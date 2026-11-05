// src/nem/nem_ext.cu - maya-nem's Nemotron-only kernels (nem_ext.cuh).
#include "nem_ext.cuh"

#include <algorithm>
#include <cmath>

namespace nem::k {

namespace {

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
inline unsigned nblocks(size_t n, int per) { return (unsigned) std::max<size_t>(1, (n + per - 1) / per); }

__global__ void conv_step_kernel(float* __restrict__ xbc, float* __restrict__ st, const float* __restrict__ w,
                                 const float* __restrict__ b, int ch, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float* sc = st + (size_t) c * (K - 1);
    const float* wc = w + (size_t) c * K;
    const float xin = xbc[c];
    float acc = b ? b[c] : 0.0f;
    for (int i = 0; i < K - 1; ++i) acc = fmaf(wc[i], sc[i], acc);
    acc = fmaf(wc[K - 1], xin, acc);
    for (int i = 0; i + 1 < K - 1; ++i) sc[i] = sc[i + 1];
    sc[K - 2] = xin;
    xbc[c] = silu(acc);
}

__global__ void conv_chunk_kernel(const float* __restrict__ x, int ldx, float* __restrict__ out,
                                  const float* __restrict__ st, const float* __restrict__ w, const float* __restrict__ b,
                                  int T, int ch, int K) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ch) return;
    const int t = (int) (i / ch), c = (int) (i % ch);
    const float* wc = w + (size_t) c * K;
    float acc = b ? b[c] : 0.0f;
    for (int j = 0; j < K; ++j) {
        const int tt = t - (K - 1) + j;
        const float v = tt >= 0 ? x[(size_t) tt * ldx + c] : st[(size_t) c * (K - 1) + (tt + K - 1)];
        acc = fmaf(wc[j], v, acc);
    }
    out[(size_t) t * ch + c] = silu(acc);
}

__global__ void conv_state_kernel(const float* __restrict__ x, int ldx, float* __restrict__ st, int T, int ch, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float v[8];
    float* sc = st + (size_t) c * (K - 1);
    for (int j = 0; j < K - 1; ++j) {   // the inputs at T - (K - 1) + j
        const int tt = T - (K - 1) + j;
        v[j] = tt >= 0 ? x[(size_t) tt * ldx + c] : sc[tt + K - 1];
    }
    for (int j = 0; j < K - 1; ++j) sc[j] = v[j];
}

// A warp RPW rows (p) of a head, a lane N / 32 state values (n = lane + 32 i) each, in registers over the T tokens.
constexpr int kScanWarps = 8, kRPW = 2, kMaxNL = 8;
__global__ void __launch_bounds__(kScanWarps * 32) scan_kernel(const float* __restrict__ xbc, int ldxbc,
                                                               const float* __restrict__ z, int ldz, int dt_off,
                                                               const float* __restrict__ dt_bias,
                                                               const float* __restrict__ A, const float* __restrict__ D,
                                                               float* __restrict__ state, float* __restrict__ y, int T,
                                                               int H, int P, int G, int N) {
    const int h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p0 = (blockIdx.x * kScanWarps + warp) * kRPW;
    if (p0 >= P) return;
    const int g = h / (H / G), NL = N / 32;
    const float a = A[h], dtb = dt_bias[h], dd = D[h];
    float s[kRPW][kMaxNL];
#pragma unroll
    for (int r = 0; r < kRPW; ++r)
#pragma unroll
        for (int i = 0; i < kMaxNL; ++i)
            if (i < NL) s[r][i] = state[((size_t) h * P + p0 + r) * N + lane + 32 * i];
    const int HP = H * P;
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * ldxbc;
        const float* zr = z + (size_t) t * ldz;
        float dt = zr[dt_off + h] + dtb;
        dt = dt <= 20.0f ? log1pf(__expf(dt)) : dt;
        const float dA = __expf(dt * a);
        float Bv[kMaxNL], Cv[kMaxNL];
#pragma unroll
        for (int i = 0; i < kMaxNL; ++i)
            if (i < NL) {
                Bv[i] = row[HP + g * N + lane + 32 * i];
                Cv[i] = row[HP + G * N + g * N + lane + 32 * i];
            }
#pragma unroll
        for (int r = 0; r < kRPW; ++r) {
            const int p = p0 + r;
            const float xv = row[h * P + p], xdt = xv * dt;
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < kMaxNL; ++i)
                if (i < NL) {
                    s[r][i] = fmaf(s[r][i], dA, Bv[i] * xdt);
                    acc = fmaf(s[r][i], Cv[i], acc);
                }
            acc = wsum(acc);
            if (lane == 0) y[(size_t) t * HP + h * P + p] = (acc + dd * xv) * silu(zr[h * P + p]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRPW; ++r)
#pragma unroll
        for (int i = 0; i < kMaxNL; ++i)
            if (i < NL) state[((size_t) h * P + p0 + r) * N + lane + 32 * i] = s[r][i];
}

__global__ void group_rmsnorm_kernel(float* __restrict__ y, const float* __restrict__ w, int groups, int n, float eps) {
    const int r = blockIdx.y, g = blockIdx.x;
    float* v = y + ((size_t) r * groups + g) * n;
    const float* wg = w + (size_t) g * n;
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += v[i] * v[i];
    ss = wsum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        t = wsum(t);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float sc = rsqrtf(red[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) v[i] = v[i] * sc * wg[i];
}

__global__ void relu2_kernel(float* __restrict__ x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = fmaxf(x[i], 0.0f);
    x[i] = v * v;
}

__global__ void relu2_f16_kernel(const float* __restrict__ x, int ldx, int rows, int ff, __half* __restrict__ h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * ff) return;
    const int r = (int) (i / ff), c = (int) (i % ff);
    const float v = fmaxf(x[(size_t) r * ldx + c], 0.0f);
    h[i] = __float2half_rn(v * v);
}

constexpr int kRouteMaxPer = 16;   // (512 experts / 32 lanes)
__global__ void route_sig_kernel(const float* __restrict__ RL, int ld, const float* __restrict__ bias, int T, int ne,
                                 int nu, int norm, float scale, int* __restrict__ sel, float* __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const int per = ne / 32;
    float sc[kRouteMaxPer], sg[kRouteMaxPer];
#pragma unroll
    for (int i = 0; i < kRouteMaxPer; ++i)
        if (i < per) {
            const int e = lane + 32 * i;
            sg[i] = 1.0f / (1.0f + expf(-RL[(size_t) t * ld + e]));
            sc[i] = sg[i] + bias[e];
        }
    float wsel = 0.0f, tot = 0.0f;
    for (int k = 0; k < nu; ++k) {
        // the lane's best (the lower id on a tie), then the warp's
        float bv = -INFINITY, bs = 0.0f;
        int be = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < kRouteMaxPer; ++i)
            if (i < per && (sc[i] > bv || (sc[i] == bv && lane + 32 * i < be))) {
                bv = sc[i];
                be = lane + 32 * i;
                bs = sg[i];
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o), os = __shfl_xor_sync(0xffffffffu, bs, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
                bs = os;
            }
        }
        if ((be & 31) == lane) sc[be >> 5] = -INFINITY;   // (the lane holding it: out of the next rounds)
        if (lane == k) wsel = bs;
        tot += bs;
        if (lane == 0) sel[(size_t) t * nu + k] = be;
    }
    if (lane < nu) {
        const float d = norm ? fmaxf(tot, 6.103515625e-5f) : 1.0f;
        w[(size_t) t * nu + lane] = wsel / d * scale;
    }
}

__global__ void lat_sum_kernel(const float* __restrict__ cpu, const float* __restrict__ gpu, const int* __restrict__ fl,
                               float* __restrict__ out, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    out[c] = (fl[0] ? cpu[c] : 0.0f) + (fl[1] ? gpu[c] : 0.0f);
}

__global__ void add2_kernel(float* __restrict__ y, const float* __restrict__ a, const float* __restrict__ b, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) y[c] += a[c] + b[c];
}

}  // namespace

void mamba_conv_step(float* xbc, float* state, const float* w, const float* b, int ch, int K, cudaStream_t s) {
    conv_step_kernel<<<nblocks((size_t) ch, 256), 256, 0, s>>>(xbc, state, w, b, ch, K);
}
void mamba_conv_chunk(const float* x, int ldx, float* out, float* state, const float* w, const float* b, int T, int ch,
                      int K, cudaStream_t s) {
    if (T <= 0) return;
    conv_chunk_kernel<<<nblocks((size_t) T * ch, 256), 256, 0, s>>>(x, ldx, out, state, w, b, T, ch, K);
    conv_state_kernel<<<nblocks((size_t) ch, 256), 256, 0, s>>>(x, ldx, state, T, ch, K);
}
void mamba_scan(const float* xbc, int ldxbc, const float* z, int ldz, int dt_off, const float* dt_bias, const float* A,
                const float* D, float* state, float* y, int T, int H, int P, int G, int N, cudaStream_t s) {
    if (T <= 0) return;
    const dim3 grid((unsigned) ((P + kScanWarps * kRPW - 1) / (kScanWarps * kRPW)), (unsigned) H);
    scan_kernel<<<grid, kScanWarps * 32, 0, s>>>(xbc, ldxbc, z, ldz, dt_off, dt_bias, A, D, state, y, T, H, P, G, N);
}
void group_rmsnorm(float* y, const float* w, int rows, int groups, int n, float eps, cudaStream_t s) {
    group_rmsnorm_kernel<<<dim3((unsigned) groups, (unsigned) rows), 256, 0, s>>>(y, w, groups, n, eps);
}
void relu2(float* x, size_t n, cudaStream_t s) {
    if (n) relu2_kernel<<<nblocks(n, 256), 256, 0, s>>>(x, n);
}
void relu2_f16(const float* x, int ldx, int rows, int ff, __half* h, cudaStream_t s) {
    const size_t n = (size_t) rows * ff;
    if (n) relu2_f16_kernel<<<nblocks(n, 256), 256, 0, s>>>(x, ldx, rows, ff, h);
}
void route_sig(const float* RL, int ld, const float* bias, int T, int ne, int nu, bool norm, float scale, int* sel,
               float* w, cudaStream_t s) {
    const int wpb = 4;
    if (T > 0)
        route_sig_kernel<<<nblocks((size_t) T, wpb), 32 * wpb, 0, s>>>(RL, ld, bias, T, ne, nu, norm ? 1 : 0, scale,
                                                                        sel, w);
}
void lat_sum(const float* cpu, const float* gpu, const int* flags, float* out, int n, cudaStream_t s) {
    lat_sum_kernel<<<nblocks((size_t) n, 256), 256, 0, s>>>(cpu, gpu, flags, out, n);
}
void add2(float* y, const float* a, const float* b, int n, cudaStream_t s) {
    add2_kernel<<<nblocks((size_t) n, 256), 256, 0, s>>>(y, a, b, n);
}

}  // namespace nem::k
