// src/ling/ling_kernels.cu - maya-ling's own device kernels (ling_kernels.cuh).
#include "ling_kernels.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <random>
#include <vector>

namespace ling::k {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// a block's sum (blockDim.x a multiple of 32, at most 1024); every thread gets it
__device__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float t = lane < nw ? part[lane] : 0.0f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}
__device__ float block_max(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float t = lane < nw ? part[lane] : -INFINITY;
    t = warp_max(t);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_(float x) { return x / (1.0f + expf(-x)); }
// llama.cpp's clamped SwiGLU for Ling (build_moe_ffn / build_ffn, swiglu_clamp_*): min(silu(g), L) clamp(u, -L, L)
__device__ __forceinline__ float swiglu_l(float g, float u, float limit) {
    float a = silu_(g);
    if (limit > 0.0f) {
        a = fminf(a, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return a * u;
}
// q8_kernel's rounding of a warp's 32 values (lane's v): the block's scale to d (lane 0), q to q
__device__ __forceinline__ void warp_q8(float v, int8_t* q, float* d) {
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    *q = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if ((threadIdx.x & 31) == 0) *d = sc;
}

inline unsigned blocks(int n, int t) { return (unsigned) ((n + t - 1) / t); }
inline unsigned blocks_l(size_t n, int t) { return (unsigned) ((n + (size_t) t - 1) / (size_t) t); }

// ------------------------------------------------------------------------------------------------ generic
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n,
                               float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * sc * w[i];
}

constexpr int kPerT = 16;
__global__ void __launch_bounds__(256) norm_q8_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                                     float* __restrict__ y, int8_t* __restrict__ q, float* __restrict__ d,
                                                     int n) {
    const int t = threadIdx.x, per = n / 256;
    float xv[kPerT];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        if (k >= per) break;
        const float xc = x[t + 256 * k];
        xv[k] = xc;
        ss += xc * xc;
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / (float) n + eps);
#pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        if (k >= per) break;
        const int c = t + 256 * k;
        const float a = xv[k] * sc * w[c];
        if (y) y[c] = a;
        warp_q8(a, q + c, d + c / 32);
    }
}

__global__ void q8_kernel(const float* __restrict__ x, int k, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const size_t at = (size_t) r * k + (size_t) b * 32 + lane;
    warp_q8(x[at], q + at, d + (size_t) r * (k / 32) + b);
}

__device__ __forceinline__ float load_f(const void* w, int type, size_t off) {
    if (type == 0) return ((const float*) w)[off];
    if (type == 1) return __half2float(((const __half*) w)[off]);
    return __bfloat162float(((const __nv_bfloat16*) w)[off]);
}
// a block a row; F32 rows of a multiple of 4 values (16-byte aligned) as float4
__global__ void gemv_f_kernel(const void* __restrict__ w, int type, size_t rb, const float* __restrict__ x,
                              float* __restrict__ y, int k) {
    const int r = blockIdx.x;
    const char* row = (const char*) w + (size_t) r * rb;
    float s = 0.0f;
    if (type == 0 && (k & 3) == 0 && (rb & 15) == 0 && ((uintptr_t) w & 15) == 0) {
        const float4* w4 = (const float4*) row;
        const float4* x4 = (const float4*) x;
        for (int c = threadIdx.x; c < k / 4; c += blockDim.x) {
            const float4 a = w4[c], b = x4[c];
            s = fmaf(a.x, b.x, s);
            s = fmaf(a.y, b.y, s);
            s = fmaf(a.z, b.z, s);
            s = fmaf(a.w, b.w, s);
        }
    } else {
        for (int c = threadIdx.x; c < k; c += blockDim.x) s += load_f(row, type, c) * x[c];
    }
    s = block_sum(s);
    if (threadIdx.x == 0) y[r] = s;
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void embed_q8_0_kernel(const uint8_t* __restrict__ w, size_t rb, const int* __restrict__ ids, int n,
                                  float* __restrict__ x) {
    const int t = blockIdx.x;
    const uint8_t* row = w + (size_t) ids[t] * rb;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint8_t* b = row + (size_t) (i >> 5) * 34;
        const float d = __half2float(*(const __half*) b);
        x[(size_t) t * n + i] = d * (float) ((const int8_t*) (b + 2))[i & 31];
    }
}
__global__ void embed_f_kernel(const uint8_t* __restrict__ w, int type, size_t rb, const int* __restrict__ ids, int n,
                               float* __restrict__ x) {
    const int t = blockIdx.x;
    const uint8_t* row = w + (size_t) ids[t] * rb;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) t * n + i] = load_f(row, type, (size_t) i);
}

__global__ void to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ a, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i] * (a ? a[0] : 1.0f);
}

// ------------------------------------------------------------------------------------------------ the FFNs
__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n,
                              float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = swiglu_l(g[i], u[i], limit);
}
__global__ void swiglu_f32_rows_kernel(const float* __restrict__ gu, int n, int ff, float limit, float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ff) return;
    const int r = i / ff, c = i % ff;
    h[i] = swiglu_l(gu[(size_t) r * 2 * ff + c], gu[(size_t) r * 2 * ff + ff + c], limit);
}
__global__ void swiglu_rows_kernel(const float* __restrict__ gu, int n, int ff, float limit, __half* __restrict__ h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * ff) return;
    const int r = (int) (i / ff), c = (int) (i % ff);
    h[i] = __float2half(swiglu_l(gu[(size_t) r * 2 * ff + c], gu[(size_t) r * 2 * ff + ff + c], limit));
}
__global__ void sum_rows_kernel(const float* __restrict__ y, int n, int m, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= m) return;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += y[(size_t) i * m + c];
    out[c] = s;
}
__global__ void moe_tail_kernel(float* __restrict__ x, const float* __restrict__ cpu, const float* __restrict__ gpu,
                                const float* __restrict__ so, const int* __restrict__ flags, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float v = 0.0f;
    if (flags[0]) v = cpu[c];
    if (flags[1]) v += gpu[c];
    x[c] += v + so[c];
}

// ------------------------------------------------------------------------------------------------ KDA
// grid (H, 3: q / k / v), block hd: thread i the head's channel c = h hd + i - its convolution with the state (slid in
// place: this thread alone reads and writes it), SiLU, the head's L2 norm (q, k); the q blocks also the gate and beta
__global__ void kda_prep1_kernel(const float* __restrict__ qr, const float* __restrict__ kr, const float* __restrict__ vr,
                                 const float* __restrict__ fr, const float* __restrict__ br, float* __restrict__ st_q,
                                 float* __restrict__ st_k, float* __restrict__ st_v, const float* __restrict__ wq,
                                 const float* __restrict__ wk, const float* __restrict__ wv,
                                 const float* __restrict__ dt_bias, const float* __restrict__ a, float lb, float eps,
                                 int hd, int K, float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ vc,
                                 float* __restrict__ ge, float* __restrict__ beta) {
    const int h = blockIdx.x, which = blockIdx.y, i = threadIdx.x, c = h * hd + i;
    const float* in = which == 0 ? qr : which == 1 ? kr : vr;
    float* st = which == 0 ? st_q : which == 1 ? st_k : st_v;
    const float* w = which == 0 ? wq : which == 1 ? wk : wv;
    float win[8];
    for (int j = 0; j < K - 1; ++j) win[j] = st[(size_t) c * (K - 1) + j];
    const float x = in[c];
    win[K - 1] = x;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) acc += win[j] * w[(size_t) c * K + j];   // (ggml_ssm_conv's order: the oldest first)
    for (int j = 0; j < K - 1; ++j) st[(size_t) c * (K - 1) + j] = win[j + 1];
    float y = silu_(acc);
    if (which < 2) {
        const float ss = block_sum(y * y);
        y *= rsqrtf(ss + eps);
        (which == 0 ? qn : kn)[c] = y;
    } else {
        vc[c] = y;
    }
    if (which == 0) {
        ge[c] = expf(lb * sigmoidf_(a[h] * (fr[c] + dt_bias[c])));
        if (i == 0) beta[h] = sigmoidf_(br[h]);
    }
}

// a thread a (token, channel) of one of q / k / v (grid.y): silu(conv) over the chunk, the state's inputs before it
__global__ void kda_conv_kernel(const float* __restrict__ qr, const float* __restrict__ kr, const float* __restrict__ vr,
                                const float* __restrict__ st_q, const float* __restrict__ st_k,
                                const float* __restrict__ st_v, const float* __restrict__ wq,
                                const float* __restrict__ wk, const float* __restrict__ wv, float* __restrict__ qc,
                                float* __restrict__ kc, float* __restrict__ vc, int T, int C, int K) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) T * C) return;
    const int which = blockIdx.y, t = (int) (idx / C), c = (int) (idx % C);
    const float* in = which == 0 ? qr : which == 1 ? kr : vr;
    const float* st = which == 0 ? st_q : which == 1 ? st_k : st_v;
    const float* w = which == 0 ? wq : which == 1 ? wk : wv;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const int tt = t - (K - 1) + j;
        const float v = tt >= 0 ? in[(size_t) tt * C + c] : st[(size_t) c * (K - 1) + (tt + K - 1)];
        acc += v * w[(size_t) c * K + j];
    }
    (which == 0 ? qc : which == 1 ? kc : vc)[idx] = silu_(acc);
}
__global__ void kda_conv_state_kernel(const float* __restrict__ qr, const float* __restrict__ kr,
                                      const float* __restrict__ vr, float* __restrict__ st_q, float* __restrict__ st_k,
                                      float* __restrict__ st_v, int T, int C, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int which = blockIdx.y;
    const float* in = which == 0 ? qr : which == 1 ? kr : vr;
    float* st = which == 0 ? st_q : which == 1 ? st_k : st_v;
    float nw[8];
    for (int j = 0; j < K - 1; ++j) {
        const int tt = T - (K - 1) + j;
        nw[j] = tt >= 0 ? in[(size_t) tt * C + c] : st[(size_t) c * (K - 1) + (tt + K - 1)];
    }
    for (int j = 0; j < K - 1; ++j) st[(size_t) c * (K - 1) + j] = nw[j];
}
// grid (T, H), block hd: the head's q and k L2-normed, its gate channels, its beta
__global__ void kda_norm_gate_kernel(float* __restrict__ qc, float* __restrict__ kc, float* __restrict__ fr,
                                     const float* __restrict__ br, float* __restrict__ beta,
                                     const float* __restrict__ dt_bias, const float* __restrict__ a, float lb,
                                     float eps, int H, int hd) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x, C = H * hd, c = h * hd + i;
    const size_t at = (size_t) t * C + c;
    const float q = qc[at], k = kc[at];
    const float sq = block_sum(q * q);
    const float sk = block_sum(k * k);
    qc[at] = q * rsqrtf(sq + eps);
    kc[at] = k * rsqrtf(sk + eps);
    fr[at] = expf(lb * sigmoidf_(a[h] * (fr[at] + dt_bias[c])));
    if (i == 0) beta[(size_t) t * H + h] = sigmoidf_(br[(size_t) t * H + h]);
}

// The recurrence, ggml-cuda's gated_delta_net layout (KDA): a warp 4 columns (values j) of a head, 8 lanes a column,
// each lane 16 of its keys (i = r 8 + lane_in_col); a column's sums over its 8 lanes.  The tokens one after another
// with the state in registers; grid (H, 128 / (4 warps x 4 columns)).
constexpr int kScanWarps = 4, kScanCols = 4, kScanLanes = 8, kScanRows = 128 / kScanLanes;
__device__ __forceinline__ float seg_sum(float v) {   // a column's 8 lanes
#pragma unroll
    for (int o = kScanLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__global__ void __launch_bounds__(32 * kScanWarps) kda_scan_kernel(const float* __restrict__ q,
                                                                   const float* __restrict__ k,
                                                                   const float* __restrict__ v,
                                                                   const float* __restrict__ ge,
                                                                   const float* __restrict__ beta,
                                                                   float* __restrict__ S, float* __restrict__ o, int T,
                                                                   int H, float scale) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int cw = lane / kScanLanes, lic = lane % kScanLanes;
    const int j = (blockIdx.y * kScanWarps + w) * kScanCols + cw;
    const int C = H * 128;
    float* sc = S + ((size_t) h * 128 + j) * 128;
    float s[kScanRows];
#pragma unroll
    for (int r = 0; r < kScanRows; ++r) s[r] = sc[r * kScanLanes + lic];
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t) t * C + (size_t) h * 128;
        float kr[kScanRows], qr[kScanRows], gr[kScanRows];
#pragma unroll
        for (int r = 0; r < kScanRows; ++r) {
            const int i = r * kScanLanes + lic;
            kr[r] = k[base + i];
            qr[r] = q[base + i];
            gr[r] = ge[base + i];
        }
        const float bv = beta[(size_t) t * H + h], vv = v[base + j];
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < kScanRows; ++r) {
            s[r] *= gr[r];
            kv = fmaf(s[r], kr[r], kv);
        }
        kv = seg_sum(kv);
        const float dl = (vv - kv) * bv;
        float at = 0.0f;
#pragma unroll
        for (int r = 0; r < kScanRows; ++r) {
            s[r] = fmaf(kr[r], dl, s[r]);
            at = fmaf(s[r], qr[r], at);
        }
        at = seg_sum(at);
        if (lic == 0) o[base + j] = at * scale;
    }
#pragma unroll
    for (int r = 0; r < kScanRows; ++r) sc[r * kScanLanes + lic] = s[r];
}

// a warp a (token, head) of 128: lane values i = lane + 32 m
__global__ void kda_out_kernel(const float* __restrict__ o, const float* __restrict__ gr, const float* __restrict__ w,
                               float eps, int T, int H, float* __restrict__ y, __half* __restrict__ yh,
                               int8_t* __restrict__ q, float* __restrict__ d) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= T * H) return;
    const int t = row / H, h = row % H, C = H * 128;
    const size_t base = (size_t) t * C + (size_t) h * 128;
    float v[4], ss = 0.0f;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        v[m] = o[base + lane + 32 * m];
        ss += v[m] * v[m];
    }
    ss = warp_sum(ss);
    const float sc = rsqrtf(ss / 128.0f + eps);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int i = lane + 32 * m;
        const float val = v[m] * sc * w[i] * sigmoidf_(gr[base + i]);
        if (y) y[base + i] = val;
        if (yh) yh[base + i] = __float2half(val);
        if (q) warp_q8(val, q + base + i, d + (base + 32 * m) / 32);
    }
}

// ------------------------------------------------------------------------------------------------ MLA
__device__ __forceinline__ void rope_pair(float x0, float x1, int pos, int i, float theta_scale, float& y0, float& y1) {
    const float theta = (float) pos * powf(theta_scale, (float) i);   // (rope_norm's theta_base; freq_scale 1, no YaRN)
    const float c = cosf(theta), s = sinf(theta);
    y0 = x0 * c - x1 * s;
    y1 = x0 * s + x1 * c;
}
// grid T H, block 32: pair i of head h's rope values
__global__ void mla_rope_q_kernel(float* __restrict__ q, int H, int pos0, const int* __restrict__ dpos,
                                  float theta_scale) {
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    const int pos = (dpos ? *dpos : pos0) + t;
    float* p = q + ((size_t) t * H + h) * 192 + 128 + 2 * i;
    float y0, y1;
    rope_pair(p[0], p[1], pos, i, theta_scale, y0, y1);
    p[0] = y0;
    p[1] = y1;
}
// grid T, block 256
__global__ void mla_kv_store_kernel(const float* __restrict__ kva, const float* __restrict__ w, float eps,
                                    __half* __restrict__ cache, int pos0, const int* __restrict__ dpos,
                                    float theta_scale) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const int pos = (dpos ? *dpos : pos0) + t;
    const float* r = kva + (size_t) t * 576;
    __half* dst = cache + (size_t) pos * 576;
    const float a0 = r[tid], a1 = r[tid + 256];
    const float ss = block_sum(a0 * a0 + a1 * a1);
    const float sc = rsqrtf(ss / 512.0f + eps);
    dst[tid] = __float2half(a0 * sc * w[tid]);
    dst[tid + 256] = __float2half(a1 * sc * w[tid + 256]);
    if (tid < 32) {
        float y0, y1;
        rope_pair(r[512 + 2 * tid], r[512 + 2 * tid + 1], pos, tid, theta_scale, y0, y1);
        dst[512 + 2 * tid] = __float2half(y0);
        dst[512 + 2 * tid + 1] = __float2half(y1);
    }
}

// The decode attention (flash-decoding, MQA): a block a (split of the keys, group of kMlaHG heads); a split's keys in
// tiles of kMlaTile - their scores (a thread a key and 4 heads, the key's row read once for them), the online softmax
// (a warp 2 heads), the weighted rows (a thread 2 of the 512 dimensions and every head of the group) - then each
// head's (max, sum, acc) of the split for the combine.  The split's size from the keys there are (the grid the same at
// every position: a CUDA graph's).
constexpr int kMlaHG = 16, kMlaTile = 64, kMlaThreads = 256, kMlaD = 576, kMlaV = 512;
__host__ __device__ inline int mla_split_keys(int n_kv, int nsp) {
    const int per = (n_kv + nsp - 1) / nsp;
    return max(kMlaTile, (per + kMlaTile - 1) / kMlaTile * kMlaTile);
}
__global__ void __launch_bounds__(kMlaThreads) mla_decode_kernel(const float* __restrict__ qa,
                                                                 const float* __restrict__ qf,
                                                                 const __half* __restrict__ cache,
                                                                 const int* __restrict__ dpos, int nsp, float scale,
                                                                 float* __restrict__ part) {
    extern __shared__ float sm[];
    const int split = blockIdx.x, hg = blockIdx.y, t = threadIdx.x;
    const int n_kv = *dpos + 1, S = mla_split_keys(n_kv, nsp);
    const int j0 = split * S, j1 = min(n_kv, j0 + S);
    if (j0 >= n_kv) return;   // (a split past the keys: the combine reads only those before)
    float* qs = sm;                           // [HG][576]
    float* sc = qs + kMlaHG * kMlaD;          // [HG][tile]
    float* mrow = sc + kMlaHG * kMlaTile;     // [HG] the running max
    float* lrow = mrow + kMlaHG;              // [HG] the running sum
    float* frow = lrow + kMlaHG;              // [HG] the tile's rescale
    for (int x = t; x < kMlaHG * kMlaD; x += kMlaThreads) {
        const int hl = x / kMlaD, d = x % kMlaD, h = hg * kMlaHG + hl;
        qs[x] = d < kMlaV ? qa[(size_t) h * kMlaV + d] : qf[(size_t) h * 192 + 128 + (d - kMlaV)];
    }
    if (t < kMlaHG) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.0f;
    }
    float acc0[kMlaHG], acc1[kMlaHG];
#pragma unroll
    for (int i = 0; i < kMlaHG; ++i) acc0[i] = acc1[i] = 0.0f;
    __syncthreads();
    for (int jt = j0; jt < j1; jt += kMlaTile) {
        const int nk = min(kMlaTile, j1 - jt);
        {   // the scores: thread t key kk, heads hq .. hq + 3
            const int kk = t & (kMlaTile - 1), hq = (t / kMlaTile) * 4;
            float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (kk < nk) {
                const uint4* kr = reinterpret_cast<const uint4*>(cache + (size_t) (jt + kk) * kMlaD);
                for (int c = 0; c < kMlaD / 8; ++c) {
                    const uint4 u = kr[c];
                    const __half2* k2 = reinterpret_cast<const __half2*>(&u);
                    float kf[8];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f2 = __half22float2(k2[e]);
                        kf[2 * e] = f2.x;
                        kf[2 * e + 1] = f2.y;
                    }
#pragma unroll
                    for (int h4 = 0; h4 < 4; ++h4) {
                        const float4* q4 = reinterpret_cast<const float4*>(qs + (size_t) (hq + h4) * kMlaD + c * 8);
                        const float4 qa4 = q4[0], qb4 = q4[1];
                        float s0 = a4[h4];
                        s0 = fmaf(qa4.x, kf[0], s0);
                        s0 = fmaf(qa4.y, kf[1], s0);
                        s0 = fmaf(qa4.z, kf[2], s0);
                        s0 = fmaf(qa4.w, kf[3], s0);
                        s0 = fmaf(qb4.x, kf[4], s0);
                        s0 = fmaf(qb4.y, kf[5], s0);
                        s0 = fmaf(qb4.z, kf[6], s0);
                        s0 = fmaf(qb4.w, kf[7], s0);
                        a4[h4] = s0;
                    }
                }
            }
#pragma unroll
            for (int h4 = 0; h4 < 4; ++h4) sc[(size_t) (hq + h4) * kMlaTile + kk] = kk < nk ? a4[h4] * scale : -INFINITY;
        }
        __syncthreads();
        {   // the online softmax: warp w the heads w, w + 8
            const int w = t >> 5, lane = t & 31;
            for (int hl = w; hl < kMlaHG; hl += kMlaThreads / 32) {
                float* sh = sc + (size_t) hl * kMlaTile;
                float mx = -INFINITY;
                for (int i = lane; i < kMlaTile; i += 32) mx = fmaxf(mx, sh[i]);
                mx = warp_max(mx);
                const float m_old = mrow[hl], m_new = fmaxf(m_old, mx);
                float l = 0.0f;
                for (int i = lane; i < kMlaTile; i += 32) {
                    const float p = i < nk ? __expf(sh[i] - m_new) : 0.0f;
                    sh[i] = p;
                    l += p;
                }
                l = warp_sum(l);
                if (lane == 0) {
                    const float f = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                    frow[hl] = f;
                    lrow[hl] = lrow[hl] * f + l;
                    mrow[hl] = m_new;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int hl = 0; hl < kMlaHG; ++hl) {
            const float f = frow[hl];
            acc0[hl] *= f;
            acc1[hl] *= f;
        }
        for (int i = 0; i < nk; ++i) {
            const __half* vr = cache + (size_t) (jt + i) * kMlaD;
            const float v0 = __half2float(vr[t]), v1 = __half2float(vr[t + kMlaThreads]);
#pragma unroll
            for (int hl = 0; hl < kMlaHG; ++hl) {
                const float p = sc[(size_t) hl * kMlaTile + i];
                acc0[hl] = fmaf(p, v0, acc0[hl]);
                acc1[hl] = fmaf(p, v1, acc1[hl]);
            }
        }
        __syncthreads();   // (the next tile's scores overwrite sc)
    }
#pragma unroll
    for (int hl = 0; hl < kMlaHG; ++hl) {
        const int h = hg * kMlaHG + hl;
        float* pp = part + ((size_t) h * nsp + split) * (kMlaV + 2);
        if (t == 0) {
            pp[0] = mrow[hl];
            pp[1] = lrow[hl];
        }
        pp[2 + t] = acc0[hl];
        pp[2 + t + kMlaThreads] = acc1[hl];
    }
}
// grid H, block 512: the splits' partial sums merged (the splits that hold keys), and the row's int8 rounding
__global__ void mla_combine_kernel(const float* __restrict__ part, int nsp, const int* __restrict__ dpos,
                                   float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int n_kv = *dpos + 1, S = mla_split_keys(n_kv, nsp);
    const int used = min(nsp, (n_kv + S - 1) / S);
    const float* pp = part + (size_t) h * nsp * (kMlaV + 2);
    float M = -INFINITY;
    for (int s = 0; s < used; ++s) M = fmaxf(M, pp[(size_t) s * (kMlaV + 2)]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < used; ++s) {
        const float* p = pp + (size_t) s * (kMlaV + 2);
        const float m = p[0];
        const float c = m == -INFINITY ? 0.0f : __expf(m - M);
        L += p[1] * c;
        o += p[2 + d] * c;
    }
    const float v = L > 0.0f ? o / L : 0.0f;
    const size_t at = (size_t) h * kMlaV + d;
    out[at] = v;
    if (oq) warp_q8(v, oq + at, od + at / 32);
}
__global__ void head_gate_kernel(const float* __restrict__ x, const float* __restrict__ g, int n, int hd,
                                 float* __restrict__ y, int8_t* __restrict__ q, float* __restrict__ d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;   // (n a multiple of 32: whole warps)
    if (c >= n) return;
    const float v = x[c] * sigmoidf_(g[c / hd]);
    y[c] = v;
    if (q) warp_q8(v, q + c, d + c / 32);
}
__global__ void head_gate_rows_kernel(const float* __restrict__ x, const float* __restrict__ g, size_t n, int H, int hd,
                                      __half* __restrict__ y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t row = i / hd;   // (token, head)
    y[i] = __float2half(x[i] * sigmoidf_(g[row]));
    (void) H;
}
__global__ void mla_qpe_f16_kernel(const float* __restrict__ q, __half* __restrict__ qcat, int H, int pos0,
                                   float theta_scale) {
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    const float* p = q + ((size_t) t * H + h) * 192 + 128 + 2 * i;
    float y0, y1;
    rope_pair(p[0], p[1], pos0 + t, i, theta_scale, y0, y1);
    __half* dst = qcat + ((size_t) t * H + h) * kMlaD + kMlaV + 2 * i;
    dst[0] = __float2half(y0);
    dst[1] = __float2half(y1);
}
__global__ void mla_softmax_kernel(float* __restrict__ S, int lds, __half* __restrict__ P, int ldp, int nk, int H, int ta,
                                   int pos0) {
    const int r = blockIdx.x, p = pos0 + ta + r / H;
    float* Sr = S + (size_t) r * lds;
    __half* Pr = P + (size_t) r * ldp;
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const float v = c <= p ? Sr[c] : -INFINITY;
        Sr[c] = v;
        mx = fmaxf(mx, v);
    }
    mx = block_max(mx);
    float sum = 0.0f;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const float v = Sr[c];
        if (v != -INFINITY) sum += expf(v - mx);
    }
    sum = block_sum(sum);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int c = threadIdx.x; c < ldp; c += blockDim.x) {
        float pv = 0.0f;
        if (c < nk) {
            const float v = Sr[c];
            if (v != -INFINITY) pv = expf(v - mx) * inv;
        }
        Pr[c] = __float2half(pv);
    }
}

// ------------------------------------------------------------------------------------------------ the routing
constexpr int kRouteMaxPer = 16;   // ne / 32
__global__ void route_kernel(const float* __restrict__ RL, int ld, const float* __restrict__ bias, int T, int ne, int ng,
                             int ngu, int nu, float scale, int* __restrict__ sel, float* __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float* rl = RL + (size_t) t * ld;
    const int per = ne / 32, mpg = per / ng;   // (a group: mpg of each lane's values, m = g mpg ..)
    float sc[kRouteMaxPer], pr[kRouteMaxPer];
#pragma unroll
    for (int m = 0; m < kRouteMaxPer; ++m) {
        if (m >= per) break;
        const int e = lane + 32 * m;
        const float p = 1.0f / (1.0f + expf(-rl[e]));
        pr[m] = p;
        sc[m] = p + bias[e];
    }
    if (ng > 1) {
        float gs[16];
        for (int g = 0; g < ng; ++g) {
            float a = -INFINITY, b = -INFINITY;
            for (int j = 0; j < mpg; ++j) {
                const float v = sc[g * mpg + j];
                if (v > a) {
                    b = a;
                    a = v;
                } else if (v > b) {
                    b = v;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float oa = __shfl_xor_sync(0xffffffffu, a, o), ob = __shfl_xor_sync(0xffffffffu, b, o);
                const float na = fmaxf(a, oa), nb = fmaxf(fminf(a, oa), fmaxf(b, ob));
                a = na;
                b = nb;
            }
            gs[g] = a + b;
        }
        unsigned keep = 0;
        for (int k = 0; k < ngu; ++k) {
            int bg = -1;
            for (int g = 0; g < ng; ++g)
                if (!(keep >> g & 1u) && (bg < 0 || gs[g] > gs[bg])) bg = g;
            keep |= 1u << bg;
        }
#pragma unroll
        for (int m = 0; m < kRouteMaxPer; ++m) {
            if (m >= per) break;
            if (!(keep >> (m / mpg) & 1u)) sc[m] = -INFINITY;
        }
    }
    int chosen[16];
    float cp[16], psum = 0.0f;
    for (int k = 0; k < nu; ++k) {
        float bv = -INFINITY;
        int be = 0x7fffffff;
#pragma unroll
        for (int m = 0; m < kRouteMaxPer; ++m) {
            if (m >= per) break;
            const int e = lane + 32 * m;
            if (sc[m] > bv || (sc[m] == bv && e < be)) {
                bv = sc[m];
                be = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
            }
        }
        float mine = 0.0f;
#pragma unroll
        for (int m = 0; m < kRouteMaxPer; ++m) {
            if (m >= per) break;
            if (m == (be >> 5)) {
                mine = pr[m];
                if ((be & 31) == lane) sc[m] = -INFINITY;
            }
        }
        const float p = __shfl_sync(0xffffffffu, mine, be & 31);
        chosen[k] = be;
        cp[k] = p;
        psum += p;
    }
    if (lane == 0) {
        const float den = fmaxf(psum, 6.103515625e-5f);
        for (int k = 0; k < nu; ++k) {
            sel[(size_t) t * nu + k] = chosen[k];
            w[(size_t) t * nu + k] = cp[k] / den * scale;
        }
    }
}
__global__ void expert_count_kernel(const int* __restrict__ sel, int n, int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + sel[i], 1);
}
__global__ void expert_lists_kernel(const int* __restrict__ sel, const float* __restrict__ w, int T, int nu,
                                    const int* __restrict__ off, int* __restrict__ list, float* __restrict__ wl) {
    const int e = blockIdx.x, lane = threadIdx.x;
    const int n = T * nu;
    int base = off[e];
    for (int j0 = 0; j0 < n; j0 += 32) {
        const int j = j0 + lane;
        const bool m = j < n && sel[j] == e;
        const unsigned b = __ballot_sync(0xffffffffu, m);
        if (m) {
            const int at = base + __popc(b & ((1u << lane) - 1u));
            list[at] = j / nu;
            wl[at] = w[j];
        }
        base += __popc(b);
    }
}
__global__ void scatter_add_kernel(const float* __restrict__ ye, const int* __restrict__ list,
                                   const float* __restrict__ wl, int rows, int n, float* __restrict__ y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * n) return;
    const int r = (int) (i / n), c = (int) (i % n);
    y[(size_t) list[r] * n + c] += wl[r] * ye[i];
}

}  // namespace

// ------------------------------------------------------------------------------------------------ launches
void rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s) {
    if (rows > 0) rmsnorm_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps);
}
bool norm_q8_ok(int n) { return n % 256 == 0 && n / 256 <= kPerT; }
void norm_q8(const float* x, const float* w, float eps, float* y, int8_t* q, float* d, int n, cudaStream_t s) {
    norm_q8_kernel<<<1, 256, 0, s>>>(x, w, eps, y, q, d, n);
}
void q8(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    const int wpb = 8;
    if (rows > 0) q8_kernel<<<dim3(blocks(k / 32, wpb), rows), 32 * wpb, 0, s>>>(x, k, q, d);
}
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s) {
    gemv_f_kernel<<<n, 256, 0, s>>>(w, type, row_bytes, x, y, k);
}
void add(float* y, const float* x, int n, cudaStream_t s) { add_kernel<<<blocks(n, 256), 256, 0, s>>>(y, x, n); }
bool embed_rows(const uint8_t* w, int type, size_t rb, const int* ids, int T, int n, float* x, cudaStream_t s) {
    if (T <= 0) return true;
    if (type == 8) embed_q8_0_kernel<<<T, 256, 0, s>>>(w, rb, ids, n, x);
    else if (type == 0 || type == 1 || type == 30) embed_f_kernel<<<T, 256, 0, s>>>(w, type, rb, ids, n, x);
    else return false;
    return true;
}
void to_f16(const float* x, __half* y, size_t n, cudaStream_t s) {
    if (n) to_f16_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, y, n);
}
void axpy(float* y, const float* x, const float* a, size_t n, cudaStream_t s) {
    if (n) axpy_kernel<<<blocks_l(n, 256), 256, 0, s>>>(y, x, a, n);
}
void swiglu(const float* g, const float* u, float* h, int n, float limit, cudaStream_t s) {
    swiglu_kernel<<<blocks(n, 256), 256, 0, s>>>(g, u, h, n, limit);
}
void swiglu_f32_rows(const float* gu, int n, int ff, float limit, float* h, cudaStream_t s) {
    if (n * ff > 0) swiglu_f32_rows_kernel<<<blocks(n * ff, 256), 256, 0, s>>>(gu, n, ff, limit, h);
}
void swiglu_rows(const float* gu, int n, int ff, float limit, __half* h, cudaStream_t s) {
    const size_t m = (size_t) n * ff;
    if (m) swiglu_rows_kernel<<<blocks_l(m, 256), 256, 0, s>>>(gu, n, ff, limit, h);
}
void sum_rows(const float* y, int n, int m, float* out, cudaStream_t s) {
    sum_rows_kernel<<<blocks(m, 256), 256, 0, s>>>(y, n, m, out);
}
void moe_tail(float* x, const float* cpu, const float* gpu, const float* so, const int* flags, int n, cudaStream_t s) {
    moe_tail_kernel<<<blocks(n, 256), 256, 0, s>>>(x, cpu, gpu, so, flags, n);
}

void kda_prep1(const Kda& w, const float* qr, const float* kr, const float* vr, const float* fr, const float* br,
               float* st_q, float* st_k, float* st_v, float* qn, float* kn, float* vc, float* ge, float* beta,
               cudaStream_t s) {
    kda_prep1_kernel<<<dim3(w.H, 3), w.hd, 0, s>>>(qr, kr, vr, fr, br, st_q, st_k, st_v, w.wq, w.wk, w.wv, w.dt_bias,
                                                   w.a, w.lb, w.eps, w.hd, w.K, qn, kn, vc, ge, beta);
}
void kda_conv(const Kda& w, const float* qr, const float* kr, const float* vr, const float* st_q, const float* st_k,
              const float* st_v, float* qc, float* kc, float* vc, int T, cudaStream_t s) {
    const int C = w.H * w.hd;
    const size_t n = (size_t) T * C;
    if (n) kda_conv_kernel<<<dim3(blocks_l(n, 256), 3), 256, 0, s>>>(qr, kr, vr, st_q, st_k, st_v, w.wq, w.wk, w.wv, qc,
                                                                     kc, vc, T, C, w.K);
}
void kda_conv_state(const Kda& w, const float* qr, const float* kr, const float* vr, float* st_q, float* st_k,
                    float* st_v, int T, cudaStream_t s) {
    const int C = w.H * w.hd;
    kda_conv_state_kernel<<<dim3(blocks(C, 256), 3), 256, 0, s>>>(qr, kr, vr, st_q, st_k, st_v, T, C, w.K);
}
void kda_norm_gate(const Kda& w, float* qc, float* kc, float* fr, const float* br, float* beta, int T, cudaStream_t s) {
    if (T > 0)
        kda_norm_gate_kernel<<<dim3(T, w.H), w.hd, 0, s>>>(qc, kc, fr, br, beta, w.dt_bias, w.a, w.lb, w.eps, w.H, w.hd);
}
void kda_scan(const float* q, const float* k, const float* v, const float* ge, const float* beta, float* S, float* o,
              int T, int H, cudaStream_t s) {
    if (T > 0)
        kda_scan_kernel<<<dim3(H, 128 / (kScanWarps * kScanCols)), 32 * kScanWarps, 0, s>>>(
            q, k, v, ge, beta, S, o, T, H, 1.0f / std::sqrt(128.0f));
}
void kda_out(const float* o, const float* gr, const float* w, float eps, int T, int H, float* y, __half* yh,
             int8_t* q, float* d, cudaStream_t s) {
    const int rows = T * H, wpb = 8;
    if (rows > 0) kda_out_kernel<<<blocks(rows, wpb), 32 * wpb, 0, s>>>(o, gr, w, eps, T, H, y, yh, q, d);
}

void mla_rope_q(float* q, int T, int H, int pos0, const int* dpos, float theta_scale, cudaStream_t s) {
    if (T > 0) mla_rope_q_kernel<<<T * H, 32, 0, s>>>(q, H, pos0, dpos, theta_scale);
}
void mla_kv_store(const float* kva, const float* w, float eps, __half* cache, int T, int pos0, const int* dpos,
                  float theta_scale, cudaStream_t s) {
    if (T > 0) mla_kv_store_kernel<<<T, 256, 0, s>>>(kva, w, eps, cache, pos0, dpos, theta_scale);
}
int mla_nsp(int rows) { return std::max(128, (rows + 1023) / 1024); }
size_t mla_scratch_floats(int H, int nsp) { return (size_t) H * nsp * (kMlaV + 2); }
void mla_decode(const float* qa, const float* qf, const __half* cache, const int* dpos, int H, float scale, int nsp,
                float* scratch, float* out, int8_t* oq, float* od, cudaStream_t s) {
    const size_t smem = ((size_t) kMlaHG * kMlaD + (size_t) kMlaHG * kMlaTile + 3 * kMlaHG) * sizeof(float);   // 41 KB
    mla_decode_kernel<<<dim3(nsp, H / kMlaHG), kMlaThreads, smem, s>>>(qa, qf, cache, dpos, nsp, scale, scratch);
    mla_combine_kernel<<<H, kMlaV, 0, s>>>(scratch, nsp, dpos, out, oq, od);
}
void head_gate(const float* x, const float* g, int H, int hd, float* y, int8_t* q, float* dq, cudaStream_t s) {
    const int n = H * hd;
    head_gate_kernel<<<blocks(n, 256), 256, 0, s>>>(x, g, n, hd, y, q, dq);
}
void head_gate_rows(const float* x, const float* g, int T, int H, int hd, __half* y, cudaStream_t s) {
    const size_t n = (size_t) T * H * hd;
    if (n) head_gate_rows_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, g, n, H, hd, y);
}
void mla_qpe_f16(const float* q, __half* qcat, int T, int H, int pos0, float theta_scale, cudaStream_t s) {
    if (T > 0) mla_qpe_f16_kernel<<<T * H, 32, 0, s>>>(q, qcat, H, pos0, theta_scale);
}
void mla_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int H, int ta, int pos0, cudaStream_t s) {
    if (M > 0) mla_softmax_kernel<<<M, 256, 0, s>>>(S, lds, P, ldp, nk, H, ta, pos0);
}

void route(const float* RL, int ld, const float* bias, int T, int ne, int ng, int ngu, int nu, float scale, int* sel,
           float* w, cudaStream_t s) {
    const int wpb = 4;
    if (T > 0) route_kernel<<<blocks(T, wpb), 32 * wpb, 0, s>>>(RL, ld, bias, T, ne, ng, ngu, nu, scale, sel, w);
}
void expert_count(const int* sel, int n, int* counts, cudaStream_t s) {
    if (n > 0) expert_count_kernel<<<blocks(n, 256), 256, 0, s>>>(sel, n, counts);
}
void expert_lists(const int* sel, const float* w, int T, int nu, int ne, const int* off, int* list, float* wl,
                  cudaStream_t s) {
    expert_lists_kernel<<<ne, 32, 0, s>>>(sel, w, T, nu, off, list, wl);
}
void scatter_add(const float* ye, const int* list, const float* wl, int rows, int n, float* y, cudaStream_t s) {
    const size_t m = (size_t) rows * n;
    if (m) scatter_add_kernel<<<blocks_l(m, 256), 256, 0, s>>>(ye, list, wl, rows, n, y);
}

// ------------------------------------------------------------------------------------------------ the self-test
namespace {
float h2f(__half h) { return __half2float(h); }
bool check(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(got[i] - want[i]));
        scale = std::max(scale, (double) std::fabs(want[i]));
    }
    const bool ok = worst <= tol * std::max(1.0, scale);
    std::fprintf(stderr, "maya-ling selftest: %-34s max |diff| %.3g (values up to %.3g) %s\n", what, worst, scale,
                 ok ? "ok" : "FAIL");
    return ok;
}
template <class T> T* up(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}
}  // namespace

bool selftest() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> N(0.0f, 1.0f);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    bool ok = true;
    // KDA: H 4 heads of 128, T 37 tokens from a random state, against the host's loop (double)
    {
        const int H = 4, T = 37, C = H * 128;
        std::vector<float> q((size_t) T * C), kk((size_t) T * C), v((size_t) T * C), ge((size_t) T * C), beta((size_t) T * H);
        std::vector<float> S0((size_t) H * 128 * 128);
        for (auto& x : S0) x = 0.1f * N(rng);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                double sq = 0, sk = 0;
                for (int i = 0; i < 128; ++i) {
                    const size_t at = (size_t) t * C + h * 128 + i;
                    q[at] = N(rng);
                    kk[at] = N(rng);
                    v[at] = N(rng);
                    ge[at] = std::exp(-5.0f * U(rng));
                    sq += (double) q[at] * q[at];
                    sk += (double) kk[at] * kk[at];
                }
                for (int i = 0; i < 128; ++i) {
                    const size_t at = (size_t) t * C + h * 128 + i;
                    q[at] = (float) (q[at] / std::sqrt(sq));
                    kk[at] = (float) (kk[at] / std::sqrt(sk));
                }
                beta[(size_t) t * H + h] = U(rng);
            }
        // host: S[h][j][i] (key i, value j)
        std::vector<double> S(S0.begin(), S0.end());
        std::vector<float> ow((size_t) T * C);
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                for (int j = 0; j < 128; ++j) {
                    double* col = S.data() + ((size_t) h * 128 + j) * 128;
                    const size_t base = (size_t) t * C + h * 128;
                    double kv = 0;
                    for (int i = 0; i < 128; ++i) {
                        col[i] *= ge[base + i];
                        kv += col[i] * kk[base + i];
                    }
                    const double d = (v[base + j] - kv) * beta[(size_t) t * H + h];
                    double o = 0;
                    for (int i = 0; i < 128; ++i) {
                        col[i] += kk[base + i] * d;
                        o += col[i] * q[base + i];
                    }
                    ow[base + j] = (float) (o / std::sqrt(128.0));
                }
        float *dq = up(q), *dk = up(kk), *dv = up(v), *dg = up(ge), *db = up(beta), *dS = up(S0), *dO = nullptr;
        cudaMalloc(&dO, (size_t) T * C * sizeof(float));
        kda_scan(dq, dk, dv, dg, db, dS, dO, T, H, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> Sw(S.begin(), S.end());
        ok = check("KDA scan: outputs", down(dO, (size_t) T * C), ow, 1e-4) && ok;
        ok = check("KDA scan: the state after", down(dS, S.size()), Sw, 1e-4) && ok;
        // the same tokens one at a time (the token path's launches)
        cudaMemcpy(dS, S0.data(), S0.size() * sizeof(float), cudaMemcpyHostToDevice);
        for (int t = 0; t < T; ++t)
            kda_scan(dq + (size_t) t * C, dk + (size_t) t * C, dv + (size_t) t * C, dg + (size_t) t * C, db + (size_t) t * H,
                     dS, dO + (size_t) t * C, 1, H, nullptr);
        cudaDeviceSynchronize();
        ok = check("KDA scan: token by token", down(dO, (size_t) T * C), ow, 1e-4) && ok;
        for (float* p : {dq, dk, dv, dg, db, dS, dO}) cudaFree(p);
    }
    // MLA decode: H 64 heads, 300 keys at position 299, against the host's softmax (double)
    {
        const int H = 64, n = 300, nsp = 128;
        const float scale = 1.0f / std::sqrt(192.0f);
        std::vector<float> qa((size_t) H * 512), qf((size_t) H * 192);
        std::vector<__half> cache((size_t) n * 576);
        std::vector<float> cf((size_t) n * 576);
        for (auto& x : qa) x = N(rng);
        for (auto& x : qf) x = N(rng);
        for (size_t i = 0; i < cache.size(); ++i) {
            cache[i] = __float2half(N(rng));
            cf[i] = h2f(cache[i]);
        }
        std::vector<float> want((size_t) H * 512);
        for (int h = 0; h < H; ++h) {
            std::vector<double> sc(n);
            double mx = -1e300;
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int d = 0; d < 512; ++d) s += (double) qa[(size_t) h * 512 + d] * cf[(size_t) j * 576 + d];
                for (int d = 0; d < 64; ++d) s += (double) qf[(size_t) h * 192 + 128 + d] * cf[(size_t) j * 576 + 512 + d];
                sc[j] = s * scale;
                mx = std::max(mx, sc[j]);
            }
            double l = 0;
            for (int j = 0; j < n; ++j) l += std::exp(sc[j] - mx);
            for (int d = 0; d < 512; ++d) {
                double o = 0;
                for (int j = 0; j < n; ++j) o += std::exp(sc[j] - mx) * cf[(size_t) j * 576 + d];
                want[(size_t) h * 512 + d] = (float) (o / l);
            }
        }
        float *dqa = up(qa), *dqf = up(qf), *dout = nullptr, *dscr = nullptr;
        __half* dc = up(cache);
        int* dp = nullptr;
        const int pos = n - 1;
        cudaMalloc(&dp, sizeof(int));
        cudaMemcpy(dp, &pos, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&dout, want.size() * sizeof(float));
        cudaMalloc(&dscr, mla_scratch_floats(H, nsp) * sizeof(float));
        mla_decode(dqa, dqf, dc, dp, H, scale, nsp, dscr, dout, nullptr, nullptr, nullptr);
        cudaDeviceSynchronize();
        ok = check("MLA decode attention", down(dout, want.size()), want, 2e-3) && ok;
        cudaFree(dqa);
        cudaFree(dqf);
        cudaFree(dout);
        cudaFree(dscr);
        cudaFree(dc);
        cudaFree(dp);
    }
    // the routing: 3 tokens of random logits against the host's choice
    {
        const int T = 3, ne = 512;
        std::vector<float> rl((size_t) T * ne), bias(ne);
        for (auto& x : rl) x = 2.0f * N(rng);
        for (auto& x : bias) x = 0.1f * N(rng);
        float *drl = up(rl), *dbias = up(bias), *dw = nullptr;
        int* dsel = nullptr;
        cudaMalloc(&dw, (size_t) T * 8 * sizeof(float));
        cudaMalloc(&dsel, (size_t) T * 8 * sizeof(int));
        route(drl, ne, dbias, T, ne, 8, 4, 8, 2.5f, dsel, dw, nullptr);
        cudaDeviceSynchronize();
        const std::vector<int> sel = down(dsel, (size_t) T * 8);
        const std::vector<float> w = down(dw, (size_t) T * 8);
        bool same = true;
        for (int t = 0; t < T; ++t) {
            std::vector<float> p(ne), s(ne);
            for (int e = 0; e < ne; ++e) {
                p[e] = 1.0f / (1.0f + std::exp(-rl[(size_t) t * ne + e]));
                s[e] = p[e] + bias[e];
            }
            std::vector<std::pair<float, int>> gsc;
            for (int g = 0; g < 8; ++g) {
                std::vector<float> v(s.begin() + g * 64, s.begin() + g * 64 + 64);
                std::sort(v.begin(), v.end(), std::greater<float>());
                gsc.push_back({v[0] + v[1], g});
            }
            std::stable_sort(gsc.begin(), gsc.end(), [](auto& a, auto& b) { return a.first > b.first; });
            std::vector<int> idx;
            for (int k = 0; k < 4; ++k)
                for (int e = gsc[k].second * 64; e < gsc[k].second * 64 + 64; ++e) idx.push_back(e);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
            double ps = 0;
            for (int k = 0; k < 8; ++k) ps += p[idx[k]];
            for (int k = 0; k < 8; ++k) {
                same = same && sel[(size_t) t * 8 + k] == idx[k] &&
                       std::fabs(w[(size_t) t * 8 + k] - (float) (p[idx[k]] / ps * 2.5)) < 1e-5f;
            }
        }
        std::fprintf(stderr, "maya-ling selftest: %-34s %s\n", "routing (groups, top 8, weights)", same ? "ok" : "FAIL");
        ok = ok && same;
        cudaFree(drl);
        cudaFree(dbias);
        cudaFree(dw);
        cudaFree(dsel);
    }
    return ok;
}

}  // namespace ling::k
