// src/hy4/hy4_kernels.cu - maya-hy4's own device kernels (ling_kernels.cuh).
#include "hy4_kernels.cuh"
#include "hy4_topk.cuh"

#include <mma.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace hy4::k {

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
// Hy4's clamped SwiGLU (llama.cpp's ggml_swiglu_clamp, as for DeepSeek 4): silu(min(g, L)) clamp(u, -L, L)
__device__ __forceinline__ float swiglu_l(float g, float u, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return silu_(g) * u;
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
                              float* __restrict__ y, int k, int f32_order) {
    const int r = blockIdx.x;
    const char* row = (const char*) w + (size_t) r * rb;
    float s = 0.0f;
    if (f32_order && type == 30 && (k & 3) == 0 && (rb & 7) == 0 && ((uintptr_t) w & 7) == 0) {
        // (BF16 rows of an F32 matrix whose values BF16 holds exactly: the F32 branch's loads and sums, the same bits)
        const uint2* w2 = (const uint2*) row;
        const float4* x4 = (const float4*) x;
        for (int c = threadIdx.x; c < k / 4; c += blockDim.x) {
            const uint2 u = w2[c];
            const float4 b = x4[c];
            s = fmaf(__uint_as_float(u.x << 16), b.x, s);
            s = fmaf(__uint_as_float(u.x & 0xffff0000u), b.y, s);
            s = fmaf(__uint_as_float(u.y << 16), b.z, s);
            s = fmaf(__uint_as_float(u.y & 0xffff0000u), b.w, s);
        }
    } else if (type == 0 && (k & 3) == 0 && (rb & 15) == 0 && ((uintptr_t) w & 15) == 0) {
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
__global__ void mla_rope_q_kernel(float* __restrict__ q, int H, int qk, int nope, int pos0, const int* __restrict__ dpos,
                                  float theta_scale) {
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    const int pos = (dpos ? *dpos : pos0) + t;
    float* p = q + ((size_t) t * H + h) * qk + nope + 2 * i;
    float y0, y1;
    rope_pair(p[0], p[1], pos, i, theta_scale, y0, y1);
    p[0] = y0;
    p[1] = y1;
}
// grid T, block 256
__global__ void mla_kv_store_kernel(const float* __restrict__ kva, const float* __restrict__ w, float eps,
                                    __half* __restrict__ cache, int pos0, const int* __restrict__ dpos,
                                    float theta_scale, int ring, int base) {
    const int t = blockIdx.x, tid = threadIdx.x;
    const int pos = (dpos ? *dpos : pos0) + t;
    const float* r = kva + (size_t) t * 576;
    __half* dst = cache + (size_t) (ring > 0 ? pos % ring : pos - base) * 576;
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
// a split's keys: n_kv over nsp splits, at least smin (and a multiple of it: 64 = the tile; fewer keys a split give more
// blocks, the decode being bound by its latency at a few thousand keys)
__host__ __device__ inline int mla_split_keys(int n_kv, int nsp, int smin = kMlaTile) {
    const int per = (n_kv + nsp - 1) / nsp;
    return max(smin, (per + smin - 1) / smin * smin);
}
__global__ void __launch_bounds__(kMlaThreads) mla_decode_kernel(const float* __restrict__ qa,
                                                                 const float* __restrict__ qf, int qk, int nope,
                                                                 const __half* __restrict__ cache,
                                                                 const int* __restrict__ dpos, int win,
                                                                 const int* __restrict__ cells,
                                                                 const int* __restrict__ ncell, int ring, int nsp,
                                                                 float scale, float* __restrict__ part, int smin) {
    extern __shared__ float sm[];
    const int split = blockIdx.x, hg = blockIdx.y, t = threadIdx.x;
    const int n_kv = *dpos + 1, j_lo = win > 0 ? max(0, n_kv - win) : 0;
    const int n = cells ? *ncell : n_kv - j_lo, S = mla_split_keys(n, nsp, smin);
    const int j0 = split * S, j1 = min(n, j0 + S);
    if (j0 >= n) return;   // (a split past the keys: the combine reads only those before)
    auto row = [&](int j) { return cells ? cells[j] : ring > 0 ? (j_lo + j) % ring : j_lo + j; };
    float* qs = sm;                           // [HG][576]
    float* sc = qs + kMlaHG * kMlaD;          // [HG][tile]
    float* mrow = sc + kMlaHG * kMlaTile;     // [HG] the running max
    float* lrow = mrow + kMlaHG;              // [HG] the running sum
    float* frow = lrow + kMlaHG;              // [HG] the tile's rescale
    for (int x = t; x < kMlaHG * kMlaD; x += kMlaThreads) {
        const int hl = x / kMlaD, d = x % kMlaD, h = hg * kMlaHG + hl;
        qs[x] = d < kMlaV ? qa[(size_t) h * kMlaV + d] : qf[(size_t) h * qk + nope + (d - kMlaV)];
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
                const uint4* kr = reinterpret_cast<const uint4*>(cache + (size_t) row(jt + kk) * kMlaD);
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
            const __half* vr = cache + (size_t) row(jt + i) * kMlaD;
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

// The decode attention on tensor cores (HY4_MLA_TC, on; sm_80 and up): mla_decode_kernel's blocks and partial sums, the
// products as FlashMLA does them - a block (a split of the keys, kMlaHG = 16 heads) of 4 warps; the heads' queries F16
// [16][576] once; a tile of 64 keys' rows F16 [64][576] (zeros past the keys); the scores Q K^T on 16x16x16 tiles (a
// warp 16 keys, 36 steps of the 576), the tile's online softmax (P F16), then P V (V = the rows' first 512) added to
// the heads' F32 sums [16][512] (a warp 8 of the 32 column tiles, 4 steps of the 64 keys), the sums first rescaled.
// (a tile of 32 keys: the block's shared memory ~91 KB, under the 99 KB a block of sm_86 / sm_89 / sm_120 may have)
// Rows in shared memory padded (576 -> 584 halves, 512 -> 516 floats): 1,152- and 2,048-byte rows put a 16x16 tile's
// 16 rows on the same banks (16-way conflicts: the products took 23 of the kernel's 26 us).
constexpr int kTcWarps = 4, kTcThreads = kTcWarps * 32, kTcTile = 32, kQS = kMlaD + 8, kAS = kMlaV + 4;
constexpr size_t kTcSmem = (size_t) kMlaHG * kQS * 2 + (size_t) kTcTile * kQS * 2 + (size_t) kMlaHG * kTcTile * 4 +
                           (size_t) kMlaHG * kTcTile * 2 + (size_t) kMlaHG * kAS * 4 + 3 * kMlaHG * 4;
__global__ void __launch_bounds__(kTcThreads) mla_decode_tc_kernel(const float* __restrict__ qa,
                                                                   const float* __restrict__ qf, int qk, int nope,
                                                                   const __half* __restrict__ cache,
                                                                   const int* __restrict__ dpos, int win,
                                                                   const int* __restrict__ cells,
                                                                   const int* __restrict__ ncell, int ring, int nsp,
                                                                   float scale, float* __restrict__ part, int smin,
                                                                   unsigned* qmax) {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    extern __shared__ __align__(128) unsigned char smb[];
    __half* qs = (__half*) smb;                                      // [16][kQS]
    __half* ks = qs + kMlaHG * kQS;                                  // [32][kQS]
    float* sc = (float*) (ks + kTcTile * kQS);                       // [16][32]
    __half* ps = (__half*) (sc + kMlaHG * kTcTile);                  // [16][32]
    float* acc = (float*) (ps + kMlaHG * kTcTile);                   // [16][kAS]
    float* mrow = acc + kMlaHG * kAS;
    float* lrow = mrow + kMlaHG;
    float* frow = lrow + kMlaHG;
    const int split = blockIdx.x, hg = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int n_kv = *dpos + 1, j_lo = win > 0 ? max(0, n_kv - win) : 0;
    const int n = cells ? *ncell : n_kv - j_lo, S = mla_split_keys(n, nsp, smin);
    const int j0 = split * S, j1 = min(n, j0 + S);
    if (j0 >= n) return;
    auto row = [&](int j) { return cells ? cells[j] : ring > 0 ? (j_lo + j) % ring : j_lo + j; };
    {
        constexpr int kQ4 = kMlaHG * kMlaD / 4, kPer = kQ4 / kTcThreads;   // (18 float4 a thread)
        float4 v[kPer];
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int x = (t + i * kTcThreads) * 4, hl = x / kMlaD, d = x % kMlaD, h = hg * kMlaHG + hl;
            v[i] = d < kMlaV ? *reinterpret_cast<const float4*>(qa + (size_t) h * kMlaV + d)
                             : *reinterpret_cast<const float4*>(qf + (size_t) h * qk + nope + (d - kMlaV));
        }
        // (scaled before F16: the scores need it anyway, and |q| gets 16x the headroom under F16's 65,504)
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int x = (t + i * kTcThreads) * 4, hl = x / kMlaD, d = x % kMlaD;
            __half2* q2 = reinterpret_cast<__half2*>(qs + (size_t) hl * kQS + d);
            q2[0] = __floats2half2_rn(v[i].x * scale, v[i].y * scale);
            q2[1] = __floats2half2_rn(v[i].z * scale, v[i].w * scale);
        }
        if (qmax) {   // (HY4_MLA_QMAX: the largest |q|, the 512 and the rope part apart - positive floats order as ints)
            float m0 = 0.0f, m1 = 0.0f;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int d = ((t + i * kTcThreads) * 4) % kMlaD;
                const float m = fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w)));
                if (d < kMlaV) m0 = fmaxf(m0, m);
                else m1 = fmaxf(m1, m);
            }
            atomicMax(qmax, __float_as_uint(m0));
            atomicMax(qmax + 1, __float_as_uint(m1));
        }
    }
    for (int x = t * 4; x < kMlaHG * kMlaV; x += kTcThreads * 4)
        *reinterpret_cast<float4*>(acc + (size_t) (x / kMlaV) * kAS + x % kMlaV) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (t < kMlaHG) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.0f;
    }
    for (int jt = j0; jt < j1; jt += kTcTile) {
        const int nk = min(kTcTile, j1 - jt);
        {   // the tile's rows (zeros past the keys): a thread's 18 16-byte pieces in flight at once
            constexpr int kR16 = kTcTile * (kMlaD / 8), kPer = kR16 / kTcThreads;
            uint4 u[kPer];
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int x = t + i * kTcThreads, kk = x / (kMlaD / 8), c = x % (kMlaD / 8);
                u[i] = kk < nk ? reinterpret_cast<const uint4*>(cache + (size_t) row(jt + kk) * kMlaD)[c]
                               : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int x = t + i * kTcThreads, kk = x / (kMlaD / 8), c = x % (kMlaD / 8);
                reinterpret_cast<uint4*>(ks + (size_t) kk * kQS)[c] = u[i];
            }
        }
        __syncthreads();
        if (warp < kTcTile / 16) {   // the scores: warp w the keys 16 w .. 16 w + 15 of the tile
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            wmma::fill_fragment(fc, 0.0f);
            for (int k0 = 0; k0 < kMlaD; k0 += 16) {
                wmma::load_matrix_sync(fa, qs + k0, kQS);
                wmma::load_matrix_sync(fb, ks + (size_t) (16 * warp) * kQS + k0, kQS);
                wmma::mma_sync(fc, fa, fb, fc);
            }
            wmma::store_matrix_sync(sc + 16 * warp, fc, kTcTile, wmma::mem_row_major);
        }
        __syncthreads();
        // the online softmax: warp w the heads 4 w .. 4 w + 3 (a key a lane)
        for (int hl = 4 * warp; hl < 4 * warp + 4; ++hl) {
            const float* sh = sc + (size_t) hl * kTcTile;
            const float v0 = lane < nk ? sh[lane] : -INFINITY;   // (the queries came scaled)
            const float mx = warp_max(v0);
            const float m_old = mrow[hl], m_new = fmaxf(m_old, mx);
            const float p0 = lane < nk ? __expf(v0 - m_new) : 0.0f;
            ps[(size_t) hl * kTcTile + lane] = __float2half(p0);
            const float l = warp_sum(p0);
            if (lane == 0) {
                const float f = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                frow[hl] = f;
                lrow[hl] = lrow[hl] * f + l;
                mrow[hl] = m_new;
            }
        }
        __syncthreads();
        for (int x = t; x < kMlaHG * kMlaV; x += kTcThreads) acc[(size_t) (x / kMlaV) * kAS + x % kMlaV] *= frow[x / kMlaV];
        __syncthreads();
        {   // P V: warp w the column tiles 8 w .. 8 w + 7 of the 512
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
            for (int ct = 8 * warp; ct < 8 * warp + 8; ++ct) {
                wmma::load_matrix_sync(fc, acc + 16 * ct, kAS, wmma::mem_row_major);
                for (int k0 = 0; k0 < kTcTile; k0 += 16) {
                    wmma::load_matrix_sync(fa, ps + k0, kTcTile);
                    wmma::load_matrix_sync(fb, ks + (size_t) k0 * kQS + 16 * ct, kQS);
                    wmma::mma_sync(fc, fa, fb, fc);
                }
                wmma::store_matrix_sync(acc + 16 * ct, fc, kAS, wmma::mem_row_major);
            }
        }
        __syncthreads();   // (the next tile's rows overwrite ks)
    }
    for (int x = t; x < kMlaHG * kMlaV; x += kTcThreads) {
        const int hl = x / kMlaV, d = x % kMlaV, h = hg * kMlaHG + hl;
        part[((size_t) h * nsp + split) * (kMlaV + 2) + 2 + d] = acc[(size_t) hl * kAS + d];
    }
    if (t < kMlaHG) {
        float* pp = part + ((size_t) (hg * kMlaHG + t) * nsp + split) * (kMlaV + 2);
        pp[0] = mrow[t];
        pp[1] = lrow[t];
    }
#endif
}
// grid H, block 512: the splits' partial sums merged (the splits that hold keys), and the row's int8 rounding
// A head's sink (sinks set): one more logit in the softmax's sum, with no value row (ggml_soft_max_add_sinks).
__global__ void mla_combine_kernel(const float* __restrict__ part, int nsp, const int* __restrict__ dpos, int win,
                                   const int* __restrict__ ncell, const float* __restrict__ sinks,
                                   float* __restrict__ out, int8_t* __restrict__ oq, float* __restrict__ od, int smin) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int n_kv = *dpos + 1, n = ncell ? *ncell : win > 0 ? min(n_kv, win) : n_kv,
              S = mla_split_keys(n, nsp, smin);
    const int used = min(nsp, (n + S - 1) / S);
    const float* pp = part + (size_t) h * nsp * (kMlaV + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < used; ++s) M = fmaxf(M, pp[(size_t) s * (kMlaV + 2)]);
    float L = sinks ? __expf(sinks[h] - M) : 0.0f, o = 0.0f;
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
// route_kernel for one token (the token path) in one block of ne threads: each thread an expert's sigmoid and score,
// its rank the scores above it (ties: the lower id first - route_kernel's order), the nu best by rank; their weights
// summed in that order as route_kernel sums them - the same choices and the same bits, in ~1 us where route_kernel's
// 16 serial warp rounds took ~18 (ng 1 only)
constexpr int kRoute1Max = 1024;
__global__ void route1_kernel(const float* __restrict__ rl, const float* __restrict__ bias, int ne, int nu, float scale,
                              int* __restrict__ sel, float* __restrict__ w) {
    __shared__ float s_sc[kRoute1Max];
    __shared__ float s_p[16];
    __shared__ int s_e[16];
    const int e = threadIdx.x;
    float p = 0.0f, sc = -INFINITY;
    if (e < ne) {
        p = 1.0f / (1.0f + expf(-rl[e]));
        sc = p + bias[e];
        if (!(sc == sc)) sc = -INFINITY;   // (a NaN last: it would rank 0 beside the top one and leave a choice unwritten)
        s_sc[e] = sc;
    }
    __syncthreads();
    if (e < ne) {
        int rank = 0;
        for (int j = 0; j < ne; ++j) {
            const float v = s_sc[j];
            rank += v > sc || (v == sc && j < e);
        }
        if (rank < nu) {
            s_e[rank] = e;
            s_p[rank] = p;
        }
    }
    __syncthreads();
    if (e == 0) {
        float psum = 0.0f;
        for (int k = 0; k < nu; ++k) psum += s_p[k];
        const float den = fmaxf(psum, 6.103515625e-5f);
        for (int k = 0; k < nu; ++k) {
            sel[k] = s_e[k];
            w[k] = s_p[k] / den * scale;
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
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ src, size_t n, float* __restrict__ dst) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __uint_as_float((unsigned) src[i] << 16);
}
void bf16_to_f32(const void* src, size_t n, float* dst, cudaStream_t s) {
    if (n) bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((const uint16_t*) src, n, dst);
}
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s,
            bool f32_order) {
    gemv_f_kernel<<<n, 256, 0, s>>>(w, type, row_bytes, x, y, k, f32_order ? 1 : 0);
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

void mla_rope_q(float* q, int T, int H, int qk, int nope, int pos0, const int* dpos, float theta_scale, cudaStream_t s) {
    if (T > 0) mla_rope_q_kernel<<<T * H, 32, 0, s>>>(q, H, qk, nope, pos0, dpos, theta_scale);
}
void mla_kv_store(const float* kva, const float* w, float eps, __half* cache, int T, int pos0, const int* dpos,
                  float theta_scale, cudaStream_t s, int ring, int base) {
    if (T > 0) mla_kv_store_kernel<<<T, 256, 0, s>>>(kva, w, eps, cache, pos0, dpos, theta_scale, ring, base);
}
int mla_nsp(int rows) { return std::max(128, (rows + 1023) / 1024); }
// (mla_init, at the load: the tensor cores' kernel and its shared memory, outside any graph capture)
static bool g_mla_tc = false;
static unsigned* g_mla_qmax = nullptr;   // (HY4_MLA_QMAX=1: the largest |q| seen, [2] on the device)
bool mla_qmax(float* out) {
    if (!g_mla_qmax) return false;
    unsigned u[2] = {0, 0};
    if (cudaMemcpy(u, g_mla_qmax, sizeof u, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    std::memcpy(out, u, sizeof u);
    return true;
}
bool mla_init() {
    const char* v = std::getenv("HY4_MLA_TC");
    g_mla_tc = false;
    if (const char* qm = std::getenv("HY4_MLA_QMAX"))
        if (std::atoi(qm) != 0 && cudaMalloc((void**) &g_mla_qmax, 2 * sizeof(unsigned)) == cudaSuccess)
            cudaMemset(g_mla_qmax, 0, 2 * sizeof(unsigned));
    if (v && std::atoi(v) == 0) return false;
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess || major < 8) {
        cudaGetLastError();
        return false;
    }
    if (cudaFuncSetAttribute(mla_decode_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcSmem) !=
        cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    g_mla_tc = true;
    return true;
}
size_t mla_scratch_floats(int H, int nsp) { return (size_t) H * nsp * (kMlaV + 2); }
void mla_decode(const float* qa, const float* qf, int qk, int nope, const __half* cache, const int* dpos, int win,
                const int* cells, const int* ncell, const float* sinks, int H, float scale, int nsp, float* scratch,
                float* out, int8_t* oq, float* od, cudaStream_t s, int ring) {
    const size_t smem = ((size_t) kMlaHG * kMlaD + (size_t) kMlaHG * kMlaTile + 3 * kMlaHG) * sizeof(float);   // 41 KB
    // (HY4_MLA_SPLIT_MIN: a split's fewest keys - 64, 32 or 16; 16 on the tensor cores, a tile a block - 10 us a call,
    //  23 with 64 - and 64 on FP32)
    static const int smin = [] {
        const char* v = std::getenv("HY4_MLA_SPLIT_MIN");
        const int m = v ? std::atoi(v) : (g_mla_tc ? 16 : kMlaTile);
        return m == 16 || m == 32 ? m : kMlaTile;
    }();
    if (g_mla_tc)
        mla_decode_tc_kernel<<<dim3(nsp, H / kMlaHG), kTcThreads, kTcSmem, s>>>(qa, qf, qk, nope, cache, dpos, win, cells,
                                                                                ncell, ring, nsp, scale, scratch, smin,
                                                                                g_mla_qmax);
    else
        mla_decode_kernel<<<dim3(nsp, H / kMlaHG), kMlaThreads, smem, s>>>(qa, qf, qk, nope, cache, dpos, win, cells,
                                                                           ncell, ring, nsp, scale, scratch, smin);
    mla_combine_kernel<<<H, kMlaV, 0, s>>>(scratch, nsp, dpos, win, cells ? ncell : nullptr, sinks, out, oq, od, smin);
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
    // (one token: route1_kernel; HY4_ROUTE1=0 the warp kernel, as before)
    static const bool one = !(std::getenv("HY4_ROUTE1") && std::atoi(std::getenv("HY4_ROUTE1")) == 0);
    if (T == 1 && ng == 1 && one && ne <= kRoute1Max && nu <= 16) {
        route1_kernel<<<1, (unsigned) ((ne + 31) / 32 * 32), 0, s>>>(RL, bias, ne, nu, scale, sel, w);
        return;
    }
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

namespace {
// ------------------------------------------------------------------------------------------------ Hy4: hyper-connections
// grid nmix, block 256: mix[i] = (fn[i] . R) * rsqrt(mean(R^2) + eps) over R's hc x n values (fn: nmix rows of hc n F32)
__global__ void hc_mix_kernel(const float* __restrict__ R, const float* __restrict__ fn, int m, float eps,
                              float* __restrict__ mix) {
    const float* w = fn + (size_t) blockIdx.x * m;
    float dot = 0.0f, ss = 0.0f;
    for (int c = threadIdx.x; c < m; c += blockDim.x) {
        const float r = R[c];
        dot = fmaf(w[c], r, dot);
        ss = fmaf(r, r, ss);
    }
    dot = block_sum(dot);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mix[blockIdx.x] = dot * rsqrtf(ss / (float) m + eps);
}
// grid n / 256: x[c] = sum_i pre_i R_i[c], pre_i = sigmoid(mix[i] scale[0] + base[i]) + hc_eps; with post set, block 0
// also post_i = sigmoid(mix[hc + i] scale[1] + base[hc + i]) mag + hc_eps
__global__ void hc_read_kernel(const float* __restrict__ R, const float* __restrict__ mix, const float* __restrict__ scale,
                               const float* __restrict__ base, int hc, int n, float hc_eps, float mag,
                               float* __restrict__ x, float* __restrict__ post) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float pre[8];
    for (int i = 0; i < hc; ++i) pre[i] = sigmoidf_(mix[i] * scale[0] + base[i]) + hc_eps;
    if (post && blockIdx.x == 0 && threadIdx.x < hc) {
        const int i = threadIdx.x;
        post[i] = sigmoidf_(mix[hc + i] * scale[1] + base[hc + i]) * mag + hc_eps;
    }
    if (c >= n) return;
    float v = 0.0f;
    for (int i = 0; i < hc; ++i) v = fmaf(pre[i], R[(size_t) i * n + c], v);
    x[c] = v;
}
// one block of 1024: y = rmsnorm(a (+ b)) w, then R_i += post_i y (identity hyper-connections: no stream mixing)
__global__ void __launch_bounds__(1024) norm_write_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                          const float* __restrict__ w, float eps,
                                                          const float* __restrict__ post, int hc, int n,
                                                          float* __restrict__ R) {
    float ss = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = a[c] + (b ? b[c] : 0.0f);
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / (float) n + eps);
    float p[8];
    for (int i = 0; i < hc; ++i) p[i] = post[i];
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float y = (a[c] + (b ? b[c] : 0.0f)) * sc * w[c];
        for (int i = 0; i < hc; ++i) R[(size_t) i * n + c] = fmaf(p[i], y, R[(size_t) i * n + c]);
    }
}

// ------------------------------------------------------------------------------------------------ a fused boundary
// One sublayer's end and the next one's start in one block of 1024 (the token path): norm_write (y = rmsnorm(a + b) w,
// R_i += post_i y), hc_read's mixes (hc_mix: nmix dots of fn's rows with R over hc n, each scaled by R's rms), its read
// (x = sum_i pre_i R_i, the next post weights) and norm_q8 of x - each phase's sums in its own kernel's order (its
// threads' loops, its block_sum's two stages: 1024 threads for norm_write, 256 for a mix as hc_mix's blocks, 256 for
// norm_q8), so the same bits as the four launches.  n a multiple of 1024 and of 256 x kPerT at most; hc <= 8; nmix
// <= 16.
__device__ __forceinline__ float group_sum(float v, float* part, int g, int wid, int lane) {   // 256-thread block_sum
    v = warp_sum(v);
    if (lane == 0) part[g * 8 + wid] = v;
    __syncthreads();
    float t = lane < 8 ? part[g * 8 + lane] : 0.0f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}
__global__ void __launch_bounds__(1024) hc_step_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                       const float* __restrict__ wpost, const float* post_in, float eps,
                                                       int hc, int n, float* R, const float* __restrict__ fn, int nmix,
                                                       const float* __restrict__ scale, const float* __restrict__ base,
                                                       float hc_eps, float mag, float* __restrict__ mix, float* x,
                                                       float* post_out, const float* __restrict__ wn, float* y,
                                                       int8_t* __restrict__ q, float* __restrict__ d) {
    __shared__ float part[32];
    __shared__ float s_mix[16];
    __shared__ float s_sc;
    {   // (a window: block r the token r's rows - a, b, x, y, q [n], R [hc][n], post [hc], mix [16], d [n / 32])
        const size_t r = blockIdx.x;
        a += r * n;
        if (b) b += r * n;
        post_in += r * hc;
        R += r * hc * n;
        mix += r * 16;
        x += r * n;
        if (post_out) post_out += r * hc;
        if (y) y += r * n;
        q += r * n;
        d += r * (n / 32);
    }
    const int t = threadIdx.x, lane = t & 31, g = t >> 8, tt = t & 255, gw = (t >> 5) & 7;
    // norm_write (its 1024 threads, its order); the post weights read before this kernel writes the next ones
    float p[8];
    for (int i = 0; i < hc; ++i) p[i] = post_in[i];
    {
        float ss = 0.0f;
        for (int c = t; c < n; c += blockDim.x) {
            const float v = a[c] + (b ? b[c] : 0.0f);
            ss = fmaf(v, v, ss);
        }
        ss = block_sum(ss);
        const float sc = rsqrtf(ss / (float) n + eps);
        for (int c = t; c < n; c += blockDim.x) {
            const float yv = (a[c] + (b ? b[c] : 0.0f)) * sc * wpost[c];
            for (int i = 0; i < hc; ++i) R[(size_t) i * n + c] = fmaf(p[i], yv, R[(size_t) i * n + c]);
        }
    }
    __syncthreads();
    // hc_mix: group g takes mixes g, g + 4, ..; each its kernel's 256-thread loop and block_sum
    const int m = hc * n;
    for (int r0 = 0; r0 < nmix; r0 += 4) {
        const int r = r0 + g;
        float dot = 0.0f, ss = 0.0f;
        if (r < nmix) {
            const float* w = fn + (size_t) r * m;
            for (int c = tt; c < m; c += 256) {
                const float rv = R[c];
                dot = fmaf(w[c], rv, dot);
                ss = fmaf(rv, rv, ss);
            }
        }
        __shared__ float gpart[32];
        dot = group_sum(dot, gpart, g, gw, lane);
        ss = group_sum(ss, gpart, g, gw, lane);
        if (tt == 0 && r < nmix) s_mix[r] = dot * rsqrtf(ss / (float) m + eps);
    }
    __syncthreads();
    // hc_read: the pre weights, the next post weights, x
    float pre[8];
    for (int i = 0; i < hc; ++i) pre[i] = sigmoidf_(s_mix[i] * scale[0] + base[i]) + hc_eps;
    if (post_out && t < hc) post_out[t] = sigmoidf_(s_mix[hc + t] * scale[1] + base[hc + t]) * mag + hc_eps;
    if (t < nmix) mix[t] = s_mix[t];
    for (int c = t; c < n; c += blockDim.x) {
        float v = 0.0f;
        for (int i = 0; i < hc; ++i) v = fmaf(pre[i], R[(size_t) i * n + c], v);
        x[c] = v;
    }
    __syncthreads();
    // norm_q8: its sum of squares in its 256 threads' order (group 0), then every thread its values' int8 rows (a warp
    // 32 consecutive ones, as there)
    if (g == 0) {
        const int per = n / 256;
        float ss = 0.0f;
        for (int k = 0; k < per; ++k) {
            const float xc = x[tt + 256 * k];
            ss += xc * xc;
        }
        ss = warp_sum(ss);
        if (lane == 0) part[gw] = ss;
    }
    __syncthreads();
    if (t < 32) {
        float v = lane < 8 ? part[lane] : 0.0f;
        v = warp_sum(v);
        if (t == 0) s_sc = rsqrtf(v / (float) n + eps);
    }
    __syncthreads();
    const float sc = s_sc;
    for (int c = t; c < n; c += blockDim.x) {
        const float av = x[c] * sc * wn[c];
        if (y) y[c] = av;
        warp_q8(av, q + c, d + c / 32);
    }
}
// grid n / 256: R_i = x for every stream (the embedding into the hc streams)
__global__ void hc_init_kernel(const float* __restrict__ x, int hc, int n, float* __restrict__ R) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float v = x[c];
    for (int i = 0; i < hc; ++i) R[(size_t) i * n + c] = v;
}

// ------------------------------------------------------------------------------------------------ Hy4: Engram
// grid cols, block dim (a multiple of 32): row r of rows [cols][dim] RMS-normed (eps) times w [cols][dim] -> y (flat) and
// its int8 rows (q8's 32-value blocks over the flat row)
__global__ void eg_rows_kernel(const float* __restrict__ rows, const float* __restrict__ w, int dim, int wcols,
                               float eps, float* __restrict__ y, int8_t* __restrict__ q, float* __restrict__ d) {
    const size_t at = (size_t) blockIdx.x * dim + threadIdx.x;
    const float v = rows[at];
    const float ss = block_sum(v * v);
    const float o = v * rsqrtf(ss / (float) dim + eps) * w[(size_t) (blockIdx.x % wcols) * dim + threadIdx.x];
    y[at] = o;
    if (q) warp_q8(o, q + at, d + at / 32);
}
// grid hc, block 1024: stream s of R updated with the projected rows kv [(hc + 1)][n] (+ bias kb, same shape): key =
// kv[s], value = kv[hc]; dot = <rms(R_s), rms(key)> . gw[s] / sqrt(n); g = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6)));
// R_s += rms(value) gamma (g + hs[s])   (llama.cpp#32517's build_engram; eps the fp32 machine epsilon)
__global__ void __launch_bounds__(1024) eg_update_kernel(const float* __restrict__ kv, const float* __restrict__ kb,
                                                         const float* __restrict__ gw, const float* __restrict__ hs,
                                                         const float* __restrict__ gamma, int hc, int n, float eps,
                                                         float* __restrict__ R) {
    const int st = blockIdx.x;
    float* Rs = R + (size_t) st * n;
    const float* K = kv + (size_t) st * n;
    const float* Kb = kb + (size_t) st * n;
    const float* V = kv + (size_t) hc * n;
    const float* Vb = kb + (size_t) hc * n;
    float sr = 0.0f, sk = 0.0f, sv = 0.0f, dt = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float r = Rs[c], kk = K[c] + Kb[c], v = V[c] + Vb[c];
        sr = fmaf(r, r, sr);
        sk = fmaf(kk, kk, sk);
        sv = fmaf(v, v, sv);
        dt = fmaf(r * kk, gw[(size_t) st * n + c], dt);
    }
    sr = block_sum(sr);
    sk = block_sum(sk);
    sv = block_sum(sv);
    dt = block_sum(dt);
    const float ir = rsqrtf(sr / (float) n + eps), ik = rsqrtf(sk / (float) n + eps), iv = rsqrtf(sv / (float) n + eps);
    const float dot = dt * ir * ik / sqrtf((float) n);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gt = sigmoidf_(dot < 0.0f ? -mag : mag);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = (V[c] + Vb[c]) * iv * gamma[c];
        Rs[c] = fmaf(v, gt + hs[(size_t) st * n + c], Rs[c]);
    }
}

// ------------------------------------------------------------------------------------------------ Hy4: the latent MoE
// grid n / 256 (n a multiple of 32): y = (flags[0] ? cpu : 0) + (flags[1] ? gpu : 0) and its int8 rows
__global__ void lat_sum_q8_kernel(const float* __restrict__ cpu, const float* __restrict__ gpu,
                                  const int* __restrict__ flags, int n, float* __restrict__ y, int8_t* __restrict__ q,
                                  float* __restrict__ d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float v = 0.0f;
    if (flags[0]) v = cpu[c];
    if (flags[1]) v += gpu[c];
    y[c] = v;
    warp_q8(v, q + c, d + c / 32);
}
}  // namespace

void hc_read(const float* R, const float* fn, int nmix, const float* scale, const float* base, int hc, int n, float eps,
             float hc_eps, float mag, float* mix, float* x, float* post, cudaStream_t s) {
    hc_mix_kernel<<<nmix, 256, 0, s>>>(R, fn, hc * n, eps, mix);
    hc_read_kernel<<<blocks(n, 256), 256, 0, s>>>(R, mix, scale, base, hc, n, hc_eps, mag, x, post);
}
void norm_write(const float* a, const float* b, const float* w, float eps, const float* post, int hc, int n, float* R,
                cudaStream_t s) {
    norm_write_kernel<<<1, 1024, 0, s>>>(a, b, w, eps, post, hc, n, R);
}
bool hc_step(const float* a, const float* b, const float* wpost, const float* post_in, float eps, int hc, int n,
             float* R, const float* fn, int nmix, const float* scale, const float* base, float hc_eps, float mag,
             float* mix, float* x, float* post_out, const float* wn, float* y, int8_t* q, float* d, cudaStream_t s,
             int T) {
    if (n % 1024 != 0 || n / 256 > kPerT || hc > 8 || nmix > 16 || 2 * hc > 16 || T < 1) return false;
    hc_step_kernel<<<T, 1024, 0, s>>>(a, b, wpost, post_in, eps, hc, n, R, fn, nmix, scale, base, hc_eps, mag, mix, x,
                                     post_out, wn, y, q, d);
    return true;
}
void hc_init(const float* x, int hc, int n, float* R, cudaStream_t s) {
    hc_init_kernel<<<blocks(n, 256), 256, 0, s>>>(x, hc, n, R);
}
void eg_rows(const float* rows, const float* w, int cols, int dim, float eps, float* y, int8_t* q, float* d,
             cudaStream_t s, int wcols) {
    eg_rows_kernel<<<cols, dim, 0, s>>>(rows, w, dim, wcols > 0 ? wcols : cols, eps, y, q, d);
}
void eg_update(const float* kv, const float* kb, const float* gw, const float* hs, const float* gamma, int hc, int n,
               float eps, float* R, cudaStream_t s) {
    eg_update_kernel<<<hc, 1024, 0, s>>>(kv, kb, gw, hs, gamma, hc, n, eps, R);
}
void lat_sum_q8(const float* cpu, const float* gpu, const int* flags, int n, float* y, int8_t* q, float* d,
                cudaStream_t s) {
    lat_sum_q8_kernel<<<blocks(n, 256), 256, 0, s>>>(cpu, gpu, flags, n, y, q, d);
}


namespace {
// ------------------------------------------------------------------------------------------------ Hy4: the indexer
// grid 1, block 2 dim: the token's value and gate rows (iv, ig [dim]) into slot pos % nring of ring [nring][2 dim]
// (value | gate; nring a multiple of kpool - a group's members in kpool slots in a row)
__global__ void idx_store_kernel(const float* __restrict__ iv, const float* __restrict__ ig, const int* __restrict__ dpos,
                                 int nring, int dim, float* __restrict__ ring) {
    const int t = threadIdx.x, slot = *dpos % nring;
    ring[(size_t) slot * 2 * dim + t] = t < dim ? iv[t] : ig[t - dim];
}
// grid 1, block dim: when pos ends a group, its pooled key into pooled[pos / kpool]: per element e, the softmax over the
// group's members m of (gate[m][e] + ape[m][e]) weighting value[m][e]; then RMS-normed (eps) times w; then RoPE (NORM
// pairs) on its last 64 values at the group's first position
__global__ void idx_pool_kernel(const float* __restrict__ ring, const float* __restrict__ ape, const float* __restrict__ w,
                                const int* __restrict__ dpos, int kpool, int nring, int dim, float eps,
                                float theta_scale, float* __restrict__ pooled) {
    const int pos = *dpos;
    if ((pos + 1) % kpool != 0) return;
    const int e = threadIdx.x;
    ring += (size_t) ((pos + 1 - kpool) % nring) * 2 * dim;   // (the group's first member's slot)
    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) mx = fmaxf(mx, ring[(size_t) m * 2 * dim + dim + e] + ape[(size_t) m * dim + e]);
    float den = 0.0f, num = 0.0f;
    for (int m = 0; m < kpool; ++m) {
        const float p = expf(ring[(size_t) m * 2 * dim + dim + e] + ape[(size_t) m * dim + e] - mx);
        den += p;
        num += p * ring[(size_t) m * 2 * dim + e];
    }
    const float v = num / den;
    const float ss = block_sum(v * v);
    __shared__ float sv[1024];
    sv[e] = v * rsqrtf(ss / (float) dim + eps) * w[e];
    __syncthreads();
    float* dst = pooled + (size_t) (pos / kpool) * dim;
    const int nope = dim - 64;
    if (e < nope) dst[e] = sv[e];
    else if (((e - nope) & 1) == 0) {
        float y0, y1;
        rope_pair(sv[e], sv[e + 1], pos + 1 - kpool, (e - nope) >> 1, theta_scale, y0, y1);
        dst[e] = y0;
        dst[e + 1] = y1;
    }
}
// grid ceil(max_pools / 8), block 256 (a warp a pool): score[g] = sum_h relu(q_h . pooled_g) w_h scale for the pools
// g < pos / kpool (the complete groups before the token's own)
__global__ void idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w, const float* __restrict__ pooled,
                                 const int* __restrict__ dpos, int kpool, int nh, int dim, float scale,
                                 float* __restrict__ score) {
    const int n_cand = *dpos / kpool;
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (blockIdx.x * 8 >= n_cand) return;
    __shared__ float sq[16 * 128];
    __shared__ float sw[16];
    for (int i = threadIdx.x; i < nh * dim; i += blockDim.x) sq[i] = q[i];
    if (threadIdx.x < nh) sw[threadIdx.x] = w[threadIdx.x] * scale;
    __syncthreads();
    if (g >= n_cand) return;
    const float* k = pooled + (size_t) g * dim;
    float kr[4];
    for (int i = 0; i < 4; ++i) kr[i] = lane + 32 * i < dim ? k[lane + 32 * i] : 0.0f;
    float acc = 0.0f;
    for (int h = 0; h < nh; ++h) {
        float d = 0.0f;
        for (int i = 0; i < 4; ++i)
            if (lane + 32 * i < dim) d = fmaf(sq[h * dim + lane + 32 * i], kr[i], d);
        d = warp_sum(d);
        acc = fmaf(fmaxf(d, 0.0f), sw[h], acc);
    }
    if (lane == 0) score[g] = acc;
}
// one block of 1024: the cells the token at pos attends to - every one up to pos while the complete groups before its
// own are no more than top_pools; else the top top_pools of them by score (their kpool cells each) and its own group's
// cells up to pos.  ncell[0] their count.
__global__ void __launch_bounds__(1024) idx_select_kernel(const float* __restrict__ score, const int* __restrict__ dpos,
                                                          int kpool, int top_pools, int* __restrict__ cells,
                                                          int* __restrict__ ncell) {
    const int pos = *dpos, n_cand = pos / kpool;
    if (n_cand <= top_pools) {
        for (int j = threadIdx.x; j <= pos; j += blockDim.x) cells[j] = j;
        if (threadIdx.x == 0) ncell[0] = pos + 1;
        return;
    }
    strata::kernels::dsa::select_top_pools(score, n_cand, top_pools, kpool, cells);
    const int base = n_cand * kpool, own = pos - base + 1;
    if (threadIdx.x < own) cells[top_pools * kpool + threadIdx.x] = base + threadIdx.x;
    if (threadIdx.x == 0) ncell[0] = top_pools * kpool + own;
}
}  // namespace

void idx_store(const float* iv, const float* ig, const int* dpos, int nring, int dim, float* ring, cudaStream_t s) {
    idx_store_kernel<<<1, 2 * dim, 0, s>>>(iv, ig, dpos, nring, dim, ring);
}
void idx_pool(const float* ring, const float* ape, const float* w, const int* dpos, int kpool, int nring, int dim,
              float eps,
              float theta_scale, float* pooled, cudaStream_t s) {
    idx_pool_kernel<<<1, dim, 0, s>>>(ring, ape, w, dpos, kpool, nring, dim, eps, theta_scale, pooled);
}
void idx_score(const float* q, const float* w, const float* pooled, const int* dpos, int kpool, int nh, int dim,
               float scale, int max_pools, float* score, cudaStream_t s) {
    idx_score_kernel<<<blocks(max_pools, 8), 256, 0, s>>>(q, w, pooled, dpos, kpool, nh, dim, scale, score);
}
void idx_select(const float* score, const int* dpos, int kpool, int top_pools, int* cells, int* ncell, cudaStream_t s) {
    idx_select_kernel<<<1, 1024, 0, s>>>(score, dpos, kpool, top_pools, cells, ncell);
}

namespace {
// ------------------------------------------------------------------------------------------------ Hy4: the prompt path
// grid (nmix, T), block 256: hc_mix_kernel a row: R [T][hc n], mix [T][nmix]
__global__ void hc_mix_rows_kernel(const float* __restrict__ R, const float* __restrict__ fn, int m, int nmix, float eps,
                                   float* __restrict__ mix) {
    const float* r = R + (size_t) blockIdx.y * m;
    const float* w = fn + (size_t) blockIdx.x * m;
    float dot = 0.0f, ss = 0.0f;
    for (int c = threadIdx.x; c < m; c += blockDim.x) {
        const float v = r[c];
        dot = fmaf(w[c], v, dot);
        ss = fmaf(v, v, ss);
    }
    dot = block_sum(dot);
    ss = block_sum(ss);
    if (threadIdx.x == 0) mix[(size_t) blockIdx.y * nmix + blockIdx.x] = dot * rsqrtf(ss / (float) m + eps);
}
// grid (n / 256, T): hc_read_kernel a row (post [T][hc] when set)
__global__ void hc_read_rows_kernel(const float* __restrict__ R, const float* __restrict__ mix, int nmix,
                                    const float* __restrict__ scale, const float* __restrict__ base, int hc, int n,
                                    float hc_eps, float mag, float* __restrict__ x, float* __restrict__ post) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    const float* mx = mix + (size_t) t * nmix;
    const float* Rt = R + (size_t) t * hc * n;
    float pre[8];
    for (int i = 0; i < hc; ++i) pre[i] = sigmoidf_(mx[i] * scale[0] + base[i]) + hc_eps;
    if (post && blockIdx.x == 0 && threadIdx.x < hc) {
        const int i = threadIdx.x;
        post[(size_t) t * hc + i] = sigmoidf_(mx[hc + i] * scale[1] + base[hc + i]) * mag + hc_eps;
    }
    if (c >= n) return;
    float v = 0.0f;
    for (int i = 0; i < hc; ++i) v = fmaf(pre[i], Rt[(size_t) i * n + c], v);
    x[(size_t) t * n + c] = v;
}
// grid T, block 1024: norm_write_kernel a row (a, b [T][n], post [T][hc], R [T][hc][n])
__global__ void __launch_bounds__(1024) norm_write_rows_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                               const float* __restrict__ w, float eps,
                                                               const float* __restrict__ post, int hc, int n,
                                                               float* __restrict__ R) {
    const size_t t = blockIdx.x;
    const float* at = a + t * n;
    const float* bt = b ? b + t * n : nullptr;
    float* Rt = R + t * hc * n;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = at[c] + (bt ? bt[c] : 0.0f);
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss);
    const float sc = rsqrtf(ss / (float) n + eps);
    float p[8];
    for (int i = 0; i < hc; ++i) p[i] = post[t * hc + i];
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float y = (at[c] + (bt ? bt[c] : 0.0f)) * sc * w[c];
        for (int i = 0; i < hc; ++i) Rt[(size_t) i * n + c] = fmaf(p[i], y, Rt[(size_t) i * n + c]);
    }
}
// grid (n / 256, T): R[t][i] = x[t] for each stream
__global__ void hc_init_rows_kernel(const float* __restrict__ x, int hc, int n, float* __restrict__ R) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const float v = x[(size_t) t * n + c];
    for (int i = 0; i < hc; ++i) R[((size_t) t * hc + i) * n + c] = v;
}
// grid (hc, T), block 1024: eg_update_kernel a row (kv [T][(hc + 1) n], R [T][hc][n])
__global__ void __launch_bounds__(1024) eg_update_rows_kernel(const float* __restrict__ kv, const float* __restrict__ kb,
                                                              const float* __restrict__ gw, const float* __restrict__ hs,
                                                              const float* __restrict__ gamma, int hc, int n, float eps,
                                                              float* __restrict__ R) {
    const int st = blockIdx.x;
    const size_t t = blockIdx.y;
    float* Rs = R + (t * hc + st) * n;
    const float* K = kv + t * (hc + 1) * n + (size_t) st * n;
    const float* Kb = kb + (size_t) st * n;
    const float* V = kv + t * (hc + 1) * n + (size_t) hc * n;
    const float* Vb = kb + (size_t) hc * n;
    float sr = 0.0f, sk = 0.0f, sv = 0.0f, dt = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float r = Rs[c], kk = K[c] + Kb[c], v = V[c] + Vb[c];
        sr = fmaf(r, r, sr);
        sk = fmaf(kk, kk, sk);
        sv = fmaf(v, v, sv);
        dt = fmaf(r * kk, gw[(size_t) st * n + c], dt);
    }
    sr = block_sum(sr);
    sk = block_sum(sk);
    sv = block_sum(sv);
    dt = block_sum(dt);
    const float ir = rsqrtf(sr / (float) n + eps), ik = rsqrtf(sk / (float) n + eps), iv = rsqrtf(sv / (float) n + eps);
    const float dot = dt * ir * ik / sqrtf((float) n);
    const float mg = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gt = sigmoidf_(dot < 0.0f ? -mg : mg);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = (V[c] + Vb[c]) * iv * gamma[c];
        Rs[c] = fmaf(v, gt + hs[(size_t) st * n + c], Rs[c]);
    }
}
// grid T H, block 32: Qcat [T][H][576]'s rope part from q [T][H][qk] (its rope values from nope on), pos = pos0 + t
__global__ void qpe_f16_kernel(const float* __restrict__ q, __half* __restrict__ qcat, int H, int qk, int nope, int pos0,
                               float theta_scale) {
    const int t = blockIdx.x / H, h = blockIdx.x % H, i = threadIdx.x;
    const float* p = q + ((size_t) t * H + h) * qk + nope + 2 * i;
    float y0, y1;
    rope_pair(p[0], p[1], pos0 + t, i, theta_scale, y0, y1);
    __half* dst = qcat + ((size_t) t * H + h) * 576 + 512 + 2 * i;
    dst[0] = __float2half(y0);
    dst[1] = __float2half(y1);
}
// grid n, block 576 / 2: rows of a ring (position p in row p % ring) into dst rows [n][576] for positions p0 .. p0 + n
__global__ void ring_get_kernel(const __half* __restrict__ ring_kv, int ring, int p0, __half* __restrict__ dst) {
    const int r = blockIdx.x;
    const __half* src = ring_kv + (size_t) ((p0 + r) % ring) * 576;
    for (int i = threadIdx.x; i < 576; i += blockDim.x) dst[(size_t) r * 576 + i] = src[i];
}
// grid n, block 288: rows [n][576] for positions p0 .. p0 + n into a ring
__global__ void ring_put_kernel(const __half* __restrict__ src, int ring, int p0, __half* __restrict__ ring_kv) {
    const int r = blockIdx.x;
    __half* dst = ring_kv + (size_t) ((p0 + r) % ring) * 576;
    for (int i = threadIdx.x; i < 576; i += blockDim.x) dst[i] = src[(size_t) r * 576 + i];
}
// The scores' softmax (rows r of a tile: token ta + r / H at qpos0 + r / H, head r % H; columns c < nk: the keys at
// positions kpos0 + c): visible when the key's position is at most the token's, within the window (win > 0: more than
// qpos - win), and - for a token whose sel_on is set - in its own group or in a group bits marks (nwords 32-bit words a
// token); S (already scaled) in place -> P (F16, ldp >= nk; the columns from nk to ldp zero); the head's sink (sinks
// set) one more logit in the sum.  grid M, block 256.
__global__ void attn_softmax_kernel(float* __restrict__ S, int lds, __half* __restrict__ P, int ldp, int nk, int H,
                                    int qpos0, int kpos0, int win, const unsigned* __restrict__ bits, int nwords,
                                    const int* __restrict__ sel_on, int kpool, const float* __restrict__ sinks) {
    const int r = blockIdx.x, tl = r / H, h = r % H, qp = qpos0 + tl;
    float* row = S + (size_t) r * lds;
    __half* prow = P + (size_t) r * ldp;
    const bool sel = bits && sel_on[tl];
    const unsigned* bt = sel ? bits + (size_t) tl * nwords : nullptr;
    const int own = sel ? qp / kpool * kpool : 0;
    auto vis = [&](int c) {
        const int kp = kpos0 + c;
        if (kp > qp) return false;
        if (win > 0 && kp <= qp - win) return false;
        if (sel && kp < own) {
            const int gq = kp / kpool;
            return ((bt[gq >> 5] >> (gq & 31)) & 1u) != 0;
        }
        return true;
    };
    float mx = sinks ? sinks[h] : -INFINITY;
    for (int c = threadIdx.x; c < nk; c += blockDim.x)
        if (vis(c)) mx = fmaxf(mx, row[c]);
    mx = block_max(mx);
    float sum = 0.0f;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const float e = vis(c) ? __expf(row[c] - mx) : 0.0f;
        row[c] = e;
        sum += e;
    }
    sum = block_sum(sum);
    if (sinks) sum += __expf(sinks[h] - mx);
    const float inv = 1.0f / sum;
    for (int c = threadIdx.x; c < ldp; c += blockDim.x) prow[c] = __float2half(c < nk ? row[c] * inv : 0.0f);
}
// The indexer's pooled keys for the groups that end in the rows [T] at positions p0 .. p0 + T (g from g0): each member
// from the rows (iv, ig [T][dim]) or, before p0, the ring (slot pos % nring); pooled as idx_pool_kernel.  grid
// n_groups, block dim.
__global__ void idx_pool_rows_kernel(const float* __restrict__ iv, const float* __restrict__ ig,
                                     const float* __restrict__ ring, const float* __restrict__ ape,
                                     const float* __restrict__ w, int p0, int g0, int kpool, int nring, int dim,
                                     float eps, float theta_scale, float* __restrict__ pooled) {
    const int gi = g0 + blockIdx.x, e = threadIdx.x;
    float gv[16], vv[16];
    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) {
        const int p = gi * kpool + m;
        const float a = p >= p0 ? ig[(size_t) (p - p0) * dim + e] : ring[(size_t) (p % nring) * 2 * dim + dim + e];
        const float v = p >= p0 ? iv[(size_t) (p - p0) * dim + e] : ring[(size_t) (p % nring) * 2 * dim + e];
        gv[m] = a + ape[(size_t) m * dim + e];
        vv[m] = v;
        mx = fmaxf(mx, gv[m]);
    }
    float den = 0.0f, num = 0.0f;
    for (int m = 0; m < kpool; ++m) {
        const float q = expf(gv[m] - mx);
        den += q;
        num += q * vv[m];
    }
    const float v = num / den;
    const float ss = block_sum(v * v);
    __shared__ float sv[1024];
    sv[e] = v * rsqrtf(ss / (float) dim + eps) * w[e];
    __syncthreads();
    float* dst = pooled + (size_t) gi * dim;
    const int nope = dim - 64;
    if (e < nope) dst[e] = sv[e];
    else if (((e - nope) & 1) == 0) {
        float y0, y1;
        rope_pair(sv[e], sv[e + 1], gi * kpool, (e - nope) >> 1, theta_scale, y0, y1);
        dst[e] = y0;
        dst[e + 1] = y1;
    }
}
// grid n, block 2 dim: the rows for positions p0 + r0 .. into the ring's slots (the open group's members)
__global__ void idx_ring_put_kernel(const float* __restrict__ iv, const float* __restrict__ ig, int p0, int r0,
                                    int nring, int dim, float* __restrict__ ring) {
    const int r = r0 + blockIdx.x, t = threadIdx.x, slot = (p0 + r) % nring;
    ring[(size_t) slot * 2 * dim + t] = t < dim ? iv[(size_t) r * dim + t] : ig[(size_t) r * dim + t - dim];
}
// grid (ceil(npool / 256), tq), block 256: S [tq][npool] = sum_h relu(A [tq nh][npool]) w [tq][nh] scale, -inf for the
// groups at or after the token's own (g >= pos / kpool)
__global__ void idx_score_rows_kernel(const float* __restrict__ A, const float* __restrict__ w, int nh, int npool,
                                      int qpos0, int kpool, float scale, float* __restrict__ S) {
    const int t = blockIdx.y, gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= npool) return;
    const int ncand = (qpos0 + t) / kpool;
    float acc = 0.0f;
    for (int h = 0; h < nh; ++h) acc = fmaf(fmaxf(A[((size_t) t * nh + h) * npool + gi], 0.0f), w[(size_t) t * nh + h], acc);
    S[(size_t) t * npool + gi] = gi < ncand ? acc * scale : -INFINITY;
}
// grid tq, block 1024: a token's selection - sel_on 0 when its complete groups before its own are no more than
// top_pools (every key visible), else the bits of its top top_pools groups (cells: a scratch row of top_pools kpool ints)
__global__ void __launch_bounds__(1024) idx_select_rows_kernel(const float* __restrict__ S, int npool, int qpos0,
                                                               int kpool, int top_pools, int* __restrict__ cells,
                                                               unsigned* __restrict__ bits, int nwords,
                                                               int* __restrict__ sel_on) {
    const int t = blockIdx.x, ncand = (qpos0 + t) / kpool;
    unsigned* bt = bits + (size_t) t * nwords;
    for (int i = threadIdx.x; i < nwords; i += blockDim.x) bt[i] = 0u;
    if (ncand <= top_pools) {
        if (threadIdx.x == 0) sel_on[t] = 0;
        return;
    }
    int* ct = cells + (size_t) t * top_pools * kpool;
    strata::kernels::dsa::select_top_pools(S + (size_t) t * npool, ncand, top_pools, kpool, ct);
    __syncthreads();
    for (int i = threadIdx.x; i < top_pools; i += blockDim.x) {
        const int gq = ct[i * kpool] / kpool;
        atomicOr(&bt[gq >> 5], 1u << (gq & 31));
    }
    if (threadIdx.x == 0) sel_on[t] = 1;
}
}  // namespace

void hc_read_rows(const float* R, const float* fn, int nmix, const float* scale, const float* base, int hc, int n,
                  float eps, float hc_eps, float mag, float* mix, float* x, float* post, int T, cudaStream_t s) {
    if (T <= 0) return;
    hc_mix_rows_kernel<<<dim3(nmix, T), 256, 0, s>>>(R, fn, hc * n, nmix, eps, mix);
    hc_read_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, s>>>(R, mix, nmix, scale, base, hc, n, hc_eps, mag, x, post);
}
void norm_write_rows(const float* a, const float* b, const float* w, float eps, const float* post, int hc, int n,
                     float* R, int T, cudaStream_t s) {
    if (T > 0) norm_write_rows_kernel<<<T, 1024, 0, s>>>(a, b, w, eps, post, hc, n, R);
}
void hc_init_rows(const float* x, int hc, int n, float* R, int T, cudaStream_t s) {
    if (T > 0) hc_init_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, s>>>(x, hc, n, R);
}
void eg_update_rows(const float* kv, const float* kb, const float* gw, const float* hs, const float* gamma, int hc, int n,
                    float eps, float* R, int T, cudaStream_t s) {
    if (T > 0) eg_update_rows_kernel<<<dim3(hc, T), 1024, 0, s>>>(kv, kb, gw, hs, gamma, hc, n, eps, R);
}
void qpe_f16(const float* q, __half* qcat, int T, int H, int qk, int nope, int pos0, float theta_scale, cudaStream_t s) {
    if (T > 0) qpe_f16_kernel<<<T * H, 32, 0, s>>>(q, qcat, H, qk, nope, pos0, theta_scale);
}
void ring_get(const __half* ring_kv, int ring, int p0, int n, __half* dst, cudaStream_t s) {
    if (n > 0) ring_get_kernel<<<n, 288, 0, s>>>(ring_kv, ring, p0, dst);
}
void ring_put(const __half* src, int ring, int p0, int n, __half* ring_kv, cudaStream_t s) {
    if (n > 0) ring_put_kernel<<<n, 288, 0, s>>>(src, ring, p0, ring_kv);
}
void attn_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int H, int qpos0, int kpos0, int win,
                  const unsigned* bits, int nwords, const int* sel_on, int kpool, const float* sinks, cudaStream_t s) {
    if (M > 0)
        attn_softmax_kernel<<<M, 256, 0, s>>>(S, lds, P, ldp, nk, H, qpos0, kpos0, win, bits, nwords, sel_on, kpool, sinks);
}
void idx_pool_rows(const float* iv, const float* ig, const float* ring, const float* ape, const float* w, int p0, int g0,
                   int n_groups, int kpool, int nring, int dim, float eps, float theta_scale, float* pooled,
                   cudaStream_t s) {
    if (n_groups > 0)
        idx_pool_rows_kernel<<<n_groups, dim, 0, s>>>(iv, ig, ring, ape, w, p0, g0, kpool, nring, dim, eps, theta_scale,
                                                       pooled);
}
void idx_ring_put(const float* iv, const float* ig, int p0, int r0, int n, int nring, int dim, float* ring,
                  cudaStream_t s) {
    if (n > 0) idx_ring_put_kernel<<<n, 2 * dim, 0, s>>>(iv, ig, p0, r0, nring, dim, ring);
}
void idx_score_rows(const float* A, const float* w, int nh, int npool, int tq, int qpos0, int kpool, float scale, float* S,
                    cudaStream_t s) {
    if (tq > 0 && npool > 0)
        idx_score_rows_kernel<<<dim3(blocks(npool, 256), tq), 256, 0, s>>>(A, w, nh, npool, qpos0, kpool, scale, S);
}
void idx_select_rows(const float* S, int npool, int tq, int qpos0, int kpool, int top_pools, int* cells, unsigned* bits,
                     int nwords, int* sel_on, cudaStream_t s) {
    if (tq > 0)
        idx_select_rows_kernel<<<tq, 1024, 0, s>>>(S, npool, qpos0, kpool, top_pools, cells, bits, nwords, sel_on);
}

bool selftest() {
    std::fprintf(stderr, "maya-hy4: no selftest yet (the token path is checked against llama.cpp's logits instead)\n");
    return true;
}

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__global__ void keepalive_kernel(unsigned long long ns) {
    const unsigned long long t0 = global_ns();
    while (global_ns() - t0 < ns) __nanosleep(20000);
}
void keepalive(unsigned long long ns, cudaStream_t s) { keepalive_kernel<<<1, 32, 0, s>>>(ns); }

// ------------------------------------------------------------------------------------------------ the device's routing
namespace {
constexpr unsigned long long kSpinNs = 30ull * 1000000000ull;   // (a spin's bound: %globaltimer's ns, the same on both
                                                                //  cards whatever their clocks)
__global__ void moe_plan_kernel(const int* __restrict__ sel, const float* __restrict__ w, int nu,
                                const float* __restrict__ lat, const int8_t* __restrict__ xq,
                                const float* __restrict__ xd, int nl, const int* __restrict__ where,
                                const uint8_t* const* __restrict__ slotptr, const uint8_t** gptr, int* need, int* plan,
                                const int* __restrict__ dpos) {
    const int t = threadIdx.x;
    __shared__ int s_main[16];
    volatile int* vp = plan;
    if (t < nu) {
        const int e = sel[t], sl = where[e];
        gptr[t] = sl >= 0 ? slotptr[sl] : nullptr;
        s_main[t] = sl >= 0;
        vp[2 + t] = e;
        vp[18 + t] = __float_as_int(w[t]);
        vp[34 + t] = sl;
    }
    // the latent (F32: the CPU lane's), its int8 rows and scales (the helper's), in 16-byte stores (the fence below
    // orders them before the sequence number)
    {
        int4* dl = (int4*) (plan + kPlanHead);
        const int4* sl = (const int4*) lat;
        for (int i = t; i < nl / 4; i += blockDim.x) dl[i] = sl[i];
        int4* dq = (int4*) (plan + kPlanHead + nl);
        const int4* sq = (const int4*) xq;
        for (int i = t; i < nl / 16; i += blockDim.x) dq[i] = sq[i];
        int4* dd = (int4*) (plan + kPlanHead + nl + nl / 4);
        const int4* sd = (const int4*) xd;
        for (int i = t; i < nl / 128; i += blockDim.x) dd[i] = sd[i];
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        int nm = 0;
        for (int i = 0; i < nu; ++i) nm += s_main[i];
        vp[1] = nm;
        *need = nu - nm;
        __threadfence_system();
        vp[0] = dpos[2];
        __threadfence_system();
    }
}

__global__ void moe_wait_kernel(const int* ans, const int* __restrict__ need, const int* __restrict__ dpos, int nl,
                                float* __restrict__ tail, unsigned* diag) {
    __shared__ int s_go;
    const int t = threadIdx.x;
    if (t == 0) {
        const int go = *need > 0;
        if (go) {
            const int want = dpos[2];
            volatile const int* va = ans;
            const unsigned long long t0 = global_ns();
            while (va[0] != want)
                if (global_ns() - t0 > kSpinNs) {   // (a dead host must not hang the GPU: 30 s)
                    if (diag) atomicAdd(diag, 1u);
                    break;
                }
            __threadfence_system();
        }
        s_go = go;
        ((int*) tail)[0] = go;
        ((int*) tail)[1] = 1;
    }
    __syncthreads();
    if (!s_go) return;
    volatile const float* vs = (volatile const float*) (ans + kAnsHead);
    for (int i = t; i < nl; i += blockDim.x) tail[4 + i] = vs[i];
}

__device__ __forceinline__ bool spin_for(const int* p, int want, unsigned* diag) {
    volatile const int* v = p;
    const unsigned long long t0 = global_ns();
    while (v[0] != want)
        if (global_ns() - t0 > kSpinNs) {   // (30 s: a dead peer must not hang the card)
            if (diag) atomicAdd(diag, 1u);
            return false;
        }
    __threadfence_system();
    return true;
}

__global__ void moe_wait2_kernel(const int* ans, const int* hans, const int* __restrict__ need,
                                 const int* __restrict__ dpos, int nl, float* __restrict__ tail, unsigned* diag) {
    __shared__ int s_go, s_cpu, s_nh;
    const int t = threadIdx.x;
    if (t == 0) {
        const int nd = *need, want = dpos[2];
        int nh = 0, cpu = 0;
        if (nd > 0) {
            spin_for(hans, want, diag);
            nh = ((volatile const int*) hans)[1];
            cpu = nd - nh > 0;
            if (cpu) spin_for(ans, want, diag);
        }
        s_go = nd > 0;
        s_cpu = cpu;
        s_nh = nh;
        ((int*) tail)[0] = nd > 0;
        ((int*) tail)[1] = 1;
    }
    __syncthreads();
    if (!s_go) return;
    volatile const float* vc = (volatile const float*) (ans + kAnsHead);
    volatile const float* vh = (volatile const float*) (hans + kAnsHead);
    for (int i = t; i < nl; i += blockDim.x) {
        float v = s_cpu ? vc[i] : 0.0f;
        if (s_nh) v = s_cpu ? v + vh[i] : vh[i];
        tail[4 + i] = v;
    }
}

__global__ void hd_plan_kernel(const int* plan, int nu, int nl, const int* __restrict__ where,
                               const uint8_t* const* __restrict__ slotptr, const uint8_t** gptr, float* __restrict__ w,
                               int8_t* __restrict__ xq, float* __restrict__ xd, int* nh, const int* __restrict__ dpos,
                               unsigned* diag) {
    __shared__ int s_here[16];
    const int t = threadIdx.x;
    if (t == 0) spin_for(plan, dpos[2], diag);
    __syncthreads();
    volatile const int* vp = plan;
    if (t < nu) {
        const int e = vp[2 + t], main_sl = vp[34 + t];
        const int hs = main_sl >= 0 ? -1 : where[e];
        gptr[t] = hs >= 0 ? slotptr[hs] : nullptr;
        w[t] = __int_as_float(vp[18 + t]);
        s_here[t] = hs >= 0;
    }
    {   // (__ldcv: fetched again over PCIe, never a cached line of the last token's)
        const int4* sq = (const int4*) (plan + kPlanHead + nl);
        int4* dq = (int4*) xq;
        for (int i = t; i < nl / 16; i += blockDim.x) dq[i] = __ldcv(sq + i);
        const int4* sd = (const int4*) (plan + kPlanHead + nl + nl / 4);
        int4* dd = (int4*) xd;
        for (int i = t; i < nl / 128; i += blockDim.x) dd[i] = __ldcv(sd + i);
    }
    __syncthreads();
    if (t == 0) {
        int n = 0;
        for (int i = 0; i < nu; ++i) n += s_here[i];
        *nh = n;
    }
}

__global__ void hd_answer_kernel(const float* __restrict__ y, int nu, const int* __restrict__ nh, int nl, int* hans,
                                 const int* __restrict__ dpos) {
    const int t = threadIdx.x;
    volatile float* vs = (volatile float*) (hans + kAnsHead);
    for (int c = t; c < nl; c += blockDim.x) {   // (sum_rows' order: row 0 first)
        float acc = 0.0f;
        for (int i = 0; i < nu; ++i) acc += y[(size_t) i * nl + c];
        vs[c] = acc;
    }
    if (t == 0) ((volatile int*) hans)[1] = *nh;
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        ((volatile int*) hans)[0] = dpos[2];
        __threadfence_system();
    }
}
}  // namespace

void moe_plan(const int* sel, const float* w, int nu, const float* lat, const int8_t* xq, const float* xd, int nl,
              const int* where, const uint8_t* const* slotptr, const uint8_t** gptr, int* need, int* plan,
              const int* dpos, cudaStream_t s) {
    moe_plan_kernel<<<1, 256, 0, s>>>(sel, w, nu, lat, xq, xd, nl, where, slotptr, gptr, need, plan, dpos);
}
void moe_wait(const int* ans, const int* need, const int* dpos, int nl, float* tail, unsigned* diag, cudaStream_t s) {
    moe_wait_kernel<<<1, 256, 0, s>>>(ans, need, dpos, nl, tail, diag);
}
void moe_wait2(const int* ans, const int* hans, const int* need, const int* dpos, int nl, float* tail, unsigned* diag,
               cudaStream_t s) {
    moe_wait2_kernel<<<1, 256, 0, s>>>(ans, hans, need, dpos, nl, tail, diag);
}
void hd_plan(const int* plan, int nu, int nl, const int* where, const uint8_t* const* slotptr, const uint8_t** gptr,
             float* w, int8_t* xq, float* xd, int* nh, const int* dpos, unsigned* diag, cudaStream_t s) {
    hd_plan_kernel<<<1, 256, 0, s>>>(plan, nu, nl, where, slotptr, gptr, w, xq, xd, nh, dpos, diag);
}
void hd_answer(const float* y, int nu, const int* nh, int nl, int* hans, const int* dpos, cudaStream_t s) {
    hd_answer_kernel<<<1, 512, 0, s>>>(y, nu, nh, nl, hans, dpos);
}

// ------------------------------------------------------------------------------------------------ a window's routing
namespace {
__global__ void moe_plan_w_kernel(const int* __restrict__ sel, const float* __restrict__ w, int T, int Tm, int nu,
                                  const float* __restrict__ lat, const int8_t* __restrict__ xq,
                                  const float* __restrict__ xd, int nl, const int* __restrict__ where,
                                  const uint8_t* const* __restrict__ slotptr, const uint8_t** gptr, int* need,
                                  int* plan, const int* __restrict__ dpos) {
    __shared__ int s_nm;
    const int t = threadIdx.x;
    if (t == 0) s_nm = 0;
    __syncthreads();
    volatile int* vp = plan;
    for (int i = t; i < T * nu; i += blockDim.x) {
        const int e = sel[i], sl = where[e];
        gptr[i] = sl >= 0 ? slotptr[sl] : nullptr;
        vp[kWinHead + i] = e;
        vp[kWinHead + 16 * Tm + i] = __float_as_int(w[i]);
        vp[kWinHead + 32 * Tm + i] = sl;
        if (sl >= 0) atomicAdd(&s_nm, 1);
    }
    const size_t L0 = winplan_lat(Tm);
    {
        int4* dl = (int4*) (plan + L0);
        const int4* sl4 = (const int4*) lat;
        for (int i = t; i < T * nl / 4; i += blockDim.x) dl[i] = sl4[i];
        int4* dq = (int4*) (plan + L0 + (size_t) Tm * nl);
        const int4* sq = (const int4*) xq;
        for (int i = t; i < T * nl / 16; i += blockDim.x) dq[i] = sq[i];
        int4* dd = (int4*) (plan + L0 + (size_t) Tm * nl + (size_t) Tm * nl / 4);
        const int4* sd = (const int4*) xd;
        for (int i = t; i < T * nl / 128; i += blockDim.x) dd[i] = sd[i];
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        vp[1] = T;
        vp[2] = s_nm;
        *need = T * nu - s_nm;
        __threadfence_system();
        vp[0] = dpos[2];
        __threadfence_system();
    }
}

__global__ void moe_wait2_w_kernel(const int* ans, const int* hans, const int* __restrict__ need,
                                   const int* __restrict__ dpos, int T, int nl, float* __restrict__ tail,
                                   unsigned* diag) {
    __shared__ int s_go, s_cpu, s_nh;
    const int t = threadIdx.x;
    if (t == 0) {
        const int nd = *need, want = dpos[2];
        int nh = 0, cpu = 0;
        if (nd > 0) {
            spin_for(hans, want, diag);
            nh = ((volatile const int*) hans)[1];
            cpu = nd - nh > 0;
            if (cpu) spin_for(ans, want, diag);
        }
        s_go = nd > 0;
        s_cpu = cpu;
        s_nh = nh;
        ((int*) tail)[0] = nd > 0;
        ((int*) tail)[1] = 1;
    }
    __syncthreads();
    if (!s_go) return;
    volatile const float* vc = (volatile const float*) (ans + kAnsHead);
    volatile const float* vh = (volatile const float*) (hans + kAnsHead);
    for (int i = t; i < T * nl; i += blockDim.x) {
        float v = s_cpu ? vc[i] : 0.0f;
        if (s_nh) v = s_cpu ? v + vh[i] : vh[i];
        tail[4 + i] = v;
    }
}

__global__ void hd_plan_w_kernel(const int* plan, int T, int Tm, int nu, int nl, const int* __restrict__ where,
                                 const uint8_t* const* __restrict__ slotptr, const uint8_t** gptr,
                                 float* __restrict__ w, int8_t* __restrict__ xq, float* __restrict__ xd, int* nh,
                                 const int* __restrict__ dpos, unsigned* diag) {
    __shared__ int s_n;
    const int t = threadIdx.x;
    if (t == 0) {
        s_n = 0;
        spin_for(plan, dpos[2], diag);
    }
    __syncthreads();
    volatile const int* vp = plan;
    for (int i = t; i < T * nu; i += blockDim.x) {
        const int e = vp[kWinHead + i], main_sl = vp[kWinHead + 32 * Tm + i];
        const int hs = main_sl >= 0 ? -1 : where[e];
        gptr[i] = hs >= 0 ? slotptr[hs] : nullptr;
        w[i] = __int_as_float(vp[kWinHead + 16 * Tm + i]);
        if (hs >= 0) atomicAdd(&s_n, 1);
    }
    {
        const size_t L0 = winplan_lat(Tm);
        const int4* sq = (const int4*) (plan + L0 + (size_t) Tm * nl);
        int4* dq = (int4*) xq;
        for (int i = t; i < T * nl / 16; i += blockDim.x) dq[i] = __ldcv(sq + i);
        const int4* sd = (const int4*) (plan + L0 + (size_t) Tm * nl + (size_t) Tm * nl / 4);
        int4* dd = (int4*) xd;
        for (int i = t; i < T * nl / 128; i += blockDim.x) dd[i] = __ldcv(sd + i);
    }
    __syncthreads();
    if (t == 0) *nh = s_n;
}

__global__ void hd_answer_w_kernel(const float* __restrict__ y, int T, int nu, const int* __restrict__ nh, int nl,
                                   int* hans, const int* __restrict__ dpos) {
    const int t = threadIdx.x;
    volatile float* vs = (volatile float*) (hans + kAnsHead);
    for (int i = t; i < T * nl; i += blockDim.x) {   // (each token's rows in its order, as sum_rows)
        const int tk = i / nl, c = i % nl;
        float acc = 0.0f;
        for (int k = 0; k < nu; ++k) acc += y[((size_t) tk * nu + k) * nl + c];
        vs[i] = acc;
    }
    if (t == 0) ((volatile int*) hans)[1] = *nh;
    __threadfence_system();
    __syncthreads();
    if (t == 0) {
        ((volatile int*) hans)[0] = dpos[2];
        __threadfence_system();
    }
}
}  // namespace

void moe_plan_w(const int* sel, const float* w, int T, int Tm, int nu, const float* lat, const int8_t* xq,
                const float* xd, int nl, const int* where, const uint8_t* const* slotptr, const uint8_t** gptr,
                int* need, int* plan, const int* dpos, cudaStream_t s) {
    moe_plan_w_kernel<<<1, 256, 0, s>>>(sel, w, T, Tm, nu, lat, xq, xd, nl, where, slotptr, gptr, need, plan, dpos);
}
void moe_wait2_w(const int* ans, const int* hans, const int* need, const int* dpos, int T, int nl, float* tail,
                 unsigned* diag, cudaStream_t s) {
    moe_wait2_w_kernel<<<1, 512, 0, s>>>(ans, hans, need, dpos, T, nl, tail, diag);
}
void hd_plan_w(const int* plan, int T, int Tm, int nu, int nl, const int* where, const uint8_t* const* slotptr,
               const uint8_t** gptr, float* w, int8_t* xq, float* xd, int* nh, const int* dpos, unsigned* diag,
               cudaStream_t s) {
    hd_plan_w_kernel<<<1, 256, 0, s>>>(plan, T, Tm, nu, nl, where, slotptr, gptr, w, xq, xd, nh, dpos, diag);
}
void hd_answer_w(const float* y, int T, int nu, const int* nh, int nl, int* hans, const int* dpos, cudaStream_t s) {
    hd_answer_w_kernel<<<1, 512, 0, s>>>(y, T, nu, nh, nl, hans, dpos);
}

}  // namespace hy4::k
