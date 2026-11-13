// src/ink/ink_kernels.cu - maya-ink's own device kernels (ink_kernels.cuh).
#include "ink_kernels.cuh"

#include <cuda_bf16.h>

#include <algorithm>

namespace ink::k {

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

// one warp a head; hd a multiple of 32, at most 256
__global__ void head_norm_kernel(float* __restrict__ x, const float* __restrict__ w, int heads, int hd, float eps,
                                 const float* __restrict__ tau) {
    const int row = blockIdx.y, h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= heads) return;
    float* p = x + ((size_t) row * heads + h) * hd;
    float v[8];
    const int per = hd / 32;
    float ss = 0.0f;
    for (int i = 0; i < per; ++i) {
        v[i] = p[lane + 32 * i];
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    float sc = rsqrtf(ss / (float) hd + eps);
    const float t = tau ? tau[row] : 1.0f;
    for (int i = 0; i < per; ++i) p[lane + 32 * i] = v[i] * sc * w[lane + 32 * i] * t;
}

// a thread a channel, the tokens in order (T small: one token, or a short chunk)
__global__ void shortconv_kernel(float* __restrict__ x, float* __restrict__ state, const float* __restrict__ kern, int T,
                                 int ch, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float w[8], win[8];
    for (int i = 0; i < K; ++i) w[i] = kern[(size_t) c * K + i];
    for (int i = 0; i < K - 1; ++i) win[i] = state[(size_t) c * (K - 1) + i];
    for (int t = 0; t < T; ++t) {
        const float xin = x[(size_t) t * ch + c];
        win[K - 1] = xin;
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += win[i] * w[i];   // ggml_ssm_conv's order: oldest input first
        x[(size_t) t * ch + c] = xin + sum;
        for (int i = 0; i < K - 1; ++i) win[i] = win[i + 1];
    }
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = win[i];
}

__global__ void rel_logits_kernel(const float* __restrict__ r, const float* __restrict__ proj,
                                  const float* __restrict__ tau, float* __restrict__ rel, int H, int d_rel, int E) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, t = blockIdx.z;
    if (e >= E) return;
    const float* rv = r + ((size_t) t * H + h) * d_rel;
    float s = 0.0f;
    for (int d = 0; d < d_rel; ++d) s += proj[(size_t) d * E + e] * rv[d];
    if (tau) s *= tau[t];
    rel[((size_t) t * H + h) * E + e] = s;
}

__global__ void kv_store_kernel(const float* __restrict__ k, const float* __restrict__ v, __half* __restrict__ K,
                                __half* __restrict__ V, int kvw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvw) return;
    K[i] = __float2half(k[i]);
    V[i] = __float2half(v[i]);
}
// (the row from the position on the device: ring > 0 pos % ring, else pos)
__global__ void kv_store_dev_kernel(const float* __restrict__ k, const float* __restrict__ v, __half* __restrict__ K,
                                    __half* __restrict__ V, const int* __restrict__ dpos, int ring, int kvw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kvw) return;
    const int pos = *dpos, slot = ring > 0 ? pos % ring : pos;
    K[(size_t) slot * kvw + i] = __float2half(k[i]);
    V[(size_t) slot * kvw + i] = __float2half(v[i]);
}

// ------------------------------------------------------------------------------------------------ fused token kernels
// The token path's small steps fused (the same arithmetic in the same order as the kernels they replace, so the same
// bits): one block of 256 threads for a row of n = 256 x kPerT channels, thread t channels t + 256 k (rmsnorm_kernel's
// order for the sum of squares, block_sum for its total), then each warp's 32-channel blocks rounded as q8_kernel does.
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
    const int lane = t & 31;
#pragma unroll
    for (int k = 0; k < kPerT; ++k) {
        if (k >= per) break;
        const int c = t + 256 * k;
        const float a = xv[k] * sc * w[c];
        if (y) y[c] = a;
        const float am = warp_max(fabsf(a));
        const float s8 = am / 127.0f;
        q[c] = (int8_t) (am > 0.0f ? __float2int_rn(a / s8) : 0);
        if (lane == 0) d[c / 32] = s8;
    }
}

// A residual's update before a norm, a thread a channel: mode 1 x += shortconv(in) (shortconv + add), mode 2
// x += shortconv(moe_tail's sum) (moe_tail).  (Over many blocks: in norm_q8's one block, 4096 channels' convolution
// loads were slower than the kernels they replaced.)
template <int MODE>
__global__ void conv_add_kernel(float* __restrict__ x, const float* __restrict__ in, const float* __restrict__ gpu,
                                const float* __restrict__ so, const int* __restrict__ flags, float* __restrict__ state,
                                const float* __restrict__ kern, int K, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float v;
    if (MODE == 2) {
        if (flags[0]) {
            v = in[c];
            if (flags[1]) v += gpu[c];
        } else {
            v = gpu[c];
        }
        v += so[c];
        v += so[n + c];
    } else {
        v = in[c];
    }
    float wk[8], win[8];
    for (int i = 0; i < K; ++i) wk[i] = kern[(size_t) c * K + i];
    for (int i = 0; i < K - 1; ++i) win[i] = state[(size_t) c * (K - 1) + i];
    win[K - 1] = v;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += win[i] * wk[i];
    const float out = v + sum;
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = win[i + 1];
    x[c] += out;
}

// grid Hkv, 64 threads: warp 0 the KV head's k (shortconv, head_norm_kernel's norm, F16 into K's row), warp 1 its v
// (shortconv, F16 into V's row); the row from the position on the device
__global__ void kv_conv_store_kernel(const float* __restrict__ kk, const float* __restrict__ vv,
                                     float* __restrict__ st_k, float* __restrict__ st_v,
                                     const float* __restrict__ sc_k, const float* __restrict__ sc_v, int K,
                                     const float* __restrict__ k_norm, float eps, __half* __restrict__ Kc,
                                     __half* __restrict__ Vc, const int* __restrict__ dpos, int ring, int hd, int kvw) {
    const int h = blockIdx.x, wv = threadIdx.x >> 5, lane = threadIdx.x & 31, per = hd / 32;
    const int pos = *dpos, slot = ring > 0 ? pos % ring : pos;
    const float* in = wv ? vv : kk;
    float* st = wv ? st_v : st_k;
    const float* kern = wv ? sc_v : sc_k;
    float v[8];
    for (int i = 0; i < per; ++i) {
        const int c = h * hd + lane + 32 * i;
        float wk[8], win[8];
        for (int j = 0; j < K; ++j) wk[j] = kern[(size_t) c * K + j];
        for (int j = 0; j < K - 1; ++j) win[j] = st[(size_t) c * (K - 1) + j];
        const float xin = in[c];
        win[K - 1] = xin;
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) sum += win[j] * wk[j];
        v[i] = xin + sum;
        for (int j = 0; j < K - 1; ++j) st[(size_t) c * (K - 1) + j] = win[j + 1];
    }
    if (wv == 0) {
        float ss = 0.0f;
        for (int i = 0; i < per; ++i) ss += v[i] * v[i];
        ss = warp_sum(ss);
        const float sc = rsqrtf(ss / (float) hd + eps);
        for (int i = 0; i < per; ++i) v[i] = v[i] * sc * k_norm[lane + 32 * i];
    }
    __half* dst = (wv ? Vc : Kc) + (size_t) slot * kvw + (size_t) h * hd;
    for (int i = 0; i < per; ++i) dst[lane + 32 * i] = __float2half(v[i]);
}

// A MoE layer's end in one kernel, a thread a channel: fo = (the CPU lane's sum, then + the VRAM experts' when both
// ran; else the one that did) + the shared experts' two outputs, then the FFN's short convolution (shortconv_kernel's
// arithmetic) and x += it.  flags[0]: the CPU lane ran, flags[1]: the VRAM experts did.
__global__ void moe_tail_kernel(float* __restrict__ x, const float* __restrict__ cpu, const float* __restrict__ gpu,
                                const float* __restrict__ so, const int* __restrict__ flags, float* __restrict__ state,
                                const float* __restrict__ kern, int n, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float v;
    if (flags[0]) {
        v = cpu[c];
        if (flags[1]) v += gpu[c];
    } else {
        v = gpu[c];
    }
    v += so[c];
    v += so[n + c];
    float w[8], win[8];
    for (int i = 0; i < K; ++i) w[i] = kern[(size_t) c * K + i];
    for (int i = 0; i < K - 1; ++i) win[i] = state[(size_t) c * (K - 1) + i];
    win[K - 1] = v;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += win[i] * w[i];   // ggml_ssm_conv's order: oldest input first
    const float out = v + sum;
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = win[i + 1];
    x[c] += out;
}

// The decode attention, a block a (split of the keys, KV head): the split's scores for the group's G query heads (a
// thread a key, its K row read once for all G - every thread's loads independent, so they are in flight together), the
// split's softmax (a warp a head), then its weighted V rows (a thread a dimension, the keys unrolled); the splits'
// (max, sum, acc) merged by attn_combine_kernel.  The split's size from the keys there are, so the grid can stay the same
// from one token to the next (a CUDA graph): a ring of n_swa rows in splits of kRingSplit keys, a full cache in at most
// nsp_max splits of a multiple of 64 keys.  dpos (when set): the query's position read on the device.
constexpr int kAttnThreads = 128, kRingSplit = 64, kMaxG = 8, kAttnMaxSplit = 1024;

__host__ __device__ inline int attn_n_kv(int pos, int ring) { return ring > 0 ? min(pos + 1, ring) : pos + 1; }
__host__ __device__ inline int attn_split_keys(int n_kv, int ring, int nsp_max) {
    if (ring > 0) return kRingSplit;
    const int per = (n_kv + nsp_max - 1) / nsp_max;
    return max(64, (per + 63) / 64 * 64);
}

// (fused) qn: q's head norm done here (head_norm_kernel's arithmetic, a warp a head, times tau when set); rr / proj: the
// bias band computed for each key's distance (rel_logits_kernel's arithmetic) instead of read from rel.
__global__ void __launch_bounds__(kAttnThreads) attn_decode_kernel(const float* __restrict__ q,
                                                                   const __half* __restrict__ K,
                                                                   const __half* __restrict__ V, int pos_arg,
                                                                   const int* __restrict__ dpos, int ring, int nsp_max,
                                                                   const float* __restrict__ rel, int E, int H, int Hkv,
                                                                   int hd, float scale, float* __restrict__ part,
                                                                   const float* __restrict__ qn,
                                                                   const float* __restrict__ tau, float eps,
                                                                   const float* __restrict__ rr,
                                                                   const float* __restrict__ proj, int d_rel) {
    extern __shared__ float sm[];   // q [G][hd], then the scores / weights [G][S], then r [G][d_rel]
    const int split = blockIdx.x, g = blockIdx.y, t = threadIdx.x, G = H / Hkv;
    const int pos = dpos ? *dpos : pos_arg, n_kv = attn_n_kv(pos, ring), S = attn_split_keys(n_kv, ring, nsp_max);
    const int j0 = split * S, j1 = min(n_kv, j0 + S);
    if (j0 >= n_kv) return;   // (a split past the keys: the combine reads only those before)
    float* qs = sm;
    float* sc = sm + (size_t) G * hd;
    float* rs = sc + (size_t) G * S;
    if (qn) {
        const int w = t >> 5, lane = t & 31, per = hd / 32;
        const float tv = tau ? tau[0] : 1.0f;
        for (int hl = w; hl < G; hl += kAttnThreads / 32) {
            const float* p = q + ((size_t) g * G + hl) * hd;
            float v[8];
            float ss = 0.0f;
            for (int i = 0; i < per; ++i) {
                v[i] = p[lane + 32 * i];
                ss += v[i] * v[i];
            }
            ss = warp_sum(ss);
            const float s0 = rsqrtf(ss / (float) hd + eps);
            for (int i = 0; i < per; ++i) qs[(size_t) hl * hd + lane + 32 * i] = v[i] * s0 * qn[lane + 32 * i] * tv;
        }
    } else {
        for (int i = t; i < G * hd; i += kAttnThreads) qs[i] = q[(size_t) g * G * hd + i];
    }
    if (rr)
        for (int i = t; i < G * d_rel; i += kAttnThreads) rs[i] = rr[(size_t) g * G * d_rel + i];
    __syncthreads();
    const size_t row = (size_t) Hkv * hd;
    // the scores: thread t takes keys j0 + t, j0 + t + 128, ...
    for (int j = j0 + t; j < j1; j += kAttnThreads) {
        const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t) j * row + (size_t) g * hd);
        float acc[kMaxG];
#pragma unroll
        for (int h = 0; h < kMaxG; ++h) acc[h] = 0.0f;
        for (int c0 = 0; c0 < hd / 8; c0 += 4) {   // 4 x 8 values in flight
            uint4 kv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) kv[u] = c0 + u < hd / 8 ? kr[c0 + u] : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (c0 + u >= hd / 8) break;
                const __half2* k2 = reinterpret_cast<const __half2*>(&kv[u]);
                float kf[8];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f2 = __half22float2(k2[e]);
                    kf[2 * e] = f2.x;
                    kf[2 * e + 1] = f2.y;
                }
                const int d0 = (c0 + u) * 8;
#pragma unroll
                for (int h = 0; h < kMaxG; ++h) {
                    if (h >= G) break;
                    const float* qh = qs + (size_t) h * hd + d0;
                    float s = acc[h];
#pragma unroll
                    for (int e = 0; e < 8; ++e) s = fmaf(qh[e], kf[e], s);
                    acc[h] = s;
                }
            }
        }
        const int dist = ring > 0 ? ((pos - j) % ring + ring) % ring : pos - j;
        if (rr) {
            const bool band = dist >= 0 && dist < E;
            float pc[32];   // (the band's column at this distance)
            if (band)
                for (int d = 0; d < d_rel; ++d) pc[d] = __ldg(proj + (size_t) d * E + dist);
            const float tv = tau ? tau[0] : 1.0f;
            for (int h = 0; h < G; ++h) {
                float s = acc[h] * scale;
                if (band) {
                    float b = 0.0f;
                    for (int d = 0; d < d_rel; ++d) b += pc[d] * rs[(size_t) h * d_rel + d];
                    if (tau) b *= tv;
                    s += b;
                }
                sc[(size_t) h * S + (j - j0)] = s;
            }
        } else {
            for (int h = 0; h < G; ++h) {
                float s = acc[h] * scale;
                if (dist >= 0 && dist < E) s += rel[(size_t) (g * G + h) * E + dist];
                sc[(size_t) h * S + (j - j0)] = s;
            }
        }
    }
    __syncthreads();
    // the split's softmax, a warp a head: (m, l) and the weights exp(s - m) in place
    const int w = t >> 5, lane = t & 31, nk = j1 - j0, nsp = gridDim.x;
    for (int h = w; h < G; h += kAttnThreads / 32) {
        float* sh = sc + (size_t) h * S;
        float m = -INFINITY;
        for (int i = lane; i < nk; i += 32) m = fmaxf(m, sh[i]);
        m = warp_max(m);
        float l = 0.0f;
        for (int i = lane; i < nk; i += 32) {
            const float p = __expf(sh[i] - m);
            sh[i] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) {
            float* pp = part + ((size_t) (g * G + h) * nsp + split) * (hd + 2);
            pp[0] = m;
            pp[1] = l;
        }
    }
    __syncthreads();
    // the weighted V rows: thread t a dimension (and t + 128, ...), the keys 8 at a time
    for (int d = t; d < hd; d += kAttnThreads) {
        float o[kMaxG];
#pragma unroll
        for (int h = 0; h < kMaxG; ++h) o[h] = 0.0f;
        const __half* vb = V + (size_t) j0 * row + (size_t) g * hd + d;
        int i = 0;
        for (; i + 8 <= nk; i += 8) {
            float vv[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) vv[u] = __half2float(vb[(size_t) (i + u) * row]);
#pragma unroll
            for (int h = 0; h < kMaxG; ++h) {
                if (h >= G) break;
                const float* ph = sc + (size_t) h * S + i;
                float a = o[h];
#pragma unroll
                for (int u = 0; u < 8; ++u) a = fmaf(ph[u], vv[u], a);
                o[h] = a;
            }
        }
        for (; i < nk; ++i) {
            const float v = __half2float(vb[(size_t) i * row]);
            for (int h = 0; h < G; ++h) o[h] = fmaf(sc[(size_t) h * S + i], v, o[h]);
        }
        for (int h = 0; h < G; ++h) part[((size_t) (g * G + h) * nsp + split) * (hd + 2) + 2 + d] = o[h];
    }
}

// grid H, blockDim hd: the splits' partial sums merged (the splits that hold keys)
__global__ void attn_combine_kernel(const float* __restrict__ part, int nsp, int pos_arg, const int* __restrict__ dpos,
                                    int ring, int nsp_max, int hd, float* __restrict__ out, int8_t* __restrict__ oq,
                                    float* __restrict__ od) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int pos = dpos ? *dpos : pos_arg, n_kv = attn_n_kv(pos, ring), S = attn_split_keys(n_kv, ring, nsp_max);
    const int used = min(nsp, (n_kv + S - 1) / S);
    const float* pp = part + (size_t) h * nsp * (hd + 2);
    float M = -INFINITY;
#pragma unroll 8
    for (int s = 0; s < used; ++s) M = fmaxf(M, pp[(size_t) s * (hd + 2)]);
    float L = 0.0f, o = 0.0f;
#pragma unroll 8
    for (int s = 0; s < used; ++s) {
        const float* p = pp + (size_t) s * (hd + 2);
        const float m = p[0];
        const float c = m == -INFINITY ? 0.0f : __expf(m - M);
        L += p[1] * c;
        o += p[2 + d] * c;
    }
    const float v = L > 0.0f ? o / L : 0.0f;
    out[(size_t) h * hd + d] = v;
    if (oq) {   // (fused) q8_kernel's rounding of the row: a warp a block of 32 (blockDim = hd, a multiple of 32)
        const float am = warp_max(fabsf(v));
        const float s8 = am / 127.0f;
        const size_t c = (size_t) h * hd + d;
        oq[c] = (int8_t) (am > 0.0f ? __float2int_rn(v / s8) : 0);
        if ((d & 31) == 0) od[c / 32] = s8;
    }
}

// a warp a block of 32 values
__global__ void q8_kernel(const float* __restrict__ x, int k, int8_t* __restrict__ q, float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * k + (size_t) b * 32 + lane];
    const float am = warp_max(fabsf(v));
    const float sc = am / 127.0f;
    const int qi = am > 0.0f ? __float2int_rn(v / sc) : 0;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) qi;
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}

__device__ __forceinline__ float load_f(const void* w, int type, size_t off) {
    if (type == 0) return ((const float*) w)[off];
    if (type == 1) return __half2float(((const __half*) w)[off]);
    return __bfloat162float(((const __nv_bfloat16*) w)[off]);
}

// a block a row
__global__ void gemv_f_kernel(const void* __restrict__ w, int type, size_t rb, const float* __restrict__ x,
                              float* __restrict__ y, int k) {
    const int r = blockIdx.x;
    const char* row = (const char*) w + (size_t) r * rb;
    float s = 0.0f;
    for (int c = threadIdx.x; c < k; c += blockDim.x) s += load_f(row, type, c) * x[c];
    s = block_sum(s);
    if (threadIdx.x == 0) y[r] = s;
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ y, int n,
                              const float* __restrict__ scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    float v = gv / (1.0f + __expf(-gv)) * u[i];
    if (scale) v *= scale[0];
    y[i] = v;
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void mul_scalar_kernel(float* __restrict__ y, const float* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= a[0];
}
__global__ void f16_to_f32_kernel(const __half* __restrict__ x, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
}

__global__ void row_to_f32_kernel(const void* __restrict__ x, int type, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = load_f(x, type, i);
}

inline unsigned blocks(int n, int t) { return (unsigned) ((n + t - 1) / t); }
inline unsigned blocks_l(size_t n, int t) { return (unsigned) ((n + (size_t) t - 1) / (size_t) t); }

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

// ------------------------------------------------------------------------------------------------ the prompt's chunks
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

__global__ void sconv_chunk_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ state,
                                   const float* __restrict__ kern, int T, int ch, int K, int add) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ch) return;
    const int t = (int) (i / ch), c = (int) (i % ch);
    float s = 0.0f;
    for (int k = 0; k < K; ++k) {   // ggml_ssm_conv's order: the oldest input first
        const int tt = t - (K - 1) + k;
        const float v = tt >= 0 ? x[(size_t) tt * ch + c] : state[(size_t) c * (K - 1) + (tt + K - 1)];
        s += v * kern[(size_t) c * K + k];
    }
    const float out = x[i] + s;
    if (add) y[i] += out;
    else y[i] = out;
}
__global__ void sconv_state_kernel(const float* __restrict__ x, float* __restrict__ state, int T, int ch, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float nw[8];
    for (int i = 0; i < K - 1; ++i) {
        const int tt = T - (K - 1) + i;
        nw[i] = tt >= 0 ? x[(size_t) tt * ch + c] : state[(size_t) c * (K - 1) + (tt + K - 1)];
    }
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = nw[i];
}

__global__ void to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void q_group_kernel(const float* __restrict__ q, __half* __restrict__ qh, int T, int H, int Hkv, int hd) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * H * hd) return;
    const int d = (int) (i % hd), h = (int) ((i / hd) % H), t = (int) (i / ((size_t) hd * H));
    const int G = H / Hkv, g = h / G, hh = h % G;
    qh[(((size_t) g * T + t) * G + hh) * hd + d] = __float2half(q[i]);
}

__global__ void ring_window_kernel(const __half* __restrict__ src, __half* __restrict__ dst, int p0, int j0, int j1,
                                   int R, int kvw, int to_ring) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) (j1 - j0) * kvw;
    if (i >= n) return;
    const int j = j0 + (int) (i / kvw), c = (int) (i % kvw);
    const size_t rr = (size_t) ((p0 + j) % R) * kvw + c, ww = (size_t) j * kvw + c;
    if (to_ring) dst[rr] = src[ww];
    else dst[ww] = src[rr];
}

__global__ void attn_softmax_kernel(float* __restrict__ S, int lds, __half* __restrict__ P, int ldp, int nk, int G,
                                    int g, int ta, int pos0, int kp0, int window, const float* __restrict__ R,
                                    const float* __restrict__ proj, int E, int H, int d_rel,
                                    const float* __restrict__ tau) {
    const int r = blockIdx.x, t = r / G, hh = r % G, h = g * G + hh, tok = ta + t, p = pos0 + tok;
    float rv[32];
    for (int d = 0; d < d_rel; ++d) rv[d] = R[((size_t) tok * H + h) * d_rel + d];
    const float tv = tau ? tau[tok] : 1.0f;
    float* Sr = S + (size_t) r * lds;
    __half* Pr = P + (size_t) r * ldp;
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const int dist = p - (kp0 + c);
        float v = -INFINITY;
        if (dist >= 0 && (window <= 0 || dist < window)) {
            v = Sr[c];
            if (dist < E) {
                float b = 0.0f;
                for (int d = 0; d < d_rel; ++d) b += rv[d] * __ldg(proj + (size_t) d * E + dist);
                v += b * tv;
            }
        }
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

__global__ void attn_scatter_kernel(const float* __restrict__ O, float* __restrict__ out, int M, int G, int g, int ta,
                                    int H, int hd) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) M * hd) return;
    const int r = (int) (i / hd), d = (int) (i % hd), t = r / G, hh = r % G;
    out[(((size_t) (ta + t)) * H + g * G + hh) * hd + d] = O[i];
}

__device__ __forceinline__ float log_sigmoid(float z) {
    return z >= 0.0f ? -log1pf(expf(-z)) : z - log1pf(expf(z));
}

// a warp a token: lane l holds experts l, l + 32, ...
__global__ void route_kernel(const float* __restrict__ RL, int ld, const float* __restrict__ bias, int T, int ne,
                             int nsh, int nu, float scale, int* __restrict__ sel, float* __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float* rl = RL + (size_t) t * ld;
    float sc[16];
    const int per = (ne + 31) / 32;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        sc[i] = e < ne ? 1.0f / (1.0f + expf(-rl[e])) + bias[e] : -INFINITY;
    }
    int chosen[32];
    for (int k = 0; k < nu; ++k) {
        float bv = -INFINITY;
        int be = 0x7fffffff;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (sc[i] > bv || (sc[i] == bv && e < be)) {
                bv = sc[i];
                be = e;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
            }
        }
        chosen[k] = be;
        if ((be & 31) == lane) sc[be >> 5] = -INFINITY;
    }
    if (lane == 0) {
        float wv[32], mx = -INFINITY;
        const int nw = nu + nsh;
        for (int i = 0; i < nw; ++i) {
            wv[i] = log_sigmoid(i < nu ? rl[chosen[i]] : rl[ne + (i - nu)]);
            mx = fmaxf(mx, wv[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < nw; ++i) {
            wv[i] = expf(wv[i] - mx);
            sum += wv[i];
        }
        for (int i = 0; i < nu; ++i) sel[(size_t) t * nu + i] = chosen[i];
        for (int i = 0; i < nw; ++i) w[(size_t) t * nw + i] = wv[i] / sum * scale;
    }
}

__global__ void expert_count_kernel(const int* __restrict__ sel, int n, int* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + sel[i], 1);
}

// a warp an expert: its tokens compacted in order
__global__ void expert_lists_kernel(const int* __restrict__ sel, const float* __restrict__ w, int T, int nu, int ldw,
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
            const int t = j / nu;
            list[at] = t;
            wl[at] = w[(size_t) t * ldw + (j % nu)];
        }
        base += __popc(b);
    }
}

__global__ void swiglu_rows_kernel(const float* __restrict__ gu, int n, int ff, const float* __restrict__ gamma,
                                   int gstride, __half* __restrict__ h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * ff) return;
    const int r = (int) (i / ff), c = (int) (i % ff);
    const float g = gu[(size_t) r * 2 * ff + c], u = gu[(size_t) r * 2 * ff + ff + c];
    float v = g / (1.0f + expf(-g)) * u;
    if (gamma) v *= gamma[(size_t) r * gstride];
    h[i] = __float2half(v);
}

__global__ void scatter_add_kernel(const float* __restrict__ ye, const int* __restrict__ list,
                                   const float* __restrict__ wl, int rows, int n, float* __restrict__ y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * n) return;
    const int r = (int) (i / n), c = (int) (i % n);
    y[(size_t) list[r] * n + c] += wl[r] * ye[i];
}

__global__ void swiglu_f32_rows_kernel(const float* __restrict__ gu, int n, int ff, float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ff) return;
    const int r = i / ff, c = i % ff;
    const float g = gu[(size_t) r * 2 * ff + c], u = gu[(size_t) r * 2 * ff + ff + c];
    h[i] = g / (1.0f + expf(-g)) * u;
}
__global__ void sum_rows_kernel(const float* __restrict__ y, int n, int m, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= m) return;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += y[(size_t) i * m + c];
    out[c] = s;
}

__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ a, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i] * (a ? a[0] : 1.0f);
}
__global__ void scale_dev_kernel(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ a,
                                 size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * a[0];
}

}  // namespace

bool embed_rows(const uint8_t* w, int type, size_t rb, const int* ids, int T, int n, float* x, cudaStream_t s) {
    if (T <= 0) return true;
    if (type == 8) embed_q8_0_kernel<<<T, 256, 0, s>>>(w, rb, ids, n, x);
    else if (type == 0 || type == 1 || type == 30) embed_f_kernel<<<T, 256, 0, s>>>(w, type, rb, ids, n, x);
    else return false;
    return true;
}
void sconv_chunk(const float* x, float* y, const float* state, const float* kern, int T, int ch, int K, bool add,
                 cudaStream_t s) {
    const size_t n = (size_t) T * ch;
    if (n) sconv_chunk_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, y, state, kern, T, ch, K, add ? 1 : 0);
}
void sconv_state(const float* x, float* state, int T, int ch, int K, cudaStream_t s) {
    sconv_state_kernel<<<blocks(ch, 256), 256, 0, s>>>(x, state, T, ch, K);
}
void to_f16(const float* x, __half* y, size_t n, cudaStream_t s) {
    if (n) to_f16_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, y, n);
}
void q_group(const float* q, __half* qh, int T, int H, int Hkv, int hd, cudaStream_t s) {
    const size_t n = (size_t) T * H * hd;
    if (n) q_group_kernel<<<blocks_l(n, 256), 256, 0, s>>>(q, qh, T, H, Hkv, hd);
}
void ring_to_window(const __half* ring, __half* win, int p0, int n, int ring_rows, int kvw, cudaStream_t s) {
    const size_t m = (size_t) n * kvw;
    if (m) ring_window_kernel<<<blocks_l(m, 256), 256, 0, s>>>(ring, win, p0, 0, n, ring_rows, kvw, 0);
}
void window_to_ring(const __half* win, __half* ring, int p0, int j0, int j1, int ring_rows, int kvw, cudaStream_t s) {
    const size_t m = (size_t) (j1 - j0) * kvw;
    if (j1 > j0) ring_window_kernel<<<blocks_l(m, 256), 256, 0, s>>>(win, ring, p0, j0, j1, ring_rows, kvw, 1);
}
void attn_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int G, int g, int ta, int pos0, int kp0,
                  int window, const float* R, const float* proj, int E, int H, int d_rel, const float* tau,
                  cudaStream_t s) {
    if (M > 0)
        attn_softmax_kernel<<<M, 256, 0, s>>>(S, lds, P, ldp, nk, G, g, ta, pos0, kp0, window, R, proj, E, H, d_rel,
                                              tau);
}
void attn_scatter(const float* O, float* out, int M, int G, int g, int ta, int H, int hd, cudaStream_t s) {
    const size_t n = (size_t) M * hd;
    if (n) attn_scatter_kernel<<<blocks_l(n, 256), 256, 0, s>>>(O, out, M, G, g, ta, H, hd);
}
void route(const float* RL, int ld, const float* bias, int T, int ne, int nsh, int nu, float scale, int* sel, float* w,
           cudaStream_t s) {
    const int wpb = 4;
    if (T > 0) route_kernel<<<blocks(T, wpb), 32 * wpb, 0, s>>>(RL, ld, bias, T, ne, nsh, nu, scale, sel, w);
}
void expert_count(const int* sel, int n, int* counts, cudaStream_t s) {
    if (n > 0) expert_count_kernel<<<blocks(n, 256), 256, 0, s>>>(sel, n, counts);
}
void expert_lists(const int* sel, const float* w, int T, int nu, int ldw, int ne, const int* off, int* list, float* wl,
                  cudaStream_t s) {
    expert_lists_kernel<<<ne, 32, 0, s>>>(sel, w, T, nu, ldw, off, list, wl);
}
void swiglu_rows(const float* gu, int n, int ff, const float* gamma, int gstride, __half* h, cudaStream_t s) {
    const size_t m = (size_t) n * ff;
    if (m) swiglu_rows_kernel<<<blocks_l(m, 256), 256, 0, s>>>(gu, n, ff, gamma, gstride, h);
}
void scatter_add(const float* ye, const int* list, const float* wl, int rows, int n, float* y, cudaStream_t s) {
    const size_t m = (size_t) rows * n;
    if (m) scatter_add_kernel<<<blocks_l(m, 256), 256, 0, s>>>(ye, list, wl, rows, n, y);
}
void axpy(float* y, const float* x, const float* a, size_t n, cudaStream_t s) {
    if (n) axpy_kernel<<<blocks_l(n, 256), 256, 0, s>>>(y, x, a, n);
}
void scale_dev(float* y, const float* x, const float* a, size_t n, cudaStream_t s) {
    if (n) scale_dev_kernel<<<blocks_l(n, 256), 256, 0, s>>>(y, x, a, n);
}
void swiglu_f32_rows(const float* gu, int n, int ff, float* h, cudaStream_t s) {
    if (n * ff > 0) swiglu_f32_rows_kernel<<<blocks(n * ff, 256), 256, 0, s>>>(gu, n, ff, h);
}
void sum_rows(const float* y, int n, int m, float* out, cudaStream_t s) {
    sum_rows_kernel<<<blocks(m, 256), 256, 0, s>>>(y, n, m, out);
}


void rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps);
}
void head_norm(float* x, const float* w, int rows, int heads, int hd, float eps, const float* tau, cudaStream_t s) {
    const int wpb = 8;
    head_norm_kernel<<<dim3(blocks(heads, wpb), rows), 32 * wpb, 0, s>>>(x, w, heads, hd, eps, tau);
}
void shortconv(float* x, float* state, const float* kern, int T, int ch, int K, cudaStream_t s) {
    shortconv_kernel<<<blocks(ch, 256), 256, 0, s>>>(x, state, kern, T, ch, K);
}
void rel_logits(const float* r, const float* proj, const float* tau, float* rel, int T, int H, int d_rel, int E,
                cudaStream_t s) {
    rel_logits_kernel<<<dim3(blocks(E, 128), H, T), 128, 0, s>>>(r, proj, tau, rel, H, d_rel, E);
}
void kv_store(const float* k, const float* v, __half* K, __half* V, int slot, int kvw, cudaStream_t s) {
    kv_store_kernel<<<blocks(kvw, 256), 256, 0, s>>>(k, v, K + (size_t) slot * kvw, V + (size_t) slot * kvw, kvw);
}
void kv_store_dev(const float* k, const float* v, __half* K, __half* V, const int* dpos, int ring, int kvw,
                  cudaStream_t s) {
    kv_store_dev_kernel<<<blocks(kvw, 256), 256, 0, s>>>(k, v, K, V, dpos, ring, kvw);
}
void moe_tail(float* x, const float* cpu, const float* gpu, const float* so, const int* flags, float* state,
              const float* kern, int n, int K, cudaStream_t s) {
    moe_tail_kernel<<<blocks(n, 256), 256, 0, s>>>(x, cpu, gpu, so, flags, state, kern, n, K);
}
int attn_nsp(int ring, int rows) {
    if (ring > 0) return (ring + kRingSplit - 1) / kRingSplit;
    return std::max(128, (rows + kAttnMaxSplit - 1) / kAttnMaxSplit);
}
size_t attn_scratch_floats(int H, int hd, int nsp) { return (size_t) H * nsp * (hd + 2); }
void attn_decode(const float* q, const __half* K, const __half* V, int rows, int pos, const int* dpos, int ring,
                 const float* rel, int E, int H, int Hkv, int hd, float scale, float* out, float* scratch,
                 cudaStream_t s, const AttnFuse* fz) {
    const int nsp = attn_nsp(ring, rows), G = H / Hkv;
    const int S_max = attn_split_keys(ring > 0 ? ring : rows, ring, nsp);
    const int d_rel = fz && fz->rr ? fz->d_rel : 0;
    const size_t smem = ((size_t) G * (hd + S_max) + (size_t) G * d_rel) * sizeof(float);
    attn_decode_kernel<<<dim3(nsp, Hkv), kAttnThreads, smem, s>>>(
        q, K, V, pos, dpos, ring, nsp, rel, E, H, Hkv, hd, scale, scratch, fz ? fz->qn : nullptr,
        fz ? fz->tau : nullptr, fz ? fz->eps : 0.0f, fz ? fz->rr : nullptr, fz ? fz->proj : nullptr, d_rel);
    attn_combine_kernel<<<H, hd, 0, s>>>(scratch, nsp, pos, dpos, ring, nsp, hd, out, fz ? fz->oq : nullptr,
                                         fz ? fz->od : nullptr);
}
bool fused_ok(int n, int hd, int K, int d_rel) {
    return n % 256 == 0 && n / 256 <= kPerT && hd % 32 == 0 && hd <= 256 && K <= 8 && d_rel <= 32;
}
void norm_q8(const float* x, const float* w, float eps, float* y, int8_t* q, float* d, int n, cudaStream_t s) {
    norm_q8_kernel<<<1, 256, 0, s>>>(x, w, eps, y, q, d, n);
}
void conv_add(int mode, float* x, const float* in, const float* gpu, const float* so, const int* flags, float* state,
              const float* kern, int K, int n, cudaStream_t s) {
    if (mode == 1) conv_add_kernel<1><<<blocks(n, 256), 256, 0, s>>>(x, in, gpu, so, flags, state, kern, K, n);
    else conv_add_kernel<2><<<blocks(n, 256), 256, 0, s>>>(x, in, gpu, so, flags, state, kern, K, n);
}
void kv_conv_store(const float* kk, const float* vv, float* st_k, float* st_v, const float* sc_k, const float* sc_v,
                   int K, const float* k_norm, float eps, __half* Kc, __half* Vc, const int* dpos, int ring, int Hkv,
                   int hd, cudaStream_t s) {
    kv_conv_store_kernel<<<Hkv, 64, 0, s>>>(kk, vv, st_k, st_v, sc_k, sc_v, K, k_norm, eps, Kc, Vc, dpos, ring, hd,
                                            Hkv * hd);
}
void q8(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s) {
    const int wpb = 8;
    q8_kernel<<<dim3(blocks(k / 32, wpb), rows), 32 * wpb, 0, s>>>(x, k, q, d);
}
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s) {
    gemv_f_kernel<<<n, 256, 0, s>>>(w, type, row_bytes, x, y, k);
}
void swiglu(const float* g, const float* u, float* y, int n, const float* scale, cudaStream_t s) {
    swiglu_kernel<<<blocks(n, 256), 256, 0, s>>>(g, u, y, n, scale);
}
void add(float* y, const float* x, int n, cudaStream_t s) { add_kernel<<<blocks(n, 256), 256, 0, s>>>(y, x, n); }
void mul_scalar(float* y, const float* a, int n, cudaStream_t s) {
    mul_scalar_kernel<<<blocks(n, 256), 256, 0, s>>>(y, a, n);
}
void f16_to_f32(const __half* x, float* y, int n, cudaStream_t s) {
    f16_to_f32_kernel<<<blocks(n, 256), 256, 0, s>>>(x, y, n);
}
void row_to_f32(const void* x, int type, float* y, int n, cudaStream_t s) {
    row_to_f32_kernel<<<blocks(n, 256), 256, 0, s>>>(x, type, y, n);
}

}  // namespace ink::k

// ================================================================================================ Inkling 2
// Kernels for Inkling 2 Small's parts (transformers' modeling_inkling2.py, huggingface/transformers#50682): the block
// attention residuals, the L2 clips, the convolutions' output scale, the dynamic k / v convolutions of the global
// layers, the per-token attention sinks and output gate, the routing weights and the softcapped expert activation.
namespace ink::k {
namespace {

constexpr int kResMax = 9, kResQ = 3;

// a block a token: softmax over the (normed) states [the nb blocks, the partial] of q . state / sqrt(E), per query
__global__ void attn_res_kernel(const float* __restrict__ b32, const __half* __restrict__ b16, size_t bstride, int nb,
                                const float* __restrict__ partial, const float* __restrict__ q, int nq,
                                float* __restrict__ out, size_t oqs, int E) {
    __shared__ float wts[kResQ][kResMax];
    const int t = blockIdx.x;
    const float* p = partial + (size_t) t * E;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) ss += p[c] * p[c];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) E + 1e-6f);
    float acc[kResMax * kResQ];
#pragma unroll
    for (int i = 0; i < kResMax * kResQ; ++i) acc[i] = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        float qv[kResQ] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < nq; ++i) qv[i] = q[(size_t) i * E + c];
        for (int j = 0; j < nb; ++j) {
            const size_t o = (size_t) j * bstride + (size_t) t * E + c;
            const float v = b16 ? __half2float(b16[o]) : b32[o];
            for (int i = 0; i < nq; ++i) acc[j * kResQ + i] += v * qv[i];
        }
        const float v = p[c] * inv;
        for (int i = 0; i < nq; ++i) acc[nb * kResQ + i] += v * qv[i];
    }
    const float sc = rsqrtf((float) E);
    // the (nb + 1) x nq sums at once: each warp's by shuffles, then the warps' partials added by one thread a sum
    __shared__ float part[kResMax * kResQ][8];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int j = 0; j <= nb; ++j)
        for (int i = 0; i < nq; ++i) {
            const float d = warp_sum(acc[j * kResQ + i]);
            if (lane == 0) part[j * kResQ + i][wid] = d;
        }
    __syncthreads();
    if (threadIdx.x < (nb + 1) * kResQ) {
        const int j = threadIdx.x / kResQ, i = threadIdx.x % kResQ;
        if (i < nq) {
            float d = 0.0f;
            for (int w = 0; w < nw; ++w) d += part[threadIdx.x][w];
            wts[i][j] = d * sc;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 0; i < nq; ++i) {
            float m = -INFINITY;
            for (int j = 0; j <= nb; ++j) m = fmaxf(m, wts[i][j]);
            float s = 0.0f;
            for (int j = 0; j <= nb; ++j) {
                wts[i][j] = expf(wts[i][j] - m);
                s += wts[i][j];
            }
            for (int j = 0; j <= nb; ++j) wts[i][j] /= s;
        }
    __syncthreads();
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        float o[kResQ] = {0.0f, 0.0f, 0.0f};
        for (int j = 0; j < nb; ++j) {
            const size_t oo = (size_t) j * bstride + (size_t) t * E + c;
            const float v = b16 ? __half2float(b16[oo]) : b32[oo];
            for (int i = 0; i < nq; ++i) o[i] += wts[i][j] * v;
        }
        const float v = p[c] * inv;
        for (int i = 0; i < nq; ++i) out[(size_t) i * oqs + (size_t) t * E + c] = o[i] + wts[i][nb] * v;
    }
}

// One token's attention residual in two wide launches (attn_res_kernel's one block a token is latency-bound at T = 1):
// the dots, a block a state (the partial's also its rsqrt of the mean square) into w [kResMax][kResQ] + inv; then the
// softmax and the mix, a thread a channel.  b32: the token's finished blocks, [nb][E] (normed).
__global__ void attn_res_dots_kernel(const float* __restrict__ b32, int nb, const float* __restrict__ partial,
                                     const float* __restrict__ q, int nq, int E, float* __restrict__ w) {
    const int j = blockIdx.x;
    const float* s = j < nb ? b32 + (size_t) j * E : partial;
    float a[kResQ] = {0.0f, 0.0f, 0.0f}, ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float v = s[c];
        ss += v * v;
        for (int i = 0; i < nq; ++i) a[i] += v * q[(size_t) i * E + c];
    }
    __shared__ float part[kResQ + 1][32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = 0; i < nq; ++i) {
        const float d = warp_sum(a[i]);
        if (lane == 0) part[i][wid] = d;
    }
    const float dss = warp_sum(ss);
    if (lane == 0) part[kResQ][wid] = dss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tss = 0.0f;
        for (int k = 0; k < nw; ++k) tss += part[kResQ][k];
        const float inv = j < nb ? 1.0f : rsqrtf(tss / (float) E + 1e-6f);
        for (int i = 0; i < nq; ++i) {
            float d = 0.0f;
            for (int k = 0; k < nw; ++k) d += part[i][k];
            w[j * kResQ + i] = d * inv * rsqrtf((float) E);
        }
        if (j == nb) w[kResMax * kResQ] = inv;
    }
}
__global__ void attn_res_mix_kernel(const float* __restrict__ b32, int nb, const float* __restrict__ partial, int nq,
                                    int E, const float* __restrict__ w, float* __restrict__ out, size_t oqs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= E) return;
    const float inv = w[kResMax * kResQ];
    for (int i = 0; i < nq; ++i) {
        float m = -INFINITY;
        for (int j = 0; j <= nb; ++j) m = fmaxf(m, w[j * kResQ + i]);
        float s = 0.0f, o = 0.0f;
        for (int j = 0; j <= nb; ++j) {
            const float p = __expf(w[j * kResQ + i] - m);
            s += p;
            o += p * (j < nb ? b32[(size_t) j * E + c] : partial[c] * inv);
        }
        out[(size_t) i * oqs + c] = o / s;
    }
}

// a block a row: y = x * rsqrt(mean(x^2) + eps), F32 or F16
__global__ void unorm_kernel(const float* __restrict__ x, float* __restrict__ y32, __half* __restrict__ y16, int E,
                             float eps) {
    const int t = blockIdx.x;
    const float* p = x + (size_t) t * E;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < E; c += blockDim.x) ss += p[c] * p[c];
    ss = block_sum(ss);
    const float inv = rsqrtf(ss / (float) E + eps);
    for (int c = threadIdx.x; c < E; c += blockDim.x) {
        const float v = p[c] * inv;
        if (y16) y16[(size_t) t * E + c] = __float2half(v);
        else y32[(size_t) t * E + c] = v;
    }
}

// a warp a (row, head): x = x * scale; x *= min(1, thr / (|x| + 1e-6)); x *= tau[row]
__global__ void head_clip_kernel(float* __restrict__ x, const float* __restrict__ scale, float thr,
                                 const float* __restrict__ tau, int heads, int hd, int rows) {
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= rows * heads) return;
    const int row = w / heads, h = w % heads;
    float* p = x + (size_t) w * hd;
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) {
        float v = p[i];
        if (scale) v *= scale[(size_t) h * hd + i];
        p[i] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
    if (tau) f *= tau[row];
    for (int i = lane; i < hd; i += 32) p[i] *= f;
}

// a warp a (row, head): x * rsqrt(mean(x^2) + eps)
__global__ void head_unorm_kernel(float* __restrict__ x, int n, int hd, float eps) {
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (w >= n) return;
    float* p = x + (size_t) w * hd;
    float ss = 0.0f;
    for (int i = lane; i < hd; i += 32) ss += p[i] * p[i];
    ss = warp_sum(ss);
    const float s = rsqrtf(ss / (float) hd + eps);
    for (int i = lane; i < hd; i += 32) p[i] *= s;
}

// the token path's short convolution in place: x = (x + conv(x)) * scale, the state the last K - 1 inputs
__global__ void sconv2_kernel(float* __restrict__ x, float* __restrict__ state, const float* __restrict__ kern, int T,
                              int ch, int K, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float w[8], win[8];
    for (int i = 0; i < K; ++i) w[i] = kern[(size_t) c * K + i];
    for (int i = 0; i < K - 1; ++i) win[i] = state[(size_t) c * (K - 1) + i];
    for (int t = 0; t < T; ++t) {
        const float xin = x[(size_t) t * ch + c];
        win[K - 1] = xin;
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) sum += win[i] * w[i];
        x[(size_t) t * ch + c] = (xin + sum) * scale;
        for (int i = 0; i < K - 1; ++i) win[i] = win[i + 1];
    }
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = win[i];
}

// a chunk: y = (x + conv(x)) * scale (the inputs before the chunk from state, unchanged here)
__global__ void sconv2_chunk_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ state,
                                    const float* __restrict__ kern, int T, int ch, int K, float scale) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ch) return;
    const int t = (int) (i / ch), c = (int) (i % ch);
    float s = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int tt = t - (K - 1) + k;
        const float v = tt >= 0 ? x[(size_t) tt * ch + c] : state[(size_t) c * (K - 1) + (tt + K - 1)];
        s += v * kern[(size_t) c * K + k];
    }
    y[i] = (x[i] + s) * scale;
}

// the global layers' dynamic convolution: taps stat[lag][c] + dyn[t][head(c) K + lag] (lag 0: this token),
// y = (x + sum_lag tap x(t - lag)) / sqrt(2K + 1); the inputs before the chunk from state (the oldest first)
__global__ void dsconv_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ state,
                              const float* __restrict__ stat, const float* __restrict__ dyn, int T, int ch, int nkv,
                              int K) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * ch) return;
    const int t = (int) (i / ch), c = (int) (i % ch), h = c / (ch / nkv);
    float s = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const int tt = t - lag;
        const float v = tt >= 0 ? x[(size_t) tt * ch + c] : state[(size_t) c * (K - 1) + (tt + K - 1)];
        const float w = stat[(size_t) lag * ch + c] + dyn[(size_t) t * nkv * K + (size_t) h * K + lag];
        s += w * v;
    }
    y[i] = (x[i] + s) * rsqrtf((float) (2 * K + 1));
}

// the decode attention splits merged with a per-head sink logit (one more softmax term with no value) and the
// output gate sigmoid(gate) (either may be null); grid H, blockDim hd
__global__ void attn_combine2_kernel(const float* __restrict__ part, int nsp, int pos_arg, const int* __restrict__ dpos,
                                     int ring, int nsp_max, int hd, const float* __restrict__ sinks,
                                     const float* __restrict__ gate, float* __restrict__ out, int8_t* __restrict__ oq,
                                     float* __restrict__ od) {
    const int h = blockIdx.x, d = threadIdx.x;
    const int pos = dpos ? *dpos : pos_arg, n_kv = attn_n_kv(pos, ring), S = attn_split_keys(n_kv, ring, nsp_max);
    const int used = min(nsp, (n_kv + S - 1) / S);
    const float* pp = part + (size_t) h * nsp * (hd + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int s = 0; s < used; ++s) M = fmaxf(M, pp[(size_t) s * (hd + 2)]);
    float L = sinks ? __expf(sinks[h] - M) : 0.0f, o = 0.0f;
    for (int s = 0; s < used; ++s) {
        const float* p = pp + (size_t) s * (hd + 2);
        const float m = p[0];
        const float c = m == -INFINITY ? 0.0f : __expf(m - M);
        L += p[1] * c;
        o += p[2 + d] * c;
    }
    float v = L > 0.0f ? o / L : 0.0f;
    if (gate) v *= 1.0f / (1.0f + __expf(-gate[(size_t) h * hd + d]));
    out[(size_t) h * hd + d] = v;
    if (oq) {
        const float am = warp_max(fabsf(v));
        const float s8 = am / 127.0f;
        const size_t c = (size_t) h * hd + d;
        oq[c] = (int8_t) (am > 0.0f ? __float2int_rn(v / s8) : 0);
        if ((d & 31) == 0) od[c / 32] = s8;
    }
}

// the chunk attention softmax as attn_softmax_kernel, plus each row sink logit sinks[token][h] in the sum
__global__ void attn_softmax2_kernel(float* __restrict__ S, int lds, __half* __restrict__ P, int ldp, int nk, int G,
                                     int g, int ta, int pos0, int kp0, int window, const float* __restrict__ R,
                                     const float* __restrict__ proj, int E, int H, int d_rel,
                                     const float* __restrict__ sinks) {
    const int r = blockIdx.x, t = r / G, hh = r % G, h = g * G + hh, tok = ta + t, p = pos0 + tok;
    float rv[32];
    for (int d = 0; d < d_rel; ++d) rv[d] = R[((size_t) tok * H + h) * d_rel + d];
    float* Sr = S + (size_t) r * lds;
    __half* Pr = P + (size_t) r * ldp;
    const float sk = sinks[(size_t) tok * H + h];
    float mx = sk;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const int dist = p - (kp0 + c);
        float v = -INFINITY;
        if (dist >= 0 && (window <= 0 || dist < window)) {
            v = Sr[c];
            if (dist < E) {
                float b = 0.0f;
                for (int d = 0; d < d_rel; ++d) b += rv[d] * __ldg(proj + (size_t) d * E + dist);
                v += b;
            }
        }
        Sr[c] = v;
        mx = fmaxf(mx, v);
    }
    mx = block_max(mx);
    float sum = 0.0f;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        const float v = Sr[c];
        if (v != -INFINITY) sum += expf(v - mx);
    }
    sum = block_sum(sum) + expf(sk - mx);
    const float inv = 1.0f / sum;
    for (int c = threadIdx.x; c < ldp; c += blockDim.x) {
        float pv = 0.0f;
        if (c < nk) {
            const float v = Sr[c];
            if (v != -INFINITY) pv = expf(v - mx) * inv;
        }
        Pr[c] = __float2half(pv);
    }
}

// a warp a token: the top nu of sigmoid(RL) + bias (ties: the lower id), weights sigmoid / their sum x scale
__global__ void route2_kernel(const float* __restrict__ RL, int ld, const float* __restrict__ bias, int T, int ne,
                              int nu, float scale, int* __restrict__ sel, float* __restrict__ w) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float* rl = RL + (size_t) t * ld;
    float sc[16];
    const int per = (ne + 31) / 32;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        sc[i] = e < ne ? 1.0f / (1.0f + expf(-rl[e])) + bias[e] : -INFINITY;
    }
    int chosen[16];
    for (int k = 0; k < nu; ++k) {
        float bv = -INFINITY;
        int be = 0x7fffffff;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (sc[i] > bv || (sc[i] == bv && e < be)) {
                bv = sc[i];
                be = e;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                be = oe;
            }
        }
        chosen[k] = be;
        if ((be & 31) == lane) sc[be >> 5] = -INFINITY;
    }
    if (lane == 0) {
        float wv[16], sum = 0.0f;
        for (int i = 0; i < nu; ++i) {
            wv[i] = 1.0f / (1.0f + expf(-rl[chosen[i]]));
            sum += wv[i];
        }
        for (int i = 0; i < nu; ++i) {
            sel[(size_t) t * nu + i] = chosen[i];
            w[(size_t) t * nu + i] = wv[i] / (sum + 1e-20f) * scale;
        }
    }
}

__device__ __forceinline__ float act_cap(float g, float u, float gcap, float ucap) {
    return tanhf(g / gcap) * gcap * (1.0f / (1.0f + expf(-g))) * (tanhf(u / ucap) * ucap);
}

// rows of gu [gate ff | up ff] -> softcap(g) sigmoid(g) softcap(u), F16 or F32 rows of ff
__global__ void act_cap_rows_kernel(const float* __restrict__ gu, int n, int ff, float gcap, float ucap,
                                    __half* __restrict__ h16, float* __restrict__ h32) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * ff) return;
    const int r = (int) (i / ff), c = (int) (i % ff);
    const float v = act_cap(gu[(size_t) r * 2 * ff + c], gu[(size_t) r * 2 * ff + ff + c], gcap, ucap);
    if (h16) h16[i] = __float2half(v);
    else h32[i] = v;
}

// a MoE layer end: fo = the CPU lane sum (flags[0]) + the VRAM experts sum (flags[1]), x = res + (fo + conv(fo)) * scale
__global__ void moe_tail2_kernel(float* __restrict__ x, const float* __restrict__ res, const float* __restrict__ cpu,
                                 const float* __restrict__ gpu, const int* __restrict__ flags, float* __restrict__ state,
                                 const float* __restrict__ kern, int n, int K, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float v = 0.0f;
    if (flags[0]) v += cpu[c];
    if (flags[1]) v += gpu[c];
    float w[8], win[8];
    for (int i = 0; i < K; ++i) w[i] = kern[(size_t) c * K + i];
    for (int i = 0; i < K - 1; ++i) win[i] = state[(size_t) c * (K - 1) + i];
    win[K - 1] = v;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) sum += win[i] * w[i];
    for (int i = 0; i < K - 1; ++i) state[(size_t) c * (K - 1) + i] = win[i + 1];
    x[c] = res[c] + (v + sum) * scale;
}

__global__ void gate_mul_kernel(float* __restrict__ o, const float* __restrict__ gate, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
__global__ void softcap_kernel(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}
__global__ void add2_kernel(float* __restrict__ y, const float* __restrict__ a, const float* __restrict__ b, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

}  // namespace

void attn_res(const float* b32, const __half* b16, size_t bstride, int nb, const float* partial, const float* q, int nq,
              float* out, size_t oqs, int T, int E, cudaStream_t s) {
    if (T > 0) attn_res_kernel<<<T, 256, 0, s>>>(b32, b16, bstride, nb, partial, q, nq, out, oqs, E);
}
void attn_res_tok(const float* b32, int nb, const float* partial, const float* q, int nq, float* out, size_t oqs, int E,
                  float* scratch, cudaStream_t s) {
    attn_res_dots_kernel<<<nb + 1, 256, 0, s>>>(b32, nb, partial, q, nq, E, scratch);
    attn_res_mix_kernel<<<blocks(E, 256), 256, 0, s>>>(b32, nb, partial, nq, E, scratch, out, oqs);
}
void unorm(const float* x, float* y32, __half* y16, int T, int E, float eps, cudaStream_t s) {
    if (T > 0) unorm_kernel<<<T, 256, 0, s>>>(x, y32, y16, E, eps);
}
void head_clip(float* x, const float* scale, float thr, const float* tau, int rows, int heads, int hd, cudaStream_t s) {
    const int wpb = 8, n = rows * heads;
    if (n > 0) head_clip_kernel<<<blocks(n, wpb), 32 * wpb, 0, s>>>(x, scale, thr, tau, heads, hd, rows);
}
void head_unorm(float* x, int rows, int heads, int hd, float eps, cudaStream_t s) {
    const int wpb = 8, n = rows * heads;
    if (n > 0) head_unorm_kernel<<<blocks(n, wpb), 32 * wpb, 0, s>>>(x, n, hd, eps);
}
void sconv2(float* x, float* state, const float* kern, int T, int ch, int K, float scale, cudaStream_t s) {
    sconv2_kernel<<<blocks(ch, 256), 256, 0, s>>>(x, state, kern, T, ch, K, scale);
}
void sconv2_chunk(const float* x, float* y, const float* state, const float* kern, int T, int ch, int K, float scale,
                  cudaStream_t s) {
    const size_t n = (size_t) T * ch;
    if (n) sconv2_chunk_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, y, state, kern, T, ch, K, scale);
}
void dsconv(const float* x, float* y, const float* state, const float* stat, const float* dyn, int T, int ch, int nkv,
            int K, cudaStream_t s) {
    const size_t n = (size_t) T * ch;
    if (n) dsconv_kernel<<<blocks_l(n, 256), 256, 0, s>>>(x, y, state, stat, dyn, T, ch, nkv, K);
}
void attn_decode2(const float* q, const __half* K, const __half* V, int rows, const int* dpos, int ring,
                  const float* rel, int E, int H, int Hkv, int hd, float scale, const float* sinks, const float* gate,
                  float* out, int8_t* oq, float* od, float* scratch, cudaStream_t s) {
    const int nsp = attn_nsp(ring, rows);
    const int S_max = attn_split_keys(ring > 0 ? ring : rows, ring, nsp);
    const size_t smem = (size_t) (H / Hkv) * (hd + S_max) * sizeof(float);
    attn_decode_kernel<<<dim3(nsp, Hkv), kAttnThreads, smem, s>>>(q, K, V, 0, dpos, ring, nsp, rel, E, H, Hkv, hd,
                                                                   scale, scratch, nullptr, nullptr, 0.0f, nullptr,
                                                                   nullptr, 0);
    attn_combine2_kernel<<<H, hd, 0, s>>>(scratch, nsp, 0, dpos, ring, nsp, hd, sinks, gate, out, oq, od);
}
void attn_softmax2(float* S, int lds, __half* P, int ldp, int M, int nk, int G, int g, int ta, int pos0, int kp0,
                   int window, const float* R, const float* proj, int E, int H, int d_rel, const float* sinks,
                   cudaStream_t s) {
    if (M > 0)
        attn_softmax2_kernel<<<M, 256, 0, s>>>(S, lds, P, ldp, nk, G, g, ta, pos0, kp0, window, R, proj, E, H, d_rel,
                                               sinks);
}
void route2(const float* RL, int ld, const float* bias, int T, int ne, int nu, float scale, int* sel, float* w,
            cudaStream_t s) {
    const int wpb = 4;
    if (T > 0) route2_kernel<<<blocks(T, wpb), 32 * wpb, 0, s>>>(RL, ld, bias, T, ne, nu, scale, sel, w);
}
void act_cap_rows(const float* gu, int n, int ff, float gcap, float ucap, __half* h16, float* h32, cudaStream_t s) {
    const size_t m = (size_t) n * ff;
    if (m) act_cap_rows_kernel<<<blocks_l(m, 256), 256, 0, s>>>(gu, n, ff, gcap, ucap, h16, h32);
}
void moe_tail2(float* x, const float* res, const float* cpu, const float* gpu, const int* flags, float* state,
               const float* kern, int n, int K, float scale, cudaStream_t s) {
    moe_tail2_kernel<<<blocks(n, 256), 256, 0, s>>>(x, res, cpu, gpu, flags, state, kern, n, K, scale);
}
void gate_mul(float* o, const float* gate, size_t n, cudaStream_t s) {
    if (n) gate_mul_kernel<<<blocks_l(n, 256), 256, 0, s>>>(o, gate, n);
}
void softcap(float* x, int n, float cap, cudaStream_t s) { softcap_kernel<<<blocks(n, 256), 256, 0, s>>>(x, n, cap); }
void add2(float* y, const float* a, const float* b, size_t n, cudaStream_t s) {
    if (n) add2_kernel<<<blocks_l(n, 256), 256, 0, s>>>(y, a, b, n);
}

}  // namespace ink::k
