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

constexpr int kSplit = 256;   // keys a block of the decode attention

// grid (splits, Hkv), a warp per query head of the group (blockDim = 32 G); each warp's (m, l, acc) for its split
__global__ void attn_decode_kernel(const float* __restrict__ q, const __half* __restrict__ K,
                                   const __half* __restrict__ V, int n_kv, int pos, int ring,
                                   const float* __restrict__ rel, int E, int H, int Hkv, int hd, float scale,
                                   float* __restrict__ part) {
    const int split = blockIdx.x, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = H / Hkv, h = g * G + w;
    const int per = hd / 32;
    float qv[8], acc[8];
    for (int i = 0; i < per; ++i) {
        qv[i] = q[(size_t) h * hd + lane + 32 * i];
        acc[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const int j0 = split * kSplit, j1 = min(n_kv, j0 + kSplit);
    const size_t row = (size_t) Hkv * hd;
    for (int j = j0; j < j1; ++j) {
        const __half* kr = K + (size_t) j * row + (size_t) g * hd;
        float s = 0.0f;
        for (int i = 0; i < per; ++i) s += qv[i] * __half2float(kr[lane + 32 * i]);
        s = warp_sum(s) * scale;
        int dist = ring > 0 ? ((pos - j) % ring + ring) % ring : pos - j;
        if (dist >= 0 && dist < E) s += rel[(size_t) h * E + dist];
        const float mn = fmaxf(m, s);
        const float c1 = __expf(m - mn), c2 = __expf(s - mn);
        l = l * c1 + c2;
        const __half* vr = V + (size_t) j * row + (size_t) g * hd;
        for (int i = 0; i < per; ++i) acc[i] = acc[i] * c1 + c2 * __half2float(vr[lane + 32 * i]);
        m = mn;
    }
    const int nsp = gridDim.x;
    float* pp = part + ((size_t) h * nsp + split) * (hd + 2);
    if (lane == 0) {
        pp[0] = m;
        pp[1] = l;
    }
    for (int i = 0; i < per; ++i) pp[2 + lane + 32 * i] = acc[i];
}

// grid H, blockDim hd: the splits' partial sums merged
__global__ void attn_combine_kernel(const float* __restrict__ part, int nsp, int hd, float* __restrict__ out) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float* pp = part + (size_t) h * nsp * (hd + 2);
    float M = -INFINITY;
    for (int s = 0; s < nsp; ++s) M = fmaxf(M, pp[(size_t) s * (hd + 2)]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < nsp; ++s) {
        const float* p = pp + (size_t) s * (hd + 2);
        if (p[0] == -INFINITY) continue;
        const float c = __expf(p[0] - M);
        L += p[1] * c;
        o += p[2 + d] * c;
    }
    out[(size_t) h * hd + d] = L > 0.0f ? o / L : 0.0f;
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
size_t attn_scratch_floats(int H, int hd, int n_kv) {
    const int nsp = (n_kv + kSplit - 1) / kSplit;
    return (size_t) H * nsp * (hd + 2);
}
void attn_decode(const float* q, const __half* K, const __half* V, int n_kv, int pos, int ring, const float* rel, int E,
                 int H, int Hkv, int hd, float scale, float* out, float* scratch, cudaStream_t s) {
    const int nsp = (n_kv + kSplit - 1) / kSplit, G = H / Hkv;
    attn_decode_kernel<<<dim3(nsp, Hkv), 32 * G, 0, s>>>(q, K, V, n_kv, pos, ring, rel, E, H, Hkv, hd, scale, scratch);
    attn_combine_kernel<<<H, hd, 0, s>>>(scratch, nsp, hd, out);
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
