// src/ink/ink_kernels.cuh - maya-ink's own device kernels: Inkling-Small's parts that the shared GGUF kernels
// (src/ds4/ds4_iq.cuh) do not cover - the RMS norms, the four short convolutions a layer, the relative-position bias
// band, the attention, the activations' int8 rounding for those kernels and the small F32 / F16 / BF16 matrices.
//
// The arithmetic follows llama.cpp's Inkling graph (ggml-org/llama.cpp#25731, src/models/inkling.cpp, MIT):
//   * sconv(x) = x + causal depthwise conv1d(x), kernel K = 4 (shortconv_kernel), state = the last K - 1 inputs;
//   * attention: no RoPE; s = (q . k) / head_dim + rel[h][pos_q - pos_k] (0 <= distance < E: rel_extent, 1024 on the
//     global layers and 512 on the sliding ones, else no bias), softmax over the visible keys (causal; the sliding
//     layers see pos_q - pos_k < n_swa = 512);
//   * the global layers' q and rel times tau = 1 + alpha log(max((pos + 1) / n_floor, 1)).
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ink::k {

/// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, rows of n (y may be x).
void rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s);
/// In place, each head of hd values of rows x heads: x * rsqrt(mean(x^2) + eps) * w (w: hd values), then times
/// tau[row] when tau is set.
void head_norm(float* x, const float* w, int rows, int heads, int hd, float eps, const float* tau, cudaStream_t s);
/// x[t][c] += sum_i kern[c][i] * in[t - (K - 1) + i] for t < T (in[-1 ..] from state, the oldest first), channels ch;
/// state [ch][K - 1] then holds the last K - 1 inputs (before the add).  T at most 64.
void shortconv(float* x, float* state, const float* kern, int T, int ch, int K, cudaStream_t s);
/// rel[t][h][e] = (sum_d proj[d][e] r[t][h][d]) * (tau ? tau[t] : 1) for e < E (proj: d_rel rows of E values).
void rel_logits(const float* r, const float* proj, const float* tau, float* rel, int T, int H, int d_rel, int E,
                cudaStream_t s);
/// One token's k / v (F32, kvw = n_head_kv x head_dim) into the caches' row `slot` (F16).
void kv_store(const float* k, const float* v, __half* K, __half* V, int slot, int kvw, cudaStream_t s);

/// One query position's attention (decode): q [H][hd] (normed, tau'd), keys / values in F16 rows of [Hkv][hd],
/// n_kv of them; key j's distance to the query is dist(j) = (pos - slot_pos(j)), where slot j holds position
///   ring == 0: j (a full cache, positions 0 .. pos);  ring > 0: pos - ((pos - j) mod ring) (a ring of ring rows).
/// rel [H][E] (the query's band; distance >= E: no bias).  out [H][hd].  scratch: attn_scratch_floats(H, hd, n_kv).
void attn_decode(const float* q, const __half* K, const __half* V, int n_kv, int pos, int ring, const float* rel, int E,
                 int H, int Hkv, int hd, float scale, float* out, float* scratch, cudaStream_t s);
size_t attn_scratch_floats(int H, int hd, int n_kv);

/// int8 rows with an F32 scale per 32 values (ds4_iq.cuh's MvJob / Args inputs): d = max|x| / 127, q = rint(x / d).
void q8(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s);

/// y[r] = sum_c W[r][c] x[c] for W of n rows of k values in F32 (type 0), F16 (1) or BF16 (30), rows_bytes apart.
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s);

/// y = silu(g) * u (n values; y may be g), then times scale[0] when scale is set (a device scalar).
void swiglu(const float* g, const float* u, float* y, int n, const float* scale, cudaStream_t s);
/// y += x (n values); y = x * a (a device scalar) when mul is set.
void add(float* y, const float* x, int n, cudaStream_t s);
void mul_scalar(float* y, const float* a, int n, cudaStream_t s);
/// F16 row to F32 (n values).
void f16_to_f32(const __half* x, float* y, int n, cudaStream_t s);
/// A row of n values in F32 (0), F16 (1) or BF16 (30) to F32.
void row_to_f32(const void* x, int type, float* y, int n, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ the prompt's chunks
/// x[t] = the embedding row of ids[t] as F32 (w: rows of n values in Q8_0 (8), F32 (0), F16 (1) or BF16 (30), rb bytes a
/// row), t < T.  false: another type.
bool embed_rows(const uint8_t* w, int type, size_t rb, const int* ids, int T, int n, float* x, cudaStream_t s);
/// The short convolution over a chunk: y[t][c] (+)= x[t][c] + sum_i kern[c][i] in(t - (K - 1) + i), in(t') = x[t'] for
/// t' >= 0, else state[c][t' + K - 1] (the inputs before the chunk, the oldest first).  add: y += (the residual), else
/// y =.  The state is not touched (sconv_state after it).
void sconv_chunk(const float* x, float* y, const float* state, const float* kern, int T, int ch, int K, bool add,
                 cudaStream_t s);
/// state[c] = the last K - 1 inputs once the chunk x (T rows) is in (the old state's where T < K - 1).
void sconv_state(const float* x, float* state, int T, int ch, int K, cudaStream_t s);
/// F32 to F16, n values.
void to_f16(const float* x, __half* y, size_t n, cudaStream_t s);
/// q [T][H][hd] (normed) to F16 rows grouped by KV head: qh [Hkv][T][G][hd] (G = H / Hkv).
void q_group(const float* q, __half* qh, int T, int H, int Hkv, int hd, cudaStream_t s);
/// A sliding layer's ring of `ring` rows (position p at row p % ring) and its window: win[j] = ring[(p0 + j) % ring] for
/// j < n (to_window), or ring[(p0 + j) % ring] = win[j] for j0 <= j < j1 (to_ring); rows of kvw F16 values.
void ring_to_window(const __half* ring, __half* win, int p0, int n, int ring_rows, int kvw, cudaStream_t s);
void window_to_ring(const __half* win, __half* ring, int p0, int j0, int j1, int ring_rows, int kvw, cudaStream_t s);
/// The chunk attention's softmax, rows r = t G + hh of a query tile (the tile's token t: the chunk's ta + t, at position
/// pos0 + ta + t; head h = g G + hh), columns c < nk the keys at positions kp0 + c: S (scores, already times 1 / hd)
/// plus the band bias - (sum_d R[token][h][d] proj[d][dist]) times tau[token] (tau null: 1) for 0 <= dist < E - over
/// the visible keys (dist >= 0, and dist < window when window > 0), the others 0; P (F16, ldp >= nk, the columns from
/// nk to ldp zero).  S is overwritten.
void attn_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int G, int g, int ta, int pos0, int kp0,
                  int window, const float* R, const float* proj, int E, int H, int d_rel, const float* tau,
                  cudaStream_t s);
/// A tile's output O [M][hd] (rows r = t G + hh) into out [T][H][hd] at token ta + t, head g G + hh.
void attn_scatter(const float* O, float* out, int M, int G, int g, int ta, int H, int hd, cudaStream_t s);
/// MoE routing, a token a warp: the top nu of sigmoid(RL[t][e]) + bias[e] over ne experts (ties: the lower id), in
/// score order -> sel [T][nu]; their weights softmax(logsigmoid([their raw logits, the nsh shared logits])) x scale ->
/// w [T][nu + nsh] (the shared experts' gammas last).  ne at most 512, nu + nsh at most 32.
void route(const float* RL, int ld, const float* bias, int T, int ne, int nsh, int nu, float scale, int* sel, float* w,
           cudaStream_t s);
/// counts[e] += the tokens that chose e (counts zeroed by the caller).
void expert_count(const int* sel, int n, int* counts, cudaStream_t s);
/// Each expert's tokens in token order: list[off[e] + i] = t, wl[...] = its weight (a block an expert; deterministic).
void expert_lists(const int* sel, const float* w, int T, int nu, int ldw, int ne, const int* off, int* list, float* wl,
                  cudaStream_t s);
/// h[r][c] = F16(silu(gu[r][c]) gu[r][ff + c] (gamma ? gamma[r * gstride] : 1)), rows of gu 2 ff F32, of h ff.
void swiglu_rows(const float* gu, int n, int ff, const float* gamma, int gstride, __half* h, cudaStream_t s);
/// y[list[i]] += wl[i] * ye[i] (rows of n values; each list entry a different row of y).
void scatter_add(const float* ye, const int* list, const float* wl, int rows, int n, float* y, cudaStream_t s);
/// y += x * (a ? a[0] : 1), n values.
void axpy(float* y, const float* x, const float* a, size_t n, cudaStream_t s);
/// y = x * a[0], n values.
void scale_dev(float* y, const float* x, const float* a, size_t n, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ a token's VRAM experts
/// h[r][c] = silu(gu[r][c]) gu[r][ff + c] (F32), rows of gu 2 ff, of h ff.
void swiglu_f32_rows(const float* gu, int n, int ff, float* h, cudaStream_t s);
/// out[c] = sum over i < n of y[i][c] (rows of m; in order of i), c < m.
void sum_rows(const float* y, int n, int m, float* out, cudaStream_t s);

}  // namespace ink::k
