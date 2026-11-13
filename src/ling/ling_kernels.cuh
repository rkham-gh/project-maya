// src/ling/ling_kernels.cuh - maya-ling's own device kernels: Ling-3.1-flash's parts that the shared GGUF kernels
// (src/ds4/ds4_iq.cuh) do not cover - the RMS norms, the activations' int8 rounding for those kernels, the F32
// matrices, the Kimi Delta Attention (KDA) layers' short convolutions, gates and recurrence, the gated MLA layers'
// RoPE, cache rows and attention, the routing and the experts' clamped SwiGLU.
//
// The arithmetic follows llama.cpp's bailingmoe3 graph (src/models/bailingmoe3.cpp, delta-net-base.cpp,
// llama-graph.cpp's build_moe_ffn / build_ffn / build_attn_mha; ggml-cuda's gated_delta_net.cu and rope.cu; MIT):
//   * KDA (63 of the 72 layers): q, k, v = silu(causal depthwise conv1d(W x), kernel 4) (no residual), q and k
//     L2-normed per head (x / sqrt(sum x^2 + eps)), beta = sigmoid(W_beta x) a head, a gate a channel
//     g = lb sigmoid(a_h (W_f x + dt_bias)) (lb = -5, a_h = exp(A_log)), then a head's 128 x 128 state S (key i,
//     value j): S[i][:] *= exp(g_i); d_j = (v_j - sum_i S[i][j] k_i) beta; S[i][j] += k_i d_j;
//     o_j = sum_i S[i][j] q_i / sqrt(128); out = rmsnorm(o) w sigmoid(W_g x) a head, then W_o;
//   * MLA (every 8th layer): MQA over the 576-value cache rows [rmsnorm(kv_a[0:512]), rope(kv_a[512:576])], the queries
//     [k_b q_nope (512), rope(q_pe) (64)] a head, scale 1/sqrt(192), each head's 512-value output through v_b (128),
//     times sigmoid(W_gate x) a head, then W_o; RoPE ggml's NORM (consecutive pairs) on the 64 rope values, theta
//     pos base^(-2i/64);
//   * the MoE: sigmoid(logits), + exp_probs_b for the choice, 8 groups of 64 (each scored by its top-2 sum), the top 4
//     groups kept, the top 8 experts there; weights the chosen sigmoids / their sum x 2.5; SwiGLU clamped where the
//     GGUF says (swiglu_clamp_exp / _shexp): min(silu(g), L) clamp(u, -L, L).
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ling::k {

// ------------------------------------------------------------------------------------------------ generic
/// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, rows of n (y may be x).
void rmsnorm(const float* x, const float* w, float* y, int rows, int n, float eps, cudaStream_t s);
/// Whether norm_q8 takes a row of n (n % 256 == 0, n / 256 <= 16).
bool norm_q8_ok(int n);
/// y = rmsnorm(x) * w (y may be null) and its int8 rows (q, d: q8's) - one block, rmsnorm's then q8's arithmetic.
void norm_q8(const float* x, const float* w, float eps, float* y, int8_t* q, float* d, int n, cudaStream_t s);
/// int8 rows with an F32 scale per 32 values (ds4_iq.cuh's MvJob inputs): d = max|x| / 127, q = rint(x / d).
void q8(const float* x, int rows, int k, int8_t* q, float* d, cudaStream_t s);
/// y[r] = sum_c W[r][c] x[c] for W of n rows of k values in F32 (type 0), F16 (1) or BF16 (30), row_bytes apart.
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s);
/// y += x (n values).
void add(float* y, const float* x, int n, cudaStream_t s);
/// x[t] = the embedding row of ids[t] as F32 (w: rows of n values in Q8_0 (8), F32 (0), F16 (1) or BF16 (30), rb
/// bytes a row), t < T.  false: another type.
bool embed_rows(const uint8_t* w, int type, size_t rb, const int* ids, int T, int n, float* x, cudaStream_t s);
/// F32 to F16, n values.
void to_f16(const float* x, __half* y, size_t n, cudaStream_t s);
/// y += x (times a[0] when a is set: a device scalar), n values.
void axpy(float* y, const float* x, const float* a, size_t n, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ the FFNs
/// h = silu(g) u, or with limit > 0: min(silu(g), limit) * clamp(u, -limit, limit) (n values; h may be g).
void swiglu(const float* g, const float* u, float* h, int n, float limit, cudaStream_t s);
/// Rows of gu [n][2 ff] (gate | up) -> h [n][ff] (F32), swiglu's arithmetic.
void swiglu_f32_rows(const float* gu, int n, int ff, float limit, float* h, cudaStream_t s);
/// The same into F16 rows (the prompt path's GEMM input).
void swiglu_rows(const float* gu, int n, int ff, float limit, __half* h, cudaStream_t s);
/// out[c] = sum over i < n of y[i][c] (rows of m; in order of i), c < m.
void sum_rows(const float* y, int n, int m, float* out, cudaStream_t s);
/// A MoE layer's end: x += ((flags[0] ? cpu : 0) + (flags[1] ? gpu : 0)) + so, n channels (cpu first, as summed).
void moe_tail(float* x, const float* cpu, const float* gpu, const float* so, const int* flags, int n, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ KDA
/// A KDA layer's per-channel constants (all device pointers): the q / k / v convolutions' kernels [C][K] (C = H hd,
/// the oldest tap first), dt_bias [C], a [H] (exp(A_log)), the gate's lower bound lb (-5) and the L2 norm's eps.
struct Kda {
    const float *wq = nullptr, *wk = nullptr, *wv = nullptr, *dt_bias = nullptr, *a = nullptr;
    int H = 0, hd = 0, K = 0;
    float lb = -5.0f, eps = 1e-6f;
};
/// One token's prep: qr / kr / vr (W x, C values) through their convolutions (the states st_* [C][K - 1], the oldest
/// first, slid in place) and SiLU, q and k L2-normed per head -> qn, kn, vc; ge = exp(lb sigmoid(a_h (fr + dt_bias)))
/// [C]; beta[h] = sigmoid(br[h]).
void kda_prep1(const Kda& w, const float* qr, const float* kr, const float* vr, const float* fr, const float* br,
               float* st_q, float* st_k, float* st_v, float* qn, float* kn, float* vc, float* ge, float* beta,
               cudaStream_t s);
/// A chunk's convolutions (T rows of C, the states before it): out = silu(conv), the states untouched (kda_conv_state).
void kda_conv(const Kda& w, const float* qr, const float* kr, const float* vr, const float* st_q, const float* st_k,
              const float* st_v, float* qc, float* kc, float* vc, int T, cudaStream_t s);
/// The states after the chunk: the last K - 1 inputs (the old state's where T < K - 1).
void kda_conv_state(const Kda& w, const float* qr, const float* kr, const float* vr, float* st_q, float* st_k,
                    float* st_v, int T, cudaStream_t s);
/// A chunk's q / k L2-normed per head in place, fr -> ge in place, beta [T][H] = sigmoid(br [T][H]).
void kda_norm_gate(const Kda& w, float* qc, float* kc, float* fr, const float* br, float* beta, int T, cudaStream_t s);
/// The recurrence over T tokens (rows of C = H 128: q / k L2-normed, ge = exp(g), v; beta [T][H]) from the state S
/// [H][128 (value j)][128 (key i)] (updated in place): o [T][C] = S^T q / sqrt(128) after each token's update.
void kda_scan(const float* q, const float* k, const float* v, const float* ge, const float* beta, float* S, float* o,
              int T, int H, cudaStream_t s);
/// y = rmsnorm(o) w sigmoid(gr) a head of 128 (rows of C): into y (F32), yh (F16) and / or the int8 rows (q, d) of
/// the row of C (any may be null).
void kda_out(const float* o, const float* gr, const float* w, float eps, int T, int H, float* y, __half* yh,
             int8_t* q, float* d, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ MLA
/// RoPE (ggml's NORM: pairs (2i, 2i + 1), theta = pos theta_scale^i) on q [T][H][192]'s last 64 values in place, pos =
/// (dpos ? *dpos : pos0) + t.
void mla_rope_q(float* q, int T, int H, int pos0, const int* dpos, float theta_scale, cudaStream_t s);
/// The cache rows of T tokens: kva [T][576] -> cache[pos][0:512] = F16(rmsnorm(kva[0:512]) w), [512:576] =
/// F16(rope(kva[512:576])), pos = (dpos ? *dpos : pos0) + t.
void mla_kv_store(const float* kva, const float* w, float eps, __half* cache, int T, int pos0, const int* dpos,
                  float theta_scale, cudaStream_t s);
/// The decode attention's splits (a fixed grid: the position read on the device).
int mla_nsp(int rows);
size_t mla_scratch_floats(int H, int nsp);
/// One token's attention at *dpos over the cache rows 0 .. *dpos: qa [H][512] (k_b q_nope), qf [H][192] (its last 64
/// the roped q_pe), scale; out [H][512] F32 and its int8 rows (oq, od: q8's over the row of H 512, null: none).
void mla_decode(const float* qa, const float* qf, const __half* cache, const int* dpos, int H, float scale, int nsp,
                float* scratch, float* out, int8_t* oq, float* od, cudaStream_t s);
/// y[h][d] = x[h][d] sigmoid(g[h]) (rows of hd, H heads) and its int8 rows (q, d; null: none), y F32 (may be x).
void head_gate(const float* x, const float* g, int H, int hd, float* y, int8_t* q, float* dq, cudaStream_t s);
/// (the prompt path) the same for T rows [T][H][hd] with gates [T][H], into F16.
void head_gate_rows(const float* x, const float* g, int T, int H, int hd, __half* y, cudaStream_t s);
/// (the prompt path) Qcat [T][H][576]'s rope part: F16(rope(q [T][H][192]'s last 64)) at [512:576], pos = pos0 + t.
void mla_qpe_f16(const float* q, __half* qcat, int T, int H, int pos0, float theta_scale, cudaStream_t s);
/// (the prompt path) the scores' softmax, rows r of a tile (token ta + r / H at pos0 + ta + r / H), columns c < nk
/// (the keys at positions c): visible when c <= that position; S (already scaled) in place -> P (F16, ldp >= nk; the
/// columns from nk to ldp zero).
void mla_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int H, int ta, int pos0, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ the routing
/// Ling's routing for T tokens (a warp a token): p = sigmoid(RL[t][e]), s = p + bias; groups of ne / ng experts
/// scored by the sum of their top 2 s, the top ngu groups kept (ties: the lower group), the top nu of s there (ties:
/// the lower id) -> sel [T][nu] in score order; w [T][nu] = p / max(sum p, 6.1e-5) x scale.  ne <= 512, nu <= 16.
void route(const float* RL, int ld, const float* bias, int T, int ne, int ng, int ngu, int nu, float scale, int* sel,
           float* w, cudaStream_t s);
/// counts[e] += the tokens that chose e (counts zeroed by the caller).
void expert_count(const int* sel, int n, int* counts, cudaStream_t s);
/// Each expert's tokens in token order: list[off[e] + i] = t, wl[...] = its weight (a warp an expert; deterministic).
void expert_lists(const int* sel, const float* w, int T, int nu, int ne, const int* off, int* list, float* wl,
                  cudaStream_t s);
/// y[list[i]] += wl[i] * ye[i] (rows of n values; each list entry a different row of y).
void scatter_add(const float* ye, const int* list, const float* wl, int rows, int n, float* y, cudaStream_t s);

/// (--selftest) the KDA recurrence and the MLA decode against host references on random data; prints, true = pass.
bool selftest();

/// One warp that sleeps for ns nanoseconds: a card's work kept "busy" for its driver's clocks.
void keepalive(unsigned long long ns, cudaStream_t s);

}  // namespace ling::k
