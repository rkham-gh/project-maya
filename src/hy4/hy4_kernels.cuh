// src/hy4/hy4_kernels.cuh - maya-hy4's own device kernels: Ling-3.1-flash's parts that the shared GGUF kernels
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

namespace hy4::k {

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
void gemv_f(const void* w, int type, size_t row_bytes, const float* x, float* y, int n, int k, cudaStream_t s,
            bool f32_order = false);   // (f32_order: BF16 rows of a BF16-exact F32 matrix, summed as the F32 rows)
/// dst[i] = src[i] (BF16 -> F32, exact), n values.
void bf16_to_f32(const void* src, size_t n, float* dst, cudaStream_t s);
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
/// RoPE (ggml's NORM: pairs (2i, 2i + 1), theta = pos theta_scale^i) on q [T][H][qk]'s 64 values from nope on, in
/// place, pos = (dpos ? *dpos : pos0) + t.
void mla_rope_q(float* q, int T, int H, int qk, int nope, int pos0, const int* dpos, float theta_scale, cudaStream_t s);
/// (ring > 0: the cache is a ring of that many rows, position p in row p % ring - mla_kv_store's and mla_decode's
/// sliding-window layers.)
/// The cache rows of T tokens: kva [T][576] -> cache[pos][0:512] = F16(rmsnorm(kva[0:512]) w), [512:576] =
/// F16(rope(kva[512:576])), pos = (dpos ? *dpos : pos0) + t.
void mla_kv_store(const float* kva, const float* w, float eps, __half* cache, int T, int pos0, const int* dpos,
                  float theta_scale, cudaStream_t s, int ring = 0, int base = 0);
/// The decode attention's splits (a fixed grid: the position read on the device).
int mla_nsp(int rows);
size_t mla_scratch_floats(int H, int nsp);
/// One token's attention at *dpos over the cache rows 0 .. *dpos (win > 0: only the last win of them, a sliding
/// window; cells set: the rows cells[0 .. *ncell) instead, the indexer's choice): qa [H][512] (k_b q_nope), qf [H][qk] (the roped q_pe at nope), scale; sinks (null: none) a logit a head
/// added to the softmax's sum; out [H][512] F32 and its int8 rows (oq, od: q8's over the row of H 512, null: none).
void mla_decode(const float* qa, const float* qf, int qk, int nope, const __half* cache, const int* dpos, int win,
                const int* cells, const int* ncell, const float* sinks, int H, float scale, int nsp, float* scratch,
                float* out, int8_t* oq, float* od, cudaStream_t s, int ring = 0);
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

// ------------------------------------------------------------------------------------------------ Hy4
/// The hyper-connections' read of the hc streams R [hc][n]: mix [nmix] = fn [nmix][hc n] . R scaled by rsqrt(mean(R^2) +
/// eps); x = sum_i pre_i R_i with pre_i = sigmoid(mix[i] scale[0] + base[i]) + hc_eps; with post set (the layers' hc,
/// nmix = 2 hc) also post [hc] = sigmoid(mix[hc + i] scale[1] + base[hc + i]) mag + hc_eps.  (The head: nmix = hc, no
/// post.)  mix: nmix device floats of scratch.
void hc_read(const float* R, const float* fn, int nmix, const float* scale, const float* base, int hc, int n, float eps,
             float hc_eps, float mag, float* mix, float* x, float* post, cudaStream_t s);
/// A sublayer's end: y = rmsnorm(a + b) w (b may be null), then R_i += post_i y for each of the hc streams.
void norm_write(const float* a, const float* b, const float* w, float eps, const float* post, int hc, int n, float* R,
                cudaStream_t s);
/// R_i = x for each of the hc streams.
void hc_init(const float* x, int hc, int n, float* R, cudaStream_t s);
/// Engram's rows [cols][dim] (F32) RMS-normed (eps) times w [cols][dim] -> y [cols dim] and its int8 rows.
void eg_rows(const float* rows, const float* w, int cols, int dim, float eps, float* y, int8_t* q, float* d,
             cudaStream_t s, int wcols = 0);   // (wcols > 0: w's rows repeat every wcols rows - several tokens'; q may be
                                               //  null)
/// Engram's update of the hc streams R [hc][n] from the projected rows kv [hc + 1][n] and its bias kb (keys, then the
/// value): gw [hc][n] the gate's weights, hs [hc][n] the per-stream offsets, gamma [n] the value's.
void eg_update(const float* kv, const float* kb, const float* gw, const float* hs, const float* gamma, int hc, int n,
               float eps, float* R, cudaStream_t s);
/// y = (flags[0] ? cpu : 0) + (flags[1] ? gpu : 0) (n values, a multiple of 32) and its int8 rows.
void lat_sum_q8(const float* cpu, const float* gpu, const int* flags, int n, float* y, int8_t* q, float* d,
                cudaStream_t s);

/// The indexer (sparse layers): the token's value and gate rows (dim each) into slot *dpos % nring of ring
/// [nring][2 dim] (nring a multiple of kpool: the open group and the next ones a draft's rejected tokens may write -
/// Geo::idx_nring).
void idx_store(const float* iv, const float* ig, const int* dpos, int nring, int dim, float* ring, cudaStream_t s);
/// When *dpos ends a group: its pooled key (softmax over the members of gate + ape, per element, weighting value; then
/// RMS-normed times w; RoPE on its last 64 values at the group's first position) into pooled[*dpos / kpool].
void idx_pool(const float* ring, const float* ape, const float* w, const int* dpos, int kpool, int nring, int dim,
              float eps,
              float theta_scale, float* pooled, cudaStream_t s);
/// score[g] = sum_h relu(q_h . pooled_g) w_h scale for the complete groups before the token's own (q [nh][dim]).
void idx_score(const float* q, const float* w, const float* pooled, const int* dpos, int kpool, int nh, int dim,
               float scale, int max_pools, float* score, cudaStream_t s);
/// The cells the token attends to: all of them up to *dpos while there are no more than top_pools complete groups
/// before its own, else the top top_pools groups' cells and its own group's; *ncell their count.
void idx_select(const float* score, const int* dpos, int kpool, int top_pools, int* cells, int* ncell, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ Hy4's prompt path
/// hc_read for T rows: R [T][hc][n], mix [T][nmix] (scratch), x [T][n], post [T][hc] (or null).
void hc_read_rows(const float* R, const float* fn, int nmix, const float* scale, const float* base, int hc, int n,
                  float eps, float hc_eps, float mag, float* mix, float* x, float* post, int T, cudaStream_t s);
/// norm_write for T rows: a, b [T][n] (b may be null), post [T][hc], R [T][hc][n].
void norm_write_rows(const float* a, const float* b, const float* w, float eps, const float* post, int hc, int n,
                     float* R, int T, cudaStream_t s);
/// R[t][i] = x[t] for each of the hc streams, T rows.
void hc_init_rows(const float* x, int hc, int n, float* R, int T, cudaStream_t s);
/// eg_update for T rows: kv [T][(hc + 1) n], R [T][hc][n].
void eg_update_rows(const float* kv, const float* kb, const float* gw, const float* hs, const float* gamma, int hc, int n,
                    float eps, float* R, int T, cudaStream_t s);
/// Qcat [T][H][576]'s rope part, F16(rope(q [T][H][qk]'s 64 values from nope on)) at [512:576], pos = pos0 + t.
void qpe_f16(const float* q, __half* qcat, int T, int H, int qk, int nope, int pos0, float theta_scale, cudaStream_t s);
/// n rows of 576 for positions p0 .. p0 + n out of a ring (row p % ring) into dst, or from src into the ring.
void ring_get(const __half* ring_kv, int ring, int p0, int n, __half* dst, cudaStream_t s);
void ring_put(const __half* src, int ring, int p0, int n, __half* ring_kv, cudaStream_t s);
/// The prompt path's scores softmax: rows of M = tokens x H (token ta + r / H at qpos0 + r / H), columns the nk keys at
/// kpos0 + c - visible at or before the token's position, within win (> 0) of it, and for a token with sel_on set in its
/// own group or a group its bits (nwords a token) mark; the sinks a logit a head in the sum; P (F16, ldp) the result.
void attn_softmax(float* S, int lds, __half* P, int ldp, int M, int nk, int H, int qpos0, int kpos0, int win,
                  const unsigned* bits, int nwords, const int* sel_on, int kpool, const float* sinks, cudaStream_t s);
/// The indexer's pooled keys of the groups g0 .. g0 + n_groups that end in the rows (iv, ig [T][dim] at positions p0 ..),
/// members before p0 from the ring; into pooled[g].
void idx_pool_rows(const float* iv, const float* ig, const float* ring, const float* ape, const float* w, int p0, int g0,
                   int n_groups, int kpool, int nring, int dim, float eps, float theta_scale, float* pooled,
                   cudaStream_t s);
/// rows r0 .. r0 + n of (iv, ig) at positions p0 + r into the ring's slots.
void idx_ring_put(const float* iv, const float* ig, int p0, int r0, int n, int nring, int dim, float* ring,
                  cudaStream_t s);
/// S [tq][npool] = sum_h relu(A [tq nh][npool]) w [tq][nh] scale, -inf from the token's own group on.
void idx_score_rows(const float* A, const float* w, int nh, int npool, int tq, int qpos0, int kpool, float scale, float* S,
                    cudaStream_t s);
/// Each of tq tokens' selection: sel_on 0 (every key) or the bits of its top top_pools groups (cells: scratch).
void idx_select_rows(const float* S, int npool, int tq, int qpos0, int kpool, int top_pools, int* cells, unsigned* bits,
                     int nwords, int* sel_on, cudaStream_t s);

// ------------------------------------------------------------------------------------------------ the device's routing
/// A MoE layer's plan and the host's answer, in host-mapped memory (ints).  Plan: [0] the token's sequence number once
/// the rest is there, [1] the experts on the main card, [2 + i] the i-th choice, [18 + i] its weight (float bits),
/// [34 + i] its main-card slot (-1: not there), the latent (F32) from kPlanHead.  Answer: [0] the sequence number once
/// the sum is there, the CPU lane's and the helper's latent sum (F32) from kAnsHead.
constexpr int kPlanHead = 64, kAnsHead = 16;
/// After the router's top nu (sel, w on the device): each choice's slot from where (the layer's [n_expert]) and its
/// matrices' pointer from slotptr (null: not on the main card) into gptr [nu], the experts the host must add into *need,
/// the plan into plan (the latent lat [nl] in it); its sequence number dpos[2] last.
/// The latent's int8 rows (xq [nl], scales xd [nl / 32]: the main card's experts' input) go in it too, after the latent:
/// at plan_q(plan, nl), plan_d(plan, nl) - the helper reads them in 16-byte pieces.
void moe_plan(const int* sel, const float* w, int nu, const float* lat, const int8_t* xq, const float* xd, int nl,
              const int* where, const uint8_t* const* slotptr, const uint8_t** gptr, int* need, int* plan,
              const int* dpos, cudaStream_t s);
/// A plan's ints: the head, the latent, its int8 rows and scales (16-byte pieces)
inline size_t plan_ints(int nl) { return ((size_t) kPlanHead + nl + nl / 4 + nl / 32 + 63) & ~(size_t) 63; }
/// With *need > 0: waits for the answer's sequence number dpos[2] (bounded: ~30 s, then diag[0] + 1), its sum into
/// tail + 4; tail's flags [0] (the host's sum to add) and [1] (the main card's, always) for lat_sum_q8.
void moe_wait(const int* ans, const int* need, const int* dpos, int nl, float* tail, unsigned* diag, cudaStream_t s);
/// moe_wait with the helper card's own answer (hans, as an answer: [1] its count of experts): with *need > 0, the
/// helper's answer for dpos[2], then the host's when the helper took fewer than *need; tail + 4 = the host's sum, plus
/// the helper's (the host's routing's order).
void moe_wait2(const int* ans, const int* hans, const int* need, const int* dpos, int nl, float* tail, unsigned* diag,
               cudaStream_t s);
/// (the helper card) A MoE layer's plan once its sequence number dpos[2] is there (bounded: diag[0] + 1): each choice
/// not on the main card that this card holds (where: the layer's [n_expert] slots here) its pointer from slotptr into
/// gptr (else null), the weights into w, the latent into lat [nl], their count into *nh.
void hd_plan(const int* plan, int nu, int nl, const int* where, const uint8_t* const* slotptr, const uint8_t** gptr,
             float* w, int8_t* xq, float* xd, int* nh, const int* dpos, unsigned* diag, cudaStream_t s);
/// (the helper card) Its experts' rows y [nu][nl] summed in order (sum_rows') and its count *nh into its answer hans,
/// the sequence number dpos[2] last.
void hd_answer(const float* y, int nu, const int* nh, int nl, int* hans, const int* dpos, cudaStream_t s);

/// norm_write(a, b, wpost, eps, post_in, hc, n, R), then hc_read(R, fn, nmix, scale, base, ..., mix, x, post_out) and
/// norm_q8(x, wn, eps, y, q, d, n) in one launch, the same bits (false, nothing launched: a shape it does not take).
bool hc_step(const float* a, const float* b, const float* wpost, const float* post_in, float eps, int hc, int n,
             float* R, const float* fn, int nmix, const float* scale, const float* base, float hc_eps, float mag,
             float* mix, float* x, float* post_out, const float* wn, float* y, int8_t* q, float* d, cudaStream_t s,
             int T = 1);   // (T tokens: a block each - a, b, x, y, q rows of n, R [T][hc][n], post [T][hc], mix [T][16])

// ------------------------------------------------------------------------------------------------ a window's routing
/// A window's plan (T <= Tm tokens, one sequence number), ints: [0] the sequence number once the rest is there, [1] T,
/// [2] the choices on the main card; from kWinHead the choices [Tm][16]: experts, then weights (float bits) at + 16 Tm,
/// then main-card slots (-1: not there) at + 32 Tm; the latents F32 [Tm][nl] at winplan_lat, their int8 rows
/// [Tm][nl] and scales [Tm][nl / 32] after.  An answer: [0] the sequence number, [1] the helper's count, the sums
/// [Tm][nl] (F32) from kAnsHead.
constexpr int kWinHead = 64;
#ifdef __CUDACC__
__host__ __device__
#endif
inline size_t winplan_lat(int Tm) { return ((size_t) kWinHead + 48 * (size_t) Tm + 15) & ~(size_t) 15; }
inline size_t winplan_ints(int Tm, int nl) {
    return (winplan_lat(Tm) + (size_t) Tm * nl + (size_t) Tm * nl / 4 + (size_t) Tm * nl / 32 + 63) & ~(size_t) 63;
}
inline size_t winans_ints(int Tm, int nl) { return ((size_t) kAnsHead + (size_t) Tm * nl + 63) & ~(size_t) 63; }
/// moe_plan for a window: T tokens' nu choices each (sel, w [T][nu]), their latents (lat, xq, xd [T][nl..]).
void moe_plan_w(const int* sel, const float* w, int T, int Tm, int nu, const float* lat, const int8_t* xq,
                const float* xd, int nl, const int* where, const uint8_t* const* slotptr, const uint8_t** gptr,
                int* need, int* plan, const int* dpos, cudaStream_t s);
/// moe_wait2 for a window: tail + 4 + t nl = the host's sum (0 where it has none) + the helper's (0 likewise).
void moe_wait2_w(const int* ans, const int* hans, const int* need, const int* dpos, int T, int nl, float* tail,
                 unsigned* diag, cudaStream_t s);
/// hd_plan for a window: the choices not on the main card that the helper holds, their weights, the int8 rows [T].
void hd_plan_w(const int* plan, int T, int Tm, int nu, int nl, const int* where, const uint8_t* const* slotptr,
               const uint8_t** gptr, float* w, int8_t* xq, float* xd, int* nh, const int* dpos, unsigned* diag,
               cudaStream_t s);
/// hd_answer for a window: each token's nu rows of y [T nu][nl] summed in order into the answer [T][nl].
void hd_answer_w(const float* y, int T, int nu, const int* nh, int nl, int* hans, const int* dpos, cudaStream_t s);

/// The decode attention's kernel chosen, at the load (HY4_MLA_TC=0: the FP32 one): true, the tensor cores'.
bool mla_init();
/// (HY4_MLA_QMAX=1) the largest |q| the tensor cores' kernel has seen so far: the 512 (out[0]) and the rope part
/// (out[1]); false: not recorded.
bool mla_qmax(float* out);

/// (--selftest) nothing yet; true.
bool selftest();

/// One warp that sleeps for ns nanoseconds: a card's work kept "busy" for its driver's clocks.
void keepalive(unsigned long long ns, cudaStream_t s);

}  // namespace hy4::k
