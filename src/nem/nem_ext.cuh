// src/nem/nem_ext.cuh - maya-nem's Nemotron-only device kernels: the Mamba-2 mixer (llama.cpp's build_mamba2_layer with
// ggml_ssm_conv / ggml_ssm_scan, src/models/mamba-base.cpp, MIT), the latent MoE's relu^2 and routing.  The rest (the
// norms, the int8 rows, the attention, the expert lists) are maya-ink's kernels in nem_kernels.cuh.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nem::k {

/// A token's causal conv over the Mamba-2 mixer's x | B | C (ch channels, in place): xbc[c] = silu(sum_i w[c][i] in(i)
/// + b[c]) with in = [state[c][0 .. K - 2], xbc[c]] (the oldest first); state then holds the last K - 1 inputs.
void mamba_conv_step(float* xbc, float* state, const float* w, const float* b, int ch, int K, cudaStream_t s);
/// A chunk's: out[t][c] = silu(sum_i w[c][i] in(t - (K - 1) + i) + b[c]) for t < T, in(t') = x[t' ldx + c] for t' >= 0,
/// else state[c][t' + K - 1]; then state = the last K - 1 inputs (the old state's where T < K - 1).  out rows of ch.
void mamba_conv_chunk(const float* x, int ldx, float* out, float* state, const float* w, const float* b, int T, int ch,
                      int K, cudaStream_t s);
/// The selective scan over T tokens (T = 1: a token's step), H heads of P values, G groups of N state values (head h
/// reads group h / (H / G)):  dt = softplus(dtr[t][h] + dt_bias[h]),  s[h][p][n] = s exp(dt A[h]) + B[t][g][n] x[t][h][p]
/// dt,  y[t][h][p] = (sum_n s C[t][g][n] + D[h] x[t][h][p]) silu(z[t][h][p]).  x, B, C: rows of ldxbc (x at 0, B at H P,
/// C at H P + G N); z, dtr: rows of ldz (z at 0, dtr at dt_off); y rows of H P; state [H][P][N] F32, carried on.
/// P a multiple of 16, N of 32 and at most 256.
void mamba_scan(const float* xbc, int ldxbc, const float* z, int ldz, int dt_off, const float* dt_bias, const float* A,
                const float* D, float* state, float* y, int T, int H, int P, int G, int N, cudaStream_t s);
/// In place, rows of groups x n values: each group y * rsqrt(mean(y^2) + eps) * w[g][..] (w [groups][n]).
void group_rmsnorm(float* y, const float* w, int rows, int groups, int n, float eps, cudaStream_t s);
/// x = relu(x)^2, n values.
void relu2(float* x, size_t n, cudaStream_t s);
/// h[r][c] = F16(relu(x[r][c])^2), x rows of ldx, h rows of ff.
void relu2_f16(const float* x, int ldx, int rows, int ff, __half* h, cudaStream_t s);
/// Nemotron's routing, a token a warp: the top nu of sigmoid(RL[t][e]) + bias[e] over ne experts (ties: the lower id),
/// in score order -> sel [T][nu]; their weights = their sigmoids (/ their sum when norm) x scale -> w [T][nu].  ne at
/// most 512 and a multiple of 32, nu at most 32.
void route_sig(const float* RL, int ld, const float* bias, int T, int ne, int nu, bool norm, float scale, int* sel,
               float* w, cudaStream_t s);
/// out[c] = (flags[0] ? cpu[c] : 0) + (flags[1] ? gpu[c] : 0), n values.
void lat_sum(const float* cpu, const float* gpu, const int* flags, float* out, int n, cudaStream_t s);
/// y += a + b, n values.
void add2(float* y, const float* a, const float* b, int n, cudaStream_t s);

}  // namespace nem::k
