// src/ink/ink_cpu.hpp - maya-ink's CPU lane: a token's routed experts that are not in VRAM, computed from host memory
// (the GGUF's mapping) with maya-ds4's CPU rows (src/ds4/ds4_gguf_cpu.hpp: ggml-cpu's dots, or ds4's own rows where
// they are their bits), Inkling's plain SwiGLU between gate / up and down.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ink {

/// A fixed set of worker threads; run() hands out jobs 0 .. n - 1 (the caller takes some too) and returns when all
/// are done.
class Pool {
public:
    explicit Pool(int workers);
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    void run(int n, const std::function<void(int)>& f);
    int threads() const;   // the workers and the caller

private:
    struct Impl;
    Impl* p;
};

/// One MoE layer's routed experts in host memory: each role's ggml type, bytes a row and base (expert e's rows start
/// at base + e x rows x row bytes), and the activations' types their rows take.
struct CpuLayer {
    int tg = 0, tu = 0, td = 0;
    size_t rg = 0, ru = 0, rd = 0;
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    int act_g = 0, act_u = 0, act_d = 0;
    size_t ab_g = 0, ab_u = 0, ab_d = 0;   // bytes of an activation of each
};

/// The layer's format from the GGUF's types; false (err) when the CPU rows take none for a type.
bool cpu_layer(int tg, int tu, int td, int n_embd, int n_ff, CpuLayer& L, std::string& err);

/// out[n_embd] = sum over i < n of w[i] * down_e(silu(gate_e x) * up_e x) for e = experts[i], x [n_embd] (F32), on the
/// pool's threads; the experts summed in the order given (the same bits every run).
void cpu_experts(Pool& pool, const CpuLayer& L, int n_embd, int n_ff, const int* experts, const float* w, int n,
                 const float* x, float* out);

}  // namespace ink
