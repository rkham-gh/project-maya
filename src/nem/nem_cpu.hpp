// src/nem/nem_cpu.hpp - maya-nem's CPU lane: a token's routed experts that are not in VRAM, computed from host memory
// (the GGUF's mapping) with maya-ds4's CPU rows (src/ds4/ds4_gguf_cpu.hpp: ggml-cpu's dots, or ds4's own rows where
// they are their bits).  Nemotron's experts live in the MoE's latent space (n_lat = 1024 values in and out): up, then
// relu(.)^2, then down - no gate.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../ds4/ds4_gguf_cpu.hpp"

namespace nem {

/// A fixed set of worker threads; run() hands out jobs 0 .. n - 1 (the caller takes some too) and returns when all
/// are done.  start() / join() split it: the workers begin at start(), the caller joins in at join().  Between runs the
/// workers spin a while (NEM_POOL_SPIN_US, default 3000: a token's layers come faster than a sleeping thread wakes on
/// Windows), then sleep.
class Pool {
public:
    explicit Pool(int workers);
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    void run(int n, const std::function<void(int)>& f);
    void start(int n, const std::function<void(int)>* f);   // f lives until join() returns
    void join();
    int threads() const;   // the workers and the caller

private:
    struct Impl;
    Impl* p;
};

/// One MoE layer's routed experts in host memory: up [n_ff rows of n_in] and down [n_in rows of n_ff] - each role's
/// ggml type, bytes a row and base (expert e's rows start at base + e x rows x row bytes) - and the activations' types
/// their rows take.
struct CpuLayer {
    int tu = 0, td = 0;
    size_t ru = 0, rd = 0;
    const uint8_t *up = nullptr, *down = nullptr;
    int act_u = 0, act_d = 0;
    size_t ab_u = 0, ab_d = 0;   // bytes of an activation of each
};

/// The layer's format from the GGUF's types; false (err) when the CPU rows take none for a type.
bool cpu_layer(int tu, int td, int n_in, int n_ff, CpuLayer& L, std::string& err);

/// out[n_in] = sum over i < n of w[i] * down_e(relu(up_e x)^2) for e = experts[i], x [n_in] (F32), on the pool's
/// threads; the experts summed in the order given (the same bits every run).  begin() starts the workers on it (the
/// caller is free until end(), which takes its share of the jobs and returns with out written): one run of (expert, band
/// of up rows) jobs, each expert's activation for its down rows made by the job that finishes its last band, then (band
/// of down rows) jobs, each summing its rows over the experts.  x, out and the arrays stay put until end().
class CpuLane {
public:
    static constexpr int kMax = 32;
    void begin(Pool& pool, const CpuLayer& L, int n_in, int n_ff, const int* experts, const float* w, int n,
               const float* x, float* out);
    void end();

private:
    void job(int j);
    Pool* pool_ = nullptr;
    const CpuLayer* L_ = nullptr;
    int n_in_ = 0, n_ff_ = 0, n_ = 0, bands_u_ = 0, bands_d_ = 0;
    int experts_[kMax] = {};
    float w_[kMax] = {};
    float* out_ = nullptr;
    std::vector<uint8_t> au_, hq_;
    std::vector<float> h_, o_;
    std::atomic<int> done_u_[kMax];
    std::atomic<int> ready_{0};
    std::function<void(int)> fn_;
};

}  // namespace nem
