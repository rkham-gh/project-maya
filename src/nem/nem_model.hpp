// src/nem/nem_model.hpp - maya-nem's model: NVIDIA Nemotron 3.5 Super (120B, 12B active; llama.cpp's
// "nemotron_h_moe" architecture, unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF) from a GGUF on one NVIDIA GPU and
// the CPU: the Mamba-2 and attention layers, the MoE layers' router, latent projections and shared expert, and the
// output head in VRAM; the routed experts read where they lie in the GGUF's mapping by the CPU lane (nem_cpu.hpp), the
// ones used most also in VRAM.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nem {

/// The model's shape, from the GGUF's nemotron_h_moe.* metadata.
struct Geo {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_ff_sh = 0, n_lat = 0;
    float exp_scale = 1.0f, eps = 1e-5f;
    bool exp_norm = true;
    // Mamba-2: d_inner = n_ssm_head x head dim, B / C n_group x d_state each, a causal conv of d_conv over x | B | C
    int d_inner = 0, d_state = 0, n_group = 0, n_ssm_head = 0, d_conv = 0;
    int vocab = 0, ctx_train = 0, eos = -1;
    std::vector<uint8_t> kind;   // per layer: kMamba, kAttn, kMoe
};
constexpr uint8_t kMamba = 0, kAttn = 1, kMoe = 2;

struct Options {
    std::string gguf;     // the first shard (...-00001-of-0000N.gguf); the others are found beside it
    int max_ctx = 262144;
    int threads = 0;      // CPU lane threads in all (0: the physical cores)
    int gpu = 0;
    int chunk = 8192;     // the prompt path's tokens a chunk (prefill): every expert a chunk routes to crosses PCIe once
    int stage = 64;       // the prompt path copies the routed experts in groups of this many (two VRAM buffers in turn)
    double cache_gb = -1; // VRAM for the most-used routed experts (-1: what is free, less NEM_VRAM_RESERVE_MIB)
    // A second card that only holds routed experts (the next most used after the main card's): a token's experts found
    // there are multiplied on it while the main card and the CPU lane do theirs.  The layers, the states and the prompt
    // path stay on `gpu`.  -1: none.
    int helper_gpu = -1;
    double helper_gb = -1; // its VRAM for them (-1: what is free, less NEM_HELPER_RESERVE_MIB)
    std::string profile;  // the experts' use scores: read at the start (the cache filled from them), save_profile()
    bool verbose = true;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool load(const Options& o, std::string& err);
    const Geo& geo() const;
    int max_ctx() const;
    /// Back to position 0, every state cleared.
    void reset();
    int pos() const;
    /// One token at pos(): the next token's logits (host memory, geo().vocab values), pos + 1.  nullptr when pos() is at
    /// max_ctx().
    const float* step(int token);
    /// n tokens at pos() as one prompt (chunks of Options::chunk on the GPU, the routed experts streamed to it): the
    /// logits after the last one (as step()), pos + n.  nullptr when they do not fit before max_ctx().  progress (when
    /// set) is called as the chunks go, with the tokens read so far.
    const float* prefill(const int* ids, int n, void (*progress)(void*, int) = nullptr, void* ctx = nullptr);

    struct Stats {
        double ms_total = 0, ms_cpu = 0;   // step()'s time, and the CPU lane's part of it
        long tokens = 0;
        double pf_ms = 0, pf_copy_gb = 0;  // prefill()'s time, and the expert bytes it copied to the GPU
        long pf_tokens = 0;
        long lookups = 0, hits = 0;        // step()'s routed experts, and those it found in VRAM
        long cache_slots = 0, cache_copies = 0;
        long hits_helper = 0;              // ... and those it found on the helper card
        long helper_slots = 0, helper_copies = 0;
        double ms_lane = 0, ms_helper = 0, ms_update = 0;   // in ms_cpu: the CPU lane's part; the wait for the helper
                                                            // after it; and the helper's swaps between tokens
        long lane_layers = 0, helper_layers = 0;            // the MoE layers that used each
    };
    Stats stats() const;
    void clear_stats();
    /// The experts' use scores into Options::profile's file (false: not written).
    bool save_profile() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace nem
