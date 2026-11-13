// src/ink/ink_model.hpp - maya-ink's model: Inkling-Small (Thinking Machines, 276B MoE, 12B active) from a GGUF
// (llama.cpp's "inkling" architecture, ggml-org/llama.cpp#25731; unsloth/Inkling-Small-GGUF) on one NVIDIA GPU and
// the CPU: the attention, the dense layers, the shared experts and the output head in VRAM; the routed experts read
// where they lie in the GGUF's mapping by the CPU lane (ink_cpu.hpp).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ink {

/// The model's shape, from the GGUF's inkling.* metadata.
struct Geo {
    int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_shexp = 0, n_dense = 0;
    float exp_scale = 1.0f, eps = 1e-6f, logit_scale = 1.0f, log_alpha = 0.0f;
    int n_swa = 0, d_rel = 0, rel_ext = 0, rel_ext_swa = 0, conv_k = 0, log_floor = 0;
    int vocab = 0, vocab_unpadded = 0, ctx_train = 0;
    std::vector<uint8_t> swa;   // per layer: 1 sliding window, 0 global
    // Inkling 2: the output's softcap, the q / k and relative states' L2 clips, the experts' gate / up softcaps, the
    // layers in an attention-residual block
    float lcap = 0.0f, qkclip = 0.0f, rclip = 0.0f, gcap = 0.0f, ucap = 0.0f;
    int res_block = 6;
};

struct Options {
    std::string gguf;     // the first shard (...-00001-of-0000N.gguf); the others are found beside it
    int max_ctx = 131072;
    int threads = 0;      // CPU lane threads in all (0: the physical cores)
    int gpu = 0;
    int chunk = 8192;     // the prompt path's tokens a chunk (prefill): every expert a chunk routes to crosses PCIe once
    int stage = 32;       // the prompt path copies the routed experts in groups of this many (two VRAM buffers in turn)
    double cache_gb = -1; // VRAM for the most-used routed experts (-1: what is free, less INK_VRAM_RESERVE_MIB)
    // A second card that only holds routed experts (the next most used after the main card's): a token's experts found
    // there are multiplied on it while the main card and the CPU lane do theirs.  The layers, the states and the prompt
    // path stay on `gpu`.  -1: none.
    int helper_gpu = -1;
    double helper_gb = -1; // its VRAM for them (-1: what is free, less INK_HELPER_RESERVE_MIB)
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
    /// One token at pos(): the next token's logits (host memory, geo().vocab values; the padded ids -inf), pos + 1.
    /// nullptr when pos() is at max_ctx().
    const float* step(int token);
    /// n tokens at pos() as one prompt (chunks of Options::chunk on the GPU, the routed experts streamed to it): the
    /// logits after the last one (as step()), pos + n.  nullptr when they do not fit before max_ctx().  progress (when
    /// set) is called as the chunks go, with the tokens read so far.
    const float* prefill(const int* ids, int n, void (*progress)(void*, int) = nullptr, void* ctx = nullptr);
    /// What later tokens overwrite of the state at pos() - every layer's short convolutions and the sliding layers'
    /// K / V rings - copied aside in VRAM (79 MB); restore() puts it back and the position with it.  The global layers'
    /// caches need nothing: a row is only ever written at the position, so the rows below it are as they were.  A
    /// conversation's next turn that does not continue the tokens so far (its last answer rendered without its
    /// thinking, or a little differently) then goes on from the last prompt's end instead of from the start
    /// (maya-ling's, src/ling).  false: no room was kept for it (INK_SNAPSHOT=1 keeps it) or nothing taken yet.
    bool snapshot();
    bool restore();

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
        long tail_skips = 0;                                // the experts INK_TAIL_KEEP left out
    };
    Stats stats() const;
    void clear_stats();
    /// The experts' use scores into Options::profile's file (false: not written).
    bool save_profile() const;
    /// (--bench-h2d) times the ways to copy the routed experts to the GPU, to stderr.
    void bench_h2d();

private:
    struct Impl;
    Impl* p;
};

}  // namespace ink
