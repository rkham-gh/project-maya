// src/hy4/hy4_model.hpp - maya-hy4's model: Tencent's Hy4 Flash (a 42-layer MoE: 12 sparse-attention and 30
// sliding-window gated MLA layers, identity hyper-connections over 4 residual streams, an Engram n-gram table read at 4
// layers, 512 latent routed experts of which 16 a token plus one shared) from a GGUF (llama.cpp's "hy_v4_flash"
// architecture, ggml-org/llama.cpp#32517; AngelSlim/Hy4-flash-GGUF) on one NVIDIA GPU and the CPU: the attention, the
// dense layers, the shared experts and the output head in VRAM; the routed experts read where they lie in the GGUF's
// mapping by the CPU lane (hy4_cpu.hpp), the most used of them in a VRAM cache (and on a helper card).
// (maya-ling's structure, src/ling in the ling-2gpu branch.)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hy4 {

/// The model's shape, from the GGUF's hy_v4_flash.* metadata.
struct Geo {
    int n_layer = 0;   // the main layers (block_count less nextn_predict_layers: the MTP block is not run)
    int n_embd = 0, n_ff = 0, n_dense = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0, n_ff_sh = 0;
    int n_lat = 0;     // the routed experts' width (expert_latent_length): routed_down E -> n_lat, routed_up back
    float exp_scale = 1.0f, eps = 1e-5f;
    int q_lora = 0, kv_lora = 0, rope_dim = 0, qk_head = 0, v_head = 0;
    float rope_base = 1e7f, rope_base_swa = 1e4f;
    int swa_window = 0;
    // the indexer (sparse layers)
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0, idx_kpool = 0, idx_tail = 0;
    // hyper-connections
    int hc = 0;
    float hc_eps = 1e-6f, hc_mag = 2.0f;
    // Engram
    int eg_ngram = 0, eg_heads = 0, eg_dim = 0, eg_cols = 0, eg_pad = 0;
    std::vector<int> eg_layers;
    int vocab = 0, ctx_train = 0, eos = -1;
    std::vector<int> n_head;           // per layer: 32 (sparse) or 64 (sliding window)
    std::vector<uint8_t> swa;          // per layer: 1 a sliding-window layer, 0 a sparse one
    std::vector<uint8_t> idx;          // per layer: 1 the layer has an indexer
    std::vector<float> clamp_exp;      // per layer: the routed experts' SwiGLU clamp (0: none)
};

struct Options {
    std::string gguf;     // the first shard (...-00001-of-0000N.gguf); the others are found beside it
    int max_ctx = 131072;
    int threads = 0;      // CPU lane threads in all (0: the physical cores)
    int gpu = 0;
    int chunk = 4096;     // the prompt path's tokens a chunk (prefill): every expert a chunk routes to crosses PCIe once
    int sub = 1024;       // ... its attention's and Engram's tokens at a time
    int stage = 32;
    double cache_gb = -1; // VRAM for the most-used routed experts (-1: what is free, less HY4_VRAM_RESERVE_MIB)
    // A second card that only holds routed experts (the next most used after the main card's): a token's experts found
    // there are multiplied on it while the main card and the CPU lane do theirs.  -1: none.
    int helper_gpu = -1;
    double helper_gb = -1; // its VRAM for them (-1: what is free, less HY4_HELPER_RESERVE_MIB)
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
    /// Back to position 0, every state cleared (failed() too).
    void reset();
    /// step() / prefill() returned nullptr because a card gave up waiting for the other (~30 s, the device's routing):
    /// that token and the conversation's state are not to be trusted - fail the request and reset().
    bool failed() const;
    int pos() const;
    /// One token at pos(): the next token's logits (host memory, geo().vocab values), pos + 1.  nullptr when pos() is
    /// at max_ctx().
    const float* step(int token);
    /// n tokens at pos() as one prompt: the logits after the last one (as step()), pos + n.  nullptr when they do not
    /// fit before max_ctx().  progress (when set) is called as they go, with the tokens read so far.
    const float* prefill(const int* ids, int n, void (*progress)(void*, int) = nullptr, void* ctx = nullptr);
    /// The position kept aside; restore() goes back to it.  Hy4 has no recurrent state: every layer's cache rows below a
    /// position are as they were (a row is only ever written at its position), and the Engram n-grams read the token
    /// history, so going back is the position alone.  false: nothing taken yet.
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
    };
    Stats stats() const;
    void clear_stats();
    /// The experts' use scores into Options::profile's file (false: not written).
    bool save_profile() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace hy4
