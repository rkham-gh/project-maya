// src/ling/ling_model.hpp - maya-ling's model: inclusionAI's Ling-3.1-flash (a 72-layer MoE: 63 Kimi Delta Attention
// layers and 9 gated MLA ones, 512 routed experts of which 8 a token, one shared) from a GGUF (llama.cpp's "bailingmoe3"
// architecture; bartowski/Ling-3.1-flash-1019-GGUF) on one NVIDIA GPU and the CPU: the attention, the dense layers,
// the shared experts and the output head in VRAM; the routed experts read where they lie in the GGUF's mapping by
// the CPU lane (ling_cpu.hpp), the most used of them in a VRAM cache.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ling {

/// The model's shape, from the GGUF's bailingmoe3.* metadata.
struct Geo {
    int n_layer = 0;   // the main layers (block_count less nextn_predict_layers: the MTP layer is not run)
    int n_embd = 0, n_head = 0, head_dim = 0, n_ff = 0;
    int n_expert = 0, n_used = 0, n_group = 1, n_group_used = 1, n_ff_exp = 0, n_ff_sh = 0, n_dense = 0;
    float exp_scale = 1.0f, eps = 1e-6f, kda_lb = -5.0f, rope_base = 10000.0f;
    int conv_k = 0, q_lora = 0, kv_lora = 0, rope_dim = 0, qk_head = 0, v_head = 0;
    int vocab = 0, ctx_train = 0, eos = -1;
    std::vector<uint8_t> mla;          // per layer: 1 a gated MLA layer, 0 a KDA layer
    std::vector<float> clamp_exp, clamp_sh;   // per layer: the SwiGLU clamps (0: none)
};

struct Options {
    std::string gguf;     // the first shard (...-00001-of-0000N.gguf); the others are found beside it
    int max_ctx = 131072;
    int threads = 0;      // CPU lane threads in all (0: the physical cores)
    int gpu = 0;
    int chunk = 8192;     // the prompt path's tokens a chunk (prefill): every expert a chunk routes to crosses PCIe once
    int sub = 2048;       // ... its attention layers' tokens at a time (the KDA / MLA buffers' size)
    int stage = 32;       // the prompt path copies the routed experts in groups of this many (two VRAM buffers in turn)
    double cache_gb = -1; // VRAM for the most-used routed experts (-1: what is free, less LING_VRAM_RESERVE_MIB)
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
    /// One token at pos(): the next token's logits (host memory, geo().vocab values), pos + 1.  nullptr when pos() is
    /// at max_ctx().
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
    };
    Stats stats() const;
    void clear_stats();
    /// The experts' use scores into Options::profile's file (false: not written).
    bool save_profile() const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace ling
