// src/ink/ink_model.cu - maya-ink's model (ink_model.hpp), branch inkling2: Inkling 2 Small (Thinking Machines, 594B
// MoE, 17B active) from the GGUF written by C:\llm\test-results\2026-11-12-inkling2\ink2_convert.py, following
// transformers' modeling_inkling2.py (huggingface/transformers#50682):
//
//   h = rmsnorm(embd[token], token_embd_norm); blocks = []
//   per layer i:  x = i == 0 ? h : attn_res(blocks, h, attn_res_q) (3 queries on the global layers: q / k / v inputs)
//                 q, r, gate, sinks from rmsnorm(x_q); k, v from rmsnorm(x_k), rmsnorm(x_v)
//                 k, v = (y + conv(y)) / sqrt(5) (sliding), dynamic taps (global: static + proj(x_k / x_v), / 3)
//                 q = l2clip(q * q_clip) (x tau on the global layers); k = l2clip(k); v = rmsnorm per head
//                 o = sigmoid(gate) * attention(q, K, V, rel band(l2clip(r * rel_clip)), sinks), scale 1 / sqrt(hd)
//                 a = (Wo o + conv(Wo o)) / sqrt(5)
//                 at i % 6 == 0: blocks += [h], res = a; else res = h + a
//                 f = rmsnorm(attn_res(blocks, res, ffn_res_q), ffn_norm); the top 8 of sigmoid(logits) + bias,
//                 weights sigmoid / their sum x 8; experts softcap(g, 4) sigmoid(g) softcap(u, 25) before down
//                 h = res + (moe + conv(moe)) / sqrt(5)
//   logits = softcap(output(rmsnorm(attn_res(blocks, h, output_res_q), output_norm)), 30), the padded rows -inf
//
// (Inkling-Small, the engine this branch came from, followed llama.cpp's graph (src/models/inkling.cpp in
// ggml-org/llama.cpp#25731) step by step:)
//
//   x = rmsnorm(embd[token], token_embd_norm)
//   per layer:  a = rmsnorm(x, attn_norm);  q, k, v, r = Wq a, Wk a, Wv a, Wr a
//               k = sconv_k(k);  v = sconv_v(v);  q, k = head rmsnorm (q times tau on the global layers)
//               rel = (attn_rel_proj^T r) per head (times tau on the global layers)
//               o = attention(q, K, V, rel) with scale 1 / head_dim;  x += sconv_attn(Wo o)
//               f = rmsnorm(x, ffn_norm);  x += sconv_mlp(ffn(f))
//   ffn: the first dense_block_count layers a SwiGLU FFN times ffn_gscale; the others the MoE - the top 6 of
//        sigmoid(logits) + exp_probs_b, weighted by softmax(logsigmoid([their raw logits, the 2 shared logits])) x
//        expert_weights_scale x ffn_gscale; the 2 shared experts' gammas (the last 2 weights) times their SwiGLU
//        before their down; the routed experts' sum, then the shared ones'
//   logits = output(rmsnorm(x, output_norm) / logit_scale_denom), the padded vocabulary rows -inf
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "ink_model.hpp"

#include "ink_cpu.hpp"
#include "ink_kernels.cuh"

#include "../ds4/ds4_iq.cuh"
#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace ink {

namespace iq = strata::ds4::gguf::iq;

namespace {

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "maya-ink: CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,             \
                         cudaGetErrorString(e_));                                                               \
            std::fflush(stderr);                                                                                \
            std::abort();                                                                                       \
        }                                                                                                       \
    } while (0)

bool is_float(int t) { return t == 0 || t == 1 || t == 30; }

size_t row_bytes_of(int type, int64_t k) {
    int e = 0, b = 0;
    if (!strata::block_geometry((uint32_t) type, e, b) || e <= 0 || k % e != 0) return 0;
    return (size_t) (k / e) * (size_t) b;
}

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {   // subnormal
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            bits = s | ((uint32_t) (127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 127 - 15) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

struct DMat {   // n rows of k values in VRAM, as the GGUF holds them
    const uint8_t* w = nullptr;
    int type = 0;
    size_t rb = 0;
    int n = 0, k = 0;
};

struct Tensor {
    const strata::TensorInfo* info = nullptr;
    const uint8_t* data = nullptr;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// (the copy timings) this process's page faults so far and its working set
unsigned long long page_faults() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.PageFaultCount;
#endif
    return 0;
}
double working_set_gb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return pmc.WorkingSetSize / 1e9;
#endif
    return 0;
}

}  // namespace

struct Layer {
    bool swa = false, dense = false;
    int E = 0;
    float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *rel_proj = nullptr;
    float *sc_k = nullptr, *sc_v = nullptr, *sc_attn = nullptr, *sc_mlp = nullptr;
    float *ffn_norm = nullptr, *gscale_d = nullptr;
    float gscale = 1.0f;
    std::vector<float> exp_bias;   // (host: a token's routing is done there)
    float* exp_bias_d = nullptr;   // (the prompt path routes on the GPU)
    DMat wq, wk, wv, wr, wo;
    DMat gate, up, down;           // dense layers
    DMat gate_inp;                 // MoE layers: n_expert + n_shexp rows
    DMat sh_gate[2], sh_up[2], sh_down[2];
    // Inkling 2
    float* in_norm[3] = {nullptr, nullptr, nullptr};    // the global layers' q / k / v input norms
    float *sc_k_stat = nullptr, *sc_v_stat = nullptr;   // their dynamic convs' static taps [K][kvw]
    float *res_q = nullptr, *ffn_q = nullptr;           // the attention residuals' queries ([nq][E], [E])
    int nq = 1;
    float *q_clip = nullptr, *rel_clip = nullptr;
    DMat wgate, wsink, kproj, vproj;
    CpuLayer cl;
    __half *K = nullptr, *V = nullptr;
    int kv_rows = 0;
    float* conv = nullptr;   // [k | v | attn | mlp] states, (conv_k - 1) values a channel
};

struct Engine::Impl {
    Options o;
    Geo g;
    int max_ctx = 0, pos = 0, kvw = 0;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Tensor> tens;
    cudaStream_t s = nullptr;
    std::vector<Layer> L;
    DMat embd, output;
    // The embedding's rows stay in the GGUF's mapping (a token needs one): its row copied up through h_erow, a prompt
    // chunk's gathered into the pinned staging and copied to pf.Erows.  INK_EMBD_VRAM=1: the table in VRAM as before.
    bool embd_host = true;
    uint8_t *h_erow = nullptr, *d_erow = nullptr;
    float *tok_norm = nullptr, *out_norm = nullptr;
    std::unique_ptr<Pool> pool;
    CpuLane lane;   // the token path's CPU experts, a layer at a time
    size_t dev_bytes = 0;

    // device scratch
    float *x = nullptr, *a = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr, *r = nullptr, *rel = nullptr;
    float *o_ = nullptr, *ao = nullptr, *f = nullptr, *fg = nullptr, *fu = nullptr, *fo = nullptr;
    float *sg = nullptr, *su = nullptr, *so = nullptr, *logits = nullptr, *rlog = nullptr;
    float *tau = nullptr, *gamma = nullptr, *attn_scr = nullptr;
    float* tokd = nullptr;   // the token's tau (tau points here), then its position (dpos) - the graphs read them
    int* dpos = nullptr;
    float* d_tail = nullptr;   // a MoE layer's end: which lanes ran (2 ints, 2 spare), the CPU lane's sum (n_embd)
    int8_t* xq = nullptr;
    float* xd = nullptr;
    uint16_t* e16 = nullptr;
    // Inkling 2's token path: the token's finished blocks (RMS-normed, [blocks][E]), the layer's residual, the
    // attention residuals' outputs [3][E], the normed inputs [3][E], the gate [H hd], the sinks [H], the dynamic taps of
    // k and v [Hkv K], k and v after a global layer's convs, the output's residual query
    float *blk = nullptr, *res = nullptr, *mix = nullptr, *a3 = nullptr, *gt = nullptr, *snk = nullptr, *dyk = nullptr,
          *dyv = nullptr, *k2 = nullptr, *v2 = nullptr, *out_q = nullptr, *ars = nullptr;
    int n_blocks() const { return (g.n_layer + g.res_block - 1) / g.res_block; }
    // host (pinned)
    float *h_f = nullptr, *h_rlog = nullptr, *h_tail = nullptr, *h_logits = nullptr, *h_tokd = nullptr,
          *h_gamma = nullptr;
    Stats st;
    // (INK_STEP_TIMES) the token path's host time by phase, printed every 64 tokens
    struct PhaseTimes {
        double att = 0, route = 0, glaunch = 0, cpu = 0, tail = 0, head = 0, adapt = 0, embed = 0, dense = 0;
        long cpu_experts = 0, tokens = 0;
        double gpu_att = 0;   // (a MoE layer's attention segment on the GPU, layer 10's, by events)
        cudaEvent_t e0 = nullptr, e1 = nullptr;
    } pt;
    // (INK_ROUTE_LOG=FILE) the routing as it goes, for replaying the expert cache's choices offline: 'T' + a token's
    // n_used experts a MoE layer, in rank order (uint16: Inkling 2 has 384; Inkling-Small's log had a byte each); 'P' +
    // layer + a prompt chunk's counts an expert (uint16); 'E' a prompt's end (the cache refilled there)
    FILE* route_log = nullptr;
    std::vector<uint16_t> route_tok;
    bool pf_open = false;   // a prompt read through the prompt path, no token written since (its buffers not lent)
    // (Engine::snapshot, INK_SNAPSHOT=1) every layer's convolution states and the sliding layers' K / V rings at
    // snap_pos, one buffer in VRAM (-1: none taken)
    uint8_t* snap = nullptr;
    size_t snap_bytes = 0;
    int snap_pos = -1;
    size_t conv_floats() const { return (size_t) (g.conv_k - 1) * (2 * (size_t) kvw + 2 * (size_t) g.n_embd); }

    template <class T> T* dalloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, n * sizeof(T)));
        CK(cudaMemset(p, 0, n * sizeof(T)));
        dev_bytes += n * sizeof(T);
        return (T*) p;
    }
    template <class T> T* halloc(size_t n) {
        void* p = nullptr;
        CK(cudaMallocHost(&p, n * sizeof(T)));
        std::memset(p, 0, n * sizeof(T));
        return (T*) p;
    }

    const Tensor* find(const std::string& name) const {
        auto it = tens.find(name);
        return it == tens.end() ? nullptr : &it->second;
    }

    bool mat(const std::string& name, int k, int n, DMat& m, std::string& err) {
        const Tensor* t = find(name);
        if (!t) {
            err = "the GGUF has no " + name;
            return false;
        }
        const auto& sh = t->info->shape;
        if (sh.size() < 2 || (int64_t) sh[0] != k || (int64_t) sh[1] != n) {
            err = name + ": shape not [" + std::to_string(k) + ", " + std::to_string(n) + "]";
            return false;
        }
        const int type = (int) t->info->type;
        if (!is_float(type) && !iq::mv_supported(type)) {
            err = name + ": type " + std::string(t->info->type_name()) + " has no kernel here";
            return false;
        }
        m.type = type;
        m.k = k;
        m.n = n;
        m.rb = row_bytes_of(type, k);
        const size_t bytes = m.rb * (size_t) n;
        uint8_t* d = dalloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, t->data, bytes, cudaMemcpyHostToDevice));
        m.w = d;
        return true;
    }
    // a bank of `count` matrices [k][n] (a 3D tensor [k, n, count]): one upload, `count` views
    bool bank(const std::string& name, int k, int n, int count, DMat* out, std::string& err) {
        const Tensor* t = find(name);
        if (!t) {
            err = "the GGUF has no " + name;
            return false;
        }
        const auto& sh = t->info->shape;
        if (sh.size() < 3 || (int64_t) sh[0] != k || (int64_t) sh[1] != n || (int64_t) sh[2] != count) {
            err = name + ": shape not [" + std::to_string(k) + ", " + std::to_string(n) + ", " + std::to_string(count) +
                  "]";
            return false;
        }
        const int type = (int) t->info->type;
        if (!is_float(type) && !iq::mv_supported(type)) {
            err = name + ": type " + std::string(t->info->type_name()) + " has no kernel here";
            return false;
        }
        const size_t rb = row_bytes_of(type, k), bytes = rb * (size_t) n * count;
        uint8_t* d = dalloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, t->data, bytes, cudaMemcpyHostToDevice));
        for (int i = 0; i < count; ++i) {
            out[i].w = d + (size_t) i * n * rb;
            out[i].type = type;
            out[i].rb = rb;
            out[i].n = n;
            out[i].k = k;
        }
        return true;
    }
    // a small tensor of `count` values as F32 (host copy in hv when asked)
    float* f32(const std::string& name, size_t count, std::string& err, std::vector<float>* hv = nullptr) {
        const Tensor* t = find(name);
        if (!t) {
            err = "the GGUF has no " + name;
            return nullptr;
        }
        if (t->info->elements() != count) {
            err = name + ": " + std::to_string(t->info->elements()) + " values, not " + std::to_string(count);
            return nullptr;
        }
        std::vector<float> v(count);
        const int type = (int) t->info->type;
        if (type == 0) std::memcpy(v.data(), t->data, count * 4);
        else if (type == 1)
            for (size_t i = 0; i < count; ++i) v[i] = half_to_float(((const uint16_t*) t->data)[i]);
        else if (type == 30)
            for (size_t i = 0; i < count; ++i) {
                const uint32_t b = (uint32_t) ((const uint16_t*) t->data)[i] << 16;
                std::memcpy(&v[i], &b, 4);
            }
        else {
            err = name + ": type " + std::string(t->info->type_name()) + " where F32 / F16 / BF16 was expected";
            return nullptr;
        }
        float* d = dalloc<float>(count);
        CK(cudaMemcpy(d, v.data(), count * 4, cudaMemcpyHostToDevice));
        if (hv) *hv = std::move(v);
        return d;
    }

    // ---------------------------------------------------------------- the GPU's products
    void quant(const float* xin, int rows, int k) { k::q8(xin, rows, k, xq, xd, s); }
    // y = alpha (W x): x quantized by quant() (rows of k; row `row`), or read as F32 for a float matrix
    void mv(const DMat& m, const float* xin, int row, float* y, float alpha = 1.0f) {
        if (is_float(m.type)) {
            k::gemv_f(m.w, m.type, m.rb, xin + (size_t) row * m.k, y, m.n, m.k, s);
            return;
        }
        iq::MvJob j;
        j.w = m.w;
        j.ld = m.rb;
        j.type = m.type;
        j.k = m.k;
        j.n = m.n;
        j.ncol = 1;
        j.xq = xq + (size_t) row * m.k;
        j.xd = xd + (size_t) row * (m.k / 32);
        j.y = y;
        j.ycs = (size_t) m.n;
        j.alpha = alpha;
        if (!iq::mv(&j, 1, s)) {
            std::fprintf(stderr, "maya-ink: no GEMV for type %d (k %d)\n", m.type, m.k);
            std::abort();
        }
    }
    // several y[i] = W[i] x(rows[i]) in one launch (x quantized by quant(); a float matrix among them: one by one)
    void mv_jobs(const DMat* const* m, const int* rows, float* const* y, int n, const float* xin) {
        bool all_q = n <= 16;
        for (int i = 0; i < n && all_q; ++i) all_q = !is_float(m[i]->type);
        if (!all_q) {
            for (int i = 0; i < n; ++i) mv(*m[i], xin, rows[i], y[i]);
            return;
        }
        iq::MvJob j[16];
        for (int i = 0; i < n; ++i) {
            j[i] = iq::MvJob{};
            j[i].w = m[i]->w;
            j[i].ld = m[i]->rb;
            j[i].type = m[i]->type;
            j[i].k = m[i]->k;
            j[i].n = m[i]->n;
            j[i].ncol = 1;
            j[i].xq = xq + (size_t) rows[i] * m[i]->k;
            j[i].xd = xd + (size_t) rows[i] * (m[i]->k / 32);
            j[i].y = y[i];
            j[i].ycs = (size_t) m[i]->n;
        }
        if (!iq::mv(j, n, s)) {
            std::fprintf(stderr, "maya-ink: no GEMV for these types\n");
            std::abort();
        }
    }

    bool load(const Options& opt, std::string& err);
    const float* step(int token);
    void tok_attention(Layer& l, const Layer* prev);
    void tok_router(const Layer& l);
    void tok_tail(const Layer& l);
    void tok_head(const Layer& last);
    int fuse = 15;   // the token path's fused kernels (tok_attention's bits)
    void tok_graphs();
    size_t tail_bytes() const { return ((size_t) 4 + g.n_embd) * sizeof(float); }
    std::vector<cudaGraphExec_t> gx;   // a MoE layer's graph (tok_graphs)
    cudaGraphExec_t gx_head = nullptr;
    int graphs = -1;   // -1: not made yet, 0: none, 1: made

    // ---------------------------------------------------------------- the prompt path (prefill)
    // A chunk of up to C tokens a layer at a time on the GPU: the dense matrices decoded to F16 and multiplied by
    // cuBLAS (F32 sums), the attention as tiles of scores (cuBLAS) and a softmax with the band bias, the router in F32;
    // each routed expert some token of the chunk chose copied to a VRAM slot (the copy stream, `stage` slots ahead),
    // decoded to F16 and multiplied for its tokens, its output added to theirs times their weights - expert by expert
    // in id order, then the shared experts: the same sums every run.
    struct PF {
        int C = 0, GE = 0;
        long group_ctr = 0;
        size_t s_elems = 0, slot_bytes = 0, w1_elems = 0, w2_elems = 0;
        uint8_t* arena = nullptr;   // the buffers below: one allocation, lent to the expert cache between prompts
        size_t arena_bytes = 0;
        uint8_t* gbuf[2] = {nullptr, nullptr};
        cudaEvent_t gready[2] = {nullptr, nullptr}, gfreed[2] = {nullptr, nullptr};
        cublasHandle_t cb = nullptr;
        cudaStream_t cs = nullptr;   // the experts' copies
        int *ids = nullptr, *sel = nullptr, *cnt = nullptr, *off = nullptr, *list = nullptr;
        float *X = nullptr, *A = nullptr, *Q = nullptr, *Kf = nullptr, *Vf = nullptr, *Kc = nullptr, *Vc = nullptr;
        float *R = nullptr, *Y = nullptr, *Ye = nullptr, *GU = nullptr, *RL = nullptr, *wts = nullptr, *wl = nullptr;
        float *S = nullptr, *Ot = nullptr, *tau = nullptr;
        float *RES = nullptr, *GATE = nullptr, *SINK = nullptr, *DYN = nullptr;   // (Inkling 2)
        __half* BLK = nullptr;   // a chunk's finished blocks, RMS-normed: [blocks][C][E]
        __half *Ah = nullptr, *Qh = nullptr, *AOh = nullptr, *Kw = nullptr, *Vw = nullptr, *Xe = nullptr, *He = nullptr;
        __half *P = nullptr, *W1 = nullptr, *W2 = nullptr;
        int *h_ids = nullptr, *h_cnt = nullptr, *h_off = nullptr;
        uint8_t* Erows = nullptr;
        float* h_tau = nullptr;
        double copy_bytes = 0, copy_ms = 0, moe_ms = 0, attn_ms = 0;
    } pf;
    // The experts' copies to the GPU: the CPU lane's threads copy into two pinned buffers in turn and the DMA takes each
    // from there (the driver's own staging of pageable memory moved 4-15 GB/s here, one thread; this holds ~14, which
    // is what the RAM gives three passes over the bytes).  INK_PF_PAGEABLE=1: the driver's staging.
    struct Stage {
        uint8_t* pin[2] = {nullptr, nullptr};
        cudaEvent_t ev[2] = {nullptr, nullptr};
        size_t cap = 0;
        int k = 0;
        bool on = false;
    } stg;
    void stage_copy(uint8_t* dst, const uint8_t* src, size_t len);
    bool pf_alloc(std::string& err);
    void f16w(const DMat& m, __half* dst);
    void gemm(const __half* X, int ldx, const __half* W, int ldw, float* Y, int ldy, int T, int n, int k,
              float alpha = 1.0f, float beta = 0.0f);
    void lin(const DMat& m, const __half* x, int T, float* y) {
        f16w(m, pf.W1);
        gemm(x, m.k, pf.W1, m.k, y, m.n, T, m.n, m.k);
    }
    // y [T][n] = x [T][k] . W^T for an F32 matrix (the sinks' and the dynamic taps' projections)
    void lin32(const DMat& m, const float* x, int T, float* y) {
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, m.n, T, m.k, &one, (const float*) m.w, m.k, x, m.k, &zero, y,
                        m.n) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "maya-ink: cuBLAS F32 product failed\n");
            std::abort();
        }
    }
    void pf_attention(const Layer& l, int T, int pos0, const __half* Kb, const __half* Vb, int kp0, int nkeys,
                      float* out);
    void pf_dense(const Layer& l, int T);
    void pf_moe(const Layer& l, int T);
    void chunk(const int* ids, int T, bool want_logits);
    const float* prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx);

    // ---------------------------------------------------------------- the VRAM expert cache
    // Slots of VRAM, each one routed expert's gate | up | down as the GGUF holds them, for the experts used most
    // (score: each time a token or a prompt's token chose it, decaying as tokens are written): a token's experts that
    // are in a slot are multiplied on the GPU while the CPU lane takes the others; a prompt's are not copied again.
    // Filled after each prompt (the best-scored in) and adapted as tokens are written (up to `per` swaps every `every`
    // tokens, each a better score than its slot's expert by `margin` x + 0.5); the scores decay by `decay` a token, so
    // they follow what the conversation uses now (the measured best, replaying ab.py's routing: ec-sim.py in
    // test-results\2026-11-04-inkling); the scores can be kept in a file (Options::profile) for the next start.
    // A prompt chunk adds each expert's count of its tokens as it is - thousands after a long prompt, where a written
    // token adds 1 and an expert every token chose settles at 99: the cache then holds what the prompt used most and no
    // swap clears the bar until those scores have decayed, some hundred tokens on.  INK_CACHE_PROMPT=1 (off by default,
    // not run yet) puts a chunk on the written tokens' scale (pf_score); 2 also decays what was there, as its tokens
    // written one by one would.
    // A swap's copy is staged in pieces (INK_CACHE_STAGE_KB, 2 MB) while the CPU waits for a layer's attention on the
    // GPU, when its RAM is idle, and the expert is used from the token after the one its last piece went up in (by
    // then long arrived; the same tokens every run).  INK_CACHE_DEFER=0: copied at the swap.
    static constexpr int kKvSlack = 4096;   // (K / V rows reserved past the position: 8 MB a global layer's K or V)
    // A slot is sized for its class: the layers whose experts take the same bytes (UD-Q4_K_XL: 38 layers' 15.2 MB, one
    // layer's 16.3 MB, one's 18.4 MB), each class's share of the slots its share of the layers.  The slots past `own`
    // lie in the prompt path's buffers (pf.arena, the class of the most layers): the cache's while no prompt is read,
    // emptied before one is (ec_reclaim) and filled again after it.  Past those, slots in the global layers' K / V
    // caches' rows beyond the position (each cache's end cut into slots): usable while no row of theirs is reserved
    // (kv_hi: the rows below it are, the position plus kKvSlack), emptied as the position reaches them (ec_kv_reserve),
    // usable again after a reset.  INK_CACHE_POOLS=0: one class, the largest bytes; INK_LEND=0 / INK_LEND_KV=0:
    // nothing lent from the prompt path's buffers / the K / V caches.
    struct EC {
        int slots = 0, ne = 0, own = 0, lend_cls = 0, arena_end = 0;   // [own, arena_end): the prompt path's
        bool lent = false;                   // the prompt path's slots usable
        std::vector<int> kv_row;             // slot -> its first K / V row (a slot in a K / V cache), else -1
        int kv_hi = 0;                       // the K / V rows below it reserved (not usable as slots)
        std::vector<uint8_t*> ptr;           // slot -> its VRAM
        std::vector<int> cls;                // slot -> its class
        std::vector<int> layer_cls;          // layer -> its experts' class (-1: a dense layer)
        std::vector<size_t> cls_bytes;       // class -> bytes a slot
        std::vector<int> owner, where;   // slot -> key (layer x n_expert + expert) or -1; key -> slot, -1, or
                                         // -2: being copied in (its slot's owner already)
        std::vector<float> score;
        long lookups = 0, hits = 0, copies = 0, tok = 0;
        float decay = 0.99f, margin = 1.5f;
        int every = 2, per = 16;
        int prompt = 0;   // how a prompt chunk's choices count (INK_CACHE_PROMPT; pf_score)
        float *gu = nullptr, *h = nullptr, *y = nullptr, *routed = nullptr, *hd = nullptr;
        int8_t* hq = nullptr;
        cudaEvent_t copied = nullptr;
        // the swaps' copies being staged (key, slot, bytes so far, bytes), and those all up, used after the token's end
        struct Pending {
            int key = 0, slot = 0;
            size_t done = 0, bytes = 0;
        };
        std::vector<Pending> pend;
        size_t pend_head = 0;
        std::vector<std::pair<int, int>> arrived, arrived_prev;   // (this token's, the last token's)
        cudaEvent_t arrived_ev = nullptr;                          // (after the last token's pieces)
        bool defer = true;
        size_t piece = (size_t) 2 << 20;
        uint8_t* apin[2] = {nullptr, nullptr};
        cudaEvent_t aev[2] = {nullptr, nullptr};
        int ak = 0;
        size_t stage_n = 0;   // (the piece the workers copy now, and where its bytes lie)
        std::vector<std::pair<const uint8_t*, size_t>> stage_parts;
        std::function<void(int)> stage_fn;
    } ec;
    void ec_alloc();
    void ec_copy(int key, int slot);
    void ec_copy_many(const std::vector<std::pair<int, int>>& todo);
    void ec_fill();
    void ec_adapt();
    void ec_reclaim();
    void ec_flush();
    bool ec_stage_begin(size_t budget);
    void ec_stage_end();
    void ec_arrive();
    bool ec_usable(int sl) const {
        if (sl < ec.own) return true;
        if (sl < ec.arena_end) return ec.lent;
        return ec.kv_row[(size_t) sl] >= ec.kv_hi;
    }
    void ec_kv_reserve(int rows);
    void pf_score(int il, const int* cnt, int T);
    // an expert's bytes [off, off + n) as they lie in its slot (gate | up | down), from the GGUF's mapping
    void ec_src_parts(int key, size_t off, size_t n, std::vector<std::pair<const uint8_t*, size_t>>& out) const;
    const uint8_t* ec_blob(int il, int e) const {
        const int sl = ec.slots ? ec.where[(size_t) il * ec.ne + e] : -1;
        return sl >= 0 ? ec.ptr[(size_t) sl] : nullptr;
    }
    void gpu_experts(const Layer& l, const int* slot, const float* w, int n);

    // ---------------------------------------------------------------- the helper card's expert cache
    // A second card (Options::helper_gpu) that holds routed experts only (maya-nem's, branch nemotron-2gpu): slots sized
    // by class as the main cache's, filled with the best-scored experts the main card does not hold.  A token's experts
    // found there get f from the host (h_f, there after the layer's sync), are multiplied on the helper's own stream
    // while the main card and the CPU lane do theirs, and their sum comes back to the host, where it is added to the CPU
    // lane's.  The main cache decides first and leaves the helper's experts alone; the helper follows it: every `every`
    // tokens up to `per` experts change places, and after a prompt up to an eighth of its tokens (64 .. fill_cap).
    // The swaps between tokens are copied by a thread of their own on a second stream while the next tokens are
    // written (a 15 MB expert from the mapping takes ~0.8 ms of the thread that copies it), and an expert is used from
    // the first update after its copy is done; the fills at the start and after a prompt are copied at once, through
    // pinned staging the CPU lane's threads fill (hc_copy_many: as the main cache's fills; INK_HELPER_STAGE=0: the
    // driver's staging of the mapping, which moved ~5 GB/s - 4.6 s for the helper's 23 GB after a long prompt).
    // Nothing of the model's state lives there, so the prompt path and reset() ignore it.
    struct HC {
        int dev = -1, slots = 0;
        cudaStream_t s = nullptr, cs = nullptr;   // the experts' launches; the swaps' copies (the worker's)
        std::vector<uint8_t*> ptr;
        std::vector<int> cls, owner, where;   // slot -> class; slot -> key or -1; key -> slot, -1, or -2: on its way
        std::vector<uint8_t> busy;            // slot -> 1 while its copy is on its way
        float *f = nullptr, *qd = nullptr, *gu = nullptr, *h = nullptr, *hd = nullptr, *y = nullptr, *routed = nullptr;
        int8_t *qq = nullptr, *hq = nullptr;
        float* h_out = nullptr;               // (pinned) the helper's sum
        // the fills' staging (hc_copy_many): two pinned buffers in turn, each one's last DMA (an event of the helper's)
        uint8_t* pin[2] = {nullptr, nullptr};
        cudaEvent_t pev[2] = {nullptr, nullptr};
        size_t pcap = 0;
        int pk = 0;
        bool stage = false;
        long hits = 0, copies = 0, tok = 0;
        int every = 2, per = 4, fill_cap = 4096;   // (see docs/INKLING.md, "Two cards")
        float margin = -1.0f, bias = 2.0f;         // a swap's bar: score > margin * the slot's + bias (-1: the main cache's)
        size_t bytes = 0;
        // the worker: (key, slot) jobs from q, the copied ones to done (both under mu); inflight is the main thread's
        // count of those it has not collected yet
        bool bg = true, quit = false;
        std::thread th;
        std::mutex mu;
        std::condition_variable cv, cv_done;
        std::vector<std::pair<int, int>> q, done;
        size_t qhead = 0;
        int inflight = 0;
        // (INK_HELPER_KEEPALIVE) a one-warp kernel kept running on the helper while tokens are written (each lasts
        // ka_ns; step() launches the next once it has ended), so its driver sees it busy and keeps its clocks up
        cudaStream_t ks = nullptr;
        unsigned long long ka_ns = 0;
        ~HC() {
            if (!th.joinable()) return;
            {
                std::lock_guard<std::mutex> lk(mu);
                quit = true;
            }
            cv.notify_all();
            th.join();
        }
    } hc;
    bool hc_steal = false;
    long pf_tokens_open = 0;   // the tokens read through the prompt path since the last written token
    void hc_alloc();
    void hc_update(int max_copies, bool hysteresis, bool bg);
    void hc_copy(int key, int slot, cudaStream_t st);
    void hc_copy_many(const std::vector<std::pair<int, int>>& todo);
    void hc_collect(bool wait);
    void hc_worker();
    void hc_experts(const Layer& l, const int* slot, const float* w, int n);

    bool save_profile(const std::string& path) const;
    bool load_profile(const std::string& path);
};

Engine::Engine() : p(new Impl) {}
Engine::~Engine() { delete p; }
const Geo& Engine::geo() const { return p->g; }
int Engine::max_ctx() const { return p->max_ctx; }
int Engine::pos() const { return p->pos; }
Engine::Stats Engine::stats() const {
    Stats r = p->st;
    r.lookups = p->ec.lookups;
    r.hits = p->ec.hits;
    r.cache_slots = p->ec.slots;
    r.cache_copies = p->ec.copies;
    r.hits_helper = p->hc.hits;
    r.helper_slots = p->hc.slots;
    r.helper_copies = p->hc.copies;
    return r;
}
void Engine::clear_stats() {
    p->st = Stats{};
    p->ec.lookups = p->ec.hits = 0;
    p->hc.hits = 0;
}
bool Engine::save_profile() const { return !p->o.profile.empty() && p->save_profile(p->o.profile); }
bool Engine::load(const Options& o, std::string& err) { return p->load(o, err); }
const float* Engine::step(int token) { return p->step(token); }
const float* Engine::prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx) {
    return p->prefill(ids, n, progress, ctx);
}

// Each layer's part of the snapshot: its convolution states, then (a sliding layer) its K and V rings.  The rings are
// kept whole: a ring's row is written again n_swa tokens later, so the tokens after the snapshot overwrite the rows the
// position then still looked back at.  A global layer's rows are written once each, at their position: the rows below
// the snapshot's stay, and those from it on are written again before they are read.
bool Engine::snapshot() {
    Impl& m = *p;
    if (!m.snap) return false;
    const size_t conv = m.conv_floats() * sizeof(float);
    uint8_t* at = m.snap;
    for (const Layer& l : m.L) {
        CK(cudaMemcpyAsync(at, l.conv, conv, cudaMemcpyDeviceToDevice, m.s));
        at += conv;
        if (!l.swa) continue;
        const size_t ring = (size_t) l.kv_rows * m.kvw * sizeof(__half);
        CK(cudaMemcpyAsync(at, l.K, ring, cudaMemcpyDeviceToDevice, m.s));
        CK(cudaMemcpyAsync(at + ring, l.V, ring, cudaMemcpyDeviceToDevice, m.s));
        at += 2 * ring;
    }
    CK(cudaStreamSynchronize(m.s));
    m.snap_pos = m.pos;
    return true;
}

bool Engine::restore() {
    Impl& m = *p;
    if (!m.snap || m.snap_pos < 0) return false;
    const size_t conv = m.conv_floats() * sizeof(float);
    const uint8_t* at = m.snap;
    for (Layer& l : m.L) {
        CK(cudaMemcpyAsync(l.conv, at, conv, cudaMemcpyDeviceToDevice, m.s));
        at += conv;
        if (!l.swa) continue;
        const size_t ring = (size_t) l.kv_rows * m.kvw * sizeof(__half);
        CK(cudaMemcpyAsync(l.K, at, ring, cudaMemcpyDeviceToDevice, m.s));
        CK(cudaMemcpyAsync(l.V, at + ring, ring, cudaMemcpyDeviceToDevice, m.s));
        at += 2 * ring;
    }
    CK(cudaStreamSynchronize(m.s));
    m.pos = m.snap_pos;   // (ec.kv_hi stays: the K / V rows reserved up to the position reached are no slots again)
    return true;
}

void Engine::reset() {
    p->pos = 0;
    p->snap_pos = -1;   // (a snapshot is of the conversation that ends here)
    p->ec.kv_hi = 0;    // (the K / V caches' slots usable again: empty, the next fill or swaps take them)
    const size_t conv = p->conv_floats();
    for (auto& l : p->L) CK(cudaMemsetAsync(l.conv, 0, conv * sizeof(float), p->s));
    CK(cudaStreamSynchronize(p->s));
}

namespace {
std::string shard_name(const std::string& first, int i, int n) {
    // ...-00001-of-00005.gguf -> ...-0000i-of-00005.gguf
    char want[32], now[32];
    std::snprintf(now, sizeof now, "-%05d-of-%05d.gguf", 1, n);
    std::snprintf(want, sizeof want, "-%05d-of-%05d.gguf", i, n);
    const size_t at = first.rfind(now);
    if (at == std::string::npos) return {};
    return first.substr(0, at) + want;
}
int meta_int(const strata::GgufFile& f, const std::string& k, int def = -1) {
    const strata::MetaValue* v = f.get(k);
    return v && v->is_num() ? (int) v->num() : def;
}
float meta_f(const strata::GgufFile& f, const std::string& k, float def = 0.0f) {
    const strata::MetaValue* v = f.get(k);
    return v && v->is_num() ? (float) v->num() : def;
}
}  // namespace

bool Engine::Impl::load(const Options& opt, std::string& err) {
    o = opt;
    const double t0 = now_ms();
    try {
        files.emplace_back(new strata::GgufFile(o.gguf));
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const strata::GgufFile& f0 = *files[0];
    const strata::MetaValue* arch = f0.get("general.architecture");
    if (!arch || arch->s != "inkling2") {
        err = "not an Inkling 2 GGUF (general.architecture " + (arch ? arch->s : std::string("missing")) + ")";
        return false;
    }
    const int nsplit = meta_int(f0, "split.count", 1);
    for (int i = 2; i <= nsplit; ++i) {
        const std::string nm = shard_name(o.gguf, i, nsplit);
        if (nm.empty()) {
            err = "the first shard's name does not end in -00001-of-" + std::to_string(nsplit) + ".gguf";
            return false;
        }
        try {
            files.emplace_back(new strata::GgufFile(nm));
        } catch (const std::exception& e) {
            err = std::string(e.what()) + " (" + nm + ")";
            return false;
        }
    }
    for (auto& fp : files)
        for (const auto& t : fp->tensors()) tens[t.name] = Tensor{&t, fp->tensor_data(t)};

    // the shape
    const std::string A = "inkling2.";
    g.n_layer = meta_int(f0, A + "block_count");
    g.n_embd = meta_int(f0, A + "embedding_length");
    g.n_ff = meta_int(f0, A + "feed_forward_length");
    g.n_head = meta_int(f0, A + "attention.head_count");
    if (const strata::MetaValue* v = f0.get(A + "attention.head_count_kv"))
        g.n_head_kv = v->type == strata::MetaType::ARRAY ? (int) v->items.at(0).num() : (int) v->num();
    g.head_dim = meta_int(f0, A + "attention.key_length");
    g.eps = meta_f(f0, A + "attention.layer_norm_rms_epsilon", 1e-6f);
    g.n_expert = meta_int(f0, A + "expert_count");
    g.n_used = meta_int(f0, A + "expert_used_count");
    g.n_ff_exp = meta_int(f0, A + "expert_feed_forward_length");
    g.n_shexp = meta_int(f0, A + "expert_shared_count");
    g.exp_scale = meta_f(f0, A + "expert_weights_scale", 1.0f);
    g.n_swa = meta_int(f0, A + "attention.sliding_window");
    g.d_rel = meta_int(f0, A + "d_rel");
    g.rel_ext = meta_int(f0, A + "rel_extent");
    g.rel_ext_swa = meta_int(f0, A + "rel_extent_swa");
    g.conv_k = meta_int(f0, A + "shortconv_kernel");
    g.n_dense = 0;
    g.n_shexp = std::max(0, g.n_shexp);
    g.lcap = meta_f(f0, A + "final_logit_softcapping", 30.0f);
    g.qkclip = meta_f(f0, A + "qk_l2_clip", 0.0f);
    g.rclip = meta_f(f0, A + "rel_l2_clip", 0.0f);
    g.gcap = meta_f(f0, A + "mlp_gate_softcap", 0.0f);
    g.ucap = meta_f(f0, A + "mlp_up_softcap", 0.0f);
    g.res_block = meta_int(f0, A + "attn_res_block", 6);
    g.logit_scale = 1.0f / meta_f(f0, A + "logit_scale_denom", 1.0f);
    g.log_floor = meta_int(f0, A + "log_scaling_n_floor", 0);
    g.log_alpha = meta_f(f0, A + "log_scaling_alpha", 0.0f);
    g.vocab = meta_int(f0, A + "vocab_size");
    g.vocab_unpadded = meta_int(f0, A + "unpadded_vocab_size", g.vocab);
    g.ctx_train = meta_int(f0, A + "context_length");
    g.swa.assign((size_t) g.n_layer, 0);
    if (const strata::MetaValue* v = f0.get(A + "attention.sliding_window_pattern"))
        for (int i = 0; i < g.n_layer && i < (int) v->items.size(); ++i) g.swa[(size_t) i] = v->items[(size_t) i].u ? 1 : 0;
    if (g.n_layer <= 0 || g.n_embd <= 0 || g.head_dim % 32 != 0 || g.head_dim > 256 || g.n_head % g.n_head_kv != 0 ||
        g.conv_k < 2 || g.conv_k > 8 || g.n_shexp != 0 || g.n_used > 16 || g.d_rel <= 0 || g.qkclip <= 0.0f ||
        g.gcap <= 0.0f || g.ucap <= 0.0f || g.res_block <= 0 || (g.n_layer + g.res_block - 1) / g.res_block > 8) {
        err = "an Inkling 2 shape this engine does not take";
        return false;
    }
    kvw = g.n_head_kv * g.head_dim;
    max_ctx = std::min(o.max_ctx, g.ctx_train > 0 ? g.ctx_train : o.max_ctx);

    CK(cudaSetDevice(o.gpu));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int E = g.n_embd, H = g.n_head, hd = g.head_dim, kc = g.conv_k;

    // the weights in VRAM
    if (const char* v = std::getenv("INK_EMBD_VRAM")) embd_host = std::atoi(v) == 0;
    if (embd_host) {
        const Tensor* t = find("token_embd.weight");
        if (!t || t->info->shape.size() < 2 || (int64_t) t->info->shape[0] != E ||
            (int64_t) t->info->shape[1] != g.vocab) {
            err = "the GGUF has no token_embd.weight of [" + std::to_string(E) + ", " + std::to_string(g.vocab) + "]";
            return false;
        }
        embd.type = (int) t->info->type;
        embd.k = E;
        embd.n = g.vocab;
        embd.rb = row_bytes_of(embd.type, E);
        embd.w = t->data;
        if (embd.rb == 0 || (!is_float(embd.type) && embd.type != 8)) {
            err = "token_embd.weight: type " + std::string(t->info->type_name()) + " has no row decoder here";
            return false;
        }
        h_erow = halloc<uint8_t>(embd.rb);
        d_erow = dalloc<uint8_t>(embd.rb);
    } else if (!mat("token_embd.weight", E, g.vocab, embd, err)) {
        return false;
    }
    if (!mat("output.weight", E, g.vocab, output, err)) return false;
    if (!(tok_norm = f32("token_embd_norm.weight", (size_t) E, err))) return false;
    if (!(out_norm = f32("output_norm.weight", (size_t) E, err))) return false;
    if (!(out_q = f32("output_res_q.weight", (size_t) E, err))) return false;
    L.resize((size_t) g.n_layer);
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        l.swa = g.swa[(size_t) il] != 0;
        l.dense = il < g.n_dense;
        l.E = l.swa ? g.rel_ext_swa : g.rel_ext;
        if (l.swa) {
            if (!(l.attn_norm = f32(b + "attn_norm.weight", (size_t) E, err))) return false;
            if (!(l.sc_k = f32(b + "shortconv_k.weight", (size_t) kc * kvw, err))) return false;
            if (!(l.sc_v = f32(b + "shortconv_v.weight", (size_t) kc * kvw, err))) return false;
        } else {
            static const char* nn[3] = {"attn_q_in_norm", "attn_k_in_norm", "attn_v_in_norm"};
            for (int i = 0; i < 3; ++i)
                if (!(l.in_norm[i] = f32(b + nn[i] + ".weight", (size_t) E, err))) return false;
            if (!(l.sc_k_stat = f32(b + "shortconv_k_static.weight", (size_t) kc * kvw, err))) return false;
            if (!(l.sc_v_stat = f32(b + "shortconv_v_static.weight", (size_t) kc * kvw, err))) return false;
            if (!mat(b + "shortconv_k_proj.weight", E, g.n_head_kv * kc, l.kproj, err) ||
                !mat(b + "shortconv_v_proj.weight", E, g.n_head_kv * kc, l.vproj, err))
                return false;
        }
        l.nq = l.swa ? 1 : 3;
        if (il > 0 && !(l.res_q = f32(b + "attn_res_q.weight", (size_t) E * l.nq, err))) return false;
        if (!(l.ffn_q = f32(b + "ffn_res_q.weight", (size_t) E, err))) return false;
        if (!(l.q_clip = f32(b + "attn_q_clip.weight", (size_t) H * hd, err))) return false;
        if (!(l.rel_clip = f32(b + "attn_rel_clip.weight", (size_t) H * g.d_rel, err))) return false;
        if (!(l.rel_proj = f32(b + "attn_rel_proj.weight", (size_t) g.d_rel * l.E, err))) return false;
        if (!(l.sc_attn = f32(b + "shortconv_attn.weight", (size_t) kc * E, err))) return false;
        if (!(l.sc_mlp = f32(b + "shortconv_mlp.weight", (size_t) kc * E, err))) return false;
        if (!(l.ffn_norm = f32(b + "ffn_norm.weight", (size_t) E, err))) return false;
        if (!mat(b + "attn_q.weight", E, H * hd, l.wq, err) || !mat(b + "attn_k.weight", E, kvw, l.wk, err) ||
            !mat(b + "attn_v.weight", E, kvw, l.wv, err) || !mat(b + "attn_r.weight", E, H * g.d_rel, l.wr, err) ||
            !mat(b + "attn_output.weight", H * hd, E, l.wo, err) || !mat(b + "attn_gate.weight", E, H * hd, l.wgate, err) ||
            !mat(b + "attn_sinks.weight", E, H, l.wsink, err))
            return false;
        if (l.dense) {
            if (!mat(b + "ffn_gate.weight", E, g.n_ff, l.gate, err) || !mat(b + "ffn_up.weight", E, g.n_ff, l.up, err) ||
                !mat(b + "ffn_down.weight", g.n_ff, E, l.down, err))
                return false;
        } else {
            const int ne = g.n_expert;
            if (!mat(b + "ffn_gate_inp.weight", E, ne + g.n_shexp, l.gate_inp, err)) return false;
            if (l.gate_inp.type != 0) {
                err = b + "ffn_gate_inp: not F32 (the prompt path's router multiplies F32 rows)";
                return false;
            }
            if (!(l.exp_bias_d = f32(b + "exp_probs_b.bias", (size_t) ne, err, &l.exp_bias))) return false;
            // the routed experts stay where they are, for the CPU lane
            const Tensor *tg = find(b + "ffn_gate_exps.weight"), *tu = find(b + "ffn_up_exps.weight"),
                         *td = find(b + "ffn_down_exps.weight");
            if (!tg || !tu || !td) {
                err = b + "ffn_*_exps: missing";
                return false;
            }
            if (!cpu_layer((int) tg->info->type, (int) tu->info->type, (int) td->info->type, E, g.n_ff_exp, l.cl, err)) {
                err = b + "ffn_*_exps: " + err;
                return false;
            }
            l.cl.gate = tg->data;
            l.cl.up = tu->data;
            l.cl.down = td->data;
            l.cl.gcap = g.gcap;
            l.cl.ucap = g.ucap;
        }
        // the caches: a full one on the global layers, a ring of n_swa rows on the sliding ones
        l.kv_rows = l.swa ? g.n_swa : max_ctx;
        l.K = dalloc<__half>((size_t) l.kv_rows * kvw);
        l.V = dalloc<__half>((size_t) l.kv_rows * kvw);
        l.conv = dalloc<float>((size_t) (kc - 1) * (2 * kvw + 2 * E));
        if (o.verbose && (il % 6 == 5 || il == g.n_layer - 1))
            std::fprintf(stderr, "maya-ink: layers 0..%d placed (%.1f GB of VRAM so far)\n", il, dev_bytes / 1e9);
    }

    // scratch
    x = dalloc<float>(E);
    a = dalloc<float>(E);
    q = dalloc<float>((size_t) H * hd);
    kk = dalloc<float>(kvw);
    vv = dalloc<float>(kvw);
    r = dalloc<float>((size_t) H * g.d_rel);
    rel = dalloc<float>((size_t) H * std::max(g.rel_ext, g.rel_ext_swa));
    o_ = dalloc<float>((size_t) H * hd);
    ao = dalloc<float>(E);
    f = dalloc<float>(E);
    fg = dalloc<float>((size_t) std::max(g.n_ff, 2 * g.n_ff_exp));
    fu = dalloc<float>((size_t) std::max(g.n_ff, 2 * g.n_ff_exp));
    fo = dalloc<float>(E);
    d_tail = dalloc<float>((size_t) 4 + E);
    sg = dalloc<float>((size_t) 2 * g.n_ff_exp);
    su = dalloc<float>((size_t) 2 * g.n_ff_exp);
    so = dalloc<float>((size_t) 2 * E);
    logits = dalloc<float>((size_t) g.vocab);
    rlog = dalloc<float>((size_t) g.n_expert + g.n_shexp);
    tokd = dalloc<float>(2);
    tau = tokd;
    dpos = (int*) (tokd + 1);
    gamma = dalloc<float>(2);
    attn_scr = dalloc<float>(
        k::attn_scratch_floats(H, hd, std::max(k::attn_nsp(0, max_ctx), k::attn_nsp(g.n_swa, g.n_swa))));
    const int kmax = std::max({3 * E, 4 * g.n_ff_exp, H * hd, 2 * g.n_ff});
    blk = dalloc<float>((size_t) n_blocks() * E);
    res = dalloc<float>(E);
    mix = dalloc<float>((size_t) 3 * E);
    a3 = dalloc<float>((size_t) 3 * E);
    gt = dalloc<float>((size_t) H * hd);
    snk = dalloc<float>(H);
    dyk = dalloc<float>((size_t) g.n_head_kv * kc);
    dyv = dalloc<float>((size_t) g.n_head_kv * kc);
    k2 = dalloc<float>(kvw);
    v2 = dalloc<float>(kvw);
    ars = dalloc<float>(k::attn_res_scratch);
    xq = dalloc<int8_t>((size_t) kmax);
    xd = dalloc<float>((size_t) kmax / 32);
    e16 = dalloc<uint16_t>(E);
    h_f = halloc<float>(E);
    h_rlog = halloc<float>((size_t) g.n_expert + g.n_shexp);
    h_tail = halloc<float>((size_t) 4 + E);
    h_logits = halloc<float>((size_t) g.vocab);
    h_tokd = halloc<float>(2);
    h_gamma = halloc<float>(2);
    // the room for Engine::snapshot (INK_SNAPSHOT=1; off by default, not run yet): every layer's convolution states,
    // the sliding layers' K / V rings (79 MB for Inkling-Small), taken before the expert cache sizes itself
    if (const char* sv = std::getenv("INK_SNAPSHOT"); sv && std::atoi(sv) != 0) {
        for (const Layer& l : L)
            snap_bytes += conv_floats() * sizeof(float) + (l.swa ? (size_t) 2 * l.kv_rows * kvw * sizeof(__half) : 0);
        snap = dalloc<uint8_t>(snap_bytes);
    }

    int threads = o.threads > 0 ? o.threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    pool.reset(new Pool(threads - 1));
    // Every routed expert's pages touched once, now: the first touch of a mapped page costs a fault (~1 us a 4 KB
    // page here even with the file in RAM), which held the prompt path's copies to 3.5 GB/s and the CPU lane's first
    // tokens back.  INK_PREFAULT=0: not done.
    const char* pv = std::getenv("INK_PREFAULT");
    if (!pv || std::atoi(pv) != 0) {
        const double tp = now_ms();
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        double gb = 0;
        for (const Layer& l : L) {
            if (l.dense) continue;
            const size_t gb_ = (size_t) g.n_ff_exp * l.cl.rg * g.n_expert, ub_ = (size_t) g.n_ff_exp * l.cl.ru * g.n_expert,
                         db_ = (size_t) E * l.cl.rd * g.n_expert;
            const std::pair<const uint8_t*, size_t> rs[3] = {{l.cl.gate, gb_}, {l.cl.up, ub_}, {l.cl.down, db_}};
            for (const auto& r0 : rs)   // (in 64 MB pieces: the pool's jobs)
                for (size_t off = 0; off < r0.second; off += (size_t) 64 << 20) {
                    parts.push_back({r0.first + off, std::min((size_t) 64 << 20, r0.second - off)});
                    gb += (double) parts.back().second / 1e9;
                }
        }
        std::atomic<unsigned> sink{0};
        pool->run((int) parts.size(), [&](int j) {
            const volatile uint8_t* p0 = parts[(size_t) j].first;
            unsigned acc = 0;
            for (size_t i = 0; i < parts[(size_t) j].second; i += 4096) acc += p0[i];
            sink.fetch_add(acc, std::memory_order_relaxed);
        });
        if (o.verbose)
            std::fprintf(stderr, "maya-ink: the routed experts' %.1f GB touched in %.1f s (working set %.1f GB)\n", gb,
                         (now_ms() - tp) / 1000.0, working_set_gb());
    }
    if (const char* rl = std::getenv("INK_ROUTE_LOG")) route_log = std::fopen(rl, "ab");
    fuse = 0;   // (Inkling 2: the fused token kernels are Inkling-Small's; the plain ones run)
    const size_t before_pf = dev_bytes;
    if (!pf_alloc(err)) return false;
    const size_t before_ec = dev_bytes;
    ec_alloc();
    if (!o.profile.empty() && load_profile(o.profile)) {
        const double tf = now_ms();
        ec_fill();
        CK(cudaStreamSynchronize(s));
        if (o.verbose)
            std::fprintf(stderr, "maya-ink: the expert profile %s: %ld experts into VRAM in %.1f s\n", o.profile.c_str(),
                         ec.copies, (now_ms() - tf) / 1000.0);
    }
    if (o.helper_gpu >= 0) {
        const double th = now_ms();
        hc_alloc();
        if (hc.slots) {
            hc_update(INT_MAX, false, false);
            if (o.verbose)
                std::fprintf(stderr,
                             "maya-ink: helper card CUDA%d: %d expert slots, %.2f GB of its VRAM; %ld experts into them in "
                             "%.1f s\n",
                             hc.dev, hc.slots, hc.bytes / 1e9, hc.copies, (now_ms() - th) / 1000.0);
        }
    }
    if (o.verbose) {
        std::string cls;
        for (size_t c = 0; c < ec.cls_bytes.size(); ++c) {
            int n_all = 0, n_l = 0;
            for (int sl = 0; sl < ec.slots; ++sl) n_all += ec.cls[(size_t) sl] == (int) c;
            for (int il = g.n_dense; il < g.n_layer; ++il) n_l += ec.layer_cls[(size_t) il] == (int) c;
            char b[128];
            std::snprintf(b, sizeof b, "%s%d of %.1f MB (%d layers)", c ? ", " : "", n_all, ec.cls_bytes[c] / 1e6, n_l);
            cls += b;
        }
        std::fprintf(stderr,
                     "maya-ink: Inkling %d layers, %d experts (top %d + %d shared), context %d; %.2f GB in VRAM (the "
                     "prompt path's chunks of %d tokens %.2f of it, %d expert slots %.2f, %d more in the prompt "
                     "path's buffers between prompts and %d in the K / V caches' rows past the position: %s), the "
                     "routed experts in the GGUF's mapping; CPU lane %d threads; loaded in %.1f s\n",
                     g.n_layer, g.n_expert, g.n_used, g.n_shexp, max_ctx, dev_bytes / 1e9, pf.C,
                     (before_ec - before_pf) / 1e9, ec.own, (dev_bytes - before_ec) / 1e9, ec.arena_end - ec.own,
                     ec.slots - ec.arena_end, cls.c_str(), pool->threads(), (now_ms() - t0) / 1000.0);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert cache
void Engine::Impl::ec_alloc() {
    const int E = g.n_embd, ff = g.n_ff_exp;
    ec.ne = g.n_expert;
    ec.where.assign((size_t) g.n_layer * ec.ne, -1);
    ec.score.assign((size_t) g.n_layer * ec.ne, 0.0f);
    ec.gu = dalloc<float>((size_t) 16 * 2 * ff);
    ec.h = dalloc<float>((size_t) 16 * ff);
    ec.hq = dalloc<int8_t>((size_t) 16 * ff);
    ec.hd = dalloc<float>((size_t) 16 * ff / 32);
    ec.y = dalloc<float>((size_t) 16 * E);
    ec.routed = dalloc<float>(E);
    CK(cudaEventCreateWithFlags(&ec.copied, cudaEventDisableTiming));
    if (const char* v = std::getenv("INK_CACHE_DECAY")) ec.decay = (float) std::atof(v);
    if (const char* v = std::getenv("INK_CACHE_EVERY")) ec.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("INK_CACHE_SWAPS")) ec.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("INK_CACHE_MARGIN")) ec.margin = (float) std::atof(v);
    if (const char* v = std::getenv("INK_CACHE_PROMPT")) ec.prompt = std::max(0, std::min(2, std::atoi(v)));
    if (const char* v = std::getenv("INK_CACHE_DEFER")) ec.defer = std::atoi(v) != 0;
    if (const char* v = std::getenv("INK_CACHE_STAGE_KB")) ec.piece = (size_t) std::max(64, std::atoi(v)) << 10;
    if (ec.defer) {
        for (int i = 0; i < 2 && ec.defer; ++i) {
            ec.defer = cudaHostAlloc((void**) &ec.apin[i], 4 * ec.piece, cudaHostAllocPortable) == cudaSuccess;
            if (ec.defer) CK(cudaEventCreateWithFlags(&ec.aev[i], cudaEventDisableTiming));
        }
        if (ec.defer) CK(cudaEventCreateWithFlags(&ec.arrived_ev, cudaEventDisableTiming));
        if (!ec.defer) cudaGetLastError();
    }
    const char* pv = std::getenv("INK_CACHE_POOLS");
    const bool pools = !pv || std::atoi(pv) != 0;
    const char* lv = std::getenv("INK_LEND");
    const bool lend = !lv || std::atoi(lv) != 0;
    // the classes: each MoE layer's expert bytes (gate | up | down as the GGUF holds them), a 256-byte multiple
    ec.layer_cls.assign((size_t) g.n_layer, -1);
    std::vector<size_t> eb((size_t) g.n_layer, 0);
    size_t eb_max = 0;
    for (int il = g.n_dense; il < g.n_layer; ++il) {
        const CpuLayer& c = L[(size_t) il].cl;
        eb[(size_t) il] = ((size_t) ff * (c.rg + c.ru) + (size_t) E * c.rd + 255) & ~(size_t) 255;
        eb_max = std::max(eb_max, eb[(size_t) il]);
    }
    for (int il = g.n_dense; il < g.n_layer; ++il) {
        const size_t b = pools ? eb[(size_t) il] : eb_max;
        auto it = std::find(ec.cls_bytes.begin(), ec.cls_bytes.end(), b);
        if (it == ec.cls_bytes.end()) {
            ec.cls_bytes.push_back(b);
            it = ec.cls_bytes.end() - 1;
        }
        ec.layer_cls[(size_t) il] = (int) (it - ec.cls_bytes.begin());
    }
    const int nc = (int) ec.cls_bytes.size(), nmoe = g.n_layer - g.n_dense;
    std::vector<int> nl((size_t) nc, 0);
    for (int il = g.n_dense; il < g.n_layer; ++il) ++nl[(size_t) ec.layer_cls[(size_t) il]];
    ec.lend_cls = (int) (std::max_element(nl.begin(), nl.end()) - nl.begin());
    // the cache's own VRAM: what is free, less the reserve, each class's slots its share of the layers
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t reserve = (size_t) 1536 << 20;
    if (const char* v = std::getenv("INK_VRAM_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
    size_t want = fr > reserve ? fr - reserve : 0;
    if (o.cache_gb >= 0.0) want = std::min(want, (size_t) (o.cache_gb * 1e9));
    double per_slot = 0;   // bytes a slot, on average over the classes' shares
    for (int c = 0; c < nc; ++c) per_slot += (double) nl[(size_t) c] / nmoe * (double) ec.cls_bytes[(size_t) c];
    auto counts = [&](size_t budget) {
        std::vector<int> n((size_t) nc, 0);
        const double N = (double) budget / per_slot;
        size_t used = 0;
        for (int c = 0; c < nc; ++c) {
            n[(size_t) c] = (int) (N * nl[(size_t) c] / nmoe);
            used += (size_t) n[(size_t) c] * ec.cls_bytes[(size_t) c];
        }
        if (budget > used) n[(size_t) ec.lend_cls] += (int) ((budget - used) / ec.cls_bytes[(size_t) ec.lend_cls]);
        return n;
    };
    std::vector<int> n = counts(want);
    uint8_t* base = nullptr;
    for (;;) {   // (one allocation; a smaller one if the free memory is in pieces)
        size_t bytes = 0;
        for (int c = 0; c < nc; ++c) bytes += (size_t) n[(size_t) c] * ec.cls_bytes[(size_t) c];
        if (bytes == 0) break;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) == cudaSuccess) {
            base = (uint8_t*) p;
            dev_bytes += bytes;
            break;
        }
        cudaGetLastError();
        want = want / 16 * 15;
        n = counts(want);
    }
    if (base) {
        uint8_t* at = base;
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < n[(size_t) c]; ++i) {
                ec.ptr.push_back(at);
                ec.cls.push_back(c);
                at += ec.cls_bytes[(size_t) c];
            }
    }
    ec.own = (int) ec.ptr.size();
    // the prompt path's buffers, lent between prompts: slots of the class of the most layers
    if (lend && base && pf.arena) {
        const size_t b = ec.cls_bytes[(size_t) ec.lend_cls];
        for (size_t at = 0; at + b <= pf.arena_bytes; at += b) {
            ec.ptr.push_back(pf.arena + at);
            ec.cls.push_back(ec.lend_cls);
        }
    }
    ec.arena_end = (int) ec.ptr.size();
    ec.lent = ec.arena_end > ec.own;
    ec.kv_row.assign(ec.ptr.size(), -1);
    const char* kv = std::getenv("INK_LEND_KV");
    if (base && (!kv || std::atoi(kv) != 0)) {
        const size_t b = ec.cls_bytes[(size_t) ec.lend_cls], row = (size_t) kvw * sizeof(__half);
        for (const Layer& l : L) {
            if (l.swa) continue;
            for (__half* buf : {l.K, l.V}) {
                const size_t bytes = (size_t) l.kv_rows * row;
                for (size_t end = bytes; end >= b && end - b >= (size_t) kKvSlack * row; end -= b) {
                    ec.ptr.push_back((uint8_t*) buf + (end - b));
                    ec.cls.push_back(ec.lend_cls);
                    ec.kv_row.push_back((int) ((end - b) / row));
                }
            }
        }
    }
    ec.slots = (int) ec.ptr.size();
    ec.kv_hi = 0;
    ec.owner.assign((size_t) ec.slots, -1);
}

// (the driver's staging; ec_copy_many's when there is no pinned staging)
void Engine::Impl::ec_copy(int key, int slot) {
    const int il = key / ec.ne, e = key % ec.ne, E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    uint8_t* d = ec.ptr[(size_t) slot];
    if (ec.owner[(size_t) slot] >= 0) ec.where[(size_t) ec.owner[(size_t) slot]] = -1;
    CK(cudaMemcpyAsync(d, c.gate + (size_t) e * gb, gb, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d + gb, c.up + (size_t) e * ub, ub, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d + gb + ub, c.down + (size_t) e * db, db, cudaMemcpyHostToDevice, s));
    ec.owner[(size_t) slot] = key;
    ec.where[(size_t) key] = slot;
    ++ec.copies;
}

// Experts (key, slot) into their slots: as many as fill a pinned staging buffer at a time, copied into it by the CPU
// lane's threads (1 MB pieces), then one DMA each on the copy stream from there; the compute stream waits for the last.
// (Called with the GPU done with the slots' old experts: after a prompt, or at the end of a token.)
void Engine::Impl::ec_copy_many(const std::vector<std::pair<int, int>>& todo) {
    if (todo.empty()) return;
    if (!stg.on) {
        for (const auto& kv : todo) ec_copy(kv.first, kv.second);
        return;
    }
    const int E = g.n_embd, ff = g.n_ff_exp;
    struct Part {
        const uint8_t* src;
        size_t off, len;
    };
    size_t i = 0;
    while (i < todo.size()) {
        // a batch: the experts that fit the buffer, laid out there as in their slots
        std::vector<Part> parts;
        std::vector<std::pair<size_t, size_t>> dma;   // (todo index, offset in the buffer)
        size_t fill = 0;
        for (; i < todo.size(); ++i) {
            const int key = todo[i].first, il = key / ec.ne, e = key % ec.ne;
            const CpuLayer& c = L[(size_t) il].cl;
            const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
            if (fill + gb + ub + db > stg.cap) {
                if (fill == 0) {   // (an expert larger than the buffer: the driver's staging)
                    ec_copy(key, todo[i].second);
                    continue;
                }
                break;
            }
            dma.push_back({i, fill});
            const std::pair<const uint8_t*, size_t> rs[3] = {
                {c.gate + (size_t) e * gb, gb}, {c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
            for (const auto& r0 : rs) {
                for (size_t o0 = 0; o0 < r0.second; o0 += (size_t) 1 << 20)
                    parts.push_back({r0.first + o0, fill + o0, std::min((size_t) 1 << 20, r0.second - o0)});
                fill += r0.second;
            }
        }
        CK(cudaEventSynchronize(stg.ev[stg.k]));   // (that buffer's last DMA is done)
        uint8_t* pin = stg.pin[stg.k];
        pool->run((int) parts.size(), [&](int j) {
            const Part& p0 = parts[(size_t) j];
            std::memcpy(pin + p0.off, p0.src, p0.len);
        });
        for (const auto& d : dma) {
            const int key = todo[d.first].first, sl = todo[d.first].second, il = key / ec.ne;
            const CpuLayer& c = L[(size_t) il].cl;
            const size_t bytes = (size_t) ff * (c.rg + c.ru) + (size_t) E * c.rd;
            CK(cudaMemcpyAsync(ec.ptr[(size_t) sl], pin + d.second, bytes, cudaMemcpyHostToDevice, pf.cs));
            if (ec.owner[(size_t) sl] >= 0) ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
            ec.owner[(size_t) sl] = key;
            ec.where[(size_t) key] = sl;
            ++ec.copies;
        }
        CK(cudaEventRecord(stg.ev[stg.k], pf.cs));
        stg.k ^= 1;
    }
    CK(cudaEventRecord(ec.copied, pf.cs));
    CK(cudaStreamWaitEvent(s, ec.copied, 0));
}

void Engine::Impl::ec_fill() {
    if (ec.slots == 0) return;
    ec_flush();
    std::vector<std::pair<int, int>> todo;
    std::vector<char> want(ec.where.size(), 0);
    for (int c = 0; c < (int) ec.cls_bytes.size(); ++c) {
        std::vector<int> keys;
        for (int il = g.n_dense; il < g.n_layer; ++il)
            if (ec.layer_cls[(size_t) il] == c)
                for (int e = 0; e < ec.ne; ++e)
                    if (ec.score[(size_t) il * ec.ne + e] > 0.0f) keys.push_back(il * ec.ne + e);
        int avail = 0;
        for (int sl = 0; sl < ec.slots; ++sl) avail += ec.cls[(size_t) sl] == c && ec_usable(sl);
        const size_t top = std::min(keys.size(), (size_t) avail);
        std::partial_sort(keys.begin(), keys.begin() + (long) top, keys.end(), [&](int a1, int b1) {
            return ec.score[(size_t) a1] > ec.score[(size_t) b1] ||
                   (ec.score[(size_t) a1] == ec.score[(size_t) b1] && a1 < b1);
        });
        for (size_t i = 0; i < top; ++i) want[(size_t) keys[i]] = 1;
        std::vector<int> spare;
        for (int sl = ec.slots - 1; sl >= 0; --sl)
            if (ec.cls[(size_t) sl] == c && ec_usable(sl) &&
                (ec.owner[(size_t) sl] < 0 || !want[(size_t) ec.owner[(size_t) sl]]))
                spare.push_back(sl);
        for (size_t i = 0; i < top && !spare.empty(); ++i)
            if (ec.where[(size_t) keys[i]] < 0) {
                todo.push_back({keys[i], spare.back()});
                spare.pop_back();
            }
    }
    ec_copy_many(todo);
}

void Engine::Impl::ec_adapt() {
    if (ec.slots == 0 || ec.per == 0) return;
    if (ec.pend.size() - ec.pend_head >= (size_t) ec.per) return;   // (the last swaps still on their way)
    const int per = ec.per;
    auto by_score = [&](int a1, int b1) { return ec.score[(size_t) a1] > ec.score[(size_t) b1]; };
    std::vector<std::pair<int, int>> todo;
    for (int c = 0; c < (int) ec.cls_bytes.size(); ++c) {
        // the class's best-scored experts not in VRAM, its worst-scored in it
        std::vector<int> best, worst;
        for (int il = g.n_dense; il < g.n_layer; ++il) {
            if (ec.layer_cls[(size_t) il] != c) continue;
            for (int e = 0; e < ec.ne; ++e) {
                const int key = il * ec.ne + e;
                if (ec.where[(size_t) key] != -1 || ec.score[(size_t) key] <= 0.0f) continue;
                // (an expert the helper card holds is off the CPU already: the main card's swaps go to the ones that
                //  are not, and the helper's slots stay as they are; INK_HELPER_STEAL=1: the main card takes them too)
                if (hc.slots && !hc_steal && hc.where[(size_t) key] != -1) continue;
                best.push_back(key);
                std::push_heap(best.begin(), best.end(), by_score);
                if ((int) best.size() > per) {
                    std::pop_heap(best.begin(), best.end(), by_score);
                    best.pop_back();
                }
            }
        }
        std::sort(best.begin(), best.end(), by_score);
        std::vector<int> free_slots;
        for (int sl = 0; sl < ec.slots; ++sl) {
            if (ec.cls[(size_t) sl] != c || !ec_usable(sl)) continue;
            const int ow = ec.owner[(size_t) sl];
            if (ow < 0) free_slots.push_back(sl);
            else if (ec.where[(size_t) ow] >= 0) worst.push_back(sl);   // (not one being copied in)
        }
        std::partial_sort(worst.begin(), worst.begin() + (long) std::min(worst.size(), (size_t) per), worst.end(),
                          [&](int a1, int b1) {
                              return ec.score[(size_t) ec.owner[(size_t) a1]] < ec.score[(size_t) ec.owner[(size_t) b1]];
                          });
        size_t wi = 0;
        for (int key : best) {
            if (!free_slots.empty()) {
                todo.push_back({key, free_slots.back()});
                free_slots.pop_back();
                continue;
            }
            if (wi >= worst.size() || (int) wi >= per) break;
            const int sl = worst[wi];
            if (ec.score[(size_t) key] <= ec.margin * ec.score[(size_t) ec.owner[(size_t) sl]] + 0.5f) break;
            todo.push_back({key, sl});
            ++wi;
        }
    }
    if (!ec.defer) {
        ec_copy_many(todo);
        return;
    }
    // the slots given to their new experts now, the bytes staged in the next tokens' gaps (ec_stage_begin)
    const int E = g.n_embd, ff = g.n_ff_exp;
    for (const auto& kv : todo) {
        const int key = kv.first, sl = kv.second;
        if (ec.owner[(size_t) sl] >= 0) ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
        ec.owner[(size_t) sl] = key;
        ec.where[(size_t) key] = -2;
        const CpuLayer& c = L[(size_t) (key / ec.ne)].cl;
        EC::Pending p;
        p.key = key;
        p.slot = sl;
        p.bytes = (size_t) ff * (c.rg + c.ru) + (size_t) E * c.rd;
        ec.pend.push_back(p);
    }
}

// The lent slots back to the prompt path: emptied, once the GPU is through with them (its experts, the copies into them)
void Engine::Impl::ec_reclaim() {
    if (!ec.lent) return;
    ec_flush();
    CK(cudaStreamSynchronize(pf.cs));
    CK(cudaStreamSynchronize(s));
    for (int sl = ec.own; sl < ec.arena_end; ++sl)
        if (ec.owner[(size_t) sl] >= 0) {
            ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
            ec.owner[(size_t) sl] = -1;
        }
    ec.lent = false;
}

// The K / V rows below `rows` about to be written: the slots there emptied first (the GPU done with them, the copies
// into them finished), and the reserve kKvSlack rows past it, so this happens once in kKvSlack tokens
void Engine::Impl::ec_kv_reserve(int rows) {
    if (rows <= ec.kv_hi || ec.slots == ec.arena_end) return;
    const int hi = std::min(max_ctx, rows + kKvSlack);
    bool any = false;
    for (int sl = ec.arena_end; sl < ec.slots && !any; ++sl) any = ec.kv_row[(size_t) sl] < hi && ec.owner[(size_t) sl] >= 0;
    if (any) {
        ec_flush();
        CK(cudaStreamSynchronize(pf.cs));
        CK(cudaStreamSynchronize(s));
        for (int sl = ec.arena_end; sl < ec.slots; ++sl)
            if (ec.kv_row[(size_t) sl] < hi && ec.owner[(size_t) sl] >= 0) {
                ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
                ec.owner[(size_t) sl] = -1;
            }
    }
    ec.kv_hi = hi;
}

// A prompt chunk's choices in layer il (cnt: the tokens of its T that chose each expert) into the scores.
//   0: the counts as they are.
//   1: on the written tokens' scale.  T tokens written one by one leave sum(decay^k, k = 1 .. T) of score to share
//      for each of a token's choices (a token's choice adds 1, then every score decays by the token's end); the chunk
//      gives each expert its share of that, cnt / T - what its tokens would have left had they chosen it evenly
//      through the chunk.  A long chunk so adds at most 99 (decay 0.99), a 48-token one 0.79 a count.
//   2: the same, the layer's scores first times decay^T, as T written tokens leave them.
// (Integers and the same float operations in the same order every run: the same scores.)
void Engine::Impl::pf_score(int il, const int* cnt, int T) {
    const int ne = g.n_expert;
    float* sc = ec.score.data() + (size_t) il * ne;
    if (ec.prompt == 0 || T <= 0) {
        for (int e = 0; e < ne; ++e)
            if (cnt[e]) sc[e] += (float) cnt[e];
        return;
    }
    const double d = std::min(std::max((double) ec.decay, 0.0), 1.0), dT = std::pow(d, (double) T);
    const double mass = d < 1.0 ? d * (1.0 - dT) / (1.0 - d) : (double) T;
    const float keep = (float) dT, per = (float) (mass / (double) T);
    if (ec.prompt >= 2)
        for (int e = 0; e < ne; ++e) sc[e] *= keep;
    for (int e = 0; e < ne; ++e)
        if (cnt[e]) sc[e] += (float) cnt[e] * per;
}

void Engine::Impl::ec_src_parts(int key, size_t off, size_t n,
                                std::vector<std::pair<const uint8_t*, size_t>>& out) const {
    const int il = key / ec.ne, e = key % ec.ne, E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    const std::pair<const uint8_t*, size_t> rs[3] = {
        {c.gate + (size_t) e * gb, gb}, {c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
    size_t at = 0;
    for (const auto& r0 : rs) {
        const size_t a0 = std::max(off, at), a1 = std::min(off + n, at + r0.second);
        if (a0 < a1) out.push_back({r0.first + (a0 - at), a1 - a0});
        at += r0.second;
    }
}

// The pending copies' next piece (up to `budget` bytes of the first one) into a pinned buffer, by the CPU lane's
// workers, while this thread waits for the GPU; ec_stage_end joins them and puts the piece's DMA on the copy stream.
bool Engine::Impl::ec_stage_begin(size_t budget) {
    if (!ec.defer || ec.pend_head >= ec.pend.size()) return false;
    EC::Pending& p = ec.pend[ec.pend_head];
    const size_t n = std::min(budget, p.bytes - p.done);
    CK(cudaEventSynchronize(ec.aev[ec.ak]));   // (that buffer's last DMA: long done)
    ec.stage_n = n;
    ec.stage_parts.clear();
    ec_src_parts(p.key, p.done, n, ec.stage_parts);
    const auto* parts = &ec.stage_parts;   // (by pointer: the workers run the job)
    // the piece in jobs of 128 KB, each its bytes from the parts it spans
    const size_t job = (size_t) 128 << 10;
    const int nj = (int) ((n + job - 1) / job);
    uint8_t* dst = ec.apin[ec.ak];
    ec.stage_fn = [parts, dst, job, n](int j) {
        const size_t a0 = (size_t) j * job, a1 = std::min(n, a0 + job);
        size_t at = 0;
        for (const auto& pr : *parts) {
            const size_t b0 = std::max(a0, at), b1 = std::min(a1, at + pr.second);
            if (b0 < b1) std::memcpy(dst + b0, pr.first + (b0 - at), b1 - b0);
            at += pr.second;
        }
    };
    pool->start(nj, &ec.stage_fn);
    return true;
}

void Engine::Impl::ec_stage_end() {
    pool->join();
    EC::Pending& p = ec.pend[ec.pend_head];
    CK(cudaMemcpyAsync(ec.ptr[(size_t) p.slot] + p.done, ec.apin[ec.ak], ec.stage_n, cudaMemcpyHostToDevice, pf.cs));
    CK(cudaEventRecord(ec.aev[ec.ak], pf.cs));
    ec.ak ^= 1;
    p.done += ec.stage_n;
    if (p.done == p.bytes) {
        ec.arrived.push_back({p.key, p.slot});
        if (++ec.pend_head == ec.pend.size()) {
            ec.pend.clear();
            ec.pend_head = 0;
        }
    }
}

// (a token's end) the experts whose last piece went up during the token before: used from the next token on (their
// DMAs, a token old, are done: the event's wait is no wait); this token's wait for the next
void Engine::Impl::ec_arrive() {
    if (!ec.arrived_prev.empty()) {
        CK(cudaEventSynchronize(ec.arrived_ev));
        for (const auto& kv : ec.arrived_prev) {
            ec.where[(size_t) kv.first] = kv.second;
            ++ec.copies;
        }
        ec.arrived_prev.clear();
    }
    if (!ec.arrived.empty()) {
        CK(cudaEventRecord(ec.arrived_ev, pf.cs));
        std::swap(ec.arrived_prev, ec.arrived);
    }
}

// Every pending copy finished now (whole experts again, through ec_copy_many), before the slots are refilled or lent back
void Engine::Impl::ec_flush() {
    if (!ec.arrived.empty() || !ec.arrived_prev.empty()) {
        CK(cudaStreamSynchronize(pf.cs));
        for (auto* v : {&ec.arrived_prev, &ec.arrived}) {
            for (const auto& kv : *v) {
                ec.where[(size_t) kv.first] = kv.second;
                ++ec.copies;
            }
            v->clear();
        }
    }
    if (ec.pend_head >= ec.pend.size()) return;
    CK(cudaStreamSynchronize(pf.cs));
    std::vector<std::pair<int, int>> todo;
    for (size_t i = ec.pend_head; i < ec.pend.size(); ++i) todo.push_back({ec.pend[i].key, ec.pend[i].slot});
    ec.pend.clear();
    ec.pend_head = 0;
    ec_copy_many(todo);
}

void Engine::Impl::gpu_experts(const Layer& l, const int* slot, const float* w, int n) {
    const int E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    iq::MvJob jobs[32];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = ec.ptr[(size_t) slot[i]];
        for (int r = 0; r < 2; ++r) {
            iq::MvJob& j = jobs[2 * i + r];
            j = iq::MvJob{};
            j.w = r == 0 ? b : b + gb;
            j.ld = r == 0 ? c.rg : c.ru;
            j.type = r == 0 ? c.tg : c.tu;
            j.k = E;
            j.n = ff;
            j.xq = xq;   // (f, quantized for the router)
            j.xd = xd;
            j.y = ec.gu + (size_t) i * 2 * ff + (size_t) r * ff;
            j.ycs = (size_t) ff;
        }
    }
    if (!iq::mv(jobs, 2 * n, s)) {
        std::fprintf(stderr, "maya-ink: no GEMV for the experts' types %d / %d\n", c.tg, c.tu);
        std::abort();
    }
    k::act_cap_rows(ec.gu, n, ff, g.gcap, g.ucap, nullptr, ec.h, s);
    k::q8(ec.h, n, ff, ec.hq, ec.hd, s);
    iq::MvJob dj[16];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = ec.ptr[(size_t) slot[i]];
        dj[i].w = b + gb + ub;
        dj[i].ld = c.rd;
        dj[i].type = c.td;
        dj[i].k = ff;
        dj[i].n = E;
        dj[i].ncol = 1;
        dj[i].xq = ec.hq + (size_t) i * ff;
        dj[i].xd = ec.hd + (size_t) i * (ff / 32);
        dj[i].y = ec.y + (size_t) i * E;
        dj[i].ycs = (size_t) E;
        dj[i].bias = nullptr;
        dj[i].alpha = w[i];
    }
    if (!iq::mv(dj, n, s)) {
        std::fprintf(stderr, "maya-ink: no GEMV for the experts' down type %d\n", c.td);
        for (int i = 0; i < n; ++i)
            std::fprintf(stderr, "  job %d: type %d k %d n %d ncol %d ld %zu ycs %zu\n", i, dj[i].type, dj[i].k,
                         dj[i].n, dj[i].ncol, dj[i].ld, dj[i].ycs);
        std::abort();
    }
    k::sum_rows(ec.y, n, E, ec.routed, s);
}

// ------------------------------------------------------------------------------------------------ the helper card
namespace {
// The calling thread on another device for a scope (a stream, its memory and its launches belong to their device).
struct OnDev {
    int prev = 0;
    explicit OnDev(int d) {
        CK(cudaGetDevice(&prev));
        CK(cudaSetDevice(d));
    }
    ~OnDev() { cudaSetDevice(prev); }
};
}  // namespace

void Engine::Impl::hc_alloc() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || o.helper_gpu >= ndev || o.helper_gpu == o.gpu || ec.cls_bytes.empty()) {
        cudaGetLastError();
        std::fprintf(stderr, "maya-ink: no helper card: --helper-gpu %d is not a second visible CUDA device (%d visible)\n",
                     o.helper_gpu, ndev);
        return;
    }
    hc.dev = o.helper_gpu;
    if (const char* v = std::getenv("INK_HELPER_EVERY")) hc.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("INK_HELPER_SWAPS")) hc.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("INK_HELPER_FILL")) hc.fill_cap = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("INK_HELPER_STEAL")) hc_steal = std::atoi(v) != 0;
    if (const char* v = std::getenv("INK_HELPER_BG")) hc.bg = std::atoi(v) != 0;
    if (const char* v = std::getenv("INK_HELPER_MARGIN")) hc.margin = (float) std::atof(v);
    if (const char* v = std::getenv("INK_HELPER_BIAS")) hc.bias = (float) std::atof(v);
    if (hc.margin < 0.0f) hc.margin = ec.margin;
    const int E = g.n_embd, ff = g.n_ff_exp, nc = (int) ec.cls_bytes.size();
    OnDev od(hc.dev);
    CK(cudaStreamCreate(&hc.s));
    CK(cudaStreamCreate(&hc.cs));
    auto dev_alloc = [&](size_t bytes) {
        void* p = nullptr;
        CK(cudaMalloc(&p, bytes));
        CK(cudaMemset(p, 0, bytes));
        hc.bytes += bytes;
        return p;
    };
    hc.f = (float*) dev_alloc((size_t) E * sizeof(float));
    hc.qq = (int8_t*) dev_alloc((size_t) E);
    hc.qd = (float*) dev_alloc((size_t) E / 32 * sizeof(float));
    hc.gu = (float*) dev_alloc((size_t) 16 * 2 * ff * sizeof(float));
    hc.h = (float*) dev_alloc((size_t) 16 * ff * sizeof(float));
    hc.hq = (int8_t*) dev_alloc((size_t) 16 * ff);
    hc.hd = (float*) dev_alloc((size_t) 16 * ff / 32 * sizeof(float));
    hc.y = (float*) dev_alloc((size_t) 16 * E * sizeof(float));
    hc.routed = (float*) dev_alloc((size_t) E * sizeof(float));
    CK(cudaHostAlloc((void**) &hc.h_out, (size_t) E * sizeof(float), cudaHostAllocPortable));
    // the fills' pinned staging: its own buffers and events (an event belongs to its device; the prompt path's staging
    // may still hold the main card's copies)
    // (INK_HELPER_STAGE: a buffer's MB, 64; 0: none - the driver's staging, as before)
    const char* hsv = std::getenv("INK_HELPER_STAGE");
    if (!hsv || std::atoi(hsv) > 0) {
        hc.pcap = (size_t) (hsv ? std::atoi(hsv) : 64) << 20;
        hc.stage = cudaHostAlloc((void**) &hc.pin[0], hc.pcap, cudaHostAllocPortable) == cudaSuccess &&
                   cudaHostAlloc((void**) &hc.pin[1], hc.pcap, cudaHostAllocPortable) == cudaSuccess;
        if (hc.stage) {
            CK(cudaEventCreateWithFlags(&hc.pev[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&hc.pev[1], cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            for (uint8_t*& b : hc.pin) {
                if (b) cudaFreeHost(b);
                b = nullptr;
            }
            std::fprintf(stderr, "maya-ink: no pinned staging for the helper card's fills (the driver's is used)\n");
        }
    }
    // the slots: this card's free VRAM less a reserve, each class's share by its layers (as the main cache's)
    std::vector<int> nl((size_t) nc, 0);
    int nmoe = 0;
    for (int il = 0; il < g.n_layer; ++il)
        if (ec.layer_cls[(size_t) il] >= 0) {
            ++nl[(size_t) ec.layer_cls[(size_t) il]];
            ++nmoe;
        }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t reserve = (size_t) 1024 << 20;
    if (const char* v = std::getenv("INK_HELPER_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
    size_t want = fr > reserve ? fr - reserve : 0;
    if (o.helper_gb >= 0.0) want = std::min(want, (size_t) (o.helper_gb * 1e9));
    double per_slot = 0;
    for (int c = 0; c < nc; ++c) per_slot += (double) nl[(size_t) c] / nmoe * (double) ec.cls_bytes[(size_t) c];
    auto counts = [&](size_t budget) {
        std::vector<int> n((size_t) nc, 0);
        const double N = (double) budget / per_slot;
        size_t used = 0;
        for (int c = 0; c < nc; ++c) {
            n[(size_t) c] = (int) (N * nl[(size_t) c] / nmoe);
            used += (size_t) n[(size_t) c] * ec.cls_bytes[(size_t) c];
        }
        if (budget > used) n[(size_t) ec.lend_cls] += (int) ((budget - used) / ec.cls_bytes[(size_t) ec.lend_cls]);
        return n;
    };
    std::vector<int> n = counts(want);
    uint8_t* base = nullptr;
    for (;;) {
        size_t bytes = 0;
        for (int c = 0; c < nc; ++c) bytes += (size_t) n[(size_t) c] * ec.cls_bytes[(size_t) c];
        if (bytes == 0) break;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) == cudaSuccess) {
            base = (uint8_t*) p;
            hc.bytes += bytes;
            break;
        }
        cudaGetLastError();
        want = want / 16 * 15;
        n = counts(want);
    }
    if (base) {
        uint8_t* at = base;
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < n[(size_t) c]; ++i) {
                hc.ptr.push_back(at);
                hc.cls.push_back(c);
                at += ec.cls_bytes[(size_t) c];
            }
    }
    hc.slots = (int) hc.ptr.size();
    hc.owner.assign((size_t) hc.slots, -1);
    hc.busy.assign((size_t) hc.slots, 0);
    hc.where.assign((size_t) g.n_layer * ec.ne, -1);
    if (hc.slots && hc.bg) hc.th = std::thread([this] { hc_worker(); });
    // The helper works in bursts of tens of microseconds a layer, so on its own its driver leaves it in P3 / P5: the
    // RTX 3090 measured 420-1,260 MHz and its memory at 5,001 or 810 MHz of 9,751 while maya-ink ran, and the experts'
    // GEMVs took 180-280 us there against 15-23 on the 5090.  A kernel kept resident while tokens are written (one
    // warp, asleep in __nanosleep, on a non-blocking stream of the lowest priority) makes the card count as busy.  Each
    // one ends by itself after INK_HELPER_KEEPALIVE_MS (50) and step() launches the next: a call that implicitly waits
    // for all of the card's work (an allocation, a plain memset) waits that long at most - one left running forever
    // hung the first token - and the card drops back to idle once the writing stops.  INK_HELPER_KEEPALIVE=0: none.
    const char* kv = std::getenv("INK_HELPER_KEEPALIVE");
    if (hc.slots && (!kv || std::atoi(kv) != 0)) {
        int lo = 0, hi = 0;
        const char* km = std::getenv("INK_HELPER_KEEPALIVE_MS");
        hc.ka_ns = (unsigned long long) std::max(1, km ? std::atoi(km) : 50) * 1000000ull;
        if (cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess &&
            cudaStreamCreateWithPriority(&hc.ks, cudaStreamNonBlocking, lo) == cudaSuccess) {
            std::fprintf(stderr, "maya-ink: helper card CUDA%d: a keep-alive kernel holds its clocks up while tokens are "
                         "written (%llu ms each)\n", hc.dev, hc.ka_ns / 1000000ull);
        } else {
            cudaGetLastError();
            hc.ks = nullptr;
            std::fprintf(stderr, "maya-ink: helper card CUDA%d: no keep-alive kernel\n", hc.dev);
        }
    }
}

// An expert's bytes (gate | up | down, as the main cache's slots) into a helper slot, on st (the helper current)
void Engine::Impl::hc_copy(int key, int slot, cudaStream_t st) {
    const int E = g.n_embd, ff = g.n_ff_exp, il = key / ec.ne, e = key % ec.ne;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    uint8_t* d = hc.ptr[(size_t) slot];
    CK(cudaMemcpyAsync(d, c.gate + (size_t) e * gb, gb, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d + gb, c.up + (size_t) e * ub, ub, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d + gb + ub, c.down + (size_t) e * db, db, cudaMemcpyHostToDevice, st));
}

// A fill's experts (key, slot) into their helper slots, the helper current: ec_copy_many's way - as many as fill a
// pinned buffer at a time, copied into it by the CPU lane's threads (1 MB pieces), then one DMA each on the helper's
// stream from there.  The same bytes into the same slots as hc_copy's; called from the token's thread between tokens
// (the pool idle), never from the swaps' thread.
void Engine::Impl::hc_copy_many(const std::vector<std::pair<int, int>>& todo) {
    const int E = g.n_embd, ff = g.n_ff_exp;
    struct Part {
        const uint8_t* src;
        size_t off, len;
    };
    size_t i = 0;
    while (i < todo.size()) {
        // a batch: the experts that fit the buffer, laid out there as in their slots
        std::vector<Part> parts;
        std::vector<std::pair<size_t, size_t>> dma;   // (todo index, offset in the buffer)
        size_t fill = 0;
        for (; i < todo.size(); ++i) {
            const int key = todo[i].first, il = key / ec.ne, e = key % ec.ne;
            const CpuLayer& c = L[(size_t) il].cl;
            const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
            if (fill + gb + ub + db > hc.pcap) {
                if (fill == 0) {   // (an expert larger than the buffer: the driver's staging)
                    hc_copy(key, todo[i].second, hc.s);
                    continue;
                }
                break;
            }
            dma.push_back({i, fill});
            const std::pair<const uint8_t*, size_t> rs[3] = {
                {c.gate + (size_t) e * gb, gb}, {c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
            for (const auto& r0 : rs) {
                for (size_t o0 = 0; o0 < r0.second; o0 += (size_t) 1 << 20)
                    parts.push_back({r0.first + o0, fill + o0, std::min((size_t) 1 << 20, r0.second - o0)});
                fill += r0.second;
            }
        }
        if (dma.empty()) continue;
        CK(cudaEventSynchronize(hc.pev[hc.pk]));   // (that buffer's last DMA is done)
        uint8_t* pin = hc.pin[hc.pk];
        pool->run((int) parts.size(), [&](int j) {
            const Part& p0 = parts[(size_t) j];
            std::memcpy(pin + p0.off, p0.src, p0.len);
        });
        for (const auto& d : dma) {
            const int key = todo[d.first].first, sl = todo[d.first].second;
            const CpuLayer& c = L[(size_t) (key / ec.ne)].cl;
            const size_t bytes = (size_t) ff * (c.rg + c.ru) + (size_t) E * c.rd;
            CK(cudaMemcpyAsync(hc.ptr[(size_t) sl], pin + d.second, bytes, cudaMemcpyHostToDevice, hc.s));
        }
        CK(cudaEventRecord(hc.pev[hc.pk], hc.s));
        hc.pk ^= 1;
    }
}

// The swaps' thread: one expert at a time from the queue into its slot (no launch reads that slot until the main
// thread has collected it)
void Engine::Impl::hc_worker() {
    CK(cudaSetDevice(hc.dev));
    for (;;) {
        std::pair<int, int> job;
        {
            std::unique_lock<std::mutex> lk(hc.mu);
            hc.cv.wait(lk, [&] { return hc.quit || hc.qhead < hc.q.size(); });
            if (hc.quit) return;
            job = hc.q[hc.qhead++];
            if (hc.qhead == hc.q.size()) {
                hc.q.clear();
                hc.qhead = 0;
            }
        }
        hc_copy(job.first, job.second, hc.cs);
        CK(cudaStreamSynchronize(hc.cs));
        {
            std::lock_guard<std::mutex> lk(hc.mu);
            hc.done.push_back(job);
        }
        hc.cv_done.notify_one();
    }
}

// The copied experts become usable (wait: all of those on their way first)
void Engine::Impl::hc_collect(bool wait) {
    if (hc.inflight == 0) return;
    std::unique_lock<std::mutex> lk(hc.mu);
    if (wait) hc.cv_done.wait(lk, [&] { return (int) hc.done.size() == hc.inflight; });
    for (const auto& kv : hc.done) {
        hc.busy[(size_t) kv.second] = 0;
        hc.where[(size_t) kv.first] = kv.second;
        ++hc.copies;
    }
    hc.inflight -= (int) hc.done.size();
    hc.done.clear();
}

// The helper's slots brought up to date, at most max_copies experts copied: the slots whose expert the main card holds
// (or is taking) are freed, then the best-scored experts outside both cards come in - into free slots, then in place
// of the worst-scored ones (with hysteresis: only one that beats its slot's expert by the main cache's margin).  Called
// between tokens, the helper idle.  bg: the copies go to the worker and count against max_copies until they are done;
// otherwise they are made here, after the worker's.
void Engine::Impl::hc_update(int max_copies, bool hysteresis, bool bg) {
    if (hc.slots == 0) return;
    bg = bg && hc.th.joinable();
    hc_collect(!bg);
    max_copies -= hc.inflight;
    if (max_copies <= 0) return;
    std::vector<std::pair<int, int>> todo;   // (key, slot)
    auto better = [&](int a1, int b1) {
        return ec.score[(size_t) a1] > ec.score[(size_t) b1] || (ec.score[(size_t) a1] == ec.score[(size_t) b1] && a1 < b1);
    };
    for (int c = 0; c < (int) ec.cls_bytes.size(); ++c) {
        std::vector<int> free_slots, res, cand;
        for (int sl = 0; sl < hc.slots; ++sl) {
            if (hc.cls[(size_t) sl] != c || hc.busy[(size_t) sl]) continue;
            const int ow = hc.owner[(size_t) sl];
            if (ow >= 0 && ec.where[(size_t) ow] != -1) {
                hc.where[(size_t) ow] = -1;
                hc.owner[(size_t) sl] = -1;
            }
            if (hc.owner[(size_t) sl] < 0) free_slots.push_back(sl);
            else res.push_back(sl);
        }
        for (int il = 0; il < g.n_layer; ++il) {
            if (ec.layer_cls[(size_t) il] != c) continue;
            for (int e = 0; e < ec.ne; ++e) {
                const size_t key = (size_t) il * ec.ne + e;
                if (ec.score[key] > 0.0f && ec.where[key] == -1 && hc.where[key] == -1) cand.push_back((int) key);
            }
        }
        const size_t left = (size_t) max_copies > todo.size() ? (size_t) max_copies - todo.size() : 0;
        const size_t top = std::min(cand.size(), left);
        if (top == 0) continue;
        std::partial_sort(cand.begin(), cand.begin() + (long) top, cand.end(), better);
        const size_t nw = std::min(res.size(), top);
        std::partial_sort(res.begin(), res.begin() + (long) nw, res.end(), [&](int a1, int b1) {
            return ec.score[(size_t) hc.owner[(size_t) a1]] < ec.score[(size_t) hc.owner[(size_t) b1]];
        });
        size_t wi = 0;
        for (size_t i = 0; i < top; ++i) {
            const int key = cand[i];
            int sl = -1;
            if (!free_slots.empty()) {
                sl = free_slots.back();
                free_slots.pop_back();
            } else {
                if (wi >= nw) break;
                const int w0 = res[wi];
                const float rs = ec.score[(size_t) hc.owner[(size_t) w0]];
                if (hysteresis ? ec.score[(size_t) key] <= hc.margin * rs + hc.bias : ec.score[(size_t) key] <= rs) break;
                hc.where[(size_t) hc.owner[(size_t) w0]] = -1;
                sl = w0;
                ++wi;
            }
            hc.owner[(size_t) sl] = key;
            hc.where[(size_t) key] = bg ? -2 : sl;
            todo.push_back({key, sl});
        }
    }
    if (todo.empty()) return;
    if (bg) {
        for (const auto& kv : todo) hc.busy[(size_t) kv.second] = 1;
        hc.inflight += (int) todo.size();
        {
            std::lock_guard<std::mutex> lk(hc.mu);
            hc.q.insert(hc.q.end(), todo.begin(), todo.end());
        }
        hc.cv.notify_one();
        return;
    }
    OnDev od(hc.dev);
    // (a fill - the start's, a prompt's - through the pinned staging; the few swaps between tokens as before)
    if (hc.stage && !hysteresis) hc_copy_many(todo);
    else
        for (const auto& kv : todo) hc_copy(kv.first, kv.second, hc.s);
    hc.copies += (long) todo.size();
    CK(cudaStreamSynchronize(hc.s));
    // (INK_HELPER_VERIFY=1, a check) every slot just filled read back against the mapping's bytes
    static const bool verify = std::getenv("INK_HELPER_VERIFY") != nullptr;
    if (verify) {
        std::vector<uint8_t> back;
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        for (const auto& kv : todo) {
            const CpuLayer& c = L[(size_t) (kv.first / ec.ne)].cl;
            const size_t bytes = (size_t) g.n_ff_exp * (c.rg + c.ru) + (size_t) g.n_embd * c.rd;
            back.resize(bytes);
            CK(cudaMemcpy(back.data(), hc.ptr[(size_t) kv.second], bytes, cudaMemcpyDeviceToHost));
            parts.clear();
            ec_src_parts(kv.first, 0, bytes, parts);
            size_t at = 0;
            for (const auto& pr : parts) {
                if (std::memcmp(back.data() + at, pr.first, pr.second) != 0) {
                    std::fprintf(stderr,
                                 "maya-ink: INK_HELPER_VERIFY: expert %d of layer %d differs in helper slot %d\n",
                                 kv.first % ec.ne, kv.first / ec.ne, kv.second);
                    std::abort();
                }
                at += pr.second;
            }
        }
        std::fprintf(stderr,
                     "maya-ink: INK_HELPER_VERIFY: %zu experts in the helper's slots hold the mapping's bytes\n",
                     todo.size());
    }
}

// A token's experts on the helper card: f up from the host, its int8 rows, then gpu_experts' launches on the helper's
// stream and buffers; their sum -> hc.h_out (there once hc.s is synchronized).
void Engine::Impl::hc_experts(const Layer& l, const int* slot, const float* w, int n) {
    const int E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    OnDev od(hc.dev);
    CK(cudaMemcpyAsync(hc.f, h_f, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, hc.s));
    k::q8(hc.f, 1, E, hc.qq, hc.qd, hc.s);
    iq::MvJob jobs[32];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = hc.ptr[(size_t) slot[i]];
        for (int r = 0; r < 2; ++r) {
            iq::MvJob& j = jobs[2 * i + r];
            j = iq::MvJob{};
            j.w = r == 0 ? b : b + gb;
            j.ld = r == 0 ? c.rg : c.ru;
            j.type = r == 0 ? c.tg : c.tu;
            j.k = E;
            j.n = ff;
            j.xq = hc.qq;
            j.xd = hc.qd;
            j.y = hc.gu + (size_t) i * 2 * ff + (size_t) r * ff;
            j.ycs = (size_t) ff;
        }
    }
    if (!iq::mv(jobs, 2 * n, hc.s)) {
        std::fprintf(stderr, "maya-ink: no GEMV for the experts' types %d / %d\n", c.tg, c.tu);
        std::abort();
    }
    k::act_cap_rows(hc.gu, n, ff, g.gcap, g.ucap, nullptr, hc.h, hc.s);
    k::q8(hc.h, n, ff, hc.hq, hc.hd, hc.s);
    iq::MvJob dj[16];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = hc.ptr[(size_t) slot[i]];
        dj[i] = iq::MvJob{};
        dj[i].w = b + gb + ub;
        dj[i].ld = c.rd;
        dj[i].type = c.td;
        dj[i].k = ff;
        dj[i].n = E;
        dj[i].ncol = 1;
        dj[i].xq = hc.hq + (size_t) i * ff;
        dj[i].xd = hc.hd + (size_t) i * (ff / 32);
        dj[i].y = hc.y + (size_t) i * E;
        dj[i].ycs = (size_t) E;
        dj[i].bias = nullptr;
        dj[i].alpha = w[i];
    }
    if (!iq::mv(dj, n, hc.s)) {
        std::fprintf(stderr, "maya-ink: no GEMV for the experts' down type %d\n", c.td);
        std::abort();
    }
    k::sum_rows(hc.y, n, E, hc.routed, hc.s);
    CK(cudaMemcpyAsync(hc.h_out, hc.routed, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, hc.s));
    // WDDM holds a stream's launches until something submits them: without this the helper's work may only start at
    // its sync after the CPU lane (maya-ds4's win_submit for its helper).  INK_HELPER_FLUSH=1: done (2026-11-12: no
    // gain here, 23.2 against 23.5 tok/s on ab.py - the lane is the longer wait - so off by default).
    static const bool flush = std::getenv("INK_HELPER_FLUSH") && std::atoi(std::getenv("INK_HELPER_FLUSH")) != 0;
    if (flush) (void) cudaStreamQuery(hc.s);
}

// (--bench-h2d) the ways to move the experts' bytes to the GPU on this machine, ~1.2 GB each
void Engine::bench_h2d() {
    Impl& m = *p;
    const Layer& l = m.L[(size_t) std::min(m.g.n_layer - 1, 10)];
    const CpuLayer& c = l.cl;
    const size_t eb = (size_t) m.g.n_ff_exp * c.rg, n_e = (size_t) m.g.n_expert, bytes = eb * n_e;
    uint8_t* dst = nullptr;
    CK(cudaMalloc(&dst, (size_t) 64 << 20));
    cudaStream_t cs = m.pf.cs;
    auto touch = [&](const uint8_t* src) {   // the pages resident first: the copy, not the disk, is timed
        volatile uint8_t acc = 0;
        for (size_t i = 0; i < bytes; i += 4096) acc ^= src[i];
        (void) acc;
    };
    auto report = [&](const char* what, double ms, size_t b) {
        std::fprintf(stderr, "maya-ink: H2D %-44s %6.2f GB/s (%.2f GB in %.0f ms)\n", what, b / ms / 1e6, b / 1e9, ms);
    };
    // (a) cudaMemcpyAsync straight from the GGUF's mapping (pageable: the driver stages it), an expert a call
    touch(c.gate);
    double t0 = now_ms();
    for (size_t e = 0; e < n_e; ++e)
        CK(cudaMemcpyAsync(dst + (e & 7) * eb, c.gate + e * eb, eb, cudaMemcpyHostToDevice, cs));
    CK(cudaStreamSynchronize(cs));
    report("pageable, an expert (4.7 MB) a call", now_ms() - t0, bytes);
    t0 = now_ms();
    for (size_t o = 0; o < bytes; o += (size_t) 64 << 20)
        CK(cudaMemcpyAsync(dst, c.gate + o, std::min((size_t) 64 << 20, bytes - o), cudaMemcpyHostToDevice, cs));
    CK(cudaStreamSynchronize(cs));
    report("pageable, 64 MB a call", now_ms() - t0, bytes);
    // (b) CPU threads copy into pinned staging, the DMA takes it from there (two buffers in turn)
    touch(c.up);
    const size_t piece = (size_t) 16 << 20;
    uint8_t* pin[2] = {nullptr, nullptr};
    cudaEvent_t ev[2];
    if (cudaHostAlloc((void**) &pin[0], piece, cudaHostAllocPortable) == cudaSuccess &&
        cudaHostAlloc((void**) &pin[1], piece, cudaHostAllocPortable) == cudaSuccess) {
        CK(cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming));
        for (int threads : {1, 4, 8}) {
            t0 = now_ms();
            int k = 0;
            for (size_t o = 0; o < bytes; o += piece, k ^= 1) {
                const size_t len = std::min(piece, bytes - o);
                CK(cudaEventSynchronize(ev[k]));   // (that buffer's last DMA done)
                const size_t per = (len + threads - 1) / threads;
                std::vector<std::thread> th;
                for (int t = 0; t < threads; ++t) {
                    const size_t a0 = (size_t) t * per, a1 = std::min(len, a0 + per);
                    if (a0 < a1) th.emplace_back([&, a0, a1] { std::memcpy(pin[k] + a0, c.up + o + a0, a1 - a0); });
                }
                for (auto& x : th) x.join();
                CK(cudaMemcpyAsync(dst + (size_t) k * piece, pin[k], len, cudaMemcpyHostToDevice, cs));
                CK(cudaEventRecord(ev[k], cs));
            }
            CK(cudaStreamSynchronize(cs));
            char what[64];
            std::snprintf(what, sizeof what, "pinned staging, %d copy thread%s", threads, threads > 1 ? "s" : "");
            report(what, now_ms() - t0, bytes);
        }
        // (b2) the DMA alone from pinned memory (the link's own speed)
        t0 = now_ms();
        for (size_t o = 0; o < bytes; o += piece)
            CK(cudaMemcpyAsync(dst + (o / piece & 1) * piece, pin[(o / piece) & 1], piece, cudaMemcpyHostToDevice, cs));
        CK(cudaStreamSynchronize(cs));
        report("pinned memory, the DMA alone", now_ms() - t0, bytes);
        cudaFreeHost(pin[0]);
        cudaFreeHost(pin[1]);
    } else {
        cudaGetLastError();
        std::fprintf(stderr, "maya-ink: H2D: no pinned staging (cudaHostAlloc failed)\n");
    }
    // (d) a layer the load touched, not touched again: are its pages still in the working set?
    {
        const Layer& l3 = m.L[(size_t) std::min(m.g.n_layer - 1, 30)];
        const uint8_t* b3 = l3.cl.gate;
        const auto f0 = page_faults();
        t0 = now_ms();
        for (size_t o = 0; o < bytes; o += (size_t) 64 << 20)
            CK(cudaMemcpyAsync(dst, b3 + o, std::min((size_t) 64 << 20, bytes - o), cudaMemcpyHostToDevice, cs));
        CK(cudaStreamSynchronize(cs));
        const double ms = now_ms() - t0;
        report("pageable, touched at load only", ms, bytes);
        std::fprintf(stderr, "maya-ink: H2D (%lu page faults in that copy; working set %.1f GB)\n",
                     (unsigned long) (page_faults() - f0), working_set_gb());
    }
    // (c) the mapping itself pinned in place (cudaHostRegister), then the DMA from it
    const Layer& l2 = m.L[(size_t) std::min(m.g.n_layer - 1, 11)];
    uint8_t* base = (uint8_t*) l2.cl.gate;
    touch(base);
    t0 = now_ms();
    const cudaError_t r = cudaHostRegister(base, bytes, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (r == cudaSuccess) {
        const double pin_ms = now_ms() - t0;
        t0 = now_ms();
        for (size_t o = 0; o < bytes; o += (size_t) 64 << 20)
            CK(cudaMemcpyAsync(dst, base + o, std::min((size_t) 64 << 20, bytes - o), cudaMemcpyHostToDevice, cs));
        CK(cudaStreamSynchronize(cs));
        report("the mapping pinned in place", now_ms() - t0, bytes);
        std::fprintf(stderr, "maya-ink: H2D (pinning %.2f GB of the mapping took %.0f ms)\n", bytes / 1e9, pin_ms);
        cudaHostUnregister(base);
    } else {
        std::fprintf(stderr, "maya-ink: H2D: cudaHostRegister on the mapping failed: %s\n", cudaGetErrorString(r));
        cudaGetLastError();
    }
    cudaFree(dst);
}

bool Engine::Impl::save_profile(const std::string& path) const {
    FILE* f = std::fopen((path + ".tmp").c_str(), "wb");
    if (!f) return false;
    const int hdr[4] = {0x504b4e49, 1, g.n_layer, g.n_expert};   // "INKP", version 1
    bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
              std::fwrite(ec.score.data(), sizeof(float), ec.score.size(), f) == ec.score.size();
    ok = std::fclose(f) == 0 && ok;
    if (!ok) return false;
    std::remove(path.c_str());
    return std::rename((path + ".tmp").c_str(), path.c_str()) == 0;
}

bool Engine::Impl::load_profile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    int hdr[4] = {0, 0, 0, 0};
    std::vector<float> v(ec.score.size());
    const bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x504b4e49 && hdr[1] == 1 &&
                    hdr[2] == g.n_layer && hdr[3] == g.n_expert &&
                    std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
    std::fclose(f);
    if (ok) ec.score = std::move(v);
    return ok;
}

// ------------------------------------------------------------------------------------------------ the prompt path
bool Engine::Impl::pf_alloc(std::string& err) {
    const int C = pf.C = std::max(64, o.chunk);
    const int E = g.n_embd, H = g.n_head, hd = g.head_dim, ne = g.n_expert, nsh = g.n_shexp, nu = g.n_used;
    const int ffe = g.n_ff_exp, ffd = g.n_ff, qw = H * hd, G = H / g.n_head_kv;
    if (cublasCreate(&pf.cb) != CUBLAS_STATUS_SUCCESS) {
        err = "cuBLAS did not start";
        return false;
    }
    cublasSetStream(pf.cb, s);
    CK(cudaStreamCreateWithFlags(&pf.cs, cudaStreamNonBlocking));
    const size_t CE = (size_t) C * E;
    size_t mib = 960;
    if (const char* v = std::getenv("INK_SCORES_MIB")) mib = (size_t) std::max(64, std::atoi(v));
    pf.s_elems = (mib << 20) / 6;   // S F32 and P F16
    pf.w1_elems = std::max({(size_t) 2 * ffd * E, (size_t) 2 * ffe * E, (size_t) qw * E});
    pf.w2_elems = std::max({(size_t) ffd * E, (size_t) ffe * E, (size_t) E * qw});
    for (const Layer& l : L)
        if (!l.dense)
            pf.slot_bytes = std::max(pf.slot_bytes, (size_t) ffe * (l.cl.rg + l.cl.ru) + (size_t) E * l.cl.rd);
    pf.GE = std::max(1, std::min(ne, o.stage));
    // Every buffer in one arena (none holds anything from one prompt to the next, so the expert cache can borrow it,
    // and each is written before it is read): the same list twice, first its size, then its pointers, 256-byte aligned.
    for (int pass = 0; pass < 2; ++pass) {
        size_t at = 0;
        auto carve = [&](auto*& ptr, size_t n) {
            using T = std::remove_reference_t<decltype(*ptr)>;
            if (pass == 1) ptr = (T*) (pf.arena + at);
            at += (n * sizeof(T) + 255) & ~(size_t) 255;
        };
        carve(pf.ids, (size_t) C);
        carve(pf.X, CE);
        carve(pf.A, CE);
        carve(pf.Q, (size_t) C * qw);
        carve(pf.Kf, (size_t) C * kvw);
        carve(pf.Vf, (size_t) C * kvw);
        carve(pf.Kc, (size_t) C * kvw);
        carve(pf.Vc, (size_t) C * kvw);
        carve(pf.R, (size_t) C * H * g.d_rel);
        carve(pf.Y, CE);
        carve(pf.Ye, CE);
        carve(pf.GU, (size_t) C * 2 * ffe);
        carve(pf.RL, (size_t) C * (ne + nsh));
        carve(pf.sel, (size_t) C * nu);
        carve(pf.wts, (size_t) C * (nu + nsh));
        carve(pf.cnt, (size_t) ne);
        carve(pf.off, (size_t) ne);
        carve(pf.list, (size_t) C * nu);
        carve(pf.wl, (size_t) C * nu);
        carve(pf.tau, (size_t) C);
        carve(pf.RES, CE);
        carve(pf.GATE, (size_t) C * qw);
        carve(pf.SINK, (size_t) C * H);
        carve(pf.DYN, (size_t) C * g.n_head_kv * g.conv_k);
        carve(pf.BLK, (size_t) n_blocks() * CE);
        carve(pf.Ah, CE);
        carve(pf.Qh, (size_t) C * qw);
        carve(pf.AOh, (size_t) C * qw);
        carve(pf.Xe, CE);
        carve(pf.He, (size_t) C * ffe);
        carve(pf.Kw, (size_t) (g.n_swa + C) * kvw);
        carve(pf.Vw, (size_t) (g.n_swa + C) * kvw);
        carve(pf.Ot, (size_t) G * C * hd);
        carve(pf.S, pf.s_elems);
        carve(pf.P, pf.s_elems);
        carve(pf.W1, pf.w1_elems);
        carve(pf.W2, pf.w2_elems);
        // two group buffers in turn, `stage` experts each (the copy of one group while the other's are multiplied)
        carve(pf.gbuf[0], (size_t) pf.GE * pf.slot_bytes);
        carve(pf.gbuf[1], (size_t) pf.GE * pf.slot_bytes);
        if (embd_host) carve(pf.Erows, (size_t) C * embd.rb);   // (a chunk's embedding rows)
        if (pass == 0) {
            pf.arena_bytes = at;
            pf.arena = dalloc<uint8_t>(at);
        }
    }
    for (int i = 0; i < 2; ++i) {
        CK(cudaEventCreateWithFlags(&pf.gready[i], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&pf.gfreed[i], cudaEventDisableTiming));
    }
    pf.h_ids = halloc<int>(C);
    pf.h_cnt = halloc<int>(ne);
    pf.h_off = halloc<int>(ne + 1);
    pf.h_tau = halloc<float>(C);
    const char* pg = std::getenv("INK_PF_PAGEABLE");
    if (!pg || std::atoi(pg) == 0) {
        stg.cap = (size_t) 64 << 20;
        stg.on = cudaHostAlloc((void**) &stg.pin[0], stg.cap, cudaHostAllocPortable) == cudaSuccess &&
                 cudaHostAlloc((void**) &stg.pin[1], stg.cap, cudaHostAllocPortable) == cudaSuccess;
        if (stg.on) {
            CK(cudaEventCreateWithFlags(&stg.ev[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stg.ev[1], cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "maya-ink: no pinned staging for the prompt path's copies (the driver's is used)\n");
        }
    }
    return true;
}

void Engine::Impl::stage_copy(uint8_t* dst, const uint8_t* src, size_t len) {
    if (!stg.on) {
        CK(cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, pf.cs));
        return;
    }
    const int nt = pool->threads();
    while (len) {
        const size_t n = std::min(len, stg.cap);
        CK(cudaEventSynchronize(stg.ev[stg.k]));   // (that buffer's last DMA is done)
        uint8_t* pin = stg.pin[stg.k];
        const size_t per = ((n + (size_t) nt - 1) / (size_t) nt + 4095) & ~(size_t) 4095;
        pool->run(nt, [&](int j) {
            const size_t a0 = (size_t) j * per;
            if (a0 < n) std::memcpy(pin + a0, src + a0, std::min(per, n - a0));
        });
        CK(cudaMemcpyAsync(dst, pin, n, cudaMemcpyHostToDevice, pf.cs));
        CK(cudaEventRecord(stg.ev[stg.k], pf.cs));
        stg.k ^= 1;
        dst += n;
        src += n;
        len -= n;
    }
}

void Engine::Impl::f16w(const DMat& m, __half* dst) {
    if (!iq::f16_rows(m.type, m.w, m.k, m.n, (uint16_t*) dst, m.k, s)) {
        std::fprintf(stderr, "maya-ink: no F16 rows for type %d\n", m.type);
        std::abort();
    }
}

void Engine::Impl::gemm(const __half* X, int ldx, const __half* W, int ldw, float* Y, int ldy, int T, int n, int k,
                        float alpha, float beta) {
    // Y [T][n] (row-major, ldy) = alpha X [T][k] . W [n][k]^T + beta Y: column-major Y^T = W . X^T
    const cublasStatus_t r = cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k, &alpha, W, CUDA_R_16F, ldw, X,
                                          CUDA_R_16F, ldx, &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ink: cuBLAS GEMM %dx%dx%d failed (%d)\n", T, n, k, (int) r);
        std::abort();
    }
}

void Engine::Impl::pf_attention(const Layer& l, int T, int pos0, const __half* Kb, const __half* Vb, int kp0,
                                int nkeys, float* out) {
    const int H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, G = H / Hkv;
    const float scale = 1.0f / std::sqrt((float) hd), one = 1.0f, zero = 0.0f;
    const int window = l.swa ? g.n_swa : 0;
    auto r8 = [](int v) { return (v + 7) & ~7; };
    // the keys a tile of the queries [ta, tb) needs: from lo(ta) (the window's start on a sliding layer) to the last
    // query's own position
    auto lo = [&](int ta) { return l.swa ? std::max(0, pos0 + ta - (g.n_swa - 1) - kp0) : 0; };
    auto hi = [&](int tb) { return std::min(nkeys, pos0 + tb - kp0); };
    int ta = 0;
    while (ta < T) {
        int tq = l.swa ? std::min(T - ta, 2 * g.n_swa) : T - ta;
        while (tq > 1 && (size_t) G * tq * (size_t) r8(hi(ta + tq) - lo(ta)) > pf.s_elems) tq = (tq + 1) / 2;
        const int tb = ta + tq, j0 = lo(ta), nk = hi(tb) - j0, ld = r8(nk), M = G * tq;
        for (int gi = 0; gi < Hkv; ++gi) {
            const __half* Qt = pf.Qh + (((size_t) gi * T + ta) * G) * hd;
            const __half* Kt = Kb + (size_t) j0 * kvw + (size_t) gi * hd;
            const __half* Vt = Vb + (size_t) j0 * kvw + (size_t) gi * hd;
            cublasStatus_t r = cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, M, hd, &scale, Kt, CUDA_R_16F, kvw,
                                            Qt, CUDA_R_16F, hd, &zero, pf.S, CUDA_R_32F, ld, CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT);
            if (r != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "maya-ink: cuBLAS scores %dx%d failed (%d)\n", M, nk, (int) r);
                std::abort();
            }
            k::attn_softmax2(pf.S, ld, pf.P, ld, M, nk, G, gi, ta, pos0, kp0 + j0, window, pf.R, l.rel_proj, l.E, H,
                             g.d_rel, pf.SINK, s);
            r = cublasGemmEx(pf.cb, CUBLAS_OP_N, CUBLAS_OP_N, hd, M, nk, &one, Vt, CUDA_R_16F, kvw, pf.P, CUDA_R_16F, ld,
                             &zero, pf.Ot, CUDA_R_32F, hd, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (r != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "maya-ink: cuBLAS attention output %dx%d failed (%d)\n", M, nk, (int) r);
                std::abort();
            }
            k::attn_scatter(pf.Ot, out, M, G, gi, ta, H, hd, s);
        }
        ta = tb;
    }
}

void Engine::Impl::pf_dense(const Layer& l, int T) {
    const int E = g.n_embd, ff = g.n_ff;
    f16w(l.gate, pf.W1);
    f16w(l.up, pf.W1 + (size_t) ff * E);
    f16w(l.down, pf.W2);
    const int TR = std::max(1, (int) ((size_t) pf.C * 2 * g.n_ff_exp / (2 * (size_t) ff)));   // the rows GU holds
    for (int r0 = 0; r0 < T; r0 += TR) {
        const int n = std::min(TR, T - r0);
        gemm(pf.Ah + (size_t) r0 * E, E, pf.W1, E, pf.GU, 2 * ff, n, 2 * ff, E);
        k::swiglu_rows(pf.GU, n, ff, nullptr, 0, pf.He, s);
        gemm(pf.He, ff, pf.W2, ff, pf.Y + (size_t) r0 * E, E, n, E, ff);
    }
    k::scale_dev(pf.Y, pf.Y, l.gscale_d, (size_t) T * E, s);
}

void Engine::Impl::pf_moe(const Layer& l, int T) {
    const int E = g.n_embd, ne = g.n_expert, nsh = g.n_shexp, nu = g.n_used, ff = g.n_ff_exp, nw = nu + nsh;
    const float one = 1.0f, zero = 0.0f;
    // the router in F32 (llama.cpp: GGML_PREC_F32_PEDANTIC), the routing on the GPU
    if (cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, ne + nsh, T, E, &one, (const float*) l.gate_inp.w, E, pf.A, E,
                    &zero, pf.RL, ne + nsh) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ink: cuBLAS router failed\n");
        std::abort();
    }
    k::route2(pf.RL, ne + nsh, l.exp_bias_d, T, ne, nu, g.exp_scale, pf.sel, pf.wts, s);
    CK(cudaMemsetAsync(pf.cnt, 0, (size_t) ne * sizeof(int), s));
    k::expert_count(pf.sel, T * nu, pf.cnt, s);
    CK(cudaMemcpyAsync(pf.h_cnt, pf.cnt, (size_t) ne * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    pf.h_off[0] = 0;
    for (int e = 0; e < ne; ++e) pf.h_off[e + 1] = pf.h_off[e] + pf.h_cnt[e];
    CK(cudaMemcpyAsync(pf.off, pf.h_off, (size_t) ne * sizeof(int), cudaMemcpyHostToDevice, s));
    k::expert_lists(pf.sel, pf.wts, T, nu, nw, ne, pf.off, pf.list, pf.wl, s);
    CK(cudaMemsetAsync(pf.Y, 0, (size_t) T * E * sizeof(float), s));
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    const int il = (int) (&l - L.data()), GE = pf.GE;
    static const bool no_copy = std::getenv("INK_PF_NOCOPY") != nullptr, no_math = std::getenv("INK_PF_NOMATH") != nullptr;
    pf_score(il, pf.h_cnt, T);
    if (route_log) {
        std::vector<uint16_t> c16((size_t) ne);
        for (int e = 0; e < ne; ++e) c16[(size_t) e] = (uint16_t) std::min(pf.h_cnt[e], 65535);
        const uint8_t hdr[2] = {'P', (uint8_t) il};
        std::fwrite(hdr, 1, 2, route_log);
        std::fwrite(c16.data(), 2, c16.size(), route_log);
    }
    // The experts in groups of GE: a group's used experts that are not in VRAM copied into one of two group buffers
    // (each run of consecutive ones in one copy a tensor - large copies are what the driver's staging moves fastest),
    // laid out as the GGUF holds them: [the gates | the ups | the downs] of the group's experts.
    for (int e0 = 0; e0 < ne; e0 += GE) {
        const int e1 = std::min(ne, e0 + GE);
        bool any = false, copy = false;
        for (int e = e0; e < e1; ++e)
            if (pf.h_cnt[e]) {
                any = true;
                copy = copy || !ec_blob(il, e);
            }
        if (!any) continue;
        int bi = -1;
        uint8_t* B = nullptr;
        if (copy) {
            bi = (int) (pf.group_ctr++ & 1);
            B = pf.gbuf[bi];
            if (pf.group_ctr > 2) CK(cudaStreamWaitEvent(pf.cs, pf.gfreed[bi], 0));
            const double tc = now_ms();
            for (int e = e0; e < e1;) {
                if (!pf.h_cnt[e] || ec_blob(il, e)) {
                    ++e;
                    continue;
                }
                int r = e;
                while (r < e1 && pf.h_cnt[r] && !ec_blob(il, r)) ++r;
                const size_t n = (size_t) (r - e), at = (size_t) (e - e0);
                if (!no_copy) {
                    stage_copy(B + at * gb, c.gate + (size_t) e * gb, n * gb);
                    stage_copy(B + GE * gb + at * ub, c.up + (size_t) e * ub, n * ub);
                    stage_copy(B + GE * (gb + ub) + at * db, c.down + (size_t) e * db, n * db);
                }
                pf.copy_bytes += (double) (n * (gb + ub + db));
                e = r;
            }
            pf.copy_ms += now_ms() - tc;
            CK(cudaEventRecord(pf.gready[bi], pf.cs));
            CK(cudaStreamWaitEvent(s, pf.gready[bi], 0));
        }
        for (int e = e0; e < e1; ++e) {
            const int rows = pf.h_cnt[e], o0 = pf.h_off[e];
            if (rows == 0 || no_math) continue;
            const uint8_t* blob = ec_blob(il, e);
            const size_t at = (size_t) (e - e0);
            const uint8_t* wg = blob ? blob : B + at * gb;
            const uint8_t* wu = blob ? blob + gb : B + GE * gb + at * ub;
            const uint8_t* wd = blob ? blob + gb + ub : B + GE * (gb + ub) + at * db;
            if (!iq::f16_rows(c.tg, wg, E, ff, (uint16_t*) pf.W1, E, s) ||
                !iq::f16_rows(c.tu, wu, E, ff, (uint16_t*) (pf.W1 + (size_t) ff * E), E, s) ||
                !iq::f16_rows(c.td, wd, ff, E, (uint16_t*) pf.W2, ff, s)) {
                std::fprintf(stderr, "maya-ink: no F16 rows for the experts' types %d / %d / %d\n", c.tg, c.tu, c.td);
                std::abort();
            }
            iq::gather_f16((const uint16_t*) pf.Ah, E, pf.list + o0, rows, E, (uint16_t*) pf.Xe, s);
            gemm(pf.Xe, E, pf.W1, E, pf.GU, 2 * ff, rows, 2 * ff, E);
            k::act_cap_rows(pf.GU, rows, ff, g.gcap, g.ucap, pf.He, nullptr, s);
            gemm(pf.He, ff, pf.W2, ff, pf.Ye, E, rows, E, ff);
            k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, E, pf.Y, s);
        }
        if (bi >= 0) CK(cudaEventRecord(pf.gfreed[bi], s));
    }
    // the shared experts, every token: their gammas (the routing's last weights) on h before the down
    for (int sh = 0; sh < nsh; ++sh) {
        f16w(l.sh_gate[sh], pf.W1);
        f16w(l.sh_up[sh], pf.W1 + (size_t) ff * E);
        f16w(l.sh_down[sh], pf.W2);
        gemm(pf.Ah, E, pf.W1, E, pf.GU, 2 * ff, T, 2 * ff, E);
        k::swiglu_rows(pf.GU, T, ff, pf.wts + nu + sh, nw, pf.He, s);
        gemm(pf.He, ff, pf.W2, ff, pf.Ye, E, T, E, ff);
        k::axpy(pf.Y, pf.Ye, nullptr, (size_t) T * E, s);
    }
}

void Engine::Impl::chunk(const int* ids, int T, bool want_logits) {
    const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, kc = g.conv_k, qw = H * hd;
    const int pos0 = pos;
    ec_kv_reserve(pos0 + T);
    const uint8_t* ew = embd.w;
    if (embd_host) {
        // the chunk's rows gathered from the mapping into the pinned staging (or one by one without it), then the
        // rows read in order
        const size_t rb = embd.rb;
        if (stg.on && stg.cap >= rb) {
            // (in staging-buffer pieces: a 16K chunk's Q8_0 rows are 71 MB, more than a buffer - maya-ling's)
            const int per = (int) std::min<size_t>((size_t) T, stg.cap / rb);   // (the rows a staging buffer holds)
            for (int t0 = 0; t0 < T; t0 += per) {
                const int n = std::min(per, T - t0);
                CK(cudaEventSynchronize(stg.ev[stg.k]));
                uint8_t* pin = stg.pin[stg.k];
                pool->run((n + 255) / 256, [&](int j) {
                    for (int t = j * 256; t < std::min(n, j * 256 + 256); ++t)
                        std::memcpy(pin + (size_t) t * rb, embd.w + (size_t) ids[t0 + t] * rb, rb);
                });
                CK(cudaMemcpyAsync(pf.Erows + (size_t) t0 * rb, pin, (size_t) n * rb, cudaMemcpyHostToDevice, s));
                CK(cudaEventRecord(stg.ev[stg.k], s));
                stg.k ^= 1;
            }
        } else {
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(pf.Erows + (size_t) t * rb, embd.w + (size_t) ids[t] * rb, rb,
                                   cudaMemcpyHostToDevice, s));
        }
        ew = pf.Erows;
        for (int t = 0; t < T; ++t) pf.h_ids[t] = t;
    } else {
        std::memcpy(pf.h_ids, ids, (size_t) T * sizeof(int));
    }
    CK(cudaMemcpyAsync(pf.ids, pf.h_ids, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, s));
    if (!k::embed_rows(ew, embd.type, embd.rb, pf.ids, T, E, pf.X, s)) {
        std::fprintf(stderr, "maya-ink: no embedding rows for type %d\n", embd.type);
        std::abort();
    }
    k::rmsnorm(pf.X, tok_norm, pf.X, T, E, g.eps, s);
    for (int t = 0; t < T; ++t)
        pf.h_tau[t] = g.log_floor > 0 ? 1.0f + g.log_alpha * std::log(std::max((float) (pos0 + t + 1) /
                                                                                 (float) g.log_floor, 1.0f))
                                      : 1.0f;
    CK(cudaMemcpyAsync(pf.tau, pf.h_tau, (size_t) T * sizeof(float), cudaMemcpyHostToDevice, s));
    static const bool times = std::getenv("INK_PF_TIMES") != nullptr;
    if (times) CK(cudaStreamSynchronize(s));
    double tl0 = now_ms();
    const float c5 = 1.0f / std::sqrt((float) (kc + 1));
    const size_t bstride = (size_t) pf.C * E;
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        float* cv = l.conv;
        float *st_k = cv, *st_v = cv + (size_t) (kc - 1) * kvw, *st_a = cv + (size_t) (kc - 1) * 2 * kvw,
              *st_m = cv + (size_t) (kc - 1) * (2 * kvw + E);
        const int nb = (il + g.res_block - 1) / g.res_block;   // the blocks finished before this layer
        const bool boundary = il % g.res_block == 0;
        // an attention input (query qi of the residual's), normed by nw: pf.A (F32) and pf.Ah (F16)
        auto input = [&](int qi, const float* nw) {
            if (il == 0) {
                k::rmsnorm(pf.X, nw, pf.A, T, E, g.eps, s);
            } else {
                k::attn_res(nullptr, pf.BLK, bstride, nb, pf.X, l.res_q + (size_t) qi * E, 1, pf.A, 0, T, E, s);
                k::rmsnorm(pf.A, nw, pf.A, T, E, g.eps, s);
            }
            k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        };
        // attention
        input(0, l.swa ? l.attn_norm : l.in_norm[0]);
        lin(l.wq, pf.Ah, T, pf.Q);
        lin(l.wr, pf.Ah, T, pf.R);
        lin(l.wgate, pf.Ah, T, pf.GATE);
        lin32(l.wsink, pf.A, T, pf.SINK);
        if (l.swa) {
            lin(l.wk, pf.Ah, T, pf.Kf);
            lin(l.wv, pf.Ah, T, pf.Vf);
            k::sconv2_chunk(pf.Kf, pf.Kc, st_k, l.sc_k, T, kvw, kc, c5, s);
            k::sconv_state(pf.Kf, st_k, T, kvw, kc, s);
            k::sconv2_chunk(pf.Vf, pf.Vc, st_v, l.sc_v, T, kvw, kc, c5, s);
            k::sconv_state(pf.Vf, st_v, T, kvw, kc, s);
        } else {
            input(1, l.in_norm[1]);
            lin(l.wk, pf.Ah, T, pf.Kf);
            lin32(l.kproj, pf.A, T, pf.DYN);
            k::dsconv(pf.Kf, pf.Kc, st_k, l.sc_k_stat, pf.DYN, T, kvw, Hkv, kc, s);
            k::sconv_state(pf.Kf, st_k, T, kvw, kc, s);
            input(2, l.in_norm[2]);
            lin(l.wv, pf.Ah, T, pf.Vf);
            lin32(l.vproj, pf.A, T, pf.DYN);
            k::dsconv(pf.Vf, pf.Vc, st_v, l.sc_v_stat, pf.DYN, T, kvw, Hkv, kc, s);
            k::sconv_state(pf.Vf, st_v, T, kvw, kc, s);
        }
        const float* tl = (!l.swa && g.log_floor > 0) ? pf.tau : nullptr;
        k::head_clip(pf.Q, l.q_clip, g.qkclip, tl, T, H, hd, s);
        k::head_clip(pf.Kc, nullptr, g.qkclip, nullptr, T, Hkv, hd, s);
        k::head_unorm(pf.Vc, T, Hkv, hd, g.eps, s);
        k::head_clip(pf.R, l.rel_clip, g.rclip, tl, T, H, g.d_rel, s);
        k::q_group(pf.Q, pf.Qh, T, H, Hkv, hd, s);
        float* AO = pf.Q;   // (free once q_group has its copy)
        if (!l.swa) {
            k::to_f16(pf.Kc, l.K + (size_t) pos0 * kvw, (size_t) T * kvw, s);
            k::to_f16(pf.Vc, l.V + (size_t) pos0 * kvw, (size_t) T * kvw, s);
            pf_attention(l, T, pos0, l.K, l.V, 0, pos0 + T, AO);
        } else {
            // the ring's last n_swa - 1 positions before the chunk, then the chunk's own, in a window; the ring then
            // keeps the last n_swa positions
            const int nprev = std::min(pos0, g.n_swa - 1), kp0 = pos0 - nprev, nk = nprev + T;
            k::ring_to_window(l.K, pf.Kw, kp0, nprev, g.n_swa, kvw, s);
            k::ring_to_window(l.V, pf.Vw, kp0, nprev, g.n_swa, kvw, s);
            k::to_f16(pf.Kc, pf.Kw + (size_t) nprev * kvw, (size_t) T * kvw, s);
            k::to_f16(pf.Vc, pf.Vw + (size_t) nprev * kvw, (size_t) T * kvw, s);
            pf_attention(l, T, pos0, pf.Kw, pf.Vw, kp0, nk, AO);
            const int j0 = std::max(nprev, nk - g.n_swa);
            k::window_to_ring(pf.Kw, l.K, kp0, j0, nk, g.n_swa, kvw, s);
            k::window_to_ring(pf.Vw, l.V, kp0, j0, nk, g.n_swa, kvw, s);
        }
        k::gate_mul(AO, pf.GATE, (size_t) T * qw, s);
        k::to_f16(AO, pf.AOh, (size_t) T * qw, s);
        lin(l.wo, pf.AOh, T, pf.Y);
        k::sconv2_chunk(pf.Y, pf.Ye, st_a, l.sc_attn, T, E, kc, c5, s);
        k::sconv_state(pf.Y, st_a, T, E, kc, s);
        if (boundary) {
            k::unorm(pf.X, nullptr, pf.BLK + (size_t) nb * bstride, T, E, 1e-6f, s);
            CK(cudaMemcpyAsync(pf.RES, pf.Ye, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        } else {
            k::add2(pf.RES, pf.X, pf.Ye, (size_t) T * E, s);
        }
        if (times) {
            CK(cudaStreamSynchronize(s));
            const double t1 = now_ms();
            pf.attn_ms += t1 - tl0;
            tl0 = t1;
        }
        // the FFN
        k::attn_res(nullptr, pf.BLK, bstride, nb + (boundary ? 1 : 0), pf.RES, l.ffn_q, 1, pf.A, 0, T, E, s);
        k::rmsnorm(pf.A, l.ffn_norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        pf_moe(l, T);
        k::sconv2_chunk(pf.Y, pf.Ye, st_m, l.sc_mlp, T, E, kc, c5, s);
        k::sconv_state(pf.Y, st_m, T, E, kc, s);
        k::add2(pf.X, pf.RES, pf.Ye, (size_t) T * E, s);
        if (times) {
            CK(cudaStreamSynchronize(s));
            const double t1 = now_ms();
            pf.moe_ms += t1 - tl0;
            tl0 = t1;
        }
    }
    if (times)
        std::fprintf(stderr,
                     "maya-ink: prompt path so far: attention %.0f ms, FFN %.0f ms (blocked in copies %.0f ms); %llu "
                     "page faults, working set %.1f GB\n",
                     pf.attn_ms, pf.moe_ms, pf.copy_ms, page_faults(), working_set_gb());
    pos = pos0 + T;
    if (want_logits) {   // the last token's: the output's residual over its blocks and x
        k::attn_res(nullptr, pf.BLK + (size_t) (T - 1) * E, bstride, n_blocks(), pf.X + (size_t) (T - 1) * E, out_q, 1,
                    a, 0, 1, E, s);
        k::rmsnorm(a, out_norm, a, 1, E, g.eps, s);
        if (is_float(output.type)) {
            k::gemv_f(output.w, output.type, output.rb, a, logits, output.n, output.k, s);
        } else {
            quant(a, 1, E);
            mv(output, a, 0, logits, g.logit_scale);
        }
        k::softcap(logits, g.vocab, g.lcap, s);
        CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    CK(cudaStreamSynchronize(s));
    if (want_logits)
        for (int i = g.vocab_unpadded; i < g.vocab; ++i) h_logits[i] = -INFINITY;
}

const float* Engine::Impl::prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx) {
    if (n <= 0 || pos + n > max_ctx) return nullptr;
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= g.vocab) return nullptr;
    const double t0 = now_ms();
    const float* lg = nullptr;
    // a short prompt (a turn's few new tokens): the token path's experts on the CPU beat copying the ones it chooses
    int min_chunk = 48;
    if (const char* v = std::getenv("INK_MIN_CHUNK")) min_chunk = std::max(1, std::atoi(v));
    if (n < min_chunk) {
        for (int i = 0; i < n; ++i) lg = step(ids[i]);
        if (progress) progress(ctx, n);
        return lg;
    }
    // the lent slots back to their buffers until the next token is written (a long prompt comes in pieces)
    ec_reclaim();
    pf_open = true;
    pf_tokens_open += n;
    for (int c0 = 0; c0 < n; c0 += pf.C) {
        const int T = std::min(pf.C, n - c0);
        chunk(ids + c0, T, c0 + T == n);
        if (progress) progress(ctx, c0 + T);
    }
    // the prompt's experts as the next tokens' best guess: the VRAM cache refilled from the scores
    if (route_log) {
        std::fputc('E', route_log);
        std::fflush(route_log);
    }
    ec_fill();
    CK(cudaStreamSynchronize(s));
    st.pf_ms += now_ms() - t0;
    st.pf_tokens += n;
    st.pf_copy_gb = pf.copy_bytes / 1e9;
    return h_logits;
}

// ------------------------------------------------------------------------------------------------ the token path
// A layer's attention for the token at *dpos (the position and tau read on the device, so the same launches serve
// every position: a CUDA graph's), x += it; prev: the MoE layer before, whose end (tok_tail's) comes first.  Fused
// (Impl::fuse's bits, INK_FUSE; the same bits of output either way): 1 the norm and its int8 rows in one kernel (after
// the previous end's / the output's convolution, kept over many blocks), the FFN's norm (f) and f's int8 rows made at
// the end; 2 k / v's convolutions, k's norm and their store in one; 4 q's norm and the bias band inside the attention;
// 8 the output's int8 rows in the attention's combine.
void Engine::Impl::tok_attention(Layer& l, const Layer* prev) {
    const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, kc = g.conv_k;
    const int il = (int) (&l - L.data());
    float* cv = l.conv;
    float *st_k = cv, *st_v = cv + (size_t) (kc - 1) * kvw, *st_a = cv + (size_t) (kc - 1) * 2 * kvw;
    const float* t = (!l.swa && g.log_floor > 0) ? tau : nullptr;
    const int ring = l.swa ? g.n_swa : 0;
    const float c5 = 1.0f / std::sqrt((float) (kc + 1));
    if (prev) tok_tail(*prev);
    const int nb = (il + g.res_block - 1) / g.res_block;
    // the normed inputs in a3: row 0 for q, r, the gate and the sinks (and k, v on a sliding layer), rows 1 / 2 for k / v
    // on a global one
    const int nr = (!l.swa && il > 0) ? 3 : 1;
    if (il == 0) {
        k::rmsnorm(x, l.swa ? l.attn_norm : l.in_norm[0], a3, 1, E, g.eps, s);
    } else {
        k::attn_res_tok(blk, nb, x, l.res_q, l.nq, mix, (size_t) E, E, ars, s);
        if (l.swa) k::rmsnorm(mix, l.attn_norm, a3, 1, E, g.eps, s);
        else
            for (int i = 0; i < 3; ++i) k::rmsnorm(mix + (size_t) i * E, l.in_norm[i], a3 + (size_t) i * E, 1, E, g.eps, s);
    }
    quant(a3, nr, E);
    const int rk = nr == 3 ? 1 : 0, rv = nr == 3 ? 2 : 0;
    {
        const DMat* ms[5] = {&l.wq, &l.wk, &l.wv, &l.wr, &l.wgate};
        const int rows[5] = {0, rk, rv, 0, 0};
        float* ys[5] = {q, kk, vv, r, gt};
        mv_jobs(ms, rows, ys, 5, a3);
    }
    mv(l.wsink, a3, 0, snk);
    float *kp = kk, *vp = vv;
    if (l.swa) {
        k::sconv2(kk, st_k, l.sc_k, 1, kvw, kc, c5, s);
        k::sconv2(vv, st_v, l.sc_v, 1, kvw, kc, c5, s);
    } else {
        mv(l.kproj, a3, rk, dyk);
        mv(l.vproj, a3, rv, dyv);
        k::dsconv(kk, k2, st_k, l.sc_k_stat, dyk, 1, kvw, Hkv, kc, s);
        k::sconv_state(kk, st_k, 1, kvw, kc, s);
        k::dsconv(vv, v2, st_v, l.sc_v_stat, dyv, 1, kvw, Hkv, kc, s);
        k::sconv_state(vv, st_v, 1, kvw, kc, s);
        kp = k2;
        vp = v2;
    }
    k::head_clip(q, l.q_clip, g.qkclip, t, 1, H, hd, s);
    k::head_clip(kp, nullptr, g.qkclip, nullptr, 1, Hkv, hd, s);
    k::head_unorm(vp, 1, Hkv, hd, g.eps, s);
    k::head_clip(r, l.rel_clip, g.rclip, t, 1, H, g.d_rel, s);
    k::rel_logits(r, l.rel_proj, nullptr, rel, 1, H, g.d_rel, l.E, s);
    k::kv_store_dev(kp, vp, l.K, l.V, dpos, ring, kvw, s);
    k::attn_decode2(q, l.K, l.V, l.kv_rows, dpos, ring, rel, l.E, H, Hkv, hd, 1.0f / std::sqrt((float) hd), snk, gt,
                    o_, xq, xd, attn_scr, s);
    mv(l.wo, o_, 0, ao);
    k::sconv2(ao, st_a, l.sc_attn, 1, E, kc, c5, s);
    const bool boundary = il % g.res_block == 0;
    if (boundary) {
        k::unorm(x, blk + (size_t) nb * E, nullptr, 1, E, 1e-6f, s);
        CK(cudaMemcpyAsync(res, ao, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s));
    } else {
        k::add2(res, x, ao, (size_t) E, s);
    }
    k::attn_res_tok(blk, nb + (boundary ? 1 : 0), res, l.ffn_q, 1, mix, (size_t) E, E, ars, s);
    k::rmsnorm(mix, l.ffn_norm, f, 1, E, g.eps, s);
    quant(f, 1, E);
}

// A layer's router: the logits of f (made by tok_attention, F32 rows), f and the logits to the host.
void Engine::Impl::tok_router(const Layer& l) {
    const int E = g.n_embd;
    mv(l.gate_inp, f, 0, rlog);
    CK(cudaMemcpyAsync(h_f, f, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_rlog, rlog, (size_t) (g.n_expert + g.n_shexp) * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// A layer's end: the CPU lane's sum and which lanes ran (h_tail, written by the host) up to the device, then
// x = res + (moe + conv(moe)) / sqrt(5).
void Engine::Impl::tok_tail(const Layer& l) {
    const int E = g.n_embd, kc = g.conv_k;
    float* st_m = l.conv + (size_t) (kc - 1) * (2 * kvw + E);
    CK(cudaMemcpyAsync(d_tail, h_tail, tail_bytes(), cudaMemcpyHostToDevice, s));
    k::moe_tail2(x, res, d_tail + 4, ec.routed, (const int*) d_tail, st_m, l.sc_mlp, E, kc,
                 1.0f / std::sqrt((float) (kc + 1)), s);
}

// The last layer's end and the output head: the output's residual over the blocks, its norm, the head, the softcap; the
// logits to the host.
void Engine::Impl::tok_head(const Layer& last) {
    const int E = g.n_embd;
    tok_tail(last);
    k::attn_res_tok(blk, n_blocks(), x, out_q, 1, a, (size_t) E, E, ars, s);
    k::rmsnorm(a, out_norm, a, 1, E, g.eps, s);
    if (is_float(output.type)) {
        k::gemv_f(output.w, output.type, output.rb, a, logits, output.n, output.k, s);
    } else {
        quant(a, 1, E);
        mv(output, a, 0, logits, g.logit_scale);
    }
    k::softcap(logits, g.vocab, g.lcap, s);
    CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// The CUDA graphs: a MoE layer's from its sync back to the next (the previous layer's end when that one is a MoE
// layer, the attention, the router), and the head's (the last layer's end, the head): two launches a layer become one,
// and the GPU runs them without the launches' gaps.  Captured on the first token; INK_GRAPHS=0: none.
void Engine::Impl::tok_graphs() {
    graphs = 0;
    const char* v = std::getenv("INK_GRAPHS");
    if ((v && std::atoi(v) == 0) || std::getenv("INK_TRACE")) return;
    gx.assign((size_t) g.n_layer, nullptr);
    auto capture = [&](auto&& body, cudaGraphExec_t& out) {
        cudaGraph_t gr = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        body();
        if (cudaStreamEndCapture(s, &gr) != cudaSuccess || !gr) return false;
        const bool ok = cudaGraphInstantiate(&out, gr, 0) == cudaSuccess;
        cudaGraphDestroy(gr);
        return ok;
    };
    bool ok = true;
    for (int il = g.n_dense; il < g.n_layer && ok; ++il) {
        Layer& l = L[(size_t) il];
        ok = capture(
            [&] {
                tok_attention(l, il > g.n_dense ? &L[(size_t) il - 1] : nullptr);
                tok_router(l);
            },
            gx[(size_t) il]);
    }
    ok = ok && capture([&] { tok_head(L[(size_t) g.n_layer - 1]); }, gx_head);
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "maya-ink: the token path's CUDA graphs could not be made; it launches its kernels one by one\n");
        for (auto& e : gx)
            if (e) cudaGraphExecDestroy(e);
        gx.clear();
        if (gx_head) cudaGraphExecDestroy(gx_head);
        gx_head = nullptr;
        return;
    }
    graphs = 1;
}

const float* Engine::Impl::step(int token) {
    if (pos >= max_ctx || token < 0 || token >= g.vocab) return nullptr;
    const double t0 = now_ms();
    double cpu_ms = 0.0;
    const int E = g.n_embd, kc = g.conv_k;
    static const bool ptimes = std::getenv("INK_STEP_TIMES") != nullptr;
    double tp = t0;   // (INK_STEP_TIMES: the last phase's end)
    auto phase = [&](double& acc) {
        if (!ptimes) return;
        const double t = now_ms();
        acc += t - tp;
        tp = t;
    };
    if (pf_open) {   // the first token after a prompt: the prompt path's buffers lent to the cache again, and filled
        pf_open = false;
        if (ec.arena_end > ec.own) {
            ec.lent = true;
            ec_fill();
        }
        // The prompt changed the scores, and the main cache with them: the helper follows, with more copies after a
        // longer prompt (an eighth of its tokens, 64 .. fill_cap).  INK_HELPER_PROMPT_BG=1 (strata-mimo's
        // MIMO_HELPER_PROMPT_BG): the copies go to the swaps' thread, so the first token doesn't wait for them; the
        // helper's slots being refilled sit out until their copy lands.
        static const bool prompt_bg = std::getenv("INK_HELPER_PROMPT_BG") && std::atoi(std::getenv("INK_HELPER_PROMPT_BG")) != 0;
        hc_update(std::min(hc.fill_cap, std::max(64, (int) std::min<long>(pf_tokens_open / 8, INT_MAX))), false,
                  prompt_bg && hc.bg);
        pf_tokens_open = 0;
    }
    if (graphs < 0) tok_graphs();
    ec_kv_reserve(pos + 1);
    // (INK_HELPER_KEEPALIVE) the helper's keep-alive kernel again once the last one has ended
    if (hc.ks && cudaStreamQuery(hc.ks) == cudaSuccess) {
        OnDev od(hc.dev);
        k::keepalive(hc.ka_ns, hc.ks);
        (void) cudaStreamQuery(hc.ks);   // (submitted now: WDDM holds launches until something does)
    }

    // the token's row, normed
    const uint8_t* erow = embd.w + (size_t) token * embd.rb;
    if (embd_host) {
        std::memcpy(h_erow, erow, embd.rb);
        CK(cudaMemcpyAsync(d_erow, h_erow, embd.rb, cudaMemcpyHostToDevice, s));
        erow = d_erow;
    }
    if (is_float(embd.type)) {
        k::row_to_f32(erow, embd.type, x, E, s);
    } else {
        if (!iq::f16_rows(embd.type, erow, E, 1, e16, E, s)) {
            std::fprintf(stderr, "maya-ink: no row decoder for the embedding's type %d\n", embd.type);
            std::abort();
        }
        k::f16_to_f32((const __half*) e16, x, E, s);
    }
    k::rmsnorm(x, tok_norm, x, 1, E, g.eps, s);
    // the global layers' log-N scale, and the position, on the device
    h_tokd[0] = g.log_floor > 0 ? 1.0f + g.log_alpha * std::log(std::max((float) (pos + 1) / (float) g.log_floor, 1.0f))
                                : 1.0f;
    std::memcpy(h_tokd + 1, &pos, sizeof(int));
    CK(cudaMemcpyAsync(tokd, h_tokd, 2 * sizeof(float), cudaMemcpyHostToDevice, s));
    phase(pt.embed);

    static const bool trace = std::getenv("INK_TRACE") != nullptr;
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] pos %d layer %d\n", pos, il);
            std::fflush(stderr);
        }
        if (l.dense) {
            float* st_m = l.conv + (size_t) (kc - 1) * (2 * kvw + E);
            tok_attention(l, nullptr);
            if (!(fuse & 1)) {   // (fused: f and its int8 rows made at the attention's end)
                k::rmsnorm(x, l.ffn_norm, f, 1, E, g.eps, s);
                quant(f, 1, E);
            }
            mv(l.gate, f, 0, fg);
            mv(l.up, f, 0, fu);
            k::swiglu(fg, fu, fg, g.n_ff, nullptr, s);
            quant(fg, 1, g.n_ff);
            mv(l.down, fg, 0, fo);
            k::mul_scalar(fo, l.gscale_d, E, s);
            k::shortconv(fo, st_m, l.sc_mlp, 1, E, kc, s);
            k::add(x, fo, E, s);
            phase(pt.dense);
            continue;
        }
        // (the previous layer's end,) the attention and the router: one graph, or their launches
        const bool ev = ptimes && il == 10;
        if (ev) {
            if (!pt.e0) {
                CK(cudaEventCreate(&pt.e0));
                CK(cudaEventCreate(&pt.e1));
            }
            CK(cudaEventRecord(pt.e0, s));
        }
        if (graphs > 0) {
            CK(cudaGraphLaunch(gx[(size_t) il], s));
        } else {
            tok_attention(l, il > g.n_dense ? &L[(size_t) il - 1] : nullptr);
            tok_router(l);
        }
        const int ne = g.n_expert, nsh = g.n_shexp, ff = g.n_ff_exp;
        if (ev) CK(cudaEventRecord(pt.e1, s));
        const bool staging = ec_stage_begin(ec.piece);   // (a piece of the swaps' copies while the GPU works)
        CK(cudaStreamSynchronize(s));
        if (staging) ec_stage_end();
        if (ev) {
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, pt.e0, pt.e1));
            pt.gpu_att += ms;
        }
        phase(pt.att);
        const double c0 = now_ms();
        // routing: the top n_used of sigmoid(logit) + bias, ties to the lower id
        int sel[16];
        {
            float sc[512];
            int idx[512];
            for (int e = 0; e < ne; ++e) {
                sc[e] = 1.0f / (1.0f + std::exp(-h_rlog[e])) + l.exp_bias[(size_t) e];
                idx[e] = e;
            }
            std::partial_sort(idx, idx + g.n_used, idx + ne,
                              [&](int a1, int b1) { return sc[a1] > sc[b1] || (sc[a1] == sc[b1] && a1 < b1); });
            for (int i = 0; i < g.n_used; ++i) sel[i] = idx[i];
        }
        // weights: the chosen sigmoid scores over their sum, x scale
        float wv[18];
        float sum = 0.0f;
        for (int i = 0; i < g.n_used; ++i) {
            wv[i] = 1.0f / (1.0f + std::exp(-h_rlog[sel[i]]));
            sum += wv[i];
        }
        for (int i = 0; i < g.n_used; ++i) wv[i] = wv[i] / (sum + 1e-20f) * g.exp_scale;
        phase(pt.route);
        // the chosen experts in VRAM go to the GPU, the others to the CPU lane, both at once.  INK_TAIL_KEEP=k (off by
        // default; strata-mimo's MIMO_TAIL_KEEP): a chosen expert that neither card holds and that ranks k-th or lower
        // (sel[] is in rank order, sigmoid + bias) is left out, its weight with it (the others keep theirs; with
        // INK_TAIL_RENORM=1 they share it).  INK_TAIL_W=w (Maya's MAYA_GLM_MISS_SKIP_W): also one whose weight is under
        // w of the token's whole (8 experts: 1/8 = 0.125 is the even share).  Only in layers INK_TAIL_FROM <= il <
        // INK_TAIL_UPTO.  Its score still counts, so the caches keep learning it.
        static const float tail_w = std::getenv("INK_TAIL_W") ? (float) std::atof(std::getenv("INK_TAIL_W")) : 0.0f;
        static const int tail_keep = std::getenv("INK_TAIL_KEEP") ? std::atoi(std::getenv("INK_TAIL_KEEP")) : 0;
        static const int tail_from = std::getenv("INK_TAIL_FROM") ? std::atoi(std::getenv("INK_TAIL_FROM")) : 0;
        static const int tail_upto = std::getenv("INK_TAIL_UPTO") ? std::atoi(std::getenv("INK_TAIL_UPTO")) : 1 << 30;
        static const bool tail_renorm = std::getenv("INK_TAIL_RENORM") && std::atoi(std::getenv("INK_TAIL_RENORM")) != 0;
        const bool tail_here = ((tail_keep > 0 && tail_keep < g.n_used) || tail_w > 0.0f) && il >= tail_from && il < tail_upto;
        int gslot[16], csel[16], hslot[16];
        float gw[16], cw[16], hw[16];
        int ng = 0, nc = 0, nh = 0;
        float dropped = 0.0f;
        if (route_log)
            for (int i = 0; i < g.n_used; ++i) route_tok.push_back((uint16_t) sel[i]);
        for (int i = 0; i < g.n_used; ++i) {
            const size_t key = (size_t) il * ne + sel[i];
            ec.score[key] += 1.0f;
            const int sl = ec.slots ? ec.where[key] : -1;
            const int hs = sl < 0 && hc.slots ? hc.where[key] : -1;
            if (sl >= 0) {
                gslot[ng] = sl;
                gw[ng++] = wv[i];
            } else if (hs >= 0) {
                hslot[nh] = hs;
                hw[nh++] = wv[i];
            } else if (tail_here && ((tail_keep > 0 && i >= tail_keep) || wv[i] < tail_w * g.exp_scale)) {
                dropped += wv[i];
                ++st.tail_skips;
            } else {
                csel[nc] = sel[i];
                cw[nc++] = wv[i];
            }
        }
        if (tail_renorm && dropped > 0.0f) {
            const float kept = g.exp_scale - dropped, fr = kept > 0.0f ? g.exp_scale / kept : 1.0f;
            for (int i = 0; i < ng; ++i) gw[i] *= fr;
            for (int i = 0; i < nh; ++i) hw[i] *= fr;
            for (int i = 0; i < nc; ++i) cw[i] *= fr;
        }
        ec.lookups += g.n_used;
        ec.hits += ng;
        hc.hits += nh;
        // the CPU lane's workers start on the rest now; this thread launches the GPUs' parts, then joins them
        static const bool early = !std::getenv("INK_LANE_EARLY") || std::atoi(std::getenv("INK_LANE_EARLY")) != 0;
        float* h_routed = h_tail + 4;
        if (nc && early) lane.begin(*pool, l.cl, E, ff, csel, cw, nc, h_f, h_routed);
        if (nh) hc_experts(l, hslot, hw, nh);   // (the helper card starts on its share first: it is the slower one)
        if (ng) gpu_experts(l, gslot, gw, ng);
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] layer %d: experts %d %d %d %d %d %d, %d on the GPU; CPU lane next\n", il,
                         sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], ng);
            std::fflush(stderr);
        }
        phase(pt.glaunch);
        if (nc && !early) lane.begin(*pool, l.cl, E, ff, csel, cw, nc, h_f, h_routed);
        if (nc) lane.end();
        const double c1 = now_ms();
        st.ms_lane += c1 - c0;
        st.lane_layers += nc > 0;
        st.helper_layers += nh > 0;
        if (nh) {   // the helper's sum joins the CPU lane's on the host (the device adds that to its own at the layer's end)
            {
                OnDev od(hc.dev);
                CK(cudaStreamSynchronize(hc.s));
            }
            if (nc)
                for (int i = 0; i < E; ++i) h_routed[i] += hc.h_out[i];
            else std::memcpy(h_routed, hc.h_out, (size_t) E * sizeof(float));
            st.ms_helper += now_ms() - c1;
        }
        cpu_ms += now_ms() - c0;
        phase(pt.cpu);
        pt.cpu_experts += nc;
        // which lanes ran, for this layer's end (the next graph's start)
        int* flags = (int*) h_tail;
        flags[0] = nc > 0 || nh > 0;
        flags[1] = ng > 0;
        phase(pt.tail);
    }

    // the last layer's end and the output head
    if (graphs > 0) {
        CK(cudaGraphLaunch(gx_head, s));
    } else {
        tok_head(L[(size_t) g.n_layer - 1]);
    }
    const bool staging = ec_stage_begin(4 * ec.piece);   // (the head's GEMV is the token's longest wait)
    CK(cudaStreamSynchronize(s));
    if (staging) ec_stage_end();
    for (int i = g.vocab_unpadded; i < g.vocab; ++i) h_logits[i] = -INFINITY;
    ++pos;
    if (route_log) {
        std::fputc('T', route_log);
        std::fwrite(route_tok.data(), sizeof(uint16_t), route_tok.size(), route_log);
        route_tok.clear();
        if (pos % 64 == 0) std::fflush(route_log);
    }
    phase(pt.head);
    // the scores decay as tokens are written; a few swaps every few tokens (the copies that went up during this token
    // in use from the next)
    if (ec.slots) {
        ec_arrive();
        for (float& v : ec.score) v *= ec.decay;
        if (++ec.tok % ec.every == 0) ec_adapt();
    }
    if (hc.slots && ++hc.tok % hc.every == 0) {
        const double u0 = now_ms();
        hc_update(hc.per, true, hc.bg);
        st.ms_update += now_ms() - u0;
    }
    phase(pt.adapt);
    st.ms_total += now_ms() - t0;
    st.ms_cpu += cpu_ms;
    ++st.tokens;
    if (ptimes && ++pt.tokens == 64) {
        const double n = (double) pt.tokens;
        std::fprintf(stderr,
                     "maya-ink: step phases, ms a token (64 tokens): embed %.2f, attention+sync %.2f, routing %.2f, GPU "
                     "launches %.2f, CPU lane %.2f (%.1f experts, %.3f ms each), tail %.2f, dense layers %.2f, head %.2f, "
                     "adapt %.2f; total %.2f; layer 10's attention on the GPU %.0f us\n",
                     pt.embed / n, pt.att / n, pt.route / n, pt.glaunch / n, pt.cpu / n, pt.cpu_experts / n,
                     pt.cpu_experts ? pt.cpu / pt.cpu_experts : 0.0, pt.tail / n, pt.dense / n, pt.head / n,
                     pt.adapt / n,
                     (pt.embed + pt.att + pt.route + pt.glaunch + pt.cpu + pt.tail + pt.dense + pt.head + pt.adapt) / n,
                     pt.gpu_att / n * 1000.0);
        const cudaEvent_t e0 = pt.e0, e1 = pt.e1;
        pt = PhaseTimes{};
        pt.e0 = e0;
        pt.e1 = e1;
    }
    return h_logits;
}

}  // namespace ink
