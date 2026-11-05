// src/ink/ink_model.cu - maya-ink's model (ink_model.hpp): the GGUF read, the weights placed, and a token's step
// through Inkling-Small's layers - llama.cpp's graph (src/models/inkling.cpp in ggml-org/llama.cpp#25731) step by step:
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <thread>

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
    float *tok_norm = nullptr, *out_norm = nullptr;
    std::unique_ptr<Pool> pool;
    size_t dev_bytes = 0;

    // device scratch
    float *x = nullptr, *a = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr, *r = nullptr, *rel = nullptr;
    float *o_ = nullptr, *ao = nullptr, *f = nullptr, *fg = nullptr, *fu = nullptr, *fo = nullptr;
    float *routed = nullptr, *sg = nullptr, *su = nullptr, *so = nullptr, *logits = nullptr, *rlog = nullptr;
    float *tau = nullptr, *gamma = nullptr, *attn_scr = nullptr;
    int8_t* xq = nullptr;
    float* xd = nullptr;
    uint16_t* e16 = nullptr;
    // host (pinned)
    float *h_f = nullptr, *h_rlog = nullptr, *h_routed = nullptr, *h_logits = nullptr, *h_tau = nullptr,
          *h_gamma = nullptr;
    Stats st;

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
        uint8_t* gbuf[2] = {nullptr, nullptr};
        cudaEvent_t gready[2] = {nullptr, nullptr}, gfreed[2] = {nullptr, nullptr};
        cublasHandle_t cb = nullptr;
        cudaStream_t cs = nullptr;   // the experts' copies
        int *ids = nullptr, *sel = nullptr, *cnt = nullptr, *off = nullptr, *list = nullptr;
        float *X = nullptr, *A = nullptr, *Q = nullptr, *Kf = nullptr, *Vf = nullptr, *Kc = nullptr, *Vc = nullptr;
        float *R = nullptr, *Y = nullptr, *Ye = nullptr, *GU = nullptr, *RL = nullptr, *wts = nullptr, *wl = nullptr;
        float *S = nullptr, *Ot = nullptr, *tau = nullptr;
        __half *Ah = nullptr, *Qh = nullptr, *AOh = nullptr, *Kw = nullptr, *Vw = nullptr, *Xe = nullptr, *He = nullptr;
        __half *P = nullptr, *W1 = nullptr, *W2 = nullptr;
        int *h_ids = nullptr, *h_cnt = nullptr, *h_off = nullptr;
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
    // Filled after each prompt (the best-scored in) and adapted as tokens are written (a few swaps every few tokens);
    // the scores can be kept in a file (Options::profile) for the next start.
    struct EC {
        int slots = 0, ne = 0;
        size_t slot_bytes = 0;
        uint8_t* base = nullptr;
        std::vector<int> owner, where;   // slot -> key (layer x n_expert + expert) or -1; key -> slot or -1
        std::vector<float> score;
        long lookups = 0, hits = 0, copies = 0, tok = 0;
        float decay = 0.998f, margin = 1.25f;
        int every = 8, per = 4;
        float *gu = nullptr, *h = nullptr, *y = nullptr, *routed = nullptr, *hd = nullptr;
        int8_t* hq = nullptr;
    } ec;
    void ec_alloc();
    void ec_copy(int key, int slot);
    void ec_fill();
    void ec_adapt();
    const uint8_t* ec_blob(int il, int e) const {
        const int sl = ec.slots ? ec.where[(size_t) il * ec.ne + e] : -1;
        return sl >= 0 ? ec.base + (size_t) sl * ec.slot_bytes : nullptr;
    }
    void gpu_experts(const Layer& l, const int* slot, const float* w, int n);
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
    return r;
}
void Engine::clear_stats() {
    p->st = Stats{};
    p->ec.lookups = p->ec.hits = 0;
}
bool Engine::save_profile() const { return !p->o.profile.empty() && p->save_profile(p->o.profile); }
bool Engine::load(const Options& o, std::string& err) { return p->load(o, err); }
const float* Engine::step(int token) { return p->step(token); }
const float* Engine::prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx) {
    return p->prefill(ids, n, progress, ctx);
}

void Engine::reset() {
    p->pos = 0;
    const int km1 = p->g.conv_k - 1;
    const size_t conv = (size_t) km1 * (2 * p->kvw + 2 * p->g.n_embd);
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
    if (!arch || arch->s != "inkling") {
        err = "not an Inkling GGUF (general.architecture " + (arch ? arch->s : std::string("missing")) + ")";
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
    const std::string A = "inkling.";
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
    g.n_dense = meta_int(f0, A + "dense_block_count");
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
        g.conv_k < 2 || g.conv_k > 8 || g.n_shexp != 2 || g.n_used > 16 || g.d_rel <= 0) {
        err = "an Inkling shape this engine does not take";
        return false;
    }
    kvw = g.n_head_kv * g.head_dim;
    max_ctx = std::min(o.max_ctx, g.ctx_train > 0 ? g.ctx_train : o.max_ctx);

    CK(cudaSetDevice(o.gpu));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int E = g.n_embd, H = g.n_head, hd = g.head_dim, kc = g.conv_k;

    // the weights in VRAM
    if (!mat("token_embd.weight", E, g.vocab, embd, err) || !mat("output.weight", E, g.vocab, output, err)) return false;
    if (!(tok_norm = f32("token_embd_norm.weight", (size_t) E, err))) return false;
    if (!(out_norm = f32("output_norm.weight", (size_t) E, err))) return false;
    L.resize((size_t) g.n_layer);
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        l.swa = g.swa[(size_t) il] != 0;
        l.dense = il < g.n_dense;
        l.E = l.swa ? g.rel_ext_swa : g.rel_ext;
        std::vector<float> gs;
        if (!(l.attn_norm = f32(b + "attn_norm.weight", (size_t) E, err))) return false;
        if (!(l.q_norm = f32(b + "attn_q_norm.weight", (size_t) hd, err))) return false;
        if (!(l.k_norm = f32(b + "attn_k_norm.weight", (size_t) hd, err))) return false;
        if (!(l.rel_proj = f32(b + "attn_rel_proj.weight", (size_t) g.d_rel * l.E, err))) return false;
        if (!(l.sc_k = f32(b + "shortconv_k.weight", (size_t) kc * kvw, err))) return false;
        if (!(l.sc_v = f32(b + "shortconv_v.weight", (size_t) kc * kvw, err))) return false;
        if (!(l.sc_attn = f32(b + "shortconv_attn.weight", (size_t) kc * E, err))) return false;
        if (!(l.sc_mlp = f32(b + "shortconv_mlp.weight", (size_t) kc * E, err))) return false;
        if (!(l.ffn_norm = f32(b + "ffn_norm.weight", (size_t) E, err))) return false;
        if (!(l.gscale_d = f32(b + "ffn_gscale.weight", 1, err, &gs))) return false;
        l.gscale = gs[0];
        if (!mat(b + "attn_q.weight", E, H * hd, l.wq, err) || !mat(b + "attn_k.weight", E, kvw, l.wk, err) ||
            !mat(b + "attn_v.weight", E, kvw, l.wv, err) || !mat(b + "attn_r.weight", E, H * g.d_rel, l.wr, err) ||
            !mat(b + "attn_output.weight", H * hd, E, l.wo, err))
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
            if (!bank(b + "ffn_gate_shexp.weight", E, g.n_ff_exp, g.n_shexp, l.sh_gate, err) ||
                !bank(b + "ffn_up_shexp.weight", E, g.n_ff_exp, g.n_shexp, l.sh_up, err) ||
                !bank(b + "ffn_down_shexp.weight", g.n_ff_exp, E, g.n_shexp, l.sh_down, err))
                return false;
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
    routed = dalloc<float>(E);
    sg = dalloc<float>((size_t) 2 * g.n_ff_exp);
    su = dalloc<float>((size_t) 2 * g.n_ff_exp);
    so = dalloc<float>((size_t) 2 * E);
    logits = dalloc<float>((size_t) g.vocab);
    rlog = dalloc<float>((size_t) g.n_expert + g.n_shexp);
    tau = dalloc<float>(1);
    gamma = dalloc<float>(2);
    attn_scr = dalloc<float>(k::attn_scratch_floats(H, hd, max_ctx));
    const int kmax = std::max(g.n_ff, 2 * g.n_ff_exp) * 2;
    xq = dalloc<int8_t>((size_t) kmax);
    xd = dalloc<float>((size_t) kmax / 32);
    e16 = dalloc<uint16_t>(E);
    h_f = halloc<float>(E);
    h_rlog = halloc<float>((size_t) g.n_expert + g.n_shexp);
    h_routed = halloc<float>(E);
    h_logits = halloc<float>((size_t) g.vocab);
    h_tau = halloc<float>(1);
    h_gamma = halloc<float>(2);

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
    if (o.verbose)
        std::fprintf(stderr,
                     "maya-ink: Inkling %d layers, %d experts (top %d + %d shared), context %d; %.2f GB in VRAM (the "
                     "prompt path's chunks of %d tokens %.2f of it, %d expert slots %.2f), the routed experts in the "
                     "GGUF's mapping; CPU lane %d threads; loaded in %.1f s\n",
                     g.n_layer, g.n_expert, g.n_used, g.n_shexp, max_ctx, dev_bytes / 1e9, pf.C,
                     (before_ec - before_pf) / 1e9, ec.slots, (dev_bytes - before_ec) / 1e9, pool->threads(),
                     (now_ms() - t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert cache
void Engine::Impl::ec_alloc() {
    const int E = g.n_embd, ff = g.n_ff_exp;
    ec.ne = g.n_expert;
    ec.slot_bytes = (pf.slot_bytes + 255) & ~(size_t) 255;
    ec.where.assign((size_t) g.n_layer * ec.ne, -1);
    ec.score.assign((size_t) g.n_layer * ec.ne, 0.0f);
    ec.gu = dalloc<float>((size_t) 16 * 2 * ff);
    ec.h = dalloc<float>((size_t) 16 * ff);
    ec.hq = dalloc<int8_t>((size_t) 16 * ff);
    ec.hd = dalloc<float>((size_t) 16 * ff / 32);
    ec.y = dalloc<float>((size_t) 16 * E);
    ec.routed = dalloc<float>(E);
    if (const char* v = std::getenv("INK_CACHE_DECAY")) ec.decay = (float) std::atof(v);
    if (const char* v = std::getenv("INK_CACHE_EVERY")) ec.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("INK_CACHE_SWAPS")) ec.per = std::max(0, std::atoi(v));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t reserve = (size_t) 1536 << 20;
    if (const char* v = std::getenv("INK_VRAM_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
    size_t want = fr > reserve ? fr - reserve : 0;
    if (o.cache_gb >= 0.0) want = std::min(want, (size_t) (o.cache_gb * 1e9));
    int n = (int) (want / ec.slot_bytes);
    while (n > 0) {   // (one allocation; a smaller one if the free memory is in pieces)
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) n * ec.slot_bytes) == cudaSuccess) {
            ec.base = (uint8_t*) p;
            dev_bytes += (size_t) n * ec.slot_bytes;
            break;
        }
        cudaGetLastError();
        n = n * 15 / 16;
    }
    ec.slots = ec.base ? n : 0;
    ec.owner.assign((size_t) ec.slots, -1);
}

void Engine::Impl::ec_copy(int key, int slot) {
    const int il = key / ec.ne, e = key % ec.ne, E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    uint8_t* d = ec.base + (size_t) slot * ec.slot_bytes;
    if (ec.owner[(size_t) slot] >= 0) ec.where[(size_t) ec.owner[(size_t) slot]] = -1;
    CK(cudaMemcpyAsync(d, c.gate + (size_t) e * gb, gb, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d + gb, c.up + (size_t) e * ub, ub, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d + gb + ub, c.down + (size_t) e * db, db, cudaMemcpyHostToDevice, s));
    ec.owner[(size_t) slot] = key;
    ec.where[(size_t) key] = slot;
    ++ec.copies;
}

void Engine::Impl::ec_fill() {
    if (ec.slots == 0) return;
    std::vector<int> keys;
    for (int il = g.n_dense; il < g.n_layer; ++il)
        for (int e = 0; e < ec.ne; ++e)
            if (ec.score[(size_t) il * ec.ne + e] > 0.0f) keys.push_back(il * ec.ne + e);
    const size_t top = std::min(keys.size(), (size_t) ec.slots);
    std::partial_sort(keys.begin(), keys.begin() + (long) top, keys.end(), [&](int a1, int b1) {
        return ec.score[(size_t) a1] > ec.score[(size_t) b1] || (ec.score[(size_t) a1] == ec.score[(size_t) b1] && a1 < b1);
    });
    std::vector<char> want(ec.where.size(), 0);
    for (size_t i = 0; i < top; ++i) want[(size_t) keys[i]] = 1;
    std::vector<int> spare;
    for (int sl = ec.slots - 1; sl >= 0; --sl)
        if (ec.owner[(size_t) sl] < 0 || !want[(size_t) ec.owner[(size_t) sl]]) spare.push_back(sl);
    for (size_t i = 0; i < top && !spare.empty(); ++i)
        if (ec.where[(size_t) keys[i]] < 0) {
            ec_copy(keys[i], spare.back());
            spare.pop_back();
        }
}

void Engine::Impl::ec_adapt() {
    if (ec.slots == 0 || ec.per == 0) return;
    // the best-scored experts not in VRAM, the worst-scored in it
    const int per = ec.per;
    std::vector<int> best, worst;
    for (int il = g.n_dense; il < g.n_layer; ++il)
        for (int e = 0; e < ec.ne; ++e) {
            const int key = il * ec.ne + e;
            if (ec.where[(size_t) key] >= 0 || ec.score[(size_t) key] <= 0.0f) continue;
            best.push_back(key);
            std::push_heap(best.begin(), best.end(), [&](int a1, int b1) { return ec.score[(size_t) a1] > ec.score[(size_t) b1]; });
            if ((int) best.size() > per) {
                std::pop_heap(best.begin(), best.end(), [&](int a1, int b1) { return ec.score[(size_t) a1] > ec.score[(size_t) b1]; });
                best.pop_back();
            }
        }
    std::sort(best.begin(), best.end(), [&](int a1, int b1) { return ec.score[(size_t) a1] > ec.score[(size_t) b1]; });
    std::vector<int> free_slots;
    for (int sl = 0; sl < ec.slots; ++sl) {
        const int key = ec.owner[(size_t) sl];
        if (key < 0) free_slots.push_back(sl);
        else worst.push_back(sl);
    }
    std::partial_sort(worst.begin(), worst.begin() + (long) std::min(worst.size(), (size_t) per), worst.end(),
                      [&](int a1, int b1) {
                          return ec.score[(size_t) ec.owner[(size_t) a1]] < ec.score[(size_t) ec.owner[(size_t) b1]];
                      });
    size_t wi = 0;
    for (int key : best) {
        if (!free_slots.empty()) {
            ec_copy(key, free_slots.back());
            free_slots.pop_back();
            continue;
        }
        if (wi >= worst.size() || (int) wi >= per) break;
        const int sl = worst[wi];
        if (ec.score[(size_t) key] <= ec.margin * ec.score[(size_t) ec.owner[(size_t) sl]] + 0.5f) break;
        ec_copy(key, sl);
        ++wi;
    }
}

void Engine::Impl::gpu_experts(const Layer& l, const int* slot, const float* w, int n) {
    const int E = g.n_embd, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    iq::MvJob jobs[32];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = ec.base + (size_t) slot[i] * ec.slot_bytes;
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
    k::swiglu_f32_rows(ec.gu, n, ff, ec.h, s);
    k::q8(ec.h, n, ff, ec.hq, ec.hd, s);
    iq::MvJob dj[16];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = ec.base + (size_t) slot[i] * ec.slot_bytes;
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
    pf.ids = dalloc<int>(C);
    pf.X = dalloc<float>(CE);
    pf.A = dalloc<float>(CE);
    pf.Q = dalloc<float>((size_t) C * qw);
    pf.Kf = dalloc<float>((size_t) C * kvw);
    pf.Vf = dalloc<float>((size_t) C * kvw);
    pf.Kc = dalloc<float>((size_t) C * kvw);
    pf.Vc = dalloc<float>((size_t) C * kvw);
    pf.R = dalloc<float>((size_t) C * H * g.d_rel);
    pf.Y = dalloc<float>(CE);
    pf.Ye = dalloc<float>(CE);
    pf.GU = dalloc<float>((size_t) C * 2 * ffe);
    pf.RL = dalloc<float>((size_t) C * (ne + nsh));
    pf.sel = dalloc<int>((size_t) C * nu);
    pf.wts = dalloc<float>((size_t) C * (nu + nsh));
    pf.cnt = dalloc<int>(ne);
    pf.off = dalloc<int>(ne);
    pf.list = dalloc<int>((size_t) C * nu);
    pf.wl = dalloc<float>((size_t) C * nu);
    pf.tau = dalloc<float>(C);
    pf.Ah = dalloc<__half>(CE);
    pf.Qh = dalloc<__half>((size_t) C * qw);
    pf.AOh = dalloc<__half>((size_t) C * qw);
    pf.Xe = dalloc<__half>(CE);
    pf.He = dalloc<__half>((size_t) C * ffe);
    pf.Kw = dalloc<__half>((size_t) (g.n_swa + C) * kvw);
    pf.Vw = dalloc<__half>((size_t) (g.n_swa + C) * kvw);
    pf.Ot = dalloc<float>((size_t) G * C * hd);
    size_t mib = 960;
    if (const char* v = std::getenv("INK_SCORES_MIB")) mib = (size_t) std::max(64, std::atoi(v));
    pf.s_elems = (mib << 20) / 6;   // S F32 and P F16
    pf.S = dalloc<float>(pf.s_elems);
    pf.P = dalloc<__half>(pf.s_elems);
    pf.w1_elems = std::max({(size_t) 2 * ffd * E, (size_t) 2 * ffe * E, (size_t) qw * E});
    pf.w2_elems = std::max({(size_t) ffd * E, (size_t) ffe * E, (size_t) E * qw});
    pf.W1 = dalloc<__half>(pf.w1_elems);
    pf.W2 = dalloc<__half>(pf.w2_elems);
    for (const Layer& l : L)
        if (!l.dense)
            pf.slot_bytes = std::max(pf.slot_bytes, (size_t) ffe * (l.cl.rg + l.cl.ru) + (size_t) E * l.cl.rd);
    // two group buffers in turn, `stage` experts each (the copy of one group while the other's are multiplied)
    pf.GE = std::max(1, std::min(ne, o.stage));
    for (int i = 0; i < 2; ++i) {
        pf.gbuf[i] = dalloc<uint8_t>((size_t) pf.GE * pf.slot_bytes);
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
    const float scale = 1.0f / (float) hd, one = 1.0f, zero = 0.0f;
    const float* tl = (!l.swa && g.log_floor > 0) ? pf.tau : nullptr;
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
            k::attn_softmax(pf.S, ld, pf.P, ld, M, nk, G, gi, ta, pos0, kp0 + j0, window, pf.R, l.rel_proj, l.E, H,
                            g.d_rel, tl, s);
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
    k::route(pf.RL, ne + nsh, l.exp_bias_d, T, ne, nsh, nu, g.exp_scale * l.gscale, pf.sel, pf.wts, s);
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
    for (int e = 0; e < ne; ++e)
        if (pf.h_cnt[e]) ec.score[(size_t) il * ne + e] += (float) pf.h_cnt[e];
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
            k::swiglu_rows(pf.GU, rows, ff, nullptr, 0, pf.He, s);
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
    std::memcpy(pf.h_ids, ids, (size_t) T * sizeof(int));
    CK(cudaMemcpyAsync(pf.ids, pf.h_ids, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, s));
    if (!k::embed_rows(embd.w, embd.type, embd.rb, pf.ids, T, E, pf.X, s)) {
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
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        float* cv = l.conv;
        float *st_k = cv, *st_v = cv + (size_t) (kc - 1) * kvw, *st_a = cv + (size_t) (kc - 1) * 2 * kvw,
              *st_m = cv + (size_t) (kc - 1) * (2 * kvw + E);
        // attention
        k::rmsnorm(pf.X, l.attn_norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        lin(l.wq, pf.Ah, T, pf.Q);
        lin(l.wk, pf.Ah, T, pf.Kf);
        lin(l.wv, pf.Ah, T, pf.Vf);
        lin(l.wr, pf.Ah, T, pf.R);
        k::sconv_chunk(pf.Kf, pf.Kc, st_k, l.sc_k, T, kvw, kc, false, s);
        k::sconv_state(pf.Kf, st_k, T, kvw, kc, s);
        k::sconv_chunk(pf.Vf, pf.Vc, st_v, l.sc_v, T, kvw, kc, false, s);
        k::sconv_state(pf.Vf, st_v, T, kvw, kc, s);
        const float* tl = (!l.swa && g.log_floor > 0) ? pf.tau : nullptr;
        k::head_norm(pf.Q, l.q_norm, T, H, hd, g.eps, tl, s);
        k::head_norm(pf.Kc, l.k_norm, T, Hkv, hd, g.eps, nullptr, s);
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
        k::to_f16(AO, pf.AOh, (size_t) T * qw, s);
        lin(l.wo, pf.AOh, T, pf.Y);
        k::sconv_chunk(pf.Y, pf.X, st_a, l.sc_attn, T, E, kc, true, s);
        k::sconv_state(pf.Y, st_a, T, E, kc, s);
        if (times) {
            CK(cudaStreamSynchronize(s));
            const double t1 = now_ms();
            pf.attn_ms += t1 - tl0;
            tl0 = t1;
        }
        // the FFN
        k::rmsnorm(pf.X, l.ffn_norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        if (l.dense) pf_dense(l, T);
        else pf_moe(l, T);
        k::sconv_chunk(pf.Y, pf.X, st_m, l.sc_mlp, T, E, kc, true, s);
        k::sconv_state(pf.Y, st_m, T, E, kc, s);
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
    if (want_logits) {
        k::rmsnorm(pf.X + (size_t) (T - 1) * E, out_norm, a, 1, E, g.eps, s);
        if (is_float(output.type)) {
            k::gemv_f(output.w, output.type, output.rb, a, logits, output.n, output.k, s);
        } else {
            quant(a, 1, E);
            mv(output, a, 0, logits, g.logit_scale);
        }
        CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    CK(cudaStreamSynchronize(s));
    if (want_logits) {
        if (is_float(output.type))
            for (int i = 0; i < g.vocab; ++i) h_logits[i] *= g.logit_scale;
        for (int i = g.vocab_unpadded; i < g.vocab; ++i) h_logits[i] = -INFINITY;
    }
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
    for (int c0 = 0; c0 < n; c0 += pf.C) {
        const int T = std::min(pf.C, n - c0);
        chunk(ids + c0, T, c0 + T == n);
        if (progress) progress(ctx, c0 + T);
    }
    // the prompt's experts as the next tokens' best guess: the VRAM cache refilled from the scores
    ec_fill();
    CK(cudaStreamSynchronize(s));
    st.pf_ms += now_ms() - t0;
    st.pf_tokens += n;
    st.pf_copy_gb = pf.copy_bytes / 1e9;
    return h_logits;
}

const float* Engine::Impl::step(int token) {
    if (pos >= max_ctx || token < 0 || token >= g.vocab) return nullptr;
    const double t0 = now_ms();
    double cpu_ms = 0.0;
    const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, kc = g.conv_k;

    // the token's row, normed
    if (is_float(embd.type)) {
        k::row_to_f32(embd.w + (size_t) token * embd.rb, embd.type, x, E, s);
    } else {
        if (!iq::f16_rows(embd.type, embd.w + (size_t) token * embd.rb, E, 1, e16, E, s)) {
            std::fprintf(stderr, "maya-ink: no row decoder for the embedding's type %d\n", embd.type);
            std::abort();
        }
        k::f16_to_f32((const __half*) e16, x, E, s);
    }
    k::rmsnorm(x, tok_norm, x, 1, E, g.eps, s);
    // the global layers' log-N scale
    h_tau[0] = g.log_floor > 0 ? 1.0f + g.log_alpha * std::log(std::max((float) (pos + 1) / (float) g.log_floor, 1.0f))
                               : 1.0f;
    CK(cudaMemcpyAsync(tau, h_tau, sizeof(float), cudaMemcpyHostToDevice, s));
    const bool tau_on = g.log_floor > 0;

    static const bool trace = std::getenv("INK_TRACE") != nullptr;
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] pos %d layer %d\n", pos, il);
            std::fflush(stderr);
        }
        float* cv = l.conv;
        float *st_k = cv, *st_v = cv + (size_t) (kc - 1) * kvw, *st_a = cv + (size_t) (kc - 1) * 2 * kvw,
              *st_m = cv + (size_t) (kc - 1) * (2 * kvw + E);
        // attention
        k::rmsnorm(x, l.attn_norm, a, 1, E, g.eps, s);
        quant(a, 1, E);
        {
            const DMat* ms[4] = {&l.wq, &l.wk, &l.wv, &l.wr};
            const int rows[4] = {0, 0, 0, 0};
            float* ys[4] = {q, kk, vv, r};
            mv_jobs(ms, rows, ys, 4, a);
        }
        k::shortconv(kk, st_k, l.sc_k, 1, kvw, kc, s);
        k::shortconv(vv, st_v, l.sc_v, 1, kvw, kc, s);
        const float* t = (!l.swa && tau_on) ? tau : nullptr;
        k::head_norm(q, l.q_norm, 1, H, hd, g.eps, t, s);
        k::head_norm(kk, l.k_norm, 1, Hkv, hd, g.eps, nullptr, s);
        k::rel_logits(r, l.rel_proj, t, rel, 1, H, g.d_rel, l.E, s);
        const int slot = l.swa ? pos % g.n_swa : pos;
        k::kv_store(kk, vv, l.K, l.V, slot, kvw, s);
        const int n_kv = l.swa ? std::min(pos + 1, g.n_swa) : pos + 1;
        k::attn_decode(q, l.K, l.V, n_kv, pos, l.swa ? g.n_swa : 0, rel, l.E, H, Hkv, hd, 1.0f / (float) hd, o_,
                       attn_scr, s);
        quant(o_, 1, H * hd);
        mv(l.wo, o_, 0, ao);
        k::shortconv(ao, st_a, l.sc_attn, 1, E, kc, s);
        k::add(x, ao, E, s);

        // the FFN
        k::rmsnorm(x, l.ffn_norm, f, 1, E, g.eps, s);
        if (l.dense) {
            quant(f, 1, E);
            mv(l.gate, f, 0, fg);
            mv(l.up, f, 0, fu);
            k::swiglu(fg, fu, fg, g.n_ff, nullptr, s);
            quant(fg, 1, g.n_ff);
            mv(l.down, fg, 0, fo);
            k::mul_scalar(fo, l.gscale_d, E, s);
        } else {
            const int ne = g.n_expert, nsh = g.n_shexp, ff = g.n_ff_exp;
            quant(f, 1, E);
            mv(l.gate_inp, f, 0, rlog);
            CK(cudaMemcpyAsync(h_f, f, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(h_rlog, rlog, (size_t) (ne + nsh) * sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
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
            // weights: softmax over logsigmoid of [the chosen raw logits, the shared logits], x scale x gscale
            float wv[18];
            const int nw = g.n_used + nsh;
            float mx = -INFINITY;
            for (int i = 0; i < nw; ++i) {
                const float z = i < g.n_used ? h_rlog[sel[i]] : h_rlog[ne + (i - g.n_used)];
                wv[i] = z >= 0.0f ? -std::log1p(std::exp(-z)) : z - std::log1p(std::exp(z));   // logsigmoid
                mx = std::max(mx, wv[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < nw; ++i) {
                wv[i] = std::exp(wv[i] - mx);
                sum += wv[i];
            }
            for (int i = 0; i < nw; ++i) wv[i] = wv[i] / sum * g.exp_scale * l.gscale;
            // the chosen experts in VRAM go to the GPU (launched first: it works while the CPU lane takes the rest)
            int gslot[16], csel[16];
            float gw[16], cw[16];
            int ng = 0, nc = 0;
            for (int i = 0; i < g.n_used; ++i) {
                const size_t key = (size_t) il * ne + sel[i];
                ec.score[key] += 1.0f;
                const int sl = ec.slots ? ec.where[key] : -1;
                if (sl >= 0) {
                    gslot[ng] = sl;
                    gw[ng++] = wv[i];
                } else {
                    csel[nc] = sel[i];
                    cw[nc++] = wv[i];
                }
            }
            ec.lookups += g.n_used;
            ec.hits += ng;
            if (ng) gpu_experts(l, gslot, gw, ng);
            h_gamma[0] = wv[g.n_used];
            h_gamma[1] = wv[g.n_used + 1];
            CK(cudaMemcpyAsync(gamma, h_gamma, 2 * sizeof(float), cudaMemcpyHostToDevice, s));
            // the shared experts, their gammas before the down (the GPU's experts have read f's int8 rows by now)
            {
                const DMat* ms[4] = {&l.sh_gate[0], &l.sh_up[0], &l.sh_gate[1], &l.sh_up[1]};
                const int rows[4] = {0, 0, 0, 0};
                float* ys[4] = {sg, su, sg + ff, su + ff};
                mv_jobs(ms, rows, ys, 2 * nsh, f);
            }
            for (int sh = 0; sh < nsh; ++sh)
                k::swiglu(sg + (size_t) sh * ff, su + (size_t) sh * ff, sg + (size_t) sh * ff, ff, gamma + sh, s);
            quant(sg, nsh, ff);
            {
                const DMat* ms[2] = {&l.sh_down[0], &l.sh_down[1]};
                const int rows[2] = {0, 1};
                float* ys[2] = {so, so + E};
                mv_jobs(ms, rows, ys, nsh, sg);
            }
            if (trace) {
                CK(cudaStreamSynchronize(s));
                std::fprintf(stderr, "[trace] layer %d: experts %d %d %d %d %d %d, %d on the GPU; CPU lane next\n", il,
                             sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], ng);
                std::fflush(stderr);
            }
            if (nc) cpu_experts(*pool, l.cl, E, ff, csel, cw, nc, h_f, h_routed);
            cpu_ms += now_ms() - c0;
            if (nc) {
                CK(cudaMemcpyAsync(routed, h_routed, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(fo, routed, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s));
                if (ng) k::add(fo, ec.routed, E, s);
            } else {
                CK(cudaMemcpyAsync(fo, ec.routed, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s));
            }
            for (int sh = 0; sh < nsh; ++sh) k::add(fo, so + (size_t) sh * E, E, s);
        }
        k::shortconv(fo, st_m, l.sc_mlp, 1, E, kc, s);
        k::add(x, fo, E, s);
    }

    // the output head
    k::rmsnorm(x, out_norm, a, 1, E, g.eps, s);
    if (is_float(output.type)) {
        k::gemv_f(output.w, output.type, output.rb, a, logits, output.n, output.k, s);
    } else {
        quant(a, 1, E);
        mv(output, a, 0, logits, g.logit_scale);
    }
    CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    if (is_float(output.type))
        for (int i = 0; i < g.vocab; ++i) h_logits[i] *= g.logit_scale;
    for (int i = g.vocab_unpadded; i < g.vocab; ++i) h_logits[i] = -INFINITY;
    ++pos;
    // the scores decay as tokens are written; a few swaps every few tokens
    if (ec.slots) {
        for (float& v : ec.score) v *= ec.decay;
        if (++ec.tok % ec.every == 0) ec_adapt();
    }
    st.ms_total += now_ms() - t0;
    st.ms_cpu += cpu_ms;
    ++st.tokens;
    return h_logits;
}

}  // namespace ink
