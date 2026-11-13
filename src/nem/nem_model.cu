// src/nem/nem_model.cu - maya-nem's model (nem_model.hpp): the GGUF read, the weights placed, and a token's step
// through Nemotron 3.5 Super's layers - llama.cpp's graph (src/models/nemotron-h.cpp and mamba-base.cpp's
// build_mamba2_layer, MIT) step by step:
//
//   x = embd[token]
//   per layer:  a = rmsnorm(x, attn_norm);  x += block(a), the block one of
//     Mamba-2:    zxBCdt = ssm_in a;  xBC = silu(conv1d(xBC) + bias) (causal, kernel d_conv, the state its last inputs);
//                 dt = softplus(dt + dt_bias), per head h (group g = h / (heads / groups)):
//                 s[h] = s[h] exp(dt A[h]) + x[h] (x) B[g] dt,  y[h] = s[h] C[g] + D[h] x[h];  y = y silu(z);
//                 y = grouped rmsnorm(y, ssm_norm) (groups of d_inner / n_group);  out = ssm_out y
//     attention:  q, k, v = Wq a, Wk a, Wv a (no RoPE);  out = Wo softmax(q k / sqrt(hd)) v  (GQA, a full cache)
//     MoE:        the top n_used of sigmoid(router a) + exp_probs_b; weights their sigmoids normalized x scale;
//                 l = latent_down a;  m = sum w_e down_e(relu(up_e l)^2)  (the experts in the latent space);
//                 out = latent_up m + shared_down relu(shared_up a)^2
//   logits = output(rmsnorm(x, output_norm))
//
// The machinery is maya-ink's (src/ink): the dense weights in VRAM and int8-activation GEMVs from maya-ds4's kernels,
// a token's routed experts in VRAM run on the GPU while the CPU lane takes the rest from the GGUF's mapping, the VRAM
// expert cache with its staged swaps and lent slots, the prompt path's chunks with cuBLAS and the experts copied
// through pinned staging.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "nem_model.hpp"

#include "nem_cpu.hpp"
#include "nem_ext.cuh"
#include "nem_kernels.cuh"

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

namespace nem {

namespace iq = strata::ds4::gguf::iq;

namespace {

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "maya-nem: CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,             \
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

// an activation's int8 rows (q8 per 32 values with an F32 scale: ds4_iq's MvJob input)
struct Q8 {
    int8_t* q = nullptr;
    float* d = nullptr;
    int k = 0;   // (values a row)
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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
    uint8_t kind = kMamba;
    float* norm = nullptr;   // attn_norm: every block's input norm
    // Mamba-2
    DMat ssm_in, ssm_out;
    float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    float *conv_st = nullptr, *ssm_st = nullptr;
    // attention
    DMat wq, wk, wv, wo;
    __half *K = nullptr, *V = nullptr;
    int kv_rows = 0;
    // MoE
    DMat gate_inp, lat_down, lat_up, sh_up, sh_down;
    std::vector<float> exp_bias;   // (host: a token's routing is done there)
    float* exp_bias_d = nullptr;   // (the prompt path routes on the GPU)
    CpuLayer cl;
};

struct Engine::Impl {
    Options o;
    Geo g;
    int max_ctx = 0, pos = 0, kvw = 0, d_in_proj = 0, conv_ch = 0, dt_off = 0, ssm_P = 0;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Tensor> tens;
    cudaStream_t s = nullptr;
    std::vector<Layer> L;
    DMat embd, output;
    // The embedding's rows stay in the GGUF's mapping (a token needs one): its row copied up through h_erow, a prompt
    // chunk's gathered into the pinned staging and copied to pf.Erows.
    uint8_t *h_erow = nullptr, *d_erow = nullptr;
    float* out_norm = nullptr;
    std::unique_ptr<Pool> pool;
    CpuLane lane;   // the token path's CPU experts, a layer at a time
    size_t dev_bytes = 0;

    // device scratch (the token path)
    float *x = nullptr, *a = nullptr, *zx = nullptr, *y = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr;
    float *o_ = nullptr, *ao = nullptr, *rlog = nullptr, *lat = nullptr, *lat_s = nullptr, *mo = nullptr;
    float *shu = nullptr, *sho = nullptr, *logits = nullptr, *attn_scr = nullptr, *d_tail = nullptr;
    int* dpos = nullptr;
    Q8 qa, qy, ql, qs, qo;   // a's, y's (the Mamba out_proj's), the latent's, the shared expert's h, the attention's
    uint16_t* e16 = nullptr;
    // host (pinned)
    float *h_lat = nullptr, *h_rlog = nullptr, *h_tail = nullptr, *h_logits = nullptr;
    int* h_pos = nullptr;
    Stats st;
    // (Engine::snapshot, NEM_SNAPSHOT=1) the Mamba-2 layers' convolution inputs and scan states at snap_pos, one buffer
    // in VRAM (-1: none taken)
    uint8_t* snap = nullptr;
    size_t snap_bytes = 0;
    int snap_pos = -1;
    size_t conv_bytes() const { return (size_t) conv_ch * (g.d_conv - 1) * sizeof(float); }
    size_t ssm_bytes() const { return (size_t) g.d_inner * g.d_state * sizeof(float); }

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
    Q8 q8alloc(int k) {
        Q8 r;
        r.k = k;
        r.q = dalloc<int8_t>((size_t) k);
        r.d = dalloc<float>((size_t) k / 32);
        return r;
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
    void quant(Q8& qq, const float* xin, int rows, int k) { k::q8(xin, rows, k, qq.q, qq.d, s); }
    // y = alpha (W x): x's int8 rows from quant() (row `row`), or x read as F32 for a float matrix
    void mv(const DMat& m, const Q8& qq, const float* xin, int row, float* yv, float alpha = 1.0f) {
        if (is_float(m.type)) {
            k::gemv_f(m.w, m.type, m.rb, xin + (size_t) row * m.k, yv, m.n, m.k, s);
            return;
        }
        iq::MvJob j;
        j.w = m.w;
        j.ld = m.rb;
        j.type = m.type;
        j.k = m.k;
        j.n = m.n;
        j.ncol = 1;
        j.xq = qq.q + (size_t) row * m.k;
        j.xd = qq.d + (size_t) row * (m.k / 32);
        j.y = yv;
        j.ycs = (size_t) m.n;
        j.alpha = alpha;
        if (!iq::mv(&j, 1, s)) {
            std::fprintf(stderr, "maya-nem: no GEMV for type %d (k %d)\n", m.type, m.k);
            std::abort();
        }
    }
    // several y[i] = W[i] x in one launch (the same input's int8 rows; quantized matrices only)
    void mv_jobs(const DMat* const* m, float* const* yv, int n, const Q8& qq, const float* xin) {
        bool all_q = n <= 16;
        for (int i = 0; i < n && all_q; ++i) all_q = !is_float(m[i]->type);
        if (!all_q) {
            for (int i = 0; i < n; ++i) mv(*m[i], qq, xin, 0, yv[i]);
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
            j[i].xq = qq.q;
            j[i].xd = qq.d;
            j[i].y = yv[i];
            j[i].ycs = (size_t) m[i]->n;
        }
        if (!iq::mv(j, n, s)) {
            std::fprintf(stderr, "maya-nem: no GEMV for these types\n");
            std::abort();
        }
    }

    bool load(const Options& opt, std::string& err);
    const float* step(int token);
    void tok_mamba(Layer& l);
    void tok_attention(Layer& l);
    // The token path in segments between the host's syncs (one a MoE layer: its routing is done on the host): segment
    // 0 the embedding, the layers before the first MoE layer and that one's router; segment j the MoE layer j - 1's end,
    // the layers up to the next MoE layer and its router (the last: the head).  The GPU's part of a segment has the
    // same launches every token (the position read on the device), so each is a CUDA graph (NEM_GRAPHS=0: launched one
    // by one); the routed experts in VRAM, the shared expert and the CPU lane go between segments.
    void tok_embed();
    void moe_pre(Layer& l);
    void moe_mid(Layer& l, int il, double& cpu_ms);
    void moe_tail(Layer& l);
    void tok_head();
    void seg(int j);
    void run_seg(int j) {
        if (graphs > 0) CK(cudaGraphLaunch(gx[(size_t) j], s));
        else seg(j);
    }
    void tok_graphs();
    std::vector<int> moes;            // the MoE layers' indices
    std::vector<cudaGraphExec_t> gx;  // a segment's graph
    int graphs = -1;                  // -1: not made yet, 0: none, 1: made

    // ---------------------------------------------------------------- the prompt path (prefill)
    // A chunk of up to C tokens a layer at a time on the GPU: the dense matrices decoded to F16 and multiplied by
    // cuBLAS (F32 sums); a Mamba-2 layer in sub-chunks of SC tokens (its projection's rows, the conv, the scan carried
    // over them); the attention as tiles of scores (cuBLAS) and a causal softmax; the router in F32; each routed expert
    // some token of the chunk chose copied to a VRAM slot (the copy stream, `stage` slots ahead), decoded to F16 and
    // multiplied for its tokens in the latent space, its output added to theirs times their weights - expert by expert
    // in id order, then the latent up-projection and the shared expert: the same sums every run.
    struct PF {
        int C = 0, SC = 0, GE = 0;
        long group_ctr = 0;
        size_t s_elems = 0, slot_bytes = 0, w1_elems = 0, w2_elems = 0;
        uint8_t* arena = nullptr;   // the buffers below: one allocation, lent to the expert cache between prompts
        size_t arena_bytes = 0;
        uint8_t* gbuf[2] = {nullptr, nullptr};
        cudaEvent_t gready[2] = {nullptr, nullptr}, gfreed[2] = {nullptr, nullptr};
        cublasHandle_t cb = nullptr;
        cudaStream_t cs = nullptr;   // the experts' copies
        int *ids = nullptr, *sel = nullptr, *cnt = nullptr, *off = nullptr, *list = nullptr;
        float *X = nullptr, *A = nullptr, *Y = nullptr, *ZX = nullptr, *XBC = nullptr, *Ys = nullptr;
        float *Q = nullptr, *Kf = nullptr, *Vf = nullptr, *S = nullptr, *Ot = nullptr;
        float *RL = nullptr, *wts = nullptr, *wl = nullptr, *Lat = nullptr, *GU = nullptr, *Ye = nullptr, *Ylat = nullptr;
        float* SU = nullptr;
        __half *Ah = nullptr, *Ysh = nullptr, *Qh = nullptr, *AOh = nullptr, *P = nullptr, *W1 = nullptr, *W2 = nullptr;
        __half *Lath = nullptr, *Xe = nullptr, *He = nullptr, *Ylh = nullptr, *SH = nullptr;
        int *h_ids = nullptr, *h_cnt = nullptr, *h_off = nullptr;
        uint8_t* Erows = nullptr;
        double copy_bytes = 0, copy_ms = 0;
    } pf;
    // The experts' copies to the GPU: the CPU lane's threads copy into two pinned buffers in turn and the DMA takes each
    // from there (the driver's own staging of pageable memory moved 4-15 GB/s on this PC; this holds ~14).
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
    void lin(const DMat& m, const __half* xin, int T, float* yv) {
        f16w(m, pf.W1);
        gemm(xin, m.k, pf.W1, m.k, yv, m.n, T, m.n, m.k);
    }
    void pf_attention(int T, int pos0, const __half* Kb, const __half* Vb, int nkeys, float* out);
    void pf_mamba(Layer& l, int T);
    void pf_attn(Layer& l, int T, int pos0);
    void pf_moe(const Layer& l, int T);
    void chunk(const int* ids, int T, bool want_logits);
    const float* prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx);
    bool pf_open = false;   // a prompt read through the prompt path, no token written since (its buffers not lent)

    // ---------------------------------------------------------------- the VRAM expert cache
    // Slots of VRAM, each one routed expert's up | down as the GGUF holds them, for the experts used most (score: each
    // time a token or a prompt's token chose it, decaying as tokens are written): a token's experts that are in a slot
    // are multiplied on the GPU while the CPU lane takes the others; a prompt's are not copied again.  Filled after each
    // prompt (the best-scored in) and adapted as tokens are written (up to `per` swaps every `every` tokens, each a
    // better score than its slot's expert by `margin` x + 0.5); the scores decay by `decay` a token.  A swap's copy is
    // staged in pieces while the CPU waits for the GPU, and the expert is used from the token after its last piece went
    // up.  A slot is sized for its class (the layers whose experts take the same bytes); the slots past `own` lie in the
    // prompt path's buffers (lent between prompts) and in the attention layers' K / V rows past the position.
    static constexpr int kKvSlack = 4096;
    struct EC {
        int slots = 0, ne = 0, own = 0, lend_cls = 0, arena_end = 0;
        bool lent = false;
        std::vector<int> kv_row;
        int kv_hi = 0;
        std::vector<uint8_t*> ptr;
        std::vector<int> cls;
        std::vector<int> layer_cls;
        std::vector<size_t> cls_bytes;
        std::vector<int> owner, where;   // slot -> key (layer x n_expert + expert) or -1; key -> slot, -1, or -2
        std::vector<float> score;
        long lookups = 0, hits = 0, copies = 0, tok = 0;
        // (maya-ink's settings after its 2026-11-04 tuning: decay 0.99, up to 16 swaps every 2 tokens, margin 1.5;
        // NEM_CACHE_DECAY / _EVERY / _SWAPS / _MARGIN change them)
        float decay = 0.99f, margin = 1.5f;
        int every = 2, per = 16;
        // How a prompt chunk's choices count (NEM_CACHE_PROMPT; pf_score).  0: each expert's count of the chunk's
        // tokens as it is - thousands after a long prompt, where a written token adds 1 and an expert every token
        // chose settles at 99: the cache then holds what the prompt used most and no swap clears the bar until those
        // scores have decayed, some hundred tokens on.  1 (off by default, not run yet): on the written tokens'
        // scale; 2: what was there decayed too, as the chunk's tokens written one by one would.
        int prompt = 0;
        float *u = nullptr, *routed = nullptr, *hd = nullptr, *y = nullptr;
        int8_t* hq = nullptr;
        cudaEvent_t copied = nullptr;
        struct Pending {
            int key = 0, slot = 0;
            size_t done = 0, bytes = 0;
        };
        std::vector<Pending> pend;
        size_t pend_head = 0;
        std::vector<std::pair<int, int>> arrived, arrived_prev;
        cudaEvent_t arrived_ev = nullptr;
        bool defer = true;
        size_t piece = (size_t) 2 << 20;
        uint8_t* apin[2] = {nullptr, nullptr};
        cudaEvent_t aev[2] = {nullptr, nullptr};
        int ak = 0;
        size_t stage_n = 0;
        std::vector<std::pair<const uint8_t*, size_t>> stage_parts;
        std::function<void(int)> stage_fn;
    } ec;
    size_t ub_of(const CpuLayer& c) const { return (size_t) g.n_ff_exp * c.ru; }
    size_t db_of(const CpuLayer& c) const { return (size_t) g.n_lat * c.rd; }
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
    void ec_src_parts(int key, size_t off, size_t n, std::vector<std::pair<const uint8_t*, size_t>>& out) const;
    const uint8_t* ec_blob(int il, int e) const {
        const int sl = ec.slots ? ec.where[(size_t) il * ec.ne + e] : -1;
        return sl >= 0 ? ec.ptr[(size_t) sl] : nullptr;
    }
    void gpu_experts(const Layer& l, const int* slot, const float* w, int n);

    // ---------------------------------------------------------------- the helper card's expert cache
    // A second card (Options::helper_gpu) that holds routed experts only: slots sized by class as the main cache's,
    // filled with the best-scored experts the main card does not hold.  A token's experts found there get the latent
    // from the host (h_lat, there after the MoE layer's sync), are multiplied on the helper's own stream while the main
    // card and the CPU lane do theirs, and their sum comes back to the host, where it is added to the CPU lane's.  The
    // main cache decides first (it adapts as before); the helper follows it: every `every` tokens up to `per` experts
    // change places (an expert the main card took is dropped, a better-scored one outside both comes in), and after a
    // prompt up to `fill_cap`.  The fills at the start and after a prompt go through pinned staging the CPU lane's
    // threads fill (hc_copy_many: as the main cache's fills; NEM_HELPER_STAGE=0: the driver's staging of the mapping,
    // as before).  Nothing of the model's state lives there, so the prompt path and reset() ignore it.
    struct HC {
        int dev = -1, slots = 0;
        cudaStream_t s = nullptr, cs = nullptr;   // the experts' launches; the swaps' copies (the worker's)
        std::vector<uint8_t*> ptr;
        std::vector<int> cls, owner, where;   // slot -> class; slot -> key or -1; key -> slot, -1, or -2: on its way
        std::vector<uint8_t> busy;            // slot -> 1 while its copy is on its way
        float *lat = nullptr, *qd = nullptr, *u = nullptr, *hd = nullptr, *y = nullptr, *routed = nullptr;
        int8_t *qq = nullptr, *hq = nullptr;
        float* h_out = nullptr;               // (pinned) the helper's sum
        // the fills' staging (hc_copy_many): two pinned buffers in turn, each one's last DMA (an event of the helper's)
        uint8_t* pin[2] = {nullptr, nullptr};
        cudaEvent_t pev[2] = {nullptr, nullptr};
        size_t pcap = 0;
        int pk = 0;
        bool stage = false;
        long hits = 0, copies = 0, tok = 0;
        // (measured 2026-11-11 on ab.py, RTX 5090 + RTX 3090: 16 swaps every 2 tokens wrote 48.2 tok/s on new prompts
        //  and 44.7 after a 34K prompt; every 8 tokens 45.6-47.8 and 33-40; every 4 46.7 and 42.0)
        int every = 2, per = 16, fill_cap = 4096;
        float margin = -1.0f, bias = 0.5f;   // a swap's bar: score > margin * the slot's + bias (-1: the main cache's)
        size_t bytes = 0;
        // (NEM_HELPER_BG=1; off by default, not run yet) maya-ink's swaps' thread: the swaps between tokens copied
        // by a thread of their own on a second stream while the next tokens are written, an expert used from the first
        // update after its copy is done - their copies leave the token's thread (0.4 ms a token here), so more of them
        // cost a token nothing.  (key, slot) jobs from q, the copied ones to done (both under mu); inflight is the main
        // thread's count of those it has not collected yet.
        bool bg = false, quit = false;
        std::thread th;
        std::mutex mu;
        std::condition_variable cv, cv_done;
        std::vector<std::pair<int, int>> q, done;
        size_t qhead = 0;
        int inflight = 0;
        // (NEM_HELPER_KEEPALIVE) a one-warp kernel kept running on the helper while tokens are written (each lasts
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

// The Mamba-2 layers' part of the state: each one's convolution inputs and scan state.  The attention layers need
// nothing: a K / V row is written once, at its position, so the rows below the snapshot's stay and those from it on are
// written again before they are read.
bool Engine::snapshot() {
    Impl& m = *p;
    if (!m.snap) return false;
    uint8_t* at = m.snap;
    for (const Layer& l : m.L) {
        if (l.kind != kMamba) continue;
        CK(cudaMemcpyAsync(at, l.conv_st, m.conv_bytes(), cudaMemcpyDeviceToDevice, m.s));
        CK(cudaMemcpyAsync(at + m.conv_bytes(), l.ssm_st, m.ssm_bytes(), cudaMemcpyDeviceToDevice, m.s));
        at += m.conv_bytes() + m.ssm_bytes();
    }
    CK(cudaStreamSynchronize(m.s));
    m.snap_pos = m.pos;
    return true;
}

bool Engine::restore() {
    Impl& m = *p;
    if (!m.snap || m.snap_pos < 0) return false;
    const uint8_t* at = m.snap;
    for (Layer& l : m.L) {
        if (l.kind != kMamba) continue;
        CK(cudaMemcpyAsync(l.conv_st, at, m.conv_bytes(), cudaMemcpyDeviceToDevice, m.s));
        CK(cudaMemcpyAsync(l.ssm_st, at + m.conv_bytes(), m.ssm_bytes(), cudaMemcpyDeviceToDevice, m.s));
        at += m.conv_bytes() + m.ssm_bytes();
    }
    CK(cudaStreamSynchronize(m.s));
    m.pos = m.snap_pos;   // (ec.kv_hi stays: the K / V rows reserved up to the position reached are no slots again)
    return true;
}

bool Engine::can_snapshot() const { return p->snap != nullptr; }

void Engine::reset() {
    p->pos = 0;
    p->snap_pos = -1;   // (a snapshot is of the conversation that ends here)
    p->ec.kv_hi = 0;    // (the K / V caches' slots usable again: empty, the next fill or swaps take them)
    const Geo& g = p->g;
    for (auto& l : p->L)
        if (l.kind == kMamba) {
            CK(cudaMemsetAsync(l.conv_st, 0, (size_t) p->conv_ch * (g.d_conv - 1) * sizeof(float), p->s));
            CK(cudaMemsetAsync(l.ssm_st, 0, (size_t) g.d_inner * g.d_state * sizeof(float), p->s));
        }
    CK(cudaStreamSynchronize(p->s));
}

namespace {
std::string shard_name(const std::string& first, int i, int n) {
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
// a per-layer value: the array's item il, or the scalar
int meta_layer(const strata::GgufFile& f, const std::string& k, int il, int def = 0) {
    const strata::MetaValue* v = f.get(k);
    if (!v) return def;
    if (v->type == strata::MetaType::ARRAY) return il < (int) v->items.size() ? (int) v->items[(size_t) il].num() : def;
    return v->is_num() ? (int) v->num() : def;
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
    if (!arch || arch->s != "nemotron_h_moe") {
        err = "not a Nemotron-H MoE GGUF (general.architecture " + (arch ? arch->s : std::string("missing")) + ")";
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
    const std::string A = "nemotron_h_moe.";
    const int n_all = meta_int(f0, A + "block_count");
    g.n_layer = n_all - std::max(0, meta_int(f0, A + "nextn_predict_layers", 0));   // (the NextN block: not used)
    g.n_embd = meta_int(f0, A + "embedding_length");
    g.n_head = meta_int(f0, A + "attention.head_count");
    g.head_dim = meta_int(f0, A + "attention.key_length", 128);
    g.eps = meta_f(f0, A + "attention.layer_norm_rms_epsilon", meta_f(f0, A + "attention.layer_norm_epsilon", 1e-5f));
    g.n_expert = meta_int(f0, A + "expert_count");
    g.n_used = meta_int(f0, A + "expert_used_count");
    g.n_ff_exp = meta_int(f0, A + "expert_feed_forward_length");
    g.n_ff_sh = meta_int(f0, A + "expert_shared_feed_forward_length");
    g.n_lat = meta_int(f0, A + "moe_latent_size", g.n_embd);
    g.exp_scale = meta_f(f0, A + "expert_weights_scale", 1.0f);
    {
        const strata::MetaValue* v = f0.get(A + "expert_weights_norm");
        g.exp_norm = v ? (v->is_num() ? v->num() != 0 : v->u != 0) : true;
    }
    g.d_conv = meta_int(f0, A + "ssm.conv_kernel");
    g.d_inner = meta_int(f0, A + "ssm.inner_size");
    g.d_state = meta_int(f0, A + "ssm.state_size");
    g.n_group = meta_int(f0, A + "ssm.group_count");
    g.n_ssm_head = meta_int(f0, A + "ssm.time_step_rank");
    g.vocab = meta_int(f0, A + "vocab_size");
    if (g.vocab <= 0)
        if (const strata::MetaValue* v = f0.get("tokenizer.ggml.tokens")) g.vocab = (int) v->items.size();
    g.ctx_train = meta_int(f0, A + "context_length");
    g.eos = meta_int(f0, "tokenizer.ggml.eos_token_id", -1);
    // Each layer's kind from its tensors, not the per-layer metadata arrays: strata's GGUF reader keeps only the first
    // 64 items of an array (gguf_reader.hpp's read_value), and Nemotron's have 89.
    g.kind.assign((size_t) g.n_layer, kMamba);
    for (int il = 0; il < g.n_layer; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        if (find(b + "ssm_in.weight")) {
            g.kind[(size_t) il] = kMamba;
        } else if (const Tensor* tk = find(b + "attn_k.weight")) {
            g.kind[(size_t) il] = kAttn;
            const int hkv = g.head_dim > 0 && tk->info->shape.size() >= 2 ? (int) (tk->info->shape[1] / g.head_dim) : 0;
            if (g.n_head_kv == 0) g.n_head_kv = hkv;
            if (hkv != g.n_head_kv) {
                err = "attention layers with different KV head counts";
                return false;
            }
        } else if (find(b + "ffn_up_exps.weight")) {
            g.kind[(size_t) il] = kMoe;
        } else {
            err = b + "*: neither a Mamba-2, an attention nor a MoE layer's tensors";
            return false;
        }
    }
    (void) meta_layer;
    ssm_P = g.n_ssm_head > 0 ? g.d_inner / g.n_ssm_head : 0;
    if (g.n_layer <= 0 || g.n_embd <= 0 || g.n_head_kv <= 0 || g.head_dim % 32 != 0 || g.head_dim > 256 ||
        g.n_head % g.n_head_kv != 0 || g.n_head / g.n_head_kv > 16 || g.n_used > CpuLane::kMax || g.n_used > 32 ||
        g.n_expert > 512 || g.n_expert % 32 != 0 || g.d_conv < 2 || g.d_conv > 8 || g.n_ssm_head <= 0 ||
        g.n_group <= 0 || g.n_ssm_head % g.n_group != 0 || ssm_P % 16 != 0 || g.d_state % 32 != 0 ||
        g.d_state > 256 || (g.d_inner / g.n_group) % 32 != 0 || g.n_lat % 32 != 0 || g.n_ff_exp % 32 != 0 ||
        g.n_ff_sh % 32 != 0 || g.vocab <= 0) {
        err = "a Nemotron-H shape this engine does not take";
        return false;
    }
    kvw = g.n_head_kv * g.head_dim;
    d_in_proj = 2 * g.d_inner + 2 * g.n_group * g.d_state + g.n_ssm_head;
    conv_ch = g.d_inner + 2 * g.n_group * g.d_state;
    dt_off = 2 * g.d_inner + 2 * g.n_group * g.d_state;
    max_ctx = std::min(o.max_ctx, g.ctx_train > 0 ? g.ctx_train : o.max_ctx);

    CK(cudaSetDevice(o.gpu));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int E = g.n_embd, H = g.n_head, hd = g.head_dim, kc = g.d_conv, NL = g.n_lat, ffe = g.n_ff_exp;

    // the weights in VRAM; the embedding's rows read from the mapping
    {
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
    }
    if (!mat("output.weight", E, g.vocab, output, err)) return false;
    if (!(out_norm = f32("output_norm.weight", (size_t) E, err))) return false;
    L.resize((size_t) g.n_layer);
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        l.kind = g.kind[(size_t) il];
        if (!(l.norm = f32(b + "attn_norm.weight", (size_t) E, err))) return false;
        if (l.kind == kMamba) {
            if (!mat(b + "ssm_in.weight", E, d_in_proj, l.ssm_in, err) ||
                !mat(b + "ssm_out.weight", g.d_inner, E, l.ssm_out, err))
                return false;
            if (!(l.conv_w = f32(b + "ssm_conv1d.weight", (size_t) kc * conv_ch, err))) return false;
            if (find(b + "ssm_conv1d.bias") && !(l.conv_b = f32(b + "ssm_conv1d.bias", (size_t) conv_ch, err))) return false;
            if (!(l.dt_b = f32(b + "ssm_dt.bias", (size_t) g.n_ssm_head, err))) return false;
            if (!(l.A = f32(b + "ssm_a", (size_t) g.n_ssm_head, err))) return false;
            if (!(l.D = f32(b + "ssm_d", (size_t) g.n_ssm_head, err))) return false;
            if (!(l.ssm_norm = f32(b + "ssm_norm.weight", (size_t) g.d_inner, err))) return false;
            l.conv_st = dalloc<float>((size_t) conv_ch * (kc - 1));
            l.ssm_st = dalloc<float>((size_t) g.d_inner * g.d_state);
        } else if (l.kind == kAttn) {
            if (!mat(b + "attn_q.weight", E, H * hd, l.wq, err) || !mat(b + "attn_k.weight", E, kvw, l.wk, err) ||
                !mat(b + "attn_v.weight", E, kvw, l.wv, err) || !mat(b + "attn_output.weight", H * hd, E, l.wo, err))
                return false;
            l.kv_rows = max_ctx;
            l.K = dalloc<__half>((size_t) l.kv_rows * kvw);
            l.V = dalloc<__half>((size_t) l.kv_rows * kvw);
        } else {
            const int ne = g.n_expert;
            if (!mat(b + "ffn_gate_inp.weight", E, ne, l.gate_inp, err)) return false;
            if (l.gate_inp.type != 0) {
                err = b + "ffn_gate_inp: not F32 (the prompt path's router multiplies F32 rows)";
                return false;
            }
            if (!(l.exp_bias_d = f32(b + "exp_probs_b.bias", (size_t) ne, err, &l.exp_bias))) return false;
            if (!mat(b + "ffn_latent_down.weight", E, NL, l.lat_down, err) ||
                !mat(b + "ffn_latent_up.weight", NL, E, l.lat_up, err) ||
                !mat(b + "ffn_up_shexp.weight", E, g.n_ff_sh, l.sh_up, err) ||
                !mat(b + "ffn_down_shexp.weight", g.n_ff_sh, E, l.sh_down, err))
                return false;
            // the routed experts stay where they are, for the CPU lane
            const Tensor *tu = find(b + "ffn_up_exps.weight"), *td = find(b + "ffn_down_exps.weight");
            if (!tu || !td) {
                err = b + "ffn_*_exps: missing";
                return false;
            }
            if (!iq::mv_supported((int) tu->info->type) || !iq::mv_supported((int) td->info->type)) {
                err = b + "ffn_*_exps: a type the GPU's kernels do not take (" + tu->info->type_name() + " / " +
                      td->info->type_name() + ")";
                return false;
            }
            if (!cpu_layer((int) tu->info->type, (int) td->info->type, NL, ffe, l.cl, err)) {
                err = b + "ffn_*_exps: " + err;
                return false;
            }
            l.cl.up = tu->data;
            l.cl.down = td->data;
        }
        if (o.verbose && (il % 11 == 10 || il == g.n_layer - 1))
            std::fprintf(stderr, "maya-nem: layers 0..%d placed (%.1f GB of VRAM so far)\n", il, dev_bytes / 1e9);
    }

    // scratch
    x = dalloc<float>(E);
    a = dalloc<float>(E);
    zx = dalloc<float>((size_t) d_in_proj);
    y = dalloc<float>((size_t) g.d_inner);
    q = dalloc<float>((size_t) H * hd);
    kk = dalloc<float>(kvw);
    vv = dalloc<float>(kvw);
    o_ = dalloc<float>((size_t) H * hd);
    ao = dalloc<float>(E);
    rlog = dalloc<float>((size_t) g.n_expert);
    lat = dalloc<float>((size_t) NL);
    lat_s = dalloc<float>((size_t) NL);
    mo = dalloc<float>(E);
    shu = dalloc<float>((size_t) g.n_ff_sh);
    sho = dalloc<float>(E);
    logits = dalloc<float>((size_t) g.vocab);
    d_tail = dalloc<float>((size_t) 4 + NL);
    dpos = dalloc<int>(1);
    attn_scr = dalloc<float>(k::attn_scratch_floats(H, hd, k::attn_nsp(0, max_ctx)));
    qa = q8alloc(E);
    qy = q8alloc(g.d_inner);
    ql = q8alloc(NL);
    qs = q8alloc(g.n_ff_sh);
    qo = q8alloc(H * hd);
    e16 = dalloc<uint16_t>(E);
    h_lat = halloc<float>((size_t) NL);
    h_rlog = halloc<float>((size_t) g.n_expert);
    h_tail = halloc<float>((size_t) 4 + NL);
    h_logits = halloc<float>((size_t) g.vocab);
    h_pos = halloc<int>(1);
    // the room for Engine::snapshot (NEM_SNAPSHOT=1; off by default, not run yet): every Mamba-2 layer's convolution
    // inputs and scan state (173 MB for Nemotron 3.5 Super), taken before the expert cache sizes itself
    if (const char* sv = std::getenv("NEM_SNAPSHOT"); sv && std::atoi(sv) != 0) {
        for (const Layer& l : L)
            if (l.kind == kMamba) snap_bytes += conv_bytes() + ssm_bytes();
        if (snap_bytes) snap = dalloc<uint8_t>(snap_bytes);
    }

    int threads = o.threads > 0 ? o.threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    pool.reset(new Pool(threads - 1));
    // Every routed expert's pages touched once, now: the first touch of a mapped page costs a fault (~1 us a 4 KB page
    // on this PC even with the file in RAM), which held the prompt path's copies and the CPU lane's first tokens back.
    // NEM_PREFAULT=0: not done.
    const char* pv = std::getenv("NEM_PREFAULT");
    if (!pv || std::atoi(pv) != 0) {
        const double tp = now_ms();
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        double gb = 0;
        for (const Layer& l : L) {
            if (l.kind != kMoe) continue;
            const size_t ub_ = ub_of(l.cl) * g.n_expert, db_ = db_of(l.cl) * g.n_expert;
            const std::pair<const uint8_t*, size_t> rs[2] = {{l.cl.up, ub_}, {l.cl.down, db_}};
            for (const auto& r0 : rs)
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
            std::fprintf(stderr, "maya-nem: the routed experts' %.1f GB touched in %.1f s (working set %.1f GB)\n", gb,
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
            std::fprintf(stderr, "maya-nem: the expert profile %s: %ld experts into VRAM in %.1f s\n", o.profile.c_str(),
                         ec.copies, (now_ms() - tf) / 1000.0);
    }
    if (o.helper_gpu >= 0) {
        const double th = now_ms();
        hc_alloc();
        if (hc.slots) {
            hc_update(INT_MAX, false, false);
            if (o.verbose)
                std::fprintf(stderr,
                             "maya-nem: helper card CUDA%d: %d expert slots, %.2f GB of its VRAM; %ld experts into them in "
                             "%.1f s\n",
                             hc.dev, hc.slots, hc.bytes / 1e9, hc.copies, (now_ms() - th) / 1000.0);
        }
    }
    if (o.verbose) {
        std::string cls;
        for (size_t c = 0; c < ec.cls_bytes.size(); ++c) {
            int n_all = 0, n_l = 0;
            for (int sl = 0; sl < ec.slots; ++sl) n_all += ec.cls[(size_t) sl] == (int) c;
            for (int il = 0; il < g.n_layer; ++il) n_l += ec.layer_cls[(size_t) il] == (int) c;
            char b[128];
            std::snprintf(b, sizeof b, "%s%d of %.2f MB (%d layers)", c ? ", " : "", n_all, ec.cls_bytes[c] / 1e6, n_l);
            cls += b;
        }
        int nm = 0, na = 0, nmoe = 0;
        for (uint8_t kd : g.kind) (kd == kMamba ? nm : kd == kAttn ? na : nmoe)++;
        std::fprintf(stderr,
                     "maya-nem: Nemotron-H %d layers (%d Mamba-2, %d attention, %d MoE), %d experts (top %d, latent %d), "
                     "context %d; %.2f GB in VRAM (the prompt path's chunks of %d tokens %.2f of it, %d expert slots "
                     "%.2f, %d more in the prompt path's buffers between prompts and %d in the K / V caches' rows past "
                     "the position: %s), the routed experts in the GGUF's mapping; CPU lane %d threads; loaded in %.1f s\n",
                     g.n_layer, nm, na, nmoe, g.n_expert, g.n_used, g.n_lat, max_ctx, dev_bytes / 1e9, pf.C,
                     (before_ec - before_pf) / 1e9, ec.own, (dev_bytes - before_ec) / 1e9, ec.arena_end - ec.own,
                     ec.slots - ec.arena_end, cls.c_str(), pool->threads(), (now_ms() - t0) / 1000.0);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert cache
void Engine::Impl::ec_alloc() {
    const int NL = g.n_lat, ff = g.n_ff_exp;
    ec.ne = g.n_expert;
    ec.where.assign((size_t) g.n_layer * ec.ne, -1);
    ec.score.assign((size_t) g.n_layer * ec.ne, 0.0f);
    ec.u = dalloc<float>((size_t) 32 * ff);
    ec.hq = dalloc<int8_t>((size_t) 32 * ff);
    ec.hd = dalloc<float>((size_t) 32 * ff / 32);
    ec.y = dalloc<float>((size_t) 32 * NL);
    ec.routed = dalloc<float>(NL);
    CK(cudaEventCreateWithFlags(&ec.copied, cudaEventDisableTiming));
    if (const char* v = std::getenv("NEM_CACHE_DECAY")) ec.decay = (float) std::atof(v);
    if (const char* v = std::getenv("NEM_CACHE_EVERY")) ec.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("NEM_CACHE_SWAPS")) ec.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("NEM_CACHE_MARGIN")) ec.margin = (float) std::atof(v);
    if (const char* v = std::getenv("NEM_CACHE_PROMPT")) ec.prompt = std::max(0, std::min(2, std::atoi(v)));
    if (const char* v = std::getenv("NEM_CACHE_DEFER")) ec.defer = std::atoi(v) != 0;
    if (const char* v = std::getenv("NEM_CACHE_STAGE_KB")) ec.piece = (size_t) std::max(64, std::atoi(v)) << 10;
    if (ec.defer) {
        for (int i = 0; i < 2 && ec.defer; ++i) {
            ec.defer = cudaHostAlloc((void**) &ec.apin[i], 4 * ec.piece, cudaHostAllocPortable) == cudaSuccess;
            if (ec.defer) CK(cudaEventCreateWithFlags(&ec.aev[i], cudaEventDisableTiming));
        }
        if (ec.defer) CK(cudaEventCreateWithFlags(&ec.arrived_ev, cudaEventDisableTiming));
        if (!ec.defer) cudaGetLastError();
    }
    const char* lv = std::getenv("NEM_LEND");
    const bool lend = !lv || std::atoi(lv) != 0;
    // the classes: each MoE layer's expert bytes (up | down as the GGUF holds them), a 256-byte multiple
    ec.layer_cls.assign((size_t) g.n_layer, -1);
    std::vector<size_t> eb((size_t) g.n_layer, 0);
    int nmoe = 0;
    for (int il = 0; il < g.n_layer; ++il) {
        if (L[(size_t) il].kind != kMoe) continue;
        ++nmoe;
        const CpuLayer& c = L[(size_t) il].cl;
        eb[(size_t) il] = (ub_of(c) + db_of(c) + 255) & ~(size_t) 255;
        auto it = std::find(ec.cls_bytes.begin(), ec.cls_bytes.end(), eb[(size_t) il]);
        if (it == ec.cls_bytes.end()) {
            ec.cls_bytes.push_back(eb[(size_t) il]);
            it = ec.cls_bytes.end() - 1;
        }
        ec.layer_cls[(size_t) il] = (int) (it - ec.cls_bytes.begin());
    }
    const int nc = (int) ec.cls_bytes.size();
    if (nc == 0) return;
    std::vector<int> nl((size_t) nc, 0);
    for (int il = 0; il < g.n_layer; ++il)
        if (ec.layer_cls[(size_t) il] >= 0) ++nl[(size_t) ec.layer_cls[(size_t) il]];
    ec.lend_cls = (int) (std::max_element(nl.begin(), nl.end()) - nl.begin());
    // the cache's own VRAM: what is free, less the reserve, each class's slots its share of the layers
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t reserve = (size_t) 1536 << 20;
    if (const char* v = std::getenv("NEM_VRAM_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
    size_t want = fr > reserve ? fr - reserve : 0;
    if (o.cache_gb >= 0.0) want = std::min(want, (size_t) (o.cache_gb * 1e9));
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
    const char* kv = std::getenv("NEM_LEND_KV");
    if (base && (!kv || std::atoi(kv) != 0)) {
        const size_t b = ec.cls_bytes[(size_t) ec.lend_cls], row = (size_t) kvw * sizeof(__half);
        for (const Layer& l : L) {
            if (l.kind != kAttn) continue;
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

void Engine::Impl::ec_copy(int key, int slot) {
    const int il = key / ec.ne, e = key % ec.ne;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t ub = ub_of(c), db = db_of(c);
    uint8_t* d = ec.ptr[(size_t) slot];
    if (ec.owner[(size_t) slot] >= 0) ec.where[(size_t) ec.owner[(size_t) slot]] = -1;
    CK(cudaMemcpyAsync(d, c.up + (size_t) e * ub, ub, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(d + ub, c.down + (size_t) e * db, db, cudaMemcpyHostToDevice, s));
    ec.owner[(size_t) slot] = key;
    ec.where[(size_t) key] = slot;
    ++ec.copies;
}

// Experts (key, slot) into their slots: as many as fill a pinned staging buffer at a time, copied into it by the CPU
// lane's threads (1 MB pieces), then one DMA each on the copy stream from there; the compute stream waits for the last.
void Engine::Impl::ec_copy_many(const std::vector<std::pair<int, int>>& todo) {
    if (todo.empty()) return;
    if (!stg.on) {
        for (const auto& kv : todo) ec_copy(kv.first, kv.second);
        return;
    }
    struct Part {
        const uint8_t* src;
        size_t off, len;
    };
    size_t i = 0;
    while (i < todo.size()) {
        std::vector<Part> parts;
        std::vector<std::pair<size_t, size_t>> dma;   // (todo index, offset in the buffer)
        size_t fill = 0;
        for (; i < todo.size(); ++i) {
            const int key = todo[i].first, il = key / ec.ne, e = key % ec.ne;
            const CpuLayer& c = L[(size_t) il].cl;
            const size_t ub = ub_of(c), db = db_of(c);
            if (fill + ub + db > stg.cap) {
                if (fill == 0) {
                    ec_copy(key, todo[i].second);
                    continue;
                }
                break;
            }
            dma.push_back({i, fill});
            const std::pair<const uint8_t*, size_t> rs[2] = {{c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
            for (const auto& r0 : rs) {
                for (size_t o0 = 0; o0 < r0.second; o0 += (size_t) 1 << 20)
                    parts.push_back({r0.first + o0, fill + o0, std::min((size_t) 1 << 20, r0.second - o0)});
                fill += r0.second;
            }
        }
        CK(cudaEventSynchronize(stg.ev[stg.k]));
        uint8_t* pin = stg.pin[stg.k];
        pool->run((int) parts.size(), [&](int j) {
            const Part& p0 = parts[(size_t) j];
            std::memcpy(pin + p0.off, p0.src, p0.len);
        });
        for (const auto& d : dma) {
            const int key = todo[d.first].first, sl = todo[d.first].second, il = key / ec.ne;
            const CpuLayer& c = L[(size_t) il].cl;
            CK(cudaMemcpyAsync(ec.ptr[(size_t) sl], pin + d.second, ub_of(c) + db_of(c), cudaMemcpyHostToDevice, pf.cs));
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
        for (int il = 0; il < g.n_layer; ++il)
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
        std::vector<int> best, worst;
        for (int il = 0; il < g.n_layer; ++il) {
            if (ec.layer_cls[(size_t) il] != c) continue;
            for (int e = 0; e < ec.ne; ++e) {
                const int key = il * ec.ne + e;
                if (ec.where[(size_t) key] != -1 || ec.score[(size_t) key] <= 0.0f) continue;
                // (an expert the helper card holds is off the CPU already: the main card's swaps go to the ones that
                //  are not, and the helper's slots stay as they are; NEM_HELPER_STEAL=1: the main card takes them too)
                if (hc.slots && !hc_steal && hc.where[(size_t) key] != -1) continue;   // (there, or on its way)
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
            else if (ec.where[(size_t) ow] >= 0) worst.push_back(sl);
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
    for (const auto& kv : todo) {
        const int key = kv.first, sl = kv.second;
        if (ec.owner[(size_t) sl] >= 0) ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
        ec.owner[(size_t) sl] = key;
        ec.where[(size_t) key] = -2;
        const CpuLayer& c = L[(size_t) (key / ec.ne)].cl;
        EC::Pending p;
        p.key = key;
        p.slot = sl;
        p.bytes = ub_of(c) + db_of(c);
        ec.pend.push_back(p);
    }
}

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
//      through the chunk.  A long chunk so adds at most 99 (decay 0.99), a 32-token one 0.85 a count.
//   2: the same, the layer's scores first times decay^T, as T written tokens leave them.
// (Integers and the same float operations in the same order every run: the same scores.)  maya-ink's.
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
    const int il = key / ec.ne, e = key % ec.ne;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t ub = ub_of(c), db = db_of(c);
    const std::pair<const uint8_t*, size_t> rs[2] = {{c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
    size_t at = 0;
    for (const auto& r0 : rs) {
        const size_t a0 = std::max(off, at), a1 = std::min(off + n, at + r0.second);
        if (a0 < a1) out.push_back({r0.first + (a0 - at), a1 - a0});
        at += r0.second;
    }
}

bool Engine::Impl::ec_stage_begin(size_t budget) {
    if (!ec.defer || ec.pend_head >= ec.pend.size()) return false;
    EC::Pending& p = ec.pend[ec.pend_head];
    const size_t n = std::min(budget, p.bytes - p.done);
    CK(cudaEventSynchronize(ec.aev[ec.ak]));
    ec.stage_n = n;
    ec.stage_parts.clear();
    ec_src_parts(p.key, p.done, n, ec.stage_parts);
    const auto* parts = &ec.stage_parts;
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

// A token's experts in VRAM, on the latent l's int8 rows (ql): u = up_e l, h = relu(u)^2, y_e = w_e down_e h, then
// their sum -> ec.routed
void Engine::Impl::gpu_experts(const Layer& l, const int* slot, const float* w, int n) {
    const int NL = g.n_lat, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t ub = ub_of(c);
    iq::MvJob jobs[32];
    for (int i = 0; i < n; ++i) {
        iq::MvJob& j = jobs[i];
        j = iq::MvJob{};
        j.w = ec.ptr[(size_t) slot[i]];
        j.ld = c.ru;
        j.type = c.tu;
        j.k = NL;
        j.n = ff;
        j.ncol = 1;
        j.xq = ql.q;
        j.xd = ql.d;
        j.y = ec.u + (size_t) i * ff;
        j.ycs = (size_t) ff;
    }
    if (!iq::mv(jobs, n, s)) {
        std::fprintf(stderr, "maya-nem: no GEMV for the experts' up type %d\n", c.tu);
        std::abort();
    }
    k::relu2(ec.u, (size_t) n * ff, s);
    k::q8(ec.u, n, ff, ec.hq, ec.hd, s);
    iq::MvJob dj[32];
    for (int i = 0; i < n; ++i) {
        dj[i] = iq::MvJob{};
        dj[i].w = ec.ptr[(size_t) slot[i]] + ub;
        dj[i].ld = c.rd;
        dj[i].type = c.td;
        dj[i].k = ff;
        dj[i].n = NL;
        dj[i].ncol = 1;
        dj[i].xq = ec.hq + (size_t) i * ff;
        dj[i].xd = ec.hd + (size_t) i * (ff / 32);
        dj[i].y = ec.y + (size_t) i * NL;
        dj[i].ycs = (size_t) NL;
        dj[i].bias = nullptr;
        dj[i].alpha = w[i];
    }
    if (!iq::mv(dj, n, s)) {
        std::fprintf(stderr, "maya-nem: no GEMV for the experts' down type %d\n", c.td);
        std::abort();
    }
    k::sum_rows(ec.y, n, NL, ec.routed, s);
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
        std::fprintf(stderr, "maya-nem: no helper card: --helper-gpu %d is not a second visible CUDA device (%d visible)\n",
                     o.helper_gpu, ndev);
        return;
    }
    hc.dev = o.helper_gpu;
    if (const char* v = std::getenv("NEM_HELPER_EVERY")) hc.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("NEM_HELPER_SWAPS")) hc.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("NEM_HELPER_FILL")) hc.fill_cap = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("NEM_HELPER_STEAL")) hc_steal = std::atoi(v) != 0;
    if (const char* v = std::getenv("NEM_HELPER_BG")) hc.bg = std::atoi(v) != 0;
    if (const char* v = std::getenv("NEM_HELPER_MARGIN")) hc.margin = (float) std::atof(v);
    if (const char* v = std::getenv("NEM_HELPER_BIAS")) hc.bias = (float) std::atof(v);
    if (hc.margin < 0.0f) hc.margin = ec.margin;
    const int NL = g.n_lat, ff = g.n_ff_exp, nc = (int) ec.cls_bytes.size();
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
    hc.lat = (float*) dev_alloc((size_t) NL * sizeof(float));
    hc.qq = (int8_t*) dev_alloc((size_t) NL);
    hc.qd = (float*) dev_alloc((size_t) NL / 32 * sizeof(float));
    hc.u = (float*) dev_alloc((size_t) 32 * ff * sizeof(float));
    hc.hq = (int8_t*) dev_alloc((size_t) 32 * ff);
    hc.hd = (float*) dev_alloc((size_t) 32 * ff / 32 * sizeof(float));
    hc.y = (float*) dev_alloc((size_t) 32 * NL * sizeof(float));
    hc.routed = (float*) dev_alloc((size_t) NL * sizeof(float));
    CK(cudaHostAlloc((void**) &hc.h_out, (size_t) NL * sizeof(float), cudaHostAllocPortable));
    // the fills' pinned staging: its own buffers and events (an event belongs to its device; the prompt path's staging
    // may still hold the main card's copies).  NEM_HELPER_STAGE: a buffer's MB, 64; 0: none - the driver's staging
    const char* hsv = std::getenv("NEM_HELPER_STAGE");
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
            std::fprintf(stderr, "maya-nem: no pinned staging for the helper card's fills (the driver's is used)\n");
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
    if (const char* v = std::getenv("NEM_HELPER_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
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
    // RTX 3090 measured 420-1,260 MHz and its memory at 5,001 or 810 MHz of 9,751 under maya-ink's Inkling 2, whose
    // expert GEMVs took 180-280 us there against 15-23 on the 5090.  A kernel kept resident while tokens are written
    // (one warp, asleep in __nanosleep, on a non-blocking stream of the lowest priority) makes the card count as busy.
    // Each one ends by itself after NEM_HELPER_KEEPALIVE_MS (50) and step() launches the next: a call that implicitly
    // waits for all of the card's work (an allocation, a plain memset) waits that long at most - one left running
    // forever hung the first token - and the card drops back to idle once the writing stops.
    // Off by default (2026-11-13: it costs the helper ~80-90 W while tokens are written, for 5-11% more speed);
    // NEM_HELPER_KEEPALIVE=1 turns it on.
    const char* kv = std::getenv("NEM_HELPER_KEEPALIVE");
    if (hc.slots && kv && std::atoi(kv) != 0) {
        int lo = 0, hi = 0;
        const char* km = std::getenv("NEM_HELPER_KEEPALIVE_MS");
        hc.ka_ns = (unsigned long long) std::max(1, km ? std::atoi(km) : 50) * 1000000ull;
        OnDev od(hc.dev);
        if (cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess &&
            cudaStreamCreateWithPriority(&hc.ks, cudaStreamNonBlocking, lo) == cudaSuccess) {
            std::fprintf(stderr, "maya-nem: helper card CUDA%d: a keep-alive kernel holds its clocks up while tokens are "
                         "written (%llu ms each)\n", hc.dev, hc.ka_ns / 1000000ull);
        } else {
            cudaGetLastError();
            hc.ks = nullptr;
            std::fprintf(stderr, "maya-nem: helper card CUDA%d: no keep-alive kernel\n", hc.dev);
        }
    }
}

// An expert's bytes (up | down, as the main cache's slots) into a helper slot, on st (the helper current)
void Engine::Impl::hc_copy(int key, int slot, cudaStream_t st) {
    const int il = key / ec.ne, e = key % ec.ne;
    const CpuLayer& c = L[(size_t) il].cl;
    const size_t ub = ub_of(c), db = db_of(c);
    uint8_t* d = hc.ptr[(size_t) slot];
    CK(cudaMemcpyAsync(d, c.up + (size_t) e * ub, ub, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d + ub, c.down + (size_t) e * db, db, cudaMemcpyHostToDevice, st));
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

// A fill's experts (key, slot) into their helper slots, the helper current: ec_copy_many's way - as many as fill a
// pinned buffer at a time, copied into it by the CPU lane's threads (1 MB pieces), then one DMA each on the helper's
// stream from there.  The same bytes into the same slots as hc_copy's; called between tokens (the pool idle).
void Engine::Impl::hc_copy_many(const std::vector<std::pair<int, int>>& todo) {
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
            const size_t ub = ub_of(c), db = db_of(c);
            if (fill + ub + db > hc.pcap) {
                if (fill == 0) {   // (an expert larger than the buffer: the driver's staging)
                    hc_copy(key, todo[i].second, hc.s);
                    continue;
                }
                break;
            }
            dma.push_back({i, fill});
            const std::pair<const uint8_t*, size_t> rs[2] = {{c.up + (size_t) e * ub, ub}, {c.down + (size_t) e * db, db}};
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
            CK(cudaMemcpyAsync(hc.ptr[(size_t) sl], pin + d.second, ub_of(c) + db_of(c), cudaMemcpyHostToDevice, hc.s));
        }
        CK(cudaEventRecord(hc.pev[hc.pk], hc.s));
        hc.pk ^= 1;
    }
}

// The helper's slots brought up to date, at most max_copies experts copied: the slots whose expert the main card holds
// (or is taking) are freed, then the best-scored experts outside both cards come in - into free slots, then in place
// of the worst-scored ones (with hysteresis: only one that beats its slot's expert by the main cache's margin).  Called
// between tokens, the helper idle.  bg (with the swaps' thread): the copies go to the worker and count against
// max_copies until they are done; otherwise they are made here, after the worker's.
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
    // (a fill - the start's, a prompt's - through the pinned staging; the swaps between tokens as before)
    if (hc.stage && !hysteresis) hc_copy_many(todo);
    else
        for (const auto& kv : todo) hc_copy(kv.first, kv.second, hc.s);
    hc.copies += (long) todo.size();
    CK(cudaStreamSynchronize(hc.s));
    // (NEM_HELPER_VERIFY=1, a check) every slot just filled read back against the mapping's bytes
    static const bool verify = std::getenv("NEM_HELPER_VERIFY") != nullptr;
    if (verify) {
        std::vector<uint8_t> back;
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        for (const auto& kv : todo) {
            const CpuLayer& c = L[(size_t) (kv.first / ec.ne)].cl;
            const size_t bytes = ub_of(c) + db_of(c);
            back.resize(bytes);
            CK(cudaMemcpy(back.data(), hc.ptr[(size_t) kv.second], bytes, cudaMemcpyDeviceToHost));
            parts.clear();
            ec_src_parts(kv.first, 0, bytes, parts);
            size_t at = 0;
            for (const auto& pr : parts) {
                if (std::memcmp(back.data() + at, pr.first, pr.second) != 0) {
                    std::fprintf(stderr,
                                 "maya-nem: NEM_HELPER_VERIFY: expert %d of layer %d differs in helper slot %d\n",
                                 kv.first % ec.ne, kv.first / ec.ne, kv.second);
                    std::abort();
                }
                at += pr.second;
            }
        }
        std::fprintf(stderr,
                     "maya-nem: NEM_HELPER_VERIFY: %zu experts in the helper's slots hold the mapping's bytes\n",
                     todo.size());
    }
}

// A token's experts on the helper card: the latent up from the host, its int8 rows, then gpu_experts' launches on the
// helper's stream and buffers; their sum -> hc.h_out (there once hc.s is synchronized).
void Engine::Impl::hc_experts(const Layer& l, const int* slot, const float* w, int n) {
    const int NL = g.n_lat, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t ub = ub_of(c);
    OnDev od(hc.dev);
    CK(cudaMemcpyAsync(hc.lat, h_lat, (size_t) NL * sizeof(float), cudaMemcpyHostToDevice, hc.s));
    k::q8(hc.lat, 1, NL, hc.qq, hc.qd, hc.s);
    iq::MvJob jobs[32];
    for (int i = 0; i < n; ++i) {
        iq::MvJob& j = jobs[i];
        j = iq::MvJob{};
        j.w = hc.ptr[(size_t) slot[i]];
        j.ld = c.ru;
        j.type = c.tu;
        j.k = NL;
        j.n = ff;
        j.ncol = 1;
        j.xq = hc.qq;
        j.xd = hc.qd;
        j.y = hc.u + (size_t) i * ff;
        j.ycs = (size_t) ff;
    }
    if (!iq::mv(jobs, n, hc.s)) {
        std::fprintf(stderr, "maya-nem: no GEMV for the experts' up type %d\n", c.tu);
        std::abort();
    }
    k::relu2(hc.u, (size_t) n * ff, hc.s);
    k::q8(hc.u, n, ff, hc.hq, hc.hd, hc.s);
    iq::MvJob dj[32];
    for (int i = 0; i < n; ++i) {
        dj[i] = iq::MvJob{};
        dj[i].w = hc.ptr[(size_t) slot[i]] + ub;
        dj[i].ld = c.rd;
        dj[i].type = c.td;
        dj[i].k = ff;
        dj[i].n = NL;
        dj[i].ncol = 1;
        dj[i].xq = hc.hq + (size_t) i * ff;
        dj[i].xd = hc.hd + (size_t) i * (ff / 32);
        dj[i].y = hc.y + (size_t) i * NL;
        dj[i].ycs = (size_t) NL;
        dj[i].bias = nullptr;
        dj[i].alpha = w[i];
    }
    if (!iq::mv(dj, n, hc.s)) {
        std::fprintf(stderr, "maya-nem: no GEMV for the experts' down type %d\n", c.td);
        std::abort();
    }
    k::sum_rows(hc.y, n, NL, hc.routed, hc.s);
    CK(cudaMemcpyAsync(hc.h_out, hc.routed, (size_t) NL * sizeof(float), cudaMemcpyDeviceToHost, hc.s));
}

bool Engine::Impl::save_profile(const std::string& path) const {
    FILE* f = std::fopen((path + ".tmp").c_str(), "wb");
    if (!f) return false;
    const int hdr[4] = {0x504d454e, 1, g.n_layer, g.n_expert};   // "NEMP", version 1
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
    const bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x504d454e && hdr[1] == 1 &&
                    hdr[2] == g.n_layer && hdr[3] == g.n_expert &&
                    std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
    std::fclose(f);
    if (ok) ec.score = std::move(v);
    return ok;
}

// ------------------------------------------------------------------------------------------------ the prompt path
bool Engine::Impl::pf_alloc(std::string& err) {
    const int C = pf.C = std::max(64, o.chunk);
    const int E = g.n_embd, H = g.n_head, hd = g.head_dim, ne = g.n_expert, nu = g.n_used, NL = g.n_lat;
    const int ffe = g.n_ff_exp, ffs = g.n_ff_sh, qw = H * hd, G = H / g.n_head_kv;
    pf.SC = std::min(C, 2048);   // (a Mamba layer's and the shared expert's sub-chunks)
    if (const char* v = std::getenv("NEM_PF_SUB")) pf.SC = std::max(64, std::min(C, std::atoi(v)));
    const int SC = pf.SC;
    if (cublasCreate(&pf.cb) != CUBLAS_STATUS_SUCCESS) {
        err = "cuBLAS did not start";
        return false;
    }
    cublasSetStream(pf.cb, s);
    CK(cudaStreamCreateWithFlags(&pf.cs, cudaStreamNonBlocking));
    const size_t CE = (size_t) C * E;
    size_t mib = 512;
    if (const char* v = std::getenv("NEM_SCORES_MIB")) mib = (size_t) std::max(64, std::atoi(v));
    pf.s_elems = (mib << 20) / 6;   // S F32 and P F16
    pf.w1_elems = std::max({(size_t) d_in_proj * E, (size_t) ffs * E, (size_t) qw * E, (size_t) NL * E,
                            (size_t) ffe * NL});
    pf.w2_elems = std::max({(size_t) E * g.d_inner, (size_t) E * ffs, (size_t) NL * ffe});
    for (const Layer& l : L)
        if (l.kind == kMoe) pf.slot_bytes = std::max(pf.slot_bytes, ub_of(l.cl) + db_of(l.cl));
    pf.GE = std::max(1, std::min(ne, o.stage));
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
        carve(pf.Y, CE);
        carve(pf.Ah, CE);
        carve(pf.ZX, (size_t) SC * d_in_proj);
        carve(pf.XBC, (size_t) SC * conv_ch);
        carve(pf.Ys, (size_t) SC * g.d_inner);
        carve(pf.Ysh, (size_t) SC * g.d_inner);
        carve(pf.Q, (size_t) C * qw);
        carve(pf.Kf, (size_t) C * kvw);
        carve(pf.Vf, (size_t) C * kvw);
        carve(pf.Qh, (size_t) C * qw);
        carve(pf.AOh, (size_t) C * qw);
        carve(pf.Ot, (size_t) G * C * hd);
        carve(pf.S, pf.s_elems);
        carve(pf.P, pf.s_elems);
        carve(pf.RL, (size_t) C * ne);
        carve(pf.sel, (size_t) C * nu);
        carve(pf.wts, (size_t) C * nu);
        carve(pf.cnt, (size_t) ne);
        carve(pf.off, (size_t) ne);
        carve(pf.list, (size_t) C * nu);
        carve(pf.wl, (size_t) C * nu);
        carve(pf.Lat, (size_t) C * NL);
        carve(pf.Lath, (size_t) C * NL);
        carve(pf.Xe, (size_t) C * NL);
        carve(pf.GU, (size_t) C * ffe);
        carve(pf.He, (size_t) C * ffe);
        carve(pf.Ye, (size_t) C * NL);
        carve(pf.Ylat, (size_t) C * NL);
        carve(pf.Ylh, (size_t) C * NL);
        carve(pf.SU, (size_t) SC * ffs);
        carve(pf.SH, (size_t) SC * ffs);
        carve(pf.W1, pf.w1_elems);
        carve(pf.W2, pf.w2_elems);
        carve(pf.gbuf[0], (size_t) pf.GE * pf.slot_bytes);
        carve(pf.gbuf[1], (size_t) pf.GE * pf.slot_bytes);
        carve(pf.Erows, (size_t) C * embd.rb);
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
    const char* pg = std::getenv("NEM_PF_PAGEABLE");
    if (!pg || std::atoi(pg) == 0) {
        stg.cap = (size_t) 64 << 20;
        stg.on = cudaHostAlloc((void**) &stg.pin[0], stg.cap, cudaHostAllocPortable) == cudaSuccess &&
                 cudaHostAlloc((void**) &stg.pin[1], stg.cap, cudaHostAllocPortable) == cudaSuccess;
        if (stg.on) {
            CK(cudaEventCreateWithFlags(&stg.ev[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stg.ev[1], cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "maya-nem: no pinned staging for the prompt path's copies (the driver's is used)\n");
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
        CK(cudaEventSynchronize(stg.ev[stg.k]));
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
        std::fprintf(stderr, "maya-nem: no F16 rows for type %d\n", m.type);
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
        std::fprintf(stderr, "maya-nem: cuBLAS GEMM %dx%dx%d failed (%d)\n", T, n, k, (int) r);
        std::abort();
    }
}

// The chunk's attention: queries pf.Qh ([Hkv][T][G][hd], positions pos0 ..), keys / values the cache's rows 0 ..
// nkeys - 1; out [T][H][hd].  Causal, no window, no bias.
void Engine::Impl::pf_attention(int T, int pos0, const __half* Kb, const __half* Vb, int nkeys, float* out) {
    const int H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, G = H / Hkv;
    const float scale = 1.0f / std::sqrt((float) hd), one = 1.0f, zero = 0.0f;
    auto r8 = [](int v) { return (v + 7) & ~7; };
    auto hi = [&](int tb) { return std::min(nkeys, pos0 + tb); };
    int ta = 0;
    while (ta < T) {
        int tq = T - ta;
        while (tq > 1 && (size_t) G * tq * (size_t) r8(hi(ta + tq)) > pf.s_elems) tq = (tq + 1) / 2;
        const int tb = ta + tq, nk = hi(tb), ld = r8(nk), M = G * tq;
        for (int gi = 0; gi < Hkv; ++gi) {
            const __half* Qt = pf.Qh + (((size_t) gi * T + ta) * G) * hd;
            const __half* Kt = Kb + (size_t) gi * hd;
            const __half* Vt = Vb + (size_t) gi * hd;
            cublasStatus_t r = cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, M, hd, &scale, Kt, CUDA_R_16F, kvw,
                                            Qt, CUDA_R_16F, hd, &zero, pf.S, CUDA_R_32F, ld, CUBLAS_COMPUTE_32F,
                                            CUBLAS_GEMM_DEFAULT);
            if (r != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "maya-nem: cuBLAS scores %dx%d failed (%d)\n", M, nk, (int) r);
                std::abort();
            }
            k::attn_softmax(pf.S, ld, pf.P, ld, M, nk, G, gi, ta, pos0, 0, 0, nullptr, nullptr, 0, H, 0, nullptr, s);
            r = cublasGemmEx(pf.cb, CUBLAS_OP_N, CUBLAS_OP_N, hd, M, nk, &one, Vt, CUDA_R_16F, kvw, pf.P, CUDA_R_16F, ld,
                             &zero, pf.Ot, CUDA_R_32F, hd, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (r != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "maya-nem: cuBLAS attention output %dx%d failed (%d)\n", M, nk, (int) r);
                std::abort();
            }
            k::attn_scatter(pf.Ot, out, M, G, gi, ta, H, hd, s);
        }
        ta = tb;
    }
}

// A Mamba-2 layer over the chunk (pf.Ah its normed input) into pf.Y, in sub-chunks of SC tokens
void Engine::Impl::pf_mamba(Layer& l, int T) {
    const int E = g.n_embd, H = g.n_ssm_head, P = ssm_P, Gr = g.n_group, N = g.d_state;
    f16w(l.ssm_in, pf.W1);
    f16w(l.ssm_out, pf.W2);
    for (int r0 = 0; r0 < T; r0 += pf.SC) {
        const int n = std::min(pf.SC, T - r0);
        gemm(pf.Ah + (size_t) r0 * E, E, pf.W1, E, pf.ZX, d_in_proj, n, d_in_proj, E);
        k::mamba_conv_chunk(pf.ZX + g.d_inner, d_in_proj, pf.XBC, l.conv_st, l.conv_w, l.conv_b, n, conv_ch, g.d_conv, s);
        k::mamba_scan(pf.XBC, conv_ch, pf.ZX, d_in_proj, dt_off, l.dt_b, l.A, l.D, l.ssm_st, pf.Ys, n, H, P, Gr, N, s);
        k::group_rmsnorm(pf.Ys, l.ssm_norm, n, Gr, g.d_inner / Gr, g.eps, s);
        k::to_f16(pf.Ys, pf.Ysh, (size_t) n * g.d_inner, s);
        gemm(pf.Ysh, g.d_inner, pf.W2, g.d_inner, pf.Y + (size_t) r0 * E, E, n, E, g.d_inner);
    }
}

void Engine::Impl::pf_attn(Layer& l, int T, int pos0) {
    const int H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim, qw = H * hd;
    lin(l.wq, pf.Ah, T, pf.Q);
    lin(l.wk, pf.Ah, T, pf.Kf);
    lin(l.wv, pf.Ah, T, pf.Vf);
    k::to_f16(pf.Kf, l.K + (size_t) pos0 * kvw, (size_t) T * kvw, s);
    k::to_f16(pf.Vf, l.V + (size_t) pos0 * kvw, (size_t) T * kvw, s);
    k::q_group(pf.Q, pf.Qh, T, H, Hkv, hd, s);
    float* AO = pf.Q;   // (free once q_group has its copy)
    pf_attention(T, pos0, l.K, l.V, pos0 + T, AO);
    k::to_f16(AO, pf.AOh, (size_t) T * qw, s);
    lin(l.wo, pf.AOh, T, pf.Y);
}

void Engine::Impl::pf_moe(const Layer& l, int T) {
    const int E = g.n_embd, ne = g.n_expert, nu = g.n_used, ff = g.n_ff_exp, NL = g.n_lat, ffs = g.n_ff_sh;
    const float one = 1.0f, zero = 0.0f;
    // the router in F32 (llama.cpp: F32 weights), the routing on the GPU
    if (cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, ne, T, E, &one, (const float*) l.gate_inp.w, E, pf.A, E, &zero,
                    pf.RL, ne) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-nem: cuBLAS router failed\n");
        std::abort();
    }
    k::route_sig(pf.RL, ne, l.exp_bias_d, T, ne, nu, g.exp_norm, g.exp_scale, pf.sel, pf.wts, s);
    CK(cudaMemsetAsync(pf.cnt, 0, (size_t) ne * sizeof(int), s));
    k::expert_count(pf.sel, T * nu, pf.cnt, s);
    CK(cudaMemcpyAsync(pf.h_cnt, pf.cnt, (size_t) ne * sizeof(int), cudaMemcpyDeviceToHost, s));
    // the latent input
    lin(l.lat_down, pf.Ah, T, pf.Lat);
    k::to_f16(pf.Lat, pf.Lath, (size_t) T * NL, s);
    CK(cudaStreamSynchronize(s));
    pf.h_off[0] = 0;
    for (int e = 0; e < ne; ++e) pf.h_off[e + 1] = pf.h_off[e] + pf.h_cnt[e];
    CK(cudaMemcpyAsync(pf.off, pf.h_off, (size_t) ne * sizeof(int), cudaMemcpyHostToDevice, s));
    k::expert_lists(pf.sel, pf.wts, T, nu, nu, ne, pf.off, pf.list, pf.wl, s);
    CK(cudaMemsetAsync(pf.Ylat, 0, (size_t) T * NL * sizeof(float), s));
    const CpuLayer& c = l.cl;
    const size_t ub = ub_of(c), db = db_of(c);
    const int il = (int) (&l - L.data()), GE = pf.GE;
    pf_score(il, pf.h_cnt, T);
    // The experts in groups of GE: a group's used experts that are not in VRAM copied into one of two group buffers
    // (each run of consecutive ones in one copy a tensor), laid out [the ups | the downs] of the group's experts.
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
                stage_copy(B + at * ub, c.up + (size_t) e * ub, n * ub);
                stage_copy(B + GE * ub + at * db, c.down + (size_t) e * db, n * db);
                pf.copy_bytes += (double) (n * (ub + db));
                e = r;
            }
            pf.copy_ms += now_ms() - tc;
            CK(cudaEventRecord(pf.gready[bi], pf.cs));
            CK(cudaStreamWaitEvent(s, pf.gready[bi], 0));
        }
        for (int e = e0; e < e1; ++e) {
            const int rows = pf.h_cnt[e], o0 = pf.h_off[e];
            if (rows == 0) continue;
            const uint8_t* blob = ec_blob(il, e);
            const size_t at = (size_t) (e - e0);
            const uint8_t* wu = blob ? blob : B + at * ub;
            const uint8_t* wd = blob ? blob + ub : B + GE * ub + at * db;
            if (!iq::f16_rows(c.tu, wu, NL, ff, (uint16_t*) pf.W1, NL, s) ||
                !iq::f16_rows(c.td, wd, ff, NL, (uint16_t*) pf.W2, ff, s)) {
                std::fprintf(stderr, "maya-nem: no F16 rows for the experts' types %d / %d\n", c.tu, c.td);
                std::abort();
            }
            iq::gather_f16((const uint16_t*) pf.Lath, NL, pf.list + o0, rows, NL, (uint16_t*) pf.Xe, s);
            gemm(pf.Xe, NL, pf.W1, NL, pf.GU, ff, rows, ff, NL);
            k::relu2_f16(pf.GU, ff, rows, ff, pf.He, s);
            gemm(pf.He, ff, pf.W2, ff, pf.Ye, NL, rows, NL, ff);
            k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, NL, pf.Ylat, s);
        }
        if (bi >= 0) CK(cudaEventRecord(pf.gfreed[bi], s));
    }
    // the latent sum up to the model's width, then the shared expert on top (in sub-chunks)
    k::to_f16(pf.Ylat, pf.Ylh, (size_t) T * NL, s);
    lin(l.lat_up, pf.Ylh, T, pf.Y);
    f16w(l.sh_up, pf.W1);
    f16w(l.sh_down, pf.W2);
    for (int r0 = 0; r0 < T; r0 += pf.SC) {
        const int n = std::min(pf.SC, T - r0);
        gemm(pf.Ah + (size_t) r0 * E, E, pf.W1, E, pf.SU, ffs, n, ffs, E);
        k::relu2_f16(pf.SU, ffs, n, ffs, pf.SH, s);
        gemm(pf.SH, ffs, pf.W2, ffs, pf.Y + (size_t) r0 * E, E, n, E, ffs, 1.0f, 1.0f);
    }
}

void Engine::Impl::chunk(const int* ids, int T, bool want_logits) {
    const int E = g.n_embd;
    const int pos0 = pos;
    ec_kv_reserve(pos0 + T);
    // the chunk's embedding rows gathered from the mapping into the pinned staging (or one by one without it)
    const size_t rb = embd.rb;
    if (stg.on && stg.cap >= rb) {
        // (in staging-buffer pieces: a 16K chunk's Q8_0 rows are 71 MB, more than a buffer, and went up a row a call)
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
            CK(cudaMemcpyAsync(pf.Erows + (size_t) t * rb, embd.w + (size_t) ids[t] * rb, rb, cudaMemcpyHostToDevice, s));
    }
    for (int t = 0; t < T; ++t) pf.h_ids[t] = t;
    CK(cudaMemcpyAsync(pf.ids, pf.h_ids, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, s));
    if (!k::embed_rows(pf.Erows, embd.type, embd.rb, pf.ids, T, E, pf.X, s)) {
        std::fprintf(stderr, "maya-nem: no embedding rows for type %d\n", embd.type);
        std::abort();
    }
    static const bool times = std::getenv("NEM_PF_TIMES") != nullptr;
    double tk[3] = {0, 0, 0};
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const double tl = times ? now_ms() : 0.0;
        k::rmsnorm(pf.X, l.norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        if (l.kind == kMamba) pf_mamba(l, T);
        else if (l.kind == kAttn) pf_attn(l, T, pos0);
        else pf_moe(l, T);
        k::add(pf.X, pf.Y, T * E, s);
        if (times) {
            CK(cudaStreamSynchronize(s));
            tk[l.kind] += now_ms() - tl;
        }
    }
    if (times)
        std::fprintf(stderr,
                     "maya-nem: a chunk of %d: Mamba-2 %.0f ms, attention %.0f ms, MoE %.0f ms (copies blocked %.0f ms "
                     "so far); %llu page faults, working set %.1f GB\n",
                     T, tk[kMamba], tk[kAttn], tk[kMoe], pf.copy_ms, page_faults(), working_set_gb());
    pos = pos0 + T;
    if (want_logits) {
        k::rmsnorm(pf.X + (size_t) (T - 1) * E, out_norm, a, 1, E, g.eps, s);
        quant(qa, a, 1, E);
        mv(output, qa, a, 0, logits);
        CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    CK(cudaStreamSynchronize(s));
}

const float* Engine::Impl::prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx) {
    if (n <= 0 || pos + n > max_ctx) return nullptr;
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= g.vocab) return nullptr;
    const double t0 = now_ms();
    const float* lg = nullptr;
    // a short prompt (a turn's few new tokens): the token path's experts on the CPU beat copying the ones it chooses
    int min_chunk = 32;
    if (const char* v = std::getenv("NEM_MIN_CHUNK")) min_chunk = std::max(1, std::atoi(v));
    if (n < min_chunk) {
        for (int i = 0; i < n; ++i) lg = step(ids[i]);
        if (progress) progress(ctx, n);
        return lg;
    }
    ec_reclaim();
    pf_open = true;
    pf_tokens_open += n;
    for (int c0 = 0; c0 < n; c0 += pf.C) {
        const int T = std::min(pf.C, n - c0);
        chunk(ids + c0, T, c0 + T == n);
        if (progress) progress(ctx, c0 + T);
    }
    ec_fill();
    CK(cudaStreamSynchronize(s));
    st.pf_ms += now_ms() - t0;
    st.pf_tokens += n;
    st.pf_copy_gb = pf.copy_bytes / 1e9;
    return h_logits;
}

// ------------------------------------------------------------------------------------------------ the token path
void Engine::Impl::tok_mamba(Layer& l) {
    const int E = g.n_embd;
    k::rmsnorm(x, l.norm, a, 1, E, g.eps, s);
    quant(qa, a, 1, E);
    mv(l.ssm_in, qa, a, 0, zx);
    k::mamba_conv_step(zx + g.d_inner, l.conv_st, l.conv_w, l.conv_b, conv_ch, g.d_conv, s);
    k::mamba_scan(zx + g.d_inner, conv_ch, zx, d_in_proj, dt_off, l.dt_b, l.A, l.D, l.ssm_st, y, 1, g.n_ssm_head, ssm_P,
                  g.n_group, g.d_state, s);
    k::group_rmsnorm(y, l.ssm_norm, 1, g.n_group, g.d_inner / g.n_group, g.eps, s);
    quant(qy, y, 1, g.d_inner);
    mv(l.ssm_out, qy, y, 0, ao);
    k::add(x, ao, E, s);
}

void Engine::Impl::tok_attention(Layer& l) {
    const int E = g.n_embd, H = g.n_head, Hkv = g.n_head_kv, hd = g.head_dim;
    k::rmsnorm(x, l.norm, a, 1, E, g.eps, s);
    quant(qa, a, 1, E);
    {
        const DMat* ms[3] = {&l.wq, &l.wk, &l.wv};
        float* ys[3] = {q, kk, vv};
        mv_jobs(ms, ys, 3, qa, a);
    }
    k::kv_store_dev(kk, vv, l.K, l.V, dpos, 0, kvw, s);
    k::attn_decode(q, l.K, l.V, l.kv_rows, 0, dpos, 0, nullptr, 0, H, Hkv, hd, 1.0f / std::sqrt((float) hd), o_,
                   attn_scr, s);
    quant(qo, o_, 1, H * hd);
    mv(l.wo, qo, o_, 0, ao);
    k::add(x, ao, E, s);
}

// A MoE layer's first part: a's norm and int8 rows, the router's logits and the latent to the host, the latent's int8
// rows (for the experts in VRAM).
void Engine::Impl::moe_pre(Layer& l) {
    const int E = g.n_embd, ne = g.n_expert, NL = g.n_lat;
    k::rmsnorm(x, l.norm, a, 1, E, g.eps, s);
    quant(qa, a, 1, E);
    mv(l.gate_inp, qa, a, 0, rlog);   // (F32: read as F32)
    mv(l.lat_down, qa, a, 0, lat);
    CK(cudaMemcpyAsync(h_rlog, rlog, (size_t) ne * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_lat, lat, (size_t) NL * sizeof(float), cudaMemcpyDeviceToHost, s));
    quant(ql, lat, 1, NL);
}

// A MoE layer's end: the CPU lane's sum (and which lanes ran) up to the device, the latent sum up to the model's width,
// x += it and the shared expert's.
void Engine::Impl::moe_tail(Layer& l) {
    const int E = g.n_embd, NL = g.n_lat;
    CK(cudaMemcpyAsync(d_tail, h_tail, ((size_t) 4 + NL) * sizeof(float), cudaMemcpyHostToDevice, s));
    k::lat_sum(d_tail + 4, ec.routed, (const int*) d_tail, lat_s, NL, s);
    quant(ql, lat_s, 1, NL);
    mv(l.lat_up, ql, lat_s, 0, mo);
    k::add2(x, mo, sho, E, s);
}

// The routing on the host (the logits there after the sync); the chosen experts in VRAM on the GPU and the shared
// expert, the others on the CPU lane, both at once; the CPU lane's sum and the lanes' flags in h_tail.
void Engine::Impl::moe_mid(Layer& l, int il, double& cpu_ms) {
    const int ne = g.n_expert, nu = g.n_used, NL = g.n_lat, ff = g.n_ff_exp;
    const double c0 = now_ms();
    // routing: the top n_used of sigmoid(logit) + bias, ties to the lower id; weights their sigmoids, normalized, x scale
    int sel[32];
    float wv[32];
    {
        float sc[512], sg[512];
        int idx[512];
        for (int e = 0; e < ne; ++e) {
            sg[e] = 1.0f / (1.0f + std::exp(-h_rlog[e]));
            sc[e] = sg[e] + l.exp_bias[(size_t) e];
            idx[e] = e;
        }
        std::partial_sort(idx, idx + nu, idx + ne,
                          [&](int a1, int b1) { return sc[a1] > sc[b1] || (sc[a1] == sc[b1] && a1 < b1); });
        float tot = 0.0f;
        for (int i = 0; i < nu; ++i) {
            sel[i] = idx[i];
            wv[i] = sg[idx[i]];
            tot += wv[i];
        }
        const float d = g.exp_norm ? std::max(tot, 6.103515625e-5f) : 1.0f;
        for (int i = 0; i < nu; ++i) wv[i] = wv[i] / d * g.exp_scale;
    }
    int gslot[32], csel[32], hslot[32];
    float gw[32], cw[32], hw[32];
    int ng = 0, nc = 0, nh = 0;
    for (int i = 0; i < nu; ++i) {
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
        } else {
            csel[nc] = sel[i];
            cw[nc++] = wv[i];
        }
    }
    ec.lookups += nu;
    ec.hits += ng;
    hc.hits += nh;
    float* h_routed = h_tail + 4;
    if (nc) lane.begin(*pool, l.cl, NL, ff, csel, cw, nc, h_lat, h_routed);
    if (nh) hc_experts(l, hslot, hw, nh);   // (the helper card starts on its share first: it is the slower one)
    if (ng) gpu_experts(l, gslot, gw, ng);
    // the shared expert (on a's int8 rows)
    mv(l.sh_up, qa, a, 0, shu);
    k::relu2(shu, (size_t) g.n_ff_sh, s);
    quant(qs, shu, 1, g.n_ff_sh);
    mv(l.sh_down, qs, shu, 0, sho);
    if (nc) lane.end();
    const double c1 = now_ms();
    st.ms_lane += c1 - c0;
    st.lane_layers += nc > 0;
    st.helper_layers += nh > 0;
    if (nh) {   // the helper's sum joins the CPU lane's on the host (the device adds that to its own in moe_tail)
        {
            OnDev od(hc.dev);
            CK(cudaStreamSynchronize(hc.s));
        }
        if (nc)
            for (int i = 0; i < NL; ++i) h_routed[i] += hc.h_out[i];
        else std::memcpy(h_routed, hc.h_out, (size_t) NL * sizeof(float));
        st.ms_helper += now_ms() - c1;
    }
    cpu_ms += now_ms() - c0;
    int* flags = (int*) h_tail;
    flags[0] = nc > 0 || nh > 0;
    flags[1] = ng > 0;
}

// the token's row (h_erow, written by the host) and its position (h_pos) up; x = the row
void Engine::Impl::tok_embed() {
    const int E = g.n_embd;
    CK(cudaMemcpyAsync(d_erow, h_erow, embd.rb, cudaMemcpyHostToDevice, s));
    if (is_float(embd.type)) {
        k::row_to_f32(d_erow, embd.type, x, E, s);
    } else {
        if (!iq::f16_rows(embd.type, d_erow, E, 1, e16, E, s)) {
            std::fprintf(stderr, "maya-nem: no row decoder for the embedding's type %d\n", embd.type);
            std::abort();
        }
        k::f16_to_f32((const __half*) e16, x, E, s);
    }
    CK(cudaMemcpyAsync(dpos, h_pos, sizeof(int), cudaMemcpyHostToDevice, s));
}

void Engine::Impl::tok_head() {
    const int E = g.n_embd;
    k::rmsnorm(x, out_norm, a, 1, E, g.eps, s);
    quant(qa, a, 1, E);
    mv(output, qa, a, 0, logits);
    CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
}

void Engine::Impl::seg(int j) {
    const int M = (int) moes.size();
    int from = 0;
    if (j == 0) {
        tok_embed();
    } else {
        moe_tail(L[(size_t) moes[(size_t) j - 1]]);
        from = moes[(size_t) j - 1] + 1;
    }
    const int to = j < M ? moes[(size_t) j] : g.n_layer;
    for (int il = from; il < to; ++il) {
        Layer& l = L[(size_t) il];
        if (l.kind == kMamba) tok_mamba(l);
        else tok_attention(l);
    }
    if (j < M) moe_pre(L[(size_t) moes[(size_t) j]]);
    else tok_head();
}

void Engine::Impl::tok_graphs() {
    graphs = 0;
    const char* v = std::getenv("NEM_GRAPHS");
    if ((v && std::atoi(v) == 0) || std::getenv("NEM_TRACE")) return;
    const int n = (int) moes.size() + 1;
    gx.assign((size_t) n, nullptr);
    bool ok = true;
    for (int j = 0; j < n && ok; ++j) {
        cudaGraph_t gr = nullptr;
        ok = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (!ok) break;
        seg(j);
        ok = cudaStreamEndCapture(s, &gr) == cudaSuccess && gr;
        ok = ok && cudaGraphInstantiate(&gx[(size_t) j], gr, 0) == cudaSuccess;
        if (gr) cudaGraphDestroy(gr);
    }
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "maya-nem: the token path's CUDA graphs could not be made; it launches its kernels one by one\n");
        for (auto& e : gx)
            if (e) cudaGraphExecDestroy(e);
        gx.clear();
        return;
    }
    graphs = 1;
}

const float* Engine::Impl::step(int token) {
    if (pos >= max_ctx || token < 0 || token >= g.vocab) return nullptr;
    const double t0 = now_ms();
    double cpu_ms = 0.0;
    if (pf_open) {   // the first token after a prompt: the prompt path's buffers lent to the cache again, and filled
        pf_open = false;
        if (ec.arena_end > ec.own) {
            ec.lent = true;
            ec_fill();
        }
        // The prompt changed the scores, and the main cache with them: the helper follows, with more copies after a
        // longer prompt (an eighth of its tokens, 64 .. fill_cap: a short turn waits ~30 ms for them, a long prompt
        // a second or two on top of its own reading).
        hc_update(std::min(hc.fill_cap, std::max(64, (int) std::min<long>(pf_tokens_open / 8, INT_MAX))), false, false);
        pf_tokens_open = 0;
    }
    if (moes.empty())
        for (int il = 0; il < g.n_layer; ++il)
            if (L[(size_t) il].kind == kMoe) moes.push_back(il);
    if (graphs < 0) tok_graphs();
    ec_kv_reserve(pos + 1);
    // (NEM_HELPER_KEEPALIVE) the helper's keep-alive kernel again once the last one has ended
    if (hc.ks && cudaStreamQuery(hc.ks) == cudaSuccess) {
        OnDev od(hc.dev);
        k::keepalive(hc.ka_ns, hc.ks);
        (void) cudaStreamQuery(hc.ks);   // (submitted now: WDDM holds launches until something does)
    }
    // the token's row and position, for segment 0
    std::memcpy(h_erow, embd.w + (size_t) token * embd.rb, embd.rb);
    h_pos[0] = pos;
    static const bool trace = std::getenv("NEM_TRACE") != nullptr;
    run_seg(0);
    for (size_t j = 0; j < moes.size(); ++j) {
        const bool staging = ec_stage_begin(ec.piece);   // (a piece of the swaps' copies while the GPU works)
        CK(cudaStreamSynchronize(s));
        if (staging) ec_stage_end();
        if (trace) {
            std::fprintf(stderr, "[trace] pos %d MoE layer %d\n", pos, moes[j]);
            std::fflush(stderr);
        }
        moe_mid(L[(size_t) moes[j]], moes[j], cpu_ms);
        run_seg((int) j + 1);
    }
    const bool staging = ec_stage_begin(4 * ec.piece);   // (the head's GEMV is the token's longest wait)
    CK(cudaStreamSynchronize(s));
    if (staging) ec_stage_end();
    ++pos;
    // the scores decay as tokens are written; a few swaps every few tokens
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
    st.ms_total += now_ms() - t0;
    st.ms_cpu += cpu_ms;
    ++st.tokens;
    return h_logits;
}

}  // namespace nem
