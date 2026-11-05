// src/ling/ling_model.cu - maya-ling's model (ling_model.hpp): the GGUF read, the weights placed, a token's step and a
// prompt's chunks through Ling-3.1-flash's layers - llama.cpp's bailingmoe3 graph (src/models/bailingmoe3.cpp) step by
// step:
//
//   x = embd[token]
//   per layer:  a = rmsnorm(x, attn_norm)
//     KDA:  q, k, v = silu(conv(W a)); q, k L2-normed a head; g = -5 sigmoid(exp(A_log) (W_f a + dt_bias));
//           beta = sigmoid(W_beta a); o = the delta rule's state read with q (ling_kernels.cuh);
//           x += W_o (rmsnorm(o) sigmoid(W_g a))                                          (63 layers)
//     MLA:  q = W_qb rmsnorm(W_qa a); kv = W_kva a; the cache row [rmsnorm(kv[0:512]), rope(kv[512:576])];
//           o_h = v_b_h softmax(q'_h K^T / sqrt(192)) K[:, 0:512] with q'_h = [k_b_h q_nope_h, rope(q_pe_h)];
//           x += W_o (o sigmoid(W_gate a))                                               (layers 7, 15, .. 71)
//     f = rmsnorm(x, ffn_norm);  x += ffn(f)
//   ffn: the first leading_dense_block_count layers a SwiGLU FFN; the others the MoE - the top 8 of sigmoid(logits) +
//        exp_probs_b in the top 4 of 8 groups (by their top-2 sums), weighted by the chosen sigmoids normalized x 2.5,
//        plus the shared expert; the SwiGLUs clamped where swiglu_clamp_exp / _shexp say
//   logits = output(rmsnorm(x, output_norm))
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "ling_model.hpp"

#include "ling_cpu.hpp"
#include "ling_kernels.cuh"

#include "../ds4/ds4_iq.cuh"
#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

namespace ling {

namespace iq = strata::ds4::gguf::iq;

namespace {

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "maya-ling: CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,            \
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
    bool mla = false, dense = false;
    float *attn_norm = nullptr, *ffn_norm = nullptr;
    DMat wo;
    // KDA
    DMat wq, wk, wv, wf, wg, wbeta;
    float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr, *dt_bias = nullptr, *a = nullptr, *o_norm = nullptr;
    float* conv = nullptr;   // the q / k / v convolutions' states: 3 x [C][K - 1]
    float* S = nullptr;      // the recurrence's state [H][128 (value)][128 (key)]
    // MLA
    DMat q_a, q_b, kv_a, k_b, v_b, wgate;   // k_b: H x 512 rows of 128 (q_nope -> the cache's 512); v_b: H x 128 of 512
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    __half* kv = nullptr;   // the cache: kv_rows rows of 576 (F16)
    int kv_rows = 0;
    // the FFN
    DMat gate, up, down;           // dense layers
    DMat gate_inp;                 // MoE layers: n_expert rows
    std::vector<float> exp_bias;   // (host: a token's routing is done there)
    float* exp_bias_d = nullptr;   // (the prompt path routes on the GPU)
    DMat sh_gate, sh_up, sh_down;
    CpuLayer cl;
    float clamp_exp = 0.0f, clamp_sh = 0.0f;
};

struct Engine::Impl {
    Options o;
    Geo g;
    int max_ctx = 0, pos = 0, C = 0;   // C: a KDA layer's channels (n_head x head_dim)
    float theta_scale = 1.0f, mla_scale = 1.0f;
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Tensor> tens;
    cudaStream_t s = nullptr;
    std::vector<Layer> L;
    DMat embd, output;
    // The embedding's rows stay in the GGUF's mapping (a token needs one): its row copied up through h_erow, a prompt
    // chunk's gathered into the pinned staging and copied to pf.Erows.  LING_EMBD_VRAM=1: the table in VRAM.
    bool embd_host = true;
    uint8_t *h_erow = nullptr, *d_erow = nullptr;
    float* out_norm = nullptr;
    std::unique_ptr<Pool> pool;
    CpuLane lane;   // the token path's CPU experts, a layer at a time
    size_t dev_bytes = 0;

    // device scratch (the token path)
    float *x = nullptr, *a = nullptr, *f = nullptr, *ao = nullptr, *fo = nullptr;
    float *qr = nullptr, *kr = nullptr, *vr = nullptr, *fr = nullptr, *gr = nullptr, *br = nullptr;
    float *qn = nullptr, *kn = nullptr, *vc = nullptr, *ge = nullptr, *beta = nullptr, *ko = nullptr;
    float *qa = nullptr, *kva = nullptr, *gh = nullptr, *qf = nullptr, *qabs = nullptr, *mo = nullptr, *mob = nullptr;
    float *fg = nullptr, *fu = nullptr, *fh = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *so = nullptr;
    float *logits = nullptr, *rlog = nullptr, *attn_scr = nullptr;
    int* dpos = nullptr;
    int attn_nsp = 128;
    float* d_tail = nullptr;   // a MoE layer's end: which lanes ran (2 ints, 2 spare), the CPU lane's sum (n_embd)
    int8_t *xq_a = nullptr, *xq_f = nullptr, *xq_h = nullptr, *xq_m = nullptr;   // int8 rows: the attention's, f's,
    float *xd_a = nullptr, *xd_f = nullptr, *xd_h = nullptr, *xd_m = nullptr;    // an FFN's hidden, MLA's inner ones
    // host (pinned)
    float *h_f = nullptr, *h_rlog = nullptr, *h_tail = nullptr, *h_logits = nullptr;
    int* h_pos = nullptr;
    Stats st;
    // (LING_STEP_TIMES) the token path's host time by phase, printed every 64 tokens
    struct PhaseTimes {
        double att = 0, route = 0, glaunch = 0, cpu = 0, tail = 0, head = 0, adapt = 0, embed = 0;
        long cpu_experts = 0, tokens = 0;
    } pt;
    // (LING_ROUTE_LOG=FILE) the routing as it goes, for replaying the expert cache's choices offline: 'T' + a token's
    // n_used experts a MoE layer (uint16 each); 'P' + layer + a prompt chunk's counts an expert (uint16); 'E' a
    // prompt's end (the cache refilled there)
    FILE* route_log = nullptr;
    std::vector<uint16_t> route_tok;
    bool pf_open = false;   // a prompt read through the prompt path, no token written since (its buffers not lent)
    // (Engine::snapshot) the KDA layers' states and convolutions at snap_pos, one buffer in VRAM (-1: none taken)
    uint8_t* snap = nullptr;
    size_t snap_bytes = 0;
    int snap_pos = -1;

    template <class T> T* dalloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T)));
        CK(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
        dev_bytes += n * sizeof(T);
        return (T*) p;
    }
    template <class T> T* halloc(size_t n) {
        void* p = nullptr;
        CK(cudaMallocHost(&p, std::max<size_t>(1, n) * sizeof(T)));
        std::memset(p, 0, std::max<size_t>(1, n) * sizeof(T));
        return (T*) p;
    }

    const Tensor* find(const std::string& name) const {
        auto it = tens.find(name);
        return it == tens.end() ? nullptr : &it->second;
    }

    // a matrix [k, n] (or a 3D bank [k, n, count] as n x count rows when count > 0) into VRAM
    bool mat(const std::string& name, int k, int n, DMat& m, std::string& err, int count = 0) {
        const Tensor* t = find(name);
        if (!t) {
            err = "the GGUF has no " + name;
            return false;
        }
        const auto& sh = t->info->shape;
        const bool ok = count > 0 ? sh.size() >= 3 && (int64_t) sh[0] == k && (int64_t) sh[1] == n && (int64_t) sh[2] == count
                                  : sh.size() >= 2 && (int64_t) sh[0] == k && (int64_t) sh[1] == n &&
                                        (sh.size() == 2 || sh[2] == 1);
        if (!ok) {
            err = name + ": shape not [" + std::to_string(k) + ", " + std::to_string(n) +
                  (count > 0 ? ", " + std::to_string(count) : std::string()) + "]";
            return false;
        }
        const int type = (int) t->info->type;
        if (!is_float(type) && !iq::mv_supported(type)) {
            err = name + ": type " + std::string(t->info->type_name()) + " has no kernel here";
            return false;
        }
        m.type = type;
        m.k = k;
        m.n = n * std::max(1, count);
        m.rb = row_bytes_of(type, k);
        if (m.rb == 0) {
            err = name + ": rows of " + std::to_string(k) + " values do not split into its type's blocks";
            return false;
        }
        const size_t bytes = m.rb * (size_t) m.n;
        uint8_t* d = dalloc<uint8_t>(bytes);
        CK(cudaMemcpy(d, t->data, bytes, cudaMemcpyHostToDevice));
        m.w = d;
        trim(t->data, bytes);
        return true;
    }
    // The mapping's pages of a tensor now in VRAM out of the working set (they stay in the file cache, available to
    // the system): ~12 GB of RAM the routed experts' 160 GB would otherwise share the machine's 192 with.
    // LING_TRIM=0: kept.
    static void trim(const void* p, size_t n) {
#ifdef _WIN32
        static const bool on = !std::getenv("LING_TRIM") || std::atoi(std::getenv("LING_TRIM")) != 0;
        if (on && n >= ((size_t) 1 << 20)) VirtualUnlock((LPVOID) p, n);   // (not locked: "fails", the pages leave)
#else
        (void) p;
        (void) n;
#endif
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

    // (LING_DUMP=1) a row of the residual stream after a layer's attention or FFN: its RMS and first values, to stderr -
    // the token path's (after each segment) and the prompt path's last token (after each part) compared line by line
    void dump(const char* what, int il, const float* dev, int at) {
        static const bool on = std::getenv("LING_DUMP") != nullptr;
        if (!on) return;
        std::vector<float> v((size_t) g.n_embd);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(v.data(), dev, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double ss = 0;
        for (float x0 : v) ss += (double) x0 * x0;
        std::fprintf(stderr, "[dump] pos %d %s %2d rms %.6g  %.5g %.5g %.5g %.5g\n", at, what, il,
                     std::sqrt(ss / v.size()), v[0], v[1], v[2], v[3]);
    }

    // ---------------------------------------------------------------- the GPU's products (the token path)
    static iq::MvJob job(const DMat& m, int row0, int rows, const int8_t* xq, const float* xd, float* y,
                         float alpha = 1.0f) {
        iq::MvJob j;
        j.w = m.w + (size_t) row0 * m.rb;
        j.ld = m.rb;
        j.type = m.type;
        j.k = m.k;
        j.n = rows;
        j.ncol = 1;
        j.xq = xq;
        j.xd = xd;
        j.y = y;
        j.ycs = (size_t) rows;
        j.alpha = alpha;
        return j;
    }
    void run_jobs(const iq::MvJob* j, int n) {
        if (n > 0 && !iq::mv(j, n, s)) {
            std::fprintf(stderr, "maya-ling: no GEMV for these jobs (type %d, k %d)\n", j[0].type, j[0].k);
            std::abort();
        }
    }
    // y = W x: x's int8 rows (xq, xd: q8 of x), or x itself (F32) for a float matrix
    void mv(const DMat& m, const float* xf, const int8_t* xq, const float* xd, float* y, float alpha = 1.0f) {
        if (is_float(m.type)) {
            k::gemv_f(m.w, m.type, m.rb, xf, y, m.n, m.k, s);
            return;
        }
        const iq::MvJob j = job(m, 0, m.n, xq, xd, y, alpha);
        run_jobs(&j, 1);
    }
    // several y[i] = W[i] x in one launch where they share x's int8 rows (float matrices: one by one)
    void mv_many(const DMat* const* m, float* const* y, int n, const float* xf, const int8_t* xq, const float* xd) {
        iq::MvJob j[16];
        int nj = 0;
        for (int i = 0; i < n; ++i) {
            if (is_float(m[i]->type)) k::gemv_f(m[i]->w, m[i]->type, m[i]->rb, xf, y[i], m[i]->n, m[i]->k, s);
            else j[nj++] = job(*m[i], 0, m[i]->n, xq, xd, y[i]);
        }
        run_jobs(j, nj);
    }

    bool load(const Options& opt, std::string& err);
    const float* step(int token);
    void tok_kda(Layer& l);
    void tok_mla(Layer& l);
    void tok_dense_ffn(const Layer& l);
    void tok_tail();
    void tok_router(const Layer& l);
    void tok_head();
    void tok_seg(int seg);   // the graphs' bodies: 0 the embedding .. the first MoE layer's router, il, n_layer: head
    void tok_graphs();
    size_t tail_bytes() const { return ((size_t) 4 + g.n_embd) * sizeof(float); }
    std::vector<cudaGraphExec_t> gx;   // a segment's graph (tok_graphs): [0] the start, [il] a MoE layer's, [n_layer] head
    int graphs = -1;                   // -1: not made yet, 0: none, 1: made
    void route_host(const Layer& l, const float* rl, int* sel, float* w) const;

    // ---------------------------------------------------------------- the prompt path (prefill)
    // A chunk of up to C tokens a layer at a time on the GPU: the dense matrices decoded to F16 and multiplied by
    // cuBLAS (F32 sums); a layer's attention in pieces of `sub` tokens (the KDA recurrence token by token over them,
    // MLA's attention as tiles of scores); the router in F32; each routed expert some token of the chunk chose copied
    // to a VRAM slot (the copy stream, `stage` slots ahead), decoded to F16 and multiplied for its tokens, its output
    // added to theirs times their weights - expert by expert in id order, then the shared expert: the same sums every
    // run.
    struct PF {
        int C = 0, Ca = 0, GE = 0;
        long group_ctr = 0;
        size_t s_elems = 0, slot_bytes = 0, w1_elems = 0, w2_elems = 0;
        uint8_t* arena = nullptr;   // the buffers below: one allocation, lent to the expert cache between prompts
        size_t arena_bytes = 0;
        uint8_t* gbuf[2] = {nullptr, nullptr};
        cudaEvent_t gready[2] = {nullptr, nullptr}, gfreed[2] = {nullptr, nullptr};
        cublasHandle_t cb = nullptr;
        cudaStream_t cs = nullptr;   // the experts' copies
        int *ids = nullptr, *sel = nullptr, *cnt = nullptr, *off = nullptr, *list = nullptr;
        float *X = nullptr, *A = nullptr, *Y = nullptr, *Ye = nullptr, *GU = nullptr, *RL = nullptr, *wts = nullptr,
              *wl = nullptr, *S = nullptr;
        __half *Ah = nullptr, *Xe = nullptr, *He = nullptr, *P = nullptr, *W1 = nullptr, *W2 = nullptr;
        // the attention's buffers (a sub-chunk's): one region, KDA's or MLA's
        uint8_t* U = nullptr;
        size_t u_bytes = 0;
        int *h_ids = nullptr, *h_cnt = nullptr, *h_off = nullptr;
        uint8_t* Erows = nullptr;
        double copy_bytes = 0, copy_ms = 0, moe_ms = 0, attn_ms = 0;
    } pf;
    // The experts' copies to the GPU: the CPU lane's threads copy into two pinned buffers in turn and the DMA takes each
    // from there (the driver's own staging of pageable memory moved 4-15 GB/s on this machine; this held ~14 for
    // maya-ink).  LING_PF_PAGEABLE=1: the driver's staging.
    struct Stage {
        uint8_t* pin[2] = {nullptr, nullptr};
        cudaEvent_t ev[2] = {nullptr, nullptr};
        size_t cap = 0;
        int k = 0;
        bool on = false;
    } stg;
    void stage_copy(uint8_t* dst, const uint8_t* src, size_t len);
    bool pf_alloc(std::string& err);
    size_t u_kda_bytes() const;
    size_t u_mla_bytes() const;
    void f16w(const DMat& m, __half* dst);
    void gemm(const __half* X, int ldx, const __half* W, int ldw, float* Y, int ldy, int T, int n, int k,
              float alpha = 1.0f, float beta = 0.0f);
    void lin(const DMat& m, const __half* xh, const float* xf, int T, float* y);   // y [T][m.n] = x W^T
    void pf_kda(Layer& l, int T);
    void pf_mla(Layer& l, int T, int pos0);
    void pf_dense(const Layer& l, int T);
    void pf_moe(const Layer& l, int T);
    void chunk(const int* ids, int T, bool want_logits);
    const float* prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx);
    // (LING_PF_PREFETCH=1) A chunk of LING_PF_PREFETCH_MIN tokens (4096) or more uses nearly every expert of every
    // layer, so a copier thread stages each MoE layer's experts that are not in the VRAM cache into one of two layer
    // buffers ([the gates | the ups | the downs] of the layer's experts, as the GGUF holds them) while the GPU computes
    // the layers before - the copies (the prompt path's bound: ~150 GB a chunk at what the RAM gives three passes) no
    // longer wait for the attention's compute, nor it for them.  Its own pinned staging, pool and copy stream; the
    // layer's GEMMs read the same weights in the same order as the group path's (the same bits).
    struct Prefetch {
        bool on = false;
        int min_T = 4096;
        size_t bytes = 0;                          // a layer buffer's
        uint8_t* buf[2] = {nullptr, nullptr};      // (in the arena)
        cudaStream_t cs = nullptr;
        cudaEvent_t ready[2] = {nullptr, nullptr}, freed[2] = {nullptr, nullptr};
        uint8_t* pin[2] = {nullptr, nullptr};
        cudaEvent_t pev[2] = {nullptr, nullptr};
        size_t pcap = 0;
        int pk = 0;
        std::unique_ptr<Pool> pool;
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        int staged = -1, freed_upto = -1;          // (layer numbers: the copier's last staged, the GPU's last done)
        double bytes_copied = 0, ms = 0;
        bool active = false;                       // a chunk's copier running
    } pre;
    void pre_stage(uint8_t* dst, const uint8_t* src, size_t len);   // (the copier's stage_copy)
    void pre_run();                                                // (the copier's loop over a chunk's MoE layers)
    void pre_start();
    void pre_join();

    // ---------------------------------------------------------------- the VRAM expert cache
    // Slots of VRAM, each one routed expert's gate | up | down as the GGUF holds them, for the experts used most
    // (score: each time a token or a prompt's token chose it, decaying as tokens are written): a token's experts that
    // are in a slot are multiplied on the GPU while the CPU lane takes the others; a prompt's are not copied again.
    // Filled after each prompt (the best-scored in) and adapted as tokens are written (up to `per` swaps every `every`
    // tokens, each a better score than its slot's expert by `margin` x + 0.5); the scores decay by `decay` a token, so
    // they follow what the conversation uses now; the scores can be kept in a file (Options::profile) for the next
    // start.  maya-ink's cache (src/ink/ink_model.cu, its measured settings), slots sized per class of layers' bytes,
    // the prompt path's arena lent between prompts and the MLA caches' rows past the position lent too.
    // A swap's copy is staged in pieces (LING_CACHE_STAGE_KB, 2 MB) while the CPU waits for a layer's attention on the
    // GPU, and the expert is used from the token after the one its last piece went up in.  LING_CACHE_DEFER=0: copied
    // at the swap.
    static constexpr int kKvSlack = 4096;   // (cache rows reserved past the position)
    struct EC {
        int slots = 0, ne = 0, own = 0, lend_cls = 0, arena_end = 0;   // [own, arena_end): the prompt path's
        bool lent = false;                   // the prompt path's slots usable
        std::vector<int> kv_row;             // slot -> its first cache row (a slot in an MLA cache), else -1
        int kv_hi = 0;                       // the cache rows below it reserved (not usable as slots)
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
        float *gu = nullptr, *h = nullptr, *y = nullptr, *routed = nullptr, *hd = nullptr;
        int8_t* hq = nullptr;
        cudaEvent_t copied = nullptr;
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
        size_t stage_n = 0;
        std::vector<std::pair<const uint8_t*, size_t>> stage_parts;
        std::function<void(int)> stage_fn;
    } ec;
    size_t expert_bytes(int il) const {
        const CpuLayer& c = L[(size_t) il].cl;
        return (size_t) g.n_ff_exp * (c.rg + c.ru) + (size_t) g.n_embd * c.rd;
    }
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
    void ec_src_parts(int key, size_t off, size_t n, std::vector<std::pair<const uint8_t*, size_t>>& out) const;
    const uint8_t* ec_blob(int il, int e) const {
        const int sl = ec.slots ? ec.where[(size_t) il * ec.ne + e] : -1;
        return sl >= 0 ? ec.ptr[(size_t) sl] : nullptr;
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

bool Engine::snapshot() {
    Impl& m = *p;
    if (!m.snap) return false;
    const size_t conv = (size_t) 3 * m.C * (m.g.conv_k - 1) * sizeof(float),
                 state = (size_t) m.g.n_head * 128 * 128 * sizeof(float);
    uint8_t* at = m.snap;
    for (const Layer& l : m.L)
        if (!l.mla) {
            CK(cudaMemcpyAsync(at, l.conv, conv, cudaMemcpyDeviceToDevice, m.s));
            CK(cudaMemcpyAsync(at + conv, l.S, state, cudaMemcpyDeviceToDevice, m.s));
            at += conv + state;
        }
    CK(cudaStreamSynchronize(m.s));
    m.snap_pos = m.pos;
    return true;
}

bool Engine::restore() {
    Impl& m = *p;
    if (!m.snap || m.snap_pos < 0) return false;
    const size_t conv = (size_t) 3 * m.C * (m.g.conv_k - 1) * sizeof(float),
                 state = (size_t) m.g.n_head * 128 * 128 * sizeof(float);
    const uint8_t* at = m.snap;
    for (Layer& l : m.L)
        if (!l.mla) {
            CK(cudaMemcpyAsync(l.conv, at, conv, cudaMemcpyDeviceToDevice, m.s));
            CK(cudaMemcpyAsync(l.S, at + conv, state, cudaMemcpyDeviceToDevice, m.s));
            at += conv + state;
        }
    CK(cudaStreamSynchronize(m.s));
    m.pos = m.snap_pos;
    return true;
}

void Engine::reset() {
    p->pos = 0;
    p->ec.kv_hi = 0;   // (the MLA caches' slots usable again: empty, the next fill or swaps take them)
    const size_t conv = (size_t) 3 * p->C * (p->g.conv_k - 1), state = (size_t) p->g.n_head * 128 * 128;
    for (auto& l : p->L)
        if (!l.mla) {
            CK(cudaMemsetAsync(l.conv, 0, conv * sizeof(float), p->s));
            CK(cudaMemsetAsync(l.S, 0, state * sizeof(float), p->s));
        }
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
// a per-layer array (all its items: metadata() keeps a sample of 64), or a number for every layer
std::vector<double> meta_arr(const strata::GgufFile& f, const std::string& k, int n, double def) {
    std::vector<double> out((size_t) n, def);
    strata::MetaValue v;
    if (!f.get_full(k, v)) return out;
    if (v.type == strata::MetaType::ARRAY) {
        for (int i = 0; i < n && i < (int) v.items.size(); ++i) out[(size_t) i] = v.items[(size_t) i].num();
    } else if (v.is_num()) {
        std::fill(out.begin(), out.end(), v.num());
    }
    return out;
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
    if (!arch || arch->s != "bailingmoe3") {
        err = "not a Ling-3 GGUF (general.architecture " + (arch ? arch->s : std::string("missing")) +
              ", bailingmoe3 expected)";
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
    const std::string A = "bailingmoe3.";
    const int n_all = meta_int(f0, A + "block_count");
    g.n_layer = n_all - std::max(0, meta_int(f0, A + "nextn_predict_layers", 0));
    g.n_embd = meta_int(f0, A + "embedding_length");
    g.n_ff = meta_int(f0, A + "feed_forward_length");
    g.n_head = meta_int(f0, A + "attention.head_count");
    g.head_dim = meta_int(f0, A + "kda.head_dim");
    g.eps = meta_f(f0, A + "attention.layer_norm_rms_epsilon", 1e-6f);
    g.n_expert = meta_int(f0, A + "expert_count");
    g.n_used = meta_int(f0, A + "expert_used_count");
    g.n_group = std::max(1, meta_int(f0, A + "expert_group_count", 1));
    g.n_group_used = std::max(1, meta_int(f0, A + "expert_group_used_count", g.n_group));
    g.n_ff_exp = meta_int(f0, A + "expert_feed_forward_length");
    g.n_ff_sh = meta_int(f0, A + "expert_shared_feed_forward_length", g.n_ff_exp);
    const int n_shexp = meta_int(f0, A + "expert_shared_count", 1);
    g.n_dense = meta_int(f0, A + "leading_dense_block_count", 0);
    g.exp_scale = meta_f(f0, A + "expert_weights_scale", 1.0f);
    const int w_norm = meta_int(f0, A + "expert_weights_norm", 0), gating = meta_int(f0, A + "expert_gating_func", 2);
    g.conv_k = meta_int(f0, A + "ssm.conv_kernel");
    g.kda_lb = meta_f(f0, A + "kda.gate_lower_bound", -5.0f);
    const int safe_gate = meta_int(f0, A + "kda.safe_gate", 1);
    g.q_lora = meta_int(f0, A + "attention.q_lora_rank", 0);
    g.kv_lora = meta_int(f0, A + "attention.kv_lora_rank");
    g.rope_dim = meta_int(f0, A + "rope.dimension_count");
    g.qk_head = meta_int(f0, A + "attention.key_length_mla");
    g.v_head = meta_int(f0, A + "attention.value_length_mla");
    g.rope_base = meta_f(f0, A + "rope.freq_base", 10000.0f);
    g.vocab = meta_int(f0, A + "vocab_size");
    if (g.vocab <= 0)
        if (const strata::MetaValue* tv = f0.get("tokenizer.ggml.tokens")) g.vocab = (int) tv->count;
    g.ctx_train = meta_int(f0, A + "context_length");
    g.eos = meta_int(f0, "tokenizer.ggml.eos_token_id", -1);
    const std::vector<double> kvh = meta_arr(f0, A + "attention.head_count_kv", n_all, 0.0);
    const std::vector<double> ce = meta_arr(f0, A + "swiglu_clamp_exp", n_all, 0.0);
    const std::vector<double> cs = meta_arr(f0, A + "swiglu_clamp_shexp", n_all, 0.0);
    g.mla.assign((size_t) g.n_layer, 0);
    g.clamp_exp.assign((size_t) g.n_layer, 0.0f);
    g.clamp_sh.assign((size_t) g.n_layer, 0.0f);
    for (int il = 0; il < g.n_layer; ++il) {
        g.mla[(size_t) il] = kvh[(size_t) il] > 0.0 ? 1 : 0;
        g.clamp_exp[(size_t) il] = ce[(size_t) il] > 1e-6 ? (float) ce[(size_t) il] : 0.0f;
        g.clamp_sh[(size_t) il] = cs[(size_t) il] > 1e-6 ? (float) cs[(size_t) il] : 0.0f;
    }
    if (g.n_layer <= 0 || g.n_embd <= 0 || g.head_dim != 128 || g.n_head % 16 != 0 || g.n_used > 16 || g.n_used < 1 ||
        g.n_expert > 512 || g.n_expert % 32 != 0 || g.n_group > 16 || (g.n_expert / 32) % g.n_group != 0 ||
        g.n_group_used > g.n_group || g.conv_k < 2 || g.conv_k > 8 || g.kv_lora != 512 || g.rope_dim != 64 ||
        g.qk_head != 192 || g.v_head != 128 || g.q_lora <= 0 || n_shexp != 1 || w_norm != 1 || gating != 2 ||
        safe_gate == 0 || !k::norm_q8_ok(g.n_embd) || g.vocab <= 0) {
        err = "a Ling-3 shape this engine does not take (it is written for Ling-3.1-flash: 128-wide KDA heads, MLA "
              "with kv_lora 512 + 64 rope dims, q_lora, sigmoid group-limited routing of at most 512 experts, one "
              "shared expert)";
        return false;
    }
    C = g.n_head * g.head_dim;
    max_ctx = std::min(o.max_ctx, g.ctx_train > 0 ? g.ctx_train : o.max_ctx);
    theta_scale = std::pow(g.rope_base, -2.0f / (float) g.rope_dim);
    mla_scale = 1.0f / std::sqrt((float) g.qk_head);

    CK(cudaSetDevice(o.gpu));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int E = g.n_embd, H = g.n_head, K = g.conv_k;

    // the weights in VRAM
    if (const char* v = std::getenv("LING_EMBD_VRAM")) embd_host = std::atoi(v) == 0;
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
    } else {
        if (!mat("token_embd.weight", E, g.vocab, embd, err)) return false;
        if (!is_float(embd.type) && embd.type != 8) {
            err = "token_embd.weight: no row decoder for its type";
            return false;
        }
    }
    if (!mat(find("output.weight") ? "output.weight" : "token_embd.weight", E, g.vocab, output, err)) return false;
    if (!(out_norm = f32("output_norm.weight", (size_t) E, err))) return false;
    L.resize((size_t) g.n_layer);
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        l.mla = g.mla[(size_t) il] != 0;
        l.dense = il < g.n_dense;
        l.clamp_exp = g.clamp_exp[(size_t) il];
        l.clamp_sh = g.clamp_sh[(size_t) il];
        if (!(l.attn_norm = f32(b + "attn_norm.weight", (size_t) E, err))) return false;
        if (!(l.ffn_norm = f32(b + "ffn_norm.weight", (size_t) E, err))) return false;
        if (!l.mla) {
            if (!mat(b + "attn_q.weight", E, C, l.wq, err) || !mat(b + "attn_k.weight", E, C, l.wk, err) ||
                !mat(b + "attn_v.weight", E, C, l.wv, err) || !mat(b + "ssm_f_a.weight", E, C, l.wf, err) ||
                !mat(b + "ssm_g_a.weight", E, C, l.wg, err) || !mat(b + "ssm_beta.weight", E, H, l.wbeta, err) ||
                !mat(b + "attn_output.weight", C, E, l.wo, err))
                return false;
            if (!(l.conv_q = f32(b + "ssm_conv1d_q.weight", (size_t) K * C, err)) ||
                !(l.conv_k = f32(b + "ssm_conv1d_k.weight", (size_t) K * C, err)) ||
                !(l.conv_v = f32(b + "ssm_conv1d_v.weight", (size_t) K * C, err)) ||
                !(l.dt_bias = f32(b + "ssm_dt.bias", (size_t) C, err)) || !(l.a = f32(b + "ssm_a", (size_t) H, err)) ||
                !(l.o_norm = f32(b + "ssm_norm.weight", (size_t) g.head_dim, err)))
                return false;
            l.conv = dalloc<float>((size_t) 3 * C * (K - 1));
            l.S = dalloc<float>((size_t) H * 128 * 128);
        } else {
            const int kvd = g.kv_lora + g.rope_dim, nope = g.qk_head - g.rope_dim;
            if (!mat(b + "attn_q_a.weight", E, g.q_lora, l.q_a, err) ||
                !mat(b + "attn_q_b.weight", g.q_lora, H * g.qk_head, l.q_b, err) ||
                !mat(b + "attn_kv_a_mqa.weight", E, kvd, l.kv_a, err) ||
                !mat(b + "attn_k_b.weight", nope, g.kv_lora, l.k_b, err, H) ||
                !mat(b + "attn_v_b.weight", g.kv_lora, g.v_head, l.v_b, err, H) ||
                !mat(b + "attn_gate.weight", E, H, l.wgate, err) ||
                !mat(b + "attn_output.weight", H * g.v_head, E, l.wo, err))
                return false;
            if (is_float(l.k_b.type) || is_float(l.v_b.type) || is_float(l.q_b.type)) {
                err = b + "attn_k_b / attn_v_b / attn_q_b: a float type (the token path takes them as int8 GEMVs)";
                return false;
            }
            if (!(l.q_a_norm = f32(b + "attn_q_a_norm.weight", (size_t) g.q_lora, err)) ||
                !(l.kv_a_norm = f32(b + "attn_kv_a_norm.weight", (size_t) g.kv_lora, err)))
                return false;
            if (!k::norm_q8_ok(g.q_lora)) {
                err = "q_lora_rank " + std::to_string(g.q_lora) + ": the fused norm takes multiples of 256 up to 4096";
                return false;
            }
            l.kv_rows = max_ctx;
            l.kv = dalloc<__half>((size_t) l.kv_rows * 576);
        }
        if (l.dense) {
            if (!mat(b + "ffn_gate.weight", E, g.n_ff, l.gate, err) || !mat(b + "ffn_up.weight", E, g.n_ff, l.up, err) ||
                !mat(b + "ffn_down.weight", g.n_ff, E, l.down, err))
                return false;
        } else {
            const int ne = g.n_expert;
            if (!mat(b + "ffn_gate_inp.weight", E, ne, l.gate_inp, err)) return false;
            if (l.gate_inp.type != 0) {
                err = b + "ffn_gate_inp: not F32 (the prompt path's router multiplies F32 rows)";
                return false;
            }
            if (!(l.exp_bias_d = f32(b + "exp_probs_b.bias", (size_t) ne, err, &l.exp_bias))) return false;
            if (!mat(b + "ffn_gate_shexp.weight", E, g.n_ff_sh, l.sh_gate, err) ||
                !mat(b + "ffn_up_shexp.weight", E, g.n_ff_sh, l.sh_up, err) ||
                !mat(b + "ffn_down_shexp.weight", g.n_ff_sh, E, l.sh_down, err))
                return false;
            // the routed experts stay where they are, for the CPU lane
            const Tensor *tg = find(b + "ffn_gate_exps.weight"), *tu = find(b + "ffn_up_exps.weight"),
                         *td = find(b + "ffn_down_exps.weight");
            if (!tg || !tu || !td) {
                err = b + "ffn_*_exps: missing";
                return false;
            }
            if (!iq::supported((int) tg->info->type) || !iq::supported((int) tu->info->type) ||
                !iq::supported((int) td->info->type)) {
                err = b + "ffn_*_exps: a type the GPU's expert kernels do not take";
                return false;
            }
            if (!cpu_layer((int) tg->info->type, (int) tu->info->type, (int) td->info->type, E, g.n_ff_exp, l.cl, err)) {
                err = b + "ffn_*_exps: " + err;
                return false;
            }
            l.cl.gate = tg->data;
            l.cl.up = tu->data;
            l.cl.down = td->data;
            l.cl.limit = l.clamp_exp;
            // (the CPU rows' choice between ggml-cpu's dots and ds4's own, timed and checked now rather than in the
            // first token's lane)
            static std::vector<int> prepared;
            for (int ty : {l.cl.tg, l.cl.tu, l.cl.td})
                if (std::find(prepared.begin(), prepared.end(), ty) == prepared.end()) {
                    strata::ds4::gguf::iq_cpu_prepare_rows(ty);
                    prepared.push_back(ty);
                }
        }
        if (o.verbose && (il % 8 == 7 || il == g.n_layer - 1))
            std::fprintf(stderr, "maya-ling: layers 0..%d placed (%.1f GB of VRAM so far)\n", il, dev_bytes / 1e9);
    }

    // scratch
    x = dalloc<float>(E);
    a = dalloc<float>(E);
    f = dalloc<float>(E);
    ao = dalloc<float>(E);
    fo = dalloc<float>(E);
    for (float** p : {&qr, &kr, &vr, &fr, &gr, &qn, &kn, &vc, &ge, &ko}) *p = dalloc<float>((size_t) C);
    br = dalloc<float>(H);
    beta = dalloc<float>(H);
    qa = dalloc<float>((size_t) g.q_lora);
    kva = dalloc<float>(576);
    gh = dalloc<float>(H);
    qf = dalloc<float>((size_t) H * g.qk_head);
    qabs = dalloc<float>((size_t) H * g.kv_lora);
    mo = dalloc<float>((size_t) H * g.kv_lora);
    mob = dalloc<float>((size_t) H * g.v_head);
    fg = dalloc<float>((size_t) g.n_ff);
    fu = dalloc<float>((size_t) g.n_ff);
    fh = dalloc<float>((size_t) g.n_ff);
    sg = dalloc<float>((size_t) g.n_ff_sh);
    su = dalloc<float>((size_t) g.n_ff_sh);
    sh = dalloc<float>((size_t) g.n_ff_sh);
    so = dalloc<float>(E);
    d_tail = dalloc<float>((size_t) 4 + E);
    logits = dalloc<float>((size_t) g.vocab);
    rlog = dalloc<float>((size_t) g.n_expert);
    dpos = dalloc<int>(2);
    attn_nsp = k::mla_nsp(max_ctx);
    attn_scr = dalloc<float>(k::mla_scratch_floats(H, attn_nsp));
    const size_t ka = (size_t) std::max({E, C, H * g.v_head}), kf = (size_t) E,
                 kh = (size_t) std::max(g.n_ff, g.n_ff_sh), km = (size_t) std::max({g.q_lora, H * g.qk_head, H * g.kv_lora});
    xq_a = dalloc<int8_t>(ka);
    xd_a = dalloc<float>(ka / 32);
    xq_f = dalloc<int8_t>(kf);
    xd_f = dalloc<float>(kf / 32);
    xq_h = dalloc<int8_t>(kh);
    xd_h = dalloc<float>(kh / 32);
    xq_m = dalloc<int8_t>(km);
    xd_m = dalloc<float>(km / 32);
    h_f = halloc<float>(E);
    h_rlog = halloc<float>((size_t) g.n_expert);
    h_tail = halloc<float>((size_t) 4 + E);
    h_logits = halloc<float>((size_t) g.vocab);
    h_pos = halloc<int>(2);
    // the room for Engine::snapshot: every KDA layer's state and convolutions (~283 MB)
    if (!std::getenv("LING_SNAPSHOT") || std::atoi(std::getenv("LING_SNAPSHOT")) != 0) {
        for (const Layer& l : L)
            if (!l.mla) snap_bytes += (size_t) 3 * C * (K - 1) * sizeof(float) + (size_t) H * 128 * 128 * sizeof(float);
        snap = dalloc<uint8_t>(snap_bytes);
    }

    int threads = o.threads > 0 ? o.threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    pool.reset(new Pool(threads - 1));
    // Every routed expert's pages touched once, now: the first touch of a mapped page costs a fault (~1 us a 4 KB
    // page on Windows even with the file in RAM), which held maya-ink's prompt-path copies to 3.5 GB/s and the CPU
    // lane's first tokens back.  LING_PREFAULT=0: not done.
    const char* pv = std::getenv("LING_PREFAULT");
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
            std::fprintf(stderr, "maya-ling: the routed experts' %.1f GB touched in %.1f s (working set %.1f GB)\n", gb,
                         (now_ms() - tp) / 1000.0, working_set_gb());
    }
    if (const char* rl = std::getenv("LING_ROUTE_LOG")) route_log = std::fopen(rl, "ab");
    const size_t before_pf = dev_bytes;
    if (!pf_alloc(err)) return false;
    const size_t before_ec = dev_bytes;
    ec_alloc();
    if (!o.profile.empty() && load_profile(o.profile)) {
        const double tf = now_ms();
        ec_fill();
        CK(cudaStreamSynchronize(s));
        if (o.verbose)
            std::fprintf(stderr, "maya-ling: the expert profile %s: %ld experts into VRAM in %.1f s\n", o.profile.c_str(),
                         ec.copies, (now_ms() - tf) / 1000.0);
    }
    if (o.verbose) {
        std::string cls;
        for (size_t c = 0; c < ec.cls_bytes.size(); ++c) {
            int n_all_ = 0, n_l = 0;
            for (int sl = 0; sl < ec.slots; ++sl) n_all_ += ec.cls[(size_t) sl] == (int) c;
            for (int il = g.n_dense; il < g.n_layer; ++il) n_l += ec.layer_cls[(size_t) il] == (int) c;
            char b[128];
            std::snprintf(b, sizeof b, "%s%d of %.2f MB (%d layers)", c ? ", " : "", n_all_, ec.cls_bytes[c] / 1e6, n_l);
            cls += b;
        }
        int n_mla = 0;
        for (int il = 0; il < g.n_layer; ++il) n_mla += g.mla[(size_t) il];
        std::fprintf(stderr,
                     "maya-ling: Ling-3 %d layers (%d KDA, %d MLA), %d experts (top %d of the top %d of %d groups + 1 "
                     "shared), context %d; %.2f GB in VRAM (the prompt path's chunks of %d tokens %.2f of it, %d expert "
                     "slots %.2f, %d more in the prompt path's buffers between prompts and %d in the MLA caches' rows "
                     "past the position: %s), the routed experts in the GGUF's mapping; CPU lane %d threads; loaded in "
                     "%.1f s\n",
                     g.n_layer, g.n_layer - n_mla, n_mla, g.n_expert, g.n_used, g.n_group_used, g.n_group, max_ctx,
                     dev_bytes / 1e9, pf.C, (before_ec - before_pf) / 1e9, ec.own, (dev_bytes - before_ec) / 1e9,
                     ec.arena_end - ec.own, ec.slots - ec.arena_end, cls.c_str(), pool->threads(),
                     (now_ms() - t0) / 1000.0);
        std::string cl;
        for (int il = 0; il < g.n_layer; ++il)
            if (g.clamp_exp[(size_t) il] > 0.0f || g.clamp_sh[(size_t) il] > 0.0f) {
                char b[64];
                std::snprintf(b, sizeof b, " %d:%g/%g", il, g.clamp_exp[(size_t) il], g.clamp_sh[(size_t) il]);
                cl += b;
            }
        std::fprintf(stderr, "maya-ling: SwiGLU clamps (layer:routed/shared)%s; eos %d\n", cl.empty() ? " none" : cl.c_str(),
                     g.eos);
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
    if (const char* v = std::getenv("LING_CACHE_DECAY")) ec.decay = (float) std::atof(v);
    if (const char* v = std::getenv("LING_CACHE_EVERY")) ec.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("LING_CACHE_SWAPS")) ec.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("LING_CACHE_MARGIN")) ec.margin = (float) std::atof(v);
    if (const char* v = std::getenv("LING_CACHE_DEFER")) ec.defer = std::atoi(v) != 0;
    if (const char* v = std::getenv("LING_CACHE_STAGE_KB")) ec.piece = (size_t) std::max(64, std::atoi(v)) << 10;
    if (ec.defer) {
        for (int i = 0; i < 2 && ec.defer; ++i) {
            ec.defer = cudaHostAlloc((void**) &ec.apin[i], 4 * ec.piece, cudaHostAllocPortable) == cudaSuccess;
            if (ec.defer) CK(cudaEventCreateWithFlags(&ec.aev[i], cudaEventDisableTiming));
        }
        if (ec.defer) CK(cudaEventCreateWithFlags(&ec.arrived_ev, cudaEventDisableTiming));
        if (!ec.defer) cudaGetLastError();
    }
    const char* lv = std::getenv("LING_LEND");
    const bool lend = !lv || std::atoi(lv) != 0;
    // the classes: each MoE layer's expert bytes (gate | up | down as the GGUF holds them), a 256-byte multiple
    ec.layer_cls.assign((size_t) g.n_layer, -1);
    std::vector<size_t> eb((size_t) g.n_layer, 0);
    for (int il = g.n_dense; il < g.n_layer; ++il) eb[(size_t) il] = (expert_bytes(il) + 255) & ~(size_t) 255;
    for (int il = g.n_dense; il < g.n_layer; ++il) {
        const size_t b = eb[(size_t) il];
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
    size_t fr_ = 0, tot = 0;
    CK(cudaMemGetInfo(&fr_, &tot));
    size_t reserve = (size_t) 1536 << 20;
    if (const char* v = std::getenv("LING_VRAM_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
    size_t want = fr_ > reserve ? fr_ - reserve : 0;
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
        void* p0 = nullptr;
        if (cudaMalloc(&p0, bytes) == cudaSuccess) {
            base = (uint8_t*) p0;
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
    const char* kvv = std::getenv("LING_LEND_KV");
    if (base && (!kvv || std::atoi(kvv) != 0)) {
        const size_t b = ec.cls_bytes[(size_t) ec.lend_cls], row = (size_t) 576 * sizeof(__half);
        for (const Layer& l : L) {
            if (!l.mla) continue;
            const size_t bytes = (size_t) l.kv_rows * row;
            for (size_t end = bytes; end >= b && end - b >= (size_t) kKvSlack * row; end -= b) {
                ec.ptr.push_back((uint8_t*) l.kv + (end - b));
                ec.cls.push_back(ec.lend_cls);
                ec.kv_row.push_back((int) ((end - b) / row));
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
            CK(cudaMemcpyAsync(ec.ptr[(size_t) sl], pin + d.second, expert_bytes(il), cudaMemcpyHostToDevice, pf.cs));
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
        std::vector<int> best, worst;
        for (int il = g.n_dense; il < g.n_layer; ++il) {
            if (ec.layer_cls[(size_t) il] != c) continue;
            for (int e = 0; e < ec.ne; ++e) {
                const int key = il * ec.ne + e;
                if (ec.where[(size_t) key] != -1 || ec.score[(size_t) key] <= 0.0f) continue;
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
    for (const auto& kv : todo) {
        const int key = kv.first, sl = kv.second;
        if (ec.owner[(size_t) sl] >= 0) ec.where[(size_t) ec.owner[(size_t) sl]] = -1;
        ec.owner[(size_t) sl] = key;
        ec.where[(size_t) key] = -2;
        EC::Pending p0;
        p0.key = key;
        p0.slot = sl;
        p0.bytes = expert_bytes(key / ec.ne);
        ec.pend.push_back(p0);
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

// The cache rows below `rows` about to be written: the slots there emptied first (the GPU done with them, the copies
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
    EC::Pending& p0 = ec.pend[ec.pend_head];
    const size_t n = std::min(budget, p0.bytes - p0.done);
    CK(cudaEventSynchronize(ec.aev[ec.ak]));   // (that buffer's last DMA: long done)
    ec.stage_n = n;
    ec.stage_parts.clear();
    ec_src_parts(p0.key, p0.done, n, ec.stage_parts);
    const auto* parts = &ec.stage_parts;   // (by pointer: the workers run the job)
    const size_t jb = (size_t) 128 << 10;
    const int nj = (int) ((n + jb - 1) / jb);
    uint8_t* dst = ec.apin[ec.ak];
    ec.stage_fn = [parts, dst, jb, n](int j) {
        const size_t a0 = (size_t) j * jb, a1 = std::min(n, a0 + jb);
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
    EC::Pending& p0 = ec.pend[ec.pend_head];
    CK(cudaMemcpyAsync(ec.ptr[(size_t) p0.slot] + p0.done, ec.apin[ec.ak], ec.stage_n, cudaMemcpyHostToDevice, pf.cs));
    CK(cudaEventRecord(ec.aev[ec.ak], pf.cs));
    ec.ak ^= 1;
    p0.done += ec.stage_n;
    if (p0.done == p0.bytes) {
        ec.arrived.push_back({p0.key, p0.slot});
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

// A token's routed experts that are in VRAM: gate / up (f's int8 rows), the clamped SwiGLU, down times their weights,
// summed in order into ec.routed
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
            j.ncol = 1;
            j.xq = xq_f;
            j.xd = xd_f;
            j.y = ec.gu + (size_t) i * 2 * ff + (size_t) r * ff;
            j.ycs = (size_t) ff;
        }
    }
    run_jobs(jobs, 2 * n);
    k::swiglu_f32_rows(ec.gu, n, ff, l.clamp_exp, ec.h, s);
    k::q8(ec.h, n, ff, ec.hq, ec.hd, s);
    iq::MvJob dj[16];
    for (int i = 0; i < n; ++i) {
        const uint8_t* b = ec.ptr[(size_t) slot[i]];
        dj[i] = iq::MvJob{};
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
        dj[i].alpha = w[i];
    }
    run_jobs(dj, n);
    k::sum_rows(ec.y, n, E, ec.routed, s);
}

bool Engine::Impl::save_profile(const std::string& path) const {
    FILE* fp = std::fopen((path + ".tmp").c_str(), "wb");
    if (!fp) return false;
    const int hdr[4] = {0x50474e4c, 1, g.n_layer, g.n_expert};   // "LNGP", version 1
    bool ok = std::fwrite(hdr, sizeof hdr, 1, fp) == 1 &&
              std::fwrite(ec.score.data(), sizeof(float), ec.score.size(), fp) == ec.score.size();
    ok = std::fclose(fp) == 0 && ok;
    if (!ok) return false;
    std::remove(path.c_str());
    return std::rename((path + ".tmp").c_str(), path.c_str()) == 0;
}

bool Engine::Impl::load_profile(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    int hdr[4] = {0, 0, 0, 0};
    std::vector<float> v(ec.score.size());
    const bool ok = std::fread(hdr, sizeof hdr, 1, fp) == 1 && hdr[0] == 0x50474e4c && hdr[1] == 1 &&
                    hdr[2] == g.n_layer && hdr[3] == g.n_expert &&
                    std::fread(v.data(), sizeof(float), v.size(), fp) == v.size();
    std::fclose(fp);
    if (ok) ec.score = std::move(v);
    return ok;
}

// ------------------------------------------------------------------------------------------------ the prompt path
size_t Engine::Impl::u_kda_bytes() const {
    const size_t Ca = (size_t) pf.Ca, H = (size_t) g.n_head;
    // QR KR VR FR GR QC KC VC (F32, C a row), BR BETA (H), OH (F16, C)
    return 8 * ((Ca * C * 4 + 255) & ~(size_t) 255) + 2 * ((Ca * H * 4 + 255) & ~(size_t) 255) +
           ((Ca * C * 2 + 255) & ~(size_t) 255);
}
size_t Engine::Impl::u_mla_bytes() const {
    const size_t Ca = (size_t) pf.Ca, H = (size_t) g.n_head;
    auto r = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    // QA (F32) QAh (F16) [q_lora]; KVA [576]; GH [H]; Q (F32) QH (F16) [H 192]; Qcat (F16) [H 576]; O16 (F16) [H 512];
    // OB (F32) OBh (F16) [H 128]
    return r(Ca * g.q_lora * 4) + r(Ca * g.q_lora * 2) + r(Ca * 576 * 4) + r(Ca * H * 4) + r(Ca * H * 192 * 4) +
           r(Ca * H * 192 * 2) + r(Ca * H * 576 * 2) + r(Ca * H * 512 * 2) + r(Ca * H * 128 * 4) + r(Ca * H * 128 * 2);
}

bool Engine::Impl::pf_alloc(std::string& err) {
    const int Cn = pf.C = std::max(64, o.chunk);
    pf.Ca = std::max(16, std::min(o.sub, Cn));
    const int E = g.n_embd, ne = g.n_expert, nu = g.n_used;
    const int ffe = g.n_ff_exp, ffd = g.n_ff, ffs = g.n_ff_sh;
    if (cublasCreate(&pf.cb) != CUBLAS_STATUS_SUCCESS) {
        err = "cuBLAS did not start";
        return false;
    }
    cublasSetStream(pf.cb, s);
    CK(cudaStreamCreateWithFlags(&pf.cs, cudaStreamNonBlocking));
    const size_t CE = (size_t) Cn * E;
    size_t mib = 768;
    if (const char* v = std::getenv("LING_SCORES_MIB")) mib = (size_t) std::max(64, std::atoi(v));
    pf.s_elems = (mib << 20) / 6;   // S F32 and P F16
    pf.w1_elems = std::max({(size_t) 2 * ffd * E, (size_t) 2 * ffe * E, (size_t) 2 * ffs * E, (size_t) C * E,
                            (size_t) g.n_head * g.qk_head * g.q_lora, (size_t) g.n_head * 512 * 128});
    pf.w2_elems = std::max({(size_t) ffd * E, (size_t) ffe * E, (size_t) ffs * E});
    for (const Layer& l : L)
        if (!l.dense) pf.slot_bytes = std::max(pf.slot_bytes, expert_bytes((int) (&l - L.data())));
    pf.GE = std::max(1, std::min(ne, o.stage));
    pf.u_bytes = std::max(u_kda_bytes(), u_mla_bytes());
    if (const char* v = std::getenv("LING_PF_PREFETCH")) pre.on = std::atoi(v) != 0;
    if (const char* v = std::getenv("LING_PF_PREFETCH_MIN")) pre.min_T = std::max(1, std::atoi(v));
    if (pre.on) {
        for (int il = g.n_dense; il < g.n_layer; ++il)
            pre.bytes = std::max(pre.bytes, ((size_t) ne * expert_bytes(il) + 255) & ~(size_t) 255);
        pre.pcap = (size_t) 64 << 20;
        bool ok = true;
        for (int i = 0; i < 2 && ok; ++i) {
            ok = cudaHostAlloc((void**) &pre.pin[i], pre.pcap, cudaHostAllocPortable) == cudaSuccess;
            if (ok) {
                CK(cudaEventCreateWithFlags(&pre.pev[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&pre.ready[i], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&pre.freed[i], cudaEventDisableTiming));
            }
        }
        if (!ok) {
            cudaGetLastError();
            pre.on = false;
            std::fprintf(stderr, "maya-ling: no pinned staging for the prompt path's prefetch (off)\n");
        } else {
            CK(cudaStreamCreateWithFlags(&pre.cs, cudaStreamNonBlocking));
            pre.pool.reset(new Pool(7));
        }
    }
    const size_t gu_cols = (size_t) std::max(ffe, ffs) * 2;
    // Every buffer in one arena (none holds anything from one prompt to the next, so the expert cache can borrow it,
    // and each is written before it is read): the same list twice, first its size, then its pointers, 256-byte aligned.
    for (int pass = 0; pass < 2; ++pass) {
        size_t at = 0;
        auto carve = [&](auto*& ptr, size_t n) {
            using T = std::remove_reference_t<decltype(*ptr)>;
            if (pass == 1) ptr = (T*) (pf.arena + at);
            at += (n * sizeof(T) + 255) & ~(size_t) 255;
        };
        carve(pf.ids, (size_t) Cn);
        carve(pf.X, CE);
        carve(pf.A, CE);
        carve(pf.Y, CE);
        carve(pf.Ye, CE);
        carve(pf.Ah, CE);
        carve(pf.Xe, CE);
        carve(pf.GU, (size_t) Cn * gu_cols);
        carve(pf.He, (size_t) Cn * std::max(ffe, ffs));
        carve(pf.RL, (size_t) Cn * ne);
        carve(pf.sel, (size_t) Cn * nu);
        carve(pf.wts, (size_t) Cn * nu);
        carve(pf.cnt, (size_t) ne);
        carve(pf.off, (size_t) ne);
        carve(pf.list, (size_t) Cn * nu);
        carve(pf.wl, (size_t) Cn * nu);
        carve(pf.U, pf.u_bytes);
        carve(pf.S, pf.s_elems);
        carve(pf.P, pf.s_elems);
        carve(pf.W1, pf.w1_elems);
        carve(pf.W2, pf.w2_elems);
        // two group buffers in turn, `stage` experts each (the copy of one group while the other's are multiplied)
        carve(pf.gbuf[0], (size_t) pf.GE * pf.slot_bytes);
        carve(pf.gbuf[1], (size_t) pf.GE * pf.slot_bytes);
        if (pre.on) {   // (the prefetch's two layer buffers)
            carve(pre.buf[0], pre.bytes);
            carve(pre.buf[1], pre.bytes);
        }
        if (embd_host) carve(pf.Erows, (size_t) Cn * embd.rb);   // (a chunk's embedding rows)
        if (pass == 0) {
            pf.arena_bytes = at;
            pf.arena = dalloc<uint8_t>(at);
        }
    }
    for (int i = 0; i < 2; ++i) {
        CK(cudaEventCreateWithFlags(&pf.gready[i], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&pf.gfreed[i], cudaEventDisableTiming));
    }
    pf.h_ids = halloc<int>(Cn);
    pf.h_cnt = halloc<int>(ne);
    pf.h_off = halloc<int>(ne + 1);
    const char* pg = std::getenv("LING_PF_PAGEABLE");
    if (!pg || std::atoi(pg) == 0) {
        stg.cap = (size_t) 64 << 20;
        stg.on = cudaHostAlloc((void**) &stg.pin[0], stg.cap, cudaHostAllocPortable) == cudaSuccess &&
                 cudaHostAlloc((void**) &stg.pin[1], stg.cap, cudaHostAllocPortable) == cudaSuccess;
        if (stg.on) {
            CK(cudaEventCreateWithFlags(&stg.ev[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stg.ev[1], cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "maya-ling: no pinned staging for the prompt path's copies (the driver's is used)\n");
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

// The copier's staging: stage_copy's, with its own pinned buffers, pool and copy stream
void Engine::Impl::pre_stage(uint8_t* dst, const uint8_t* src, size_t len) {
    const int nt = pre.pool->threads();
    while (len) {
        const size_t n = std::min(len, pre.pcap);
        CK(cudaEventSynchronize(pre.pev[pre.pk]));   // (that buffer's last DMA is done)
        uint8_t* pin = pre.pin[pre.pk];
        const size_t per = ((n + (size_t) nt - 1) / (size_t) nt + 4095) & ~(size_t) 4095;
        pre.pool->run(nt, [&](int j) {
            const size_t a0 = (size_t) j * per;
            if (a0 < n) std::memcpy(pin + a0, src + a0, std::min(per, n - a0));
        });
        CK(cudaMemcpyAsync(dst, pin, n, cudaMemcpyHostToDevice, pre.cs));
        CK(cudaEventRecord(pre.pev[pre.pk], pre.cs));
        pre.pk ^= 1;
        dst += n;
        src += n;
        len -= n;
    }
}

// The copier's loop over a chunk's MoE layers: layer j's experts not in the VRAM cache into buffer j & 1, once the GPU
// is through with the layer two before (its `freed` event, recorded after that layer's GEMMs); then the layer's `ready`
void Engine::Impl::pre_run() {
    CK(cudaSetDevice(o.gpu));
    const int ne = g.n_expert, E = g.n_embd, ff = g.n_ff_exp;
    const double t0 = now_ms();
    for (int il = g.n_dense, j = 0; il < g.n_layer; ++il, ++j) {
        const int b = j & 1;
        if (j >= 2) {
            {
                std::unique_lock<std::mutex> lk(pre.mu);
                pre.cv.wait(lk, [&] { return pre.freed_upto >= il - 2; });
            }
            CK(cudaEventSynchronize(pre.freed[b]));
        }
        const CpuLayer& c = L[(size_t) il].cl;
        const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
        uint8_t* B = pre.buf[b];
        for (int e = 0; e < ne;) {
            if (ec_blob(il, e)) {
                ++e;
                continue;
            }
            int r = e;
            while (r < ne && !ec_blob(il, r)) ++r;
            const size_t n = (size_t) (r - e);
            pre_stage(B + (size_t) e * gb, c.gate + (size_t) e * gb, n * gb);
            pre_stage(B + (size_t) ne * gb + (size_t) e * ub, c.up + (size_t) e * ub, n * ub);
            pre_stage(B + (size_t) ne * (gb + ub) + (size_t) e * db, c.down + (size_t) e * db, n * db);
            pre.bytes_copied += (double) (n * (gb + ub + db));
            e = r;
        }
        CK(cudaEventRecord(pre.ready[b], pre.cs));
        {
            std::lock_guard<std::mutex> lk(pre.mu);
            pre.staged = il;
        }
        pre.cv.notify_all();
    }
    pre.ms += now_ms() - t0;
}

void Engine::Impl::pre_start() {
    pre.staged = pre.freed_upto = -1;
    pre.bytes_copied = 0;
    pre.active = true;
    pre.th = std::thread([this] { pre_run(); });
}

void Engine::Impl::pre_join() {
    if (!pre.active) return;
    pre.th.join();
    pre.active = false;
    pf.copy_bytes += pre.bytes_copied;
}

void Engine::Impl::f16w(const DMat& m, __half* dst) {
    if (!iq::f16_rows(m.type, m.w, m.k, m.n, (uint16_t*) dst, m.k, s)) {
        std::fprintf(stderr, "maya-ling: no F16 rows for type %d\n", m.type);
        std::abort();
    }
}

void Engine::Impl::gemm(const __half* X, int ldx, const __half* W, int ldw, float* Y, int ldy, int T, int n, int k,
                        float alpha, float beta_) {
    // Y [T][n] (row-major, ldy) = alpha X [T][k] . W [n][k]^T + beta Y: column-major Y^T = W . X^T
    const cublasStatus_t r = cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k, &alpha, W, CUDA_R_16F, ldw, X,
                                          CUDA_R_16F, ldx, &beta_, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ling: cuBLAS GEMM %dx%dx%d failed (%d)\n", T, n, k, (int) r);
        std::abort();
    }
}

// y [T][n] = x W^T: a float matrix in F32 (x the F32 rows: cublasSgemm, as llama.cpp's F32 weights), else W decoded to
// F16 and x's F16 rows
void Engine::Impl::lin(const DMat& m, const __half* xh, const float* xf, int T, float* y) {
    if (m.type == 0) {
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, m.n, T, m.k, &one, (const float*) m.w, m.k, xf, m.k, &zero, y,
                        m.n) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "maya-ling: cuBLAS F32 GEMM failed\n");
            std::abort();
        }
        return;
    }
    if (is_float(m.type)) {
        std::fprintf(stderr, "maya-ling: a F16 / BF16 matrix in the prompt path (not taken)\n");
        std::abort();
    }
    f16w(m, pf.W1);
    gemm(xh, m.k, pf.W1, m.k, y, m.n, T, m.n, m.k);
}

void Engine::Impl::pf_kda(Layer& l, int T) {
    const int H = g.n_head, K = g.conv_k;
    const size_t Ca = (size_t) pf.Ca;
    uint8_t* at = pf.U;
    auto take = [&](size_t bytes) {
        uint8_t* p0 = at;
        at += (bytes + 255) & ~(size_t) 255;
        return p0;
    };
    float *QR = (float*) take(Ca * C * 4), *KR = (float*) take(Ca * C * 4), *VR = (float*) take(Ca * C * 4),
          *FR = (float*) take(Ca * C * 4), *GR = (float*) take(Ca * C * 4), *QC = (float*) take(Ca * C * 4),
          *KC = (float*) take(Ca * C * 4), *VC = (float*) take(Ca * C * 4), *BR = (float*) take(Ca * H * 4),
          *BT = (float*) take(Ca * H * 4);
    __half* OH = (__half*) take(Ca * C * 2);
    lin(l.wq, pf.Ah, pf.A, T, QR);
    lin(l.wk, pf.Ah, pf.A, T, KR);
    lin(l.wv, pf.Ah, pf.A, T, VR);
    lin(l.wf, pf.Ah, pf.A, T, FR);
    lin(l.wg, pf.Ah, pf.A, T, GR);
    lin(l.wbeta, pf.Ah, pf.A, T, BR);
    k::Kda kw;
    kw.wq = l.conv_q;
    kw.wk = l.conv_k;
    kw.wv = l.conv_v;
    kw.dt_bias = l.dt_bias;
    kw.a = l.a;
    kw.H = H;
    kw.hd = g.head_dim;
    kw.K = K;
    kw.lb = g.kda_lb;
    kw.eps = g.eps;
    float *st_q = l.conv, *st_k = l.conv + (size_t) C * (K - 1), *st_v = l.conv + (size_t) 2 * C * (K - 1);
    k::kda_conv(kw, QR, KR, VR, st_q, st_k, st_v, QC, KC, VC, T, s);
    k::kda_conv_state(kw, QR, KR, VR, st_q, st_k, st_v, T, s);
    k::kda_norm_gate(kw, QC, KC, FR, BR, BT, T, s);
    k::kda_scan(QC, KC, VC, FR, BT, l.S, QR, T, H, s);   // (the output into QR: free by now)
    k::kda_out(QR, GR, l.o_norm, g.eps, T, H, nullptr, OH, nullptr, nullptr, s);
    f16w(l.wo, pf.W1);
    gemm(OH, C, pf.W1, C, pf.Y, g.n_embd, T, g.n_embd, C);
}

void Engine::Impl::pf_mla(Layer& l, int T, int pos0) {
    const int H = g.n_head, E = g.n_embd, ql = g.q_lora;
    const size_t Ca = (size_t) pf.Ca;
    uint8_t* at = pf.U;
    auto take = [&](size_t bytes) {
        uint8_t* p0 = at;
        at += (bytes + 255) & ~(size_t) 255;
        return p0;
    };
    float* QA = (float*) take(Ca * ql * 4);
    __half* QAh = (__half*) take(Ca * ql * 2);
    float* KVA = (float*) take(Ca * 576 * 4);
    float* GH = (float*) take(Ca * H * 4);
    float* Q = (float*) take(Ca * H * 192 * 4);
    __half* QH = (__half*) take(Ca * H * 192 * 2);
    __half* Qcat = (__half*) take(Ca * H * 576 * 2);
    __half* O16 = (__half*) take(Ca * H * 512 * 2);
    float* OB = (float*) take(Ca * H * 128 * 4);
    __half* OBh = (__half*) take(Ca * H * 128 * 2);
    lin(l.q_a, pf.Ah, pf.A, T, QA);
    lin(l.kv_a, pf.Ah, pf.A, T, KVA);
    lin(l.wgate, pf.Ah, pf.A, T, GH);
    k::rmsnorm(QA, l.q_a_norm, QA, T, ql, g.eps, s);
    k::to_f16(QA, QAh, (size_t) T * ql, s);
    lin(l.q_b, QAh, nullptr, T, Q);
    k::to_f16(Q, QH, (size_t) T * H * 192, s);
    // q's nope part through k_b a head: Qcat[t][h][0:512] (F16), its rope part after it
    const float one = 1.0f, zero = 0.0f;
    f16w(l.k_b, pf.W1);   // [H][512][128]
    cublasStatus_t r = cublasGemmStridedBatchedEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, 512, T, 128, &one, pf.W1, CUDA_R_16F,
                                                  128, (long long) 512 * 128, QH, CUDA_R_16F, H * 192, 192, &zero, Qcat,
                                                  CUDA_R_16F, H * 576, 576, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ling: cuBLAS k_b batch failed (%d)\n", (int) r);
        std::abort();
    }
    k::mla_qpe_f16(Q, Qcat, T, H, pos0, theta_scale, s);
    k::mla_kv_store(KVA, l.kv_a_norm, g.eps, l.kv, T, pos0, nullptr, theta_scale, s);
    // the attention: tiles of query tokens, each [tq x H rows] x [the keys 0 .. its last position]
    auto r8 = [](int v) { return (v + 7) & ~7; };
    int ta = 0;
    while (ta < T) {
        int tq = T - ta;
        while (tq > 1 && (size_t) H * tq * (size_t) r8(pos0 + ta + tq) > pf.s_elems) tq = (tq + 1) / 2;
        const int tb = ta + tq, nk = pos0 + tb, ld = r8(nk), M = H * tq;
        r = cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, M, 576, &mla_scale, l.kv, CUDA_R_16F, 576,
                         Qcat + (size_t) ta * H * 576, CUDA_R_16F, 576, &zero, pf.S, CUDA_R_32F, ld, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "maya-ling: cuBLAS scores %dx%d failed (%d)\n", M, nk, (int) r);
            std::abort();
        }
        k::mla_softmax(pf.S, ld, pf.P, ld, M, nk, H, ta, pos0, s);
        r = cublasGemmEx(pf.cb, CUBLAS_OP_N, CUBLAS_OP_N, 512, M, nk, &one, l.kv, CUDA_R_16F, 576, pf.P, CUDA_R_16F, ld,
                         &zero, O16 + (size_t) ta * H * 512, CUDA_R_16F, 512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "maya-ling: cuBLAS attention output %dx%d failed (%d)\n", M, nk, (int) r);
            std::abort();
        }
        ta = tb;
    }
    // each head's 512 through v_b (128), times its gate, then W_o
    f16w(l.v_b, pf.W1);   // [H][128][512]
    r = cublasGemmStridedBatchedEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, 128, T, 512, &one, pf.W1, CUDA_R_16F, 512,
                                   (long long) 128 * 512, O16, CUDA_R_16F, H * 512, 512, &zero, OB, CUDA_R_32F, H * 128,
                                   128, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ling: cuBLAS v_b batch failed (%d)\n", (int) r);
        std::abort();
    }
    k::head_gate_rows(OB, GH, T, H, 128, OBh, s);
    f16w(l.wo, pf.W1);
    gemm(OBh, H * 128, pf.W1, H * 128, pf.Y, E, T, E, H * 128);
}

void Engine::Impl::pf_dense(const Layer& l, int T) {
    const int E = g.n_embd, ff = g.n_ff;
    f16w(l.gate, pf.W1);
    f16w(l.up, pf.W1 + (size_t) ff * E);
    f16w(l.down, pf.W2);
    const int TR = std::max(1, (int) ((size_t) pf.C * 2 * std::max(g.n_ff_exp, g.n_ff_sh) / (2 * (size_t) ff)));
    for (int r0 = 0; r0 < T; r0 += TR) {
        const int n = std::min(TR, T - r0);
        gemm(pf.Ah + (size_t) r0 * E, E, pf.W1, E, pf.GU, 2 * ff, n, 2 * ff, E);
        k::swiglu_rows(pf.GU, n, ff, l.clamp_sh, pf.He, s);
        gemm(pf.He, ff, pf.W2, ff, pf.Y + (size_t) r0 * E, E, n, E, ff);
    }
}

void Engine::Impl::pf_moe(const Layer& l, int T) {
    const int E = g.n_embd, ne = g.n_expert, nu = g.n_used, ff = g.n_ff_exp, ffs = g.n_ff_sh;
    const float one = 1.0f, zero = 0.0f;
    // the router in F32 (llama.cpp's F32 weights), the routing on the GPU
    if (cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, ne, T, E, &one, (const float*) l.gate_inp.w, E, pf.A, E, &zero,
                    pf.RL, ne) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-ling: cuBLAS router failed\n");
        std::abort();
    }
    k::route(pf.RL, ne, l.exp_bias_d, T, ne, g.n_group, g.n_group_used, nu, g.exp_scale, pf.sel, pf.wts, s);
    CK(cudaMemsetAsync(pf.cnt, 0, (size_t) ne * sizeof(int), s));
    k::expert_count(pf.sel, T * nu, pf.cnt, s);
    CK(cudaMemcpyAsync(pf.h_cnt, pf.cnt, (size_t) ne * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    pf.h_off[0] = 0;
    for (int e = 0; e < ne; ++e) pf.h_off[e + 1] = pf.h_off[e] + pf.h_cnt[e];
    CK(cudaMemcpyAsync(pf.off, pf.h_off, (size_t) ne * sizeof(int), cudaMemcpyHostToDevice, s));
    k::expert_lists(pf.sel, pf.wts, T, nu, ne, pf.off, pf.list, pf.wl, s);
    CK(cudaMemsetAsync(pf.Y, 0, (size_t) T * E * sizeof(float), s));
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) E * c.rd;
    const int il = (int) (&l - L.data()), GE = pf.GE;
    static const bool no_copy = std::getenv("LING_PF_NOCOPY") != nullptr;
    for (int e = 0; e < ne; ++e)
        if (pf.h_cnt[e]) ec.score[(size_t) il * ne + e] += (float) pf.h_cnt[e];
    if (route_log) {
        std::vector<uint16_t> c16((size_t) ne);
        for (int e = 0; e < ne; ++e) c16[(size_t) e] = (uint16_t) std::min(pf.h_cnt[e], 65535);
        const uint8_t hdr[2] = {'P', (uint8_t) il};
        std::fwrite(hdr, 1, 2, route_log);
        std::fwrite(c16.data(), 2, c16.size(), route_log);
    }
    // The prefetch's way (a copier running for this chunk): every expert not in VRAM already on its way into the
    // layer's buffer; once it is there, each used expert's GEMMs as below, in id order
    if (pre.active) {
        const int b = (il - g.n_dense) & 1;
        {
            std::unique_lock<std::mutex> lk(pre.mu);
            pre.cv.wait(lk, [&] { return pre.staged >= il; });
        }
        CK(cudaStreamWaitEvent(s, pre.ready[b], 0));
        const uint8_t* B = pre.buf[b];
        for (int e = 0; e < ne; ++e) {
            const int rows = pf.h_cnt[e], o0 = pf.h_off[e];
            if (rows == 0) continue;
            const uint8_t* blob = ec_blob(il, e);
            const uint8_t* wg = blob ? blob : B + (size_t) e * gb;
            const uint8_t* wu = blob ? blob + gb : B + (size_t) ne * gb + (size_t) e * ub;
            const uint8_t* wd = blob ? blob + gb + ub : B + (size_t) ne * (gb + ub) + (size_t) e * db;
            if (!iq::f16_rows(c.tg, wg, E, ff, (uint16_t*) pf.W1, E, s) ||
                !iq::f16_rows(c.tu, wu, E, ff, (uint16_t*) (pf.W1 + (size_t) ff * E), E, s) ||
                !iq::f16_rows(c.td, wd, ff, E, (uint16_t*) pf.W2, ff, s)) {
                std::fprintf(stderr, "maya-ling: no F16 rows for the experts' types %d / %d / %d\n", c.tg, c.tu, c.td);
                std::abort();
            }
            iq::gather_f16((const uint16_t*) pf.Ah, E, pf.list + o0, rows, E, (uint16_t*) pf.Xe, s);
            gemm(pf.Xe, E, pf.W1, E, pf.GU, 2 * ff, rows, 2 * ff, E);
            k::swiglu_rows(pf.GU, rows, ff, l.clamp_exp, pf.He, s);
            gemm(pf.He, ff, pf.W2, ff, pf.Ye, E, rows, E, ff);
            k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, E, pf.Y, s);
        }
        CK(cudaEventRecord(pre.freed[b], s));
        {
            std::lock_guard<std::mutex> lk(pre.mu);
            pre.freed_upto = il;
        }
        pre.cv.notify_all();
    }
    // The experts in groups of GE: a group's used experts that are not in VRAM copied into one of two group buffers
    // (each run of consecutive ones in one copy a tensor), laid out as [the gates | the ups | the downs] of the group.
    for (int e0 = 0; e0 < ne && !pre.active; e0 += GE) {
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
            if (rows == 0) continue;
            const uint8_t* blob = ec_blob(il, e);
            const size_t at = (size_t) (e - e0);
            const uint8_t* wg = blob ? blob : B + at * gb;
            const uint8_t* wu = blob ? blob + gb : B + GE * gb + at * ub;
            const uint8_t* wd = blob ? blob + gb + ub : B + GE * (gb + ub) + at * db;
            if (!iq::f16_rows(c.tg, wg, E, ff, (uint16_t*) pf.W1, E, s) ||
                !iq::f16_rows(c.tu, wu, E, ff, (uint16_t*) (pf.W1 + (size_t) ff * E), E, s) ||
                !iq::f16_rows(c.td, wd, ff, E, (uint16_t*) pf.W2, ff, s)) {
                std::fprintf(stderr, "maya-ling: no F16 rows for the experts' types %d / %d / %d\n", c.tg, c.tu, c.td);
                std::abort();
            }
            iq::gather_f16((const uint16_t*) pf.Ah, E, pf.list + o0, rows, E, (uint16_t*) pf.Xe, s);
            gemm(pf.Xe, E, pf.W1, E, pf.GU, 2 * ff, rows, 2 * ff, E);
            k::swiglu_rows(pf.GU, rows, ff, l.clamp_exp, pf.He, s);
            gemm(pf.He, ff, pf.W2, ff, pf.Ye, E, rows, E, ff);
            k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, E, pf.Y, s);
        }
        if (bi >= 0) CK(cudaEventRecord(pf.gfreed[bi], s));
    }
    // the shared expert, every token
    f16w(l.sh_gate, pf.W1);
    f16w(l.sh_up, pf.W1 + (size_t) ffs * E);
    f16w(l.sh_down, pf.W2);
    gemm(pf.Ah, E, pf.W1, E, pf.GU, 2 * ffs, T, 2 * ffs, E);
    k::swiglu_rows(pf.GU, T, ffs, l.clamp_sh, pf.He, s);
    gemm(pf.He, ffs, pf.W2, ffs, pf.Ye, E, T, E, ffs);
    k::axpy(pf.Y, pf.Ye, nullptr, (size_t) T * E, s);
}

void Engine::Impl::chunk(const int* ids, int T, bool want_logits) {
    const int E = g.n_embd;
    const int pos0 = pos;
    ec_kv_reserve(pos0 + T);
    const uint8_t* ew = embd.w;
    if (embd_host) {
        // the chunk's rows gathered from the mapping into the pinned staging (or one by one without it), then the
        // rows read in order
        const size_t rb = embd.rb;
        if (stg.on && stg.cap >= rb) {
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
        std::fprintf(stderr, "maya-ling: no embedding rows for type %d\n", embd.type);
        std::abort();
    }
    static const bool times = std::getenv("LING_PF_TIMES") != nullptr;
    if (times) CK(cudaStreamSynchronize(s));
    if (pre.on && T >= pre.min_T) pre_start();   // (the copier: every MoE layer's experts, while the GPU computes)
    double tl0 = now_ms();
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        // the attention, `sub` tokens at a time
        for (int s0 = 0; s0 < T; s0 += pf.Ca) {
            const int Ts = std::min(pf.Ca, T - s0);
            float* Xs = pf.X + (size_t) s0 * E;
            k::rmsnorm(Xs, l.attn_norm, pf.A, Ts, E, g.eps, s);
            k::to_f16(pf.A, pf.Ah, (size_t) Ts * E, s);
            if (l.mla) pf_mla(l, Ts, pos0 + s0);
            else pf_kda(l, Ts);
            k::axpy(Xs, pf.Y, nullptr, (size_t) Ts * E, s);
        }
        dump("attn", il, pf.X + (size_t) (T - 1) * E, pos0 + T - 1);
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
        k::axpy(pf.X, pf.Y, nullptr, (size_t) T * E, s);
        dump("ffn ", il, pf.X + (size_t) (T - 1) * E, pos0 + T - 1);
        if (times) {
            CK(cudaStreamSynchronize(s));
            const double t1 = now_ms();
            pf.moe_ms += t1 - tl0;
            tl0 = t1;
        }
    }
    pre_join();
    if (times)
        std::fprintf(stderr,
                     "maya-ling: prompt path so far: attention %.0f ms, FFN %.0f ms (blocked in copies %.0f ms; the "
                     "prefetch's copier %.0f ms); %llu page faults, working set %.1f GB\n",
                     pf.attn_ms, pf.moe_ms, pf.copy_ms, pre.ms, page_faults(), working_set_gb());
    pos = pos0 + T;
    if (want_logits) {
        k::norm_q8(pf.X + (size_t) (T - 1) * E, out_norm, g.eps, a, xq_a, xd_a, E, s);
        mv(output, a, xq_a, xd_a, logits);
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
    // a short prompt (a turn's few new tokens): the token path's experts on the CPU beat copying the ones it chooses -
    // a chunk of a few hundred tokens touches nearly every one of a layer's 512 experts (LING_MIN_CHUNK)
    int min_chunk = 256;
    if (const char* v = std::getenv("LING_MIN_CHUNK")) min_chunk = std::max(1, std::atoi(v));
    if (n < min_chunk) {
        for (int i = 0; i < n; ++i) lg = step(ids[i]);
        if (progress) progress(ctx, n);
        return lg;
    }
    // the lent slots back to their buffers until the next token is written (a long prompt comes in pieces)
    ec_reclaim();
    pf_open = true;
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
// A KDA layer's attention for the token: x += it
void Engine::Impl::tok_kda(Layer& l) {
    const int E = g.n_embd, H = g.n_head, K = g.conv_k;
    k::norm_q8(x, l.attn_norm, g.eps, a, xq_a, xd_a, E, s);
    {
        const DMat* ms[6] = {&l.wq, &l.wk, &l.wv, &l.wf, &l.wg, &l.wbeta};
        float* ys[6] = {qr, kr, vr, fr, gr, br};
        mv_many(ms, ys, 6, a, xq_a, xd_a);
    }
    k::Kda kw;
    kw.wq = l.conv_q;
    kw.wk = l.conv_k;
    kw.wv = l.conv_v;
    kw.dt_bias = l.dt_bias;
    kw.a = l.a;
    kw.H = H;
    kw.hd = g.head_dim;
    kw.K = K;
    kw.lb = g.kda_lb;
    kw.eps = g.eps;
    float *st_q = l.conv, *st_k = l.conv + (size_t) C * (K - 1), *st_v = l.conv + (size_t) 2 * C * (K - 1);
    k::kda_prep1(kw, qr, kr, vr, fr, br, st_q, st_k, st_v, qn, kn, vc, ge, beta, s);
    k::kda_scan(qn, kn, vc, ge, beta, l.S, ko, 1, H, s);
    k::kda_out(ko, gr, l.o_norm, g.eps, 1, H, nullptr, nullptr, xq_a, xd_a, s);
    mv(l.wo, nullptr, xq_a, xd_a, ao);
    k::add(x, ao, E, s);
}

// A gated MLA layer's attention for the token at *dpos: x += it
void Engine::Impl::tok_mla(Layer& l) {
    const int E = g.n_embd, H = g.n_head;
    k::norm_q8(x, l.attn_norm, g.eps, a, xq_a, xd_a, E, s);
    {
        const DMat* ms[3] = {&l.q_a, &l.kv_a, &l.wgate};
        float* ys[3] = {qa, kva, gh};
        mv_many(ms, ys, 3, a, xq_a, xd_a);
    }
    k::norm_q8(qa, l.q_a_norm, g.eps, nullptr, xq_m, xd_m, g.q_lora, s);
    mv(l.q_b, nullptr, xq_m, xd_m, qf);
    k::q8(qf, 1, H * g.qk_head, xq_m, xd_m, s);   // (q_nope's int8 rows: head h's at 192 h, 32-value blocks)
    {
        // k_b a head: q_nope (128) -> the cache's 512
        iq::MvJob j[64];
        for (int h0 = 0; h0 < H; h0 += 64) {
            const int nh = std::min(64, H - h0);
            for (int i = 0; i < nh; ++i) {
                const int h = h0 + i;
                j[i] = job(l.k_b, h * g.kv_lora, g.kv_lora, xq_m + (size_t) h * g.qk_head,
                           xd_m + (size_t) h * g.qk_head / 32, qabs + (size_t) h * g.kv_lora);
            }
            run_jobs(j, nh);
        }
    }
    k::mla_rope_q(qf, 1, H, 0, dpos, theta_scale, s);
    k::mla_kv_store(kva, l.kv_a_norm, g.eps, l.kv, 1, 0, dpos, theta_scale, s);
    k::mla_decode(qabs, qf, l.kv, dpos, H, mla_scale, attn_nsp, attn_scr, mo, xq_m, xd_m, s);
    {
        // v_b a head: the 512 -> 128
        iq::MvJob j[64];
        for (int h0 = 0; h0 < H; h0 += 64) {
            const int nh = std::min(64, H - h0);
            for (int i = 0; i < nh; ++i) {
                const int h = h0 + i;
                j[i] = job(l.v_b, h * g.v_head, g.v_head, xq_m + (size_t) h * g.kv_lora,
                           xd_m + (size_t) h * g.kv_lora / 32, mob + (size_t) h * g.v_head);
            }
            run_jobs(j, nh);
        }
    }
    k::head_gate(mob, gh, H, g.v_head, mob, xq_a, xd_a, s);
    mv(l.wo, nullptr, xq_a, xd_a, ao);
    k::add(x, ao, E, s);
}

// A dense layer's FFN: x += it
void Engine::Impl::tok_dense_ffn(const Layer& l) {
    const int E = g.n_embd;
    k::norm_q8(x, l.ffn_norm, g.eps, f, xq_f, xd_f, E, s);
    {
        const DMat* ms[2] = {&l.gate, &l.up};
        float* ys[2] = {fg, fu};
        mv_many(ms, ys, 2, f, xq_f, xd_f);
    }
    k::swiglu(fg, fu, fh, g.n_ff, l.clamp_sh, s);
    k::q8(fh, 1, g.n_ff, xq_h, xd_h, s);
    mv(l.down, fh, xq_h, xd_h, fo);
    k::add(x, fo, E, s);
}

// A MoE layer's router: f = rmsnorm(x) and its int8 rows (the VRAM experts' and the shared expert's input), the
// logits, f and the logits to the host.
void Engine::Impl::tok_router(const Layer& l) {
    const int E = g.n_embd;
    k::norm_q8(x, l.ffn_norm, g.eps, f, xq_f, xd_f, E, s);
    mv(l.gate_inp, f, xq_f, xd_f, rlog);
    CK(cudaMemcpyAsync(h_f, f, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_rlog, rlog, (size_t) g.n_expert * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// A MoE layer's end: the CPU lane's sum and which lanes ran (h_tail, written by the host) up to the device, then
// x += the routed experts' sum + the shared expert's
void Engine::Impl::tok_tail() {
    CK(cudaMemcpyAsync(d_tail, h_tail, tail_bytes(), cudaMemcpyHostToDevice, s));
    k::moe_tail(x, d_tail + 4, ec.routed, so, (const int*) d_tail, g.n_embd, s);
}

// The output head: the logits to the host
void Engine::Impl::tok_head() {
    k::norm_q8(x, out_norm, g.eps, a, xq_a, xd_a, g.n_embd, s);
    mv(output, a, xq_a, xd_a, logits);
    CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// The token path's segments between its host syncs: 0 the embedding (d_erow), the dense layers, the first MoE layer's
// attention and router; il (a MoE layer after the first) the layer before's end, its attention and router; n_layer
// the last layer's end and the head.
void Engine::Impl::tok_seg(int seg) {
    if (seg == g.n_layer) {
        tok_tail();
        tok_head();
        return;
    }
    if (seg == 0) {
        // the token's row: row dpos[1] of d_erow (the row copied up: 0) or of the table in VRAM (the token)
        if (!k::embed_rows(embd_host ? d_erow : embd.w, embd.type, embd.rb, dpos + 1, 1, g.n_embd, x, s)) {
            std::fprintf(stderr, "maya-ling: no embedding rows for type %d\n", embd.type);
            std::abort();
        }
        for (int il = 0; il <= g.n_dense && il < g.n_layer; ++il) {
            Layer& l = L[(size_t) il];
            if (l.mla) tok_mla(l);
            else tok_kda(l);
            if (l.dense) tok_dense_ffn(l);
            else tok_router(l);
        }
        return;
    }
    Layer& l = L[(size_t) seg];   // (a MoE layer after the first: the layer before's end first)
    tok_tail();
    if (l.mla) tok_mla(l);
    else tok_kda(l);
    tok_router(l);
}

void Engine::Impl::tok_graphs() {
    graphs = 0;
    const char* v = std::getenv("LING_GRAPHS");
    if ((v && std::atoi(v) == 0) || std::getenv("LING_TRACE")) return;
    gx.assign((size_t) g.n_layer + 1, nullptr);
    auto capture = [&](int seg, cudaGraphExec_t& out) {
        cudaGraph_t gr_ = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        tok_seg(seg);
        if (cudaStreamEndCapture(s, &gr_) != cudaSuccess || !gr_) return false;
        const bool ok = cudaGraphInstantiate(&out, gr_, 0) == cudaSuccess;
        cudaGraphDestroy(gr_);
        return ok;
    };
    bool ok = capture(0, gx[0]);
    for (int il = g.n_dense + 1; il < g.n_layer && ok; ++il) ok = capture(il, gx[(size_t) il]);
    ok = ok && capture(g.n_layer, gx[(size_t) g.n_layer]);
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "maya-ling: the token path's CUDA graphs could not be made; it launches its kernels one by one\n");
        for (auto& e : gx)
            if (e) cudaGraphExecDestroy(e);
        gx.clear();
        return;
    }
    graphs = 1;
}

// Ling's routing on the host (route_kernel's arithmetic): sigmoid, + bias for the choice, the top groups by their top-2
// sums, the top n_used there (ties: the lower id); the weights the chosen sigmoids normalized x scale
void Engine::Impl::route_host(const Layer& l, const float* rl, int* sel, float* w) const {
    const int ne = g.n_expert, ng = g.n_group, epg = ne / ng;
    float p[512], sc[512];
    for (int e = 0; e < ne; ++e) {
        p[e] = 1.0f / (1.0f + std::exp(-rl[e]));
        sc[e] = p[e] + l.exp_bias[(size_t) e];
    }
    if (ng > 1) {
        float gs[16];
        for (int gi = 0; gi < ng; ++gi) {
            float a1 = -INFINITY, b1 = -INFINITY;
            for (int e = gi * epg; e < (gi + 1) * epg; ++e) {
                const float v = sc[e];
                if (v > a1) {
                    b1 = a1;
                    a1 = v;
                } else if (v > b1) {
                    b1 = v;
                }
            }
            gs[gi] = a1 + b1;
        }
        unsigned keep = 0;
        for (int kq = 0; kq < g.n_group_used; ++kq) {
            int bg = -1;
            for (int gi = 0; gi < ng; ++gi)
                if (!(keep >> gi & 1u) && (bg < 0 || gs[gi] > gs[bg])) bg = gi;
            keep |= 1u << bg;
        }
        for (int e = 0; e < ne; ++e)
            if (!(keep >> (e / epg) & 1u)) sc[e] = -INFINITY;
    }
    int idx[512];
    for (int e = 0; e < ne; ++e) idx[e] = e;
    std::partial_sort(idx, idx + g.n_used, idx + ne,
                      [&](int a1, int b1) { return sc[a1] > sc[b1] || (sc[a1] == sc[b1] && a1 < b1); });
    float sum = 0.0f;
    for (int i = 0; i < g.n_used; ++i) {
        sel[i] = idx[i];
        sum += p[idx[i]];
    }
    const float den = std::max(sum, 6.103515625e-5f);
    for (int i = 0; i < g.n_used; ++i) w[i] = p[sel[i]] / den * g.exp_scale;
}

const float* Engine::Impl::step(int token) {
    if (pos >= max_ctx || token < 0 || token >= g.vocab) return nullptr;
    const double t0 = now_ms();
    double cpu_ms = 0.0;
    const int E = g.n_embd;
    static const bool ptimes = std::getenv("LING_STEP_TIMES") != nullptr;
    double tp = t0;
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
    }
    if (graphs < 0) tok_graphs();
    ec_kv_reserve(pos + 1);

    // the token's row (to d_erow, decoded by segment 0), the position and the row's index on the device
    if (embd_host) {
        std::memcpy(h_erow, embd.w + (size_t) token * embd.rb, embd.rb);
        CK(cudaMemcpyAsync(d_erow, h_erow, embd.rb, cudaMemcpyHostToDevice, s));
    }
    h_pos[0] = pos;
    h_pos[1] = embd_host ? 0 : token;
    CK(cudaMemcpyAsync(dpos, h_pos, 2 * sizeof(int), cudaMemcpyHostToDevice, s));
    phase(pt.embed);

    static const bool trace = std::getenv("LING_TRACE") != nullptr;
    const int ne = g.n_expert, ff = g.n_ff_exp;
    for (int il = g.n_dense; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const int seg = il == g.n_dense ? 0 : il;
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] pos %d layer %d\n", pos, il);
            std::fflush(stderr);
        }
        if (graphs > 0) CK(cudaGraphLaunch(gx[(size_t) seg], s));
        else tok_seg(seg);
        const bool staging = ec_stage_begin(ec.piece);   // (a piece of the swaps' copies while the GPU works)
        CK(cudaStreamSynchronize(s));
        if (staging) ec_stage_end();
        dump("attn", il, x, pos);
        phase(pt.att);
        const double c0 = now_ms();
        int sel[16];
        float wv[16];
        route_host(l, h_rlog, sel, wv);
        phase(pt.route);
        // the chosen experts in VRAM go to the GPU, the others to the CPU lane, both at once
        int gslot[16], csel[16];
        float gw[16], cw[16];
        int ng = 0, nc = 0;
        if (route_log)
            for (int i = 0; i < g.n_used; ++i) route_tok.push_back((uint16_t) sel[i]);
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
        // the CPU lane's workers start on the rest now; this thread launches the GPU's part, then joins them
        float* h_routed = h_tail + 4;
        if (nc) lane.begin(*pool, l.cl, E, ff, csel, cw, nc, h_f, h_routed);
        if (ng) gpu_experts(l, gslot, gw, ng);
        // the shared expert (f's int8 rows)
        {
            const DMat* ms[2] = {&l.sh_gate, &l.sh_up};
            float* ys[2] = {sg, su};
            mv_many(ms, ys, 2, f, xq_f, xd_f);
        }
        k::swiglu(sg, su, sh, g.n_ff_sh, l.clamp_sh, s);
        k::q8(sh, 1, g.n_ff_sh, xq_h, xd_h, s);
        mv(l.sh_down, sh, xq_h, xd_h, so);
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] layer %d: experts %d %d %d %d %d %d %d %d, %d on the GPU; CPU lane next\n", il,
                         sel[0], sel[1], sel[2], sel[3], sel[4], sel[5], sel[6], sel[7], ng);
            std::fflush(stderr);
        }
        phase(pt.glaunch);
        if (nc) lane.end();
        cpu_ms += now_ms() - c0;
        phase(pt.cpu);
        pt.cpu_experts += nc;
        int* flags = (int*) h_tail;   // which lanes ran, for this layer's end (the next segment's start)
        flags[0] = nc > 0;
        flags[1] = ng > 0;
        phase(pt.tail);
    }

    // the last layer's end and the output head
    if (graphs > 0) CK(cudaGraphLaunch(gx[(size_t) g.n_layer], s));
    else tok_seg(g.n_layer);
    const bool staging = ec_stage_begin(4 * ec.piece);   // (the head's GEMV is the token's longest wait)
    CK(cudaStreamSynchronize(s));
    if (staging) ec_stage_end();
    ++pos;
    if (route_log) {
        std::fputc('T', route_log);
        std::fwrite(route_tok.data(), 2, route_tok.size(), route_log);
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
    phase(pt.adapt);
    st.ms_total += now_ms() - t0;
    st.ms_cpu += cpu_ms;
    ++st.tokens;
    if (ptimes && ++pt.tokens == 64) {
        const double n = (double) pt.tokens;
        std::fprintf(stderr,
                     "maya-ling: step phases, ms a token (64 tokens): embed %.2f, attention+sync %.2f, routing %.2f, GPU "
                     "launches %.2f, CPU lane %.2f (%.1f experts, %.3f ms each), tail %.2f, head %.2f, adapt %.2f; "
                     "total %.2f\n",
                     pt.embed / n, pt.att / n, pt.route / n, pt.glaunch / n, pt.cpu / n, pt.cpu_experts / n,
                     pt.cpu_experts ? pt.cpu / pt.cpu_experts : 0.0, pt.tail / n, pt.head / n, pt.adapt / n,
                     (pt.embed + pt.att + pt.route + pt.glaunch + pt.cpu + pt.tail + pt.head + pt.adapt) / n);
        pt = PhaseTimes{};
    }
    return h_logits;
}

}  // namespace ling
