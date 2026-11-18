// src/hy4/hy4_model.cu - maya-hy4's model (hy4_model.hpp): the GGUF read, the weights placed and a token's step
// through Hy4 Flash's layers - llama.cpp's hy_v4_flash graph (ggml-org/llama.cpp#32517, src/models/hy-v4-flash.cpp)
// step by step:
//
//   R_s = embd[token] for each of the hc = 4 residual streams
//   per layer:  (Engram layers) R_s += its n-gram rows' gated delta (eg_update)
//     x = sum_s pre_s R_s (pre, post from the streams: hc_read); a = rmsnorm(x, attn_norm)
//     MLA:  qr = rmsnorm(W_qa a); q = W_qb qr [H][192 nope | 64 rope]; kv = W_kva a; the cache row
//           [rmsnorm(kv[0:512]), rope(kv[512:576])]; o_h = v_b_h softmax(q'_h K^T / 16 + the head's sink) K[:, 0:512]
//           with q'_h = [k_b_h q_nope_h, rope(q_pe_h)] over the last 512 keys (sliding-window layers, rope base 1e4)
//           or the indexer's choice (sparse layers, 1e7: past 2048 keys, the top 511 groups of 4 by the scores of the
//           pooled keys, and the token's own group);
//           R_s += post_s rmsnorm(W_o (o sigmoid(W_gate a)), post_attention_norm)
//     x = sum_s pre_s R_s (the FFN's); f = rmsnorm(x, ffn_norm)
//     FFN:  layer 0 a SwiGLU FFN; the others the MoE - the top 16 of sigmoid(W_r f) + exp_probs_b, weighted by the
//           chosen sigmoids normalized x 3.998, the routed experts (SwiGLU clamped at 10) on l = W_rdown f (1536),
//           routed_up their sum, plus the shared expert on f;
//           R_s += post_s rmsnorm(that, post_ffw_norm)
//   logits = output(rmsnorm(sum_s pre_s R_s (the head's), output_norm))
//
// The machinery is maya-ling's (src/ling): the dense weights in VRAM and int8-activation GEMVs from maya-ds4's kernels,
// a token's routed experts in VRAM run on the GPU while the CPU lane takes the rest from the GGUF's mapping, the VRAM
// expert cache with its staged swaps and lent slots, the helper card.  The routed experts are in the latent width
// (n_lat) wherever maya-ling's were in n_embd.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "hy4_model.hpp"

#include "hy4_cpu.hpp"
#include "hy4_kernels.cuh"

#include "../ds4/ds4_iq.cuh"
#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <immintrin.h>
#include <atomic>
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

namespace hy4 {

namespace iq = strata::ds4::gguf::iq;

namespace {

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        cudaError_t e_ = (x);                                                                                   \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "maya-hy4: CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,            \
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
    bool swa = false, idx = false, dense = false;
    int H = 0;                     // its heads (32 sparse, 64 sliding window)
    float theta_scale = 1.0f;      // its RoPE: base^(-2 / 64) (1e7 sparse, 1e4 sliding window)
    float *attn_norm = nullptr, *ffn_norm = nullptr, *post_attn = nullptr, *post_ffn = nullptr, *sinks = nullptr;
    // the gated MLA
    DMat q_a, q_b, kv_a, k_b, v_b, wgate, wo;   // k_b: H x 512 rows of 192 (q_nope -> the cache's 512); v_b: H x 256 of 512
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    __half* kv = nullptr;   // the cache: kv_rows rows of 576 (F16); a sliding-window layer's a ring (ring rows)
    int kv_rows = 0, ring = 0;
    // the indexer (the sparse layers): its key and query projections, the key's norm, the heads' weights (F32), the
    // pooling's value and gate projections, position embedding [kpool][dim] and norm; the open group's value | gate
    // rows (ring [kpool][2 dim]) and the complete groups' pooled keys (pooled [max_ctx / kpool][dim])
    DMat idx_k, idx_qb, idx_proj, idx_v, idx_g;
    float *idx_knorm = nullptr, *idx_ape = nullptr, *idx_pnorm = nullptr;
    float *idx_ring = nullptr, *idx_pooled = nullptr;
    // the hyper-connections: before the attention (a) and before the FFN (f); fn [2 hc][hc E], scale [2], base [2 hc]
    float *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr;
    float *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr;
    // Engram (eg >= 0: its module, the n-gram rows read before this layer's attention)
    int eg = -1;
    DMat eg_kv;   // [(hc + 1) E] rows of cols x dim
    float *eg_kv_b = nullptr, *eg_gate = nullptr, *eg_lnorm = nullptr, *eg_gamma = nullptr, *eg_hs = nullptr;
    // the FFN
    DMat gate, up, down;           // the dense layer
    DMat gate_inp;                 // MoE layers: n_expert rows (F32)
    DMat lat_down, lat_up;         // E -> n_lat before the routed experts, n_lat -> E after
    std::vector<float> exp_bias;   // (host: a token's routing is done there)
    float* exp_bias_d = nullptr;
    DMat sh_gate, sh_up, sh_down;  // the shared expert, at the full width (no clamp)
    CpuLayer cl;
    float clamp_exp = 0.0f;
};

constexpr float kEgEps = 1.1920929e-07f;
// the sliding-window layers' cache rows: the window and room past it (the prompt path's chunks will need more)
constexpr int kSwaRing = 1024;   // (Engram's norms: the fp32 machine epsilon, as the reference)

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
    // chunk's gathered into the pinned staging and copied to pf.Erows.  HY4_EMBD_VRAM=1: the table in VRAM.
    bool embd_host = true;
    uint8_t *h_erow = nullptr, *d_erow = nullptr;
    float* out_norm = nullptr;
    std::unique_ptr<Pool> pool;
    CpuLane lane;   // the token path's CPU experts, a layer at a time
    size_t dev_bytes = 0;

    // device scratch (the token path)
    float *x = nullptr, *a = nullptr, *f = nullptr, *ao = nullptr, *fo = nullptr;
    float* R = nullptr;                          // the hc residual streams [hc][E]
    float *head_fn = nullptr, *head_scale = nullptr, *head_base = nullptr;   // the output head's hc read
    float *hmix = nullptr, *hpost = nullptr;     // a hyper-connection's mixes [2 hc], its post weights [hc]
    float *qa = nullptr, *kva = nullptr, *gh = nullptr, *qf = nullptr, *qabs = nullptr, *mo = nullptr, *mob = nullptr;
    float *fg = nullptr, *fu = nullptr, *fh = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *so = nullptr;
    float *lat = nullptr, *lsum = nullptr;       // the routed experts' input (n_lat), their sum (n_lat)
    float *logits = nullptr, *rlog = nullptr, *attn_scr = nullptr;
    int* dpos = nullptr;
    int attn_nsp = 128;
    float* d_tail = nullptr;   // a MoE layer's end: which lanes ran (2 ints, 2 spare), the CPU lane's sum (n_lat)
    int8_t *xq_a = nullptr, *xq_f = nullptr, *xq_h = nullptr, *xq_m = nullptr, *xq_l = nullptr, *xq_e = nullptr;
    float *xd_a = nullptr, *xd_f = nullptr, *xd_h = nullptr, *xd_m = nullptr, *xd_l = nullptr, *xd_e = nullptr;
    // Engram: a token's rows for every module [n_eg][cols][dim] (F32, from h_eg), the normed rows, the projection
    float *eg_rows_d = nullptr, *eg_y = nullptr, *eg_kvo = nullptr;
    // the indexer's: the token's key (raw, normed), value, gate, the heads' queries [nh][dim] and weights [nh], the
    // groups' scores, the cells chosen and their count; snap_ring the open groups' rows at snap_pos
    float *ik = nullptr, *ikn = nullptr, *iv = nullptr, *ig = nullptr, *iq = nullptr, *iw = nullptr, *iscore = nullptr;
    int *cells = nullptr, *ncell = nullptr;
    int8_t* xq_i = nullptr;
    float* xd_i = nullptr;
    float* snap_ring = nullptr;
    uint8_t* snap_kv = nullptr;   // the sliding-window layers' rings at snap_pos
    int max_pools = 0;
    // host (pinned)
    float *h_f = nullptr, *h_rlog = nullptr, *h_tail = nullptr, *h_logits = nullptr, *h_eg = nullptr;
    int* h_pos = nullptr;   // [0] the position, [1] the token (or 0), [2] the token's sequence number (tseq)
    // The device's routing (the token path; HY4_DEV_ROUTE=0: the host's, as before).  Each MoE layer routes on the
    // device, runs its experts in the main card's cache from a table there (where: every key's slot, uploaded when it
    // changed; slotptr: the slots' pointers) and its shared expert in the segment's graph, and publishes its plan to
    // host-mapped memory; the host reads the plans in order (no syncs), runs the CPU lane's and the helper's experts
    // and answers there; the device waits for an answer only where the plan sent experts to the host.
    bool dr = false;
    int *d_where = nullptr, *h_where = nullptr;
    std::vector<int> where_up;   // (the table as last uploaded)
    const uint8_t** d_slotptr = nullptr;
    const uint8_t** d_gptr = nullptr;
    int *d_sel = nullptr, *d_need = nullptr;
    float* d_wsel = nullptr;
    int *m_plan = nullptr, *m_ans = nullptr, *md_plan = nullptr, *md_ans = nullptr;   // (host / device pointers)
    unsigned *m_diag = nullptr, *md_diag = nullptr;
    unsigned diag_seen = 0;
    bool fault = false;   // (a card gave up waiting: see step_dr's end; reset() clears it)
    size_t plan_ints = 0, ans_ints = 0;
    int tseq = 0;
    void dr_alloc();
    void dr_where();
    void tok_moe(const Layer& l, int il);
    void step_dr(double& cpu_ms, double& tp);
    // ... and the helper card's experts from its own queued work (hd_alloc): its slots for every key (hwhere, as
    // where), their pointers, a layer's choices there, weights, latent and count, the position; its answers in
    // host-mapped memory (m_hans; mdh_*: the helper's device pointers); its per-layer graphs
    bool hd = false;
    int *hd_where = nullptr, *hh_where = nullptr, *hh_pos = nullptr;
    std::vector<int> hwhere_up;
    const uint8_t** hd_slotptr = nullptr;
    const uint8_t** hd_gptr = nullptr;
    float *hd_w = nullptr, *hd_lat = nullptr;
    int *hd_nh = nullptr, *hd_dpos = nullptr;
    int *m_hans = nullptr, *md_hans = nullptr, *mdh_hans = nullptr, *mdh_plan = nullptr;
    unsigned* mdh_diag = nullptr;
    std::vector<cudaGraphExec_t> hgx;
    int hgraphs = -1;
    unsigned hdiag_seen = 0;
    void hd_alloc();
    void hd_layer(const Layer& l, int il);
    void hd_graphs();
    void hd_start();
    void hd_launch(int il);
    // Engram's tables and the token history its n-grams read (hist[p]: the token at position p)
    const uint8_t* eg_table = nullptr;
    size_t eg_rb = 0;
    std::vector<uint64_t> eg_mult, eg_off, eg_prime;
    std::vector<int64_t> eg_map;
    std::vector<int> hist;
    float fp8_lut[256];
    void eg_fill(int token);   // h_eg for the token at pos (its n-grams with hist's)
    Stats st;
    // (HY4_STEP_TIMES) the token path's host time by phase, printed every 64 tokens
    struct PhaseTimes {
        double att = 0, route = 0, glaunch = 0, cpu = 0, tail = 0, head = 0, adapt = 0, embed = 0;
        long cpu_experts = 0, tokens = 0;
    } pt;
    // (HY4_ROUTE_LOG=FILE) the routing as it goes, for replaying the expert cache's choices offline: 'T' + a token's
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
    // HY4_TRIM=0: kept.
    static void trim(const void* p, size_t n) {
#ifdef _WIN32
        static const bool on = !std::getenv("HY4_TRIM") || std::atoi(std::getenv("HY4_TRIM")) != 0;
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

    // (HY4_DUMP=1) a row of the residual stream after a layer's attention or FFN: its RMS and first values, to stderr -
    // the token path's (after each segment) and the prompt path's last token (after each part) compared line by line
    void dump(const char* what, int il, const float* dev, int at) {
        static const bool on = std::getenv("HY4_DUMP") != nullptr;
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
            std::fprintf(stderr, "maya-hy4: no GEMV for these jobs (type %d, k %d)\n", j[0].type, j[0].k);
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
    void tok_attn(Layer& l);
    void tok_dense_ffn(const Layer& l);
    void tok_tail(const Layer& l);
    void tok_router(const Layer& l);
    void tok_head();
    void tok_seg(int seg);   // the graphs' bodies: 0 the embedding .. the first MoE layer's router, il, n_layer: head
    void tok_graphs();
    size_t tail_bytes() const { return ((size_t) 4 + g.n_lat) * sizeof(float); }
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
        // Hy4's: the hc streams [C][hc][E], the second output (the shared expert's), the hc mixes and posts, the latent
        // rows (in, out, F32 and F16), Engram's piece (rows F32 from h_egr, normed, F16, projected), the attention's
        // piece, the indexer's (keys, values, gates, weights, queries, the choice's bits and sel_on, the scoring tile)
        float *R = nullptr, *Y2 = nullptr, *MIX = nullptr, *POST = nullptr, *LAT = nullptr, *YL = nullptr;
        __half *LATh = nullptr, *YLh = nullptr;
        float *EGR = nullptr, *EGY = nullptr, *KVO = nullptr, *h_egr = nullptr;
        __half* EGYh = nullptr;
        float *QA = nullptr, *KVA = nullptr, *GATE = nullptr, *Q = nullptr, *OB = nullptr;
        __half *QAh = nullptr, *QH = nullptr, *Qcat = nullptr, *O16 = nullptr, *OBh = nullptr, *Kbuf = nullptr;
        float *IK = nullptr, *IKN = nullptr, *IV = nullptr, *IG = nullptr, *IW = nullptr, *IQ = nullptr, *IA = nullptr,
              *IS = nullptr;
        __half* IKNh = nullptr;
        unsigned* bits = nullptr;
        int *sel_on = nullptr, *icells = nullptr;
        int nwords = 0, top_pools = 0, tqi = 1;
    } pf;
    // The experts' copies to the GPU: the CPU lane's threads copy into two pinned buffers in turn and the DMA takes each
    // from there (the driver's own staging of pageable memory moved 4-15 GB/s on this machine; this held ~14 for
    // maya-ink).  HY4_PF_PAGEABLE=1: the driver's staging.
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
    void pf_index(Layer& l, const float* A_, const __half* Ah_, int T, int p0);
    void pf_attn(Layer& l, const float* A_, const __half* Ah_, int T, int p0, float* Y_);
    void eg_row(int e, int token, int p, float* dst) const;
    void pf_moe(const Layer& l, int T);
    void chunk(const int* ids, int T, bool want_logits);
    const float* prefill(const int* ids, int n, void (*progress)(void*, int), void* ctx);
    // (HY4_PF_PREFETCH=1) A chunk of HY4_PF_PREFETCH_MIN tokens (4096) or more uses nearly every expert of every
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
    // A swap's copy is staged in pieces (HY4_CACHE_STAGE_KB, 2 MB) while the CPU waits for a layer's attention on the
    // GPU, and the expert is used from the token after the one its last piece went up in.  HY4_CACHE_DEFER=0: copied
    // at the swap.
    // A prompt chunk's counts go on the written tokens' scale (HY4_CACHE_PROMPT=2, the default since cc69fc0; pf_score):
    // each expert gets its share of what T written tokens would leave, and the layer's scores first decay as those
    // tokens would.  0 is the old raw counts - thousands after a long prompt, where a written token adds 1 and an expert
    // every token chose settles at 99, so no swap cleared the bar for some hundred tokens; 1 scales without the decay.
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
        // how a prompt chunk's choices count (HY4_CACHE_PROMPT; pf_score).  2: on the written tokens' scale, the layer's
        // scores decayed as T tokens would - raw counts (0) leave ~1,000 an expert after 32K against decode's ~1 a token,
        // and the cache cannot swap for hundreds of tokens (the review's meas-v4: 32K 20.0 -> 18.0 ms a token)
        int prompt = 2;
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
        return (size_t) g.n_ff_exp * (c.rg + c.ru) + (size_t) g.n_lat * c.rd;
    }
    void ec_alloc();
    void ec_copy(int key, int slot);
    void ec_copy_many(const std::vector<std::pair<int, int>>& todo);
    void ec_fill();
    void ec_adapt();
    void ec_reclaim();
    void ec_flush();
    bool ec_stage_begin(size_t budget, bool wait = true);
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
    // A second card (Options::helper_gpu) that holds routed experts only (maya-nem's, branch nemotron-2gpu): slots sized
    // by class as the main cache's, filled with the best-scored experts the main card does not hold.  A token's experts
    // found there get f from the host (h_f, there after the layer's sync), are multiplied on the helper's own stream
    // while the main card and the CPU lane do theirs, and their sum comes back to the host, where it is added to the CPU
    // lane's.  The main cache decides first and leaves the helper's experts alone; the helper follows it: every `every`
    // tokens up to `per` experts change places, and after a prompt up to an eighth of its tokens (64 .. fill_cap).
    // The swaps between tokens are copied by a thread of their own on a second stream while the next tokens are
    // written, and an expert is used from the first update after its copy is done; the fills at the start and after
    // a prompt are copied at once, through pinned staging the CPU lane's threads fill (hc_copy_many: as the main
    // cache's fills; HY4_HELPER_STAGE=0: the driver's staging of the mapping, as before).
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
        int every = 1, per = 32, fill_cap = 4096;  // (see docs/LING.md, "Two cards")
        float margin = -1.0f, bias = 0.5f;        // a swap's bar: score > margin * the slot's + bias (-1: the main cache's)
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
        // (HY4_HELPER_KEEPALIVE) a one-warp kernel kept running on the helper while tokens are written (each lasts
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
    void hc_experts(const Layer& l, const int* slot, const float* w, int n, const float* xin = nullptr);

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

bool Engine::snapshot() {
    Impl& m = *p;
    // (the indexer's open groups: their slots are written again by the tokens after the snapshot)
    const size_t ring = (size_t) m.g.idx_kpool * 2 * m.g.idx_dim * sizeof(float);
    uint8_t* at = (uint8_t*) m.snap_ring;
    for (const Layer& l : m.L)
        if (l.idx) {
            CK(cudaMemcpyAsync(at, l.idx_ring, ring, cudaMemcpyDeviceToDevice, m.s));
            at += ring;
        }
    // (and the sliding-window layers' rings: the rows the window still reads are written again past the snapshot)
    at = (uint8_t*) m.snap_kv;
    for (const Layer& l : m.L)
        if (l.ring) {
            CK(cudaMemcpyAsync(at, l.kv, (size_t) l.ring * 576 * sizeof(__half), cudaMemcpyDeviceToDevice, m.s));
            at += (size_t) l.ring * 576 * sizeof(__half);
        }
    CK(cudaStreamSynchronize(m.s));
    m.snap_pos = m.pos;
    return true;
}

bool Engine::restore() {
    Impl& m = *p;
    if (m.snap_pos < 0) return false;
    const size_t ring = (size_t) m.g.idx_kpool * 2 * m.g.idx_dim * sizeof(float);
    const uint8_t* at = (const uint8_t*) m.snap_ring;
    for (Layer& l : m.L)
        if (l.idx) {
            CK(cudaMemcpyAsync(l.idx_ring, at, ring, cudaMemcpyDeviceToDevice, m.s));
            at += ring;
        }
    at = (const uint8_t*) m.snap_kv;
    for (Layer& l : m.L)
        if (l.ring) {
            CK(cudaMemcpyAsync(l.kv, at, (size_t) l.ring * 576 * sizeof(__half), cudaMemcpyDeviceToDevice, m.s));
            at += (size_t) l.ring * 576 * sizeof(__half);
        }
    CK(cudaStreamSynchronize(m.s));
    m.pos = m.snap_pos;
    return true;
}

bool Engine::failed() const { return p->fault; }

void Engine::reset() {
    p->fault = false;
    p->pos = 0;
    p->snap_pos = -1;
    p->ec.kv_hi = 0;   // (the caches' slots usable again: empty, the next fill or swaps take them)
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
    if (!arch || arch->s != "hy_v4_flash") {
        err = "not a Hy4 Flash GGUF (general.architecture " + (arch ? arch->s : std::string("missing")) +
              ", hy_v4_flash expected)";
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
    const std::string A = "hy_v4_flash.";
    const int n_all = meta_int(f0, A + "block_count");
    g.n_layer = n_all - std::max(0, meta_int(f0, A + "nextn_predict_layers", 0));
    g.n_embd = meta_int(f0, A + "embedding_length");
    g.n_ff = meta_int(f0, A + "feed_forward_length");
    g.eps = meta_f(f0, A + "attention.layer_norm_rms_epsilon", 1e-5f);
    g.n_expert = meta_int(f0, A + "expert_count");
    g.n_used = meta_int(f0, A + "expert_used_count");
    g.n_ff_exp = meta_int(f0, A + "expert_feed_forward_length");
    const int n_shexp = meta_int(f0, A + "expert_shared_count", 1);
    g.n_ff_sh = g.n_ff_exp * std::max(1, n_shexp);
    g.n_dense = meta_int(f0, A + "leading_dense_block_count", 0);
    g.n_lat = meta_int(f0, A + "expert_latent_length", g.n_embd);
    g.exp_scale = meta_f(f0, A + "expert_weights_scale", 1.0f);
    const int w_norm = meta_int(f0, A + "expert_weights_norm", 0), gating = meta_int(f0, A + "expert_gating_func", 2);
    g.q_lora = meta_int(f0, A + "attention.q_lora_rank", 0);
    g.kv_lora = meta_int(f0, A + "attention.kv_lora_rank");
    g.rope_dim = meta_int(f0, A + "rope.dimension_count");
    g.qk_head = meta_int(f0, A + "attention.key_length_mla");
    g.v_head = meta_int(f0, A + "attention.value_length_mla");
    g.rope_base = meta_f(f0, A + "rope.freq_base", 1e7f);
    g.rope_base_swa = meta_f(f0, A + "rope.freq_base_swa", 1e4f);
    g.swa_window = meta_int(f0, A + "attention.sliding_window", 0);
    g.idx_heads = meta_int(f0, A + "attention.indexer.head_count", 0);
    g.idx_dim = meta_int(f0, A + "attention.indexer.key_length", 0);
    g.idx_top_k = meta_int(f0, A + "attention.indexer.top_k", 0);
    g.idx_kpool = meta_int(f0, A + "attention.indexer.kpool", 0);
    g.idx_tail = meta_int(f0, A + "attention.indexer.kpool_select_tail", 0);
    g.hc = meta_int(f0, A + "hyper_connection.count", 0);
    g.hc_eps = meta_f(f0, A + "hyper_connection.epsilon", 1e-6f);
    g.hc_mag = meta_f(f0, A + "hyper_connection.magnitude", 2.0f);
    g.vocab = meta_int(f0, A + "vocab_size");
    if (g.vocab <= 0)
        if (const strata::MetaValue* tv = f0.get("tokenizer.ggml.tokens")) g.vocab = (int) tv->count;
    g.ctx_train = meta_int(f0, A + "context_length");
    g.eos = meta_int(f0, "tokenizer.ggml.eos_token_id", -1);
    {
        const std::vector<double> nh = meta_arr(f0, A + "attention.head_count", n_all, 0.0);
        const std::vector<double> sw = meta_arr(f0, A + "attention.sliding_window_pattern", n_all, 0.0);
        const std::vector<double> it = meta_arr(f0, A + "attention.indexer.types", n_all, 0.0);
        const std::vector<double> ce = meta_arr(f0, A + "swiglu_clamp_exp", n_all, 0.0);
        g.n_head.assign((size_t) g.n_layer, 0);
        g.swa.assign((size_t) g.n_layer, 0);
        g.idx.assign((size_t) g.n_layer, 0);
        g.clamp_exp.assign((size_t) g.n_layer, 0.0f);
        for (int il = 0; il < g.n_layer; ++il) {
            g.n_head[(size_t) il] = (int) nh[(size_t) il];
            g.swa[(size_t) il] = sw[(size_t) il] != 0.0 ? 1 : 0;
            g.idx[(size_t) il] = it[(size_t) il] != 0.0 ? 1 : 0;
            g.clamp_exp[(size_t) il] = ce[(size_t) il] > 1e-6 ? (float) ce[(size_t) il] : 0.0f;
        }
    }
    // Engram's hashing: the modules' layers, multipliers [module][ngram], each column's bucket offset and count
    // [module][cols], the token map (a token id to the id its n-grams hash) and the pad
    {
        strata::MetaValue v;
        if (f0.get_full(A + "ple.layers", v))
            for (const auto& it : v.items) g.eg_layers.push_back((int) it.u);
        g.eg_ngram = meta_int(f0, A + "ple.ngram_size", 0);
        g.eg_heads = meta_int(f0, A + "ple.heads_per_ngram", 0);
        g.eg_dim = meta_int(f0, A + "embedding_length_per_layer_input", 0);
        g.eg_cols = g.eg_ngram >= 2 ? (g.eg_ngram - 1) * g.eg_heads : 0;
        g.eg_pad = meta_int(f0, A + "engram.pad_token_id", 0);
        auto u64s = [&](const char* k, std::vector<uint64_t>& out) {
            strata::MetaValue mv_;
            out.clear();
            if (f0.get_full(A + k, mv_))
                for (const auto& it : mv_.items) out.push_back(it.u);
        };
        u64s("ple.layer_multipliers", eg_mult);
        u64s("ple.head_offsets", eg_off);
        u64s("ple.head_vocab_sizes", eg_prime);
        strata::MetaValue tm;
        if (f0.get_full(A + "engram.token_map", tm))
            for (const auto& it : tm.items) eg_map.push_back((int64_t) it.u);
        const size_t ne_ = g.eg_layers.size();
        if (ne_ && (eg_mult.size() != ne_ * g.eg_ngram || eg_off.size() != ne_ * g.eg_cols ||
                    eg_prime.size() != ne_ * g.eg_cols || g.eg_pad >= (int) eg_map.size() || g.eg_dim % 32 != 0 ||
                    g.eg_dim > 1024)) {
            err = "the Engram metadata is inconsistent (ple.*, engram.*)";
            return false;
        }
    }
    const int E = g.n_embd;
    const int Hmax = *std::max_element(g.n_head.begin(), g.n_head.end());
    if (g.n_layer <= 0 || E <= 0 || g.n_used > 16 || g.n_used < 1 || g.n_expert > 512 || g.n_expert % 32 != 0 ||
        g.kv_lora != 512 || g.rope_dim != 64 || g.qk_head - g.rope_dim <= 0 || (g.qk_head - g.rope_dim) % 32 != 0 ||
        g.v_head % 32 != 0 || g.q_lora <= 0 || n_shexp != 1 || w_norm != 1 || gating != 2 || g.hc < 1 || g.hc > 8 ||
        g.n_lat <= 0 || g.n_lat % 256 != 0 || Hmax % 16 != 0 || !k::norm_q8_ok(E) || !k::norm_q8_ok(g.q_lora) ||
        g.vocab <= 0) {
        err = "a Hy4 Flash shape this engine does not take (it is written for Hy4 Flash: MLA with kv_lora 512 + 64 rope "
              "dims, heads a multiple of 16, sigmoid routing of at most 16 of at most 512 latent experts, one shared "
              "expert, 1-8 hyper-connection streams)";
        return false;
    }
    C = 0;
    max_ctx = std::min(o.max_ctx, g.ctx_train > 0 ? g.ctx_train : o.max_ctx);
    mla_scale = 1.0f / std::sqrt((float) g.qk_head);

    CK(cudaSetDevice(o.gpu));
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // the weights in VRAM
    if (const char* v = std::getenv("HY4_EMBD_VRAM")) embd_host = std::atoi(v) == 0;
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
    if (!(head_fn = f32("output_hc_fn.weight", (size_t) g.hc * g.hc * E, err)) ||
        !(head_scale = f32("output_hc_scale.weight", 1, err)) || !(head_base = f32("output_hc_base.weight", (size_t) g.hc, err)))
        return false;
    // the Engram table: rows of eg_dim F8_E4M3 values in the GGUF's mapping
    if (!g.eg_layers.empty()) {
        const Tensor* t = find("per_layer_token_embd.weight");
        if (!t || (int) t->info->type != 43 || t->info->shape.size() < 2 || (int64_t) t->info->shape[0] != g.eg_dim) {
            err = "per_layer_token_embd.weight: not rows of " + std::to_string(g.eg_dim) + " F8_E4M3 values";
            return false;
        }
        eg_table = t->data;
        eg_rb = (size_t) g.eg_dim;
        const uint64_t rows = t->info->shape[1];
        for (size_t c = 0; c < eg_off.size(); ++c)
            if (eg_prime[c] < 2 || eg_off[c] + eg_prime[c] > rows) {
                err = "Engram column " + std::to_string(c) + ": its buckets run past the table's rows";
                return false;
            }
        // F8_E4M3 (1 sign, 4 exponent (bias 7), 3 mantissa; 0x7f / 0xff NaN, no infinities) to F32
        for (int b = 0; b < 256; ++b) {
            const int sgn = b >> 7, ex = (b >> 3) & 15, mt = b & 7;
            float v = ex == 0 ? std::ldexp((float) mt, -9) : std::ldexp(1.0f + mt / 8.0f, ex - 7);
            if (ex == 15 && mt == 7) v = NAN;
            fp8_lut[b] = sgn ? -v : v;
        }
    }
    L.resize((size_t) g.n_layer);
    const int kvd = g.kv_lora + g.rope_dim, nope = g.qk_head - g.rope_dim, HC = g.hc;
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        l.swa = g.swa[(size_t) il] != 0;
        l.idx = g.idx[(size_t) il] != 0;
        l.dense = il < g.n_dense;
        l.H = g.n_head[(size_t) il];
        l.theta_scale = std::pow(l.swa ? g.rope_base_swa : g.rope_base, -2.0f / (float) g.rope_dim);
        l.clamp_exp = g.clamp_exp[(size_t) il];
        const int H = l.H;
        if (!(l.attn_norm = f32(b + "attn_norm.weight", (size_t) E, err)) ||
            !(l.ffn_norm = f32(b + "ffn_norm.weight", (size_t) E, err)) ||
            !(l.post_attn = f32(b + "post_attention_norm.weight", (size_t) E, err)) ||
            !(l.post_ffn = f32(b + "post_ffw_norm.weight", (size_t) E, err)) ||
            !(l.sinks = f32(b + "attn_sinks.weight", (size_t) H, err)))
            return false;
        if (!mat(b + "attn_q_a.weight", E, g.q_lora, l.q_a, err) ||
            !mat(b + "attn_q_b.weight", g.q_lora, H * g.qk_head, l.q_b, err) ||
            !mat(b + "attn_kv_a_mqa.weight", E, kvd, l.kv_a, err) ||
            !mat(b + "attn_k_b.weight", nope, g.kv_lora, l.k_b, err, H) ||
            !mat(b + "attn_v_b.weight", g.kv_lora, g.v_head, l.v_b, err, H) ||
            !mat(b + "attn_gate.weight", E, H * g.v_head, l.wgate, err) ||
            !mat(b + "attn_output.weight", H * g.v_head, E, l.wo, err))
            return false;
        if (is_float(l.k_b.type) || is_float(l.v_b.type) || is_float(l.q_b.type) || is_float(l.wo.type)) {
            err = b + "attn_k_b / attn_v_b / attn_q_b / attn_output: a float type (the token path takes them as int8 "
                      "GEMVs)";
            return false;
        }
        if (!(l.q_a_norm = f32(b + "attn_q_a_norm.weight", (size_t) g.q_lora, err)) ||
            !(l.kv_a_norm = f32(b + "attn_kv_a_norm.weight", (size_t) g.kv_lora, err)))
            return false;
        if (!(l.hca_fn = f32(b + "hc_attn_fn.weight", (size_t) 2 * HC * HC * E, err)) ||
            !(l.hca_scale = f32(b + "hc_attn_scale.weight", 2, err)) ||
            !(l.hca_base = f32(b + "hc_attn_base.weight", (size_t) 2 * HC, err)) ||
            !(l.hcf_fn = f32(b + "hc_ffn_fn.weight", (size_t) 2 * HC * HC * E, err)) ||
            !(l.hcf_scale = f32(b + "hc_ffn_scale.weight", 2, err)) ||
            !(l.hcf_base = f32(b + "hc_ffn_base.weight", (size_t) 2 * HC, err)))
            return false;
        // (the sparse layers' attention is dense for now: the indexer's selection comes next, and below its top_k
        //  tokens of context it keeps every one anyway)
        // a sliding-window layer reads only its last swa_window rows: a ring of kSwaRing rows (position p in row
        // p % kSwaRing), not a row for every position - 30 layers x 1.1 KB a token of VRAM the expert cache keeps
        l.ring = l.swa ? kSwaRing : 0;
        l.kv_rows = l.swa ? kSwaRing : max_ctx;
        l.kv = dalloc<__half>((size_t) l.kv_rows * 576);
        if (l.idx) {
            const int D = g.idx_dim, NH = g.idx_heads, KP = g.idx_kpool;
            if (!mat(b + "indexer.attn_k.weight", E, D, l.idx_k, err) ||
                !mat(b + "indexer.attn_q_b.weight", g.q_lora, NH * D, l.idx_qb, err) ||
                !mat(b + "indexer.proj.weight", E, NH, l.idx_proj, err) ||
                !mat(b + "indexer_compressor_value.weight", D, D, l.idx_v, err) ||
                !mat(b + "indexer_compressor_gate.weight", D, D, l.idx_g, err))
                return false;
            if (!(l.idx_knorm = f32(b + "indexer.k_norm.weight", (size_t) D, err)) ||
                !(l.idx_ape = f32(b + "indexer_compressor_ape.weight", (size_t) KP * D, err)) ||
                !(l.idx_pnorm = f32(b + "indexer_compressor_norm.weight", (size_t) D, err)))
                return false;
            l.idx_ring = dalloc<float>((size_t) KP * 2 * D);
            l.idx_pooled = dalloc<float>((size_t) (max_ctx / KP + 1) * D);
        }
        for (size_t e = 0; e < g.eg_layers.size(); ++e)
            if (g.eg_layers[e] == il) l.eg = (int) e;
        if (l.eg >= 0) {
            const int cd = g.eg_cols * g.eg_dim;
            if (!mat(b + "engram_kv.weight", cd, (HC + 1) * E, l.eg_kv, err)) return false;
            if (!(l.eg_kv_b = f32(b + "engram_kv_b.weight", (size_t) (HC + 1) * E, err)) ||
                !(l.eg_gate = f32(b + "engram_gate.weight", (size_t) HC * E, err)) ||
                !(l.eg_lnorm = f32(b + "engram_lookup_norm.weight", (size_t) cd, err)) ||
                !(l.eg_gamma = f32(b + "engram_pre_gamma.weight", (size_t) E, err)) ||
                !(l.eg_hs = f32(b + "engram_hc_scale.weight", (size_t) HC * E, err)))
                return false;
        }
        if (l.dense) {
            if (!mat(b + "ffn_gate.weight", E, g.n_ff, l.gate, err) || !mat(b + "ffn_up.weight", E, g.n_ff, l.up, err) ||
                !mat(b + "ffn_down.weight", g.n_ff, E, l.down, err))
                return false;
        } else {
            const int ne = g.n_expert, NL = g.n_lat;
            if (!mat(b + "ffn_gate_inp.weight", E, ne, l.gate_inp, err)) return false;
            if (!(l.exp_bias_d = f32(b + "exp_probs_b.bias", (size_t) ne, err, &l.exp_bias))) return false;
            if (!mat(b + "ffn_routed_down.weight", E, NL, l.lat_down, err) ||
                !mat(b + "ffn_routed_up.weight", NL, E, l.lat_up, err))
                return false;
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
            if (!cpu_layer((int) tg->info->type, (int) tu->info->type, (int) td->info->type, NL, g.n_ff_exp, l.cl, err)) {
                err = b + "ffn_*_exps: " + err;
                return false;
            }
            l.cl.gate = tg->data;
            l.cl.up = tu->data;
            l.cl.down = td->data;
            l.cl.limit = l.clamp_exp;
            static std::vector<int> prepared;
            for (int ty : {l.cl.tg, l.cl.tu, l.cl.td})
                if (std::find(prepared.begin(), prepared.end(), ty) == prepared.end()) {
                    strata::ds4::gguf::iq_cpu_prepare_rows(ty);
                    prepared.push_back(ty);
                }
        }
        if (o.verbose && (il % 8 == 7 || il == g.n_layer - 1))
            std::fprintf(stderr, "maya-hy4: layers 0..%d placed (%.1f GB of VRAM so far)\n", il, dev_bytes / 1e9);
    }

    // scratch
    const int NL = g.n_lat, ffm = std::max(g.n_ff, g.n_ff_sh);
    x = dalloc<float>(E);
    a = dalloc<float>(E);
    f = dalloc<float>(E);
    ao = dalloc<float>(E);
    fo = dalloc<float>(E);
    R = dalloc<float>((size_t) HC * E);
    hmix = dalloc<float>(16);
    hpost = dalloc<float>(8);
    qa = dalloc<float>((size_t) g.q_lora);
    kva = dalloc<float>(576);
    gh = dalloc<float>((size_t) Hmax * g.v_head);
    qf = dalloc<float>((size_t) Hmax * g.qk_head);
    qabs = dalloc<float>((size_t) Hmax * g.kv_lora);
    mo = dalloc<float>((size_t) Hmax * g.kv_lora);
    mob = dalloc<float>((size_t) Hmax * g.v_head);
    fg = dalloc<float>((size_t) g.n_ff);
    fu = dalloc<float>((size_t) g.n_ff);
    fh = dalloc<float>((size_t) g.n_ff);
    sg = dalloc<float>((size_t) g.n_ff_sh);
    su = dalloc<float>((size_t) g.n_ff_sh);
    sh = dalloc<float>((size_t) g.n_ff_sh);
    so = dalloc<float>(E);
    lat = dalloc<float>((size_t) NL);
    lsum = dalloc<float>((size_t) NL);
    d_tail = dalloc<float>((size_t) 4 + NL);
    logits = dalloc<float>((size_t) g.vocab);
    rlog = dalloc<float>((size_t) g.n_expert);
    dpos = dalloc<int>(4);
    attn_nsp = k::mla_nsp(max_ctx);
    attn_scr = dalloc<float>(k::mla_scratch_floats(Hmax, attn_nsp));
    const size_t ka = (size_t) std::max(E, Hmax * g.v_head), kf = (size_t) E, kh = (size_t) ffm,
                 km = (size_t) std::max({g.q_lora, Hmax * g.qk_head, Hmax * g.kv_lora});
    xq_a = dalloc<int8_t>(ka);
    xd_a = dalloc<float>(ka / 32);
    xq_f = dalloc<int8_t>(kf);
    xd_f = dalloc<float>(kf / 32);
    xq_h = dalloc<int8_t>(kh);
    xd_h = dalloc<float>(kh / 32);
    xq_m = dalloc<int8_t>(km);
    xd_m = dalloc<float>(km / 32);
    xq_l = dalloc<int8_t>((size_t) NL);
    xd_l = dalloc<float>((size_t) NL / 32);
    h_f = halloc<float>((size_t) NL);
    h_rlog = halloc<float>((size_t) g.n_expert);
    h_tail = halloc<float>((size_t) 4 + NL);
    h_logits = halloc<float>((size_t) g.vocab);
    h_pos = halloc<int>(4);
    if (!g.eg_layers.empty()) {
        const size_t cd = (size_t) g.eg_cols * g.eg_dim;
        eg_rows_d = dalloc<float>(g.eg_layers.size() * cd);
        h_eg = halloc<float>(g.eg_layers.size() * cd);
        eg_y = dalloc<float>(cd);
        xq_e = dalloc<int8_t>(cd);
        xd_e = dalloc<float>(cd / 32);
        eg_kvo = dalloc<float>((size_t) (HC + 1) * E);
    }
    hist.reserve((size_t) max_ctx);
    {
        size_t rb = 0;
        for (const Layer& l : L) rb += (size_t) l.ring * 576 * sizeof(__half);
        if (rb) snap_kv = dalloc<uint8_t>(rb);
    }
    {
        int n_idx = 0;
        for (const Layer& l : L) n_idx += l.idx;
        if (n_idx) {
            const int D = g.idx_dim, NH = g.idx_heads, KP = g.idx_kpool;
            if (KP < 2 || D != 128 || NH > 16 || g.idx_top_k % KP != 0 || !g.idx_tail || g.idx_top_k / KP - 1 > 1024 ||
                g.rope_dim >= D) {
                err = "an indexer this engine does not take (it is written for Hy4 Flash's: 128 wide, at most 16 heads, "
                      "a pool size dividing top_k, the token's own group kept)";
                return false;
            }
            max_pools = max_ctx / KP + 1;
            ik = dalloc<float>(D);
            ikn = dalloc<float>(D);
            iv = dalloc<float>(D);
            ig = dalloc<float>(D);
            iq = dalloc<float>((size_t) NH * D);
            iw = dalloc<float>(NH);
            iscore = dalloc<float>((size_t) max_pools);
            cells = dalloc<int>((size_t) g.idx_top_k + KP);
            ncell = dalloc<int>(1);
            xq_i = dalloc<int8_t>(D);
            xd_i = dalloc<float>(D / 32);
            snap_ring = dalloc<float>((size_t) n_idx * KP * 2 * D);
        }
    }

    int threads = o.threads > 0 ? o.threads : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    pool.reset(new Pool(threads - 1));
    // Every routed expert's pages touched once, now: the first touch of a mapped page costs a fault (~1 us a 4 KB
    // page on Windows even with the file in RAM), which held maya-ink's prompt-path copies to 3.5 GB/s and the CPU
    // lane's first tokens back.  HY4_PREFAULT=0: not done.
    const char* pv = std::getenv("HY4_PREFAULT");
    if (!pv || std::atoi(pv) != 0) {
        const double tp = now_ms();
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        double gb = 0;
        for (const Layer& l : L) {
            if (l.dense) continue;
            const size_t gb_ = (size_t) g.n_ff_exp * l.cl.rg * g.n_expert, ub_ = (size_t) g.n_ff_exp * l.cl.ru * g.n_expert,
                         db_ = (size_t) g.n_lat * l.cl.rd * g.n_expert;
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
            std::fprintf(stderr, "maya-hy4: the routed experts' %.1f GB touched in %.1f s (working set %.1f GB)\n", gb,
                         (now_ms() - tp) / 1000.0, working_set_gb());
    }
    if (const char* rl = std::getenv("HY4_ROUTE_LOG")) route_log = std::fopen(rl, "ab");
    const size_t before_pf = dev_bytes;
    if (!pf_alloc(err)) return false;
    const size_t before_ec = dev_bytes;
    ec_alloc();
    dr_alloc();
    if (!o.profile.empty() && load_profile(o.profile)) {
        const double tf = now_ms();
        ec_fill();
        CK(cudaStreamSynchronize(s));
        if (o.verbose)
            std::fprintf(stderr, "maya-hy4: the expert profile %s: %ld experts into VRAM in %.1f s\n", o.profile.c_str(),
                         ec.copies, (now_ms() - tf) / 1000.0);
    }
    if (o.helper_gpu >= 0) {
        const double th = now_ms();
        hc_alloc();
        hd_alloc();
        if (hc.slots) {
            hc_update(INT_MAX, false, false);
            if (o.verbose)
                std::fprintf(stderr,
                             "maya-hy4: helper card CUDA%d: %d expert slots, %.2f GB of its VRAM; %ld experts into them in "
                             "%.1f s\n",
                             hc.dev, hc.slots, hc.bytes / 1e9, hc.copies, (now_ms() - th) / 1000.0);
        }
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
        int n_swa = 0;
        for (int il = 0; il < g.n_layer; ++il) n_swa += g.swa[(size_t) il];
        std::fprintf(stderr,
                     "maya-hy4: Hy4 Flash %d layers (%d sliding-window, %d sparse), %d experts (top %d + 1 shared, latent "
                     "%d), %d hc streams, Engram at %zu layers, context %d; %.2f GB in VRAM (%d expert slots %.2f, %d in the "
                     "caches' rows past the position: %s), the routed experts in the GGUF's mapping; CPU lane %d threads; "
                     "loaded in %.1f s; eos %d\n",
                     g.n_layer, n_swa, g.n_layer - n_swa, g.n_expert, g.n_used, g.n_lat, g.hc, g.eg_layers.size(), max_ctx,
                     dev_bytes / 1e9, ec.own, (dev_bytes - before_ec) / 1e9, ec.slots - ec.arena_end, cls.c_str(),
                     pool->threads(), (now_ms() - t0) / 1000.0, g.eos);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ the expert cache
void Engine::Impl::ec_alloc() {
    const int E = g.n_lat, ff = g.n_ff_exp;
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
    if (const char* v = std::getenv("HY4_CACHE_DECAY")) ec.decay = (float) std::atof(v);
    if (const char* v = std::getenv("HY4_CACHE_EVERY")) ec.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("HY4_CACHE_SWAPS")) ec.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("HY4_CACHE_MARGIN")) ec.margin = (float) std::atof(v);
    if (const char* v = std::getenv("HY4_CACHE_PROMPT")) ec.prompt = std::max(0, std::min(2, std::atoi(v)));
    if (const char* v = std::getenv("HY4_CACHE_DEFER")) ec.defer = std::atoi(v) != 0;
    if (const char* v = std::getenv("HY4_CACHE_STAGE_KB")) ec.piece = (size_t) std::max(64, std::atoi(v)) << 10;
    if (ec.defer) {
        for (int i = 0; i < 2 && ec.defer; ++i) {
            ec.defer = cudaHostAlloc((void**) &ec.apin[i], 4 * ec.piece, cudaHostAllocPortable) == cudaSuccess;
            if (ec.defer) CK(cudaEventCreateWithFlags(&ec.aev[i], cudaEventDisableTiming));
        }
        if (ec.defer) CK(cudaEventCreateWithFlags(&ec.arrived_ev, cudaEventDisableTiming));
        if (!ec.defer) cudaGetLastError();
    }
    const char* lv = std::getenv("HY4_LEND");
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
    if (const char* v = std::getenv("HY4_VRAM_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
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
    const char* kvv = std::getenv("HY4_LEND_KV");
    if (base && (!kvv || std::atoi(kvv) != 0)) {
        const size_t b = ec.cls_bytes[(size_t) ec.lend_cls], row = (size_t) 576 * sizeof(__half);
        for (const Layer& l : L) {
            if (l.ring) continue;
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
    const int il = key / ec.ne, e = key % ec.ne, E = g.n_lat, ff = g.n_ff_exp;
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
    const int E = g.n_lat, ff = g.n_ff_exp;
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
                // (an expert the helper card holds is off the CPU already: the main card's swaps go to the ones that
                //  are not, and the helper's slots stay as they are; HY4_HELPER_STEAL=1: the main card takes them too)
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

// A prompt chunk's choices in layer il (cnt: the tokens of its T that chose each expert) into the scores.
//   0: the counts as they are.
//   1: on the written tokens' scale.  T tokens written one by one leave sum(decay^k, k = 1 .. T) of score to share
//      for each of a token's choices (a token's choice adds 1, then every score decays by the token's end); the chunk
//      gives each expert its share of that, cnt / T - what its tokens would have left had they chosen it evenly
//      through the chunk.  A long chunk so adds at most 99 (decay 0.99), a 256-token one 0.36 a count.
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
    const int il = key / ec.ne, e = key % ec.ne, E = g.n_lat, ff = g.n_ff_exp;
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
bool Engine::Impl::ec_stage_begin(size_t budget, bool wait) {
    if (!ec.defer || ec.pend_head >= ec.pend.size()) return false;
    // (wait false: none when that buffer's last DMA is still on its way - the device's routing must not block here)
    if (!wait && cudaEventQuery(ec.aev[ec.ak]) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
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
    // (submitted now: WDDM holds a stream's work until something flushes it, and the device's routing has no sync
    //  before the next piece waits for this one)
    (void) cudaStreamQuery(pf.cs);
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
    const int E = g.n_lat, ff = g.n_ff_exp;
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
            j.xq = xq_l;
            j.xd = xd_l;
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
        std::fprintf(stderr, "maya-hy4: no helper card: --helper-gpu %d is not a second visible CUDA device (%d visible)\n",
                     o.helper_gpu, ndev);
        return;
    }
    hc.dev = o.helper_gpu;
    if (const char* v = std::getenv("HY4_HELPER_EVERY")) hc.every = std::max(1, std::atoi(v));
    if (const char* v = std::getenv("HY4_HELPER_SWAPS")) hc.per = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("HY4_HELPER_FILL")) hc.fill_cap = std::max(0, std::atoi(v));
    if (const char* v = std::getenv("HY4_HELPER_STEAL")) hc_steal = std::atoi(v) != 0;
    if (const char* v = std::getenv("HY4_HELPER_BG")) hc.bg = std::atoi(v) != 0;
    if (const char* v = std::getenv("HY4_HELPER_MARGIN")) hc.margin = (float) std::atof(v);
    if (const char* v = std::getenv("HY4_HELPER_BIAS")) hc.bias = (float) std::atof(v);
    if (hc.margin < 0.0f) hc.margin = ec.margin;
    const int E = g.n_lat, ff = g.n_ff_exp, nc = (int) ec.cls_bytes.size();
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
    // may still hold the main card's copies).  HY4_HELPER_STAGE: a buffer's MB, 64; 0: none - the driver's staging
    const char* hsv = std::getenv("HY4_HELPER_STAGE");
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
            std::fprintf(stderr, "maya-hy4: no pinned staging for the helper card's fills (the driver's is used)\n");
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
    if (const char* v = std::getenv("HY4_HELPER_RESERVE_MIB")) reserve = (size_t) std::max(0, std::atoi(v)) << 20;
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
    // Each one ends by itself after HY4_HELPER_KEEPALIVE_MS (50) and step() launches the next: a call that implicitly
    // waits for all of the card's work (an allocation, a plain memset) waits that long at most - one left running
    // forever hung the first token - and the card drops back to idle once the writing stops.
    // Off by default (2026-11-13: it costs the helper ~80-90 W while tokens are written, for 5-11% more speed);
    // HY4_HELPER_KEEPALIVE=1 turns it on.
    const char* kv = std::getenv("HY4_HELPER_KEEPALIVE");
    if (hc.slots && kv && std::atoi(kv) != 0) {
        int lo = 0, hi = 0;
        const char* km = std::getenv("HY4_HELPER_KEEPALIVE_MS");
        hc.ka_ns = (unsigned long long) std::max(1, km ? std::atoi(km) : 50) * 1000000ull;
        OnDev od(hc.dev);
        if (cudaDeviceGetStreamPriorityRange(&lo, &hi) == cudaSuccess &&
            cudaStreamCreateWithPriority(&hc.ks, cudaStreamNonBlocking, lo) == cudaSuccess) {
            std::fprintf(stderr, "maya-hy4: helper card CUDA%d: a keep-alive kernel holds its clocks up while tokens are "
                         "written (%llu ms each)\n", hc.dev, hc.ka_ns / 1000000ull);
        } else {
            cudaGetLastError();
            hc.ks = nullptr;
            std::fprintf(stderr, "maya-hy4: helper card CUDA%d: no keep-alive kernel\n", hc.dev);
        }
    }
}

// An expert's bytes (gate | up | down, as the main cache's slots) into a helper slot, on st (the helper current)
void Engine::Impl::hc_copy(int key, int slot, cudaStream_t st) {
    const int E = g.n_lat, ff = g.n_ff_exp, il = key / ec.ne, e = key % ec.ne;
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
    const int E = g.n_lat, ff = g.n_ff_exp;
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
            CK(cudaMemcpyAsync(hc.ptr[(size_t) sl], pin + d.second, expert_bytes(key / ec.ne), cudaMemcpyHostToDevice,
                               hc.s));
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
    // (HY4_HELPER_VERIFY=1, a check) every slot just filled read back against the mapping's bytes
    static const bool verify = std::getenv("HY4_HELPER_VERIFY") != nullptr;
    if (verify) {
        std::vector<uint8_t> back;
        std::vector<std::pair<const uint8_t*, size_t>> parts;
        for (const auto& kv : todo) {
            const size_t bytes = expert_bytes(kv.first / ec.ne);
            back.resize(bytes);
            CK(cudaMemcpy(back.data(), hc.ptr[(size_t) kv.second], bytes, cudaMemcpyDeviceToHost));
            parts.clear();
            ec_src_parts(kv.first, 0, bytes, parts);
            size_t at = 0;
            for (const auto& pr : parts) {
                if (std::memcmp(back.data() + at, pr.first, pr.second) != 0) {
                    std::fprintf(stderr,
                                 "maya-hy4: HY4_HELPER_VERIFY: expert %d of layer %d differs in helper slot %d\n",
                                 kv.first % ec.ne, kv.first / ec.ne, kv.second);
                    std::abort();
                }
                at += pr.second;
            }
        }
        std::fprintf(stderr,
                     "maya-hy4: HY4_HELPER_VERIFY: %zu experts in the helper's slots hold the mapping's bytes\n",
                     todo.size());
    }
}

// A token's experts on the helper card: f up from the host, its int8 rows, then gpu_experts' launches on the helper's
// stream and buffers; their sum -> hc.h_out (there once hc.s is synchronized).
void Engine::Impl::hc_experts(const Layer& l, const int* slot, const float* w, int n, const float* xin) {
    const int E = g.n_lat, ff = g.n_ff_exp;
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    auto run = [&](const iq::MvJob* j, int nj) {
        if (nj > 0 && !iq::mv(j, nj, hc.s)) {
            std::fprintf(stderr, "maya-hy4: no GEMV for these jobs (type %d, k %d)\n", j[0].type, j[0].k);
            std::abort();
        }
    };
    OnDev od(hc.dev);
    CK(cudaMemcpyAsync(hc.f, xin ? xin : h_f, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, hc.s));
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
            j.ncol = 1;
            j.xq = hc.qq;
            j.xd = hc.qd;
            j.y = hc.gu + (size_t) i * 2 * ff + (size_t) r * ff;
            j.ycs = (size_t) ff;
        }
    }
    run(jobs, 2 * n);
    k::swiglu_f32_rows(hc.gu, n, ff, l.clamp_exp, hc.h, hc.s);
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
        dj[i].alpha = w[i];
    }
    run(dj, n);
    k::sum_rows(hc.y, n, E, hc.routed, hc.s);
    CK(cudaMemcpyAsync(hc.h_out, hc.routed, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, hc.s));
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
// ------------------------------------------------------------------------------------------------ the prompt path
// A chunk of up to C tokens a layer at a time on the GPU (maya-ling's): the dense matrices decoded to F16 and
// multiplied by cuBLAS (F32 sums); the hc streams' reads and writes, the norms and the Engram rows row by row; a layer's
// attention in pieces of `sub` tokens - the sliding-window layers' keys gathered from their ring with the piece's own
// into one buffer, the sparse layers' over their whole cache with the indexer's choice as a mask (its pooled keys made
// for the groups the piece ends, each token's top 511 groups by an F32 GEMM of its queries against them); the scores'
// softmax with the window / choice, the causal mask and each head's sink; the router in F32; the routed experts in the
// latent width - each one some token of the chunk chose copied to a VRAM slot (the copy stream, `stage` slots ahead),
// decoded to F16 and multiplied for its tokens, its output added to theirs times their weights, expert by expert in id
// order - then routed_up, the shared expert on the full width, the sum post-normed into the streams.
bool Engine::Impl::pf_alloc(std::string& err) {
    const int Cn = pf.C = std::max(64, o.chunk);
    pf.Ca = std::max(16, std::min(o.sub, Cn));
    const int Ca = pf.Ca, E = g.n_embd, HC = g.hc, NL = g.n_lat, ne = g.n_expert, nu = g.n_used;
    const int ffe = g.n_ff_exp, ffd = g.n_ff, ffs = g.n_ff_sh, ql = g.q_lora, QK = g.qk_head, V = g.v_head;
    const int Hm = *std::max_element(g.n_head.begin(), g.n_head.end()), nope = QK - g.rope_dim;
    const int cd = g.eg_cols * g.eg_dim, D = g.idx_dim, NH = g.idx_heads, KP = std::max(1, g.idx_kpool);
    if (cublasCreate(&pf.cb) != CUBLAS_STATUS_SUCCESS) {
        err = "cuBLAS did not start";
        return false;
    }
    cublasSetStream(pf.cb, s);
    CK(cudaStreamCreateWithFlags(&pf.cs, cudaStreamNonBlocking));
    const size_t CE = (size_t) Cn * E;
    size_t mib = 768;
    if (const char* v = std::getenv("HY4_SCORES_MIB")) mib = (size_t) std::max(64, std::atoi(v));
    pf.s_elems = (mib << 20) / 6;   // S F32 and P F16
    pf.w1_elems = std::max({(size_t) 2 * ffd * E, (size_t) 2 * ffe * NL, (size_t) 2 * ffs * E, (size_t) Hm * V * E,
                            (size_t) Hm * QK * ql, (size_t) Hm * 512 * nope, (size_t) Hm * V * 512,
                            (size_t) (HC + 1) * E * std::max(cd, 1), (size_t) NL * E, (size_t) NH * D * ql});
    pf.w2_elems = std::max({(size_t) ffd * E, (size_t) ffe * NL, (size_t) ffs * E});
    for (const Layer& l : L)
        if (!l.dense) pf.slot_bytes = std::max(pf.slot_bytes, expert_bytes((int) (&l - L.data())));
    pf.GE = std::max(1, std::min(ne, o.stage));
    pf.nwords = (max_pools + 31) / 32;
    pf.top_pools = g.idx_kpool ? g.idx_top_k / KP - g.idx_tail : 0;
    // the indexer's scoring tile: tokens whose [heads][pools] scores take at most HY4_IDX_MIB (256 MB)
    size_t imib = 256;
    if (const char* v = std::getenv("HY4_IDX_MIB")) imib = (size_t) std::max(16, std::atoi(v));
    pf.tqi = max_pools ? (int) std::max<size_t>(1, std::min<size_t>(Ca, (imib << 20) / ((size_t) NH * max_pools * 4)))
                       : 1;
    if (const char* v = std::getenv("HY4_PF_PREFETCH")) pre.on = std::atoi(v) != 0;
    if (const char* v = std::getenv("HY4_PF_PREFETCH_MIN")) pre.min_T = std::max(1, std::atoi(v));
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
            std::fprintf(stderr, "maya-hy4: no pinned staging for the prompt path's prefetch (off)\n");
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
        carve(pf.R, (size_t) Cn * HC * E);
        carve(pf.X, CE);
        carve(pf.A, CE);
        carve(pf.Y, CE);
        carve(pf.Y2, CE);
        carve(pf.Ah, CE);
        carve(pf.MIX, (size_t) Cn * 2 * HC);
        carve(pf.POST, (size_t) Cn * HC);
        carve(pf.RL, (size_t) Cn * ne);
        carve(pf.sel, (size_t) Cn * nu);
        carve(pf.wts, (size_t) Cn * nu);
        carve(pf.cnt, (size_t) ne);
        carve(pf.off, (size_t) ne);
        carve(pf.list, (size_t) Cn * nu);
        carve(pf.wl, (size_t) Cn * nu);
        carve(pf.LAT, (size_t) Cn * NL);
        carve(pf.LATh, (size_t) Cn * NL);
        carve(pf.YL, (size_t) Cn * NL);
        carve(pf.YLh, (size_t) Cn * NL);
        carve(pf.Xe, (size_t) Cn * NL);
        carve(pf.Ye, (size_t) Cn * std::max(E, NL));
        carve(pf.GU, (size_t) Cn * gu_cols);
        carve(pf.He, (size_t) Cn * std::max(ffe, ffs));
        // Engram (a piece of `sub` rows at a time)
        carve(pf.EGR, (size_t) Ca * std::max(cd, 1));
        carve(pf.EGY, (size_t) Ca * std::max(cd, 1));
        carve(pf.EGYh, (size_t) Ca * std::max(cd, 1));
        carve(pf.KVO, (size_t) Ca * (HC + 1) * E);
        // the attention's piece
        carve(pf.QA, (size_t) Ca * ql);
        carve(pf.QAh, (size_t) Ca * ql);
        carve(pf.KVA, (size_t) Ca * 576);
        carve(pf.GATE, (size_t) Ca * Hm * V);
        carve(pf.Q, (size_t) Ca * Hm * QK);
        carve(pf.QH, (size_t) Ca * Hm * QK);
        carve(pf.Qcat, (size_t) Ca * Hm * 576);
        carve(pf.O16, (size_t) Ca * Hm * 512);
        carve(pf.OB, (size_t) Ca * Hm * V);
        carve(pf.OBh, (size_t) Ca * Hm * V);
        carve(pf.Kbuf, (size_t) (Ca + std::max(g.swa_window, 1)) * 576);
        if (max_pools) {   // the indexer's
            carve(pf.IK, (size_t) Ca * D);
            carve(pf.IKN, (size_t) Ca * D);
            carve(pf.IKNh, (size_t) Ca * D);
            carve(pf.IV, (size_t) Ca * D);
            carve(pf.IG, (size_t) Ca * D);
            carve(pf.IW, (size_t) Ca * NH);
            carve(pf.IQ, (size_t) Ca * NH * D);
            carve(pf.bits, (size_t) Ca * pf.nwords);
            carve(pf.sel_on, (size_t) Ca);
            carve(pf.IA, (size_t) pf.tqi * NH * max_pools);
            carve(pf.IS, (size_t) pf.tqi * max_pools);
            carve(pf.icells, (size_t) pf.tqi * std::max(1, pf.top_pools) * KP);
        }
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
    if (cd) pf.h_egr = halloc<float>((size_t) Ca * cd);
    const char* pg = std::getenv("HY4_PF_PAGEABLE");
    if (!pg || std::atoi(pg) == 0) {
        stg.cap = (size_t) 64 << 20;
        stg.on = cudaHostAlloc((void**) &stg.pin[0], stg.cap, cudaHostAllocPortable) == cudaSuccess &&
                 cudaHostAlloc((void**) &stg.pin[1], stg.cap, cudaHostAllocPortable) == cudaSuccess;
        if (stg.on) {
            CK(cudaEventCreateWithFlags(&stg.ev[0], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stg.ev[1], cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "maya-hy4: no pinned staging for the prompt path's copies (the driver's is used)\n");
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
    const int ne = g.n_expert, E = g.n_lat, ff = g.n_ff_exp;   // (the experts' width: the latent)
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
        std::fprintf(stderr, "maya-hy4: no F16 rows for type %d\n", m.type);
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
        std::fprintf(stderr, "maya-hy4: cuBLAS GEMM %dx%dx%d failed (%d)\n", T, n, k, (int) r);
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
            std::fprintf(stderr, "maya-hy4: cuBLAS F32 GEMM failed\n");
            std::abort();
        }
        return;
    }
    if (is_float(m.type) && m.type != 30) {
        std::fprintf(stderr, "maya-hy4: a F16 matrix in the prompt path (not taken)\n");
        std::abort();
    }
    f16w(m, pf.W1);
    gemm(xh, m.k, pf.W1, m.k, y, m.n, T, m.n, m.k);
}

namespace {
void cublas_ok(cublasStatus_t r, const char* what) {
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "maya-hy4: cuBLAS %s failed (%d)\n", what, (int) r);
        std::abort();
    }
}
}  // namespace

// The indexer for a piece of T tokens at p0 (a sparse layer): its keys' value and gate rows, the pooled keys of the
// groups the piece ends, the open group's rows into the ring; then each token's choice (pf.bits / pf.sel_on)
void Engine::Impl::pf_index(Layer& l, const float* A_, const __half* Ah_, int T, int p0) {
    const int D = g.idx_dim, NH = g.idx_heads, KP = g.idx_kpool;
    lin(l.idx_k, Ah_, A_, T, pf.IK);
    lin(l.idx_proj, Ah_, A_, T, pf.IW);
    k::rmsnorm(pf.IK, l.idx_knorm, pf.IKN, T, D, g.eps, s);
    k::to_f16(pf.IKN, pf.IKNh, (size_t) T * D, s);
    lin(l.idx_v, pf.IKNh, pf.IKN, T, pf.IV);
    lin(l.idx_g, pf.IKNh, pf.IKN, T, pf.IG);
    const int g0 = p0 / KP, g1 = (p0 + T) / KP;   // (the groups that end in the piece: g0 .. g1)
    k::idx_pool_rows(pf.IV, pf.IG, l.idx_ring, l.idx_ape, l.idx_pnorm, p0, g0, g1 - g0, KP, D, g.eps, l.theta_scale,
                     l.idx_pooled, s);
    const int open0 = std::max(p0, g1 * KP);      // (the open group's members in the piece)
    k::idx_ring_put(pf.IV, pf.IG, p0, open0 - p0, p0 + T - open0, KP, D, l.idx_ring, s);
    lin(l.idx_qb, pf.QAh, nullptr, T, pf.IQ);
    k::mla_rope_q(pf.IQ, T, NH, D, D - g.rope_dim, p0, nullptr, l.theta_scale, s);
    const int npool = (p0 + T - 1) / KP;          // (the most groups a token of the piece can choose from)
    if (npool <= pf.top_pools) {                  // (every token sees every key)
        CK(cudaMemsetAsync(pf.sel_on, 0, (size_t) T * sizeof(int), s));
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    for (int t0 = 0; t0 < T; t0 += pf.tqi) {
        const int tq = std::min(pf.tqi, T - t0);
        cublas_ok(cublasSgemm(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, npool, tq * NH, D, &one, l.idx_pooled, D,
                              pf.IQ + (size_t) t0 * NH * D, D, &zero, pf.IA, npool),
                  "indexer scores");
        k::idx_score_rows(pf.IA, pf.IW + (size_t) t0 * NH, NH, npool, tq, p0 + t0, KP,
                          1.0f / std::sqrt((float) (D * NH)), pf.IS, s);
        k::idx_select_rows(pf.IS, npool, tq, p0 + t0, KP, pf.top_pools, pf.icells, pf.bits + (size_t) t0 * pf.nwords,
                           pf.nwords, pf.sel_on + t0, s);
    }
}

// A layer's attention for a piece of T tokens at p0 (A_ / Ah_ its normed rows): Y_ [T][E]
void Engine::Impl::pf_attn(Layer& l, const float* A_, const __half* Ah_, int T, int p0, float* Y_) {
    const int H = l.H, QK = g.qk_head, V = g.v_head, ql = g.q_lora, nope = QK - g.rope_dim, win = g.swa_window;
    const float one = 1.0f, zero = 0.0f;
    lin(l.q_a, Ah_, A_, T, pf.QA);
    lin(l.kv_a, Ah_, A_, T, pf.KVA);
    lin(l.wgate, Ah_, A_, T, pf.GATE);
    k::rmsnorm(pf.QA, l.q_a_norm, pf.QA, T, ql, g.eps, s);
    k::to_f16(pf.QA, pf.QAh, (size_t) T * ql, s);
    lin(l.q_b, pf.QAh, nullptr, T, pf.Q);
    k::to_f16(pf.Q, pf.QH, (size_t) T * H * QK, s);
    // q's nope part through k_b a head: Qcat[t][h][0:512] (F16), its rope part after it
    f16w(l.k_b, pf.W1);   // [H][512][nope]
    cublas_ok(cublasGemmStridedBatchedEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, 512, T, nope, &one, pf.W1, CUDA_R_16F, nope,
                                         (long long) 512 * nope, pf.QH, CUDA_R_16F, H * QK, QK, &zero, pf.Qcat,
                                         CUDA_R_16F, H * 576, 576, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
              "k_b batch");
    k::qpe_f16(pf.Q, pf.Qcat, T, H, QK, nope, p0, l.theta_scale, s);
    if (l.idx) pf_index(l, A_, Ah_, T, p0);
    // the keys: a sliding-window layer's from its ring (the window before the piece) and the piece's own, in one
    // buffer from position kb0; a sparse layer's its cache
    const __half* K = l.kv;
    int kb0 = 0;
    if (l.ring) {
        kb0 = std::max(0, p0 - (win - 1));
        k::ring_get(l.kv, l.ring, kb0, p0 - kb0, pf.Kbuf, s);
        k::mla_kv_store(pf.KVA, l.kv_a_norm, g.eps, pf.Kbuf, T, p0, nullptr, l.theta_scale, s, 0, kb0);
        const int nput = std::min(T, l.ring);
        k::ring_put(pf.Kbuf + (size_t) (p0 + T - nput - kb0) * 576, l.ring, p0 + T - nput, nput, l.kv, s);
        K = pf.Kbuf;
    } else {
        k::mla_kv_store(pf.KVA, l.kv_a_norm, g.eps, l.kv, T, p0, nullptr, l.theta_scale, s);
    }
    // the attention: tiles of query tokens, each [tq x H rows] x [the keys it can see]
    auto r8 = [](int v) { return (v + 7) & ~7; };
    auto first_key = [&](int ta) { return l.ring ? std::max(kb0, p0 + ta - (win - 1)) : 0; };
    int ta = 0;
    while (ta < T) {
        int tq = T - ta;
        while (tq > 1 && (size_t) H * tq * (size_t) r8(p0 + ta + tq - first_key(ta)) > pf.s_elems) tq = (tq + 1) / 2;
        const int tb = ta + tq, k0 = first_key(ta), nk = p0 + tb - k0, ld = r8(nk), M = H * tq;
        cublas_ok(cublasGemmEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, M, 576, &mla_scale, K + (size_t) (k0 - kb0) * 576,
                               CUDA_R_16F, 576, pf.Qcat + (size_t) ta * H * 576, CUDA_R_16F, 576, &zero, pf.S,
                               CUDA_R_32F, ld, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                  "scores");
        k::attn_softmax(pf.S, ld, pf.P, ld, M, nk, H, p0 + ta, k0, l.ring ? win : 0,
                        l.idx ? pf.bits + (size_t) ta * pf.nwords : nullptr, pf.nwords, l.idx ? pf.sel_on + ta : nullptr,
                        std::max(1, g.idx_kpool), l.sinks, s);
        cublas_ok(cublasGemmEx(pf.cb, CUBLAS_OP_N, CUBLAS_OP_N, 512, M, nk, &one, K + (size_t) (k0 - kb0) * 576,
                               CUDA_R_16F, 576, pf.P, CUDA_R_16F, ld, &zero, pf.O16 + (size_t) ta * H * 512, CUDA_R_16F,
                               512, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                  "attention output");
        ta = tb;
    }
    // each head's 512 through v_b, times the gate (a value), then W_o
    f16w(l.v_b, pf.W1);   // [H][V][512]
    cublas_ok(cublasGemmStridedBatchedEx(pf.cb, CUBLAS_OP_T, CUBLAS_OP_N, V, T, 512, &one, pf.W1, CUDA_R_16F, 512,
                                         (long long) V * 512, pf.O16, CUDA_R_16F, H * 512, 512, &zero, pf.OB, CUDA_R_32F,
                                         H * V, V, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
              "v_b batch");
    k::head_gate_rows(pf.OB, pf.GATE, T, H * V, 1, pf.OBh, s);
    lin(l.wo, pf.OBh, nullptr, T, Y_);
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
        k::swiglu_rows(pf.GU, n, ff, 0.0f, pf.He, s);
        gemm(pf.He, ff, pf.W2, ff, pf.Y + (size_t) r0 * E, E, n, E, ff);
    }
}

// A MoE layer for the chunk: Y = routed_up(the routed experts' sum, in the latent width), Y2 = the shared expert's
void Engine::Impl::pf_moe(const Layer& l, int T) {
    const int E = g.n_embd, NL = g.n_lat, ne = g.n_expert, nu = g.n_used, ff = g.n_ff_exp, ffs = g.n_ff_sh;
    // the router in F32 (llama.cpp's F32 weights), the routing on the GPU; the latent rows
    lin(l.gate_inp, pf.Ah, pf.A, T, pf.RL);
    lin(l.lat_down, pf.Ah, pf.A, T, pf.LAT);
    k::to_f16(pf.LAT, pf.LATh, (size_t) T * NL, s);
    k::route(pf.RL, ne, l.exp_bias_d, T, ne, 1, 1, nu, g.exp_scale, pf.sel, pf.wts, s);
    CK(cudaMemsetAsync(pf.cnt, 0, (size_t) ne * sizeof(int), s));
    k::expert_count(pf.sel, T * nu, pf.cnt, s);
    CK(cudaMemcpyAsync(pf.h_cnt, pf.cnt, (size_t) ne * sizeof(int), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    pf.h_off[0] = 0;
    for (int e = 0; e < ne; ++e) pf.h_off[e + 1] = pf.h_off[e] + pf.h_cnt[e];
    CK(cudaMemcpyAsync(pf.off, pf.h_off, (size_t) ne * sizeof(int), cudaMemcpyHostToDevice, s));
    k::expert_lists(pf.sel, pf.wts, T, nu, ne, pf.off, pf.list, pf.wl, s);
    CK(cudaMemsetAsync(pf.YL, 0, (size_t) T * NL * sizeof(float), s));
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru, db = (size_t) NL * c.rd;
    const int il = (int) (&l - L.data()), GE = pf.GE;
    pf_score(il, pf.h_cnt, T);
    if (route_log) {
        std::vector<uint16_t> c16((size_t) ne);
        for (int e = 0; e < ne; ++e) c16[(size_t) e] = (uint16_t) std::min(pf.h_cnt[e], 65535);
        const uint8_t hdr[2] = {'P', (uint8_t) il};
        std::fwrite(hdr, 1, 2, route_log);
        std::fwrite(c16.data(), 2, c16.size(), route_log);
    }
    // The prefetch's way (a copier running for this chunk): every expert not in VRAM already on its way into the
    // layer's buffer; once it is there, each used expert's GEMMs as below, in id order
    auto experts = [&](int e, const uint8_t* wg, const uint8_t* wu, const uint8_t* wd) {
        const int rows = pf.h_cnt[e], o0 = pf.h_off[e];
        if (!iq::f16_rows(c.tg, wg, NL, ff, (uint16_t*) pf.W1, NL, s) ||
            !iq::f16_rows(c.tu, wu, NL, ff, (uint16_t*) (pf.W1 + (size_t) ff * NL), NL, s) ||
            !iq::f16_rows(c.td, wd, ff, NL, (uint16_t*) pf.W2, ff, s)) {
            std::fprintf(stderr, "maya-hy4: no F16 rows for the experts' types %d / %d / %d\n", c.tg, c.tu, c.td);
            std::abort();
        }
        iq::gather_f16((const uint16_t*) pf.LATh, NL, pf.list + o0, rows, NL, (uint16_t*) pf.Xe, s);
        gemm(pf.Xe, NL, pf.W1, NL, pf.GU, 2 * ff, rows, 2 * ff, NL);
        k::swiglu_rows(pf.GU, rows, ff, l.clamp_exp, pf.He, s);
        gemm(pf.He, ff, pf.W2, ff, pf.Ye, NL, rows, NL, ff);
        k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, NL, pf.YL, s);
    };
    if (pre.active) {
        const int b = (il - g.n_dense) & 1;
        {
            std::unique_lock<std::mutex> lk(pre.mu);
            pre.cv.wait(lk, [&] { return pre.staged >= il; });
        }
        CK(cudaStreamWaitEvent(s, pre.ready[b], 0));
        const uint8_t* B = pre.buf[b];
        for (int e = 0; e < ne; ++e) {
            if (pf.h_cnt[e] == 0) continue;
            const uint8_t* blob = ec_blob(il, e);
            experts(e, blob ? blob : B + (size_t) e * gb, blob ? blob + gb : B + (size_t) ne * gb + (size_t) e * ub,
                    blob ? blob + gb + ub : B + (size_t) ne * (gb + ub) + (size_t) e * db);
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
                stage_copy(B + at * gb, c.gate + (size_t) e * gb, n * gb);
                stage_copy(B + GE * gb + at * ub, c.up + (size_t) e * ub, n * ub);
                stage_copy(B + GE * (gb + ub) + at * db, c.down + (size_t) e * db, n * db);
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
            if (!iq::f16_rows(c.tg, wg, NL, ff, (uint16_t*) pf.W1, NL, s) ||
                !iq::f16_rows(c.tu, wu, NL, ff, (uint16_t*) (pf.W1 + (size_t) ff * NL), NL, s) ||
                !iq::f16_rows(c.td, wd, ff, NL, (uint16_t*) pf.W2, ff, s)) {
                std::fprintf(stderr, "maya-hy4: no F16 rows for the experts' types %d / %d / %d\n", c.tg, c.tu, c.td);
                std::abort();
            }
            iq::gather_f16((const uint16_t*) pf.LATh, NL, pf.list + o0, rows, NL, (uint16_t*) pf.Xe, s);
            gemm(pf.Xe, NL, pf.W1, NL, pf.GU, 2 * ff, rows, 2 * ff, NL);
            k::swiglu_rows(pf.GU, rows, ff, l.clamp_exp, pf.He, s);
            gemm(pf.He, ff, pf.W2, ff, pf.Ye, NL, rows, NL, ff);
            k::scatter_add(pf.Ye, pf.list + o0, pf.wl + o0, rows, NL, pf.YL, s);
        }
        if (bi >= 0) CK(cudaEventRecord(pf.gfreed[bi], s));
    }
    // routed_up, and the shared expert on every token (no clamp)
    k::to_f16(pf.YL, pf.YLh, (size_t) T * NL, s);
    lin(l.lat_up, pf.YLh, pf.YL, T, pf.Y);
    f16w(l.sh_gate, pf.W1);
    f16w(l.sh_up, pf.W1 + (size_t) ffs * E);
    f16w(l.sh_down, pf.W2);
    gemm(pf.Ah, E, pf.W1, E, pf.GU, 2 * ffs, T, 2 * ffs, E);
    k::swiglu_rows(pf.GU, T, ffs, 0.0f, pf.He, s);
    gemm(pf.He, ffs, pf.W2, ffs, pf.Y2, E, T, E, ffs);
}

void Engine::Impl::chunk(const int* ids, int T, bool want_logits) {
    const int E = g.n_embd, HC = g.hc, Ca = pf.Ca;
    const int pos0 = pos;
    ec_kv_reserve(pos0 + T);
    if ((int) hist.size() < pos0 + T) hist.resize((size_t) pos0 + T);
    for (int t = 0; t < T; ++t) hist[(size_t) pos0 + t] = ids[t];
    const uint8_t* ew = embd.w;
    if (embd_host) {
        const size_t rb = embd.rb;
        if (stg.on && stg.cap >= rb) {
            const int per = (int) std::min<size_t>((size_t) T, stg.cap / rb);
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
        std::fprintf(stderr, "maya-hy4: no embedding rows for type %d\n", embd.type);
        std::abort();
    }
    k::hc_init_rows(pf.X, HC, E, pf.R, T, s);
    static const bool times = std::getenv("HY4_PF_TIMES") != nullptr;
    if (times) CK(cudaStreamSynchronize(s));
    if (pre.on && T >= pre.min_T) pre_start();   // (the copier: every MoE layer's experts, while the GPU computes)
    double tl0 = now_ms();
    const int cd = g.eg_cols * g.eg_dim;
    for (int il = 0; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        // Engram: the rows of each piece hashed on the host (the history holds the chunk's tokens), copied up, normed,
        // projected and written into the streams
        if (l.eg >= 0) {
            for (int s0 = 0; s0 < T; s0 += Ca) {
                const int Ts = std::min(Ca, T - s0);
                CK(cudaStreamSynchronize(s));   // (h_egr's last piece copied)
                pool->run((Ts + 63) / 64, [&](int j) {
                    for (int t = j * 64; t < std::min(Ts, j * 64 + 64); ++t)
                        eg_row(l.eg, ids[s0 + t], pos0 + s0 + t, pf.h_egr + (size_t) t * cd);
                });
                CK(cudaMemcpyAsync(pf.EGR, pf.h_egr, (size_t) Ts * cd * sizeof(float), cudaMemcpyHostToDevice, s));
                k::eg_rows(pf.EGR, l.eg_lnorm, Ts * g.eg_cols, g.eg_dim, kEgEps, pf.EGY, nullptr, nullptr, s, g.eg_cols);
                k::to_f16(pf.EGY, pf.EGYh, (size_t) Ts * cd, s);
                lin(l.eg_kv, pf.EGYh, pf.EGY, Ts, pf.KVO);
                k::eg_update_rows(pf.KVO, l.eg_kv_b, l.eg_gate, l.eg_hs, l.eg_gamma, HC, E, kEgEps,
                                  pf.R + (size_t) s0 * HC * E, Ts, s);
            }
        }
        // the attention, `sub` tokens at a time
        k::hc_read_rows(pf.R, l.hca_fn, 2 * HC, l.hca_scale, l.hca_base, HC, E, g.eps, g.hc_eps, g.hc_mag, pf.MIX, pf.X,
                        pf.POST, T, s);
        k::rmsnorm(pf.X, l.attn_norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        for (int s0 = 0; s0 < T; s0 += Ca) {
            const int Ts = std::min(Ca, T - s0);
            pf_attn(l, pf.A + (size_t) s0 * E, pf.Ah + (size_t) s0 * E, Ts, pos0 + s0, pf.Y + (size_t) s0 * E);
        }
        k::norm_write_rows(pf.Y, nullptr, l.post_attn, g.eps, pf.POST, HC, E, pf.R, T, s);
        dump("attn", il, pf.R + (size_t) (T - 1) * HC * E, pos0 + T - 1);
        if (times) {
            CK(cudaStreamSynchronize(s));
            const double t1 = now_ms();
            pf.attn_ms += t1 - tl0;
            tl0 = t1;
        }
        // the FFN
        k::hc_read_rows(pf.R, l.hcf_fn, 2 * HC, l.hcf_scale, l.hcf_base, HC, E, g.eps, g.hc_eps, g.hc_mag, pf.MIX, pf.X,
                        pf.POST, T, s);
        k::rmsnorm(pf.X, l.ffn_norm, pf.A, T, E, g.eps, s);
        k::to_f16(pf.A, pf.Ah, (size_t) T * E, s);
        if (l.dense) {
            pf_dense(l, T);
            k::norm_write_rows(pf.Y, nullptr, l.post_ffn, g.eps, pf.POST, HC, E, pf.R, T, s);
        } else {
            pf_moe(l, T);
            k::norm_write_rows(pf.Y, pf.Y2, l.post_ffn, g.eps, pf.POST, HC, E, pf.R, T, s);
        }
        dump("ffn ", il, pf.R + (size_t) (T - 1) * HC * E, pos0 + T - 1);
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
                     "maya-hy4: prompt path so far: attention %.0f ms, FFN %.0f ms (blocked in copies %.0f ms; the "
                     "prefetch's copier %.0f ms); working set %.1f GB\n",
                     pf.attn_ms, pf.moe_ms, pf.copy_ms, pre.ms, working_set_gb());
    pos = pos0 + T;
    if (want_logits) {
        // the head on the last token's streams
        k::hc_read(pf.R + (size_t) (T - 1) * HC * E, head_fn, HC, head_scale, head_base, HC, E, g.eps, g.hc_eps,
                   g.hc_mag, hmix, x, nullptr, s);
        k::norm_q8(x, out_norm, g.eps, a, xq_a, xd_a, E, s);
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
    // a short prompt (a turn's few new tokens): the token path's experts on the CPU beat copying the ones it chooses
    // (HY4_MIN_CHUNK)
    int min_chunk = 256;
    if (const char* v = std::getenv("HY4_MIN_CHUNK")) min_chunk = std::max(1, std::atoi(v));
    if (n < min_chunk) {
        for (int i = 0; i < n && (i == 0 || lg); ++i) lg = step(ids[i]);   // (nullptr: failed())
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

// Engram's rows for the token at pos (llama.cpp#32517's engram_hash): each module's columns hash the n-gram ending at
// it - the token and the eg_ngram - 1 before it (hist), each through the token map, a position before 0 the pad's - as
// rolling = t0 m0 ^ t1 m1 ^ ... (uint64, wrapping) up to order o, row = rolling % prime + offset; the rows' F8_E4M3
// values to h_eg [module][col][dim] as F32.
void Engine::Impl::eg_row(int e, int token, int p, float* dst) const {
    const int ng = g.eg_ngram, nh = g.eg_heads, nc = g.eg_cols, dim = g.eg_dim;
    const int64_t pad = eg_map[(size_t) g.eg_pad];
    auto cid = [&](int t) -> int64_t { return t < 0 || t >= (int) eg_map.size() ? pad : eg_map[(size_t) t]; };
    uint64_t toks[8];
    toks[0] = (uint64_t) cid(token);
    for (int s1 = 1; s1 < ng; ++s1) toks[s1] = (uint64_t) (p - s1 < 0 ? pad : cid(hist[(size_t) (p - s1)]));
    const uint64_t* mult = eg_mult.data() + (size_t) e * ng;
    const uint64_t* prime = eg_prime.data() + (size_t) e * nc;
    const uint64_t* off = eg_off.data() + (size_t) e * nc;
    uint64_t rolling = toks[0] * mult[0];
    for (int o1 = 1; o1 < ng; ++o1) {
        rolling ^= toks[o1] * mult[o1];
        for (int h = 0; h < nh; ++h) {
            const int col = (o1 - 1) * nh + h;
            const uint64_t row = rolling % prime[col] + off[col];
            const uint8_t* src = eg_table + row * eg_rb;
            float* d = dst + (size_t) col * dim;
            for (int i = 0; i < dim; ++i) d[i] = fp8_lut[src[i]];
        }
    }
}
void Engine::Impl::eg_fill(int token) {
    if (!eg_table) return;
    for (size_t e = 0; e < g.eg_layers.size(); ++e)
        eg_row((int) e, token, pos, h_eg + e * g.eg_cols * g.eg_dim);
}

// A layer's attention for the token at *dpos (Engram first where the layer has a module): the hc streams read, the
// gated MLA (a sliding window of g.swa_window keys on the sliding-window layers, every key so far on the sparse ones;
// each head's sink in the softmax), its post-norm written back to the streams
void Engine::Impl::tok_attn(Layer& l) {
    const int E = g.n_embd, H = l.H, HC = g.hc, nope = g.qk_head - g.rope_dim;
    if (l.eg >= 0) {
        const int cd = g.eg_cols * g.eg_dim;
        k::eg_rows(eg_rows_d + (size_t) l.eg * cd, l.eg_lnorm, g.eg_cols, g.eg_dim, kEgEps, eg_y, xq_e, xd_e, s);
        mv(l.eg_kv, eg_y, xq_e, xd_e, eg_kvo);
        k::eg_update(eg_kvo, l.eg_kv_b, l.eg_gate, l.eg_hs, l.eg_gamma, HC, E, kEgEps, R, s);
    }
    k::hc_read(R, l.hca_fn, 2 * HC, l.hca_scale, l.hca_base, HC, E, g.eps, g.hc_eps, g.hc_mag, hmix, x, hpost, s);
    k::norm_q8(x, l.attn_norm, g.eps, a, xq_a, xd_a, E, s);
    {
        const DMat* ms[5] = {&l.q_a, &l.kv_a, &l.wgate, &l.idx_k, &l.idx_proj};
        float* ys[5] = {qa, kva, gh, ik, iw};
        mv_many(ms, ys, l.idx ? 5 : 3, a, xq_a, xd_a);
    }
    k::norm_q8(qa, l.q_a_norm, g.eps, nullptr, xq_m, xd_m, g.q_lora, s);
    {
        const DMat* ms[2] = {&l.q_b, &l.idx_qb};
        float* ys[2] = {qf, iq};
        mv_many(ms, ys, l.idx ? 2 : 1, nullptr, xq_m, xd_m);
    }
    if (l.idx) {
        // the indexer: the key normed, its value and gate rows into the open group (pooled when the group ends), the
        // query roped at the token's position, the complete groups scored and the cells chosen
        const int D = g.idx_dim, NH = g.idx_heads, KP = g.idx_kpool;
        k::eg_rows(ik, l.idx_knorm, 1, D, g.eps, ikn, xq_i, xd_i, s);
        {
            const DMat* ms[2] = {&l.idx_v, &l.idx_g};
            float* ys[2] = {iv, ig};
            mv_many(ms, ys, 2, ikn, xq_i, xd_i);
        }
        k::idx_store(iv, ig, dpos, KP, D, l.idx_ring, s);
        k::idx_pool(l.idx_ring, l.idx_ape, l.idx_pnorm, dpos, KP, D, g.eps, l.theta_scale, l.idx_pooled, s);
        k::mla_rope_q(iq, 1, NH, D, D - g.rope_dim, 0, dpos, l.theta_scale, s);
        k::idx_score(iq, iw, l.idx_pooled, dpos, KP, NH, D, 1.0f / std::sqrt((float) (D * NH)), max_pools, iscore, s);
        k::idx_select(iscore, dpos, KP, g.idx_top_k / KP - g.idx_tail, cells, ncell, s);
    }
    k::q8(qf, 1, H * g.qk_head, xq_m, xd_m, s);   // (q_nope's int8 rows: head h's at qk_head h, 32-value blocks)
    {
        // k_b a head: q_nope -> the cache's 512
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
    k::mla_rope_q(qf, 1, H, g.qk_head, nope, 0, dpos, l.theta_scale, s);
    k::mla_kv_store(kva, l.kv_a_norm, g.eps, l.kv, 1, 0, dpos, l.theta_scale, s, l.ring);
    // (HY4_IDX_DENSE=1: the sparse layers attend to every key, the indexer's choice ignored - a check of what it does)
    static const bool dense = std::getenv("HY4_IDX_DENSE") && std::atoi(std::getenv("HY4_IDX_DENSE")) != 0;
    const bool sel = l.idx && !dense;
    k::mla_decode(qabs, qf, g.qk_head, nope, l.kv, dpos, l.swa ? g.swa_window : 0, sel ? cells : nullptr,
                  sel ? ncell : nullptr, l.sinks, H, mla_scale, attn_nsp, attn_scr, mo, xq_m, xd_m, s, l.ring);
    {
        // v_b a head: the 512 -> v_head
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
    k::head_gate(mob, gh, H * g.v_head, 1, mob, xq_a, xd_a, s);   // (the gate a value: sigmoid(W_gate a) elementwise)
    mv(l.wo, nullptr, xq_a, xd_a, ao);
    k::norm_write(ao, nullptr, l.post_attn, g.eps, hpost, HC, E, R, s);
}

// The dense layer's FFN (no clamp), its post-norm written back to the streams
void Engine::Impl::tok_dense_ffn(const Layer& l) {
    const int E = g.n_embd, HC = g.hc;
    k::hc_read(R, l.hcf_fn, 2 * HC, l.hcf_scale, l.hcf_base, HC, E, g.eps, g.hc_eps, g.hc_mag, hmix, x, hpost, s);
    k::norm_q8(x, l.ffn_norm, g.eps, f, xq_f, xd_f, E, s);
    {
        const DMat* ms[2] = {&l.gate, &l.up};
        float* ys[2] = {fg, fu};
        mv_many(ms, ys, 2, f, xq_f, xd_f);
    }
    k::swiglu(fg, fu, fh, g.n_ff, 0.0f, s);
    k::q8(fh, 1, g.n_ff, xq_h, xd_h, s);
    mv(l.down, fh, xq_h, xd_h, fo);
    k::norm_write(fo, nullptr, l.post_ffn, g.eps, hpost, HC, E, R, s);
}

// A MoE layer's router: f = rmsnorm(the streams' read) and its int8 rows (the shared expert's and the latent
// projection's input), the logits (F32 router), the latent l = routed_down f and its int8 rows (the VRAM experts'
// input); l and the logits to the host.
void Engine::Impl::tok_router(const Layer& l) {
    const int E = g.n_embd, HC = g.hc;
    k::hc_read(R, l.hcf_fn, 2 * HC, l.hcf_scale, l.hcf_base, HC, E, g.eps, g.hc_eps, g.hc_mag, hmix, x, hpost, s);
    k::norm_q8(x, l.ffn_norm, g.eps, f, xq_f, xd_f, E, s);
    mv(l.gate_inp, f, xq_f, xd_f, rlog);
    mv(l.lat_down, f, xq_f, xd_f, lat);
    k::q8(lat, 1, g.n_lat, xq_l, xd_l, s);
    if (dr) return;   // (the device's routing: tok_moe next)
    CK(cudaMemcpyAsync(h_f, lat, (size_t) g.n_lat * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(h_rlog, rlog, (size_t) g.n_expert * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// A MoE layer's end: the CPU lane's latent sum and which lanes ran (h_tail, written by the host) up to the device; the
// latent sum (+ the VRAM experts') through routed_up, plus the shared expert's, post-normed, written back to the streams
void Engine::Impl::tok_tail(const Layer& l) {
    if (!dr) CK(cudaMemcpyAsync(d_tail, h_tail, tail_bytes(), cudaMemcpyHostToDevice, s));   // (else moe_wait's)
    k::lat_sum_q8(d_tail + 4, ec.routed, (const int*) d_tail, g.n_lat, lsum, xq_l, xd_l, s);
    mv(l.lat_up, lsum, xq_l, xd_l, fo);
    k::norm_write(fo, so, l.post_ffn, g.eps, hpost, g.hc, g.n_embd, R, s);
}

// The output head: the streams' read (pre only), the norm, the logits to the host
void Engine::Impl::tok_head() {
    const int E = g.n_embd, HC = g.hc;
    k::hc_read(R, head_fn, HC, head_scale, head_base, HC, E, g.eps, g.hc_eps, g.hc_mag, hmix, x, nullptr, s);
    k::norm_q8(x, out_norm, g.eps, a, xq_a, xd_a, E, s);
    mv(output, a, xq_a, xd_a, logits);
    CK(cudaMemcpyAsync(h_logits, logits, (size_t) g.vocab * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// The token path's segments between its host syncs: 0 the embedding (d_erow) into the streams, the Engram rows up, the
// dense layers, the first MoE layer's attention and router; il (a MoE layer after the first) the layer before's end,
// its attention and router; n_layer the last layer's end and the head.
void Engine::Impl::tok_seg(int seg) {
    if (seg == g.n_layer) {
        tok_tail(L[(size_t) g.n_layer - 1]);
        tok_head();
        return;
    }
    if (seg == 0) {
        if (!k::embed_rows(embd_host ? d_erow : embd.w, embd.type, embd.rb, dpos + 1, 1, g.n_embd, x, s)) {
            std::fprintf(stderr, "maya-hy4: no embedding rows for type %d\n", embd.type);
            std::abort();
        }
        k::hc_init(x, g.hc, g.n_embd, R, s);
        if (eg_table)
            CK(cudaMemcpyAsync(eg_rows_d, h_eg, g.eg_layers.size() * g.eg_cols * g.eg_dim * sizeof(float),
                               cudaMemcpyHostToDevice, s));
        for (int il = 0; il <= g.n_dense && il < g.n_layer; ++il) {
            Layer& l = L[(size_t) il];
            tok_attn(l);
            if (l.dense) tok_dense_ffn(l);
            else {
                tok_router(l);
                if (dr) tok_moe(l, il);
            }
        }
        return;
    }
    Layer& l = L[(size_t) seg];   // (a MoE layer after the first: the layer before's end first)
    tok_tail(L[(size_t) seg - 1]);
    tok_attn(l);
    tok_router(l);
    if (dr) tok_moe(l, seg);
}

void Engine::Impl::tok_graphs() {
    graphs = 0;
    const char* v = std::getenv("HY4_GRAPHS");
    if ((v && std::atoi(v) == 0) || std::getenv("HY4_TRACE")) return;
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
        std::fprintf(stderr, "maya-hy4: the token path's CUDA graphs could not be made; it launches its kernels one by one\n");
        for (auto& e : gx)
            if (e) cudaGraphExecDestroy(e);
        gx.clear();
        return;
    }
    graphs = 1;
}

// Hy4's routing on the host (llama.cpp's build_moe_ffn with sigmoid gating): sigmoid, + bias for the choice, the top
// n_used (ties: the lower id); the weights the chosen sigmoids normalized x scale
void Engine::Impl::route_host(const Layer& l, const float* rl, int* sel, float* w) const {
    const int ne = g.n_expert;
    float p[512], sc[512];
    for (int e = 0; e < ne; ++e) {
        p[e] = 1.0f / (1.0f + std::exp(-rl[e]));
        sc[e] = p[e] + l.exp_bias[(size_t) e];
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

// The helper card's own queued work (with the device's routing): each MoE layer's experts there read from the plan in
// host-mapped memory by the card itself (hd_plan waits for it), its sum answered there (hd_answer) for the main card's
// moe_wait2; the host runs only the CPU lane's.  HY4_DEV_HELPER=0: the host launches the helper's experts, as before.
void Engine::Impl::hd_alloc() {
    const char* v = std::getenv("HY4_DEV_HELPER");
    hd = dr && hc.slots > 0 && !(v && std::atoi(v) == 0);
    if (!hd) return;
    const int NL = g.n_lat;
    const size_t nk = (size_t) g.n_layer * g.n_expert;
    void* ha = nullptr;
    if (cudaHostAlloc(&ha, (size_t) g.n_layer * ans_ints * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable) !=
        cudaSuccess) {
        cudaGetLastError();
        hd = false;
        std::fprintf(stderr, "maya-hy4: no host-mapped memory for the helper card's answers: the host launches its experts\n");
        return;
    }
    m_hans = (int*) ha;
    std::memset(m_hans, 0, (size_t) g.n_layer * ans_ints * sizeof(int));
    CK(cudaHostGetDevicePointer((void**) &md_hans, m_hans, 0));
    auto portable = [&](size_t bytes) {   // (pinned for the helper's context too)
        void* p = nullptr;
        CK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
        std::memset(p, 0, bytes);
        return p;
    };
    hh_where = (int*) portable(nk * sizeof(int));
    hh_pos = (int*) portable(4 * sizeof(int));
    hwhere_up.assign(nk, INT_MIN);
    OnDev od(hc.dev);
    CK(cudaHostGetDevicePointer((void**) &mdh_hans, m_hans, 0));
    CK(cudaHostGetDevicePointer((void**) &mdh_plan, m_plan, 0));
    CK(cudaHostGetDevicePointer((void**) &mdh_diag, m_diag, 0));
    auto dev_alloc = [&](size_t bytes) {
        void* p = nullptr;
        CK(cudaMalloc(&p, bytes));
        CK(cudaMemset(p, 0, bytes));
        hc.bytes += bytes;
        return p;
    };
    hd_where = (int*) dev_alloc(nk * sizeof(int));
    hd_slotptr = (const uint8_t**) dev_alloc((size_t) hc.slots * sizeof(uint8_t*));
    CK(cudaMemcpy((void*) hd_slotptr, hc.ptr.data(), (size_t) hc.slots * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    hd_gptr = (const uint8_t**) dev_alloc(16 * sizeof(uint8_t*));
    hd_w = (float*) dev_alloc(16 * sizeof(float));
    hd_lat = (float*) dev_alloc((size_t) NL * sizeof(float));
    hd_nh = (int*) dev_alloc(sizeof(int));
    hd_dpos = (int*) dev_alloc(4 * sizeof(int));
    hgx.assign((size_t) g.n_layer, nullptr);
}

// A MoE layer's work on the helper card, on hc.s (the helper current): hc_experts' launches with the device's choices
void Engine::Impl::hd_layer(const Layer& l, int il) {
    const int NL = g.n_lat, ff = g.n_ff_exp, nu = g.n_used, ne = g.n_expert;
    const CpuLayer& c = l.cl;
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    k::hd_plan(mdh_plan + (size_t) il * plan_ints, nu, NL, hd_where + (size_t) il * ne, hd_slotptr, hd_gptr, hd_w, hd_lat,
               hd_nh, hd_dpos, mdh_diag + 1, hc.s);
    k::q8(hd_lat, 1, NL, hc.qq, hc.qd, hc.s);
    auto tab = [&](int type, size_t off, size_t ld, int kk, int n, const int8_t* xq, const float* xd, size_t xs,
                   float* y, size_t ys, const float* alpha) {
        if (!iq::mv_table(type, hd_gptr, off, ld, kk, n, nu, xq, xd, xs, y, ys, alpha, hc.s)) {
            std::fprintf(stderr, "maya-hy4: no table GEMV for the experts' type %d (k %d) on the helper\n", type, kk);
            std::abort();
        }
    };
    tab(c.tg, 0, c.rg, NL, ff, hc.qq, hc.qd, 0, hc.gu, (size_t) 2 * ff, nullptr);
    tab(c.tu, gb, c.ru, NL, ff, hc.qq, hc.qd, 0, hc.gu + ff, (size_t) 2 * ff, nullptr);
    k::swiglu_f32_rows(hc.gu, nu, ff, l.clamp_exp, hc.h, hc.s);
    k::q8(hc.h, nu, ff, hc.hq, hc.hd, hc.s);
    tab(c.td, gb + ub, c.rd, ff, NL, hc.hq, hc.hd, (size_t) ff, hc.y, (size_t) NL, hd_w);
    k::sum_rows(hc.y, nu, NL, hc.routed, hc.s);
    k::hd_answer(hc.routed, hd_nh, NL, mdh_hans + (size_t) il * ans_ints, hd_dpos, hc.s);
}

void Engine::Impl::hd_graphs() {
    hgraphs = 0;
    const char* v = std::getenv("HY4_GRAPHS");
    if (v && std::atoi(v) == 0) return;
    OnDev od(hc.dev);
    bool ok = true;
    for (int il = g.n_dense; il < g.n_layer && ok; ++il) {
        cudaGraph_t gr_ = nullptr;
        ok = cudaStreamBeginCapture(hc.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (!ok) break;
        hd_layer(L[(size_t) il], il);
        ok = cudaStreamEndCapture(hc.s, &gr_) == cudaSuccess && gr_ &&
             cudaGraphInstantiate(&hgx[(size_t) il], gr_, 0) == cudaSuccess;
        if (gr_) cudaGraphDestroy(gr_);
    }
    if (!ok) {
        cudaGetLastError();
        std::fprintf(stderr, "maya-hy4: the helper card's CUDA graphs could not be made; it launches its kernels one by one\n");
        for (auto& e : hgx)
            if (e) cudaGraphExecDestroy(e);
        hgx.assign((size_t) g.n_layer, nullptr);
        return;
    }
    hgraphs = 1;
}

// The helper's slots up to its card when they changed, the token's position and sequence number with them; then its
// first MoE layer's work
void Engine::Impl::hd_start() {
    if (hgraphs < 0) hd_graphs();
    OnDev od(hc.dev);
    const size_t nk = hwhere_up.size();
    bool diff = false;
    for (size_t i = 0; i < nk; ++i) {
        const int v = hc.where[i] >= 0 ? hc.where[i] : -1;
        hh_where[i] = v;
        diff |= v != hwhere_up[i];
    }
    if (diff) {
        std::memcpy(hwhere_up.data(), hh_where, nk * sizeof(int));
        CK(cudaMemcpyAsync(hd_where, hh_where, nk * sizeof(int), cudaMemcpyHostToDevice, hc.s));
    }
    std::memcpy(hh_pos, h_pos, 3 * sizeof(int));
    CK(cudaMemcpyAsync(hd_dpos, hh_pos, 3 * sizeof(int), cudaMemcpyHostToDevice, hc.s));
    hd_launch(g.n_dense);
}

void Engine::Impl::hd_launch(int il) {
    OnDev od(hc.dev);
    if (hgraphs > 0) CK(cudaGraphLaunch(hgx[(size_t) il], hc.s));
    else hd_layer(L[(size_t) il], il);
    (void) cudaStreamQuery(hc.s);   // (submitted now)
}

// ------------------------------------------------------------------------------------------------ the device's routing
void Engine::Impl::dr_alloc() {
    const char* v = std::getenv("HY4_DEV_ROUTE");
    dr = !(v && std::atoi(v) == 0) && g.n_used <= 16;
    if (!dr) return;
    const int NL = g.n_lat;
    const size_t nk = (size_t) g.n_layer * g.n_expert;
    plan_ints = ((size_t) k::kPlanHead + NL + 63) & ~(size_t) 63;
    ans_ints = ((size_t) k::kAnsHead + NL + 63) & ~(size_t) 63;
    const unsigned fl = cudaHostAllocMapped | cudaHostAllocPortable;
    void *hp = nullptr, *ha = nullptr, *hd = nullptr;
    if (cudaHostAlloc(&hp, (size_t) g.n_layer * plan_ints * sizeof(int), fl) != cudaSuccess ||
        cudaHostAlloc(&ha, (size_t) g.n_layer * ans_ints * sizeof(int), fl) != cudaSuccess ||
        cudaHostAlloc(&hd, 64, fl) != cudaSuccess) {
        cudaGetLastError();
        dr = false;
        std::fprintf(stderr, "maya-hy4: no host-mapped memory for the device's routing: the host routes\n");
        return;
    }
    m_plan = (int*) hp;
    m_ans = (int*) ha;
    m_diag = (unsigned*) hd;
    std::memset(m_plan, 0, (size_t) g.n_layer * plan_ints * sizeof(int));
    std::memset(m_ans, 0, (size_t) g.n_layer * ans_ints * sizeof(int));
    std::memset(m_diag, 0, 64);
    CK(cudaHostGetDevicePointer((void**) &md_plan, m_plan, 0));
    CK(cudaHostGetDevicePointer((void**) &md_ans, m_ans, 0));
    CK(cudaHostGetDevicePointer((void**) &md_diag, m_diag, 0));
    d_where = dalloc<int>(nk);
    h_where = halloc<int>(nk);
    where_up.assign(nk, INT_MIN);
    d_slotptr = (const uint8_t**) dalloc<const uint8_t*>((size_t) std::max(1, ec.slots));
    if (ec.slots)
        CK(cudaMemcpy((void*) d_slotptr, ec.ptr.data(), (size_t) ec.slots * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    d_gptr = (const uint8_t**) dalloc<const uint8_t*>(16);
    d_sel = dalloc<int>(16);
    d_wsel = dalloc<float>(16);
    d_need = dalloc<int>(1);
}

// The experts' slots up to the device when they changed since the last upload (they change only between tokens:
// swaps and arrivals at a token's end, the lent slots before one)
void Engine::Impl::dr_where() {
    const size_t nk = where_up.size();
    bool diff = false;
    for (size_t i = 0; i < nk; ++i) {
        const int v = ec.where[i];
        h_where[i] = v;
        diff |= v != where_up[i];
    }
    if (!diff) return;
    std::memcpy(where_up.data(), h_where, nk * sizeof(int));
    CK(cudaMemcpyAsync(d_where, h_where, nk * sizeof(int), cudaMemcpyHostToDevice, s));
}

// A MoE layer on the device after its router: the top n_used, the plan out to the host, the experts in the main card's
// cache (gpu_experts' sums in the choices' order, the others' rows 0), the shared expert; then the host's answer when
// the plan sent it experts (into d_tail for tok_tail)
void Engine::Impl::tok_moe(const Layer& l, int il) {
    const int NL = g.n_lat, ff = g.n_ff_exp, nu = g.n_used, ne = g.n_expert;
    const CpuLayer& c = l.cl;
    k::route(rlog, ne, l.exp_bias_d, 1, ne, 1, 1, nu, g.exp_scale, d_sel, d_wsel, s);
    k::moe_plan(d_sel, d_wsel, nu, lat, NL, d_where + (size_t) il * ne, d_slotptr, d_gptr, d_need,
                md_plan + (size_t) il * plan_ints, dpos, s);
    const size_t gb = (size_t) ff * c.rg, ub = (size_t) ff * c.ru;
    auto tab = [&](int type, size_t off, size_t ld, int kk, int n, const int8_t* xq, const float* xd, size_t xs,
                   float* y, size_t ys, const float* alpha) {
        if (!iq::mv_table(type, d_gptr, off, ld, kk, n, nu, xq, xd, xs, y, ys, alpha, s)) {
            std::fprintf(stderr, "maya-hy4: no table GEMV for the experts' type %d (k %d)\n", type, kk);
            std::abort();
        }
    };
    tab(c.tg, 0, c.rg, NL, ff, xq_l, xd_l, 0, ec.gu, (size_t) 2 * ff, nullptr);
    tab(c.tu, gb, c.ru, NL, ff, xq_l, xd_l, 0, ec.gu + ff, (size_t) 2 * ff, nullptr);
    k::swiglu_f32_rows(ec.gu, nu, ff, l.clamp_exp, ec.h, s);
    k::q8(ec.h, nu, ff, ec.hq, ec.hd, s);
    tab(c.td, gb + ub, c.rd, ff, NL, ec.hq, ec.hd, (size_t) ff, ec.y, (size_t) NL, d_wsel);
    k::sum_rows(ec.y, nu, NL, ec.routed, s);
    // the shared expert (f's int8 rows), unclamped
    {
        const DMat* ms[2] = {&l.sh_gate, &l.sh_up};
        float* ys[2] = {sg, su};
        mv_many(ms, ys, 2, f, xq_f, xd_f);
    }
    k::swiglu(sg, su, sh, g.n_ff_sh, 0.0f, s);
    k::q8(sh, 1, g.n_ff_sh, xq_h, xd_h, s);
    mv(l.sh_down, sh, xq_h, xd_h, so);
    if (hd) k::moe_wait2(md_ans + (size_t) il * ans_ints, md_hans + (size_t) il * ans_ints, d_need, dpos, NL, d_tail, md_diag, s);
    else k::moe_wait(md_ans + (size_t) il * ans_ints, d_need, dpos, NL, d_tail, md_diag, s);
}

// step()'s layers with the device's routing: every segment launched at once (graphs; else each layer's as the host
// gets to it), then each MoE layer's plan read in order: the scores, the helper's and the CPU lane's experts, their sum
// answered; the head waited for.
void Engine::Impl::step_dr(double& cpu_ms, double& tp) {
    static const bool ptimes = std::getenv("HY4_STEP_TIMES") != nullptr;
    auto phase = [&](double& acc) {
        if (!ptimes) return;
        const double t = now_ms();
        acc += t - tp;
        tp = t;
    };
    const int NL = g.n_lat, ne = g.n_expert, ff = g.n_ff_exp, nu = g.n_used, seq = tseq;
    static const bool tr = std::getenv("HY4_DR_TRACE") != nullptr;   // (each layer's host side, as it goes)
    auto trace = [&](const char* what, int il, int a0, int a1, int a2) {
        if (!tr) return;
        std::fprintf(stderr, "[dr] token %d layer %d: %s %d %d %d\n", seq, il, what, a0, a1, a2);
        std::fflush(stderr);
    };
    dr_where();
    // HY4_DR_LAZY=0: every segment's graph launched at once; else each the moment the layer before has its plan (the
    // stream then holds a segment or two: work on the other streams - the swaps' copies - is not queued behind a token)
    static const bool lazy = [] {
        const char* v = std::getenv("HY4_DR_LAZY");
        return !(v && std::atoi(v) == 0);
    }();
    if (hd) hd_start();
    if (graphs > 0) {
        CK(cudaGraphLaunch(gx[0], s));
        if (!lazy) {
            for (int il = g.n_dense + 1; il < g.n_layer; ++il) CK(cudaGraphLaunch(gx[(size_t) il], s));
            CK(cudaGraphLaunch(gx[(size_t) g.n_layer], s));
        }
        (void) cudaStreamQuery(s);   // (submitted now: WDDM holds launches until something does)
    }
    bool staging = false;
    for (int il = g.n_dense; il < g.n_layer; ++il) {
        Layer& l = L[(size_t) il];
        if (graphs <= 0) {
            tok_seg(il == g.n_dense ? 0 : il);
            (void) cudaStreamQuery(s);
        }
        volatile const int* pl = m_plan + (size_t) il * plan_ints;
        if (pl[0] != seq) {
            const double tw = now_ms();
            for (unsigned spin = 1; pl[0] != seq; ++spin) {
                _mm_pause();
                if ((spin & 0xfffu) == 0 && now_ms() - tw > 60000.0) {
                    std::fprintf(stderr, "maya-hy4: layer %d's plan did not come in 60 s (token %d, saw %d)\n", il, seq,
                                 pl[0]);
                    std::abort();
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        trace("plan", il, pl[1], 0, 0);
        if (graphs > 0 && lazy) {   // (the next segment behind this layer's wait)
            CK(cudaGraphLaunch(gx[(size_t) (il + 1)], s));
            (void) cudaStreamQuery(s);
        }
        if (hd && il + 1 < g.n_layer) hd_launch(il + 1);   // (the helper's next layer, waiting for its plan)
        if (staging) {
            ec_stage_end();
            staging = false;
            trace("staged", il, 0, 0, 0);
        }
        phase(pt.att);
        const int nmain = pl[1];
        int csel[16], hslot[16];
        float cw[16], hw[16];
        int nc = 0, nh = 0;
        for (int i = 0; i < nu; ++i) {
            const int e = pl[2 + i], sl = pl[34 + i], wb = pl[18 + i];
            float wi;
            std::memcpy(&wi, &wb, sizeof(float));
            const size_t key = (size_t) il * ne + e;
            if (route_log) route_tok.push_back((uint16_t) e);
            ec.score[key] += 1.0f;
            if (sl >= 0) continue;
            const int hs = hc.slots ? hc.where[key] : -1;
            if (hs >= 0) {
                hslot[nh] = hs;
                hw[nh++] = wi;
            } else {
                csel[nc] = e;
                cw[nc++] = wi;
            }
        }
        ec.lookups += nu;
        ec.hits += nmain;
        hc.hits += nh;
        phase(pt.route);
        const int nh_host = hd ? 0 : nh;   // (hd: the helper card takes its own)
        if (nc || nh_host) {
            const double c0 = now_ms();
            const float* xin = (const float*) (pl + k::kPlanHead);
            int* ans = m_ans + (size_t) il * ans_ints;
            float* part = (float*) (ans + k::kAnsHead);
            if (nc) lane.begin(*pool, l.cl, NL, ff, csel, cw, nc, xin, part);
            if (nh_host) hc_experts(l, hslot, hw, nh, xin);
            phase(pt.glaunch);
            if (nc) lane.end();
            const double c1 = now_ms();
            st.ms_lane += c1 - c0;
            if (nh_host) {
                {
                    OnDev od(hc.dev);
                    CK(cudaStreamSynchronize(hc.s));
                }
                if (nc)
                    for (int i = 0; i < NL; ++i) part[i] += hc.h_out[i];
                else std::memcpy(part, hc.h_out, (size_t) NL * sizeof(float));
                st.ms_helper += now_ms() - c1;
            }
            // (HY4_DR_STALL_AT=<n>, a test of the fault path: token n's first answer 40 s late - the main card gives up)
            static const int stall_at = std::getenv("HY4_DR_STALL_AT") ? std::atoi(std::getenv("HY4_DR_STALL_AT")) : -1;
            static bool stalled = false;
            if (seq == stall_at && !stalled) {
                stalled = true;
                std::fprintf(stderr, "maya-hy4: HY4_DR_STALL_AT: token %d, layer %d's answer 40 s late\n", seq, il);
                std::this_thread::sleep_for(std::chrono::seconds(40));
            }
            std::atomic_thread_fence(std::memory_order_release);
            ((volatile int*) ans)[0] = seq;
            trace("answered", il, nc, nh, 0);
            cpu_ms += now_ms() - c0;
            phase(pt.cpu);
        }
        pt.cpu_experts += nc;
        st.lane_layers += nc > 0;
        st.helper_layers += nh > 0;
        staging = ec_stage_begin(ec.piece, false);   // (a piece of the swaps' copies while the GPU goes on)
        trace("stage begun", il, staging, 0, 0);
    }
    if (graphs <= 0) tok_seg(g.n_layer);
    if (staging) ec_stage_end();
    const bool st4 = ec_stage_begin(4 * ec.piece, false);   // (the head's GEMV is the token's longest wait)
    trace("head sync", g.n_layer, st4, 0, 0);
    CK(cudaStreamSynchronize(s));
    if (st4) ec_stage_end();
    trace("done", g.n_layer, 0, 0, 0);
    phase(pt.tail);
    // A card that gave up waiting (~30 s) went on with what it had: the helper with a stale plan (its answer then
    // stamped as this token's), the main card without an answer - the token, and this conversation's state, are not
    // to be trusted.  fault: step() returns nullptr, the caller fails the request and resets.
    if (hd && m_diag[1] != hdiag_seen) {
        hdiag_seen = m_diag[1];
        fault = true;
        std::fprintf(stderr, "maya-hy4: the helper card gave up waiting for a plan (%u times so far)\n", hdiag_seen);
    }
    if (m_diag[0] != diag_seen) {
        diag_seen = m_diag[0];
        fault = true;
        std::fprintf(stderr, "maya-hy4: the device gave up waiting for the host's answer (%u times so far)\n", diag_seen);
    }
}

const float* Engine::Impl::step(int token) {
    if (pos >= max_ctx || token < 0 || token >= g.vocab) return nullptr;
    const double t0 = now_ms();
    double cpu_ms = 0.0;
    const int NL = g.n_lat;
    static const bool ptimes = std::getenv("HY4_STEP_TIMES") != nullptr;
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
        // The prompt changed the scores, and the main cache with them: the helper follows, with more copies after a
        // longer prompt (an eighth of its tokens, 64 .. fill_cap).
        hc_update(std::min(hc.fill_cap, std::max(64, (int) std::min<long>(pf_tokens_open / 8, INT_MAX))), false, false);
        pf_tokens_open = 0;
    }
    if (graphs < 0) tok_graphs();
    ec_kv_reserve(pos + 1);
    // (HY4_HELPER_KEEPALIVE) the helper's keep-alive kernel again once the last one has ended
    if (hc.ks && cudaStreamQuery(hc.ks) == cudaSuccess) {
        OnDev od(hc.dev);
        k::keepalive(hc.ka_ns, hc.ks);
        (void) cudaStreamQuery(hc.ks);   // (submitted now: WDDM holds launches until something does)
    }

    // the token's row (to d_erow, decoded by segment 0), its Engram rows (h_eg, copied up by segment 0), the position
    // and the row's index on the device
    if (embd_host) {
        std::memcpy(h_erow, embd.w + (size_t) token * embd.rb, embd.rb);
        CK(cudaMemcpyAsync(d_erow, h_erow, embd.rb, cudaMemcpyHostToDevice, s));
    }
    eg_fill(token);
    if ((int) hist.size() <= pos) hist.resize((size_t) pos + 1);
    hist[(size_t) pos] = token;
    h_pos[0] = pos;
    h_pos[1] = embd_host ? 0 : token;
    h_pos[2] = ++tseq;
    CK(cudaMemcpyAsync(dpos, h_pos, 3 * sizeof(int), cudaMemcpyHostToDevice, s));
    phase(pt.embed);
    if (dr) {
        step_dr(cpu_ms, tp);
        if (fault) {   // (a card gave up waiting: the token is not to be trusted)
            ++pos;
            return nullptr;
        }
    }
    else {   // the host's routing: a sync each MoE layer

    static const bool trace = std::getenv("HY4_TRACE") != nullptr;
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
        dump("streams", il, R, pos);
        phase(pt.att);
        const double c0 = now_ms();
        int sel[16];
        float wv[16];
        route_host(l, h_rlog, sel, wv);
        phase(pt.route);
        // the chosen experts in VRAM go to the GPU, the others to the CPU lane, both at once
        int gslot[16], csel[16], hslot[16];
        float gw[16], cw[16], hw[16];
        int ng = 0, nc = 0, nh = 0;
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
            } else {
                csel[nc] = sel[i];
                cw[nc++] = wv[i];
            }
        }
        ec.lookups += g.n_used;
        ec.hits += ng;
        hc.hits += nh;
        // the CPU lane's workers start on the rest now; this thread launches the GPUs' parts, then joins them
        float* h_routed = h_tail + 4;
        if (nc) lane.begin(*pool, l.cl, NL, ff, csel, cw, nc, h_f, h_routed);
        if (nh) hc_experts(l, hslot, hw, nh);   // (the helper card starts on its share first: it is the slower one)
        if (ng) gpu_experts(l, gslot, gw, ng);
        // the shared expert (f's int8 rows), unclamped
        {
            const DMat* ms[2] = {&l.sh_gate, &l.sh_up};
            float* ys[2] = {sg, su};
            mv_many(ms, ys, 2, f, xq_f, xd_f);
        }
        k::swiglu(sg, su, sh, g.n_ff_sh, 0.0f, s);
        k::q8(sh, 1, g.n_ff_sh, xq_h, xd_h, s);
        mv(l.sh_down, sh, xq_h, xd_h, so);
        if (trace) {
            CK(cudaStreamSynchronize(s));
            std::fprintf(stderr, "[trace] layer %d: experts %d %d %d %d .., %d on the GPU; CPU lane next\n", il, sel[0],
                         sel[1], sel[2], sel[3], ng);
            std::fflush(stderr);
        }
        phase(pt.glaunch);
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
                for (int i = 0; i < NL; ++i) h_routed[i] += hc.h_out[i];
            else std::memcpy(h_routed, hc.h_out, (size_t) NL * sizeof(float));
            st.ms_helper += now_ms() - c1;
        }
        cpu_ms += now_ms() - c0;
        phase(pt.cpu);
        pt.cpu_experts += nc;
        int* flags = (int*) h_tail;   // which lanes ran, for this layer's end (the next segment's start)
        flags[0] = nc > 0 || nh > 0;
        flags[1] = ng > 0;
        phase(pt.tail);
    }

    // the last layer's end and the output head
    {
        if (graphs > 0) CK(cudaGraphLaunch(gx[(size_t) g.n_layer], s));
        else tok_seg(g.n_layer);
        const bool staging = ec_stage_begin(4 * ec.piece);   // (the head's GEMV is the token's longest wait)
        CK(cudaStreamSynchronize(s));
        if (staging) ec_stage_end();
    }
    }   // (the host's routing)
    dump("final", g.n_layer, R, pos);
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
                     "maya-hy4: step phases, ms a token (64 tokens): embed %.2f, attention+sync %.2f, routing %.2f, GPU "
                     "launches %.2f, CPU lane %.2f (%.1f experts, %.3f ms each), tail %.2f, head %.2f, adapt %.2f; "
                     "total %.2f\n",
                     pt.embed / n, pt.att / n, pt.route / n, pt.glaunch / n, pt.cpu / n, pt.cpu_experts / n,
                     pt.cpu_experts ? pt.cpu / pt.cpu_experts : 0.0, pt.tail / n, pt.head / n, pt.adapt / n,
                     (pt.embed + pt.att + pt.route + pt.glaunch + pt.cpu + pt.tail + pt.head + pt.adapt) / n);
        pt = PhaseTimes{};
    }
    return h_logits;
}

}  // namespace hy4
