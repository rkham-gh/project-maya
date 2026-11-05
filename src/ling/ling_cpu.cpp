// src/ling/ling_cpu.cpp - maya-ling's CPU lane (ling_cpu.hpp).
#include "ling_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define LING_PAUSE() _mm_pause()
#else
#define LING_PAUSE() std::this_thread::yield()
#endif

namespace ling {

namespace dg = strata::ds4::gguf;

// Each run is a generation every worker takes part in once: start() hands it out (its function and count written
// before the generation's number goes up), join() returns when all of them are through with it, so a worker can never
// carry one run's job counter or function into the next.  A worker waits for the next generation spinning (a token's
// layers come every few hundred microseconds; a sleeping thread takes tens of them to wake on Windows), and sleeps on
// the condition variable once none has come for LING_POOL_SPIN_US.
struct Pool::Impl {
    std::vector<std::thread> th;
    int workers = 0;
    std::mutex mu;
    std::condition_variable cv;
    const std::function<void(int)>* f = nullptr;
    int n = 0;
    std::atomic<int> next{0}, active{0}, sleepers{0};
    std::atomic<unsigned> gen{0};
    std::atomic<bool> quit{false};
    std::chrono::microseconds spin{3000};

    void work(const std::function<void(int)>* fn, int count) {
        for (;;) {
            const int j = next.fetch_add(1, std::memory_order_relaxed);
            if (j >= count) break;
            (*fn)(j);
        }
    }
    void loop() {
        unsigned seen = 0;
        for (;;) {
            unsigned g = gen.load(std::memory_order_acquire);
            if (g == seen) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; (g = gen.load(std::memory_order_acquire)) == seen;) {
                    if (quit.load(std::memory_order_relaxed)) return;
                    LING_PAUSE();
                    if (++k == 64) {
                        k = 0;
                        if (std::chrono::steady_clock::now() - t0 > spin) break;
                    }
                }
                if (g == seen) {
                    std::unique_lock<std::mutex> lk(mu);
                    sleepers.fetch_add(1, std::memory_order_seq_cst);
                    cv.wait(lk, [&] { return quit.load() || gen.load(std::memory_order_seq_cst) != seen; });
                    sleepers.fetch_sub(1, std::memory_order_relaxed);
                    if (quit.load()) return;
                    g = gen.load(std::memory_order_acquire);
                }
            }
            seen = g;
            work(f, n);
            active.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
};

Pool::Pool(int workers) : p(new Impl) {
    p->workers = std::max(0, workers);
    if (const char* v = std::getenv("LING_POOL_SPIN_US")) p->spin = std::chrono::microseconds(std::max(0, std::atoi(v)));
    for (int i = 0; i < p->workers; ++i) p->th.emplace_back([this] { p->loop(); });
}
Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->quit = true;
    }
    p->cv.notify_all();
    for (auto& t : p->th) t.join();
    delete p;
}
int Pool::threads() const { return (int) p->th.size() + 1; }

void Pool::start(int n, const std::function<void(int)>* f) {
    p->f = f;
    p->n = n;
    p->next.store(0, std::memory_order_relaxed);
    p->active.store(p->workers, std::memory_order_relaxed);
    p->gen.fetch_add(1, std::memory_order_seq_cst);
    if (p->sleepers.load(std::memory_order_seq_cst) > 0) {
        std::lock_guard<std::mutex> lk(p->mu);
        p->cv.notify_all();
    }
}

void Pool::join() {
    p->work(p->f, p->n);
    while (p->active.load(std::memory_order_acquire) > 0) LING_PAUSE();
}

void Pool::run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    start(n, &f);
    join();
}

bool cpu_layer(int tg, int tu, int td, int n_embd, int n_ff, CpuLayer& L, std::string& err) {
    dg::IqCpu a, b;
    if (!dg::iq_cpu_fmt(tg, td, n_embd, n_ff, a, err)) return false;
    if (!dg::iq_cpu_fmt(tu, td, n_embd, n_ff, b, err)) return false;
    L.tg = tg;
    L.tu = tu;
    L.td = td;
    L.rg = a.gu_row;
    L.ru = b.gu_row;
    L.rd = a.d_row;
    L.act_g = a.gu_act;
    L.act_u = b.gu_act;
    L.act_d = a.d_act;
    L.ab_g = a.act_bytes;
    L.ab_u = b.act_bytes;
    L.ab_d = a.h_bytes;
    L.fmt = a;
    return true;
}

namespace {
constexpr int kRowsGU = 64, kRowsD = 128;
}

void CpuLane::begin(Pool& pool, const CpuLayer& L, int n_embd, int n_ff, const int* experts, const float* w, int n,
                    const float* x, float* out) {
    pool_ = &pool;
    L_ = &L;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    n_ = std::min(n, kMax);
    out_ = out;
    if (n_ <= 0) {
        std::fill(out, out + n_embd, 0.0f);
        pool_ = nullptr;
        return;
    }
    for (int i = 0; i < n_; ++i) {
        experts_[i] = experts[i];
        w_[i] = w[i];
        done_gu_[i].store(0, std::memory_order_relaxed);
    }
    ready_.store(0, std::memory_order_relaxed);
    // one pass for gate and up (iq_cpu_gate_up, DeepSeek's clamp form) only where there is no clamp; Ling's clamp
    // (on silu(g), not on g) takes the two matrices' rows and its own epilogue
    same_ = L.tg == L.tu && L.act_g == L.act_u && L.limit <= 0.0f;
    one_act_ = L.act_g == L.act_u;
    ag_.resize(L.ab_g);
    au_.resize(L.ab_u);
    h_.resize((size_t) kMax * n_ff * 2);
    o_.resize((size_t) kMax * n_embd);
    hq_.resize((size_t) kMax * L.ab_d);
    dg::iq_cpu_quant(L.act_g, x, n_embd, ag_.data());
    if (!one_act_) dg::iq_cpu_quant(L.act_u, x, n_embd, au_.data());
    bands_gu_ = (n_ff + kRowsGU - 1) / kRowsGU;
    bands_d_ = (n_embd + kRowsD - 1) / kRowsD;
    if (!fn_) fn_ = [this](int j) { job(j); };
    pool.start(n_ * bands_gu_ + bands_d_, &fn_);
}

void CpuLane::job(int j) {
    const CpuLayer& L = *L_;
    const int n_ff = n_ff_, n_embd = n_embd_;
    if (j < n_ * bands_gu_) {
        // gate / up: rows [r0, r1) of expert i, swiglu(gate, up) into h_i
        const int i = j / bands_gu_, r0 = (j % bands_gu_) * kRowsGU, r1 = std::min(n_ff, r0 + kRowsGU);
        const size_t e = (size_t) experts_[i];
        float* hi = h_.data() + (size_t) i * n_ff;
        const void* act = ag_.data();
        if (same_) {
            dg::iq_cpu_gate_up(L.fmt, L.gate + e * n_ff * L.rg, L.up + e * n_ff * L.ru, &act, 1, &hi, r0, r1, 0.0f);
        } else {
            float* ui = h_.data() + (size_t) (kMax + i) * n_ff;
            const void* actu = one_act_ ? ag_.data() : au_.data();
            dg::iq_cpu_rows(L.tg, L.gate + e * n_ff * L.rg, L.rg, n_embd, &act, 1, &hi, r0, r1);
            dg::iq_cpu_rows(L.tu, L.up + e * n_ff * L.ru, L.ru, n_embd, &actu, 1, &ui, r0, r1);
            const float lim = L.limit;
            for (int r = r0; r < r1; ++r) {
                const float g = hi[r];
                float a = g / (1.0f + std::exp(-g)), u = ui[r];
                if (lim > 0.0f) {
                    a = std::fmin(a, lim);
                    u = std::fmin(std::fmax(u, -lim), lim);
                }
                hi[r] = a * u;
            }
        }
        // the expert's last band: its h as the down rows' activation
        if (done_gu_[i].fetch_add(1, std::memory_order_acq_rel) == bands_gu_ - 1) {
            dg::iq_cpu_quant(L.act_d, hi, n_ff, hq_.data() + (size_t) i * L.ab_d);
            ready_.fetch_add(1, std::memory_order_acq_rel);
        }
        return;
    }
    // down: rows [r0, r1) of every expert (once all their activations are made), then their weighted sum there
    const int b = j - n_ * bands_gu_, r0 = b * kRowsD, r1 = std::min(n_embd, r0 + kRowsD);
    while (ready_.load(std::memory_order_acquire) < n_) LING_PAUSE();
    for (int i = 0; i < n_; ++i) {
        const size_t e = (size_t) experts_[i];
        const void* act = hq_.data() + (size_t) i * L.ab_d;
        float* oi = o_.data() + (size_t) i * n_embd;
        dg::iq_cpu_rows(L.td, L.down + e * n_embd * L.rd, L.rd, n_ff, &act, 1, &oi, r0, r1);
    }
    for (int c = r0; c < r1; ++c) out_[c] = 0.0f;
    for (int i = 0; i < n_; ++i) {
        const float* oi = o_.data() + (size_t) i * n_embd;
        for (int c = r0; c < r1; ++c) out_[c] += w_[i] * oi[c];
    }
}

void CpuLane::end() {
    if (!pool_) return;
    pool_->join();
    pool_ = nullptr;
}

}  // namespace ling
