// src/nem/nem_cpu.cpp - maya-nem's CPU lane (nem_cpu.hpp).  The pool is maya-ink's (src/ink/ink_cpu.cpp).
#include "nem_cpu.hpp"

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
#define NEM_PAUSE() _mm_pause()
#else
#define NEM_PAUSE() std::this_thread::yield()
#endif

namespace nem {

namespace dg = strata::ds4::gguf;

// Each run is a generation every worker takes part in once: start() hands it out (its function and count written
// before the generation's number goes up), join() returns when all of them are through with it, so a worker can never
// carry one run's job counter or function into the next.  A worker waits for the next generation spinning (a token's
// layers come every few hundred microseconds; a sleeping thread takes tens of them to wake on Windows), and sleeps on
// the condition variable once none has come for NEM_POOL_SPIN_US.
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
                    NEM_PAUSE();
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
    if (const char* v = std::getenv("NEM_POOL_SPIN_US")) p->spin = std::chrono::microseconds(std::max(0, std::atoi(v)));
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
    while (p->active.load(std::memory_order_acquire) > 0) NEM_PAUSE();
}

void Pool::run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    start(n, &f);
    join();
}

bool cpu_layer(int tu, int td, int n_in, int n_ff, CpuLayer& L, std::string& err) {
    dg::IqCpu a;   // (up in the gate / up role: rows of n_in values; down: rows of n_ff)
    if (!dg::iq_cpu_fmt(tu, td, n_in, n_ff, a, err)) return false;
    L.tu = tu;
    L.td = td;
    L.ru = a.gu_row;
    L.rd = a.d_row;
    L.act_u = a.gu_act;
    L.act_d = a.d_act;
    L.ab_u = a.act_bytes;
    L.ab_d = a.h_bytes;
    return true;
}

namespace {
constexpr int kRowsU = 128, kRowsD = 64;
}

void CpuLane::begin(Pool& pool, const CpuLayer& L, int n_in, int n_ff, const int* experts, const float* w, int n,
                    const float* x, float* out) {
    pool_ = &pool;
    L_ = &L;
    n_in_ = n_in;
    n_ff_ = n_ff;
    n_ = std::min(n, kMax);
    out_ = out;
    if (n_ <= 0) {
        std::fill(out, out + n_in, 0.0f);
        pool_ = nullptr;
        return;
    }
    for (int i = 0; i < n_; ++i) {
        experts_[i] = experts[i];
        w_[i] = w[i];
        done_u_[i].store(0, std::memory_order_relaxed);
    }
    ready_.store(0, std::memory_order_relaxed);
    au_.resize(L.ab_u);
    h_.resize((size_t) kMax * n_ff);
    o_.resize((size_t) kMax * n_in);
    hq_.resize((size_t) kMax * L.ab_d);
    dg::iq_cpu_quant(L.act_u, x, n_in, au_.data());
    bands_u_ = (n_ff + kRowsU - 1) / kRowsU;
    bands_d_ = (n_in + kRowsD - 1) / kRowsD;
    if (!fn_) fn_ = [this](int j) { job(j); };
    pool.start(n_ * bands_u_ + bands_d_, &fn_);
}

void CpuLane::job(int j) {
    const CpuLayer& L = *L_;
    const int n_ff = n_ff_, n_in = n_in_;
    if (j < n_ * bands_u_) {
        // up: rows [r0, r1) of expert i, relu(.)^2 into h_i
        const int i = j / bands_u_, r0 = (j % bands_u_) * kRowsU, r1 = std::min(n_ff, r0 + kRowsU);
        const size_t e = (size_t) experts_[i];
        float* hi = h_.data() + (size_t) i * n_ff;
        const void* act = au_.data();
        dg::iq_cpu_rows(L.tu, L.up + e * n_ff * L.ru, L.ru, n_in, &act, 1, &hi, r0, r1);
        for (int r = r0; r < r1; ++r) {
            const float v = hi[r] > 0.0f ? hi[r] : 0.0f;
            hi[r] = v * v;
        }
        // the expert's last band: its h as the down rows' activation
        if (done_u_[i].fetch_add(1, std::memory_order_acq_rel) == bands_u_ - 1) {
            dg::iq_cpu_quant(L.act_d, hi, n_ff, hq_.data() + (size_t) i * L.ab_d);
            ready_.fetch_add(1, std::memory_order_acq_rel);
        }
        return;
    }
    // down: rows [r0, r1) of every expert (once all their activations are made), then their weighted sum there
    const int b = j - n_ * bands_u_, r0 = b * kRowsD, r1 = std::min(n_in, r0 + kRowsD);
    while (ready_.load(std::memory_order_acquire) < n_) NEM_PAUSE();
    for (int i = 0; i < n_; ++i) {
        const size_t e = (size_t) experts_[i];
        const void* act = hq_.data() + (size_t) i * L.ab_d;
        float* oi = o_.data() + (size_t) i * n_in;
        dg::iq_cpu_rows(L.td, L.down + e * n_in * L.rd, L.rd, n_ff, &act, 1, &oi, r0, r1);
    }
    for (int c = r0; c < r1; ++c) out_[c] = 0.0f;
    for (int i = 0; i < n_; ++i) {
        const float* oi = o_.data() + (size_t) i * n_in;
        for (int c = r0; c < r1; ++c) out_[c] += w_[i] * oi[c];
    }
}

void CpuLane::end() {
    if (!pool_) return;
    pool_->join();
    pool_ = nullptr;
}

}  // namespace nem
