// src/ink/ink_cpu.cpp - maya-ink's CPU lane (ink_cpu.hpp).
#include "ink_cpu.hpp"

#include "../ds4/ds4_gguf_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace ink {

namespace dg = strata::ds4::gguf;

// Each run is a generation every worker takes part in once: run() returns when all of them are through with it, so a
// worker can never carry one run's job counter or function into the next.
struct Pool::Impl {
    std::vector<std::thread> th;
    int workers = 0;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    const std::function<void(int)>* f = nullptr;
    std::atomic<int> next{0};
    int n = 0, finished = 0;
    unsigned gen = 0;
    bool quit = false;

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
            const std::function<void(int)>* fn;
            int count;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
                fn = f;
                count = n;
            }
            work(fn, count);
            {
                std::lock_guard<std::mutex> lk(mu);
                if (++finished == workers) done_cv.notify_all();
            }
        }
    }
};

Pool::Pool(int workers) : p(new Impl) {
    p->workers = std::max(0, workers);
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

void Pool::run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->f = &f;
        p->n = n;
        p->finished = 0;
        p->next.store(0, std::memory_order_relaxed);
        ++p->gen;
    }
    p->cv.notify_all();
    p->work(&f, n);
    std::unique_lock<std::mutex> lk(p->mu);
    p->done_cv.wait(lk, [&] { return p->finished == p->workers; });
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
    return true;
}

void cpu_experts(Pool& pool, const CpuLayer& L, int n_embd, int n_ff, const int* experts, const float* w, int n,
                 const float* x, float* out) {
    constexpr int kRowsGU = 128, kRowsD = 256, kMax = 16;
    n = std::min(n, kMax);
    std::fill(out, out + n_embd, 0.0f);
    if (n == 0) return;
    // (the caller's buffers, by pointer: a lambda's [&] does not capture a thread_local - each worker would use its own)
    thread_local std::vector<uint8_t> ag_v, au_v, hq_v;
    thread_local std::vector<float> h_v, o_v;
    ag_v.resize(L.ab_g);
    au_v.resize(L.ab_u);
    const bool same = L.tg == L.tu && L.act_g == L.act_u;
    h_v.resize((size_t) n * n_ff * (same ? 1 : 2));
    o_v.resize((size_t) n * n_embd);
    hq_v.resize((size_t) n * L.ab_d);
    uint8_t *const ag = ag_v.data(), *const au = au_v.data(), *const hq = hq_v.data();
    float *const h = h_v.data(), *const o = o_v.data();
    dg::iq_cpu_quant(L.act_g, x, n_embd, ag);
    if (!same) dg::iq_cpu_quant(L.act_u, x, n_embd, au);
    dg::IqCpu f;
    std::string err;
    dg::iq_cpu_fmt(L.tg, L.td, n_embd, n_ff, f, err);

    // gate / up: (expert, row band) jobs
    const int bands_gu = (n_ff + kRowsGU - 1) / kRowsGU;
    const std::function<void(int)> gu = [&](int j) {
        const int i = j / bands_gu, r0 = (j % bands_gu) * kRowsGU, r1 = std::min(n_ff, r0 + kRowsGU);
        const size_t e = (size_t) experts[i];
        float* hi = h + (size_t) i * n_ff;
        const void* act = ag;
        if (same) {
            dg::iq_cpu_gate_up(f, L.gate + e * n_ff * L.rg, L.up + e * n_ff * L.ru, &act, 1, &hi, r0, r1, 0.0f);
        } else {
            float* ui = h + (size_t) (n + i) * n_ff;
            const void* actu = au;
            dg::iq_cpu_rows(L.tg, L.gate + e * n_ff * L.rg, L.rg, n_embd, &act, 1, &hi, r0, r1);
            dg::iq_cpu_rows(L.tu, L.up + e * n_ff * L.ru, L.ru, n_embd, &actu, 1, &ui, r0, r1);
            for (int r = r0; r < r1; ++r) {
                const float g = hi[r];
                hi[r] = g / (1.0f + std::exp(-g)) * ui[r];
            }
        }
    };
    pool.run(n * bands_gu, gu);
    for (int i = 0; i < n; ++i) dg::iq_cpu_quant(L.act_d, h + (size_t) i * n_ff, n_ff, hq + (size_t) i * L.ab_d);

    // down: (expert, row band) jobs
    const int bands_d = (n_embd + kRowsD - 1) / kRowsD;
    const std::function<void(int)> dn = [&](int j) {
        const int i = j / bands_d, r0 = (j % bands_d) * kRowsD, r1 = std::min(n_embd, r0 + kRowsD);
        const size_t e = (size_t) experts[i];
        const void* act = hq + (size_t) i * L.ab_d;
        float* oi = o + (size_t) i * n_embd;
        dg::iq_cpu_rows(L.td, L.down + e * n_embd * L.rd, L.rd, n_ff, &act, 1, &oi, r0, r1);
    };
    pool.run(n * bands_d, dn);
    for (int i = 0; i < n; ++i) {
        const float* oi = o + (size_t) i * n_embd;
        for (int c = 0; c < n_embd; ++c) out[c] += w[i] * oi[c];
    }
}

}  // namespace ink
