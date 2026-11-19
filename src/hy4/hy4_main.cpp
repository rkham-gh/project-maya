// src/hy4/hy4_main.cpp - maya-hy4: Tencent's Hy4 Flash from a GGUF on one NVIDIA GPU (and a helper card) and the CPU.
//
//   maya-hy4 --gguf <...-00001-of-0000N.gguf> --serve [--max-context N] [--threads N] [--gpu N] [--profile FILE]
//            [--dfly model-00148-of-00148.safetensors]   (or HY4_DFLY=<file>: DFly2, Tencent's drafter, below)
//
// --serve speaks Maya's engine protocol (serve/server.py's StrataEngine): INFO lines, then `READY <context> stop`; per
// request `GEN <max_new> [key=value ...] <id,id,...>` answered by `PP <read> <total> <ms> <tok/s> <reused>` while the
// prompt is read, a `T <id>` line a token, and `DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel>
// <accepted> <offered> <reused>`; `STOP` ends a request early, `QUIT` the engine.  A prompt that continues the last
// conversation (its tokens so far, the generated ones included) reads only the rest, one that continues the last
// prompt's end goes on from its snapshot: the sliding-window and indexer rings hold only the last positions, so
// anything else starts over.  Keys: temperature, top_p, top_k, min_p, seed; lookup=0|1 (the prompt lookup's
// drafts, below; suffix_draft=0|1 too, server.py's strata_tune key), window=N (their window's cap).  A request ends at
// the GGUF's eos (<|role_end|>).
//
// Prompt lookup (maya-ds4's): the conversation's tokens so far suffix-matched (strata::spec::SuffixDrafter, a match of
// HY4_LOOKUP_MIN tokens or more, 3; 0: off) propose what followed them; DraftPolicy weighs a window of them against a
// plain step by this PC's measured round times and the matches' acceptance, and a chosen window is verified in one pass
// (Engine::step_window): each row sampled as a plain step's would be, kept while it equals the draft, the first that
// differs (or the bonus row after the last draft) the round's last token - the same tokens a plain decode samples.
//
// DFly2 (--dfly; hy4_dfly.inc): each round drafts 5 tokens after the newest one from the target's state at the row
// before it; DraftPolicy's cost-aware window (maya-ds4's DS4_DRAFT_COST) takes as many as pay by their probabilities -
// calibrated online against what held - and the round times it measured at this depth (the drafting included); the
// lookup's proposal competes as beside maya-ds4's drafter (choose(); a trusted long match is verified without
// drafting).
// Drafting rests for 48 rounds when its rounds commit fewer tokens a millisecond than plain ones (16 rounds judged; a
// plain round every 32 keeps that rate). Keys: dfly=0|1; spec=1 (server.py's strata_tune: the plain token loop, no
// drafting at all) or spec=N (windows of N rows at most).
//
//   maya-hy4 --gguf <...> --ids @prompt.txt [--gen N | --follow @tokens.txt] [--top N] [--dump-logits out.bin]
//
// The prompt goes through the prompt path (Engine::prefill: chunks of --chunk tokens, default 8192, on the GPU with the
// routed experts streamed to it through --stage VRAM slots); --step reads it a token at a time instead (and prompts
// under HY4_MIN_CHUNK tokens, 256, take the token path anyway - in windows).
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt, each
// printed with the top --top (default 5) as id:log-probability (for a comparison with llama.cpp's).  --follow: the
// given tokens fed instead of the greedy ones, each line ending with that token's own log-probability, so another
// engine's continuation is scored position by position.  --dump-logits: the logits after the prompt (float32, vocab).
//
//   maya-hy4 --gguf <...> --runs runs.txt    several such runs with one load: a line a run, "NAME pf|step @ids.txt
//                                             (@follow.txt | N)", the state reset before each, its lines between
//                                             "RUN NAME" and "END NAME ok|failed"
//
//   maya-hy4 --selftest      the KDA recurrence, the MLA decode attention and the routing against host references
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "hy4_kernels.cuh"
#include "hy4_model.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

std::vector<int> parse_ids(const std::string& a) {
    std::string src = a;
    if (!a.empty() && a[0] == '@') {
        std::ifstream f(a.substr(1));
        std::stringstream ss;
        ss << f.rdbuf();
        src = ss.str();
    }
    for (char& c : src)
        if (c == ',') c = ' ';
    std::vector<int> v;
    std::stringstream ss(src);
    long long x;
    while (ss >> x) v.push_back((int) x);
    return v;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Sampling {
    float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    long long seed = -1;
};

int argmax(const float* l, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (l[i] > l[best]) best = i;
    return best;
}

// A token from the logits at the request's sampling (maya-ink's sampler).  The candidates are the finite logits
// (top_k of them when set); their probabilities are softmax(logit / temperature) normalized over the candidates; min_p
// keeps those at least min_p x the most likely one, top_p the most likely ones until their sum reaches it.  With
// neither (temperature alone) the token is drawn from the whole distribution in one pass, in id order - nothing
// sorted; otherwise only as many of the most likely ones are sorted as the cut needs.
int sample(const float* logits, int V, const Sampling& sp, std::mt19937_64& rng) {
    if (sp.temperature <= 0.0f) return argmax(logits, V);
    const double inv_t = 1.0 / sp.temperature;
    const int mxi = argmax(logits, V);
    const double mx = logits[mxi];
    if (sp.top_k <= 0 && sp.top_p >= 1.0f && sp.min_p <= 0.0f) {
        static thread_local std::vector<float> p;
        p.resize((size_t) V);
        const float fmx = (float) mx, fit = (float) inv_t;
        for (int i = 0; i < V; ++i) p[(size_t) i] = std::exp((logits[i] - fmx) * fit);   // (-inf: 0)
        double sum = 0.0;
        for (int i = 0; i < V; ++i) sum += p[(size_t) i];
        std::uniform_real_distribution<double> U(0.0, sum);
        double u = U(rng);
        int last = mxi;
        for (int i = 0; i < V; ++i) {
            if (!(p[(size_t) i] > 0.0f)) continue;
            last = i;
            u -= p[(size_t) i];
            if (u <= 0.0) return i;
        }
        return last;
    }
    std::vector<int> idx;
    idx.reserve((size_t) V);
    for (int i = 0; i < V; ++i)
        if (std::isfinite(logits[i])) idx.push_back(i);
    auto by = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
    if (sp.top_k > 0 && (size_t) sp.top_k < idx.size()) {
        std::nth_element(idx.begin(), idx.begin() + sp.top_k, idx.end(), by);
        idx.resize((size_t) sp.top_k);
    }
    double sum = 0.0;   // (the candidates' total, for the normalization)
    for (int i : idx) sum += std::exp(((double) logits[i] - mx) * inv_t);
    const double pmin = sp.min_p > 0.0f ? (double) sp.min_p : 0.0;
    size_t sorted = 0, keep = idx.size();
    std::vector<double> p;
    for (size_t m = std::min<size_t>(256, idx.size());; m = std::min(idx.size(), m * 8)) {
        if (m > sorted) {
            std::partial_sort(idx.begin() + (long) sorted, idx.begin() + (long) m, idx.end(), by);
            sorted = m;
        }
        p.resize(sorted);
        for (size_t i = 0; i < sorted; ++i) p[i] = std::exp(((double) logits[idx[i]] - mx) * inv_t) / sum;
        keep = sorted;
        bool cut = sorted == idx.size();
        if (pmin > 0.0)
            for (size_t i = 1; i < sorted; ++i)
                if (p[i] < pmin * p[0]) {
                    keep = i;
                    cut = true;
                    break;
                }
        if (sp.top_p < 1.0f) {
            double c = 0.0;
            for (size_t i = 0; i < keep; ++i) {
                c += p[i];
                if (c >= sp.top_p) {
                    keep = i + 1;
                    cut = true;
                    break;
                }
            }
        } else if (pmin <= 0.0) {
            cut = true;   // (top_k alone: all of them)
            if (sorted < idx.size()) {
                std::sort(idx.begin() + (long) sorted, idx.end(), by);
                sorted = idx.size();
                continue;
            }
        }
        if (cut) break;
    }
    double tot = 0.0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> U(0.0, tot);
    double u = U(rng);
    for (size_t i = 0; i < keep; ++i) {
        u -= p[i];
        if (u <= 0.0) return idx[i];
    }
    return idx[keep - 1];
}

// stdin on a thread: a GEN waits for its turn, a STOP reaches the request that runs
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    void start() {
        std::thread([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_all();
                // (QUIT is the last line read: a thread waiting in getline holds stdin's lock, which exit() takes
                //  to flush the streams - on glibc the engine then never ended while its stdin stayed open)
                if (l == "QUIT") return;
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || !lines.empty(); });
        if (lines.empty()) return false;
        out = lines.front();
        lines.pop_front();
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    hy4::Options opt;
    std::string ids_arg, dump, follow_arg, runs_arg;
    int gen = 0, top = 5;
    bool serve = false, by_step = false, selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-hy4: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--gguf") opt.gguf = val();
        else if (a == "--max-context") opt.max_ctx = std::atoi(val().c_str());
        else if (a == "--threads") opt.threads = std::atoi(val().c_str());
        else if (a == "--gpu") opt.gpu = std::atoi(val().c_str());
        else if (a == "--serve") serve = true;
        else if (a == "--ids") ids_arg = val();
        else if (a == "--gen") gen = std::atoi(val().c_str());
        else if (a == "--top") top = std::max(1, std::atoi(val().c_str()));
        else if (a == "--follow") follow_arg = val();
        else if (a == "--runs") runs_arg = val();
        else if (a == "--step") by_step = true;
        else if (a == "--chunk") opt.chunk = std::max(64, std::atoi(val().c_str()));
        else if (a == "--sub") opt.sub = std::max(16, std::atoi(val().c_str()));
        else if (a == "--stage") opt.stage = std::atoi(val().c_str());
        else if (a == "--cache-gb") opt.cache_gb = std::atof(val().c_str());
        else if (a == "--helper-gpu") opt.helper_gpu = std::atoi(val().c_str());
        else if (a == "--helper-gb") opt.helper_gb = std::atof(val().c_str());
        // (Maya's server adds `--layer-split auto` when a config names several GPUs: here the second visible card
        //  becomes the helper card - this engine's layers all stay on the first)
        else if (a == "--layer-split") {
            (void) val();
            if (opt.helper_gpu < 0) opt.helper_gpu = 1;
        }
        else if (a == "--profile") opt.profile = val();
        else if (a == "--dfly") opt.dfly = val();
        else if (a == "--dump-logits") dump = val();
        else if (a == "--quiet") opt.verbose = false;
        else if (a == "--selftest") selftest = true;
        else {
            std::fprintf(stderr, "maya-hy4: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (selftest) {
        const bool ok = hy4::k::selftest();
        std::fprintf(stderr, "maya-hy4 selftest: %s\n", ok ? "all passed" : "FAILED");
        return ok ? 0 : 1;
    }
    if (opt.gguf.empty()) {
        std::fprintf(stderr, "usage: maya-hy4 --gguf <first shard> (--serve | --ids @file [--gen N]) | --selftest\n");
        return 2;
    }
    if (const char* v = std::getenv("HY4_HELPER_GPU")) opt.helper_gpu = std::atoi(v);   // (-1: no helper card)
    if (const char* v = std::getenv("HY4_DFLY")) opt.dfly = v;   // (DFly2's safetensors; "": none)
    hy4::Engine eng;
    std::string err;
    if (!eng.load(opt, err)) {
        std::fprintf(stderr, "maya-hy4: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        return 1;
    }
    const hy4::Geo& g = eng.geo();
    const int V = g.vocab;
    const int eos = g.eos;   // <|role_end|> (the GGUF's tokenizer.ggml.eos_token_id)

    // One test run: the prompt (the prompt path, or the token path with by_step), then greedy tokens or the follow's,
    // each step's line "pick id:lp ..." (and " | pick:lp" when following); false: the prompt did not fit.
    auto run_one = [&](const std::vector<int>& ids, bool step_path, const std::vector<int>& follow, int n_gen,
                       const std::string& dump_to) -> bool {
        const double t0 = now_ms();
        const float* lg = nullptr;
        if (step_path)
            for (size_t i = 0; i < ids.size() && (i == 0 || lg); ++i) lg = eng.step(ids[i]);
        else
            for (size_t i = 0; i < ids.size() && (i == 0 || lg); i += (size_t) opt.chunk)
                lg = eng.prefill(ids.data() + i, (int) std::min(ids.size() - i, (size_t) opt.chunk));
        const double t1 = now_ms();
        const auto ps = eng.stats();
        std::fprintf(stderr, "maya-hy4: prompt %zu tokens in %.1f s (%.1f tok/s; %s, %.1f GB of experts copied)\n",
                     ids.size(), (t1 - t0) / 1000.0, ids.size() * 1000.0 / (t1 - t0),
                     step_path || (int) ids.size() < 256 ? "the token path" : "the prompt path", ps.pf_copy_gb);
        if (!lg) {
            if (eng.failed()) {
                std::fprintf(stderr, "maya-hy4: FAILED - a card gave up waiting for the other; reset\n");
                eng.reset();
            } else {
                std::fprintf(stderr, "maya-hy4: the prompt did not fit\n");
            }
            return false;
        }
        if (!dump_to.empty()) {
            std::ofstream f(dump_to, std::ios::binary);
            f.write((const char*) lg, (std::streamsize) V * sizeof(float));
        }
        eng.clear_stats();
        const int steps = follow.empty() ? n_gen : (int) follow.size();
        // (HY4_RUNS_DFLY=1, a check of DFly2: a greedy run decoded through its drafts - every draft verified in a
        //  window, a line a token (" d" after one that was a kept draft), the drafts' acceptance by position at the
        //  end)
        static const bool rdfly = std::getenv("HY4_RUNS_DFLY") && std::atoi(std::getenv("HY4_RUNS_DFLY")) != 0;
        // (HY4_RUNS_DFLY_T=1 runs it without a drafter too: plain steps, the same accounting)
        static const bool rplain = std::getenv("HY4_RUNS_DFLY_T") && std::atoi(std::getenv("HY4_RUNS_DFLY_T")) == 1;
        if (rdfly && follow.empty() && ((eng.dfly() && eng.window_max() >= 2) || rplain)) {
            int n = 1;
            long rounds = 0, wins = 0, off[8] = {}, acc[8] = {};
            long cb_n[10] = {}, cb_a[10] = {}, c2_n[10] = {}, c2_a[10] = {};   // (held by confidence tenths)
            double dms = 0.0;
            // (HY4_RUNS_DFLY_T=<T>: windows of T rows at most - the tokens a second at a fixed size; 1: plain steps,
            //  the same accounting.  HY4_RUNS_SKIP=<n>: the counts and the clock from the n-th token on - past a
            //  prompt's refill and the caches' first swaps)
            static const int rT = std::getenv("HY4_RUNS_DFLY_T") ? std::atoi(std::getenv("HY4_RUNS_DFLY_T")) : 0;
            static const int rskip = std::getenv("HY4_RUNS_SKIP") ? std::atoi(std::getenv("HY4_RUNS_SKIP")) : 0;
            // (HY4_RUNS_NOEOS=1: the end token never picked - the next best instead, so a run goes its whole length)
            static const bool noeos = std::getenv("HY4_RUNS_NOEOS") && std::atoi(std::getenv("HY4_RUNS_NOEOS")) != 0;
            auto pick = [&](const float* l) {
                if (!noeos) return argmax(l, V);
                int b = eos == 0 ? 1 : 0;
                for (int v = 0; v < V; ++v)
                    if (v != eos && l[v] > l[b]) b = v;
                return b;
            };
            const int wmx = rT >= 1 ? std::min(rT, eng.window_max()) : eng.window_max();
            // (HY4_RUNS_DFLY_SHADOW=1: DFly2 drafts at every position - its block's rows written into the shared
            //  layers' caches past the position - and is never verified: plain steps, whose logits
            //  (HY4_LOGITS_DUMP) must be the plain run's to the bit with frozen caches)
            static const bool shadow = std::getenv("HY4_RUNS_DFLY_SHADOW") &&
                                       std::atoi(std::getenv("HY4_RUNS_DFLY_SHADOW")) != 0;
            static FILE* sdump =
                std::getenv("HY4_LOGITS_DUMP") ? std::fopen(std::getenv("HY4_LOGITS_DUMP"), "wb") : nullptr;
            if (sdump) std::fwrite(lg, sizeof(float), (size_t) V, sdump);
            int t = pick(lg);
            std::printf("%d\n", t);
            double g0 = now_ms();
            int n0 = 0;
            bool skipped = rskip <= 0;
            while (n < steps && t != eos) {
                if (!skipped && n >= rskip) {
                    skipped = true;
                    n0 = n;
                    g0 = now_ms();
                    rounds = wins = 0;
                    dms = 0.0;
                    for (int i = 0; i < 8; ++i) off[i] = acc[i] = 0;
                    for (int i = 0; i < 10; ++i) cb_n[i] = cb_a[i] = c2_n[i] = c2_a[i] = 0;
                }
                int dr[8], wt[8];
                float cf[8];
                const int room = std::min(wmx, steps - n);
                int nd = room >= 2 || shadow ? eng.dfly_draft(t, shadow ? 5 : room - 1, dr, cf) : -1;
                ++rounds;
                if (shadow) nd = 0;
                if (nd <= 0) {
                    lg = eng.step(t);
                    if (!lg) break;
                    if (sdump) std::fwrite(lg, sizeof(float), (size_t) V, sdump);
                    t = pick(lg);
                    std::printf("%d\n", t);
                    ++n;
                    continue;
                }
                dms += eng.dfly_ms();
                ++wins;
                wt[0] = t;
                for (int i = 0; i < nd; ++i) wt[i + 1] = dr[i];
                const float* lw = eng.step_window(wt, nd + 1);
                if (!lw) break;
                int kept = 1;
                for (int i = 0; i <= nd; ++i) {
                    const int sv = pick(lw + (size_t) i * V);
                    const bool hit = i < nd && sv == dr[i];
                    if (i < nd) {   // (draft i reached: its confidence's tenth, held or not)
                        ++off[i];
                        const int cb = std::min(9, std::max(0, (int) (cf[i] * 10.0f)));
                        ++cb_n[cb];
                        cb_a[cb] += hit;
                        const int c2 = std::min(9, std::max(0, (int) (eng.dfly_conf2(i) * 10.0f)));
                        ++c2_n[c2];
                        c2_a[c2] += hit;
                    }
                    std::printf("%d%s\n", sv, hit ? " d" : "");
                    ++n;
                    t = sv;
                    if (!hit || sv == eos) break;
                    ++acc[i];
                    ++kept;
                }
                eng.window_keep(kept);
            }
            std::fflush(stdout);
            long a_all = 0;
            std::string per;
            for (int i = 0; i < 8; ++i) {
                a_all += acc[i];
                if (off[i]) {
                    char b[48];
                    std::snprintf(b, sizeof b, " %d:%.2f(%ld)", i + 1, (double) acc[i] / off[i], off[i]);
                    per += b;
                }
            }
            const double ms = now_ms() - g0;
            std::string cbs;
            for (int i = 0; i < 10; ++i)
                if (cb_n[i]) {
                    char b[48];
                    std::snprintf(b, sizeof b, " %.1f:%.2f(%ld)", i / 10.0, (double) cb_a[i] / cb_n[i], cb_n[i]);
                    cbs += b;
                }
            std::string c2s;
            for (int i = 0; i < 10; ++i)
                if (c2_n[i]) {
                    char b[48];
                    std::snprintf(b, sizeof b, " %.1f:%.2f(%ld)", i / 10.0, (double) c2_a[i] / c2_n[i], c2_n[i]);
                    c2s += b;
                }
            std::fprintf(stderr, "maya-hy4: DFly2 run: drafts held by confidence:%s; by the other confidence:%s\n",
                         cbs.c_str(), c2s.c_str());
            std::fprintf(stderr,
                         "maya-hy4: DFly2 run: %d tokens in %ld rounds (%ld windows), %.2f tokens a window, %ld drafts "
                         "kept; draft k's acceptance after k - 1 held:%s; drafting %.2f ms a window; %.1f tok/s\n",
                         n - n0, rounds, wins, wins ? (double) (a_all + wins) / wins : 0.0, a_all, per.c_str(),
                         wins ? dms / wins : 0.0, (n - n0) * 1000.0 / ms);
            return !eng.failed();
        }
        // (HY4_LOGITS_DUMP=<file>: every position's logits appended, F32 [vocab] each - a KL check's input)
        static FILE* ldump = std::getenv("HY4_LOGITS_DUMP") ? std::fopen(std::getenv("HY4_LOGITS_DUMP"), "wb") : nullptr;
        // (HY4_RUNS_WINDOW=<T>, a check of windowed decode: a follow fed T tokens a pass - the same lines as one by one)
        static const int rwin = std::getenv("HY4_RUNS_WINDOW") ? std::atoi(std::getenv("HY4_RUNS_WINDOW")) : 0;
        // (HY4_RUNS_REJECT=1, a check of a draft's rollback: each window's tokens past a kept count of 1 .. T (in turn)
        //  wrong ones, window_keep(kept) - the lines still the follow's own; 2: one kept a window; 3 / 4: as 1 / 2
        //  with the follow's own tokens past the kept ones - a rollback that leaves nothing behind gives 1 / 2's bits)
        static const int rrej = std::getenv("HY4_RUNS_REJECT") ? std::atoi(std::getenv("HY4_RUNS_REJECT")) : 0;
        std::vector<int> wt;
        long nwin = 0;
        std::vector<float> wbuf;
        int wleft = 0;   // (positions of the window's logits still to print)
        for (int n = 0; n < steps && lg; ++n) {
            if (ldump) std::fwrite(lg, sizeof(float), (size_t) V, ldump);
            std::vector<int> idx((size_t) V);
            for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
            std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            double se = 0.0;
            const double mx = lg[idx[0]];
            for (int i = 0; i < V; ++i)
                if (std::isfinite(lg[i])) se += std::exp((double) lg[i] - mx);
            const double lse = mx + std::log(se);
            const int pick = follow.empty() ? idx[0] : follow[(size_t) n];
            std::printf("%d", pick);
            for (int i = 0; i < top; ++i) std::printf(" %d:%.5f", idx[(size_t) i], lg[idx[(size_t) i]] - lse);
            if (!follow.empty()) std::printf(" | %d:%.5f", pick, lg[pick] - lse);
            std::printf("\n");
            std::fflush(stdout);
            if (follow.empty() && pick == eos) break;
            if (wleft > 0) {   // (the window's next position)
                lg += V;
                --wleft;
                continue;
            }
            const int W = !follow.empty() && rwin >= 2 ? std::min(rwin, steps - n) : 1;
            if (W >= 2 && W <= eng.window_max()) {
                int kept = W;
                wt.assign(follow.begin() + n, follow.begin() + n + W);
                if (rrej) {
                    kept = rrej % 2 == 0 ? 1 : 1 + (int) (nwin % W);
                    if (rrej <= 2)
                        for (int j = kept; j < W; ++j) wt[(size_t) j] = (wt[(size_t) j] + 7919 * (j + 1)) % V;
                }
                ++nwin;
                const float* lw = eng.step_window(wt.data(), W);
                if (!lw) {
                    lg = nullptr;
                    break;
                }
                eng.window_keep(kept);
                wbuf.assign(lw, lw + (size_t) kept * V);
                lg = wbuf.data();
                wleft = kept - 1;
                continue;
            }
            lg = eng.step(pick);
        }
        if (ldump) std::fflush(ldump);
        if (std::getenv("HY4_VERIFY_FILLS") && std::atoi(std::getenv("HY4_VERIFY_FILLS")) != 0) {
            long checked = 0;
            const long bad = eng.verify_slots(&checked);
            std::fprintf(stderr, "maya-hy4: HY4_VERIFY_FILLS: %ld settled expert slots on both cards read back, %ld differ "
                         "from the mapping\n", checked, bad);
        }
        const bool failed = eng.failed();   // (the run's output is cut short: not ok, so its END says so)
        if (failed) {
            std::fprintf(stderr, "maya-hy4: FAILED - a card gave up waiting for the other; reset\n");
            eng.reset();
        }
        const auto st = eng.stats();
        if (st.tokens > 0)
            std::fprintf(stderr,
                         "maya-hy4: %ld tokens at %.2f tok/s (the CPU lane and the helper card's wait %.0f%% of the time; "
                         "%.1f%% of the experts in VRAM, %ld slots, %ld copies; %.1f%% on the helper card, %ld slots, %ld "
                         "copies)\n",
                         st.tokens, st.tokens * 1000.0 / st.ms_total, 100.0 * st.ms_cpu / st.ms_total,
                         st.lookups ? 100.0 * st.hits / st.lookups : 0.0, st.cache_slots, st.cache_copies,
                         st.lookups ? 100.0 * st.hits_helper / st.lookups : 0.0, st.helper_slots, st.helper_copies);
        if (st.tokens > 0)
            std::fprintf(stderr,
                         "maya-hy4: a token's %.2f ms: the CPU lane %.2f (%.1f MoE layers a token), the wait for the helper "
                         "card after it %.2f (%.1f layers), the helper's swaps %.2f, the rest (the main card) %.2f\n",
                         st.ms_total / st.tokens, st.ms_lane / st.tokens, (double) st.lane_layers / st.tokens,
                         st.ms_helper / st.tokens, (double) st.helper_layers / st.tokens, st.ms_update / st.tokens,
                         (st.ms_total - st.ms_lane - st.ms_helper - st.ms_update) / st.tokens);
        return !failed;
    };

    if (!runs_arg.empty()) {
        // --runs FILE: a line a run, "NAME pf|step @ids.txt (@follow.txt | N)" - the model loaded once, the state reset
        // before each; the runs' lines between "RUN NAME" and "END NAME"
        std::ifstream rf(runs_arg);
        std::string ln;
        while (std::getline(rf, ln)) {
            std::stringstream ls(ln);
            std::string name, mode, ids_f, last_f;
            if (!(ls >> name >> mode >> ids_f >> last_f)) continue;
            const std::vector<int> ids = parse_ids(ids_f);
            const bool fol = !last_f.empty() && last_f[0] == '@';
            const std::vector<int> follow = fol ? parse_ids(last_f) : std::vector<int>();
            eng.reset();
            std::printf("RUN %s\n", name.c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "RUN %s\n", name.c_str());   // (the HY4_DUMP lines' run)
            const bool ok = !ids.empty() && run_one(ids, mode == "step", follow, fol ? 0 : std::atoi(last_f.c_str()), "");
            std::printf("END %s %s\n", name.c_str(), ok ? "ok" : "failed");
            std::fflush(stdout);
        }
        eng.save_profile();
        return 0;
    }
    if (!serve) {
        const std::vector<int> ids = parse_ids(ids_arg);
        if (ids.empty()) {
            std::fprintf(stderr, "maya-hy4: no prompt ids\n");
            return 2;
        }
        const std::vector<int> follow = follow_arg.empty() ? std::vector<int>() : parse_ids(follow_arg);
        const bool ok = run_one(ids, by_step, follow, gen, dump);
        eng.save_profile();
        return ok ? 0 : 1;
    }

    // --serve
    std::printf("INFO engine=maya-hy4-%s model=Hy4-flash layers=%d experts=%d vocab=%d\n", kVersion, g.n_layer,
                g.n_expert, V);
    std::printf("READY %d stop\n", eng.max_ctx());
    std::fflush(stdout);
    Input in;
    in.start();
    std::vector<int> hist;   // the tokens the state holds (prompt and generated, the last sampled one not fed)
    std::vector<float> last; // the logits after hist
    // the last prompt's end (Engine::snapshot): its tokens and the logits after them - a request that continues those
    // but not hist (the last answer re-rendered differently) goes on from there instead of from the start
    std::vector<int> snap_hist;
    std::vector<float> snap_last;
    bool have_snap = false;
    std::mt19937_64 rng(std::random_device{}());
    // the prompt lookup (above): its drafter a request, its policy over the engine's life (the round times: this PC's)
    const int wmax = eng.window_max();
    const int lk_min = std::getenv("HY4_LOOKUP_MIN") ? std::atoi(std::getenv("HY4_LOOKUP_MIN")) : 3;
    strata::spec::SuffixDrafter sfx(std::max(1, lk_min), 64, (size_t) eng.max_ctx());
    strata::spec::DraftPolicy policy(std::max(1, wmax));
    // DFly2 (--dfly): its drafts' window by DraftPolicy's cost-aware window (maya-ds4's DS4_DRAFT_COST: the drafts'
    // chance to hold from their confidences, calibrated online, against the measured round times by depth)
    const bool df_have = eng.dfly() && wmax >= 2;
    strata::spec::DraftPolicy dcost(std::max(1, wmax));
    int ddr[8];
    float dconf[8];
    // DFly2's rest: the tokens a millisecond of the rounds that drafted and of plain rounds (decayed sums over the
    // engine's life, by DraftPolicy's depth bands: < 8K, < 64K, longer - a token costs more deep, and a plain rate from
    // another depth would judge drafting wrongly for hundreds of rounds); when drafting does worse over kDfJudge rounds,
    // kDfRest rounds without it, then it is tried again.  Plain rounds come from the rest, the first token after a
    // prompt, and a probe every kDfProbe rounds
    constexpr int kDfJudge = 16, kDfRest = 48, kDfProbe = 32;
    double df_tok[3] = {}, df_ms[3] = {}, pl_tok[3] = {}, pl_ms[3] = {};
    int df_n[3] = {}, df_tlast = 1;
    auto band_of = [](size_t pos) { return pos < 8192 ? 0 : pos < 65536 ? 1 : 2; };
    long long df_rests = 0;
    std::vector<int32_t> dbuf((size_t) std::max(1, wmax));
    std::vector<int> wtoks((size_t) std::max(1, wmax));
    std::string line;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) continue;
        in.stop.store(false);
        std::stringstream ss(line);
        std::string w;
        ss >> w;
        int max_new = 0;
        ss >> max_new;
        Sampling sp;
        int think_budget = 0, think_end = -1;   // (the server's thinking budget: </think> forced after that many)
        int lk_on = lk_min > 0 && wmax >= 2 ? 1 : 0, wcap = wmax;
        int df_on = df_have ? 1 : 0;   // (dfly=0|1; spec=1: no drafting at all, spec=N: windows of N at most)
        // (lookup_trust=0|1, server.py's strata_tune key: a long trusted lookup match verified without DFly2's draft;
        //  HY4_LOOKUP_TRUST, 0: on code edits trusted lookups took the place of DFly2's longer windows - a first
        //  measurement, its order not rotated; see test-results/2026-11-18-hy4/NOTES.md)
        static const int trust_env = std::getenv("HY4_LOOKUP_TRUST") ? std::atoi(std::getenv("HY4_LOOKUP_TRUST")) : 0;
        int lk_trust = trust_env != 0;
        std::vector<std::string> fields;
        while (ss >> w) fields.push_back(w);
        if (fields.empty()) continue;
        const std::vector<int> ids = parse_ids(fields.back());
        fields.pop_back();
        for (const auto& kv : fields) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
            if (k == "temperature") sp.temperature = std::stof(v);
            else if (k == "top_p") sp.top_p = std::stof(v);
            else if (k == "top_k") sp.top_k = std::atoi(v.c_str());
            else if (k == "min_p") sp.min_p = std::stof(v);
            else if (k == "seed") sp.seed = std::atoll(v.c_str());
            else if (k == "think_budget") think_budget = std::atoi(v.c_str());
            else if (k == "think_end") think_end = std::atoi(v.c_str());
            else if (k == "lookup" || k == "suffix_draft")   // (suffix_draft: server.py's strata_tune key)
                lk_on = lk_min > 0 && wmax >= 2 && std::atoi(v.c_str()) != 0 ? 1 : 0;
            else if (k == "window") wcap = std::max(1, std::min(wmax, std::atoi(v.c_str())));
            else if (k == "dfly") df_on = df_have && std::atoi(v.c_str()) != 0 ? 1 : 0;
            else if (k == "lookup_trust") lk_trust = std::atoi(v.c_str()) != 0;
            else if (k == "spec") {   // (server.py's strata_tune spec: 1 the plain loop, N a window's rows at most)
                const int n = std::atoi(v.c_str());
                if (n == 1) df_on = lk_on = 0;
                else if (n >= 2) wcap = std::min(wmax, n);
            }
        }
        bool in_think = think_budget > 0 && think_end >= 0 && think_end < V;
        if (sp.seed >= 0) rng.seed((uint64_t) sp.seed);
        // the prompt: what continues hist is read on from there, what continues the last prompt's end from there,
        // anything else from the start
        size_t reuse = 0;
        if (ids.size() > hist.size() && std::equal(hist.begin(), hist.end(), ids.begin())) reuse = hist.size();
        else if (ids.size() == hist.size() && ids == hist && !last.empty()) reuse = hist.size();
        else if (have_snap && !snap_hist.empty() && ids.size() >= snap_hist.size() &&
                 std::equal(snap_hist.begin(), snap_hist.end(), ids.begin()) &&
                 (ids.size() > snap_hist.size() || !snap_last.empty()) && eng.restore()) {
            hist = snap_hist;
            last = snap_last;
            reuse = hist.size();
        }
        if (reuse < hist.size() || (reuse == 0 && !hist.empty())) {
            eng.reset();
            hist.clear();
            last.clear();
            have_snap = false;   // (reset() drops the engine's snapshot)
            reuse = 0;
        }
        if ((int) ids.size() >= eng.max_ctx()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%d)\n", ids.size(), eng.max_ctx());
            std::printf("DONE 0 %zu 0 0 length 0 0 0\n", ids.size());
            std::fflush(stdout);
            continue;
        }
        const double p0 = now_ms();
        const float* lg = last.empty() ? nullptr : last.data();
        const size_t todo = ids.size() - reuse;
        // the prompt's new tokens in pieces of nearly equal size, at most a chunk each (STOP is looked at between them)
        const size_t pieces = (todo + (size_t) opt.chunk - 1) / (size_t) opt.chunk;
        const size_t per = pieces ? (todo + pieces - 1) / pieces : 0;
        for (size_t i = 0; i < todo; i += per) {
            const size_t m = std::min(per, todo - i);
            lg = eng.prefill(ids.data() + reuse + i, (int) m);
            hist.insert(hist.end(), ids.begin() + (long) (reuse + i), ids.begin() + (long) (reuse + i + m));
            const double ms = now_ms() - p0;
            std::printf("PP %zu %zu %.0f %.1f %zu\n", i + m, todo, ms, (i + m) * 1000.0 / std::max(ms, 1.0), reuse);
            std::fflush(stdout);
            if (!lg || in.stop.load()) break;
        }
        // the prompt read whole: its end kept (the next turn's prompt holds this one and the answer, perhaps rendered
        // apart from the tokens written now)
        if (lg && hist.size() == ids.size() && !in.stop.load() && eng.snapshot()) {
            snap_hist = hist;
            snap_last.assign(lg, lg + V);
            have_snap = true;
        }
        const double p1 = now_ms();
        int generated = 0;
        std::string finish = "length";
        if (in.stop.load()) finish = "cancel";
        // the tokens: each sampled from the logits after the ones before it (lg: after hist), printed, then fed - a
        // round feeds the last one and, where the policy takes them, the lookup's drafts after it in one window, and
        // keeps the drafts while the rows sampled equal them
        long long offered = 0, accepted = 0;
        long long rounds = 0, wins[17] = {};   // (the rounds, the windows by their size: the policy's choices)
        long long df_off = 0, df_acc = 0, lk_off = 0, lk_acc = 0;   // (the drafts verified and kept, by their source)
        long long df_rounds = 0, lk_trusted = 0, df_restr = 0;   // (rounds that drafted, trusted lookups, rested ones)
        long long df_refill = 0;                                  // (... and rounds in a refill, no drafting)
        int df_rest = 0;
        // a token sampled: the thinking budget's forcing, printed, counted; then whether the request ends at it - 1
        // unfed (its end token, STOP, the context full), 2 fed (max_new: fed as the plain loop did), 0 goes on
        auto emit = [&](int t, int& end) -> int {
            if (in_think) {   // (the reasoning closed by the model, or at the budget by the engine)
                if (t != think_end && generated >= think_budget) t = think_end;
                if (t == think_end) in_think = false;
            }
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++generated;
            if (lk_on) sfx.append((int32_t) t);   // (the drafter's history: every printed token, fed or about to be)
            end = 0;
            if (t == eos) {
                finish = "stop";
                end = 1;
            } else if (in.stop.load()) {
                finish = "cancel";
                end = 1;
            } else if ((int) hist.size() + 1 >= eng.max_ctx()) {   // (hist: the fed tokens, pos() but in a window)
                end = 1;
            } else if (generated >= max_new) {
                end = 2;
            }
            return t;
        };
        if (lk_on) sfx.reset(), sfx.append(hist.data(), hist.size());
        // a round's tokens and time into DFly2's rest (above): a drafted round's or a plain one's (lookup windows
        // count for neither); drafting worse over kDfJudge rounds - a rest
        auto df_judge = [&](bool drafted, int tok_n, double ms) {
            if (!df_on) return;
            const int b = band_of(hist.size());
            if (drafted) {
                df_tok[b] = 0.95 * df_tok[b] + tok_n;
                df_ms[b] = 0.95 * df_ms[b] + ms;
                if (++df_n[b] >= kDfJudge && pl_ms[b] > 0 && df_tok[b] / df_ms[b] < pl_tok[b] / pl_ms[b]) {
                    df_rest = kDfRest;
                    df_n[b] = 0;
                    df_tok[b] = df_ms[b] = 0;
                    ++df_rests;
                }
            } else if (tok_n == 1) {
                pl_tok[b] = 0.9 * pl_tok[b] + 1;
                pl_ms[b] = 0.9 * pl_ms[b] + ms;
            }
        };
        int end = finish == "cancel" || max_new <= 0 || !lg ? 1 : 0;
        int t = end ? -1 : emit(sample(lg, V, sp, rng), end);
        while (end == 0) {
            // tokens this round may feed: t and the drafts after it - a window's T rows print up to T tokens (max_new's
            // last then the bonus row's, fed after the loop) and take positions before the context's last
            const int room = std::min({wcap, max_new - generated, eng.max_ctx() - 1 - (int) hist.size()});
            const double r0 = now_ms();
            int match = 0, kd = 0;
            if (lk_on && room >= 2) {
                policy.set_context((long long) hist.size());   // (the round costs by depth: < 8K, < 64K, longer)
                kd = sfx.propose(room - 1, dbuf.data());
                match = sfx.last_match();
            }
            // the window: DFly2's drafts by the cost-aware window, the lookup's where the policy expects more of it
            // (maya-ds4's round: a lookup whose first token is the drafter's), or the lookup's alone without DFly2
            int T = 1, nd = -1;
            bool from_lk = false, trusted = false;
            bool draft = df_on && room >= 2;
            // (no drafting while the caches' refill after a prompt runs: every round is slow then, windows the most,
            //  and its round times would stand for this depth's afterwards - nor do those rounds judge the rest)
            const bool refill = df_on && eng.refilling();
            if (refill) draft = false, ++df_refill;
            if (draft && df_rest > 0) {   // (resting: drafting has paid less than plain rounds)
                --df_rest;
                ++df_restr;
                draft = false;
            } else if (draft && rounds % kDfProbe == kDfProbe - 1) {
                draft = false;   // (a plain round now and then: the rate drafting is judged against)
            }
            if (draft && lk_trust && kd > 0 && policy.trust(df_tlast, kd, match)) {   // (a long trusted match: verified alone)
                const strata::spec::DraftPolicy::Pick pk = policy.choose(df_tlast, kd, match);
                if (pk.lookup && pk.t >= 2) T = pk.t, from_lk = trusted = true, draft = false, ++lk_trusted;
            }
            if (draft) nd = eng.dfly_draft(t, room - 1, ddr, dconf);
            if (nd >= 0) ++df_rounds;
            if (nd > 0) {
                dcost.set_context((long long) hist.size());
                double q[16];
                dcost.chain(dconf, nd, q);
                T = dcost.window(q, nd, room);
                df_tlast = T;
                if (kd > 0 && (int) dbuf[0] == ddr[0]) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, kd, match);
                    if (pk.lookup) T = pk.t, from_lk = true;
                }
            } else if (kd > 0 && !trusted) {
                const strata::spec::DraftPolicy::Pick pk = policy.choose(1, kd, match);
                if (pk.lookup && pk.t >= 2) T = pk.t, from_lk = true;
            }
            ++rounds;
            if (T >= 2) {
                wtoks[0] = t;
                for (int i = 1; i < T; ++i) wtoks[(size_t) i] = from_lk ? (int) dbuf[(size_t) (i - 1)] : ddr[i - 1];
                const float* lw = eng.step_window(wtoks.data(), T);
                if (!lw && !eng.failed()) {   // (no window after all: plain steps for the request)
                    lk_on = df_on = 0;
                    continue;
                }
                if (!lw) break;   // (a card gave up: failed(), below)
                hist.push_back(t);
                int kept = 1;
                for (int i = 0; i < T; ++i) {
                    const int s = emit(sample(lw + (size_t) i * V, V, sp, rng), end);
                    lg = lw + (size_t) i * V;   // (the logits after hist: row kept - 1)
                    if (i < T - 1 && s == wtoks[(size_t) i + 1] && end == 0) {
                        ++kept;
                        hist.push_back(s);
                        continue;
                    }
                    t = s;   // (the round's last: fed by the next round)
                    break;
                }
                eng.window_keep(kept);
                offered += T - 1;
                accepted += kept - 1;
                ++wins[T];
                const double ms = now_ms() - r0;   // (the round's time: the drafting, the window, the sampling)
                if (from_lk) lk_off += T - 1, lk_acc += kept - 1;
                else df_off += T - 1, df_acc += kept - 1;
                if (nd >= 0) dcost.observe_cost(T, ms);
                if (nd > 0 && !from_lk) dcost.observe_drafts(dconf, T - 1, kept - 1);
                // (a trusted round's time with a draft's added: the policy's costs stay those of drafted rounds)
                if (lk_on) policy.observe(from_lk, T, kept - 1, match, trusted ? ms + eng.dfly_ms() : ms);
                if (!refill) df_judge(nd >= 0, kept, ms);
                continue;
            }
            lg = eng.step(t);
            hist.push_back(t);
            if (!lg) break;
            ++wins[1];
            t = emit(sample(lg, V, sp, rng), end);
            const double ms = now_ms() - r0;   // (after the sampling, as a window's)
            if (nd >= 0) dcost.observe_cost(1, ms);   // (drafted, none verified)
            if (lk_on) policy.observe(false, 1, 0, match, ms);
            if (!refill) df_judge(nd >= 0, 1, ms);
        }
        if (end == 2 && !eng.failed()) {   // (max_new's token fed, as the plain loop did: the logits after it kept)
            lg = eng.step(t);
            hist.push_back(t);
        }
        if (eng.failed()) {   // (a card gave up waiting for the other: this conversation's state is gone)
            // (serve/server.py keys on "the conversation was dropped" - DROPPED, dropped_state(): keep the phrase)
            std::printf("ERR maya-hy4: a card gave up waiting for the other (the device's routing); the conversation "
                        "was dropped\n");
            std::printf("DONE %d %zu %.1f %.1f length %lld %lld %zu\n", generated, ids.size(), p1 - p0, now_ms() - p1,
                        accepted, offered, reuse);
            std::fflush(stdout);
            eng.reset();
            hist.clear();
            last.clear();
            have_snap = false;
            eng.clear_stats();
            continue;
        }
        if (lg && finish != "stop") last.assign(lg, lg + V);
        else last.clear();
        // (a generated end token is not fed: the next prompt carries the turn's end itself)
        const auto st = eng.stats();
        std::printf("STAT hy4 cache_hit=%.3f cache_slots=%ld cache_copies=%ld helper_hit=%.3f helper_slots=%ld "
                    "helper_copies=%ld tok_s=%.2f\n",
                    st.lookups ? (double) st.hits / st.lookups : 0.0, st.cache_slots, st.cache_copies,
                    st.lookups ? (double) st.hits_helper / st.lookups : 0.0, st.helper_slots, st.helper_copies,
                    st.ms_total > 0 ? st.tokens * 1000.0 / st.ms_total : 0.0);
        if (lk_on || df_on) {   // (the rounds: the windows by size, the drafts kept of those verified, by source)
            std::string ws;
            for (int i = 1; i <= 16; ++i)
                if (wins[i]) ws += " " + std::to_string(i) + ":" + std::to_string(wins[i]);
            std::fprintf(stderr,
                         "maya-hy4: drafts %lld of %lld kept (DFly2 %lld of %lld, lookup %lld of %lld), %lld rounds for "
                         "%d tokens, windows%s; DFly2 drafted in %lld rounds, rested %lld (%lld rests so far), "
                         "%lld in a refill; %lld trusted lookups\n",
                         accepted, offered, df_acc, df_off, lk_acc, lk_off, rounds, generated, ws.c_str(), df_rounds,
                         df_restr, df_rests, df_refill, lk_trusted);
            if (df_on) std::fprintf(stderr, "maya-hy4: DFly2's round costs: %s\n", dcost.cost_text().c_str());
        }
        std::printf("DONE %d %zu %.1f %.1f %s %lld %lld %zu\n", generated, ids.size(), p1 - p0, now_ms() - p1,
                    finish.c_str(), accepted, offered, reuse);
        std::fflush(stdout);
        eng.clear_stats();
        eng.save_profile();   // (the scores for the next start)
    }
    eng.save_profile();
    return 0;
}
