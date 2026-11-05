// src/ling/ling_main.cpp - maya-ling: inclusionAI's Ling-3.1-flash from a GGUF on one NVIDIA GPU and the CPU.
//
//   maya-ling --gguf <...-00001-of-0000N.gguf> --serve [--max-context N] [--threads N] [--gpu N] [--profile FILE]
//
// --serve speaks Maya's engine protocol (serve/server.py's StrataEngine): INFO lines, then `READY <context> stop`; per
// request `GEN <max_new> [key=value ...] <id,id,...>` answered by `PP <read> <total> <ms> <tok/s> <reused>` while the
// prompt is read, a `T <id>` line a token, and `DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel>
// 0 0 <reused>`; `STOP` ends a request early, `QUIT` the engine.  A prompt that continues the last conversation (its
// tokens so far, the generated ones included) reads only the rest: the KDA layers' state cannot go back, so anything
// else starts over.  Keys: temperature, top_p, top_k, min_p, seed.  A request ends at the GGUF's eos (<|role_end|>).
//
//   maya-ling --gguf <...> --ids @prompt.txt [--gen N | --follow @tokens.txt] [--top N] [--dump-logits out.bin]
//
// The prompt goes through the prompt path (Engine::prefill: chunks of --chunk tokens, default 8192, on the GPU with the
// routed experts streamed to it through --stage VRAM slots); --step reads it a token at a time instead (and prompts
// under LING_MIN_CHUNK tokens, 256, take the token path anyway).
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt, each
// printed with the top --top (default 5) as id:log-probability (for a comparison with llama.cpp's).  --follow: the
// given tokens fed instead of the greedy ones, each line ending with that token's own log-probability, so another
// engine's continuation is scored position by position.  --dump-logits: the logits after the prompt (float32, vocab).
//
//   maya-ling --gguf <...> --runs runs.txt    several such runs with one load: a line a run, "NAME pf|step @ids.txt
//                                             (@follow.txt | N)", the state reset before each, its lines between
//                                             "RUN NAME" and "END NAME ok|failed"
//
//   maya-ling --selftest      the KDA recurrence, the MLA decode attention and the routing against host references
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

#include "ling_kernels.cuh"
#include "ling_model.hpp"

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
    ling::Options opt;
    std::string ids_arg, dump, follow_arg, runs_arg;
    int gen = 0, top = 5;
    bool serve = false, by_step = false, selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "maya-ling: %s needs a value\n", a.c_str());
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
        else if (a == "--profile") opt.profile = val();
        else if (a == "--dump-logits") dump = val();
        else if (a == "--quiet") opt.verbose = false;
        else if (a == "--selftest") selftest = true;
        else {
            std::fprintf(stderr, "maya-ling: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (selftest) {
        const bool ok = ling::k::selftest();
        std::fprintf(stderr, "maya-ling selftest: %s\n", ok ? "all passed" : "FAILED");
        return ok ? 0 : 1;
    }
    if (opt.gguf.empty()) {
        std::fprintf(stderr, "usage: maya-ling --gguf <first shard> (--serve | --ids @file [--gen N]) | --selftest\n");
        return 2;
    }
    ling::Engine eng;
    std::string err;
    if (!eng.load(opt, err)) {
        std::fprintf(stderr, "maya-ling: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        return 1;
    }
    const ling::Geo& g = eng.geo();
    const int V = g.vocab;
    const int eos = g.eos;   // <|role_end|> (the GGUF's tokenizer.ggml.eos_token_id)

    // One test run: the prompt (the prompt path, or the token path with by_step), then greedy tokens or the follow's,
    // each step's line "pick id:lp ..." (and " | pick:lp" when following); false: the prompt did not fit.
    auto run_one = [&](const std::vector<int>& ids, bool step_path, const std::vector<int>& follow, int n_gen,
                       const std::string& dump_to) -> bool {
        const double t0 = now_ms();
        const float* lg = nullptr;
        if (step_path)
            for (int id : ids) lg = eng.step(id);
        else
            for (size_t i = 0; i < ids.size(); i += (size_t) opt.chunk)
                lg = eng.prefill(ids.data() + i, (int) std::min(ids.size() - i, (size_t) opt.chunk));
        const double t1 = now_ms();
        const auto ps = eng.stats();
        std::fprintf(stderr, "maya-ling: prompt %zu tokens in %.1f s (%.1f tok/s; %s, %.1f GB of experts copied)\n",
                     ids.size(), (t1 - t0) / 1000.0, ids.size() * 1000.0 / (t1 - t0),
                     step_path || (int) ids.size() < 256 ? "the token path" : "the prompt path", ps.pf_copy_gb);
        if (!lg) {
            std::fprintf(stderr, "maya-ling: the prompt did not fit\n");
            return false;
        }
        if (!dump_to.empty()) {
            std::ofstream f(dump_to, std::ios::binary);
            f.write((const char*) lg, (std::streamsize) V * sizeof(float));
        }
        eng.clear_stats();
        const int steps = follow.empty() ? n_gen : (int) follow.size();
        for (int n = 0; n < steps && lg; ++n) {
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
            lg = eng.step(pick);
        }
        const auto st = eng.stats();
        if (st.tokens > 0)
            std::fprintf(stderr,
                         "maya-ling: %ld tokens at %.2f tok/s (the CPU lane %.0f%% of the time; %.1f%% of the experts in "
                         "VRAM, %ld slots, %ld copies)\n",
                         st.tokens, st.tokens * 1000.0 / st.ms_total, 100.0 * st.ms_cpu / st.ms_total,
                         st.lookups ? 100.0 * st.hits / st.lookups : 0.0, st.cache_slots, st.cache_copies);
        return true;
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
            std::fprintf(stderr, "maya-ling: no prompt ids\n");
            return 2;
        }
        const std::vector<int> follow = follow_arg.empty() ? std::vector<int>() : parse_ids(follow_arg);
        const bool ok = run_one(ids, by_step, follow, gen, dump);
        eng.save_profile();
        return ok ? 0 : 1;
    }

    // --serve
    std::printf("INFO engine=maya-ling-%s model=Ling-3.1-flash layers=%d experts=%d vocab=%d\n", kVersion, g.n_layer,
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
        while (finish != "cancel" && generated < max_new && lg) {
            int t = sample(lg, V, sp, rng);
            if (in_think) {   // (the reasoning closed by the model, or at the budget by the engine)
                if (t != think_end && generated >= think_budget) t = think_end;
                if (t == think_end) in_think = false;
            }
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++generated;
            if (t == eos) {
                finish = "stop";
                break;
            }
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (eng.pos() + 1 >= eng.max_ctx()) break;
            lg = eng.step(t);
            hist.push_back(t);
        }
        if (lg && finish != "stop") last.assign(lg, lg + V);
        else last.clear();
        // (a generated end token is not fed: the next prompt carries the turn's end itself)
        const auto st = eng.stats();
        std::printf("STAT ling cache_hit=%.3f cache_slots=%ld cache_copies=%ld\n",
                    st.lookups ? (double) st.hits / st.lookups : 0.0, st.cache_slots, st.cache_copies);
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 %zu\n", generated, ids.size(), p1 - p0, now_ms() - p1, finish.c_str(),
                    reuse);
        std::fflush(stdout);
        eng.clear_stats();
        eng.save_profile();   // (the scores for the next start)
    }
    eng.save_profile();
    return 0;
}
