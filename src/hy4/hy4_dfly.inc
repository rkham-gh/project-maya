// src/hy4/hy4_dfly.inc - DFly2, Tencent's drafter for Hy4 Flash (model.dfly_layers.* of the checkpoint's last shard,
// model-00148-of-00148.safetensors): included by hy4_model.cu, members of Engine::Impl.
//
// The reference is vLLM's (PR 68296: vllm/models/hy_v4/nvidia/dfly.py, dfly_speculator.py, dfly_inputs.py,
// dfly_target_reuse.py). A draft is a block of 6 rows at positions P .. P + 5: row 0 the newest token (the one the
// target sampled and has not fed yet, at P), rows 1..5 the mask token (120009). Row 0's embedding is fused with the
// target's final hidden of the row before it (mtp_fusion, "slot0": e + gate * eh_proj(cat(enorm(e), hnorm(h))));
// then 5 dense layers (pre-norm residual: gated MLA + post_proj_norm, SiLU MLP), final_norm. Draft layer i shares
// target layer 25 + 4 i's cache and index selection: its rows write their own latent rows (kv_a, kv_a_norm, rope) into
// that layer's cache at P + t, and attend - causally - to the target's selection at the row before P (every key when
// the context is short) and the block's rows up to their own, through the draft layer's own kv_b (k_b / v_b here).
// The drafts come one at a time: draft k (the token at P + k) = argmax lm_head(h_k + hidden_correction(h_k,
// embed(token at P + k - 1))), the target's head. The cache rows past P that the drafts wrote are rewritten when the
// target decodes those positions (its verify window), before anything reads them.
//
// Here the matrices are Q8_0 (0.9 GB in VRAM; the checkpoint's BF16, quantized at the load), the drafts greedy (the
// serve loop verifies by sampling each row and keeping a draft while it equals the row's token, exact for any
// deterministic draft), the confidence of a draft its softmax probability (DraftPolicy's cost-aware window calibrates
// it). vLLM ignores the checkpoint's confidence_head; so does this.

namespace {
// The safetensors file: its header's tensors (name -> dtype, shape, data) over the file read whole
struct SafeT {
    std::vector<uint8_t> buf;
    size_t base = 0;
    struct T {
        std::string dtype;
        std::vector<int64_t> shape;
        size_t off = 0, end = 0;
    };
    std::map<std::string, T> ts;
    bool open(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        const size_t sz = (size_t) f.tellg();
        f.seekg(0);
        buf.resize(sz);
        if (!f.read((char*) buf.data(), (std::streamsize) sz) || sz < 8) {
            err = "cannot read " + path;
            return false;
        }
        uint64_t n = 0;
        std::memcpy(&n, buf.data(), 8);
        if (8 + n > sz) {
            err = path + ": not a safetensors file";
            return false;
        }
        base = 8 + (size_t) n;
        const std::string h((const char*) buf.data() + 8, (size_t) n);
        // {"name":{"dtype":"BF16","shape":[a,b],"data_offsets":[s,e]}, ...}: the objects are flat
        size_t i = 0;
        while ((i = h.find('"', i)) != std::string::npos) {
            const size_t j = h.find('"', i + 1);
            if (j == std::string::npos) break;
            const std::string name = h.substr(i + 1, j - i - 1);
            size_t k = j + 1;
            while (k < h.size() && (h[k] == ':' || h[k] == ' ')) ++k;
            if (k >= h.size() || h[k] != '{') {
                i = j + 1;
                continue;
            }
            const size_t e = h.find('}', k);
            if (e == std::string::npos) break;
            const std::string obj = h.substr(k, e - k);
            i = e + 1;
            if (name == "__metadata__") continue;
            T t;
            const size_t dt = obj.find("\"dtype\"");
            if (dt != std::string::npos) {
                const size_t a = obj.find('"', obj.find(':', dt) + 1), b = obj.find('"', a + 1);
                t.dtype = obj.substr(a + 1, b - a - 1);
            }
            auto nums = [&](const char* key) {
                std::vector<int64_t> v;
                const size_t p = obj.find(key);
                if (p == std::string::npos) return v;
                const size_t a = obj.find('[', p), b = obj.find(']', a);
                std::stringstream ss(obj.substr(a + 1, b - a - 1));
                std::string w;
                while (std::getline(ss, w, ',')) v.push_back(std::atoll(w.c_str()));
                return v;
            };
            t.shape = nums("\"shape\"");
            const std::vector<int64_t> od = nums("\"data_offsets\"");
            if (od.size() == 2) t.off = (size_t) od[0], t.end = (size_t) od[1];
            ts[name] = t;
        }
        return true;
    }
    // BF16 values of a tensor of this shape, or nullptr (err set)
    const uint16_t* bf16(const std::string& name, std::initializer_list<int64_t> shape, std::string& err) const {
        auto it = ts.find(name);
        if (it == ts.end()) {
            err = "the drafter has no " + name;
            return nullptr;
        }
        const T& t = it->second;
        if (t.dtype != "BF16" || t.shape != std::vector<int64_t>(shape)) {
            err = name + ": not BF16 of the expected shape";
            return nullptr;
        }
        size_t n = 1;
        for (int64_t d : shape) n *= (size_t) d;
        if (t.end - t.off != n * 2 || base + t.end > buf.size()) {
            err = name + ": its bytes do not match its shape";
            return nullptr;
        }
        return (const uint16_t*) (buf.data() + base + t.off);
    }
};
inline float bf16f(uint16_t v) {
    const uint32_t u = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
}  // namespace

// rows n of k values (value(i), i = row k + column) -> Q8_0 in VRAM (as mat()'s to_q8: llama.cpp's quantize_row_q8_0_ref)
template <class F> void Engine::Impl::q8_upload(F value, int k, int n, DMat& m) {
    const size_t nb = (size_t) k / 32, rbq = nb * 34;
    std::vector<uint8_t> h(rbq * (size_t) n);
    const int nth = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            for (size_t r = (size_t) w; r < (size_t) n; r += (size_t) nth)
                for (size_t b = 0; b < nb; ++b) {
                    const size_t i0 = r * (size_t) k + b * 32;
                    float amax = 0.0f;
                    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(value(i0 + j)));
                    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
                    uint8_t* o = h.data() + r * rbq + b * 34;
                    const __half hd = __float2half(d);
                    std::memcpy(o, &hd, 2);
                    for (int j = 0; j < 32; ++j) ((int8_t*) (o + 2))[j] = (int8_t) std::lround(value(i0 + j) * id);
                }
        });
    for (auto& x : th) x.join();
    m.type = 8;
    m.k = k;
    m.n = n;
    m.rb = rbq;
    uint8_t* d = dalloc<uint8_t>(h.size());
    CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    m.w = d;
}

bool Engine::Impl::dfly_load(const std::string& path, std::string& err) {
    const double t0 = now_ms();
    const int E = g.n_embd, H = kDflH, QK = g.qk_head, NOPE = g.qk_head - g.rope_dim, VH = g.v_head, KL = g.kv_lora,
              QL = g.q_lora, FF = kDflFF;
    // the target's geometry the drafter is made for (the sparse layers': 32 heads of 192 + 64, v 256, latent 512)
    if (E != 3072 || QK != 256 || NOPE != 192 || VH != 256 || KL != 512 || QL != 1024 || g.vocab <= kDflMask) {
        err = "DFly2: the model's attention geometry is not Hy4 Flash's";
        return false;
    }
    for (int i = 0; i < kDflL; ++i) {
        const int il = kDflTgt0 + kDflTgtStep * i;
        if (il >= g.n_layer || !L[(size_t) il].idx || L[(size_t) il].swa || L[(size_t) il].H != H) {
            err = "DFly2: target layer " + std::to_string(il) + " is not a sparse layer of 32 heads";
            return false;
        }
    }
    SafeT st;
    if (!st.open(path, err)) return false;
    const std::string P0 = "model.dfly_layers.";
    auto vec = [&](const std::string& name, int64_t n) -> float* {
        const uint16_t* v = st.bf16(P0 + name, {n}, err);
        if (!v) return nullptr;
        std::vector<float> h((size_t) n);
        for (int64_t i = 0; i < n; ++i) h[(size_t) i] = bf16f(v[i]);
        float* d = dalloc<float>((size_t) n);
        CK(cudaMemcpy(d, h.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice));
        return d;
    };
    auto mq = [&](const std::string& name, int64_t n, int64_t k, DMat& m) -> bool {   // a [n out][k in] weight
        const uint16_t* v = st.bf16(P0 + name, {n, k}, err);
        if (!v) return false;
        static const int bf = std::getenv("HY4_DFLY_BF16") ? std::atoi(std::getenv("HY4_DFLY_BF16")) : 0;
        if (bf) {   // (HY4_DFLY_BF16=1, a check: the checkpoint's BF16 kept - every GEMV a column at a time, F32 input;
                    //  2: Q8_0's values as BF16 - Q8_0's weights with F32 activations)
            m.type = 30;
            m.k = (int) k;
            m.n = (int) n;
            m.rb = (size_t) k * 2;
            std::vector<uint16_t> h(v, v + (size_t) n * k);
            if (bf == 2)
                for (size_t b0 = 0; b0 < h.size(); b0 += 32) {
                    float amax = 0.0f;
                    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(bf16f(h[b0 + j])));
                    const float dq = __half2float(__float2half(amax / 127.0f)), id = dq != 0.0f ? 1.0f / dq : 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const float q = (float) std::lround(bf16f(h[b0 + j]) * id) * dq;
                        uint32_t u;
                        std::memcpy(&u, &q, 4);
                        h[b0 + j] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
                    }
                }
            uint8_t* d = dalloc<uint8_t>(h.size() * 2);
            CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
            m.w = d;
            return true;
        }
        q8_upload([v](size_t i) { return bf16f(v[i]); }, (int) k, (int) n, m);
        return true;
    };
    dfl.assign((size_t) kDflL, DflLayer{});
    for (int i = 0; i < kDflL; ++i) {
        DflLayer& d = dfl[(size_t) i];
        const std::string b = std::to_string(i) + ".";
        d.tgt = kDflTgt0 + kDflTgtStep * i;
        if (!(d.attn_norm = vec(b + "input_layernorm.weight", E)) ||
            !(d.ffn_norm = vec(b + "post_attention_layernorm.weight", E)) ||
            !(d.post_proj = vec(b + "self_attn.post_proj_norm.weight", E)) ||
            !(d.q_a_norm = vec(b + "self_attn.q_a_layernorm.weight", QL)) ||
            !(d.kv_a_norm = vec(b + "self_attn.kv_a_layernorm.weight", KL)) ||
            !(d.sinks = vec(b + "self_attn.learnable_sink_param", H)) ||
            !mq(b + "self_attn.q_a_proj.weight", QL, E, d.q_a) || !mq(b + "self_attn.q_b_proj.weight", H * QK, QL, d.q_b) ||
            !mq(b + "self_attn.kv_a_proj_with_mqa.weight", KL + g.rope_dim, E, d.kv_a) ||
            !mq(b + "self_attn.linear_gate.weight", H * VH, E, d.wgate) ||
            !mq(b + "self_attn.o_proj.weight", E, H * VH, d.wo) || !mq(b + "mlp.gate_proj.weight", FF, E, d.gate) ||
            !mq(b + "mlp.up_proj.weight", FF, E, d.up) || !mq(b + "mlp.down_proj.weight", E, FF, d.down))
            return false;
        // kv_b [H (192 + 256)][512]: k_b a head's 512 rows of 192 (row c: W_UK[h][0..192)[c]) and v_b its 256 rows of
        // 512 - the layouts of the GGUF's attn_k_b / attn_v_b (mv_strided a head at a time)
        const uint16_t* kvb = st.bf16(P0 + b + "self_attn.kv_b_proj.weight", {(int64_t) H * (NOPE + VH), KL}, err);
        if (!kvb) return false;
        q8_upload([=](size_t idx) {
            const size_t r = idx / (size_t) NOPE, j = idx % (size_t) NOPE, h = r / (size_t) KL, c = r % (size_t) KL;
            return bf16f(kvb[(h * (size_t) (NOPE + VH) + j) * (size_t) KL + c]);
        }, NOPE, KL * H, d.k_b);
        q8_upload([=](size_t idx) {
            const size_t r = idx / (size_t) KL, c = idx % (size_t) KL, h = r / (size_t) VH, j = r % (size_t) VH;
            return bf16f(kvb[(h * (size_t) (NOPE + VH) + (size_t) NOPE + j) * (size_t) KL + c]);
        }, KL, VH * H, d.v_b);
        L[(size_t) d.tgt].dfl = i;
    }
    if (!(dfl_final = vec("final_norm.weight", E)) || !(dfl_enorm = vec("mtp_fusion.enorm.weight", E)) ||
        !(dfl_hnorm = vec("mtp_fusion.hnorm.weight", E)) || !(dfl_mgate = vec("mtp_fusion.gate", E)) ||
        !(dfl_c_enorm = vec("hidden_correction.embed_norm.weight", E)) ||
        !(dfl_c_hnorm = vec("hidden_correction.hidden_norm.weight", E)) ||
        !mq("mtp_fusion.eh_proj.weight", E, 2 * E, dfl_eh) || !mq("hidden_correction.gate_proj.weight", E, 2 * E, dfl_cg) ||
        !mq("hidden_correction.up_proj.weight", E, 2 * E, dfl_cu) || !mq("hidden_correction.down_proj.weight", E, E, dfl_cd))
        return false;
    {   // the confidence head (3072 -> 1, BF16; its logit read back beside each draft's probability)
        const uint16_t* w = st.bf16(P0 + "confidence_head.proj.weight", {1, E}, err);
        const uint16_t* bb = st.bf16(P0 + "confidence_head.proj.bias", {1}, err);
        if (!w || !bb) return false;
        dfl_ch.type = 30;
        dfl_ch.k = E;
        dfl_ch.n = 1;
        dfl_ch.rb = (size_t) E * 2;
        uint8_t* d = dalloc<uint8_t>((size_t) E * 2);
        CK(cudaMemcpy(d, w, (size_t) E * 2, cudaMemcpyHostToDevice));
        dfl_ch.w = d;
        dfl_ch_b = bf16f(bb[0]);
    }
    // the source row's state (the token path's and the windows' rows: up to 8) and the pass's buffers
    const int T = kDflT, R = 8;
    dfl_cn = (size_t) g.idx_top_k + g.idx_kpool;
    dfl_ccap = dfl_cn + (size_t) T;
    dfl_hid = dalloc<float>((size_t) R * E);
    dfl_scells = dalloc<int>((size_t) R * kDflL * dfl_cn);
    dfl_sncell = dalloc<int>((size_t) R * kDflL);
    dd_shid = dalloc<float>((size_t) E);
    dd_scells = dalloc<int>((size_t) kDflL * dfl_cn);
    dd_sncell = dalloc<int>((size_t) kDflL);
    dd_x = dalloc<float>((size_t) T * E);
    dd_n = dalloc<float>((size_t) T * E);
    dd_h = dalloc<float>((size_t) T * E);
    dd_qa = dalloc<float>((size_t) T * QL);
    dd_qn = dalloc<float>((size_t) T * QL);
    dd_kva = dalloc<float>((size_t) T * (KL + g.rope_dim));
    dd_gh = dalloc<float>((size_t) T * H * VH);
    dd_qf = dalloc<float>((size_t) T * H * QK);
    dd_qabs = dalloc<float>((size_t) T * H * KL);
    dd_mo = dalloc<float>((size_t) T * H * KL);
    dd_mob = dalloc<float>((size_t) T * H * VH);
    dd_ao = dalloc<float>((size_t) T * E);
    dd_fg = dalloc<float>((size_t) T * FF);
    dd_fu = dalloc<float>((size_t) T * FF);
    dd_fh = dalloc<float>((size_t) T * FF);
    dd_fo = dalloc<float>((size_t) T * E);
    dd_cat = dalloc<float>((size_t) 2 * E);
    dd_hp = dalloc<float>((size_t) E);
    dd_e = dalloc<float>((size_t) E);
    dd_logits = dalloc<float>((size_t) g.vocab);
    dd_amx = dalloc<float>((size_t) 3 * 128);
    // the drafts' head: the target's, or (HY4_DFLY_VOCAB=<n>) its first n rows - a draft vocabulary of the n lowest ids
    // (the tokenizer's merges' order: the most frequent first), argmax over those alone
    dfl_head = output;
    if (const char* v = std::getenv("HY4_DFLY_VOCAB"))
        if (std::atoi(v) > 0 && std::atoi(v) < g.vocab) dfl_head.n = std::atoi(v);
    const size_t nq = std::max({(size_t) T * FF, (size_t) T * H * QK, (size_t) T * H * VH, (size_t) 2 * E});
    dd_xq = dalloc<int8_t>(nq);
    dd_xd = dalloc<float>(nq / 32);
    dd_xq2 = dalloc<int8_t>((size_t) T * H * KL);
    dd_xd2 = dalloc<float>((size_t) T * H * KL / 32);
    dd_cells = dalloc<int>((size_t) kDflL * T * dfl_ccap);
    dd_ncell = dalloc<int>((size_t) kDflL * T);
    // in: [4 T] positions, then the block's tokens [T] (tok, the mask x 5); seq: tok then the drafts (each draft's
    // embedding read from it); out: the drafts [kDflN] (int), their probabilities and confidence-head logits (float)
    dd_in = dalloc<int>((size_t) 5 * T);
    dh_in = halloc<int>((size_t) 5 * T);
    dd_seq = dalloc<int>((size_t) T);
    dd_out = dalloc<float>((size_t) 3 * kDflN);
    dh_out = halloc<float>((size_t) 3 * kDflN);
    // the embedding table where the device reads its rows: in VRAM already, else a pinned copy of the mapping's
    // (each row read over the bus by the gather; no host copy between the drafts)
    if (embd_host) {
        const size_t tb = (size_t) g.vocab * embd.rb;
        void* hp = nullptr;
        if (cudaHostAlloc(&hp, tb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            err = "DFly2: no pinned copy of the embedding table (" + std::to_string(tb >> 20) + " MiB)";
            return false;
        }
        std::memcpy(hp, embd.w, tb);
        CK(cudaHostGetDevicePointer((void**) &dfl_embd, hp, 0));
    } else {
        dfl_embd = embd.w;
    }
    if (o.verbose)
        std::fprintf(stderr,
                     "maya-hy4: DFly2 drafter (%s): %d layers over target layers %d..%d, %s, loaded in %.1f s; the "
                     "embedding table %s (%.0f MiB)\n",
                     path.c_str(), kDflL, kDflTgt0, kDflTgt0 + kDflTgtStep * (kDflL - 1),
                     dfl[0].q_a.type == 8 ? "Q8_0" : "BF16", (now_ms() - t0) / 1000.0,
                     embd_host ? "pinned and mapped" : "in VRAM already", (double) g.vocab * embd.rb / 1048576.0);
    return true;
}

// The block's pass on the stream, from the source row's state (dd_shid, dd_scells, dd_sncell) and dh_in's positions and
// tokens: no host work and no syncs, so it is a graph (dfly_draft captures it at its first call), its end the drafts,
// their probabilities and confidence logits in dh_out.  dump (HY4_DFLY_DUMP, eager only): its inputs and outputs saved.
void Engine::Impl::dfl_pass(bool dump) {
    const int E = g.n_embd, T = kDflT, H = kDflH, QK = g.qk_head, NOPE = g.qk_head - g.rope_dim, VH = g.v_head,
              KL = g.kv_lora, QL = g.q_lora, FF = kDflFF, KVD = KL + g.rope_dim;
    const float eps = kDflEps;
    static const char* dpre = std::getenv("HY4_DFLY_DUMP");
    auto dsave = [&](const char* tag, const void* dev, size_t bytes) {
        std::vector<uint8_t> h(bytes);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
        std::ofstream f(std::string(dpre) + "." + tag, std::ios::binary);
        f.write((const char*) h.data(), (std::streamsize) bytes);
    };
    CK(cudaMemcpyAsync(dd_in, dh_in, (size_t) 5 * T * sizeof(int), cudaMemcpyHostToDevice, s));
    const int* dpos = dd_in;
    const int* btok = dd_in + 4 * T;
    CK(cudaMemcpyAsync(dd_seq, btok, sizeof(int), cudaMemcpyDeviceToDevice, s));   // (seq[0] = tok)
    k::embed_rows(dfl_embd, embd.type, embd.rb, btok, T, E, dd_x, s);
    if (dump) {
        const int P = dh_in[0];
        dsave("x0", dd_x, (size_t) T * E * sizeof(float));
        dsave("hid", dd_shid, (size_t) E * sizeof(float));
        for (int i = 0; i < kDflL; ++i) {
            int n = 0;
            CK(cudaMemcpy(&n, dd_sncell + i, sizeof(int), cudaMemcpyDeviceToHost));
            const std::string ti = std::to_string(i);
            dsave(("cells" + ti).c_str(), dd_scells + (size_t) i * dfl_cn, (size_t) n * sizeof(int));
            dsave(("kv" + ti).c_str(), L[(size_t) dfl[(size_t) i].tgt].kv, (size_t) P * 576 * sizeof(__half));
        }
        std::ofstream f(std::string(dpre) + ".meta");
        f << "tok " << dh_in[4 * T] << "\nP " << P << "\ntheta_scale " << L[(size_t) dfl[0].tgt].theta_scale << "\n";
    }
    k::dfly_cells(dd_scells, dd_sncell, dfl_cn, kDflL, dpos, T, dd_cells, dd_ncell, dfl_ccap, s);
    // row 0: mtp_fusion with the target's final hidden of the row before it
    k::rmsnorm(dd_x, dfl_enorm, dd_cat, 1, E, eps, s);
    k::rmsnorm(dd_shid, dfl_hnorm, dd_cat + E, 1, E, eps, s);
    k::q8(dd_cat, 1, 2 * E, dd_xq, dd_xd, s);
    mvw(dfl_eh, dd_cat, dd_xq, dd_xd, dd_fo, 1);
    k::gate_add(dd_x, dfl_mgate, dd_fo, E, s);
    for (int i = 0; i < kDflL; ++i) {
        const DflLayer& d = dfl[(size_t) i];
        Layer& tl = L[(size_t) d.tgt];
        k::norm_q8_rows(dd_x, d.attn_norm, eps, dd_n, dd_xq, dd_xd, E, T, s);
        {
            const DMat* ms[3] = {&d.q_a, &d.kv_a, &d.wgate};
            float* ys[3] = {dd_qa, dd_kva, dd_gh};
            mvw_many(ms, ys, 3, dd_n, dd_xq, dd_xd, T);
        }
        k::norm_q8_rows(dd_qa, d.q_a_norm, eps, dd_qn, dd_xq, dd_xd, QL, T, s);
        mvw(d.q_b, dd_qn, dd_xq, dd_xd, dd_qf, T);
        k::q8(dd_qf, T, H * QK, dd_xq, dd_xd, s);
        for (int t = 0; t < T; ++t)
            if (!iq::mv_strided(d.k_b.type, d.k_b.w, (size_t) KL * d.k_b.rb, d.k_b.rb, d.k_b.k, KL, H,
                                dd_xq + (size_t) t * H * QK, dd_xd + (size_t) t * H * QK / 32, (size_t) QK,
                                dd_qabs + (size_t) t * H * KL, (size_t) KL, s)) {
                std::fprintf(stderr, "maya-hy4: DFly2: no strided GEMV for k_b\n");
                std::abort();
            }
        for (int t = 0; t < T; ++t) {
            const int* dp = dpos + 4 * t;
            float* qft = dd_qf + (size_t) t * H * QK;
            k::mla_rope_q(qft, 1, H, QK, NOPE, 0, dp, tl.theta_scale, s);
            k::mla_kv_store(dd_kva + (size_t) t * KVD, d.kv_a_norm, eps, tl.kv, 1, 0, dp, tl.theta_scale, s, 0);
            k::mla_decode(dd_qabs + (size_t) t * H * KL, qft, QK, NOPE, tl.kv, dp, 0,
                          dd_cells + ((size_t) i * T + t) * dfl_ccap, dd_ncell + (size_t) i * T + t, d.sinks, H,
                          mla_scale, attn_nsp, attn_scr, dd_mo + (size_t) t * H * KL, dd_xq2 + (size_t) t * H * KL,
                          dd_xd2 + (size_t) t * H * KL / 32, s, 0);
        }
        for (int t = 0; t < T; ++t)
            if (!iq::mv_strided(d.v_b.type, d.v_b.w, (size_t) VH * d.v_b.rb, d.v_b.rb, d.v_b.k, VH, H,
                                dd_xq2 + (size_t) t * H * KL, dd_xd2 + (size_t) t * H * KL / 32, (size_t) KL,
                                dd_mob + (size_t) t * H * VH, (size_t) VH, s)) {
                std::fprintf(stderr, "maya-hy4: DFly2: no strided GEMV for v_b\n");
                std::abort();
            }
        k::head_gate(dd_mob, dd_gh, H * VH * T, 1, dd_mob, dd_xq, dd_xd, s);   // (sigmoid(linear_gate x) a value)
        mvw(d.wo, dd_mob, dd_xq, dd_xd, dd_ao, T);
        k::rmsnorm(dd_ao, d.post_proj, dd_n, T, E, eps, s);
        k::add(dd_x, dd_n, T * E, s);
        k::norm_q8_rows(dd_x, d.ffn_norm, eps, dd_n, dd_xq, dd_xd, E, T, s);
        {
            const DMat* ms[2] = {&d.gate, &d.up};
            float* ys[2] = {dd_fg, dd_fu};
            mvw_many(ms, ys, 2, dd_n, dd_xq, dd_xd, T);
        }
        k::swiglu(dd_fg, dd_fu, dd_fh, FF * T, 0.0f, s);
        k::q8(dd_fh, T, FF, dd_xq, dd_xd, s);
        mvw(d.down, dd_fh, dd_xq, dd_xd, dd_fo, T);
        k::add(dd_x, dd_fo, T * E, s);
    }
    k::rmsnorm(dd_x, dfl_final, dd_h, T, E, eps, s);
    if (dump) dsave("h", dd_h, (size_t) T * E * sizeof(float));
    // the drafts one at a time: row k's hidden corrected by the token before it (seq[k - 1]: its embedding gathered on
    // the device), the target's head, its argmax into seq[k] and its probability; the confidence head on the corrected
    // hidden
    int* dtok = (int*) dd_out;
    float* dprob = dd_out + kDflN;
    float* dch = dd_out + 2 * kDflN;
    for (int kk = 1; kk < T; ++kk) {
        k::embed_rows(dfl_embd, embd.type, embd.rb, dd_seq + (kk - 1), 1, E, dd_e, s);
        k::rmsnorm(dd_h + (size_t) kk * E, dfl_c_hnorm, dd_cat, 1, E, eps, s);   // (the hidden first, then the token's)
        k::rmsnorm(dd_e, dfl_c_enorm, dd_cat + E, 1, E, eps, s);
        k::q8(dd_cat, 1, 2 * E, dd_xq, dd_xd, s);
        {
            const DMat* ms[2] = {&dfl_cg, &dfl_cu};
            float* ys[2] = {dd_fg, dd_fu};
            mvw_many(ms, ys, 2, dd_cat, dd_xq, dd_xd, 1);
        }
        k::swiglu(dd_fg, dd_fu, dd_fh, E, 0.0f, s);
        k::q8(dd_fh, 1, E, dd_xq, dd_xd, s);
        mvw(dfl_cd, dd_fh, dd_xq, dd_xd, dd_fo, 1);
        CK(cudaMemcpyAsync(dd_hp, dd_h + (size_t) kk * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s));
        k::add(dd_hp, dd_fo, E, s);
        if (dump) {
            const std::string tk = std::to_string(kk);
            dsave(("e" + tk).c_str(), dd_e, (size_t) E * sizeof(float));
            dsave(("hp" + tk).c_str(), dd_hp, (size_t) E * sizeof(float));
        }
        k::q8(dd_hp, 1, E, dd_xq, dd_xd, s);
        mvw(dfl_head, dd_hp, dd_xq, dd_xd, dd_logits, 1);
        k::argmax_prob(dd_logits, dfl_head.n, dd_seq + kk, dprob + (kk - 1), dd_amx, s);
        mvw(dfl_ch, dd_hp, nullptr, nullptr, dch + (kk - 1), 1);
    }
    CK(cudaMemcpyAsync(dtok, dd_seq + 1, (size_t) kDflN * sizeof(int), cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(dh_out, dd_out, (size_t) 3 * kDflN * sizeof(float), cudaMemcpyDeviceToHost, s));
}

// The drafts after tok (the newest token, at pos(), not fed yet): up to max_n (5 at most), their confidences; -1 when
// the drafter cannot (none loaded, a window pending, the row before pos() not the last one the target decoded - after
// a prompt, the first token steps - or no room for the block).  The pass makes all 5 (one graph); max_n only cuts.
int Engine::Impl::dfly_draft(int tok, int max_n, int* out, float* conf) {
    if (dfl.empty() || fault || wp_T > 0 || dfl_P != pos || pos + kDflT > max_ctx || tok < 0 || tok >= g.vocab) return -1;
    max_n = std::min(max_n, kDflN);
    if (max_n <= 0) return 0;
    const double t0 = now_ms();
    const int T = kDflT, P = pos, r = dfl_row;
    for (int t = 0; t < T; ++t) {
        dh_in[4 * t + 0] = P + t;
        dh_in[4 * t + 1] = dh_in[4 * t + 2] = dh_in[4 * t + 3] = 0;
        dh_in[4 * T + t] = t == 0 ? tok : kDflMask;
    }
    // the source row's state where the pass reads it
    CK(cudaMemcpyAsync(dd_shid, dfl_hid + (size_t) r * g.n_embd, (size_t) g.n_embd * sizeof(float),
                       cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(dd_scells, dfl_scells + (size_t) r * kDflL * dfl_cn, (size_t) kDflL * dfl_cn * sizeof(int),
                       cudaMemcpyDeviceToDevice, s));
    CK(cudaMemcpyAsync(dd_sncell, dfl_sncell + (size_t) r * kDflL, (size_t) kDflL * sizeof(int),
                       cudaMemcpyDeviceToDevice, s));
    // (HY4_DFLY_DUMP=<prefix>: call HY4_DFLY_DUMP_AT's (3rd) inputs and outputs to <prefix>.*, eager)
    static int ncall = 0;
    static const bool dpre = std::getenv("HY4_DFLY_DUMP") != nullptr;
    static const int dat = std::getenv("HY4_DFLY_DUMP_AT") ? std::atoi(std::getenv("HY4_DFLY_DUMP_AT")) : 3;
    static const bool use_graph = !(std::getenv("HY4_DFLY_GRAPH") && std::atoi(std::getenv("HY4_DFLY_GRAPH")) == 0);
    const bool dump = dpre && ++ncall == dat;
    if (dump || !use_graph) {
        dfl_pass(dump);
    } else {
        if (!dfl_graph) {
            cudaGraph_t gr = nullptr;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            dfl_pass(false);
            CK(cudaStreamEndCapture(s, &gr));
            CK(cudaGraphInstantiate(&dfl_graph, gr, 0));
            CK(cudaGraphDestroy(gr));
        }
        CK(cudaGraphLaunch(dfl_graph, s));
    }
    CK(cudaStreamSynchronize(s));
    const int* otok = (const int*) dh_out;
    static const int cmode = std::getenv("HY4_DFLY_CONF") ? std::atoi(std::getenv("HY4_DFLY_CONF")) : 0;
    for (int i = 0; i < max_n; ++i) {
        out[i] = otok[i];
        const float pr = dh_out[kDflN + i], ch = 1.0f / (1.0f + std::exp(-(dh_out[2 * kDflN + i] + dfl_ch_b)));
        conf[i] = cmode == 1 ? ch : pr;   // (HY4_DFLY_CONF=1: the confidence head's; 0: the draft's probability)
        dfl_conf2[i] = cmode == 1 ? pr : ch;
    }
    if (dump) {
        std::ofstream f(std::string(std::getenv("HY4_DFLY_DUMP")) + ".meta", std::ios::app);
        for (int i = 0; i < kDflN; ++i)
            f << "draft" << (i + 1) << " " << otok[i] << " " << dh_out[kDflN + i] << "\n";
    }
    dfl_ms = now_ms() - t0;
    return max_n;
}
