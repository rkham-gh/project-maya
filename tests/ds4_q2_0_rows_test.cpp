// Independent scalar Q2_0 decode/dot, including odd block counts and Q8_0 -128.
#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <utility>

namespace gg = strata::ds4::gguf;

static float scale(const uint8_t* p) {
    ggml_fp16_t h;
    std::memcpy(&h, p, 2);
    return ggml_fp16_to_fp32(h);
}

int main(int argc, char**) {
    // An enabled run must actually enter the backport. CTest marks unsupported
    // hardware skipped rather than silently testing the fallback twice.
    if (argc > 1 && !gg::iq_cpu_own_rows(42)) return 77;
    std::mt19937 rng(719);
    int failures = 0;
    for (int k : {64, 128, 192, 1024, 2688}) for (int nt : {1, 2, 3, 4, 5, 9}) {
        constexpr int rows = 11;
        const size_t rb = (size_t) (k / 64) * 18, ab = (size_t) (k / 32) * 34;
        std::vector<uint8_t> w(rb * rows), act(ab * nt);
        for (auto& x : w) x = (uint8_t) rng();
        for (auto& x : act) x = (uint8_t) rng();
        auto scales = [&](std::vector<uint8_t>& data, size_t block) {
            for (size_t b = 0; b < data.size() / block; ++b) {
                const float s = b % 7 == 0 ? 0.0f : (b % 2 ? -1.0f : 1.0f) * (float) (1 + rng() % 24) / 1024.0f;
                const ggml_fp16_t h = ggml_fp32_to_fp16(s);
                std::memcpy(data.data() + b * block, &h, 2);
            }
        };
        scales(w, 18); scales(act, 34);
        for (size_t b = 0; b < act.size() / 34; ++b) {
            act[b * 34 + 2] = 128;  // -128
            act[b * 34 + 3] = 127;
        }
        std::vector<const void*> ap((size_t) nt);
        std::vector<float> out((size_t) nt * rows, -12345.0f);
        std::vector<float*> op((size_t) nt);
        for (int t = 0; t < nt; ++t) { ap[t] = act.data() + (size_t) t * ab; op[t] = out.data() + t * rows; }
        for (const auto range : {std::pair<int, int>{0, rows}, {2, 9}, {3, 3}}) {
            std::fill(out.begin(), out.end(), -12345.0f);
            gg::iq_cpu_rows(42, w.data(), rb, k, ap.data(), nt, op.data(), range.first, range.second);
            for (int t = 0; t < nt; ++t) for (int r = 0; r < rows; ++r) {
                double ref = -12345.0, abs_sum = 0;
                if (r >= range.first && r < range.second) {
                    ref = 0;
                    for (int c = 0; c < k; ++c) {
                        const uint8_t* wb = w.data() + (size_t) r * rb + (c / 64) * 18;
                        const uint8_t* yb = act.data() + (size_t) t * ab + (c / 32) * 34;
                        const int q = ((wb[2 + (c % 64) / 4] >> (2 * (c % 4))) & 3) - 1;
                        const int y = (int) (int8_t) yb[2 + c % 32];
                        const double v = (double) scale(wb) * scale(yb) * q * y;
                        ref += v; abs_sum += std::abs(v);
                    }
                }
                const double got = op[t][r];
                if (!std::isfinite(got) || std::abs(got - ref) > 2e-6 * abs_sum + 1e-7) ++failures;
            }
        }
    }
    std::printf("Q2_0 scalar reference: %d failures\n", failures);
    return failures ? 1 : 0;
}
