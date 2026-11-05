// Q2_0 x Q8_0: adapted from llama.cpp's ggml-cpu/arch/x86/quants.c (MIT).
// Copyright (c) 2023-2026 The ggml authors. See third_party/ggml/LICENSE.
// Included inside ds4_iq_cpu.cpp's anonymous namespace, after its ISA helpers.
// Q2_0 uses Q8_0 activations, NOT the Q8_K inputs of the IQ row kernels.
#if defined(IQ_X86)
IQ_AVX2 inline __m256i q20_unpack32(const uint8_t* qs) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) qs));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 12));
    v = _mm256_or_si256(v, _mm256_slli_epi32(v, 6));
    return _mm256_and_si256(v, _mm256_set1_epi32(0x03030303));
}

IQ_AVX2 inline __m256 q20_dot32(__m256i codes, const int8_t* qs) {
    const __m256i y = _mm256_loadu_si256((const __m256i*) qs);
    // Subtract the signed activation sum, so codes 0..3 mean weights -1..2.
    const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(codes, y),
                                      _mm256_maddubs_epi16(_mm256_set1_epi8(1), y));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

template <int NT>
IQ_AVX2 void q20_rows(const uint8_t* w, size_t rb, int k, const void* const* act,
                      float* const* out, int r0, int r1) {
    const int nb = k / QK2_0;
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    for (int r = r0; r < r1; ++r) {
        const block_q2_0* x = (const block_q2_0*) (w + (size_t) r * rb);
        __m256 acc[NT][2];
        for (int t = 0; t < NT; ++t) acc[t][0] = acc[t][1] = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const float dx = ggml_fp16_to_fp32(x[b].d);
            const __m256i v0 = q20_unpack32(x[b].qs), v1 = q20_unpack32(x[b].qs + 8);
            for (int t = 0; t < NT; ++t) {
                const block_q8_0* a = y[t] + 2 * b;
                const __m256 d0 = _mm256_set1_ps(dx * ggml_fp16_to_fp32(a[0].d));
                const __m256 d1 = _mm256_set1_ps(dx * ggml_fp16_to_fp32(a[1].d));
                acc[t][b & 1] = _mm256_fmadd_ps(d0, q20_dot32(v0, a[0].qs), acc[t][b & 1]);
                acc[t][b & 1] = _mm256_fmadd_ps(d1, q20_dot32(v1, a[1].qs), acc[t][b & 1]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 v = _mm256_add_ps(acc[t][0], acc[t][1]);
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            out[t][r] = _mm_cvtss_f32(s);
        }
    }
}
#endif

bool q20_enabled() {
#if defined(IQ_X86)
    static const bool enabled = [] {
        const char* v = std::getenv("DS4_Q2_0_AVX2");
        return v && std::atoi(v) != 0 && isa().avx2;
    }();
    return enabled;
#else
    return false;
#endif
}

bool q20_dispatch(int type, const uint8_t* w, size_t rb, int k, const void* const* act,
                  int nt, float* const* out, int r0, int r1) {
    if (type != 42 || k <= 0 || k % 64 || !q20_enabled()) return false;
#if defined(IQ_X86)
    for (int t = 0; t < nt; t += 4) {
        switch (std::min(4, nt - t)) {
            case 1: q20_rows<1>(w, rb, k, act + t, out + t, r0, r1); break;
            case 2: q20_rows<2>(w, rb, k, act + t, out + t, r0, r1); break;
            case 3: q20_rows<3>(w, rb, k, act + t, out + t, r0, r1); break;
            case 4: q20_rows<4>(w, rb, k, act + t, out + t, r0, r1); break;
        }
    }
#endif
    return true;
}
