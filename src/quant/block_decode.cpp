#include "quant/block_decode.h"

#include "quant/block_tables.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

namespace xs::quant {

namespace {

// Block fields are little-endian, as GGUF stores them.
uint16_t u16_at(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
uint32_t u32_at(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
           static_cast<uint32_t>(p[3]) << 24;
}
float f16_at(const uint8_t* p) {
    return f16_to_f32(u16_at(p));
}

// One decoder per format writes the block_layout(f).weights values of one block, with the float operations of ggml's
// dequantize_row_* (ggml-quants.c) in their order. Every product among them is exact in f32 (an f16 times factors
// whose significands take at most 12 bits between them), so the one rounding is that of an added minimum: a value is
// ggml's bit for bit in whatever order its factors are multiplied, its last product fused with that add or not.

// Q4_0: f16 d, 16 bytes of 4-bit codes (column j in the low nibble of byte j, column j + 16 in its high nibble).
// w = (q - 8) * d.
void q4_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = static_cast<float>((qs[j] & 0x0F) - 8) * d;
        y[j + 16] = static_cast<float>((qs[j] >> 4) - 8) * d;
    }
}

// Q4_1: f16 d, f16 m, codes as Q4_0. w = q * d + m.
void q4_1(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float m = f16_at(b + 2);
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; ++j) {
        const float lo = static_cast<float>(qs[j] & 0x0F) * d;
        const float hi = static_cast<float>(qs[j] >> 4) * d;
        y[j] = lo + m;
        y[j + 16] = hi + m;
    }
}

// Q5_0: f16 d, 32 high bits (bit j: column j), low nibbles as Q4_0. w = (q - 16) * d.
void q5_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint32_t qh = u32_at(b + 2);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        const int lo = (qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10);
        const int hi = (qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10);
        y[j] = static_cast<float>(lo - 16) * d;
        y[j + 16] = static_cast<float>(hi - 16) * d;
    }
}

// Q5_1: f16 d, f16 m, high bits and low nibbles as Q5_0. w = q * d + m.
void q5_1(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float m = f16_at(b + 2);
    const uint32_t qh = u32_at(b + 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; ++j) {
        const int lo = (qs[j] & 0x0F) | static_cast<int>(((qh >> j) << 4) & 0x10);
        const int hi = (qs[j] >> 4) | static_cast<int>((qh >> (j + 12)) & 0x10);
        const float lo_d = static_cast<float>(lo) * d;
        const float hi_d = static_cast<float>(hi) * d;
        y[j] = lo_d + m;
        y[j + 16] = hi_d + m;
    }
}

// Q8_0: f16 d, 32 int8 codes. w = q * d.
void q8_0(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    for (int j = 0; j < 32; ++j)
        y[j] = static_cast<float>(static_cast<int8_t>(b[2 + j])) * d;
}

// Q2_K: 16 4-bit pairs (scale | min << 4) for 16 sub-blocks of 16, 64 bytes of 2-bit codes, f16 d, f16 dmin.
// Weight p: sub-block p / 16, code (qs[32 * (p / 128) + p % 32] >> 2 * (p % 128 / 32)) & 3;
// w = (d * scale) * q - dmin * min.
void q2_k(const uint8_t* b, float* y) {
    const uint8_t* scales = b;
    const uint8_t* q = b + 16;
    const float d = f16_at(b + 80);
    const float min = f16_at(b + 82);
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        for (int shift = 0; shift < 8; shift += 2) {
            for (int half = 0; half < 2; ++half) {
                const uint8_t sc = scales[is++];
                const float dl = d * static_cast<float>(sc & 0xF);
                const float ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16; ++l) {
                    const float v = dl * static_cast<float>((q[l + 16 * half] >> shift) & 3);
                    *y++ = v - ml;
                }
            }
        }
        q += 32;
    }
}

// Q3_K's 16 6-bit sub-block scales, packed in 12 bytes: the low 4 bits of scale k in nibble k % 8 of bytes 0..7
// (low nibbles first), the high 2 bits in bits 2 * (k / 4) of byte 8 + k % 4. Stored biased by 32.
int q3_k_scale(const uint8_t* s, int k) {
    const int low = k < 8 ? s[k] & 0xF : s[k - 8] >> 4;
    const int high = (s[8 + k % 4] >> (2 * (k / 4))) & 3;
    return (low | high << 4) - 32;
}

// Q3_K: 32 bytes of high bits (bit p / 32 of byte p % 32: the code's third bit, set meaning no -4), 64 bytes of
// 2-bit codes laid out as Q2_K's, 12 bytes of scales (q3_k_scale), f16 d. Weight p: sub-block p / 16;
// w = (d * scale) * (q - (high bit ? 0 : 4)).
void q3_k(const uint8_t* b, float* y) {
    const uint8_t* hm = b;
    const uint8_t* q = b + 32;
    const uint8_t* scales = b + 96;
    const float d_all = f16_at(b + 108);
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        for (int shift = 0; shift < 8; shift += 2) {
            for (int half = 0; half < 2; ++half) {
                const float dl = d_all * static_cast<float>(q3_k_scale(scales, is++));
                for (int l = 0; l < 16; ++l) {
                    const int i = l + 16 * half;
                    y[0] = dl * static_cast<float>(((q[i] >> shift) & 3) - ((hm[i] & m) != 0 ? 0 : 4));
                    ++y;
                }
            }
            m = static_cast<uint8_t>(m << 1);
        }
        q += 32;
    }
}

// The 6-bit scale and minimum of sub-block j (0..7) of Q4_K and Q5_K, packed in 12 bytes: j < 4 in the low 6 bits of
// bytes j and j + 4; j >= 4 in the nibbles of byte j + 4, their top 2 bits in the top bits of bytes j - 4 and j.
void k4_scale_min(int j, const uint8_t* q, int* sc, int* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K: f16 d, f16 dmin, 12 bytes of scales and minimums (k4_scale_min), 128 bytes of 4-bit codes: sub-blocks 2i and
// 2i + 1 (of 32) in the low and high nibbles of bytes 32i..32i + 31. w = (d * scale) * q - dmin * min.
void q4_k(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float min = f16_at(b + 2);
    const uint8_t* scales = b + 4;
    const uint8_t* q = b + 16;
    for (int is = 0; is < 8; is += 2) {
        int sc = 0, mn = 0;
        k4_scale_min(is, scales, &sc, &mn);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(mn);
        k4_scale_min(is + 1, scales, &sc, &mn);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(mn);
        for (int l = 0; l < 32; ++l) {
            const float v = d1 * static_cast<float>(q[l] & 0xF);
            *y++ = v - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const float v = d2 * static_cast<float>(q[l] >> 4);
            *y++ = v - m2;
        }
        q += 32;
    }
}

// Q5_K: Q4_K plus 32 bytes of fifth bits (bit s of byte p % 32 for sub-block s), before the nibbles.
// w = (d * scale) * (q + 16 * bit) - dmin * min.
void q5_k(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const float min = f16_at(b + 2);
    const uint8_t* scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int u1 = 1, u2 = 2;
    for (int is = 0; is < 8; is += 2) {
        int sc = 0, mn = 0;
        k4_scale_min(is, scales, &sc, &mn);
        const float d1 = d * static_cast<float>(sc);
        const float m1 = min * static_cast<float>(mn);
        k4_scale_min(is + 1, scales, &sc, &mn);
        const float d2 = d * static_cast<float>(sc);
        const float m2 = min * static_cast<float>(mn);
        for (int l = 0; l < 32; ++l) {
            const float v = d1 * static_cast<float>((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0));
            *y++ = v - m1;
        }
        for (int l = 0; l < 32; ++l) {
            const float v = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0));
            *y++ = v - m2;
        }
        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// Q6_K: 128 bytes of low nibbles, 64 bytes of 2-bit high parts, 16 int8 sub-block scales (of 16 weights), f16 d.
// In each half of 128 weights (ql + 64 * half, qh + 32 * half), weight 32 * k + l (k < 4, l < 32) has its low
// nibble from ql[l + 32 * (k % 2)] (low nibble for k < 2), its high part from bits 2k of qh[l], scale 8 * half + 2k +
// l / 16; w = (d * scale) * (q - 32).
void q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const auto* sc = reinterpret_cast<const int8_t*>(b + 192);
    const float d = f16_at(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// Byte j of a grid entry: the magnitude of weight j of its group.
uint8_t grid_byte(uint64_t entry, int j) {
    return static_cast<uint8_t>(entry >> (8 * j));
}

// 8 weights from a grid entry and a sign byte: w_j = (db * grid_j) * (bit j set ? -1 : 1).
void signed_group(float db, uint64_t entry, uint8_t signs, float* y) {
    for (int j = 0; j < 8; ++j) {
        const float v = db * static_cast<float>(grid_byte(entry, j));
        y[j] = (signs & kmask_iq2xs[j]) != 0 ? -v : v;
    }
}

// IQ2_XXS: f16 d, then per 32 weights 8 bytes: 4 grid indexes (iq2xxs_grid) of 8 weights, and a 32-bit word whose
// bits 7l..7l + 6 pick group l's signs (ksigns_iq2xs) and whose top 4 bits s give db = (d * (0.5 + s)) * 0.25.
void iq2_xxs(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t* g = b + 2 + 8 * ib;
        const uint32_t aux = u32_at(g + 4);
        const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l)
            signed_group(db, iq2xxs_grid[g[l]], ksigns_iq2xs[(aux >> (7 * l)) & 127], y + 8 * (4 * ib + l));
    }
}

// IQ2_XS: f16 d, 32 16-bit words (a 9-bit iq2xs_grid index, 7 sign bits above it) of 8 weights each, 8 bytes of two
// 4-bit scales per 32 weights (low nibble: the first 16). db = (d * (0.5 + s)) * 0.25.
void iq2_xs(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* scales = b + 66;
    for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + static_cast<float>(scales[ib] & 0xF)) * 0.25f,
                             d * (0.5f + static_cast<float>(scales[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            const uint16_t q = u16_at(b + 2 + 2 * (4 * ib + l));
            signed_group(db[l / 2], iq2xs_grid[q & 511], ksigns_iq2xs[q >> 9], y + 8 * (4 * ib + l));
        }
    }
}

// IQ2_S: f16 d, 32 bytes of low grid index bits, 32 sign bytes (one per 8 weights), 8 bytes of 2 high index bits
// per 8 weights (bits 2l of byte ib), 8 bytes of scales as IQ2_XS. Index: qs | (qh << (8 - 2l) & 0x300) (iq2s_grid).
void iq2_s(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    const uint8_t* signs = b + 34;
    const uint8_t* qh = b + 66;
    const uint8_t* scales = b + 74;
    for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + static_cast<float>(scales[ib] & 0xF)) * 0.25f,
                             d * (0.5f + static_cast<float>(scales[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            const int index = qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300);
            signed_group(db[l / 2], iq2s_grid[index], signs[4 * ib + l], y + 8 * (4 * ib + l));
        }
    }
}

// IQ3_XXS: f16 d, 64 iq3xxs_grid indexes of 4 weights, then per 32 weights a 32-bit word of 4 x 7 sign bits
// (ksigns_iq2xs, one byte per 8 weights) and a 4-bit scale s at the top: db = (d * (0.5 + s)) * 0.5.
void iq3_xxs(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t* qs = b + 2 + 8 * ib;
        const uint32_t aux = u32_at(b + 66 + 4 * ib);
        const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint64_t entry = iq3xxs_grid[qs[2 * l]] | uint64_t{iq3xxs_grid[qs[2 * l + 1]]} << 32;
            signed_group(db, entry, ksigns_iq2xs[(aux >> (7 * l)) & 127], y + 8 * (4 * ib + l));
        }
    }
}

// IQ3_S: f16 d, 64 bytes of low grid index bits (4 weights each), 8 bytes of ninth bits (bit k of byte ib for index
// k of those 32 weights), 32 sign bytes (one per 8 weights), 4 bytes of two 4-bit scales per 64 weights (low nibble:
// the first 32): db = d * (1 + 2s).
void iq3_s(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    const uint8_t* qh = b + 66;
    const uint8_t* signs = b + 74;
    const uint8_t* scales = b + 106;
    for (int ib = 0; ib < 8; ++ib) {
        const int s = ib % 2 == 0 ? scales[ib / 2] & 0xF : scales[ib / 2] >> 4;
        const float db = d * static_cast<float>(1 + 2 * s);
        for (int l = 0; l < 4; ++l) {
            const int i1 = qs[8 * ib + 2 * l] | ((qh[ib] << (8 - 2 * l)) & 256);
            const int i2 = qs[8 * ib + 2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256);
            const uint64_t entry = iq3s_grid[i1] | uint64_t{iq3s_grid[i2]} << 32;
            signed_group(db, entry, signs[4 * ib + l], y + 8 * (4 * ib + l));
        }
    }
}

// 8 weights of an IQ1 grid entry: w_j = dl * (grid_j + delta), grid_j in {-1, 0, 1}.
void iq1_group(float dl, uint64_t entry, float delta, float* y) {
    for (int j = 0; j < 8; ++j)
        y[j] = dl * (static_cast<float>(static_cast<int8_t>(grid_byte(entry, j))) + delta);
}

// IQ1_S: f16 d, 32 bytes of low iq1s_grid index bits (8 weights each), then per 32 weights a 16-bit word: 3 high
// index bits per group (bits 3l..3l + 2), a 3-bit scale s (bits 12..14) and the delta's sign (bit 15).
// dl = d * (2s + 1); w = dl * (grid + delta).
void iq1_s(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    for (int ib = 0; ib < 8; ++ib) {
        const uint16_t qh = u16_at(b + 34 + 2 * ib);
        const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) != 0 ? -kIq1Delta : kIq1Delta;
        for (int l = 0; l < 4; ++l)
            iq1_group(dl, iq1s_grid[qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)], delta, y + 8 * (4 * ib + l));
    }
}

// IQ1_M: 32 bytes of low iq1s_grid index bits, 16 bytes of 3 high index bits and a delta sign bit per 8 weights
// (a nibble each, low first), 4 16-bit scale words: 3-bit scales per 16 weights in their low 12 bits, and in their
// top nibbles the 4 nibbles of the f16 d. dl = d * (2s + 1); w = dl * (grid + delta).
void iq1_m(const uint8_t* b, float* y) {
    const uint8_t* qs = b;
    const uint8_t* qh = b + 32;
    uint16_t sc[4];
    for (int k = 0; k < 4; ++k)
        sc[k] = u16_at(b + 48 + 2 * k);
    const float d = f16_to_f32(
        static_cast<uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
    for (int ib = 0; ib < 8; ++ib) {
        const int shift = 6 * (ib % 2);
        const float dl[2] = {d * static_cast<float>(2 * ((sc[ib / 2] >> shift) & 7) + 1),
                             d * static_cast<float>(2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1)};
        for (int l = 0; l < 4; ++l) {
            const uint8_t h = qh[2 * ib + l / 2] >> (4 * (l % 2));  // this group's nibble
            const int index = qs[4 * ib + l] | ((h & 7) << 8);
            const float delta = (h & 8) != 0 ? -kIq1Delta : kIq1Delta;
            iq1_group(dl[l / 2], iq1s_grid[index], delta, y + 8 * (4 * ib + l));
        }
    }
}

// IQ4_NL: f16 d, 16 bytes of 4-bit kvalues_iq4nl indexes laid out as Q4_0's. w = d * kvalues[q].
void iq4_nl(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
        y[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
    }
}

// IQ4_XS: f16 d, a 16-bit word of the high 2 bits of 8 6-bit scales, 4 bytes of their low nibbles (low first), then
// per 32 weights 16 bytes laid out as IQ4_NL's. w = (d * (scale - 32)) * kvalues[q].
void iq4_xs(const uint8_t* b, float* y) {
    const float d = f16_at(b);
    const uint16_t scales_h = u16_at(b + 2);
    const uint8_t* scales_l = b + 4;
    const uint8_t* qs = b + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * static_cast<float>(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0xF]);
            y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
        y += 32;
        qs += 16;
    }
}

using BlockDecoder = void (*)(const uint8_t* block, float* out);

BlockDecoder decoder(BlockFormat f) {
    switch (f) {
        case BlockFormat::Q4_0:
            return q4_0;
        case BlockFormat::Q4_1:
            return q4_1;
        case BlockFormat::Q5_0:
            return q5_0;
        case BlockFormat::Q5_1:
            return q5_1;
        case BlockFormat::Q8_0:
            return q8_0;
        case BlockFormat::Q2_K:
            return q2_k;
        case BlockFormat::Q3_K:
            return q3_k;
        case BlockFormat::Q4_K:
            return q4_k;
        case BlockFormat::Q5_K:
            return q5_k;
        case BlockFormat::Q6_K:
            return q6_k;
        case BlockFormat::IQ1_S:
            return iq1_s;
        case BlockFormat::IQ1_M:
            return iq1_m;
        case BlockFormat::IQ2_XXS:
            return iq2_xxs;
        case BlockFormat::IQ2_XS:
            return iq2_xs;
        case BlockFormat::IQ2_S:
            return iq2_s;
        case BlockFormat::IQ3_XXS:
            return iq3_xxs;
        case BlockFormat::IQ3_S:
            return iq3_s;
        case BlockFormat::IQ4_NL:
            return iq4_nl;
        case BlockFormat::IQ4_XS:
            return iq4_xs;
    }
    fail("unknown block format ", static_cast<int>(f));
}

}  // namespace

void decode_blocks(BlockFormat format, const uint8_t* blocks, int64_t count, float* out) {
    const BlockDecoder fn = decoder(format);
    const BlockLayout l = block_layout(format);
    for (int64_t i = 0; i < count; ++i)
        fn(blocks + i * l.bytes, out + i * l.weights);
}

void decode_block_rows(const TensorView& v, int64_t first, int64_t count, float* out) {
    XS_CHECK(v.dtype == DType::Block && !v.shape.empty() && v.data != nullptr,
             "decode_block_rows needs a block format weight, got ", dtype_name(v.dtype), " ", dims_to_string(v.shape));
    const BlockLayout l = block_layout(v.block);
    const int64_t cols = v.shape.back();
    XS_CHECK(cols > 0 && cols % l.weights == 0, block_format_name(v.block), " weight ", dims_to_string(v.shape), ": ",
             cols, " columns are not whole blocks of ", l.weights);
    XS_CHECK(first >= 0 && count >= 0 && first + count <= matrix_rows(v.shape), block_format_name(v.block), ": rows ",
             first, " to ", first + count, " of a ", dims_to_string(v.shape), " weight");
    const size_t row_bytes = block_row_bytes(v.block, cols);
    decode_blocks(v.block, static_cast<const uint8_t*>(v.data) + static_cast<size_t>(first) * row_bytes,
                  count * (cols / l.weights), out);
}

}  // namespace xs::quant
