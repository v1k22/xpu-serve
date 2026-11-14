#include "quant/kv_turbo.h"

#include <cmath>
#include <cstring>

#include "util/error.h"
#include "util/half.h"

namespace xs::quant::turbo {

namespace {

bool flipped(const uint64_t* signs, int i) { return ((signs[i / 64] >> (i % 64)) & 1u) != 0; }

// The 16 lane sums added pairwise, lanes 1 apart, then 2, 4 and 8 apart.
float sum_lanes(float* lane) {
    for (int m = 1; m < kLanes; m <<= 1) {
        float next[kLanes];
        for (int l = 0; l < kLanes; ++l)
            next[l] = lane[l] + lane[l ^ m];
        std::memcpy(lane, next, sizeof(next));
    }
    return lane[0];
}

// The Walsh-Hadamard transform of 128 values in place, unnormalized.
void butterflies(float* x) {
    for (int h = 1; h < kGroup; h *= 2) {
        for (int i = 0; i < kGroup; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

void put_code(uint8_t* codes, int bits, int i, uint32_t code) {
    if (bits == 4) {
        codes[i / 2] |= static_cast<uint8_t>(code << (4 * (i % 2)));
        return;
    }
    codes[i / 4] |= static_cast<uint8_t>((code & 3u) << (2 * (i % 4)));
    if (bits == 3) codes[kGroup / 4 + i / 8] |= static_cast<uint8_t>((code >> 2) << (i % 8));
}

template <int Bits>
uint16_t quantize(const float* x, uint8_t* codes) {
    float lane[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        float s = 0.0f;
        for (int e = 0; e < kLaneValues; ++e)
            s = std::fma(x[l * kLaneValues + e], x[l * kLaneValues + e], s);
        lane[l] = s;
    }
    const float n = sum_lanes(lane);

    float w[kGroup];
    for (int i = 0; i < kGroup; ++i)
        w[i] = flipped(kSignsBefore, i) ? -x[i] : x[i];
    butterflies(w);

    float edge[kMagnitudes<Bits> - 1];
    for (int j = 0; j < kMagnitudes<Bits> - 1; ++j)
        edge[j] = edge_sq<Bits>(j) * n;
    std::memset(codes, 0, static_cast<size_t>(group_bytes(Bits)));
    float chosen[kGroup];
    for (int i = 0; i < kGroup; ++i) {
        const float v = flipped(kSignsAfter, i) ? -w[i] : w[i];
        const bool negative = v < 0.0f;
        const float sq = v * v;
        uint32_t m = 0;
        for (int j = 0; j < kMagnitudes<Bits> - 1; ++j)
            m += (negative ? sq > edge[j] : sq >= edge[j]) ? 1u : 0u;
        put_code(codes, Bits, i, code_of<Bits>(negative, m));
        chosen[i] = kTables<Bits>.mag_sq[m];
    }
    for (int l = 0; l < kLanes; ++l) {
        float s = 0.0f;
        for (int e = 0; e < kLaneValues; ++e)
            s += chosen[l * kLaneValues + e];
        lane[l] = s;
    }
    const float r = sum_lanes(lane);
    return f32_to_f16(n > 0.0f ? std::sqrt(n / r) : 0.0f);
}

template <int Bits>
void decode(const uint8_t* codes, uint16_t norm, float* rotated) {
    const float scale = f16_to_f32(norm);
    for (int i = 0; i < kGroup; ++i)
        rotated[i] = kTables<Bits>.centroid[code_at(codes, Bits, i)] * scale;
}

void check_bits(int bits) { XS_CHECK(bits >= 2 && bits <= 4, "TurboQuant KV formats have 2, 3 or 4 bits, got ", bits); }

}  // namespace

uint32_t code_at(const uint8_t* codes, int bits, int i) {
    if (bits == 4) return (codes[i / 2] >> (4 * (i % 2))) & 15u;
    const uint32_t low = (codes[i / 4] >> (2 * (i % 4))) & 3u;
    if (bits == 2) return low;
    return low | (((codes[kGroup / 4 + i / 8] >> (i % 8)) & 1u) << 2);
}

uint16_t quantize_group(const float* x, int bits, uint8_t* codes) {
    check_bits(bits);
    return bits == 4 ? quantize<4>(x, codes) : bits == 3 ? quantize<3>(x, codes) : quantize<2>(x, codes);
}

void decode_group(const uint8_t* codes, uint16_t norm, int bits, float* rotated) {
    check_bits(bits);
    if (bits == 4) return decode<4>(codes, norm, rotated);
    if (bits == 3) return decode<3>(codes, norm, rotated);
    decode<2>(codes, norm, rotated);
}

void rotate_group(float* x, bool inverse) {
    const uint64_t* first = inverse ? kSignsAfter : kSignsBefore;
    const uint64_t* last = inverse ? kSignsBefore : kSignsAfter;
    for (int i = 0; i < kGroup; ++i)
        x[i] = flipped(first, i) ? -x[i] : x[i];
    butterflies(x);
    for (int i = 0; i < kGroup; ++i) {
        const float y = x[i] * kInvSqrtGroup;
        x[i] = flipped(last, i) ? -y : y;
    }
}

void round_trip(float* x, int64_t rows, int64_t width, int bits) {
    check_bits(bits);
    XS_CHECK(width % kGroup == 0, "TurboQuant KV formats need rows of a multiple of ", kGroup, " values, got ", width);
    uint8_t codes[kGroup / 2];
    for (int64_t g = 0; g < rows * (width / kGroup); ++g) {
        float* group = x + g * kGroup;
        const uint16_t norm = quantize_group(group, bits, codes);
        decode_group(codes, norm, bits, group);
        rotate_group(group, true);
    }
}

}  // namespace xs::quant::turbo
