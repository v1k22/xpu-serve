#pragma once

// The TurboQuant KV cache formats turbo4, turbo3 and turbo2 (arXiv 2504.19874 without its QJL stage, as
// TheTom/llama-cpp-turboquant stores them). A row of a cache (one token's head) is cut into groups of 128 values. A
// group is rotated: sign flips, a Walsh-Hadamard transform, sign flips, which spreads any row's coordinates like a
// Gaussian's. Each rotated coordinate is rounded to the nearest of 16, 8 or 4 fixed centroids and stored as that code,
// beside one f16 norm per group chosen so the decoded group has the row's own norm: 66, 50 or 34 bytes per 128 values.
// A decoded group stays rotated. The rotation is orthogonal, so the dot product with a query rotated the same way is
// the dot product of the originals, and a weighted sum of decoded groups is rotated back once.
//
// The constants here are plain constexpr data, which device kernels include too. Host and device run the same
// arithmetic in the same order (quantize_group below spells it out), so they store the same bits.

#include <cstdint>

namespace xs::quant::turbo {

inline constexpr int kGroup = 128;
inline constexpr int kLanes = 16;                              // a group as 16 lanes of 8 consecutive values
inline constexpr int kLaneValues = kGroup / kLanes;
inline constexpr float kInvSqrtGroup = 0.08838834764831845f;  // 1 / sqrt(128)

// Bytes of codes per group.
constexpr int group_bytes(int bits) { return kGroup * bits / 8; }

// The rotation's sign flips before and after the transform: bit i % 64 of word i / 64 set where value i changes sign.
inline constexpr uint64_t kSignsBefore[2] = {0x42f85949e46a0359ull, 0x49f250a9a4c63bbbull};
inline constexpr uint64_t kSignsAfter[2] = {0x3f628fdc16acee90ull, 0xbea56562b7a5357aull};
// The sign bits of a lane's 8 values, bit e for value 8 * lane + e.
constexpr uint32_t lane_signs(uint64_t low, uint64_t high, int lane) {
    return static_cast<uint32_t>((lane < 8 ? low : high) >> (8 * (lane % 8))) & 0xffu;
}

// A format's centroids are +-mags (Lloyd-Max for a Gaussian of variance 1 / 128), `edges` the borders between
// neighbouring magnitudes. Code c of `Bits` bits names centroid c in ascending order: the upper half of the codes is
// +mags[c - half], the lower half -mags[half - 1 - c].
template <int Bits>
struct Codebook;
template <>
struct Codebook<2> {
    static constexpr float mags[2] = {0.039994f, 0.133462f};
    static constexpr float edges[1] = {0.086728f};
};
template <>
struct Codebook<3> {
    static constexpr float mags[4] = {0.021663f, 0.066822f, 0.118786f, 0.190207f};
    static constexpr float edges[3] = {0.044243f, 0.092804f, 0.154496f};
};
template <>
struct Codebook<4> {
    static constexpr float mags[8] = {0.011349f, 0.034299f, 0.058050f, 0.083292f,
                                      0.111036f, 0.143016f, 0.182877f, 0.241529f};
    static constexpr float edges[7] = {0.022824f, 0.046174f, 0.070671f, 0.097164f, 0.127026f, 0.162947f, 0.212203f};
};
template <int Bits>
inline constexpr int kMagnitudes = 1 << (Bits - 1);
// A coordinate w of a group's transform (before its 1 / sqrt(128) and the group's 1 / norm) lies at or beyond edge j
// when w * w >= edge_sq(j) * n, n the group's squared norm: the comparison that picks a code without a division.
template <int Bits>
constexpr float edge_sq(int j) {
    const double e = Codebook<Bits>::edges[j];
    return static_cast<float>(e * e * kGroup);
}
template <int Bits>
constexpr uint32_t code_of(bool negative, uint32_t m) {
    return negative ? kMagnitudes<Bits> - 1 - m : kMagnitudes<Bits> + m;
}
// The centroid of every code, and the square of every magnitude.
template <int Bits>
struct Tables {
    float centroid[1 << Bits] = {};
    float mag_sq[1 << (Bits - 1)] = {};
    constexpr Tables() {
        for (int m = 0; m < kMagnitudes<Bits>; ++m) {
            centroid[code_of<Bits>(false, static_cast<uint32_t>(m))] = Codebook<Bits>::mags[m];
            centroid[code_of<Bits>(true, static_cast<uint32_t>(m))] = -Codebook<Bits>::mags[m];
            mag_sq[m] = Codebook<Bits>::mags[m] * Codebook<Bits>::mags[m];
        }
    }
};
template <int Bits>
inline constexpr Tables<Bits> kTables{};

// A group's codes in memory, value i of 128. 4 bits: nibble i % 2 of byte i / 2, the low one first. 2 bits: bits
// 2 * (i % 4) and up of byte i / 4. 3 bits: its low two bits as the 2-bit layout in the first 32 bytes, its third bit
// as bit i % 8 of byte 32 + i / 8. A lane's 8 values so fill 4, 2 or 2 + 1 whole bytes.
uint32_t code_at(const uint8_t* codes, int bits, int i);

// 128 values -> their codes (group_bytes(bits) of them) and the group's norm as f16 bits. The arithmetic every
// backend repeats, in single precision:
//   n    the sum of squares: each lane's 8 values fused-multiply-added in order, the 16 lane sums added pairwise
//        (lanes 1 apart, then 2, 4 and 8 apart)
//   w    the values with the first signs flipped, through the butterflies of spans 1, 2, ... 64, second signs flipped
//   code negative when w < 0; its magnitude the count of edges j with w * w >= edge_sq(j) * n (> for a negative w)
//   norm sqrt(n / r), r the sum of the chosen mags' squares in the same order as n (plain adds); 0 when n is 0
uint16_t quantize_group(const float* x, int bits, uint8_t* codes);
// The 128 rotated values a group stores: centroid * norm.
void decode_group(const uint8_t* codes, uint16_t norm, int bits, float* rotated);
// 128 values rotated in place, or rotated back (`inverse`).
void rotate_group(float* x, bool inverse);

// Replaces each row of x [rows, width] by the value its groups store, rotated back; width a multiple of 128.
void round_trip(float* x, int64_t rows, int64_t width, int bits);

}  // namespace xs::quant::turbo
