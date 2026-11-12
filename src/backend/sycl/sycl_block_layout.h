#pragma once

// GGUF block-format weights on the device (tensor/block_format.h). A block interleaves the codes of 256 columns with
// their scales, at offsets no kernel can load a vector from; the device holds the same bits regrouped into streams.
// Nothing is rounded or re-encoded: every stream is a field of the blocks, moved.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensor/block_format.h"

namespace xs::sycl_be {

// A weight [N, K] as streams, each row-major and 64-byte aligned, a row's part a whole number of bytes. With
// U = K / 32 units (the columns [32 u, 32 u + 32) of a row) and B = K / 256 blocks a row, stream by stream:
//
//   Q4_K     nibbles [K/2]   header [16 B]: d, dmin (f16), scales[12]
//   Q5_K     nibbles [K/2]   bit 4 [K/8]         header [16 B]
//   Q6_K     nibbles [K/2]   bits 4-5 [K/4]      scales i8 [K/16]      d f16 [2 B]
//   Q3_K     bits 0-1 [K/4]  bit 2 [K/8]         scales[12] [12 B]     d f16 [2 B]
//   Q2_K     bits 0-1 [K/4]  scale|min [K/16]    d, dmin f16 [4 B]
//   IQ4_NL   nibbles [K/2]   d f16 [2 U]
//   IQ4_XS   nibbles [K/2]   header [8 B]: d (f16), scales_h (u16), scales_l[4]
//   IQ3_XXS  indexes [K/4]   signs and scale u32 [K/8]   d f16 [2 B]
//   IQ3_S    indexes [K/4]   signs [K/8]         index bit 8 [K/32]    scales [K/64]   d f16 [2 B]
//   IQ2_XXS  indexes and signs: 8 bytes a unit [K/4]     d f16 [2 B]
//   IQ2_S    indexes [K/8]   signs [K/8]         index bits 8-9 [K/32] scales [K/32]   d f16 [2 B]
//
// Codes are in column order: nibble, bit pair or bit c of a row's stream is column c's (the low nibble of a byte the
// even column), where a block scatters them by halves and quarters. Scales keep the block's own encoding.
inline constexpr int kBlockStreams = 5;

// Whether the block kernels read `f` (the formats listed above).
bool block_on_device(BlockFormat f);
// Bytes a row of `cols` columns takes in each stream (0 past the format's last).
void block_stream_rows(BlockFormat f, int64_t cols, int64_t out[kBlockStreams]);

struct BlockImage {
    std::vector<uint8_t> bytes;         // every stream, each starting on a 64-byte boundary
    size_t offset[kBlockStreams] = {};  // of each stream in `bytes`
    int streams = 0;
};
// The streams of `rows` rows of `cols` columns stored as blocks of `f` (block_on_device).
BlockImage block_image(BlockFormat f, const uint8_t* blocks, int64_t rows, int64_t cols);

// The formats a pack already has kernels for, rewritten as that pack format with a group of 32 (the same values):
// Q8_0 as q8, Q4_1 as q4, Q4_0 as q4 with every bias -8 scale.
bool block_as_pack(BlockFormat f);
struct PackImage {
    bool q4 = false;
    std::vector<uint8_t> codes;    // q8: i8 [rows, cols]; q4: [rows, cols / 2], the even column in the low nibble
    std::vector<uint16_t> scales;  // f16 [rows, cols / 32]
    std::vector<uint16_t> biases;  // q4 only
};
PackImage block_pack_image(BlockFormat f, const uint8_t* blocks, int64_t rows, int64_t cols);

}  // namespace xs::sycl_be
