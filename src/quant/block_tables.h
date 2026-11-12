#pragma once

// The tables GGUF's i-quant formats (tensor/block_format.h) index, and their one constant: plain constexpr data that
// the host decoders (quant/block_decode.cpp) and device kernels both take from here. They are ggml's own tables,
// unchanged (third_party/ggml). Byte j of a grid entry is the level of weight j of its group, read as
// (entry >> 8 * j) & 0xFF: unsigned for the iq2 and iq3 grids (a magnitude: the signs are stored apart), int8 (-1, 0
// or 1) for iq1s_grid.

#include <array>
#include <cstdint>

#include <ggml/quant_tables.h>

namespace xs::quant {

using ggml::kmask_iq2xs;    // uint8_t[8]: 1 << j, the sign bit of weight j of a group of 8
using ggml::ksigns_iq2xs;   // uint8_t[128]: 7 stored sign bits -> 8, the 8th making the count of set bits even
using ggml::kvalues_iq4nl;  // int8_t[16]: the levels a 4-bit IQ4_NL / IQ4_XS index picks

using ggml::iq1s_grid;    // uint64_t[2048]: IQ1_S and IQ1_M; 8 weights an entry
using ggml::iq2s_grid;    // uint64_t[1024]: IQ2_S; 8 weights an entry
using ggml::iq2xs_grid;   // uint64_t[512]: IQ2_XS; 8 weights an entry
using ggml::iq2xxs_grid;  // uint64_t[256]: IQ2_XXS; 8 weights an entry
using ggml::iq3s_grid;    // uint32_t[512]: IQ3_S; 4 weights an entry
using ggml::iq3xxs_grid;  // uint32_t[256]: IQ3_XXS; 4 weights an entry

// What IQ1_S and IQ1_M add to every grid value of a group of 8 weights, negated when the group's delta bit is set.
inline constexpr float kIq1Delta = 0.125f;

}  // namespace xs::quant
