#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace xs {

// The block formats of GGUF files (ggml's quantization types). A weight [rows, cols] in one of them is its rows one
// after another, each row cols / weights blocks of `bytes` bytes: a block holds the codes of `weights` consecutive
// columns of one row and their scales. The engine keeps such a weight byte for byte as the file stores it
// (DType::Block; TensorView::block names the format) and decodes blocks where it reads them: quant/block_decode.h on
// the host, the backends' kernels on a device. The value of every weight is the one ggml's dequantization gives.
enum class BlockFormat : uint8_t {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,  // 32 weights: codes of 4, 5 or 8 bits, one scale (and, _1, one minimum)
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,  // 256 weights in sub-blocks of 16 or 32 with small scales under one f16
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,  // 256 weights: indexes into tables of 8 magnitudes, signs apart
    IQ3_XXS,
    IQ3_S,  // 256 weights: indexes into tables of 4 magnitudes, signs apart
    IQ4_NL,
    IQ4_XS,  // 32 / 256 weights: 4-bit indexes into 16 unevenly spaced values
};
inline constexpr int kBlockFormats = 19;

struct BlockLayout {
    int32_t bytes = 0;    // of one block
    int32_t weights = 0;  // it holds: 32 or 256
};

constexpr BlockLayout block_layout(BlockFormat f) {
    switch (f) {
        case BlockFormat::Q4_0:
            return {18, 32};
        case BlockFormat::Q4_1:
            return {20, 32};
        case BlockFormat::Q5_0:
            return {22, 32};
        case BlockFormat::Q5_1:
            return {24, 32};
        case BlockFormat::Q8_0:
            return {34, 32};
        case BlockFormat::Q2_K:
            return {84, 256};
        case BlockFormat::Q3_K:
            return {110, 256};
        case BlockFormat::Q4_K:
            return {144, 256};
        case BlockFormat::Q5_K:
            return {176, 256};
        case BlockFormat::Q6_K:
            return {210, 256};
        case BlockFormat::IQ1_S:
            return {50, 256};
        case BlockFormat::IQ1_M:
            return {56, 256};
        case BlockFormat::IQ2_XXS:
            return {66, 256};
        case BlockFormat::IQ2_XS:
            return {74, 256};
        case BlockFormat::IQ2_S:
            return {82, 256};
        case BlockFormat::IQ3_XXS:
            return {98, 256};
        case BlockFormat::IQ3_S:
            return {110, 256};
        case BlockFormat::IQ4_NL:
            return {18, 32};
        case BlockFormat::IQ4_XS:
            return {136, 256};
    }
    return {};
}

// GGUF's name of the format ("Q4_K", "IQ3_XXS"), which is also how a pack's manifest names it.
std::string_view block_format_name(BlockFormat f);
std::optional<BlockFormat> block_format_from_name(std::string_view name);

// Bytes of `numel` weights; numel must be a whole number of blocks.
size_t block_storage_bytes(BlockFormat f, int64_t numel);
// Bytes of one row of `cols` columns; cols must be a whole number of blocks.
size_t block_row_bytes(BlockFormat f, int64_t cols);

}  // namespace xs
