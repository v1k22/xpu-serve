#include "tensor/block_format.h"

#include <array>

#include "util/error.h"

namespace xs {

namespace {

constexpr std::array<std::string_view, kBlockFormats> kNames = {
    "Q4_0",  "Q4_1",  "Q5_0",    "Q5_1",   "Q8_0",  "Q2_K",    "Q3_K",  "Q4_K",   "Q5_K",   "Q6_K",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
};

}  // namespace

std::string_view block_format_name(BlockFormat f) {
    const auto i = static_cast<size_t>(f);
    return i < kNames.size() ? kNames[i] : std::string_view("?");
}

std::optional<BlockFormat> block_format_from_name(std::string_view name) {
    for (size_t i = 0; i < kNames.size(); ++i) {
        if (kNames[i] == name) return static_cast<BlockFormat>(i);
    }
    return std::nullopt;
}

size_t block_storage_bytes(BlockFormat f, int64_t numel) {
    const BlockLayout l = block_layout(f);
    XS_CHECK(numel >= 0 && numel % l.weights == 0, block_format_name(f), ": ", numel,
             " weights are not whole blocks of ", l.weights);
    return static_cast<size_t>(numel / l.weights) * static_cast<size_t>(l.bytes);
}

size_t block_row_bytes(BlockFormat f, int64_t cols) {
    return block_storage_bytes(f, cols);
}

}  // namespace xs
