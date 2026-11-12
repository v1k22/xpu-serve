#pragma once

// GGUF block formats on the host (tensor/block_format.h): the values a block holds, exactly as ggml's dequantization
// computes them (the same float operations in the same order, so the results match bit for bit).

#include <cstdint>

#include "tensor/tensor.h"

namespace xs::quant {

// The float values of `count` consecutive blocks of `format`, block_layout(format).weights each, into `out`.
void decode_blocks(BlockFormat format, const uint8_t* blocks, int64_t count, float* out);

// Rows [first, first + count) of a Block view (a matrix of numel / cols rows), row-major into `out`. read_rows
// (quant.h) calls it for DType::Block.
void decode_block_rows(const TensorView& v, int64_t first, int64_t count, float* out);

}  // namespace xs::quant
