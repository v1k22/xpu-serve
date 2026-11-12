#pragma once

// Kernels over GGUF block-format weights in their device layout (sycl_block_layout.h). They mirror the pack kernels of
// sycl_kernels.h: a GEMV that reads each weight once for up to kGemvMaxRows inputs, the weight written out as f16 or
// f32 for a GEMM, and rows gathered by id.

#include <cstdint>
#include <span>

#include <sycl/sycl.hpp>

#include "backend/sycl/sycl_block_layout.h"

namespace xs::sycl_be {

// A weight [N, K] on the device: block_image's streams, and the table its indexes point into where it has one.
struct BlockWeight {
    BlockFormat format = BlockFormat::Q4_K;
    const uint8_t* s[kBlockStreams] = {};
    const uint32_t* grid = nullptr;
};

// The table of magnitudes of a codebook format as u32 words in index order (four magnitudes a word, one a byte, the
// first lowest; an index of IQ2 takes two words), or empty for a format without one.
std::span<const uint32_t> block_grid(BlockFormat f);

// y [M, N] = x [M, K] @ w^T for M <= kGemvMaxRows; x 16-byte aligned.
void gemv_block(sycl::queue& q, const float* x, const BlockWeight& w, float* y, int64_t m, int64_t n, int64_t k);
// out [N, K] = w as f16 or f32: the values ggml's dequantization gives, rounded to T.
template <typename T>
void dequantize_block(sycl::queue& q, const BlockWeight& w, T* out, int64_t n, int64_t k);
// Rows of a table [rows, width] by id, as f32.
template <typename I>
void gather_block(sycl::queue& q, const BlockWeight& table, const I* ids, float* out, int64_t count, int64_t width);

}  // namespace xs::sycl_be
