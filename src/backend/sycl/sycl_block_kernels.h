#pragma once

// Kernels over GGUF block-format weights in their device layout (sycl_block_layout.h). They mirror the pack kernels of
// sycl_kernels.h: a GEMV that reads each weight once for up to kGemvMaxRows inputs, and its matrix-engine version; the
// same GEMV on each token's experts of a bank; the weight written out as f16 or f32 for a GEMM; rows gathered by id.

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
// The same on the matrix engines (needs xmx_device), x read in f16: the cost hardly grows with M (a speculative
// verify). `scratch` (gemv_block_xmx_scratch bytes, 64-byte aligned) holds x in f16 and its sums while it runs.
bool gemv_block_xmx_fits(const BlockWeight& w, int64_t k);
size_t gemv_block_xmx_scratch(int64_t m, int64_t k);
void gemv_block_xmx(sycl::queue& q, const float* x, const BlockWeight& w, float* y, int64_t m, int64_t n, int64_t k,
                    int64_t compute_units, void* scratch);
// The same for kGemvMaxRows < M <= kQuantGemmMaxRows rows (a weight gemv_block_xmx_fits): x in f16, 64-byte aligned;
// y in f32 or f16, the weight read once per 32 rows and never written out. `add` (an f32 y): y += the product.
template <typename Y>
void gemm_block_xmx(sycl::queue& q, const sycl::half* x, const BlockWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                    int64_t compute_units, bool add = false);
// ExpertMatMul over a bank (rows = E * N of K columns): expert_matmul_quant's contract (sycl_kernels.h), x 16-byte
// aligned.
template <typename I>
void expert_matmul_block(sycl::queue& q, const float* x, bool per_slot, const BlockWeight& w, const I* idx, float* out,
                         int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k, int32_t* grouping,
                         int64_t xmx_units);
// ExpertMatMul over a bank for many pairs on the matrix engines (needs xmx_device), expert_matmul_quant's contract:
// the pairs grouped by expert, an expert's weights read once per 64 of its pairs and never written out. x is read in
// f16 and the weights rounded to f16, as a GEMM over the dequantized bank reads them. `scratch`
// (expert_gemm_block_xmx_scratch bytes, 64-byte aligned) holds the grouping and x in f16 while it runs.
bool expert_gemm_block_xmx_fits(const BlockWeight& w, int64_t pairs, int64_t k);
size_t expert_gemm_block_xmx_scratch(int64_t pairs, int64_t experts, int64_t k);
template <typename I>
void expert_gemm_block_xmx(sycl::queue& q, const float* x, bool per_slot, const BlockWeight& w, const I* idx,
                           float* out, int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k,
                           int64_t compute_units, void* scratch);
// Rows [first, ...) of a weight of K columns, as a weight of its own (an expert of a bank).
BlockWeight block_rows(const BlockWeight& w, int64_t first, int64_t k);
// out [N, K] = w as f16 or f32: the values ggml's dequantization gives, rounded to T.
template <typename T>
void dequantize_block(sycl::queue& q, const BlockWeight& w, T* out, int64_t n, int64_t k);
// Rows of a table [rows, width] by id, as f32.
template <typename I>
void gather_block(sycl::queue& q, const BlockWeight& table, const I* ids, float* out, int64_t count, int64_t width);

}  // namespace xs::sycl_be
