#pragma once

// MatMul and Attention of the SYCL backend: hand-written kernels for few rows (decode), oneMKL GEMMs (XMX) beyond.

#include <cstddef>
#include <cstdint>

#include "backend/sycl/sycl_kernels.h"

namespace xs::sycl_be {

// Device scratch memory owned by the session. The pointer stays valid until the next get(); every op on the
// in-order queue may reuse it because ops run one after another.
class Scratch {
public:
    virtual ~Scratch() = default;
    virtual void* get(size_t bytes) = 0;
};

// y [M, N] = x [M, K] @ w [N, K]^T
template <typename W>
void matmul(sycl::queue& q, Scratch& scratch, const float* x, const W* w, float* y, int64_t m, int64_t n, int64_t k);

// Auto: attention_rows up to kAttentionRowsMax queries, GEMMs beyond. Rows and Gemm force one (tests).
enum class AttentionPath : uint8_t { Auto, Rows, Gemm };
inline constexpr int64_t kAttentionRowsMax = 8;

// out [T, Hq, D] from q [T, Hq, D] and k, v [N, Hkv, D] (graph.h Attention).
template <typename KV>
void attention(sycl::queue& q, Scratch& scratch, const float* qp, const KV* k, const KV* v, float* out,
               const AttentionShape& s, AttentionPath path);

}  // namespace xs::sycl_be
