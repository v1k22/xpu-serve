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
    // Like get(), in USM host memory (pinned, readable by the device), for small arrays the host writes and kernels
    // read (oneMKL's grouped GEMM parameters).
    virtual void* get_host(size_t bytes) = 0;
};

// y [M, N] = x [M, K] @ w [N, K]^T
template <typename W>
void matmul(sycl::queue& q, Scratch& scratch, const float* x, const W* w, float* y, int64_t m, int64_t n, int64_t k);
// The same through the GEMM path with x already in f16 (M > kGemvMaxRows).
void matmul_half_x(sycl::queue& q, const half* x, const half* w, float* y, int64_t m, int64_t n, int64_t k);
// y [M, N] += x [M, K] @ w [N, K]^T through the GEMM path (M > kGemvMaxRows); x in f16, or f32 (converted in scratch).
template <typename X>
void matmul_accumulate(sycl::queue& q, Scratch& scratch, const X* x, const half* w, float* y, int64_t m, int64_t n,
                       int64_t k);
// The GEMM path writing y in f16 (an attention operand); x in f32 (converted in scratch) or f16.
template <typename X>
void matmul_half_y(sycl::queue& q, Scratch& scratch, const X* x, const half* w, half* y, int64_t m, int64_t n, int64_t k);

// y [M, N] = x [M, K] @ dequant(w)^T: the quantized GEMV up to kGemvMaxRows rows; up to kQuantGemmMaxRows with
// `use_quant_gemm` (a device with matrix engines where it wins) quant_gemm, which reads the weight once; beyond (or when
// `f32`), w is dequantized once (to f16, or f32 when `f32`) and multiplied with oneMKL's GEMM.
void matmul_quant(sycl::queue& q, Scratch& scratch, const float* x, const QuantWeight& w, float* y, int64_t m,
                  int64_t n, int64_t k, bool f32, bool use_quant_gemm, int64_t compute_units);
// The same with x in f16 (plan_half_values keeps quant_gemm's inputs in f16, saving a conversion kernel per call):
// quant_gemm when it fits, else x widened to f32 and matmul_quant.
void matmul_quant_half_x(sycl::queue& q, Scratch& scratch, const half* x, const QuantWeight& w, float* y, int64_t m,
                         int64_t n, int64_t k, bool use_quant_gemm, int64_t compute_units);

// ExpertMatMul over a quantized bank (expert_matmul_quant's contract) through oneMKL, for many (token, slot) pairs
// (prefill): the pairs are grouped by expert, each expert's input rows gathered, its bank dequantized (to f16, or f32
// when `f32`) in runs of at most kExpertDequantBytes, and every expert's GEMM runs in one grouped gemm_batch.
inline constexpr int64_t kExpertPairsGemm = 256;  // from this many pairs up
inline constexpr size_t kExpertDequantBytes = size_t{256} << 20;
template <typename I>
void expert_matmul_quant_gemm(sycl::queue& q, Scratch& scratch, const float* x, bool per_slot, const QuantWeight& w,
                              const I* idx, float* out, int64_t tokens, int64_t slots, int64_t experts, int64_t n,
                              int64_t k, bool f32);

// Auto: attention_rows up to kAttentionRowsMax queries; beyond, flash_attention where the device and shape allow
// it, GEMMs otherwise. Rows, Gemm and Flash force one (tests); Flash takes GEMMs where it does not fit.
enum class AttentionPath : uint8_t { Auto, Rows, Gemm, Flash };
inline constexpr int64_t kAttentionRowsMax = 8;

struct AttentionPlan {
    AttentionPath path = AttentionPath::Auto;
    bool xmx = false;  // xmx_device(): the matrix-engine kernels run
    int64_t compute_units = 0;
};

// Keys and values in two parts, k / v holding the first `first_len` (an Attention over Concat(k1, k2) with the
// concats fused away). Only the GEMM path takes them: head_dim past the fused kernel's, more queries than the rows'.
struct KvSecond {
    const void* k = nullptr;
    const void* v = nullptr;
    int64_t first_len = 0;
};

// out [T, Hq, D] from q [T, Hq, D] and k, v [N, Hkv, D] (graph.h Attention). An f16 q takes a matrix path.
template <typename KV, typename QT = float>
void attention(sycl::queue& q, Scratch& scratch, const QT* qp, const KV* k, const KV* v, float* out,
               const AttentionShape& s, const AttentionPlan& plan, const KvSecond& second = {});
// True when attention() runs on the matrix engines (flash or GEMMs) for this shape: f16 operands welcome.
bool attention_matrix_path(const AttentionShape& s, AttentionPath path);

// attention() over an int8 KV cache. The rows path reads the codes; the GEMM path first dequantizes the keys the
// queries can see to CT in `kv_scratch` (a second scratch: the GEMM path uses `scratch` for its own buffers).
template <typename CT>
void attention_i8(sycl::queue& q, Scratch& scratch, Scratch& kv_scratch, const float* qp, const KvI8& kv, float* out,
                  const AttentionShape& s, const AttentionPlan& plan);

}  // namespace xs::sycl_be
