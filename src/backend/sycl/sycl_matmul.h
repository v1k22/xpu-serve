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

// How a quantized matmul runs: up to kGemvMaxRows rows the GEMV, or from xmx_gemv_rows rows its matrix-engine
// version (gemv_quant_xmx, whose cost hardly grows with the rows: a speculative verify); beyond, a matrix-engine kernel
// that dequantizes the weight on the way (quant_gemm for a few rows, quant_gemm_tiled for many) or, past those, the
// weight dequantized once to f16 (f32 in f32 precision) and multiplied by oneMKL's GEMM. The paths past the GEMVs
// read x in f16.
enum class QuantPath : uint8_t { Gemv, GemvXmx, FewRows, Tiled, Dequant };
struct QuantMatmulPlan {  // what the device favours, set once per session
    bool f32 = false;           // f32 precision: the GEMV, else dequantize to f32 and oneMKL
    int64_t xmx_gemv_rows = 0;  // gemv_quant_xmx from this many rows up to kGemvMaxRows (0: never)
    int64_t few_rows = 0;       // quant_gemm up to this many rows, at most kQuantGemmMaxRows (0: never)
    int64_t tiled_rows = 0;     // quant_gemm_tiled up to this many rows (0: never)
    int64_t compute_units = 0;  // the matrix-engine kernels' split of K
};
// The path for these shapes and this weight (x assumed 64-byte aligned; matmul_quant falls back to Dequant if not).
QuantPath quant_path(int64_t m, int64_t n, int64_t k, const QuantWeight& w, const QuantMatmulPlan& plan);
// Whether `count` quantized matmuls of one x (QuantGemvPart) at m <= kGemvMaxRows rows run on gemv_quant_xmx.
bool xmx_gemv(int64_t m, const QuantGemvPart* parts, int count, int64_t k, const QuantMatmulPlan& plan);
// y [M, N] = x [M, K] @ dequant(w)^T along quant_path, or with `add` y += x @ dequant(w)^T (a residual folded in).
// x in f32 (converted in scratch where a path reads f16), or in f16 for M > kGemvMaxRows; y in f32, or in f16 for
// M > kGemvMaxRows (plan_half_values keeps such values in f16, saving the conversions). `add` needs an f32 y and
// M > kGemvMaxRows.
template <typename X, typename Y>
void matmul_quant(sycl::queue& q, Scratch& scratch, const X* x, const QuantWeight& w, Y* y, int64_t m, int64_t n,
                  int64_t k, const QuantMatmulPlan& plan, bool add = false);

// Whether `count` quantized matmuls of one x at m rows all take quant_gemm (QuantPath::FewRows) and so can share a
// launch (quant_gemm_parts); x assumed 64-byte aligned.
template <typename Y>
bool quant_parts_few_rows(int64_t m, const QuantPart<Y>* parts, int count, int64_t k, const QuantMatmulPlan& plan);
// y_p [M, N_p] = x @ dequant(w_p)^T for such parts in one launch; x in f32 (converted once in scratch) or f16. An f16
// x that is not 64-byte aligned runs each part by matmul_quant instead.
template <typename X, typename Y>
void matmul_quant_parts(sycl::queue& q, Scratch& scratch, const X* x, const QuantPart<Y>* parts, int count, int64_t m,
                        int64_t k, const QuantMatmulPlan& plan);

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
    // GEMM path: bytes of scores and probabilities that stay in the device's cache between the GEMMs and the softmax
    // of a chunk of queries; 0 where streaming them through memory costs less than the smaller GEMMs (set per device).
    int64_t score_cache = 0;
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
