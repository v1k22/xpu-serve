#pragma once

// Device kernels of the SYCL backend. Every launcher enqueues on an in-order queue and returns without waiting.
// Activations are f32; weights and the KV cache are f16 or f32 (the W / KV template parameters). Shapes follow the
// step-graph op spec (graph/graph.h); callers have checked them.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace xs::sycl_be {

using half = sycl::half;

inline constexpr int kMaxRank = 6;

enum class UnaryOp : uint8_t { Scale, Tanh, GeluTanh, Softcap, Sigmoid, Silu, Softplus, Clamp };
// y = op(x); `a` is the scale factor, softcap or lower bound, `b` the upper bound.
void unary(sycl::queue& q, UnaryOp op, const float* x, float* y, int64_t n, float a, float b);

// y = act(x) * b, same shapes; act is Silu, GeluTanh or Sigmoid (a fused activation-times-gate).
void act_mul(sycl::queue& q, UnaryOp act, const float* x, const float* b, float* y, int64_t n);

enum class BinaryOp : uint8_t { Add, Mul, Div };
// numpy broadcasting over at most kMaxRank dims: out shape and the element strides of a and b per out dim (0 where
// the operand is broadcast).
struct Broadcast {
    int rank = 0;
    int64_t out[kMaxRank] = {};
    int64_t stride_a[kMaxRank] = {};
    int64_t stride_b[kMaxRank] = {};
};
void binary(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n, const Broadcast& shape);
// Fast paths: same shape, and b repeating every `period` elements of a (a row vector over the leading dims).
void binary_same(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n);
void binary_period(sycl::queue& q, BinaryOp op, const float* a, const float* b, float* y, int64_t n, int64_t period);

// Rows of `width`.
void softmax(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width);
// The k largest per row of x [rows, width]: values and indices [rows, k], largest first, ties to the lower index.
void topk(sycl::queue& q, const float* x, float* values, int32_t* indices, int64_t rows, int64_t width, int64_t k);
// y [outer, inner] = the sum over the middle axis of x [outer, dim, inner].
void reduce_sum(sycl::queue& q, const float* x, float* y, int64_t outer, int64_t dim, int64_t inner);
// out [T, S, N] = x[t] (per_slot: x[t, s]) @ w[idx[t, s]]^T, w [E, N, K]; expert ids outside [0, E) read expert 0.
template <typename W, typename I>
void expert_matmul(sycl::queue& q, const float* x, bool per_slot, const W* w, const I* idx, float* out, int64_t tokens,
                   int64_t slots, int64_t experts, int64_t n, int64_t k);

// Rows of `width`; w may be null.
void rms_norm(sycl::queue& q, const float* x, const float* w, float* y, int64_t rows, int64_t width, float eps);

template <typename T, typename I>
void gather(sycl::queue& q, const T* table, const I* ids, float* out, int64_t n, int64_t width);

// x [T, H, D]; positions [T] or [T, sections]; inv_freq [rotary / (2 sections)].
template <typename I>
void rope(sycl::queue& q, const float* x, const I* positions, const float* inv_freq, float* y, int64_t tokens,
          int64_t heads, int64_t dim, int64_t rotary, int64_t sections);

// For each of `outer` blocks: rows [begin, begin + len) of `dim` rows of `inner` elements of `elem` bytes.
void slice(sycl::queue& q, const void* x, void* y, size_t elem, int64_t outer, int64_t dim, int64_t begin, int64_t len,
           int64_t inner);

// out[i] = x[sum_d index_d(i) * stride[d]] with index_d over the out shape.
struct Permute {
    int rank = 0;
    int64_t out[kMaxRank] = {};
    int64_t stride[kMaxRank] = {};
};
void transpose(sycl::queue& q, const void* x, void* y, size_t elem, int64_t n, const Permute& p);

void to_half(sycl::queue& q, const float* x, half* y, int64_t n);

// x [T, C], w [C, K]; reads the previous K-1 inputs from state_in [K-1, C] and writes the new ones to state_out.
// `silu` applies SiLU to the output (a fused Silu op).
void causal_conv(sycl::queue& q, const float* x, const float* w, const float* state_in, float* state_out, float* y,
                 int64_t tokens, int64_t channels, int64_t kernel, bool silu);

// Gated delta rule (graph.h GatedDelta). state_in / state_out [Hv, Dk, Dv]; Dk a multiple of 32 up to 512.
void gated_delta(sycl::queue& q, const float* qp, const float* kp, const float* vp, const float* g, const float* beta,
                 const float* state_in, float* state_out, float* out, int64_t tokens, int64_t k_heads,
                 int64_t v_heads, int64_t dk, int64_t dv, float qk_l2norm_eps);

struct AttentionShape {
    int64_t tokens = 0;    // T, queries
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t kv_len = 0;    // N, keys
    float scale = 1.0f;
    float softcap = 0.0f;
    int64_t window = 0;
    bool causal = true;
};

// An int8 KV cache: codes [N, Hkv, D] and one f32 scale per (key, head); the value is code * scale.
struct KvI8 {
    const int8_t* k = nullptr;
    const int8_t* v = nullptr;
    const float* k_scale = nullptr;
    const float* v_scale = nullptr;
};
// codes [rows, width] and scales [rows] of x, one sub-group per row, rounded as quant::quantize_kv_row_i8.
void quantize_kv_i8(sycl::queue& q, const float* x, int8_t* codes, float* scales, int64_t rows, int64_t width);
// out [rows, width] = codes * scale of the row.
template <typename CT>
void dequantize_kv_i8(sycl::queue& q, const int8_t* codes, const float* scales, CT* out, int64_t rows, int64_t width);

// One work-group per (query, head, split), online softmax over its share of the visible keys. For few queries.
// split == 1 writes `out`; otherwise each work-group writes [D partial sums, max, denominator] to `part`
// ([T * Hq * split, D + 2]) and attention_combine merges them into `out` (flash-decoding for long contexts).
template <typename KV>
void attention_rows(sycl::queue& q, const float* qp, const KV* k, const KV* v, float* out, const AttentionShape& s,
                    int64_t split, float* part);
// attention_rows reading an int8 cache (scales applied per key: to the score for k, to the probability for v).
void attention_rows_i8(sycl::queue& q, const float* qp, const KvI8& kv, float* out, const AttentionShape& s,
                       int64_t split, float* part);
void attention_combine(sycl::queue& q, const float* part, float* out, int64_t rows, int64_t split, int64_t head_dim);

// Prefill attention for queries [t0, t0 + tc) through GEMMs, steps of attention_gemm (sycl_matmul.cpp):
// q -> [Hkv, tc, G, D] packed in CT
template <typename CT>
void attention_pack_q(sycl::queue& q, const float* qp, CT* packed, const AttentionShape& s, int64_t t0, int64_t tc);
// scores [Hkv, tc, G, keys] (keys [k_lo, k_lo + keys)) -> probabilities in CT, masked keys 0.
template <typename CT>
void attention_softmax(sycl::queue& q, const float* scores, CT* probs, const AttentionShape& s, int64_t t0, int64_t tc,
                       int64_t k_lo, int64_t keys);
// [Hkv, tc, G, D] -> out rows t0.. of [T, Hq, D]
void attention_unpack(sycl::queue& q, const float* packed, float* out, const AttentionShape& s, int64_t t0, int64_t tc);

// y [M, N] = x [M, K] @ w [N, K]^T for M <= kGemvMaxRows, reading each weight once.
inline constexpr int64_t kGemvMaxRows = 8;
template <typename W>
void gemv(sycl::queue& q, const float* x, const W* w, float* y, int64_t m, int64_t n, int64_t k);

// A quantized weight [N, K] on the device (pack formats, tensor.h QuantParams): Q8 codes int8 [N, K]; Q4 codes
// [N, K / 2] with the even column in the low nibble; f16 scales and (Q4) biases [N, K / group]; group >= 32.
struct QuantWeight {
    bool q4 = true;
    const uint8_t* codes = nullptr;
    const half* scales = nullptr;
    const half* biases = nullptr;
    const float* input_scales = nullptr;  // AWQ: [K], multiplies column k of the dequantized weight; or null
    int64_t group = 0;
};
// y [M, N] = x [M, K] @ dequant(w)^T for M <= kGemvMaxRows; K a multiple of 32.
void gemv_quant(sycl::queue& q, const float* x, const QuantWeight& w, float* y, int64_t m, int64_t n, int64_t k);
// ExpertMatMul over a quantized expert bank (rows = E * N of K columns): out [T, S, N], x [T, K] or (per_slot)
// [T, S, K], expert ids outside [0, E) read expert 0. Up to kExpertPairsDirect (token, slot) pairs run one GEMV each;
// more are grouped by expert on the device in `grouping` (expert_grouping_bytes) so an expert's rows are read once
// per kGemvMaxRows of its pairs.
inline constexpr int64_t kExpertPairsDirect = 32;
size_t expert_grouping_bytes(int64_t pairs, int64_t experts);
// Groups the (token, slot) pairs by expert in `grouping` (expert_grouping_bytes): offsets [E + 1] at grouping + E,
// the pairs in expert order at grouping + 4 E + 2 (the order inside an expert is not deterministic).
template <typename I>
void expert_group(sycl::queue& q, const I* idx, int64_t pairs, int64_t experts, int32_t* grouping);
inline const int32_t* expert_offsets(const int32_t* grouping, int64_t experts) { return grouping + experts; }
inline const int32_t* expert_order(const int32_t* grouping, int64_t experts) { return grouping + 4 * experts + 2; }
// rows [pairs, K] in CT: row i is x's row for pair order[i] (x [T, K], or per_slot [T, S, K]).
template <typename CT>
void expert_gather_rows(sycl::queue& q, const float* x, bool per_slot, int64_t slots, const int32_t* order,
                        int64_t pairs, int64_t k, CT* rows);
// out row order[i] (of [pairs, N]) = rows row i.
void expert_scatter_rows(sycl::queue& q, const float* rows, const int32_t* order, int64_t pairs, int64_t n, float* out);
template <typename I>
void expert_matmul_quant(sycl::queue& q, const float* x, bool per_slot, const QuantWeight& w, const I* idx, float* out,
                         int64_t tokens, int64_t slots, int64_t experts, int64_t n, int64_t k, int32_t* grouping);
// out [N, K] = dequant(w) as f16 or f32, for the GEMM path.
template <typename T>
void dequantize(sycl::queue& q, const QuantWeight& w, T* out, int64_t n, int64_t k);
// Rows of a quantized table [rows, width] by id, as f32.
template <typename I>
void gather_quant(sycl::queue& q, const QuantWeight& table, const I* ids, float* out, int64_t count, int64_t width);

}  // namespace xs::sycl_be
