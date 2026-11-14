#pragma once

// Device kernels of the SYCL backend. Every launcher enqueues on an in-order queue and returns without waiting.
// Activations are f32; weights and the KV cache are f16 or f32 (the W / KV template parameters). Shapes follow the
// step-graph op spec (graph/graph.h); callers have checked them.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quant/kv_format.h"

namespace xs::sycl_be {

using half = sycl::half;

inline constexpr int kMaxRank = 6;

enum class UnaryOp : uint8_t { Scale, Tanh, GeluTanh, Softcap, Sigmoid, Silu, Softplus, Clamp, Gelu };
// y = op(x); `a` is the scale factor, softcap or lower bound, `b` the upper bound.
// Elementwise and norm kernels write f32 or f16 (Y = float or half): an f16 output feeds a GEMM directly.
template <typename Y>
void unary(sycl::queue& q, UnaryOp op, const float* x, Y* y, int64_t n, float a, float b);

// y = act(x) * b, same shapes; act is Silu, GeluTanh or Sigmoid (a fused activation-times-gate).
template <typename X, typename Y>
void act_mul(sycl::queue& q, UnaryOp act, const X* x, const X* b, Y* y, int64_t n);
// y = act(x + bias), bias repeating every `period` elements (a matmul's bias folded into its activation).
template <typename Y>
void unary_bias(sycl::queue& q, UnaryOp act, const float* x, const float* bias, Y* y, int64_t n, int64_t period);

enum class BinaryOp : uint8_t { Add, Mul, Div };
// numpy broadcasting over at most kMaxRank dims: out shape and the element strides of a and b per out dim (0 where
// the operand is broadcast).
struct Broadcast {
    int rank = 0;
    int64_t out[kMaxRank] = {};
    int64_t stride_a[kMaxRank] = {};
    int64_t stride_b[kMaxRank] = {};
};
template <typename Y>
void binary(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n, const Broadcast& shape);
// Fast paths: same shape, and b repeating every `period` elements of a (a row vector over the leading dims).
template <typename Y>
void binary_same(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n);
template <typename Y>
void binary_period(sycl::queue& q, BinaryOp op, const float* a, const float* b, Y* y, int64_t n, int64_t period);
// y = a + b * c, c repeating every `period` elements (period == n: the same shape): a gated residual.
template <typename Y>
void add_mul(sycl::queue& q, const float* a, const float* b, const float* c, Y* y, int64_t n, int64_t period);
// y = a + (b + c), c repeating every `period` elements (a residual plus a matmul with its bias).
template <typename Y>
void add3(sycl::queue& q, const float* a, const float* b, const float* c, Y* y, int64_t n, int64_t period);

// Rows of `width`.
void softmax(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width);
// The k largest per row of x [rows, width]: values and indices [rows, k], largest first, ties to the lower index.
// `scratch`: topk_scratch bytes (0 for most shapes: then it may be null).
size_t topk_scratch(int64_t rows, int64_t width);
void topk(sycl::queue& q, const float* x, float* values, int32_t* indices, int64_t rows, int64_t width, int64_t k,
          void* scratch);
// SignMatMul (graph.h): y [M, N] = x [M, K] @ w^T for the one-bit weight of signs [N, K / 32] and scales [N], meant
// for a row or two of x. x is rounded to 12 bits of each row's largest magnitude. `scratch` (sign_matmul_scratch
// bytes, 64-byte aligned) holds a row's bit planes while it runs.
size_t sign_matmul_scratch(int64_t k);
void sign_matmul(sycl::queue& q, const float* x, const uint32_t* signs, const float* scales, float* y, int64_t m,
                 int64_t n, int64_t k, void* scratch);
// y's row t is x's row t - 1, its row 0 zeros; rows of `width` values.
void prev_row(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width);
// y [outer, inner] = the sum over the middle axis of x [outer, dim, inner].
void reduce_sum(sycl::queue& q, const float* x, float* y, int64_t outer, int64_t dim, int64_t inner);
// out [T, S, N] = x[t] (per_slot: x[t, s]) @ w[idx[t, s]]^T, w [E, N, K]; expert ids outside [0, E) read expert 0.
template <typename W, typename I>
void expert_matmul(sycl::queue& q, const float* x, bool per_slot, const W* w, const I* idx, float* out, int64_t tokens,
                   int64_t slots, int64_t experts, int64_t n, int64_t k);

// Rows of `width`: x / rms(x) [* w] [+ bias]; w and bias may be null.
template <typename Y>
void rms_norm(sycl::queue& q, const float* x, const float* w, const float* bias, Y* y, int64_t rows, int64_t width,
              float eps);
// Rows of `width`: (x - mean) / sqrt(var + eps) * w + b; with `shift` (a row, or null), of x + shift.
template <typename Y>
void layer_norm(sycl::queue& q, const float* x, const float* shift, const float* w, const float* b, Y* y, int64_t rows,
                int64_t width, float eps);

// Concat as one copy kernel: `outer` rows, each the parts' rows (part k: bytes[k] from src[k] + o * bytes[k]) laid end
// to end in `out`. Up to kConcatParts parts.
inline constexpr int kConcatParts = 8;
struct ConcatParts {
    int count = 0;
    const void* src[kConcatParts] = {};
    size_t bytes[kConcatParts] = {};  // per outer row
};
void concat(sycl::queue& q, const ConcatParts& parts, void* out, int64_t outer);

template <typename T, typename I>
void gather(sycl::queue& q, const T* table, const I* ids, float* out, int64_t n, int64_t width);
// Rows of a table of 32-bit integers, copied as they are.
template <typename I>
void gather_bits(sycl::queue& q, const uint32_t* table, const I* ids, uint32_t* out, int64_t n, int64_t width);

// x [T, H, D]; positions [T] or [T, sections]; inv_freq [rotary / (2 sections)].
template <typename I, typename Y>
void rope(sycl::queue& q, const float* x, const I* positions, const float* inv_freq, Y* y, int64_t tokens,
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
// Up to kCopyParts copies in one launch: dst[p][0, count[p]) = src[p][0, count[p]) (a queue memcpy each costs a
// launch of the driver's, several times a kernel's on an integrated GPU).
inline constexpr int kCopyParts = 4;
struct CopyPart {
    const float* src = nullptr;
    float* dst = nullptr;
    int64_t count = 0;
};
void copy_parts(sycl::queue& q, const CopyPart* parts, int count);

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
    int64_t window = 0;  // > 0: keys within window - 1 of the query, before it (and after it when not causal)
    bool causal = true;
    const float* key_bias = nullptr;  // [N], added to key j's scores (non-causal only); null: none
    int64_t block = 0;                // > 0: queries see the keys of their own run of `block` tokens (T == N)
    // [T] (T == N, not causal): token i sees token j iff either is in group 0 or both share a group; null: none
    const int32_t* groups = nullptr;
};
// Whether query i sees key j under the tokens' groups (graph.h, AttentionAttrs::grouped).
inline bool group_sees(const int32_t* groups, int64_t i, int64_t j) {
    const int32_t gi = groups[i], gj = groups[j];
    return gi == 0 || gj == 0 || gi == gj;
}
// The most keys one query sees.
inline int64_t attention_visible_keys(const AttentionShape& s) {
    if (s.block > 0) return s.block;
    if (s.window <= 0) return s.kv_len;
    return std::min(s.causal ? s.window : 2 * s.window - 1, s.kv_len);
}
// The per-row kernels cover these head dims; others take the GEMM path.
inline bool attention_rows_head_dim(int64_t d) {
    return d == 16 || d == 32 || d == 64 || d == 128 || d == 192 || d == 256 || d == 512;
}

// The keys or the values of a KV cache in a quantized format (quant/kv_format.h): rows [N * Hkv], each
// quant::kv_row_bytes of codes and of scales (I8: int8 codes and an f32 scale; TurboQuant: its codes and an f16 norm
// per 128 values).
struct KvStream {
    quant::KvFormat format = quant::KvFormat::I8;
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
};
struct KvQuantized {
    KvStream k, v;
};
// The pairs of formats the kernels read: both float, both the same quantized format, or int8 keys with TurboQuant
// values (the keys decide which keys a query attends to, so they are the side worth more bits).
bool kv_formats_supported(const quant::KvFormats& f);
// The codes and scales of x [rows, width] in `format`: the bits quant::round_trip_kv decodes its values from.
void quantize_kv(sycl::queue& q, quant::KvFormat format, const float* x, uint8_t* codes, uint8_t* scales, int64_t rows,
                 int64_t width);
// out [rows, width] = the values the rows store. A TurboQuant format's stay rotated (kv_rotate): a query rotated the
// same way has the dot products of the originals, and a weighted sum of such rows is rotated back.
template <typename CT>
void dequantize_kv(sycl::queue& q, const KvStream& s, CT* out, int64_t rows, int64_t width);
// y [rows, width] = x with each group of 128 values rotated as the TurboQuant formats rotate a row, or rotated back
// (`inverse`); y may be x.
void kv_rotate(sycl::queue& q, const float* x, float* y, int64_t rows, int64_t width, bool inverse);

// How attention_rows lays a shape out: `heads` query heads a work-group (heads of one token that share a KV head, so
// each key and value is read once for all of them), and its sub-groups.
struct RowsLayout {
    int heads = 1;
    int sub_groups = 16;
};
RowsLayout attention_rows_layout(const AttentionShape& s);
// One work-group per (query, RowsLayout::heads query heads, split), an online softmax per head over its share of the
// visible keys. For few queries. split == 1 writes `out`; otherwise each work-group writes [D partial sums, max,
// denominator] per head to `part` ([T * Hq * split, D + 2]) and attention_combine merges them into `out`
// (flash-decoding for long contexts).
template <typename KV>
void attention_rows(sycl::queue& q, const float* qp, const KV* k, const KV* v, float* out, const AttentionShape& s,
                    int64_t split, float* part);
// attention_rows reading a quantized cache: the codes of each key and value, decoded in registers.
void attention_rows_quantized(sycl::queue& q, const float* qp, const KvQuantized& kv, float* out,
                              const AttentionShape& s, int64_t split, float* part);
void attention_combine(sycl::queue& q, const float* part, float* out, int64_t rows, int64_t split, int64_t head_dim);

// Fused attention on the matrix engines (sycl_flash_attention.cpp): an online softmax over blocks of keys, the
// scores never leaving registers. head_dim up to 128, k and v f16, q f32 or f16. split > 1 shares each
// row's keys between work-groups, which write [D partial sums, max, denominator] rows to `part` (attention_combine's
// layout) instead of `out`.
// The device runs the matrix-engine kernels (this one and quant_gemm): f16 8x16x16 tiles accumulating f32 on 16-lane
// sub-groups (Xe2 and later).
bool xmx_device(const sycl::device& d);
// The shape and the operands' addresses suit it (head_dim, the 2D block loads' alignment and extents).
bool flash_attention_fits(const AttentionShape& s, const void* k, const void* v);
// Work-groups per row's keys: enough to fill `compute_units`, at least two key blocks each.
int64_t flash_attention_split(const AttentionShape& s, int64_t compute_units);
template <typename QT>
void flash_attention(sycl::queue& q, const QT* qp, const half* k, const half* v, float* out, const AttentionShape& s,
                     int64_t split, float* part);

// Prefill attention for queries [t0, t0 + tc) through GEMMs, steps of attention_gemm (sycl_matmul.cpp):
// q -> [Hkv, tc, G, D] packed in CT
template <typename CT, typename QT>
void attention_pack_q(sycl::queue& q, const QT* qp, CT* packed, const AttentionShape& s, int64_t t0, int64_t tc);
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
// Or, with `block` set, a weight in a GGUF block format (sycl_block_kernels.h) and none of the other fields: the
// GEMV, dequantize and gather below read it through the block kernels, and no other kernel takes it.
struct BlockWeight;
struct QuantWeight {
    bool q4 = true;
    const uint8_t* codes = nullptr;
    const half* scales = nullptr;
    const half* biases = nullptr;
    const float* input_scales = nullptr;  // AWQ: [K], multiplies column k of the dequantized weight; or null
    int64_t group = 0;
    const BlockWeight* block = nullptr;  // on the host; kernels never follow it
};
// y [M, N] = x [M, K] @ dequant(w)^T for M <= kGemvMaxRows; K a multiple of 32.
void gemv_quant(sycl::queue& q, const float* x, const QuantWeight& w, float* y, int64_t m, int64_t n, int64_t k);
// Up to kQuantGemvParts weights that read the same x, in one launch: y_p [M, N_p] = x @ dequant(w_p)^T. The parts
// match in format and in carrying AWQ input scales (quant_gemv_parts_match); groups and N may differ.
inline constexpr int kQuantGemvParts = 4;
template <typename Y>
struct QuantPart {
    QuantWeight w;
    Y* y = nullptr;
    int64_t n = 0;
};
using QuantGemvPart = QuantPart<float>;
// Calls f.template operator()<Q4, AWQ>() for w's format and whether it carries AWQ input scales (the kernels'
// compile-time variants).
template <typename F>
void quant_dispatch(const QuantWeight& w, F&& f) {
    const bool awq = w.input_scales != nullptr;
    if (w.q4) {
        awq ? f.template operator()<true, true>() : f.template operator()<true, false>();
    } else {
        awq ? f.template operator()<false, true>() : f.template operator()<false, false>();
    }
}
bool quant_gemv_parts_match(const QuantWeight& a, const QuantWeight& b);
void gemv_quant_parts(sycl::queue& q, const float* x, const QuantGemvPart* parts, int count, int64_t m, int64_t k);
// The same on the matrix engines (sycl_quant_gemm.cpp; needs xmx_device): streaming the weight costs about the same
// at any M up to kGemvMaxRows, where the GEMV's arithmetic grows with M (it stays ahead at one or two rows). Q4 parts
// of one group size (32, 64 or 128) and AWQ-ness, N a multiple of 16 and K of 256 (gemv_quant_xmx_fits). `scratch`
// (gemv_quant_xmx_scratch bytes, 64-byte aligned) holds x in f16 and its group sums while the launch runs.
bool gemv_quant_xmx_fits(const QuantGemvPart* parts, int count, int64_t k);
size_t gemv_quant_xmx_scratch(const QuantGemvPart* parts, int count, int64_t m, int64_t k);
void gemv_quant_xmx(sycl::queue& q, const float* x, const QuantGemvPart* parts, int count, int64_t m, int64_t k,
                    int64_t compute_units, void* scratch);
// The same for kGemvMaxRows < M <= kQuantGemmMaxRows on the matrix engines (sycl_quant_gemm.cpp; needs xmx_device),
// x in f16: the weight is read once and dequantized on the way. `slices` sub-groups of a work-group divide K and add
// their partial sums in local memory (quant_gemm_slices picks enough to keep the memory busy).
inline constexpr int64_t kQuantGemmMaxRows = 64;
inline constexpr int64_t kQuantGemmMaxSlices = 8;
// The matrix-engine kernels read x by 2D block loads: its base 64-byte aligned (the *_fits checks cover the shapes and
// the weight).
bool block_load_aligned(const void* x);
bool quant_gemm_fits(int64_t m, int64_t n, int64_t k, const QuantWeight& w);
int64_t quant_gemm_slices(int64_t n, int64_t k, int64_t compute_units);
// With `add` the kernels add their product into y (a residual folded in) instead of storing it.
template <typename Y>
void quant_gemm(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                int64_t slices, bool add = false);
// Up to kQuantGemvParts weights that read the same x, in one launch (a launch costs about what streaming a megabyte
// does, so attention's q, k and v or an MLP's gate and up share one): parts that match as for the GEMV
// (quant_gemv_parts_match), each fitting quant_gemm; `slices` from quant_gemm_slices of their N together.
template <typename Y>
void quant_gemm_parts(sycl::queue& q, const half* x, const QuantPart<Y>* parts, int count, int64_t m, int64_t k,
                      int64_t slices);
// The same for any number of rows (needs xmx_device): a work-group dequantizes each weight tile once into local
// memory for 512 rows, so the weight is read once per 512 rows and never written out in f16.
bool quant_gemm_tiled_fits(int64_t m, int64_t n, int64_t k, const QuantWeight& w);
template <typename Y>
void quant_gemm_tiled(sycl::queue& q, const half* x, const QuantWeight& w, Y* y, int64_t m, int64_t n, int64_t k,
                      bool add = false);
// ExpertMatMul over a quantized expert bank (rows = E * N of K columns): out [T, S, N], x [T, K] or (per_slot)
// [T, S, K], expert ids outside [0, E) read expert 0. Up to kExpertPairsDirect (token, slot) pairs run one GEMV each;
// more are grouped by expert on the device in `grouping` (expert_grouping_bytes) so an expert's rows are read once
// per kGemvMaxRows of its pairs. The direct path covers a speculative verify (up to kGemvMaxRows tokens with eight
// experts each): its tokens share few experts, and the grouped path's 8-row GEMV streams a weight at well under half
// the one-row GEMV's rate.
inline constexpr int64_t kExpertPairsDirect = 64;
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
