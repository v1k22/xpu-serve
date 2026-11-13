#include "backend/sycl/sycl_matmul.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include <oneapi/mkl/blas.hpp>

#include "backend/sycl/sycl_block_kernels.h"
#include "util/error.h"

namespace xs::sycl_be {

namespace {

namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

constexpr int64_t kScoreBudget = int64_t{512} << 20;  // bytes of scores + probabilities per attention chunk
constexpr int64_t kCacheChunkQueries = 64;            // fewest queries worth a chunk that fits the cache
constexpr int64_t kTargetWorkGroups = 160;            // decode attention: work-groups to keep the GPU busy
constexpr int64_t kMinSplitKeys = 512;                // fewest keys worth a split of their own

size_t align_up(size_t n) {
    return (n + 255) & ~size_t{255};
}

// oneMKL submits to the queue it is given, so on the in-order queue its kernels run after everything submitted before
// and before everything after. No barriers around the calls: a barrier between busy kernels stalls the GPU (about
// 50 us each on Arc B390; the two per GEMM were 13 % of a pi05 call).
template <typename Fn>
void ordered(sycl::queue& q, Fn&& gemm) {
    (void)q;
    gemm(std::vector<sycl::event>{});
}

// c_h [M, N] = alpha * a_h [M, K] @ op(b_h) + beta * c_h for each of `heads` matrices a `stride` apart (op(b) is b^T,
// b [N, K], with `tb` trans; b [K, N] otherwise). One head is a plain GEMM, which launches faster than a batch of one.
template <typename A, typename C>
void gemm_heads(sycl::queue& q, transpose tb, int64_t m, int64_t n, int64_t k, float alpha, const A* a, int64_t lda,
                int64_t stride_a, const A* b, int64_t ldb, int64_t stride_b, float beta, C* c, int64_t ldc,
                int64_t stride_c, int64_t heads) {
    if (heads == 1) {
        blas::gemm(q, transpose::nontrans, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        blas::gemm_batch(q, transpose::nontrans, tb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                         stride_c, heads);
    }
}

}  // namespace

template <typename W>
void matmul(sycl::queue& q, Scratch& scratch, const float* x, const W* w, float* y, int64_t m, int64_t n, int64_t k) {
    if (m <= kGemvMaxRows) {
        gemv(q, x, w, y, m, n, k);
        return;
    }
    if constexpr (std::is_same_v<W, half>) {
        auto* xh = static_cast<half*>(scratch.get(static_cast<size_t>(m * k) * sizeof(half)));
        to_half(q, x, xh, m * k);
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, xh, k, w, k, 0.0f, y, n, deps);
        });
    } else {
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, x, k, w, k, 0.0f, y, n, deps);
        });
    }
}

template <typename X>
void matmul_accumulate(sycl::queue& q, Scratch& scratch, const X* x, const half* w, float* y, int64_t m, int64_t n,
                       int64_t k) {
    const half* xh = nullptr;
    if constexpr (std::is_same_v<X, half>) {
        xh = x;
    } else {
        auto* t = static_cast<half*>(scratch.get(static_cast<size_t>(m * k) * sizeof(half)));
        to_half(q, x, t, m * k);
        xh = t;
    }
    ordered(q, [&](const std::vector<sycl::event>& deps) {
        return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, xh, k, w, k, 1.0f, y, n, deps);
    });
}

template void matmul_accumulate<float>(sycl::queue&, Scratch&, const float*, const half*, float*, int64_t, int64_t,
                                       int64_t);
template void matmul_accumulate<half>(sycl::queue&, Scratch&, const half*, const half*, float*, int64_t, int64_t,
                                      int64_t);

void matmul_half_x(sycl::queue& q, const half* x, const half* w, float* y, int64_t m, int64_t n, int64_t k) {
    ordered(q, [&](const std::vector<sycl::event>& deps) {
        return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, x, k, w, k, 0.0f, y, n, deps);
    });
}

template <typename X>
void matmul_half_y(sycl::queue& q, Scratch& scratch, const X* x, const half* w, half* y, int64_t m, int64_t n, int64_t k) {
    const half* xh = nullptr;
    if constexpr (std::is_same_v<X, half>) {
        xh = x;
    } else {
        auto* conv = static_cast<half*>(scratch.get(static_cast<size_t>(m * k) * sizeof(half)));
        to_half(q, x, conv, m * k);
        xh = conv;
    }
    ordered(q, [&](const std::vector<sycl::event>& deps) {
        return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, half(1.0f), xh, k, w, k, half(0.0f), y, n,
                          deps);
    });
}

template void matmul_half_y<float>(sycl::queue&, Scratch&, const float*, const half*, half*, int64_t, int64_t, int64_t);
template void matmul_half_y<half>(sycl::queue&, Scratch&, const half*, const half*, half*, int64_t, int64_t, int64_t);

bool xmx_gemv(int64_t m, const QuantGemvPart* parts, int count, int64_t k, const QuantMatmulPlan& plan) {
    return !plan.f32 && plan.xmx_gemv_rows > 0 && m >= plan.xmx_gemv_rows && m <= kGemvMaxRows &&
           gemv_quant_xmx_fits(parts, count, k);
}

QuantPath quant_path(int64_t m, int64_t n, int64_t k, const QuantWeight& w, const QuantMatmulPlan& plan) {
    if (w.block != nullptr) {
        if (m > kGemvMaxRows) return QuantPath::Dequant;
        const bool xmx = !plan.f32 && plan.xmx_block_rows > 0 && m >= plan.xmx_block_rows &&
                         gemv_block_xmx_fits(*w.block, k);
        return xmx ? QuantPath::GemvXmx : QuantPath::Gemv;
    }
    if (m <= kGemvMaxRows) {
        const QuantGemvPart part{w, nullptr, n};
        return xmx_gemv(m, &part, 1, k, plan) ? QuantPath::GemvXmx : QuantPath::Gemv;
    }
    if (plan.f32) return QuantPath::Dequant;
    if (m <= plan.few_rows && quant_gemm_fits(m, n, k, w)) return QuantPath::FewRows;
    if (m <= plan.tiled_rows && quant_gemm_tiled_fits(m, n, k, w)) return QuantPath::Tiled;
    return QuantPath::Dequant;
}

template <typename X, typename Y>
void matmul_quant(sycl::queue& q, Scratch& scratch, const X* x, const QuantWeight& w, Y* y, int64_t m, int64_t n,
                  int64_t k, const QuantMatmulPlan& plan, bool add) {
    constexpr bool kHalfX = std::is_same_v<X, half>;
    constexpr bool kHalfY = std::is_same_v<Y, half>;
    QuantPath path = quant_path(m, n, k, w, plan);
    const bool gemv = path == QuantPath::Gemv || path == QuantPath::GemvXmx;
    XS_CHECK(!gemv || (!kHalfX && !kHalfY && !add), "sycl matmul_quant: the GEMVs take f32 x and y");
    XS_CHECK(!add || !kHalfY, "sycl matmul_quant: adding into an f16 y");
    if (gemv) {
        if constexpr (!kHalfX && !kHalfY) {
            if (path == QuantPath::GemvXmx && w.block != nullptr) {
                gemv_block_xmx(q, x, *w.block, y, m, n, k, plan.compute_units,
                               scratch.get(gemv_block_xmx_scratch(m, k)));
            } else if (path == QuantPath::GemvXmx) {
                const QuantGemvPart part{w, y, n};
                gemv_quant_xmx(q, x, &part, 1, m, k, plan.compute_units,
                               scratch.get(gemv_quant_xmx_scratch(&part, 1, m, k)));
            } else {
                if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {  // the GEMV loads x in float4s
                    auto* aligned = static_cast<float*>(scratch.get(static_cast<size_t>(m * k) * sizeof(float)));
                    q.memcpy(aligned, x, static_cast<size_t>(m * k) * sizeof(float));
                    x = aligned;
                }
                gemv_quant(q, x, w, y, m, n, k);
            }
        }
        return;
    }
    const float beta = add ? 1.0f : 0.0f;
    if (plan.f32) {  // the weight dequantized to f32 for an f32 GEMM
        XS_CHECK(!kHalfX && !kHalfY, "sycl matmul_quant: f16 values in f32 precision");
        if constexpr (!kHalfX && !kHalfY) {
            auto* wf = static_cast<float*>(scratch.get(static_cast<size_t>(n * k) * sizeof(float)));
            dequantize(q, w, wf, n, k);
            ordered(q, [&](const std::vector<sycl::event>& deps) {
                return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, x, k, wf, k, beta, y, n, deps);
            });
        }
        return;
    }
    // the f16 paths: an f32 x is converted into scratch (aligned), behind the dequantized weight when there is one
    if constexpr (kHalfX) {
        if (path != QuantPath::Dequant && !block_load_aligned(x)) path = QuantPath::Dequant;
    }
    const size_t w_bytes = path == QuantPath::Dequant ? align_up(static_cast<size_t>(n * k) * sizeof(half)) : 0;
    const size_t x_bytes = kHalfX ? 0 : static_cast<size_t>(m * k) * sizeof(half);
    auto* base = static_cast<char*>(scratch.get(w_bytes + x_bytes));
    const half* xh = nullptr;
    if constexpr (kHalfX) {
        xh = x;
    } else {
        auto* conv = reinterpret_cast<half*>(base + w_bytes);
        to_half(q, x, conv, m * k);
        xh = conv;
    }
    switch (path) {
        case QuantPath::FewRows:
            quant_gemm(q, xh, w, y, m, n, k, quant_gemm_slices(n, k, plan.compute_units), add);
            return;
        case QuantPath::Tiled:
            quant_gemm_tiled(q, xh, w, y, m, n, k, add);
            return;
        default: {
            auto* wh = reinterpret_cast<half*>(base);
            dequantize(q, w, wh, n, k);
            ordered(q, [&](const std::vector<sycl::event>& deps) {
                if constexpr (kHalfY) {
                    return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, half(1.0f), xh, k, wh, k,
                                      half(0.0f), y, n, deps);
                } else {
                    return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, xh, k, wh, k, beta, y, n,
                                      deps);
                }
            });
        }
    }
}

template <typename Y>
bool quant_parts_few_rows(int64_t m, const QuantPart<Y>* parts, int count, int64_t k, const QuantMatmulPlan& plan) {
    if (count < 2 || count > kQuantGemvParts) return false;
    for (int j = 0; j < count; ++j) {
        if (quant_path(m, parts[j].n, k, parts[j].w, plan) != QuantPath::FewRows ||
            !quant_gemv_parts_match(parts[j].w, parts[0].w))
            return false;
    }
    return true;
}

template <typename X, typename Y>
void matmul_quant_parts(sycl::queue& q, Scratch& scratch, const X* x, const QuantPart<Y>* parts, int count, int64_t m,
                        int64_t k, const QuantMatmulPlan& plan) {
    XS_CHECK(quant_parts_few_rows(m, parts, count, k, plan),
             "sycl matmul_quant_parts: the parts do not share a launch");
    const half* xh = nullptr;
    if constexpr (std::is_same_v<X, half>) {
        if (!block_load_aligned(x)) {
            for (int j = 0; j < count; ++j) matmul_quant(q, scratch, x, parts[j].w, parts[j].y, m, parts[j].n, k, plan);
            return;
        }
        xh = x;
    } else {
        auto* conv = static_cast<half*>(scratch.get(static_cast<size_t>(m * k) * sizeof(half)));
        to_half(q, x, conv, m * k);
        xh = conv;
    }
    int64_t n = 0;
    for (int j = 0; j < count; ++j) n += parts[j].n;
    quant_gemm_parts(q, xh, parts, count, m, k, quant_gemm_slices(n, k, plan.compute_units));
}

template bool quant_parts_few_rows<float>(int64_t, const QuantPart<float>*, int, int64_t, const QuantMatmulPlan&);
template bool quant_parts_few_rows<half>(int64_t, const QuantPart<half>*, int, int64_t, const QuantMatmulPlan&);
template void matmul_quant_parts<float, float>(sycl::queue&, Scratch&, const float*, const QuantPart<float>*, int,
                                               int64_t, int64_t, const QuantMatmulPlan&);
template void matmul_quant_parts<half, float>(sycl::queue&, Scratch&, const half*, const QuantPart<float>*, int,
                                              int64_t, int64_t, const QuantMatmulPlan&);
template void matmul_quant_parts<float, half>(sycl::queue&, Scratch&, const float*, const QuantPart<half>*, int,
                                              int64_t, int64_t, const QuantMatmulPlan&);
template void matmul_quant_parts<half, half>(sycl::queue&, Scratch&, const half*, const QuantPart<half>*, int, int64_t,
                                             int64_t, const QuantMatmulPlan&);

template void matmul_quant<float, float>(sycl::queue&, Scratch&, const float*, const QuantWeight&, float*, int64_t,
                                         int64_t, int64_t, const QuantMatmulPlan&, bool);
template void matmul_quant<half, float>(sycl::queue&, Scratch&, const half*, const QuantWeight&, float*, int64_t,
                                        int64_t, int64_t, const QuantMatmulPlan&, bool);
template void matmul_quant<float, half>(sycl::queue&, Scratch&, const float*, const QuantWeight&, half*, int64_t,
                                        int64_t, int64_t, const QuantMatmulPlan&, bool);
template void matmul_quant<half, half>(sycl::queue&, Scratch&, const half*, const QuantWeight&, half*, int64_t,
                                       int64_t, int64_t, const QuantMatmulPlan&, bool);

namespace {

bool rows_path(const AttentionShape& s, AttentionPath path) {
    if (!attention_rows_head_dim(s.head_dim)) return false;
    return path == AttentionPath::Rows || (path == AttentionPath::Auto && s.tokens <= kAttentionRowsMax);
}

// The rows path: enough work-groups to fill the GPU, each with at least kMinSplitKeys keys. `run(split, part)`
// launches the per-row kernel.
template <typename Run>
void attention_by_rows(sycl::queue& q, Scratch& scratch, float* out, const AttentionShape& s, const Run& run) {
    const int64_t rows = s.tokens * s.q_heads;
    const int64_t keys = attention_visible_keys(s);
    const int64_t split = std::clamp<int64_t>(std::min(kTargetWorkGroups / rows, keys / kMinSplitKeys), 1, 64);
    if (split == 1) {
        run(int64_t{1}, static_cast<float*>(nullptr));
        return;
    }
    auto* part = static_cast<float*>(scratch.get(static_cast<size_t>(rows * split * (s.head_dim + 2)) * sizeof(float)));
    run(split, part);
    attention_combine(q, part, out, rows, split, s.head_dim);
}

}  // namespace

bool attention_matrix_path(const AttentionShape& s, AttentionPath path) { return !rows_path(s, path); }

template <typename KV, typename QT>
void attention(sycl::queue& q, Scratch& scratch, const QT* qp, const KV* k, const KV* v, float* out,
               const AttentionShape& s, const AttentionPlan& plan, const KvSecond& second) {
    XS_CHECK(second.k == nullptr || (!rows_path(s, plan.path) && s.head_dim > 128),
             "sycl attention: keys in two parts take the GEMM path only");
    bool fused = false;  // the fused matrix-engine kernel runs this shape
    if constexpr (std::is_same_v<KV, half>) {
        const bool want = plan.path == AttentionPath::Auto || plan.path == AttentionPath::Flash;
        fused = want && plan.xmx && second.k == nullptr && !rows_path(s, plan.path) && flash_attention_fits(s, k, v);
    }
    if (s.block > 0 && !fused) {  // the fused kernel takes every block in one launch; the other paths one at a time
        XS_CHECK(second.k == nullptr && s.tokens == s.kv_len, "sycl attention: blocks need as many keys as queries");
        AttentionShape one = s;
        one.block = 0;
        one.tokens = one.kv_len = s.block;
        for (int64_t b0 = 0; b0 < s.tokens; b0 += s.block) {
            one.key_bias = s.key_bias != nullptr ? s.key_bias + b0 : nullptr;
            attention(q, scratch, qp + b0 * s.q_heads * s.head_dim, k + b0 * s.kv_heads * s.head_dim,
                      v + b0 * s.kv_heads * s.head_dim, out + b0 * s.q_heads * s.head_dim, one, plan);
        }
        return;
    }
    if (rows_path(s, plan.path)) {
        if constexpr (std::is_same_v<QT, float>) {
            attention_by_rows(q, scratch, out, s,
                              [&](int64_t split, float* part) { attention_rows(q, qp, k, v, out, s, split, part); });
            return;
        } else {
            fail("sycl attention: an f16 query takes a matrix path only");
        }
    }
    if constexpr (std::is_same_v<KV, half>) {
        if (fused) {
            const int64_t split = flash_attention_split(s, plan.compute_units);
            float* part = nullptr;
            if (split > 1) {
                part = static_cast<float*>(
                    scratch.get(static_cast<size_t>(s.tokens * s.q_heads * split * (s.head_dim + 2)) * sizeof(float)));
            }
            flash_attention(q, qp, k, v, out, s, split, part);
            if (split > 1) attention_combine(q, part, out, s.tokens * s.q_heads, split, s.head_dim);
            return;
        }
    }
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t d = s.head_dim;
    const int64_t hkv = s.kv_heads;
    const int64_t past = s.kv_len - s.tokens;
    const int64_t per_query =
        std::max<int64_t>(s.q_heads * s.kv_len * static_cast<int64_t>(sizeof(float) + sizeof(KV)), 1);
    // Queries a chunk: as many as the scratch budget holds, or, where the device's memory is the bound, as fit its
    // cache (AttentionPlan::score_cache) when that is a chunk worth its three launches.
    const int64_t budget = plan.score_cache / per_query >= kCacheChunkQueries ? plan.score_cache : kScoreBudget;
    const int64_t tc = std::clamp<int64_t>(budget / per_query, std::min<int64_t>(16, s.tokens), s.tokens);

    const size_t q_bytes = align_up(static_cast<size_t>(hkv * tc * group * d) * sizeof(KV));
    const size_t s_bytes = align_up(static_cast<size_t>(hkv * tc * group * s.kv_len) * sizeof(float));
    const size_t p_bytes = align_up(static_cast<size_t>(hkv * tc * group * s.kv_len) * sizeof(KV));
    const size_t o_bytes = align_up(static_cast<size_t>(hkv * tc * group * d) * sizeof(float));
    auto* base = static_cast<char*>(scratch.get(q_bytes + s_bytes + p_bytes + o_bytes));
    auto* packed_q = reinterpret_cast<KV*>(base);
    auto* scores = reinterpret_cast<float*>(base + q_bytes);
    auto* probs = reinterpret_cast<KV*>(base + q_bytes + s_bytes);
    auto* packed_o = reinterpret_cast<float*>(base + q_bytes + s_bytes + p_bytes);

    for (int64_t t0 = 0; t0 < s.tokens; t0 += tc) {
        const int64_t cur = std::min(tc, s.tokens - t0);
        int64_t k_lo = 0;
        int64_t k_hi = s.kv_len;
        if (s.causal) k_hi = past + t0 + cur;
        if (s.window > 0) {
            k_lo = std::max<int64_t>(0, past + t0 - s.window + 1);
            k_hi = std::min<int64_t>(k_hi, past + t0 + cur - 1 + s.window);
        }
        const int64_t keys = k_hi - k_lo;
        const int64_t rows = cur * group;
        // The keys [k_lo, k_hi) in at most two parts: [k_lo, a_hi) of k / v, then [b_lo, k_hi) of the second.
        const int64_t first = second.k != nullptr ? second.first_len : s.kv_len;
        const int64_t n1 = std::max<int64_t>(0, std::min(k_hi, first) - k_lo);
        const int64_t b_lo = std::max(k_lo, first);
        const int64_t n2 = std::max<int64_t>(0, k_hi - b_lo);
        const KV* k2 = static_cast<const KV*>(second.k) + (b_lo - first) * hkv * d;
        const KV* v2 = static_cast<const KV*>(second.v) + (b_lo - first) * hkv * d;
        // One KV head and q already in KV's type: q's rows t0.. are the packed layout [1, cur, G, D].
        const KV* qa = packed_q;
        if constexpr (std::is_same_v<QT, KV>) {
            if (hkv == 1) qa = qp + t0 * s.q_heads * d;
        }
        if (qa == packed_q) attention_pack_q(q, qp, packed_q, s, t0, cur);
        if (n1 > 0) {
            gemm_heads(q, transpose::trans, rows, n1, d, s.scale, qa, d, rows * d, k + k_lo * hkv * d, hkv * d, d, 0.0f,
                       scores, keys, rows * keys, hkv);
        }
        if (n2 > 0) {  // the second part's scores: the columns after the first's
            gemm_heads(q, transpose::trans, rows, n2, d, s.scale, qa, d, rows * d, k2, hkv * d, d, 0.0f, scores + n1,
                       keys, rows * keys, hkv);
        }
        attention_softmax(q, scores, probs, s, t0, cur, k_lo, keys);
        // One KV head: the packed layout [1, cur, G, D] is the output's rows t0.. already, so the GEMM writes there.
        float* o = hkv == 1 ? out + t0 * s.q_heads * d : packed_o;
        if (n1 > 0) {
            gemm_heads(q, transpose::nontrans, rows, d, n1, 1.0f, probs, keys, rows * keys, v + k_lo * hkv * d, hkv * d,
                       d, 0.0f, o, d, rows * d, hkv);
        }
        if (n2 > 0) {  // adds the second part's P V
            gemm_heads(q, transpose::nontrans, rows, d, n2, 1.0f, probs + n1, keys, rows * keys, v2, hkv * d, d,
                       n1 > 0 ? 1.0f : 0.0f, o, d, rows * d, hkv);
        }
        if (hkv != 1) attention_unpack(q, packed_o, out, s, t0, cur);
    }
}

template <typename CT>
void attention_i8(sycl::queue& q, Scratch& scratch, Scratch& kv_scratch, const float* qp, const KvI8& kv, float* out,
                  const AttentionShape& s, const AttentionPlan& plan) {
    if (rows_path(s, plan.path)) {
        attention_by_rows(q, scratch, out, s,
                          [&](int64_t split, float* part) { attention_rows_i8(q, qp, kv, out, s, split, part); });
        return;
    }
    // Keys before `first` are outside every query's window: drop them, which shifts all positions alike.
    const int64_t past = s.kv_len - s.tokens;
    const int64_t first = s.window > 0 ? std::max<int64_t>(0, past - s.window + 1) : 0;
    const int64_t rows = (s.kv_len - first) * s.kv_heads;
    const size_t half_bytes = align_up(static_cast<size_t>(rows * s.head_dim) * sizeof(CT));
    auto* k = static_cast<CT*>(kv_scratch.get(2 * half_bytes));
    auto* v = reinterpret_cast<CT*>(reinterpret_cast<char*>(k) + half_bytes);
    const int64_t skip = first * s.kv_heads;
    dequantize_kv_i8(q, kv.k + skip * s.head_dim, kv.k_scale + skip, k, rows, s.head_dim);
    dequantize_kv_i8(q, kv.v + skip * s.head_dim, kv.v_scale + skip, v, rows, s.head_dim);
    AttentionShape visible = s;
    visible.kv_len = s.kv_len - first;
    attention(q, scratch, qp, k, v, out, visible, plan);
}

namespace {

// Rows [first, first + count) of a quantized weight, as a weight of its own.
QuantWeight quant_rows(const QuantWeight& w, int64_t first, int64_t k) {
    QuantWeight r = w;
    const int64_t row_bytes = w.q4 ? k / 2 : k;
    const int64_t groups = k / w.group;
    r.codes = w.codes + first * row_bytes;
    r.scales = w.scales + first * groups;
    if (w.biases != nullptr) r.biases = w.biases + first * groups;
    return r;
}

template <typename CT, typename I>
void expert_gemm_t(sycl::queue& q, Scratch& scratch, const float* x, bool per_slot, const QuantWeight& w, const I* idx,
                   float* out, int64_t pairs, int64_t slots, int64_t experts, int64_t n, int64_t k) {
    const int64_t expert_bytes = n * k * static_cast<int64_t>(sizeof(CT));
    const int64_t batch = std::max<int64_t>(1, static_cast<int64_t>(kExpertDequantBytes) / expert_bytes);
    const size_t g_bytes = align_up(expert_grouping_bytes(pairs, experts));
    const size_t x_bytes = align_up(static_cast<size_t>(pairs * k) * sizeof(CT));
    const size_t y_bytes = align_up(static_cast<size_t>(pairs * n) * sizeof(float));
    const size_t w_bytes = align_up(static_cast<size_t>(std::min(batch, experts) * expert_bytes));
    auto* base = static_cast<char*>(scratch.get(g_bytes + x_bytes + y_bytes + w_bytes));
    auto* grouping = reinterpret_cast<int32_t*>(base);
    auto* rows = reinterpret_cast<CT*>(base + g_bytes);
    auto* y = reinterpret_cast<float*>(base + g_bytes + x_bytes);
    auto* wd = reinterpret_cast<CT*>(base + g_bytes + x_bytes + y_bytes);
    // Per group: m, n, k, lda, ldb, ldc, size (int64), transa, transb, alpha, beta, and the a / b / c pointers.
    auto* params = static_cast<char*>(scratch.get_host(static_cast<size_t>(experts) * 10 * sizeof(int64_t)));
    const auto column = [&](int i) { return params + static_cast<size_t>(i * experts) * sizeof(int64_t); };
    auto* pm = reinterpret_cast<int64_t*>(column(0));
    auto* pn = reinterpret_cast<int64_t*>(column(1));
    auto* pk = reinterpret_cast<int64_t*>(column(2));
    auto* plda = reinterpret_cast<int64_t*>(column(3));
    auto* pldb = reinterpret_cast<int64_t*>(column(4));
    auto* pldc = reinterpret_cast<int64_t*>(column(5));
    auto* psize = reinterpret_cast<int64_t*>(column(6));
    auto* pa = reinterpret_cast<const CT**>(column(7));
    auto* pb = reinterpret_cast<const CT**>(column(8));
    auto* pc = reinterpret_cast<float**>(column(9));
    std::vector<transpose> ta(static_cast<size_t>(experts), transpose::nontrans);
    std::vector<transpose> tb(static_cast<size_t>(experts), transpose::trans);
    std::vector<float> alpha(static_cast<size_t>(experts), 1.0f), beta(static_cast<size_t>(experts), 0.0f);

    expert_group(q, idx, pairs, experts, grouping);
    const int32_t* order = expert_order(grouping, experts);
    expert_gather_rows(q, x, per_slot, slots, order, pairs, k, rows);
    std::vector<int32_t> offsets(static_cast<size_t>(experts + 1));
    q.memcpy(offsets.data(), expert_offsets(grouping, experts), offsets.size() * sizeof(int32_t)).wait();

    for (int64_t e0 = 0; e0 < experts; e0 += batch) {
        const int64_t e1 = std::min(experts, e0 + batch);
        // Dequantize the runs of experts that have pairs; set up one group per such expert.
        int64_t groups = 0;
        for (int64_t e = e0; e < e1;) {
            if (offsets[static_cast<size_t>(e + 1)] == offsets[static_cast<size_t>(e)]) {
                ++e;
                continue;
            }
            int64_t run_end = e;
            while (run_end < e1 && offsets[static_cast<size_t>(run_end + 1)] > offsets[static_cast<size_t>(run_end)])
                ++run_end;
            dequantize(q, quant_rows(w, e * n, k), wd + (e - e0) * n * k, (run_end - e) * n, k);
            for (; e < run_end; ++e) {
                const int64_t first = offsets[static_cast<size_t>(e)];
                pm[groups] = offsets[static_cast<size_t>(e + 1)] - first;
                pn[groups] = n;
                pk[groups] = k;
                plda[groups] = k;
                pldb[groups] = k;
                pldc[groups] = n;
                psize[groups] = 1;
                pa[groups] = rows + first * k;
                pb[groups] = wd + (e - e0) * n * k;
                pc[groups] = y + first * n;
                ++groups;
            }
        }
        if (groups == 0) continue;
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm_batch(q, ta.data(), tb.data(), pm, pn, pk, alpha.data(), pa, plda, pb, pldb, beta.data(), pc,
                                    pldc, groups, psize, deps);
        });
        q.wait();  // the next batch rewrites the dequantized weights and the group arrays
    }
    expert_scatter_rows(q, y, order, pairs, n, out);
}

}  // namespace

template <typename I>
void expert_matmul_quant_gemm(sycl::queue& q, Scratch& scratch, const float* x, bool per_slot, const QuantWeight& w,
                              const I* idx, float* out, int64_t tokens, int64_t slots, int64_t experts, int64_t n,
                              int64_t k, bool f32) {
    if (f32) {
        expert_gemm_t<float>(q, scratch, x, per_slot, w, idx, out, tokens * slots, slots, experts, n, k);
    } else {
        expert_gemm_t<half>(q, scratch, x, per_slot, w, idx, out, tokens * slots, slots, experts, n, k);
    }
}

template void expert_matmul_quant_gemm<int32_t>(sycl::queue&, Scratch&, const float*, bool, const QuantWeight&,
                                                const int32_t*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                bool);
template void expert_matmul_quant_gemm<int64_t>(sycl::queue&, Scratch&, const float*, bool, const QuantWeight&,
                                                const int64_t*, float*, int64_t, int64_t, int64_t, int64_t, int64_t,
                                                bool);

template void matmul<half>(sycl::queue&, Scratch&, const float*, const half*, float*, int64_t, int64_t, int64_t);
template void matmul<float>(sycl::queue&, Scratch&, const float*, const float*, float*, int64_t, int64_t, int64_t);
template void attention<half, float>(sycl::queue&, Scratch&, const float*, const half*, const half*, float*,
                                     const AttentionShape&, const AttentionPlan&, const KvSecond&);
template void attention<float, float>(sycl::queue&, Scratch&, const float*, const float*, const float*, float*,
                                      const AttentionShape&, const AttentionPlan&, const KvSecond&);
template void attention<half, half>(sycl::queue&, Scratch&, const half*, const half*, const half*, float*,
                                    const AttentionShape&, const AttentionPlan&, const KvSecond&);
template void attention<float, half>(sycl::queue&, Scratch&, const half*, const float*, const float*, float*,
                                     const AttentionShape&, const AttentionPlan&, const KvSecond&);
template void attention_i8<half>(sycl::queue&, Scratch&, Scratch&, const float*, const KvI8&, float*,
                                 const AttentionShape&, const AttentionPlan&);
template void attention_i8<float>(sycl::queue&, Scratch&, Scratch&, const float*, const KvI8&, float*,
                                  const AttentionShape&, const AttentionPlan&);

}  // namespace xs::sycl_be
