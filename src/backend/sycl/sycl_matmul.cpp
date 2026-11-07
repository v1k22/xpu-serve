#include "backend/sycl/sycl_matmul.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include <oneapi/mkl/blas.hpp>

namespace xs::sycl_be {

namespace {

namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

constexpr int64_t kScoreBudget = int64_t{512} << 20;  // bytes of scores + probabilities per attention chunk
constexpr int64_t kTargetWorkGroups = 160;            // decode attention: work-groups to keep the GPU busy
constexpr int64_t kMinSplitKeys = 512;                // fewest keys worth a split of their own

size_t align_up(size_t n) {
    return (n + 255) & ~size_t{255};
}

// oneMKL orders its kernels by the dependency list, not by the queue being in-order: hand it everything submitted
// so far, and make later work wait for it.
template <typename Fn>
void ordered(sycl::queue& q, Fn&& gemm) {
    const sycl::event before = q.ext_oneapi_submit_barrier();
    const sycl::event done = gemm(std::vector<sycl::event>{before});
    q.ext_oneapi_submit_barrier({done});
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

template <typename KV>
void attention(sycl::queue& q, Scratch& scratch, const float* qp, const KV* k, const KV* v, float* out,
               const AttentionShape& s, AttentionPath path) {
    if (path == AttentionPath::Rows || (path == AttentionPath::Auto && s.tokens <= kAttentionRowsMax)) {
        // Enough work-groups to fill the GPU, each with at least kMinSplitKeys keys.
        const int64_t rows = s.tokens * s.q_heads;
        const int64_t keys = s.causal && s.window > 0 ? std::min(s.window, s.kv_len) : s.kv_len;
        const int64_t split = std::clamp<int64_t>(std::min(kTargetWorkGroups / rows, keys / kMinSplitKeys), 1, 64);
        if (split == 1) {
            attention_rows(q, qp, k, v, out, s, 1, static_cast<float*>(nullptr));
            return;
        }
        auto* part = static_cast<float*>(scratch.get(static_cast<size_t>(rows * split * (s.head_dim + 2)) * sizeof(float)));
        attention_rows(q, qp, k, v, out, s, split, part);
        attention_combine(q, part, out, rows, split, s.head_dim);
        return;
    }
    const int64_t group = s.q_heads / s.kv_heads;
    const int64_t d = s.head_dim;
    const int64_t hkv = s.kv_heads;
    const int64_t past = s.kv_len - s.tokens;
    const int64_t per_query = s.q_heads * s.kv_len * static_cast<int64_t>(sizeof(float) + sizeof(KV));
    const int64_t tc = std::clamp<int64_t>(kScoreBudget / std::max<int64_t>(per_query, 1), std::min<int64_t>(16, s.tokens),
                                           s.tokens);

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
        if (s.causal) {
            k_hi = past + t0 + cur;
            if (s.window > 0) k_lo = std::max<int64_t>(0, past + t0 - s.window + 1);
        }
        const int64_t keys = k_hi - k_lo;
        const int64_t rows = cur * group;
        attention_pack_q(q, qp, packed_q, s, t0, cur);
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm_batch(q, transpose::nontrans, transpose::trans, rows, keys, d, s.scale, packed_q, d,
                                    rows * d, k + k_lo * hkv * d, hkv * d, d, 0.0f, scores, keys, rows * keys, hkv,
                                    deps);
        });
        attention_softmax(q, scores, probs, s, t0, cur, k_lo, keys);
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm_batch(q, transpose::nontrans, transpose::nontrans, rows, d, keys, 1.0f, probs, keys,
                                    rows * keys, v + k_lo * hkv * d, hkv * d, d, 0.0f, packed_o, d, rows * d, hkv,
                                    deps);
        });
        attention_unpack(q, packed_o, out, s, t0, cur);
    }
}

template void matmul<half>(sycl::queue&, Scratch&, const float*, const half*, float*, int64_t, int64_t, int64_t);
template void matmul<float>(sycl::queue&, Scratch&, const float*, const float*, float*, int64_t, int64_t, int64_t);
template void attention<half>(sycl::queue&, Scratch&, const float*, const half*, const half*, float*,
                              const AttentionShape&, AttentionPath);
template void attention<float>(sycl::queue&, Scratch&, const float*, const float*, const float*, float*,
                               const AttentionShape&, AttentionPath);

}  // namespace xs::sycl_be
