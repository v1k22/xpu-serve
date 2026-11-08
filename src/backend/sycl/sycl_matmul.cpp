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

void matmul_quant(sycl::queue& q, Scratch& scratch, const float* x, const QuantWeight& w, float* y, int64_t m,
                  int64_t n, int64_t k, bool f32) {
    if (m <= kGemvMaxRows) {
        if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {  // the kernel loads x in float4s
            auto* aligned = static_cast<float*>(scratch.get(static_cast<size_t>(m * k) * sizeof(float)));
            q.memcpy(aligned, x, static_cast<size_t>(m * k) * sizeof(float));
            x = aligned;
        }
        gemv_quant(q, x, w, y, m, n, k);
        return;
    }
    if (f32) {
        auto* wf = static_cast<float*>(scratch.get(static_cast<size_t>(n * k) * sizeof(float)));
        dequantize(q, w, wf, n, k);
        ordered(q, [&](const std::vector<sycl::event>& deps) {
            return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, x, k, wf, k, 0.0f, y, n, deps);
        });
        return;
    }
    const size_t w_bytes = align_up(static_cast<size_t>(n * k) * sizeof(half));
    auto* base = static_cast<char*>(scratch.get(w_bytes + static_cast<size_t>(m * k) * sizeof(half)));
    auto* wh = reinterpret_cast<half*>(base);
    auto* xh = reinterpret_cast<half*>(base + w_bytes);
    dequantize(q, w, wh, n, k);
    to_half(q, x, xh, m * k);
    ordered(q, [&](const std::vector<sycl::event>& deps) {
        return blas::gemm(q, transpose::nontrans, transpose::trans, m, n, k, 1.0f, xh, k, wh, k, 0.0f, y, n, deps);
    });
}

namespace {

bool rows_path(const AttentionShape& s, AttentionPath path) {
    return path == AttentionPath::Rows || (path == AttentionPath::Auto && s.tokens <= kAttentionRowsMax);
}

// The rows path: enough work-groups to fill the GPU, each with at least kMinSplitKeys keys. `run(split, part)`
// launches the per-row kernel.
template <typename Run>
void attention_by_rows(sycl::queue& q, Scratch& scratch, float* out, const AttentionShape& s, const Run& run) {
    const int64_t rows = s.tokens * s.q_heads;
    const int64_t keys = s.causal && s.window > 0 ? std::min(s.window, s.kv_len) : s.kv_len;
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

template <typename KV>
void attention(sycl::queue& q, Scratch& scratch, const float* qp, const KV* k, const KV* v, float* out,
               const AttentionShape& s, AttentionPath path) {
    if (rows_path(s, path)) {
        attention_by_rows(q, scratch, out, s,
                          [&](int64_t split, float* part) { attention_rows(q, qp, k, v, out, s, split, part); });
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

template <typename CT>
void attention_i8(sycl::queue& q, Scratch& scratch, Scratch& kv_scratch, const float* qp, const KvI8& kv, float* out,
                  const AttentionShape& s, AttentionPath path) {
    if (rows_path(s, path)) {
        attention_by_rows(q, scratch, out, s,
                          [&](int64_t split, float* part) { attention_rows_i8(q, qp, kv, out, s, split, part); });
        return;
    }
    // Keys before `first` are outside every query's window: drop them, which shifts all positions alike.
    const int64_t past = s.kv_len - s.tokens;
    const int64_t first = s.causal && s.window > 0 ? std::max<int64_t>(0, past - s.window + 1) : 0;
    const int64_t rows = (s.kv_len - first) * s.kv_heads;
    const size_t half_bytes = align_up(static_cast<size_t>(rows * s.head_dim) * sizeof(CT));
    auto* k = static_cast<CT*>(kv_scratch.get(2 * half_bytes));
    auto* v = reinterpret_cast<CT*>(reinterpret_cast<char*>(k) + half_bytes);
    const int64_t skip = first * s.kv_heads;
    dequantize_kv_i8(q, kv.k + skip * s.head_dim, kv.k_scale + skip, k, rows, s.head_dim);
    dequantize_kv_i8(q, kv.v + skip * s.head_dim, kv.v_scale + skip, v, rows, s.head_dim);
    AttentionShape visible = s;
    visible.kv_len = s.kv_len - first;
    attention(q, scratch, qp, k, v, out, visible, AttentionPath::Gemm);
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
template void attention<half>(sycl::queue&, Scratch&, const float*, const half*, const half*, float*,
                              const AttentionShape&, AttentionPath);
template void attention<float>(sycl::queue&, Scratch&, const float*, const float*, const float*, float*,
                               const AttentionShape&, AttentionPath);
template void attention_i8<half>(sycl::queue&, Scratch&, Scratch&, const float*, const KvI8&, float*,
                                 const AttentionShape&, AttentionPath);
template void attention_i8<float>(sycl::queue&, Scratch&, Scratch&, const float*, const KvI8&, float*,
                                  const AttentionShape&, AttentionPath);

}  // namespace xs::sycl_be
