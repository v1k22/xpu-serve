#pragma once

// Ops over rows of several a token (a vision tower's patches, R of each merged token): Attention over q [T, R, H, D]
// and Rope over x [T, R, H, D] are the same as over the T * R rows laid out [T * R, H, D]; shared by the cpu-ref and
// device op tests.

#include "backend_test_util.h"
#include "graph/graph.h"
#include "reference_ops.h"

namespace xs::test::token_rows {

constexpr int64_t kTokens = 5, kRows = 4, kHeads = 4, kKvHeads = 2, kDim = 16;

// Tensor64 views of [T, R, ...] as [T * R, ...].
inline Tensor64 flat(const Tensor64& t) {
    Tensor64 f = t;
    std::vector<int64_t> shape{t.shape[0] * t.shape[1]};
    shape.insert(shape.end(), t.shape.begin() + 2, t.shape.end());
    f.shape = shape;
    return f;
}

inline graph::Graph attention_graph(float scale) {
    graph::Graph g;
    graph::AttentionAttrs attrs;
    attrs.causal = false;
    attrs.scale = scale;
    g.output("out", g.attention(g.input("q", DType::F32, {graph::kT, kRows, kHeads, kDim}),
                                g.input("k", DType::F32, {graph::kT, kRows, kKvHeads, kDim}),
                                g.input("v", DType::F32, {graph::kT, kRows, kKvHeads, kDim}), attrs));
    return g;
}

// Attention within each token's rows only: the planted bug.
inline Tensor64 per_token(const Tensor64& q, const Tensor64& k, const Tensor64& v, const ref::AttentionSpec& spec) {
    Tensor64 out(q.shape);
    const int64_t qs = kRows * kHeads * kDim, ks = kRows * kKvHeads * kDim;
    for (int64_t t = 0; t < q.shape[0]; ++t) {
        Tensor64 qt({kRows, kHeads, kDim}), kt({kRows, kKvHeads, kDim}), vt({kRows, kKvHeads, kDim});
        for (int64_t i = 0; i < qs; ++i) qt[i] = q[t * qs + i];
        for (int64_t i = 0; i < ks; ++i) {
            kt[i] = k[t * ks + i];
            vt[i] = v[t * ks + i];
        }
        const Tensor64 o = ref::attention(qt, kt, vt, spec);
        for (int64_t i = 0; i < qs; ++i) out[t * qs + i] = o[i];
    }
    return out;
}

inline graph::Graph rope_graph(const std::vector<float>& inv_freq, bool axial) {
    graph::Graph g;
    const graph::Value freq = g.constant(make_tensor(DType::F32, {static_cast<int64_t>(inv_freq.size())}, inv_freq));
    const Dims pos = axial ? Dims{graph::kT, kRows, 2} : Dims{graph::kT, kRows};
    g.output("out", g.rope(g.input("x", DType::F32, {graph::kT, kRows, kHeads, kDim}), g.input("pos", DType::I32, pos),
                           freq));
    return g;
}

}  // namespace xs::test::token_rows
