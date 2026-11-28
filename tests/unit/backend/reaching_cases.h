#pragma once

// Causal attention reaching ahead (graph.h AttentionAttrs::reaching: an image's tokens attending to each other both
// ways, Gemma 4's use_bidirectional_attention "vision"): steps through a KV cache whose rows carry their reach, text
// rows causal; shared by the cpu-ref and device op tests. A step's runs are images: each row of one sees to the run's
// end. With a window the keys behind a row are bounded and those ahead are not (planted: bounded both ways).

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "backend_test_util.h"
#include "graph/graph.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::reaching {

struct Case {
    int64_t heads = 4, kv_heads = 2, head_dim = 64;
    int64_t window = 0;
    // A step: its length and its image runs, [first, end) each.
    struct Step {
        int64_t tokens;
        std::vector<std::pair<int64_t, int64_t>> runs;
    };
    std::vector<Step> steps;
};

inline graph::Graph graph_of(const Case& c) {
    graph::Graph g;
    const int slot = g.add_kv_slot(c.kv_heads, c.head_dim, c.window, "layer");
    const graph::Value q = g.input("q", DType::F32, {graph::kT, c.heads, c.head_dim});
    const auto [k_all, v_all] = g.kv_cache(slot, g.input("k", DType::F32, {graph::kT, c.kv_heads, c.head_dim}),
                                           g.input("v", DType::F32, {graph::kT, c.kv_heads, c.head_dim}));
    graph::AttentionAttrs a;
    a.scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
    a.sliding_window = c.window;
    a.reaching = true;
    g.output("out", g.attention_reaching(q, k_all, v_all, g.input("reach", DType::I32, {graph::kT}), a));
    return g;
}

// Each row's reach: to its run's end inside an image run, 0 for text.
inline std::vector<int32_t> reach_of(const Case::Step& s) {
    std::vector<int32_t> r(static_cast<size_t>(s.tokens), 0);
    for (const auto& [first, end] : s.runs) {
        for (int64_t i = first; i < end; ++i) r[static_cast<size_t>(i)] = static_cast<int32_t>(end - 1 - i);
    }
    return r;
}

// Runs every step through `session`; `check(step, out, expected, planted)` compares each step's output with the
// reference and with the planted reference (the window bounding the keys ahead too).
inline void run(const Case& c, Session& session, uint32_t seed,
                const std::function<void(size_t, const HostTensor&, const Tensor64&, const Tensor64&)>& check) {
    TensorRng rng(seed);
    Tensor64 k_all, v_all;
    const auto append = [](Tensor64& all, const Tensor64& rows) {
        if (all.shape.empty()) {
            all = rows;
            return;
        }
        all.shape[0] += rows.shape[0];
        all.data.insert(all.data.end(), rows.data.begin(), rows.data.end());
    };
    for (size_t i = 0; i < c.steps.size(); ++i) {
        const Case::Step& s = c.steps[i];
        const HostTensor q = rng.uniform(DType::F32, {s.tokens, c.heads, c.head_dim}, 2.0f);
        const HostTensor k = rng.uniform(DType::F32, {s.tokens, c.kv_heads, c.head_dim}, 2.0f);
        const HostTensor v = rng.uniform(DType::F32, {s.tokens, c.kv_heads, c.head_dim}, 2.0f);
        append(k_all, to_f64(k));
        append(v_all, to_f64(v));
        const std::vector<int32_t> reach = reach_of(s);
        TensorMap out;
        session.run({{"q", q}, {"k", k}, {"v", v}, {"reach", make_tensor(DType::I32, {s.tokens}, reach)}}, out);
        ref::AttentionSpec spec;
        spec.scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
        spec.sliding_window = c.window;
        spec.reach = reach;
        const Tensor64 expected = ref::attention(to_f64(q), k_all, v_all, spec);
        spec.reach_bounded = true;
        check(i, out.at("out"), expected, ref::attention(to_f64(q), k_all, v_all, spec));
    }
}

// Images of 7 and 8 rows among text, over a cache: unwindowed, and with a window shorter than the second image (its
// first rows reach past the window ahead of them).
inline std::vector<Case> cases() {
    Case open;
    open.steps = {{5, {}}, {20, {{2, 9}, {11, 19}}}, {1, {}}, {3, {{0, 3}}}};
    Case windowed = open;
    windowed.window = 6;
    Case one_kv = open;
    one_kv.heads = 8;
    one_kv.kv_heads = 1;
    one_kv.head_dim = 256;
    return {open, windowed, one_kv};
}

inline std::string describe(const Case& c) {
    return str_cat("H", c.heads, "/", c.kv_heads, " D", c.head_dim, " w", c.window);
}

}  // namespace xs::test::reaching
