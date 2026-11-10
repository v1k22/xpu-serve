#pragma once

// Backend-agnostic cases for SignMatMul and the shortlist head (models/common/blocks.h), against fp64 references.
// Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "backend_test_util.h"
#include "models/common/blocks.h"
#include "quant/quant.h"
#include "reference_ops.h"

namespace xs::test::shortlist {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;  // of the largest magnitude a score can take: the row scale times sum |x|
    double rtol = 1e-4;  // the exact product of the shortlisted rows
};

inline TensorMap run(const Env& env, const Graph& g, const WeightStore& w, const TensorMap& inputs) {
    TensorMap out;
    env.compile(g, w)->run(inputs, out);
    return out;
}

// Scores of every row for a few shapes: words loaded a cache line at a time (K a multiple of 512) and one by one, row
// counts that fill no sub-group, one and two rows of x, an x with a few large entries (what its rounding must keep).
inline void sign_matmul_shapes(const Env& env) {
    TensorRng rng(71);
    struct Shape {
        int64_t n, k, m;
    };
    for (const Shape s : {Shape{40, 64, 1}, Shape{1000, 512, 1}, Shape{130, 1536, 2}, Shape{33, 96, 3}}) {
        CAPTURE(s.n);
        CAPTURE(s.k);
        const HostTensor w = rng.uniform(DType::F32, {s.n, s.k});
        const quant::SignSketch sketch = quant::sign_sketch(w.view());
        HostTensor x = rng.uniform(DType::F32, {s.m, s.k});
        for (int64_t c = 0; c < s.k; c += 37) x.data<float>()[c] *= 9.0f;
        Graph g;
        g.output("y", g.sign_matmul(g.input("x", DType::F32, {kT, s.k}), g.constant(sketch.signs),
                                    g.constant(sketch.scales)));
        const TensorMap out = run(env, g, MemoryWeights{}, {{"x", x}});
        const Tensor64 expected = ref::sign_matmul(to_f64(x), to_ints(sketch.signs), to_f64(sketch.scales));
        const std::vector<float> got = out.at("y").to_f32();
        REQUIRE(out.at("y").shape() == Dims{s.m, s.n});
        double worst = 0;  // relative to scale * sum |x|, the largest a score can be
        for (int64_t r = 0; r < s.m; ++r) {
            double mass = 0;
            for (int64_t c = 0; c < s.k; ++c) mass += std::abs(x.data<float>()[r * s.k + c]);
            for (int64_t i = 0; i < s.n; ++i) {
                const double bound = mass * sketch.scales.data<float>()[i];
                worst = std::max(worst, std::abs(got[static_cast<size_t>(r * s.n + i)] - expected[r * s.n + i]) / bound);
            }
        }
        CAPTURE(worst);
        CHECK(worst <= env.atol);
    }
}

// The shortlist of a q4 head: the candidates are the best rows by the sketch's score, their logits the exact products,
// and an x near a row of the head picks that row (as the whole head would).
inline void shortlist_head_picks(const Env& env) {
    TensorRng rng(72);
    const int64_t vocab = 3000, hidden = 512, count = 48;
    const HostTensor dense = rng.uniform(DType::F32, {vocab, hidden}, 1.0f / std::sqrt(static_cast<float>(hidden)));
    MemoryWeights w;
    w.add_quantized("head", quant::quantize(dense.view(), DType::Q4, 32));
    const quant::SignSketch sketch = quant::sign_sketch(w.get("head"));
    Graph g;
    const Value x = g.input("x", DType::F32, {1, hidden});
    const blocks::Shortlist s = blocks::shortlist_head(g, x, g.weight(w, "head"), sketch, count);
    g.output("logits", s.logits);
    g.output("candidates", s.candidates);
    const auto session = env.compile(g, w);
    const Tensor64 head = to_f64(w.tensor("head"));
    for (const int64_t target : {7, 1234, 2999}) {
        CAPTURE(target);
        HostTensor xv = rng.uniform(DType::F32, {1, hidden}, 0.02f);
        for (int64_t c = 0; c < hidden; ++c) xv.data<float>()[c] += static_cast<float>(head[target * hidden + c]);
        TensorMap out;
        session->run({{"x", xv}}, out);
        const std::vector<int64_t> candidates = to_ints(out.at("candidates"));
        const std::vector<float> logits = out.at("logits").to_f32();
        REQUIRE(candidates.size() == static_cast<size_t>(count));
        REQUIRE(logits.size() == static_cast<size_t>(count));
        // every candidate scores at least as high as every row left out (within the backend's rounding of a score)
        const Tensor64 scores = ref::sign_matmul(to_f64(xv), to_ints(sketch.signs), to_f64(sketch.scales));
        std::vector<bool> in(static_cast<size_t>(vocab), false);
        double lowest_in = 1e300, mass = 0;
        for (int64_t c = 0; c < hidden; ++c) mass += std::abs(xv.data<float>()[c]);
        for (const int64_t c : candidates) {
            REQUIRE((c >= 0 && c < vocab && !in[static_cast<size_t>(c)]));
            in[static_cast<size_t>(c)] = true;
            lowest_in = std::min(lowest_in, scores[c]);
        }
        double highest_out = -1e300, scale = 0;
        for (int64_t i = 0; i < vocab; ++i) {
            if (!in[static_cast<size_t>(i)]) highest_out = std::max(highest_out, scores[i]);
            scale = std::max(scale, static_cast<double>(sketch.scales.data<float>()[i]));
        }
        CHECK(lowest_in >= highest_out - 2 * env.atol * mass * scale);
        // the candidates' logits are the head's
        const Tensor64 full = ref::matmul(to_f64(xv), head);
        double peak = 0;
        for (int64_t i = 0; i < vocab; ++i) peak = std::max(peak, std::abs(full[i]));
        int64_t best = 0;
        for (int64_t j = 0; j < count; ++j) {
            const double want = full[candidates[static_cast<size_t>(j)]];
            CHECK(std::abs(logits[static_cast<size_t>(j)] - want) <= 1e-5 + env.rtol * peak);
            if (logits[static_cast<size_t>(j)] > logits[static_cast<size_t>(best)]) best = j;
        }
        CHECK(candidates[static_cast<size_t>(best)] == target);
    }
}

}  // namespace xs::test::shortlist
