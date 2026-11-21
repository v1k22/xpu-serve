#pragma once

// Backend-agnostic cases for the ops of decoders without rotary positions (Inkling 2): L2Clip, DynamicConv over
// several steps against one fp64 pass over the whole sequence, and causal attention with a relative bias and a sink
// over a KV cache fed in steps. Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "backend_test_util.h"
#include "recurrent_ops_cases.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::relative {

using graph::Graph;
using graph::kT;
using graph::Value;
using recurrent::concat_rows;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;
    double rtol = 1e-5;
};

inline HostTensor to_host(const Tensor64& t) {
    return HostTensor::from_vector(DType::F32, t.shape, std::vector<float>(t.data.begin(), t.data.end()));
}

// Rows on both sides of the threshold, a row of zeros, and head dims a kernel's lanes do not divide.
inline void l2_clip(const Env& env) {
    TensorRng rng(41);
    for (const int64_t dim : {16, 128, 100}) {
        CAPTURE(dim);
        const float threshold = 0.4f * std::sqrt(static_cast<float>(dim));
        HostTensor x = rng.uniform(DType::F32, {6, 3, dim});
        for (int64_t i = 0; i < 3 * dim; ++i) x.data<float>()[i] *= 0.25f;  // token 0: shorter than the threshold
        std::fill_n(x.data<float>() + 3 * dim, dim, 0.0f);                  // token 1, head 0: zeros
        Graph g;
        g.output("out", g.l2_clip(g.input("x", DType::F32, {kT, 3, dim}), threshold));
        TensorMap out;
        env.compile(g, MemoryWeights{})->run({{"x", x}}, out);
        const Tensor64 expected = ref::l2_clip(to_f64(x), threshold, 1e-6);
        CHECK(all_close(out.at("out"), expected, env.atol, env.rtol));
        // The case holds rows of both kinds: some pass unchanged, some are scaled.
        const Tensor64 in = to_f64(x);
        int64_t kept = 0, clipped = 0;
        for (int64_t r = 0; r < 18; ++r) {
            (std::abs(expected[r * dim + 1] - in[r * dim + 1]) < 1e-12 ? kept : clipped) += 1;
        }
        CHECK(kept >= 3);
        CHECK(clipped >= 3);
        // Planted bug: every row scaled to the threshold (a norm, not a clip).
        CHECK_FALSE(all_close(out.at("out"), ref::l2_clip(ref::l2_clip(to_f64(x), 1e-3, 0.0), 1e9, 0.0), env.atol,
                              env.rtol));
    }
}

inline void dynamic_conv_steps(const Env& env) {
    TensorRng rng(42);
    struct Shape {
        int64_t channels, groups, kernel;
    };
    for (const Shape& s : {Shape{48, 4, 4}, Shape{1024, 8, 4}, Shape{40, 40, 3}}) {
        CAPTURE(s.channels);
        MemoryWeights w;
        w.add("w", rng.uniform(DType::BF16, {s.kernel, s.channels}));
        Graph g;
        const int slot = g.add_conv_slot(s.channels, s.kernel, "conv");
        g.output("out", g.dynamic_conv(slot, g.input("x", DType::F32, {kT, s.channels}), g.weight(w, "w"),
                                       g.input("taps", DType::F32, {kT, s.groups, s.kernel})));
        const auto session = env.compile(g, w);
        std::vector<HostTensor> xs, taps, outs;
        for (const int64_t t : {5, 1, 1, 3, 2}) {  // steps shorter than the kernel must reach back across runs
            xs.push_back(rng.uniform(DType::F32, {t, s.channels}));
            taps.push_back(rng.uniform(DType::F32, {t, s.groups, s.kernel}));
            TensorMap out;
            session->run({{"x", xs.back()}, {"taps", taps.back()}}, out);
            outs.push_back(out.at("out").clone());
        }
        const Tensor64 x = concat_rows(xs), tp = concat_rows(taps), weight = to_f64(w.tensor("w"));
        const HostTensor got = to_host(concat_rows(outs));
        CHECK(all_close(got, ref::dynamic_conv(x, weight, tp), env.atol, env.rtol));
        // Planted bug: tap j weighting the token K - 1 - j back (CausalConv's order).
        CHECK_FALSE(all_close(got, ref::dynamic_conv(x, weight, tp, false), env.atol, env.rtol));

        session->reset();
        TensorMap again;
        session->run({{"x", xs.front()}, {"taps", taps.front()}}, again);
        CHECK(all_close(again.at("out"), outs.front(), env.atol, env.rtol));
        CHECK_THROWS_AS(session->truncate(2), xs::Error);
    }
}

// Causal attention with a relative bias and a sink over a cache fed in steps: a decode step, steps on the per-row
// and the matrix paths, an extent shorter and longer than the cache, with and without a window.
inline void relative_attention(const Env& env) {
    TensorRng rng(43);
    struct Shape {
        int64_t heads, kv_heads, dim, extent, window;
        std::vector<int64_t> steps;
    };
    for (const Shape& s : {Shape{4, 2, 32, 6, 0, {5, 1, 1, 9, 3}}, Shape{6, 2, 128, 64, 0, {40, 1, 20, 1}},
                           Shape{3, 1, 64, 12, 12, {7, 1, 30, 1, 2}}, Shape{24, 8, 128, 16, 16, {33, 1}}}) {
        CAPTURE(s.heads);
        CAPTURE(s.dim);
        graph::AttentionAttrs attrs;
        attrs.scale = 1.0f / std::sqrt(static_cast<float>(s.dim));
        attrs.sliding_window = s.window;
        attrs.rel_extent = s.extent;
        Graph g;
        const int slot = g.add_kv_slot(s.kv_heads, s.dim, s.window, "kv");
        const auto [k_all, v_all] = g.kv_cache(slot, g.input("k", DType::F32, {kT, s.kv_heads, s.dim}),
                                               g.input("v", DType::F32, {kT, s.kv_heads, s.dim}));
        g.output("out", g.attention_relative(g.input("q", DType::F32, {kT, s.heads, s.dim}), k_all, v_all,
                                             g.input("rel", DType::F32, {kT, s.heads, s.extent}),
                                             g.input("sink", DType::F32, {kT, s.heads}), attrs));
        const auto session = env.compile(g, MemoryWeights{});
        std::vector<HostTensor> ks, vs;
        for (const int64_t t : s.steps) {
            CAPTURE(t);
            const HostTensor q = rng.uniform(DType::F32, {t, s.heads, s.dim}, 2.0f);
            const HostTensor rel = rng.uniform(DType::F32, {t, s.heads, s.extent}, 3.0f);
            const HostTensor sink = rng.uniform(DType::F32, {t, s.heads}, 3.0f);
            ks.push_back(rng.uniform(DType::F32, {t, s.kv_heads, s.dim}, 2.0f));
            vs.push_back(rng.uniform(DType::F32, {t, s.kv_heads, s.dim}));
            TensorMap out;
            session->run({{"q", q}, {"k", ks.back()}, {"v", vs.back()}, {"rel", rel}, {"sink", sink}}, out);
            ref::AttentionSpec spec;
            spec.scale = attrs.scale;
            spec.sliding_window = s.window;
            spec.rel = to_f64(rel);
            spec.sink = to_f64(sink);
            const Tensor64 k = concat_rows(ks), v = concat_rows(vs);
            CHECK(all_close(out.at("out"), ref::attention(to_f64(q), k, v, spec), env.atol, env.rtol));
            // Planted bugs: the sink left out of the softmax; the bias read toward the future; no bias.
            ref::AttentionSpec no_sink = spec;
            no_sink.sink = Tensor64{};
            CHECK_FALSE(all_close(out.at("out"), ref::attention(to_f64(q), k, v, no_sink), env.atol, env.rtol));
            ref::AttentionSpec flipped = spec;
            flipped.rel_sign = -1;
            CHECK_FALSE(all_close(out.at("out"), ref::attention(to_f64(q), k, v, flipped), env.atol, env.rtol));
            ref::AttentionSpec no_rel = spec;
            no_rel.rel = Tensor64{};
            CHECK_FALSE(all_close(out.at("out"), ref::attention(to_f64(q), k, v, no_rel), env.atol, env.rtol));
        }
    }
}

// Attention that is not causal over a window anchored at the step: every query of a step sees the step's keys and
// the window before the step, whichever its row (a block drafter's block over its context).
inline void anchored_window_attention(const Env& env) {
    TensorRng rng(44);
    struct Shape {
        int64_t heads, kv_heads, dim, window;
        std::vector<int64_t> steps;
    };
    for (const Shape& s : {Shape{4, 2, 32, 6, {5, 8, 8, 3}}, Shape{24, 8, 128, 16, {40, 8, 8}},
                           Shape{3, 1, 64, 100, {7, 8, 20}}}) {
        CAPTURE(s.heads);
        CAPTURE(s.window);
        graph::AttentionAttrs attrs;
        attrs.scale = 1.0f / std::sqrt(static_cast<float>(s.dim));
        attrs.causal = false;
        attrs.sliding_window = s.window;
        attrs.anchored = true;
        Graph g;
        const int slot = g.add_kv_slot(s.kv_heads, s.dim, 0, "kv");
        const auto [k_all, v_all] = g.kv_cache(slot, g.input("k", DType::F32, {kT, s.kv_heads, s.dim}),
                                               g.input("v", DType::F32, {kT, s.kv_heads, s.dim}));
        g.output("out", g.attention(g.input("q", DType::F32, {kT, s.heads, s.dim}), k_all, v_all, attrs));
        const auto session = env.compile(g, MemoryWeights{});
        std::vector<HostTensor> ks, vs;
        int64_t differ = 0;
        for (const int64_t t : s.steps) {
            CAPTURE(t);
            const HostTensor q = rng.uniform(DType::F32, {t, s.heads, s.dim}, 2.0f);
            ks.push_back(rng.uniform(DType::F32, {t, s.kv_heads, s.dim}, 2.0f));
            vs.push_back(rng.uniform(DType::F32, {t, s.kv_heads, s.dim}));
            TensorMap out;
            session->run({{"q", q}, {"k", ks.back()}, {"v", vs.back()}}, out);
            ref::AttentionSpec spec;
            spec.scale = attrs.scale;
            spec.causal = false;
            spec.sliding_window = s.window;
            spec.anchored = true;
            const Tensor64 k = concat_rows(ks), v = concat_rows(vs);
            CHECK(all_close(out.at("out"), ref::attention(to_f64(q), k, v, spec), env.atol, env.rtol));
            // Planted bug: a window around each row (the rule without the anchor) sees other keys once the cache is
            // longer than the window.
            ref::AttentionSpec per_row = spec;
            per_row.anchored = false;
            per_row.sliding_window = s.window + 1;
            differ += all_close(out.at("out"), ref::attention(to_f64(q), k, v, per_row), env.atol, env.rtol) ? 0 : 1;
        }
        // (where the cache outgrows the window: the last shape's never does, and both rules see every key)
        int64_t total = 0;
        for (const int64_t t : s.steps) total += t;
        if (total > s.window + 8) CHECK(differ >= 1);
    }
}

}  // namespace xs::test::relative
