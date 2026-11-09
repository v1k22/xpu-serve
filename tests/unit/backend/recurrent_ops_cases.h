#pragma once

// Backend-agnostic cases for the ops of hybrid linear-attention models (Qwen3.5): sigmoid, silu, softplus, partial
// rotary, and the stateful CausalConv and GatedDelta checked over several steps against one fp64 pass over the whole
// sequence. Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "backend_test_util.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::recurrent {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;
    double rtol = 1e-5;
};

// Concatenates step tensors along axis 0.
inline Tensor64 concat_rows(const std::vector<HostTensor>& parts) {
    Tensor64 out;
    for (const HostTensor& p : parts) {
        const Tensor64 t = to_f64(p);
        if (out.shape.empty()) {
            out = t;
            continue;
        }
        out.shape[0] += t.shape[0];
        out.data.insert(out.data.end(), t.data.begin(), t.data.end());
    }
    return out;
}

inline void unary_ops(const Env& env) {
    TensorRng rng(21);
    const HostTensor x = rng.uniform(DType::F32, {3, 40}, 12.0f);
    struct Case {
        const char* name;
        Value (Graph::*build)(Value);
        double (*ref)(double);
    };
    const Case cases[] = {
        {"sigmoid", &Graph::sigmoid, [](double v) { return 1.0 / (1.0 + std::exp(-v)); }},
        {"silu", &Graph::silu, [](double v) { return v / (1.0 + std::exp(-v)); }},
        {"softplus", &Graph::softplus, [](double v) { return std::log1p(std::exp(v)); }},
    };
    for (const Case& c : cases) {
        CAPTURE(c.name);
        Graph g;
        g.output("out", (g.*c.build)(g.input("x", DType::F32, {kT, 40})));
        TensorMap out;
        env.compile(g, MemoryWeights{})->run({{"x", x}}, out);
        CHECK(all_close(out.at("out"), ref::map(to_f64(x), c.ref), env.atol, env.rtol));
    }
}

inline void partial_rope(const Env& env) {
    TensorRng rng(22);
    const int64_t heads = 2, dim = 256, rotary = 64;
    const HostTensor x = rng.uniform(DType::F32, {5, heads, dim});
    std::vector<float> inv_freq(rotary / 2);
    for (size_t i = 0; i < inv_freq.size(); ++i)
        inv_freq[i] = std::pow(1e7f, -2.0f * static_cast<float>(i) / rotary);
    const HostTensor pos = arange_i32(7, 12);
    Graph g;
    const Value freq = g.constant(make_tensor(DType::F32, {rotary / 2}, inv_freq));
    g.output("out",
             g.rope(g.input("x", DType::F32, {kT, heads, dim}), g.input("pos", DType::I32, {kT}), freq, rotary));
    TensorMap out;
    env.compile(g, MemoryWeights{})->run({{"x", x}, {"pos", pos}}, out);
    const std::vector<double> f64(inv_freq.begin(), inv_freq.end());
    CHECK(all_close(out.at("out"), ref::partial_rope(to_f64(x), to_ints(pos), f64, rotary), env.atol, env.rtol));
    // Planted bug: rotate-half over the whole head with the tail frequencies zero pairs c with c + D/2, not c + R/2.
    std::vector<double> padded(static_cast<size_t>(dim / 2), 0.0);
    std::copy(f64.begin(), f64.end(), padded.begin());
    CHECK_FALSE(all_close(out.at("out"), ref::rope(to_f64(x), to_ints(pos), padded), env.atol, env.rtol));
}

inline void causal_conv_steps(const Env& env) {
    TensorRng rng(23);
    const int64_t channels = 48, kernel = 4;
    MemoryWeights w;
    w.add("w", rng.uniform(DType::BF16, {channels, kernel}));
    Graph g;
    const int slot = g.add_conv_slot(channels, kernel, "conv");
    g.output("out", g.causal_conv(slot, g.input("x", DType::F32, {kT, channels}), g.weight(w, "w")));
    const auto session = env.compile(g, w);
    std::vector<HostTensor> xs, outs;
    for (const int64_t t : {5, 1, 1, 3, 2}) {  // steps shorter than the kernel must reach back across runs
        xs.push_back(rng.uniform(DType::F32, {t, channels}));
        TensorMap out;
        session->run({{"x", xs.back()}}, out);
        outs.push_back(out.at("out").clone());
    }
    const Tensor64 expected = ref::causal_conv(concat_rows(xs), to_f64(w.tensor("w")));
    const Tensor64 got = concat_rows(outs);
    double worst = 0;
    for (int64_t i = 0; i < got.numel(); ++i) worst = std::max(worst, std::abs(got[i] - expected[i]) - env.rtol * std::abs(expected[i]));
    CHECK(worst <= env.atol);

    // reset starts from zeros again; a cut back into the sequence is refused.
    session->reset();
    TensorMap again;
    session->run({{"x", xs.front()}}, again);
    CHECK(all_close(again.at("out"), outs.front(), env.atol, env.rtol));
    CHECK_THROWS_AS(session->truncate(2), xs::Error);
    session->truncate(session->num_cached());  // keeping everything is fine
}

inline void gated_delta_steps(const Env& env) {
    TensorRng rng(24);
    struct Dims {
        int64_t dk, dv;
        std::vector<int64_t> steps;
    };
    // Small heads, and Qwen3.5's 128 x 128 heads over a step longer than a backend's staging block; the short steps
    // after a long one carry the state between a backend's prefill and decode kernels.
    for (const Dims& dims : {Dims{32, 16, {7, 1, 4}}, Dims{64, 32, {12, 1, 3}}, Dims{128, 128, {70, 1, 4}}})
    for (const float eps : {0.0f, 1e-6f}) {
        const int64_t hk = 2, hv = 4, dk = dims.dk, dv = dims.dv;
        CAPTURE(dk);
        CAPTURE(eps);
        Graph g;
        const int slot = g.add_recurrent_slot(hv, dk, dv, "gdn");
        g.output("out", g.gated_delta(slot, g.input("q", DType::F32, {kT, hk, dk}), g.input("k", DType::F32, {kT, hk, dk}),
                                      g.input("v", DType::F32, {kT, hv, dv}), g.input("g", DType::F32, {kT, hv}),
                                      g.input("beta", DType::F32, {kT, hv}), eps));
        const auto session = env.compile(g, MemoryWeights{});
        std::vector<HostTensor> qs, ks, vs, gs, betas, outs;
        // Without l2norm, k must stay small so the recurrence stays bounded (as l2norm guarantees in the model).
        const float k_range = eps > 0 ? 1.0f : 0.2f * std::sqrt(32.0f / static_cast<float>(dk));
        for (const int64_t t : dims.steps) {
            qs.push_back(rng.uniform(DType::F32, {t, hk, dk}));
            ks.push_back(rng.uniform(DType::F32, {t, hk, dk}, k_range));
            vs.push_back(rng.uniform(DType::F32, {t, hv, dv}));
            HostTensor gt = rng.uniform(DType::F32, {t, hv}, 0.5f);
            HostTensor bt = rng.uniform(DType::F32, {t, hv}, 0.5f);
            for (int64_t i = 0; i < gt.numel(); ++i) {
                gt.data<float>()[i] = -0.6f + gt.data<float>()[i];  // log decay in [-1.1, -0.1]
                bt.data<float>()[i] = 0.5f + bt.data<float>()[i];   // beta in [0, 1]
            }
            gs.push_back(gt);
            betas.push_back(bt);
            TensorMap out;
            session->run({{"q", qs.back()}, {"k", ks.back()}, {"v", vs.back()}, {"g", gs.back()}, {"beta", betas.back()}},
                         out);
            outs.push_back(out.at("out").clone());
        }
        const Tensor64 q = concat_rows(qs), k = concat_rows(ks), v = concat_rows(vs), gg = concat_rows(gs),
                       b = concat_rows(betas);
        const Tensor64 got = concat_rows(outs);
        HostTensor got_t = HostTensor::from_vector(DType::F32, got.shape, std::vector<float>(got.data.begin(), got.data.end()));
        CHECK(all_close(got_t, ref::gated_delta(q, k, v, gg, b, eps), env.atol, env.rtol));
        // Planted bug: interleaved head mapping (h % Hk) instead of grouped (h / (Hv / Hk)) must not match.
        const auto interleaved = [](int64_t h, int64_t hk_, int64_t) { return h % hk_; };
        CHECK_FALSE(all_close(got_t, ref::gated_delta(q, k, v, gg, b, eps, interleaved), env.atol, env.rtol));
        CHECK_THROWS_AS(session->truncate(3), xs::Error);
    }
}

}  // namespace xs::test::recurrent
