#pragma once

// Backend-agnostic cases for sessions of several programs (Backend::compile over a list of graphs): state slots shared
// by label, KvRead, and truncate inside a short step (SessionOptions::rewind_tokens) for KV, conv and recurrent state.
// Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "backend_test_util.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::session {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(std::span<const Graph* const>, const WeightStore&, int64_t rewind_tokens)>
        compile;
    double atol = 1e-5;
    double rtol = 1e-5;
};

// One layer of each kind of state: a causal conv, a gated delta rule and attention over a KV cache, each fed from its
// own inputs.
struct HybridLayer {
    static constexpr int64_t kChannels = 24, kKernel = 4, kHk = 2, kHv = 4, kDk = 32, kDv = 16, kHkv = 2, kD = 32;
    MemoryWeights weights;
    Graph graph;

    HybridLayer() {
        TensorRng rng(41);
        weights.add("conv_w", rng.uniform(DType::BF16, {kChannels, kKernel}));
        const int conv = graph.add_conv_slot(kChannels, kKernel, "conv");
        graph.output("conv", graph.causal_conv(conv, graph.input("x", DType::F32, {kT, kChannels}),
                                               graph.weight(weights, "conv_w")));
        const int gdn = graph.add_recurrent_slot(kHv, kDk, kDv, "gdn");
        graph.output("gdn", graph.gated_delta(gdn, graph.input("q", DType::F32, {kT, kHk, kDk}),
                                              graph.input("k", DType::F32, {kT, kHk, kDk}),
                                              graph.input("v", DType::F32, {kT, kHv, kDv}),
                                              graph.input("g", DType::F32, {kT, kHv}),
                                              graph.input("beta", DType::F32, {kT, kHv}), 1e-6f));
        const int kv = graph.add_kv_slot(kHkv, kD, 0, "attn");
        const auto [k_all, v_all] =
            graph.kv_cache(kv, graph.input("ak", DType::F32, {kT, kHkv, kD}), graph.input("av", DType::F32, {kT, kHkv, kD}));
        graph.output("attn", graph.attention(graph.input("aq", DType::F32, {kT, 2 * kHkv, kD}), k_all, v_all,
                                             graph::AttentionAttrs{0.2f}));
    }

    // `tokens` rows of every input.
    static TensorMap inputs(TensorRng& rng, int64_t tokens) {
        TensorMap in;
        in["x"] = rng.uniform(DType::F32, {tokens, kChannels});
        in["q"] = rng.uniform(DType::F32, {tokens, kHk, kDk});
        in["k"] = rng.uniform(DType::F32, {tokens, kHk, kDk});
        in["v"] = rng.uniform(DType::F32, {tokens, kHv, kDv});
        HostTensor g = rng.uniform(DType::F32, {tokens, kHv}, 0.5f), beta = rng.uniform(DType::F32, {tokens, kHv}, 0.5f);
        for (int64_t i = 0; i < g.numel(); ++i) {
            g.data<float>()[i] -= 0.6f;     // log decay in [-1.1, -0.1]
            beta.data<float>()[i] += 0.5f;  // beta in [0, 1]
        }
        in["g"] = std::move(g);
        in["beta"] = std::move(beta);
        in["ak"] = rng.uniform(DType::F32, {tokens, kHkv, kD});
        in["av"] = rng.uniform(DType::F32, {tokens, kHkv, kD});
        in["aq"] = rng.uniform(DType::F32, {tokens, 2 * kHkv, kD});
        return in;
    }

    // The first `n` rows of every input.
    static TensorMap head(const TensorMap& in, int64_t n) {
        TensorMap out;
        for (const auto& [name, t] : in) {
            const Tensor64 all = to_f64(t);
            Dims shape = t.shape();
            shape[0] = n;
            const auto row = static_cast<size_t>(all.numel() / t.dim(0));
            out[name] = HostTensor::from_vector(
                DType::F32, shape, std::vector<float>(all.data.begin(), all.data.begin() + static_cast<long>(row * n)));
        }
        return out;
    }
};

inline bool outputs_close(const TensorMap& a, const TensorMap& b, const Env& env) {
    bool ok = true;
    for (const char* name : {"conv", "gdn", "attn"}) {
        CAPTURE(name);
        const bool close = all_close(a.at(name), b.at(name), env.atol, env.rtol);
        CHECK(close);
        ok = ok && close;
    }
    return ok;
}

// After a step of up to rewind_tokens tokens, truncating to any count inside it and continuing gives what a session
// that never ran the dropped tokens gives; counts it cannot rebuild are refused and change nothing.
inline void truncate_inside_step(const Env& env) {
    const HybridLayer layer;
    const Graph* programs[] = {&layer.graph};
    TensorRng rng(42);
    const TensorMap prefix = HybridLayer::inputs(rng, 5), step = HybridLayer::inputs(rng, 3),
                    next = HybridLayer::inputs(rng, 2);
    for (const int64_t keep : {0, 1, 2, 3}) {
        CAPTURE(keep);
        const auto spec = env.compile(programs, layer.weights, 4);
        const auto plain = env.compile(programs, layer.weights, 0);
        TensorMap out, want;
        spec->run(prefix, out);
        plain->run(prefix, want);
        CHECK_THROWS_AS(spec->truncate(4), xs::Error);  // the prefix (5 tokens) is longer than rewind_tokens
        spec->run(step, out);
        spec->truncate(5 + keep);
        CHECK(spec->num_cached() == 5 + keep);
        if (keep > 0) plain->run(HybridLayer::head(step, keep), want);
        spec->run(next, out);
        plain->run(next, want);
        outputs_close(out, want, env);
        // The last step started at 5 + keep: below that is refused, and the refusal leaves every slot as it was.
        CHECK_THROWS_AS(spec->truncate(4 + keep), xs::Error);
        CHECK(spec->num_cached() == 7 + keep);
        const TensorMap after = HybridLayer::inputs(rng, 1);
        spec->run(after, out);
        plain->run(after, want);
        outputs_close(out, want, env);
    }
    // Without rewind_tokens only whole sequences: no count inside a short step either.
    const auto plain = env.compile(programs, layer.weights, 0);
    TensorMap out;
    plain->run(prefix, out);
    plain->run(step, out);
    CHECK_THROWS_AS(plain->truncate(6), xs::Error);
    CHECK(plain->num_cached() == 8);
    plain->truncate(0);
    CHECK(plain->num_cached() == 0);
}

// Programs that name the same KV slot share it: two writers extend one sequence, a KvRead reader attends to all of it
// without appending, and a slot declared with two shapes is refused.
inline void shared_kv_slot(const Env& env) {
    constexpr int64_t kHkv = 2, kD = 32, kHq = 4;
    const auto writer = [&](int64_t head_dim) {
        auto g = std::make_unique<Graph>();
        const int slot = g->add_kv_slot(kHkv, head_dim, 0, "layer");
        const auto [k_all, v_all] = g->kv_cache(slot, g->input("k", DType::F32, {kT, kHkv, head_dim}),
                                                g->input("v", DType::F32, {kT, kHkv, head_dim}));
        g->output("out", g->attention(g->input("q", DType::F32, {kT, kHq, head_dim}), k_all, v_all,
                                      graph::AttentionAttrs{0.25f}));
        return g;
    };
    const auto reader = [&](int64_t head_dim) {
        auto g = std::make_unique<Graph>();
        const int slot = g->add_kv_slot(kHkv, head_dim, 0, "layer");
        const auto [k_all, v_all] = g->kv_read(slot);
        g->output("out", g->attention(g->input("q", DType::F32, {kT, kHq, head_dim}), k_all, v_all,
                                      graph::AttentionAttrs{0.25f}));
        return g;
    };
    const auto first = writer(kD), second = writer(kD), read = reader(kD);
    const Graph* programs[] = {first.get(), second.get(), read.get()};
    const auto session = env.compile(programs, MemoryWeights{}, 0);
    TensorRng rng(43);
    std::vector<HostTensor> ks, vs;
    const auto write = [&](size_t program, int64_t tokens) {
        ks.push_back(rng.uniform(DType::F32, {tokens, kHkv, kD}));
        vs.push_back(rng.uniform(DType::F32, {tokens, kHkv, kD}));
        TensorMap out;
        session->run(program, {{"k", ks.back()}, {"v", vs.back()}, {"q", rng.uniform(DType::F32, {tokens, kHq, kD})}},
                     out);
    };
    const auto rows = [](const std::vector<HostTensor>& parts, int64_t n) {
        Tensor64 all;
        for (const HostTensor& p : parts) {
            const Tensor64 t = to_f64(p);
            if (all.shape.empty()) {
                all.shape = t.shape;
                all.shape[0] = 0;
            }
            all.shape[0] += t.shape[0];
            all.data.insert(all.data.end(), t.data.begin(), t.data.end());
        }
        all.data.resize(static_cast<size_t>(all.numel() / all.shape[0] * n));
        all.shape[0] = n;
        return all;
    };
    const auto check_reader = [&](int64_t len) {
        CAPTURE(len);
        const HostTensor q = rng.uniform(DType::F32, {1, kHq, kD});
        TensorMap out;
        session->run(2, {{"q", q}}, out);
        ref::AttentionSpec spec;
        spec.scale = 0.25;
        CHECK(all_close(out.at("out"), ref::attention(to_f64(q), rows(ks, len), rows(vs, len), spec), env.atol,
                        env.rtol));
        CHECK(session->num_cached(2) == 0);  // a reader writes nothing
    };
    write(0, 3);
    write(1, 2);
    CHECK(session->num_cached(0) == 5);
    CHECK(session->num_cached(1) == 5);
    check_reader(5);
    session->truncate(1, 4);
    CHECK(session->num_cached(0) == 4);
    check_reader(4);
    session->reset();
    ks.clear();
    vs.clear();
    CHECK(session->num_cached(0) == 0);
    write(1, 2);
    check_reader(2);

    const auto wide = reader(2 * kD);
    const Graph* mismatched[] = {first.get(), wide.get()};
    CHECK_THROWS_AS(env.compile(mismatched, MemoryWeights{}, 0), xs::Error);
}

}  // namespace xs::test::session
