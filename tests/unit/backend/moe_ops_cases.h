#pragma once

// Backend-agnostic cases for the mixture-of-experts ops (Div, Softmax, TopK, ReduceSum, ExpertMatMul) and the shared
// MoE block (models/common/blocks.h), against fp64 references. Each backend's test file runs them with its own
// tolerance.

#include <doctest/doctest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "backend_test_util.h"
#include "models/common/blocks.h"
#include "quant/quant.h"
#include "reference_ops.h"
#include "util/error.h"

namespace xs::test::moe {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;  // elementwise ops
    double rtol = 1e-5;
    double mm_atol = 1e-5;  // ops with an inner product over weights (expert matmuls, the block)
    double mm_rtol = 1e-4;
};

inline TensorMap run(const Env& env, const Graph& g, const WeightStore& w, const TensorMap& inputs) {
    TensorMap out;
    env.compile(g, w)->run(inputs, out);
    return out;
}

inline HostTensor ids_i32(Dims shape, const std::vector<int32_t>& v) {
    return HostTensor::from_vector(DType::I32, std::move(shape), v);
}

inline void div_softmax_reduce(const Env& env) {
    TensorRng rng(31);
    const HostTensor a = rng.uniform(DType::F32, {6, 4}, 3.0f);
    HostTensor b = rng.uniform(DType::F32, {6, 1}, 1.0f);
    for (float& v : b.span<float>())
        v = 1.5f + v;  // keep away from zero
    HostTensor big = rng.uniform(DType::F32, {6, 9}, 40.0f);  // softmax must subtract the row max
    const HostTensor r3 = rng.uniform(DType::F32, {6, 3, 5});
    Graph g;
    g.output("div", g.div(g.input("a", DType::F32, {kT, 4}), g.input("b", DType::F32, {kT, 1})));
    g.output("softmax", g.softmax(g.input("big", DType::F32, {kT, 9})));
    const Value r = g.input("r3", DType::F32, {kT, 3, 5});
    g.output("sum1", g.reduce_sum(r, 1));
    g.output("sum2", g.reduce_sum(r, 2));
    const TensorMap out = run(env, g, MemoryWeights{}, {{"a", a}, {"b", b}, {"big", big}, {"r3", r3}});
    CHECK(all_close(out.at("div"), ref::broadcast(to_f64(a), to_f64(b), [](double x, double y) { return x / y; }),
                    env.atol, env.rtol));
    CHECK(all_close(out.at("softmax"), ref::softmax(to_f64(big)), env.atol, env.rtol));
    CHECK(all_close(out.at("sum1"), ref::reduce_sum(to_f64(r3), 1), env.atol, env.rtol));
    CHECK(all_close(out.at("sum2"), ref::reduce_sum(to_f64(r3), 2), env.atol, env.rtol));
}

inline void topk_rows(const Env& env) {
    TensorRng rng(32);
    HostTensor x = rng.uniform(DType::F32, {5, 16});
    x.data<float>()[3] = x.data<float>()[11] = 2.0f;  // a tie in row 0: index 3 before 11
    Graph g;
    const auto [values, indices] = g.topk(g.input("x", DType::F32, {kT, 16}), 4);
    g.output("values", values);
    g.output("indices", indices);
    const TensorMap out = run(env, g, MemoryWeights{}, {{"x", x}});
    const auto [ref_values, ref_indices] = ref::topk(to_f64(x), 4);
    CHECK(all_close(out.at("values"), ref_values, 0, 0));  // selection copies values exactly
    CHECK(to_ints(out.at("indices")) == ref_indices);
    CHECK(to_ints(out.at("indices"))[0] == 3);
    CHECK(to_ints(out.at("indices"))[1] == 11);
}

inline void expert_matmul_shared_and_per_slot(const Env& env) {
    TensorRng rng(33);
    const int64_t experts = 6, n = 40, k = 64, slots = 3;
    MemoryWeights w;
    w.add("w", rng.uniform(DType::BF16, {experts, n, k}, 1.0f / std::sqrt(static_cast<float>(k))));
    for (const int64_t tokens : {1, 5}) {
        CAPTURE(tokens);
        std::vector<int32_t> idx;
        for (int64_t i = 0; i < tokens * slots; ++i)
            idx.push_back(static_cast<int32_t>((i * 5 + 2) % experts));  // repeats experts across tokens
        const HostTensor ids = ids_i32({tokens, slots}, idx);
        const HostTensor x2 = rng.uniform(DType::F32, {tokens, k});
        const HostTensor x3 = rng.uniform(DType::F32, {tokens, slots, k});
        Graph g;
        const Value i = g.input("idx", DType::I32, {kT, slots});
        g.output("shared", g.expert_matmul(g.input("x2", DType::F32, {kT, k}), g.weight(w, "w"), i));
        g.output("per_slot", g.expert_matmul(g.input("x3", DType::F32, {kT, slots, k}), g.weight(w, "w"), i));
        const TensorMap out = run(env, g, w, {{"idx", ids}, {"x2", x2}, {"x3", x3}});
        const std::vector<int64_t> idx64(idx.begin(), idx.end());
        CHECK(all_close(out.at("shared"), ref::expert_matmul(to_f64(x2), to_f64(w.tensor("w")), idx64, slots),
                        env.mm_atol, env.mm_rtol));
        CHECK(all_close(out.at("per_slot"), ref::expert_matmul(to_f64(x3), to_f64(w.tensor("w")), idx64, slots),
                        env.mm_atol, env.mm_rtol));
    }
}

// Q8 / Q4 expert banks against the dequantized bank: few (token, slot) pairs (one GEMV each on SYCL), more (grouped by
// expert) and many (grouped GEMMs), shared and per-slot inputs, experts repeated across tokens, N not a multiple of
// the row blocks.
struct BankShape {
    int64_t experts, n, k, slots;
};

inline void expert_matmul_quantized(const Env& env) {
    TensorRng rng(36);
    for (const BankShape s : {BankShape{6, 40, 128, 3}, BankShape{8, 128, 64, 2}}) {
    const int64_t experts = s.experts, n = s.n, k = s.k, slots = s.slots;
    CAPTURE(experts);
    CAPTURE(k);
    const HostTensor dense = rng.uniform(DType::F32, {experts, n, k}, 1.0f / std::sqrt(static_cast<float>(k)));
    for (const DType format : {DType::Q8, DType::Q4}) {
        for (const int64_t group : {32, 64}) {
            if (k % group != 0) continue;
            CAPTURE(dtype_name(format));
            CAPTURE(group);
            MemoryWeights w;
            w.add_quantized("w", quant::quantize(dense.view(), format, group));
            for (const int64_t tokens : {1, 5, 13, 40, 150}) {  // up to 450 pairs
                CAPTURE(tokens);
                std::vector<int32_t> idx;
                for (int64_t i = 0; i < tokens * slots; ++i)
                    idx.push_back(static_cast<int32_t>((i * 5 + 2 + i / 7) % experts));
                const HostTensor ids = ids_i32({tokens, slots}, idx);
                const HostTensor x2 = rng.uniform(DType::F32, {tokens, k});
                const HostTensor x3 = rng.uniform(DType::F32, {tokens, slots, k});
                Graph g;
                const Value i = g.input("idx", DType::I32, {kT, slots});
                g.output("shared", g.expert_matmul(g.input("x2", DType::F32, {kT, k}), g.weight(w, "w"), i));
                g.output("per_slot", g.expert_matmul(g.input("x3", DType::F32, {kT, slots, k}), g.weight(w, "w"), i));
                const TensorMap out = run(env, g, w, {{"idx", ids}, {"x2", x2}, {"x3", x3}});
                const std::vector<int64_t> idx64(idx.begin(), idx.end());
                CHECK(all_close(out.at("shared"), ref::expert_matmul(to_f64(x2), to_f64(w.tensor("w")), idx64, slots),
                                env.mm_atol, env.mm_rtol));
                CHECK(all_close(out.at("per_slot"),
                                ref::expert_matmul(to_f64(x3), to_f64(w.tensor("w")), idx64, slots), env.mm_atol,
                                env.mm_rtol));
            }
        }
    }
    }
}

// The MoE block against an fp64 composition of the same math. Router weights are scaled so the chosen experts are
// clearly ahead of the rest (a near tie could pick differently at f16).
// `quantized`: q4 banks with gate and up fused into one [E, 2 I, hidden] bank (Qwen3.5 MoE's layout); `tokens` past
// 16 takes the grouped path for quantized banks on SYCL.
inline void moe_block(const Env& env, blocks::Router router, bool shared, bool quantized = false, int64_t tokens = 7) {
    TensorRng rng(router == blocks::Router::Softmax ? 34 : 35);
    const int64_t experts = 8, top_k = 2, hidden = 64, inter = quantized ? 64 : 48;
    MemoryWeights w;
    const float s_in = 1.0f / std::sqrt(static_cast<float>(hidden));
    w.add("router", rng.uniform(DType::F32, {experts, hidden}, (quantized ? 24.0f : 6.0f) * s_in));
    if (quantized) {
        const HostTensor gate_up = rng.uniform(DType::F32, {experts, 2 * inter, hidden}, s_in);
        const HostTensor down = rng.uniform(DType::F32, {experts, hidden, inter}, 1.0f / std::sqrt(static_cast<float>(inter)));
        w.add_quantized("gate_up", quant::quantize(gate_up.view(), DType::Q4, 32));
        w.add_quantized("down", quant::quantize(down.view(), DType::Q4, 32));
        // gate and up as separate dense banks, for the reference below.
        HostTensor gate(DType::F32, {experts, inter, hidden}), up(DType::F32, {experts, inter, hidden});
        const float* both = w.tensor("gate_up").data<float>();
        for (int64_t e = 0; e < experts; ++e) {
            std::copy(both + (e * 2 * inter) * hidden, both + (e * 2 * inter + inter) * hidden,
                      gate.data<float>() + e * inter * hidden);
            std::copy(both + (e * 2 * inter + inter) * hidden, both + (e * 2 * inter + 2 * inter) * hidden,
                      up.data<float>() + e * inter * hidden);
        }
        w.add("gate", std::move(gate));
        w.add("up", std::move(up));
    } else {
        w.add("gate", rng.uniform(DType::BF16, {experts, inter, hidden}, s_in));
        w.add("up", rng.uniform(DType::BF16, {experts, inter, hidden}, s_in));
        w.add("down", rng.uniform(DType::BF16, {experts, hidden, inter}, 1.0f / std::sqrt(static_cast<float>(inter))));
    }
    w.add("s_gate", rng.uniform(DType::BF16, {inter, hidden}, s_in));
    w.add("s_up", rng.uniform(DType::BF16, {inter, hidden}, s_in));
    w.add("s_down", rng.uniform(DType::BF16, {hidden, inter}, 1.0f / std::sqrt(static_cast<float>(inter))));
    w.add("s_gg", rng.uniform(DType::F32, {1, hidden}, s_in));
    // Tokens whose k-th and (k+1)-th router logits are within 0.05 are drawn again: both routers rank by logit, and a
    // near tie could pick differently at f16 without a bug.
    HostTensor x = rng.uniform(DType::F32, {tokens, hidden});
    const Tensor64 router_w = to_f64(w.tensor("router"));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int attempt = 0;; ++attempt) {
            REQUIRE(attempt < 100);
            std::vector<double> row(static_cast<size_t>(experts), 0.0);
            for (int64_t e = 0; e < experts; ++e)
                for (int64_t c = 0; c < hidden; ++c)
                    row[static_cast<size_t>(e)] += router_w[e * hidden + c] * x.data<float>()[t * hidden + c];
            std::sort(row.rbegin(), row.rend());
            if (row[static_cast<size_t>(top_k - 1)] - row[static_cast<size_t>(top_k)] > 0.05) break;
            const HostTensor redraw = rng.uniform(DType::F32, {1, hidden});
            std::copy(redraw.data<float>(), redraw.data<float>() + hidden, x.data<float>() + t * hidden);
        }
    }

    Graph g;
    blocks::MoeSpec spec;
    spec.experts = experts;
    spec.top_k = top_k;
    spec.router = router;
    spec.routed_scale = router == blocks::Router::Sigmoid ? 2.5f : 1.0f;
    blocks::MoeParts parts;
    parts.router = g.weight(w, "router");
    if (quantized) {
        parts.gate_up = g.weight(w, "gate_up");
    } else {
        parts.gate = g.weight(w, "gate");
        parts.up = g.weight(w, "up");
    }
    parts.down = g.weight(w, "down");
    if (shared) {
        parts.shared_gate = blocks::matmul_by(g, g.weight(w, "s_gate"));
        parts.shared_up = blocks::matmul_by(g, g.weight(w, "s_up"));
        parts.shared_down = blocks::matmul_by(g, g.weight(w, "s_down"));
        parts.shared_expert_gate = g.weight(w, "s_gg");
    }
    g.output("out", blocks::moe(g, g.input("x", DType::F32, {kT, hidden}), spec, parts));
    const HostTensor out = run(env, g, w, {{"x", x}}).at("out");

    // fp64: router, top-k, renormalize, scale, experts, weighted sum, shared expert.
    const Tensor64 x64 = to_f64(x);
    const Tensor64 logits = ref::matmul(x64, router_w);
    const Tensor64 probs = router == blocks::Router::Softmax
                               ? ref::softmax(logits)
                               : ref::map(logits, [](double v) { return 1.0 / (1.0 + std::exp(-v)); });
    const auto [weights, idx] = ref::topk(probs, top_k);
    const auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    const Tensor64 gate = ref::map(ref::expert_matmul(x64, to_f64(w.tensor("gate")), idx, top_k), silu);
    const Tensor64 up = ref::expert_matmul(x64, to_f64(w.tensor("up")), idx, top_k);
    const Tensor64 h = ref::broadcast(gate, up, [](double a, double b) { return a * b; });
    const Tensor64 y = ref::expert_matmul(h, to_f64(w.tensor("down")), idx, top_k);
    Tensor64 expected({tokens, hidden});
    for (int64_t t = 0; t < tokens; ++t) {
        double norm = 0;
        for (int64_t j = 0; j < top_k; ++j)
            norm += weights[t * top_k + j];
        for (int64_t j = 0; j < top_k; ++j) {
            const double wj = weights[t * top_k + j] / norm * spec.routed_scale;
            for (int64_t c = 0; c < hidden; ++c)
                expected[t * hidden + c] += wj * y[(t * top_k + j) * hidden + c];
        }
    }
    if (shared) {
        const Tensor64 sg = ref::map(ref::matmul(x64, to_f64(w.tensor("s_gate"))), silu);
        const Tensor64 su = ref::matmul(x64, to_f64(w.tensor("s_up")));
        const Tensor64 s = ref::matmul(ref::broadcast(sg, su, [](double a, double b) { return a * b; }),
                                       to_f64(w.tensor("s_down")));
        const Tensor64 gg = ref::matmul(x64, to_f64(w.tensor("s_gg")));
        for (int64_t t = 0; t < tokens; ++t) {
            const double gate_t = 1.0 / (1.0 + std::exp(-gg[t]));
            for (int64_t c = 0; c < hidden; ++c)
                expected[t * hidden + c] += gate_t * s[t * hidden + c];
        }
    }
    CHECK(all_close(out, expected, env.mm_atol, env.mm_rtol));
    // Planted bug: without the renormalization the block must not match.
    Tensor64 unnormalized = expected;
    for (double& v : unnormalized.data)
        v *= 1.7;
    CHECK_FALSE(all_close(out, unnormalized, env.mm_atol, env.mm_rtol));
}

}  // namespace xs::test::moe
