#pragma once

// Backend-agnostic cases for the ops robot policies add (LayerNorm, Concat, attention with a key bias over a static
// number of queries, head dims outside the per-row kernels), against fp64 references. Each backend's test file runs
// them with its own tolerance.

#include <doctest/doctest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "backend_test_util.h"
#include "reference_ops.h"

namespace xs::test::policy_ops {

using graph::AttentionAttrs;
using graph::Graph;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const SessionOptions&)> compile;
    std::function<SessionOptions()> options;
    std::vector<std::string> attention_impls{"auto"};
    double atol = 1e-5;
    double rtol = 1e-5;
    double attn_atol = 1e-5;
    double attn_rtol = 1e-5;
};

inline TensorMap run(const Env& env, const Graph& g, const TensorMap& inputs, SessionOptions options) {
    TensorMap out;
    env.compile(g, options)->run(inputs, out);
    return out;
}

inline void layer_norm_rows(const Env& env) {
    TensorRng rng(41);
    HostTensor x = rng.uniform(DType::F32, {7, 1152}, 3.0f);
    for (float& v : x.span<float>())
        v += 2.0f;  // a mean far from zero: the kernel must subtract it
    const HostTensor w = rng.uniform(DType::F32, {1152});
    const HostTensor b = rng.uniform(DType::F32, {1152});
    Graph g;
    g.output("out", g.layer_norm(g.input("x", DType::F32, {7, 1152}), g.input("w", DType::F32, {1152}),
                                 g.input("b", DType::F32, {1152}), 1e-6f));
    const TensorMap out = run(env, g, {{"x", x}, {"w", w}, {"b", b}}, env.options());
    CHECK(all_close(out.at("out"), ref::layer_norm(to_f64(x), to_f64(w), to_f64(b), 1e-6), env.atol, env.rtol));
}

inline void concat_axes(const Env& env) {
    TensorRng rng(42);
    const HostTensor a = rng.uniform(DType::F32, {3, 4, 5});
    const HostTensor b = rng.uniform(DType::F32, {2, 4, 5});
    const HostTensor c = rng.uniform(DType::F32, {3, 6, 5});
    const HostTensor i0 = HostTensor::from_vector(DType::I32, {2}, std::vector<int32_t>{7, 8});
    const HostTensor i1 = HostTensor::from_vector(DType::I32, {3}, std::vector<int32_t>{-1, 0, 9});
    Graph g;
    const Value va = g.input("a", DType::F32, {3, 4, 5});
    g.output("rows", g.concat({va, g.input("b", DType::F32, {2, 4, 5})}, 0));
    g.output("middle", g.concat({va, g.input("c", DType::F32, {3, 6, 5}), va}, 1));
    g.output("ints", g.concat({g.input("i0", DType::I32, {2}), g.input("i1", DType::I32, {3})}, 0));
    const TensorMap out = run(env, g, {{"a", a}, {"b", b}, {"c", c}, {"i0", i0}, {"i1", i1}}, env.options());
    const Tensor64 fa = to_f64(a), fb = to_f64(b), fc = to_f64(c);
    Tensor64 rows({5, 4, 5});
    std::copy(fa.data.begin(), fa.data.end(), rows.data.begin());
    std::copy(fb.data.begin(), fb.data.end(), rows.data.begin() + 60);
    CHECK(all_close(out.at("rows"), rows, 0, 0));
    Tensor64 middle({3, 14, 5});
    for (int64_t o = 0; o < 3; ++o) {
        std::copy(fa.data.begin() + o * 20, fa.data.begin() + (o + 1) * 20, middle.data.begin() + o * 70);
        std::copy(fc.data.begin() + o * 30, fc.data.begin() + (o + 1) * 30, middle.data.begin() + o * 70 + 20);
        std::copy(fa.data.begin() + o * 20, fa.data.begin() + (o + 1) * 20, middle.data.begin() + o * 70 + 50);
    }
    CHECK(all_close(out.at("middle"), middle, 0, 0));
    CHECK(to_ints(out.at("ints")) == std::vector<int64_t>{7, 8, -1, 0, 9});
}

// Non-causal attention over a static number of queries with keys masked by a -inf bias: SigLIP's shape (head dim 72,
// 16 heads, outside the per-row kernels) and the action expert's (50 queries over a padded prefix, MQA, head dim 256,
// and the same at 128). Operands as graph inputs, and through a reshape (an f16 session stores those in f16, which the
// fused matrix-engine kernel needs): 300 keys against 50 queries of an MQA head at 128 split the keys between
// work-groups there. Keys and values also as Concat(prefix part, the last 50), the expert's form, which the GEMM path
// reads in place (two GEMMs per product) at head dim 256.
inline void attention_key_bias(const Env& env) {
    TensorRng rng(43);
    struct Case {
        int64_t queries, keys, heads, kv_heads, dim;
    };
    for (const Case c : {Case{40, 40, 16, 16, 72}, Case{50, 300, 8, 1, 256}, Case{50, 300, 8, 1, 128}}) {
        CAPTURE(c.dim);
        const HostTensor q = rng.uniform(DType::F32, {c.queries, c.heads, c.dim});
        const HostTensor k = rng.uniform(DType::F32, {c.keys, c.kv_heads, c.dim});
        const HostTensor v = rng.uniform(DType::F32, {c.keys, c.kv_heads, c.dim});
        std::vector<float> bias(static_cast<size_t>(c.keys), 0.0f);
        for (int64_t j = 0; j < c.keys; ++j) {
            if (j % 7 == 3 || (j >= c.keys - 9 && j < c.keys - 2)) bias[static_cast<size_t>(j)] = -INFINITY;
        }
        const HostTensor key_bias = HostTensor::from_vector(DType::F32, {c.keys}, bias);
        AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = 1.0f / std::sqrt(static_cast<float>(c.dim));
        ref::AttentionSpec spec;
        spec.scale = attrs.scale;
        spec.causal = false;
        spec.key_bias.assign(bias.begin(), bias.end());
        const Tensor64 expected = ref::attention(to_f64(q), to_f64(k), to_f64(v), spec);
        spec.key_bias.clear();
        const Tensor64 unmasked = ref::attention(to_f64(q), to_f64(k), to_f64(v), spec);
        for (const bool reshaped : {false, true}) {
            for (const bool parts : {false, true}) {
            CAPTURE(reshaped);
            CAPTURE(parts);
            Graph g;
            const auto operand = [&](const char* name, const Dims& shape) {
                if (!reshaped) return g.input(name, DType::F32, shape);
                return g.reshape(g.input(name, DType::F32, {shape[0], shape[1] * shape[2]}), shape);
            };
            const int64_t tail = c.queries;  // the second part: the last 50 keys
            const auto kv = [&](const char* whole, const char* a, const char* b) {
                if (!parts) return operand(whole, {c.keys, c.kv_heads, c.dim});
                return g.concat({operand(a, {c.keys - tail, c.kv_heads, c.dim}), operand(b, {tail, c.kv_heads, c.dim})}, 0);
            };
            g.output("out", g.attention(operand("q", {c.queries, c.heads, c.dim}), kv("k", "k1", "k2"), kv("v", "v1", "v2"),
                                        g.input("bias", DType::F32, {c.keys}), attrs));
            const auto flat = [&](const HostTensor& t) {
                return reshaped ? t.reshaped({t.shape()[0], t.shape()[1] * t.shape()[2]}) : t.clone();
            };
            TensorMap inputs = {{"q", flat(q)}, {"bias", key_bias}};
            if (parts) {
                inputs["k1"] = flat(rows(k, 0, c.keys - tail));
                inputs["k2"] = flat(rows(k, c.keys - tail, c.keys));
                inputs["v1"] = flat(rows(v, 0, c.keys - tail));
                inputs["v2"] = flat(rows(v, c.keys - tail, c.keys));
            } else {
                inputs["k"] = flat(k);
                inputs["v"] = flat(v);
            }
            for (const std::string& impl : env.attention_impls) {
                CAPTURE(impl);
                SessionOptions o = env.options();
                if (impl != "auto") o.extra["attention"] = impl;
                const HostTensor out = run(env, g, inputs, o).at("out");
                CHECK(all_close(out, expected, env.attn_atol, env.attn_rtol));
                CHECK_FALSE(all_close(out, unmasked, env.attn_atol, env.attn_rtol));  // planted: the mask matters
            }
            }
        }
    }
}

}  // namespace xs::test::policy_ops
