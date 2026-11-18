#pragma once

// Backend-agnostic cases for Conv2d, against an fp64 reference: Qwen3-ASR's audio front end (3 x 3, stride 2, padding
// 1, over one input channel and then many), and the 1-D convolutions of speech codecs (causal and dilated, depthwise,
// grouped), a symbolic batch. Each backend's test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <functional>
#include <memory>

#include "backend_test_util.h"
#include "reference_ops.h"

namespace xs::test::conv {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;
    double rtol = 1e-4;
};

struct Case {
    const char* name;
    Dims x, w;
    graph::Conv2dAttrs attrs;
    bool bias = true;
    bool symbolic_batch = false;
};

inline ref::Conv2dSpec spec_of(const graph::Conv2dAttrs& a) {
    return {a.stride_h, a.stride_w, a.pad_top, a.pad_bottom, a.pad_left, a.pad_right, a.dilation_h, a.dilation_w,
            a.groups};
}

inline void run_case(const Env& env, const Case& c, uint32_t seed) {
    CAPTURE(c.name);
    TensorRng rng(seed);
    const HostTensor x = rng.uniform(DType::F32, c.x, 2.0f);
    const HostTensor w = rng.uniform(DType::F32, c.w, 0.5f);
    const HostTensor b = rng.uniform(DType::F32, {c.w[0]}, 1.0f);
    MemoryWeights weights;
    weights.add("w", w);
    weights.add("b", b);
    Graph g;
    Dims xin = c.x;
    if (c.symbolic_batch) xin[0] = kT;
    const Value out = g.conv2d(g.input("x", DType::F32, xin), g.weight(weights, "w"),
                               c.bias ? g.weight(weights, "b") : Value{}, c.attrs);
    g.output("out", out);
    TensorMap result;
    env.compile(g, weights)->run({{"x", x}}, result);
    const Tensor64 b64 = to_f64(b);
    const Tensor64 expected = ref::conv2d(to_f64(x), to_f64(w), c.bias ? &b64 : nullptr, spec_of(c.attrs));
    CHECK(all_close(result.at("out"), expected, env.atol, env.rtol));
}

inline void cases(const Env& env) {
    graph::Conv2dAttrs s2;  // Qwen3-ASR: 3 x 3, stride 2, padding 1
    s2.stride_h = s2.stride_w = 2;
    s2.pad_top = s2.pad_bottom = s2.pad_left = s2.pad_right = 1;
    graph::Conv2dAttrs causal;  // a codec's causal, dilated 1-D convolution: kernel 7, dilation 3, padded on the left
    causal.dilation_w = 3;
    causal.pad_left = 18;
    graph::Conv2dAttrs depthwise = causal;
    depthwise.dilation_w = 1;
    depthwise.pad_left = 6;
    depthwise.groups = 8;
    graph::Conv2dAttrs grouped;
    grouped.groups = 2;
    grouped.pad_top = grouped.pad_left = 1;
    const Case all[] = {
        {"one input channel, stride 2", {3, 1, 128, 100}, {16, 1, 3, 3}, s2},
        {"many channels, stride 2, odd sizes", {2, 16, 33, 25}, {8, 16, 3, 3}, s2},
        {"causal dilated 1-D", {2, 6, 1, 50}, {5, 6, 1, 7}, causal},
        {"depthwise 1-D", {2, 8, 1, 40}, {8, 1, 1, 7}, depthwise, false},
        {"grouped, padded on two sides only", {1, 6, 9, 10}, {4, 3, 3, 3}, grouped},
        {"symbolic batch", {4, 2, 8, 9}, {3, 2, 3, 3}, s2, true, true},
    };
    uint32_t seed = 50;
    for (const Case& c : all) run_case(env, c, seed++);
}

// planted: a kernel read flipped (convolution instead of cross-correlation) is not the op
inline void flipped_kernel_differs(const Env& env) {
    TensorRng rng(70);
    const HostTensor x = rng.uniform(DType::F32, {1, 2, 6, 6});
    const HostTensor w = rng.uniform(DType::F32, {3, 2, 3, 3});
    HostTensor flipped(DType::F32, w.shape());
    for (int64_t i = 0; i < w.numel(); ++i) {
        const int64_t v = i % 3, u = (i / 3) % 3, rest = i / 9;
        flipped.data<float>()[i] = w.data<float>()[rest * 9 + (2 - u) * 3 + (2 - v)];
    }
    MemoryWeights weights;
    weights.add("w", w);
    Graph g;
    g.output("out", g.conv2d(g.input("x", DType::F32, {1, 2, 6, 6}), g.weight(weights, "w"), Value{}, {}));
    TensorMap result;
    env.compile(g, weights)->run({{"x", x}}, result);
    CHECK_FALSE(all_close(result.at("out"), ref::conv2d(to_f64(x), to_f64(flipped), nullptr, {}), env.atol, env.rtol));
}

}  // namespace xs::test::conv
