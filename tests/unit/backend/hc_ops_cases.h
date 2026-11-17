#pragma once

// Backend-agnostic cases for the hyper-connection ops (MhcRoute, StreamMix), against fp64 references. Each backend's
// test file runs them with its own tolerance.

#include <doctest/doctest.h>

#include <functional>
#include <memory>

#include "backend_test_util.h"
#include "reference_ops.h"

namespace xs::test::hc {

using graph::Graph;
using graph::kT;
using graph::Value;

struct Env {
    std::function<std::unique_ptr<Session>(const Graph&, const WeightStore&)> compile;
    double atol = 1e-5;
    double rtol = 1e-5;
};

inline TensorMap run(const Env& env, const Graph& g, const TensorMap& inputs) {
    TensorMap out;
    env.compile(g, MemoryWeights{})->run(inputs, out);
    return out;
}

// 2 streams (SenseNova Loop's) and 4, with a bias and without, router outputs as large as a trained router's and
// larger (the logsumexp must subtract its peak); 20 iterations as the checkpoints run, and 1 and 3, which a backend
// that ran a fixed count, or the column step first, would get wrong.
inline void mhc_route(const Env& env) {
    TensorRng rng(41);
    const HostTensor z2 = rng.uniform(DType::F32, {7, 8}, 4.0f);
    const HostTensor b2 = rng.uniform(DType::F32, {8}, 2.0f);
    HostTensor z4 = rng.uniform(DType::F32, {7, 24}, 4.0f);
    z4.data<float>()[13] = 60.0f;
    z4.data<float>()[100] = -60.0f;
    Graph g;
    const Value v2 = g.input("z2", DType::F32, {kT, 8});
    const Value v4 = g.input("z4", DType::F32, {kT, 24});
    const auto out_maps = [&](const std::string& name, const graph::MhcMaps& m) {
        g.output(name + ".pre", m.pre);
        g.output(name + ".post", m.post);
        g.output(name + ".mix", m.mix);
    };
    out_maps("s2_20", g.mhc_route(v2, g.input("b2", DType::F32, {8}), 2, 20));
    out_maps("s2_1", g.mhc_route(v2, Value{}, 2, 1));
    out_maps("s4_20", g.mhc_route(v4, Value{}, 4, 20));
    out_maps("s4_3", g.mhc_route(v4, Value{}, 4, 3));
    const TensorMap out = run(env, g, {{"z2", z2}, {"b2", b2}, {"z4", z4}});
    const Tensor64 bias = to_f64(b2);
    const auto check = [&](const std::string& name, const ref::MhcMaps& e) {
        CAPTURE(name);
        CHECK(all_close(out.at(name + ".pre"), e.pre, env.atol, env.rtol));
        CHECK(all_close(out.at(name + ".post"), e.post, env.atol, env.rtol));
        CHECK(all_close(out.at(name + ".mix"), e.mix, env.atol, env.rtol));
    };
    check("s2_20", ref::mhc_route(to_f64(z2), &bias, 2, 20));
    check("s2_1", ref::mhc_route(to_f64(z2), nullptr, 2, 1));
    check("s4_20", ref::mhc_route(to_f64(z4), nullptr, 4, 20));
    check("s4_3", ref::mhc_route(to_f64(z4), nullptr, 4, 3));
    // planted: one iteration is not three; the bias counts
    CHECK_FALSE(all_close(out.at("s4_3.mix"), ref::mhc_route(to_f64(z4), nullptr, 4, 1).mix, env.atol, env.rtol));
    CHECK_FALSE(all_close(out.at("s2_20.mix"), ref::mhc_route(to_f64(z2), nullptr, 2, 20).mix, env.atol, env.rtol));
    // after the last column step every column of a mix sums to 1
    const std::vector<float> m = out.at("s2_20.mix").to_f32();
    for (size_t r = 0; r < 7; ++r) {
        CHECK(m[r * 4 + 0] + m[r * 4 + 2] == doctest::Approx(1.0).epsilon(1e-3));
        CHECK(m[r * 4 + 1] + m[r * 4 + 3] == doctest::Approx(1.0).epsilon(1e-3));
    }
}

// Streams of a width no multiple of a lane count and of a model's width; one token (a decode) and several; the read
// (P = 1), the mix alone and the mix with the branch's update.
inline void stream_mix(const Env& env, int64_t tokens) {
    for (const int64_t d : {int64_t{100}, int64_t{4096}}) {
        CAPTURE(d);
        TensorRng rng(static_cast<uint32_t>(42 + d + tokens));
        const HostTensor mix = rng.uniform(DType::F32, {tokens, 2, 2});
        const HostTensor pre = rng.uniform(DType::F32, {tokens, 1, 2});
        const HostTensor post = rng.uniform(DType::F32, {tokens, 2}, 2.0f);
        const HostTensor x = rng.uniform(DType::F32, {tokens, 2, d}, 3.0f);
        const HostTensor y = rng.uniform(DType::F32, {tokens, d}, 3.0f);
        const HostTensor m4 = rng.uniform(DType::F32, {tokens, 4, 4});
        const HostTensor x4 = rng.uniform(DType::F32, {tokens, 4, d}, 3.0f);
        Graph g;
        const Value vx = g.input("x", DType::F32, {kT, 2, d});
        const Value vmix = g.input("mix", DType::F32, {kT, 2, 2});
        g.output("read", g.stream_mix(g.input("pre", DType::F32, {kT, 1, 2}), vx));
        g.output("mix", g.stream_mix(vmix, vx));
        const Value vpost = g.input("post", DType::F32, {kT, 2});
        g.output("merge", g.stream_mix(vmix, vx, vpost, g.input("y", DType::F32, {kT, d})));
        g.output("mix4", g.stream_mix(g.input("m4", DType::F32, {kT, 4, 4}), g.input("x4", DType::F32, {kT, 4, d})));
        const TensorMap out =
            run(env, g, {{"x", x}, {"mix", mix}, {"pre", pre}, {"post", post}, {"y", y}, {"m4", m4}, {"x4", x4}});
        const Tensor64 post64 = to_f64(post), y64 = to_f64(y);
        CHECK(all_close(out.at("read"), ref::stream_mix(to_f64(pre), to_f64(x)), env.atol, env.rtol));
        CHECK(all_close(out.at("mix"), ref::stream_mix(to_f64(mix), to_f64(x)), env.atol, env.rtol));
        CHECK(all_close(out.at("merge"), ref::stream_mix(to_f64(mix), to_f64(x), &post64, &y64), env.atol, env.rtol));
        CHECK(all_close(out.at("mix4"), ref::stream_mix(to_f64(m4), to_f64(x4)), env.atol, env.rtol));
        // planted: the mix transposed
        CHECK_FALSE(all_close(out.at("mix"), ref::stream_mix(ref::transpose(to_f64(mix), {0, 2, 1}), to_f64(x)),
                              env.atol, env.rtol));
    }
}

}  // namespace xs::test::hc
