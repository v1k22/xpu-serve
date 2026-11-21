// The ops of decoders without rotary positions on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64
// references (relative_ops_cases.h).
#include "device_test_support.h"
#include "relative_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::relative::Env device_env() {
    xs::test::relative::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w, const std::string& attention) {
        xs::SessionOptions o = dev::options();
        if (attention != "auto") o.extra["attention"] = attention;
        return dev::backend().compile(g, w, o);
    };
    env.attention_impls = dev::attention_impls();
    const dev::Tolerance tol = dev::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
    env.atol = tol.atol;
    env.rtol = tol.rtol;
    return env;
}

// Over an int8 cache, whole or (`chunk`) the fused kernel's keys a chunk at a time.
xs::test::relative::Env int8_env(const std::string& chunk) {
    xs::test::relative::Env env = device_env();
    env.kv = {xs::quant::KvFormat::I8, xs::quant::KvFormat::I8};
    env.compile = [kv = env.kv, chunk](const xs::graph::Graph& g, const xs::WeightStore& w,
                                       const std::string& attention) {
        xs::SessionOptions o = dev::options();
        if (attention != "auto") o.extra["attention"] = attention;
        if (!chunk.empty()) o.extra["kv_chunk"] = chunk;
        o.kv = kv;
        return dev::backend().compile(g, w, o);
    };
    // f32: an int8 code the device rounds the other way at a tie moves a key by a step.
    const dev::Tolerance tol = dev::tolerance({1e-3, 1e-3}, {5e-3, 5e-3});
    env.atol = tol.atol;
    env.rtol = tol.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device relative ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("l2 clip scales long vectors down and passes short ones") { xs::test::relative::l2_clip(device_env()); }
    TEST_CASE("dynamic conv carries its window across steps, taps per token and group") {
        xs::test::relative::dynamic_conv_steps(device_env());
    }
    TEST_CASE("causal attention with a relative bias and a sink, on the per-row, GEMM and fused paths") {
        xs::test::relative::relative_attention(device_env());
    }
    TEST_CASE("the same over an int8 cache, whole and by chunks of 64 keys (the sink in the first chunk alone)") {
        xs::test::relative::relative_attention(int8_env(""));
        xs::test::relative::relative_attention(int8_env("64"));
    }
    TEST_CASE("attention over a window anchored at the step: a drafter's block over its context") {
        xs::test::relative::anchored_window_attention(device_env());
    }
}
