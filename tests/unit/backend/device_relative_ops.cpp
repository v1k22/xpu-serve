// The ops of decoders without rotary positions on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64
// references (relative_ops_cases.h).
#include "device_test_support.h"
#include "relative_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::relative::Env device_env() {
    xs::test::relative::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    const dev::Tolerance tol = dev::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
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
    TEST_CASE("causal attention with a relative bias and a sink") {
        xs::test::relative::relative_attention(device_env());
    }
    TEST_CASE("attention over a window anchored at the step: a drafter's block over its context") {
        xs::test::relative::anchored_window_attention(device_env());
    }
}
