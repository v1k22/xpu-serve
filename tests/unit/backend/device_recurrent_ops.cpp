// The ops of hybrid linear-attention models on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64
// references (recurrent_ops_cases.h).
#include "device_test_support.h"
#include "recurrent_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::recurrent::Env device_env() {
    xs::test::recurrent::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    const dev::Tolerance tol = dev::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
    env.atol = tol.atol;
    env.rtol = tol.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device recurrent ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("sigmoid, silu and softplus") { xs::test::recurrent::unary_ops(device_env()); }
    TEST_CASE("partial rotary rope") { xs::test::recurrent::partial_rope(device_env()); }
    TEST_CASE("causal conv carries its window across steps") { xs::test::recurrent::causal_conv_steps(device_env()); }
    TEST_CASE("gated delta rule carries its state across steps, grouped heads") {
        xs::test::recurrent::gated_delta_steps(device_env());
    }
}
