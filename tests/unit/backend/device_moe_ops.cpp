// The mixture-of-experts ops and block on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64
// references (moe_ops_cases.h).
#include "device_test_support.h"
#include "moe_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::moe::Env device_env() {
    xs::test::moe::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    const dev::Tolerance elementwise = dev::tolerance({1e-5, 1e-5}, {2e-3, 2e-3});
    const dev::Tolerance mm = dev::tolerance({1e-5, 1e-4}, {1e-2, 1e-2});
    env.atol = elementwise.atol;
    env.rtol = elementwise.rtol;
    env.mm_atol = mm.atol;
    env.mm_rtol = mm.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device moe ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("div, softmax and reduce_sum") { xs::test::moe::div_softmax_reduce(device_env()); }
    TEST_CASE("top-k values and indices, ties to the lower index") { xs::test::moe::topk_rows(device_env()); }
    TEST_CASE("expert matmul with shared and per-slot inputs") {
        xs::test::moe::expert_matmul_shared_and_per_slot(device_env());
    }
    TEST_CASE("quantized expert banks (q8, q4): few pairs and pairs grouped by expert") {
        xs::test::moe::expert_matmul_quantized(device_env());
    }
    TEST_CASE("moe block: softmax router, renormalized top-k") {
        xs::test::moe::moe_block(device_env(), xs::blocks::Router::Softmax, false);
    }
    TEST_CASE("moe block: q4 fused gate_up and down banks, 5 and 40 tokens") {
        xs::test::moe::moe_block(device_env(), xs::blocks::Router::Softmax, true, true, 5);
        xs::test::moe::moe_block(device_env(), xs::blocks::Router::Softmax, true, true, 40);
    }
    TEST_CASE("moe block: sigmoid router, routed scale, gated shared expert") {
        xs::test::moe::moe_block(device_env(), xs::blocks::Router::Sigmoid, true);
    }
}
