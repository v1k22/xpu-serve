// The ops robot policies add, on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64 references
// (policy_ops_cases.h).
#include "device_test_support.h"
#include "policy_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::policy_ops::Env device_env() {
    xs::test::policy_ops::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::SessionOptions& o) {
        return dev::backend().compile(g, xs::test::MemoryWeights{}, o);
    };
    env.options = [] { return dev::options(); };
    env.attention_impls = dev::attention_impls();
    const dev::Tolerance elementwise = dev::tolerance({1e-5, 1e-5}, {2e-3, 2e-3});
    const dev::Tolerance attn = dev::tolerance({1e-5, 1e-5}, {4e-3, 4e-3});
    env.atol = elementwise.atol;
    env.rtol = elementwise.rtol;
    env.attn_atol = attn.atol;
    env.attn_rtol = attn.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device policy ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("layer norm subtracts the mean") { xs::test::policy_ops::layer_norm_rows(device_env()); }
    TEST_CASE("concat along the first, a middle and an int axis") { xs::test::policy_ops::concat_axes(device_env()); }
    TEST_CASE("attention with a key bias over static queries, head dims 72, 128 and 256, every path") {
        xs::test::policy_ops::attention_key_bias(device_env());
    }
}
