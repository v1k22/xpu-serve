// The hyper-connection ops on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64 references
// (hc_ops_cases.h).
#include "device_test_support.h"
#include "hc_ops_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::hc::Env device_env() {
    xs::test::hc::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    const dev::Tolerance t = dev::tolerance({1e-5, 1e-5}, {1e-4, 1e-4});
    env.atol = t.atol;
    env.rtol = t.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device hyper-connection ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("mhc_route: 2 and 4 streams, a bias or none, 1, 3 and 20 iterations, large logits") {
        xs::test::hc::mhc_route(device_env());
    }
    TEST_CASE("stream_mix: read, mix, mix with the branch's update; one token and several") {
        xs::test::hc::stream_mix(device_env(), 1);
        xs::test::hc::stream_mix(device_env(), 5);
    }
}
