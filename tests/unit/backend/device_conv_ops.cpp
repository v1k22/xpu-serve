// Conv2d on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against an fp64 reference (conv_ops_cases.h).
#include "conv_ops_cases.h"
#include "device_test_support.h"

namespace dev = xs::test::device;

namespace {

xs::test::conv::Env device_env() {
    xs::test::conv::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    // f16: the patches and weights are rounded to f16 for the GEMM (products accumulate in f32)
    const dev::Tolerance t = dev::tolerance({1e-5, 1e-4}, {1e-2, 1e-2});
    env.atol = t.atol;
    env.rtol = t.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device conv ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("conv2d: strided 3 x 3, causal and dilated 1-D, depthwise, grouped, symbolic batch") {
        xs::test::conv::cases(device_env());
    }
    TEST_CASE("conv2d: a flipped kernel is not the op") { xs::test::conv::flipped_kernel_differs(device_env()); }
}
