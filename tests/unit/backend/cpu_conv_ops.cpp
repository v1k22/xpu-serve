// Conv2d on cpu-ref, against an fp64 reference (conv_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "conv_ops_cases.h"

namespace {

xs::test::conv::Env cpu_env() {
    xs::test::conv::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref conv ops") {
    TEST_CASE("conv2d: strided 3 x 3, causal and dilated 1-D, depthwise, grouped, symbolic batch") {
        xs::test::conv::cases(cpu_env());
    }
    TEST_CASE("conv2d: a flipped kernel is not the op") { xs::test::conv::flipped_kernel_differs(cpu_env()); }
}
