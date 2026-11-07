// The ops of hybrid linear-attention models on the OpenVINO device under test (XS_OV_TEST_DEVICE), against fp64
// references (recurrent_ops_cases.h). GatedDelta lowers to OpenVINO's GatedDeltaNet.
#include "ov_test_support.h"
#include "recurrent_ops_cases.h"

namespace {

xs::test::recurrent::Env ov_env() {
    namespace ovt = xs::test::ov;
    xs::test::recurrent::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return ovt::backend().compile(g, w, ovt::options());
    };
    const ovt::Tolerance tol = ovt::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
    env.atol = tol.atol;
    env.rtol = tol.rtol;
    return env;
}

}  // namespace

TEST_SUITE("openvino recurrent ops") {
    TEST_CASE("sigmoid, silu and softplus") { xs::test::recurrent::unary_ops(ov_env()); }
    TEST_CASE("partial rotary rope") { xs::test::recurrent::partial_rope(ov_env()); }
    TEST_CASE("causal conv carries its window across steps") { xs::test::recurrent::causal_conv_steps(ov_env()); }
    TEST_CASE("gated delta rule carries its state across steps, grouped heads") {
        xs::test::recurrent::gated_delta_steps(ov_env());
    }
}
