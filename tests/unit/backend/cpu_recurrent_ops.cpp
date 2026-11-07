// The ops of hybrid linear-attention models on cpu-ref, against fp64 references (recurrent_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "recurrent_ops_cases.h"

namespace {

xs::test::recurrent::Env cpu_env() {
    xs::test::recurrent::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref recurrent ops") {
    TEST_CASE("sigmoid, silu and softplus") { xs::test::recurrent::unary_ops(cpu_env()); }
    TEST_CASE("partial rotary rope") { xs::test::recurrent::partial_rope(cpu_env()); }
    TEST_CASE("causal conv carries its window across steps") { xs::test::recurrent::causal_conv_steps(cpu_env()); }
    TEST_CASE("gated delta rule carries its state across steps, grouped heads") {
        xs::test::recurrent::gated_delta_steps(cpu_env());
    }
}
