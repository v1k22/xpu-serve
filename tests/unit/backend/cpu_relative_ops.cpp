// The ops of decoders without rotary positions on cpu-ref, against fp64 references (relative_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "relative_ops_cases.h"

namespace {

xs::test::relative::Env cpu_env() {
    xs::test::relative::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w, const std::string&) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref relative ops") {
    TEST_CASE("l2 clip scales long vectors down and passes short ones") { xs::test::relative::l2_clip(cpu_env()); }
    TEST_CASE("dynamic conv carries its window across steps, taps per token and group") {
        xs::test::relative::dynamic_conv_steps(cpu_env());
    }
    TEST_CASE("causal attention with a relative bias and a sink") {
        xs::test::relative::relative_attention(cpu_env());
    }
    TEST_CASE("attention over a window anchored at the step: a drafter's block over its context") {
        xs::test::relative::anchored_window_attention(cpu_env());
    }
}
