// The ops robot policies add, on cpu-ref, against fp64 references (policy_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "policy_ops_cases.h"

namespace {

xs::test::policy_ops::Env cpu_env() {
    xs::test::policy_ops::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::SessionOptions& o) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, xs::test::MemoryWeights{}, o);
    };
    env.options = [] { return xs::SessionOptions{}; };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref policy ops") {
    TEST_CASE("layer norm subtracts the mean") { xs::test::policy_ops::layer_norm_rows(cpu_env()); }
    TEST_CASE("concat along the first, a middle and an int axis") { xs::test::policy_ops::concat_axes(cpu_env()); }
    TEST_CASE("concats of computed values that share a first part") {
        xs::test::policy_ops::concat_of_computed(cpu_env());
    }
    TEST_CASE("attention with a key bias over static queries, head dims 72, 128 and 256") {
        xs::test::policy_ops::attention_key_bias(cpu_env());
    }
    TEST_CASE("attention within blocks of tokens, head dims 72 and 256, every path") {
        xs::test::policy_ops::attention_blocks(cpu_env());
    }
}
