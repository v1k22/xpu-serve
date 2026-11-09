// The mixture-of-experts ops and block on cpu-ref, against fp64 references (moe_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "moe_ops_cases.h"

namespace {

xs::test::moe::Env cpu_env() {
    xs::test::moe::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref moe ops") {
    TEST_CASE("div, softmax and reduce_sum") { xs::test::moe::div_softmax_reduce(cpu_env()); }
    TEST_CASE("top-k values and indices, ties to the lower index") { xs::test::moe::topk_rows(cpu_env()); }
    TEST_CASE("expert matmul with shared and per-slot inputs") {
        xs::test::moe::expert_matmul_shared_and_per_slot(cpu_env());
    }
    TEST_CASE("quantized expert banks (q8, q4): few pairs and pairs grouped by expert") {
        xs::test::moe::expert_matmul_quantized(cpu_env());
    }
    TEST_CASE("moe block: softmax router, renormalized top-k") {
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, false);
    }
    TEST_CASE("moe block: q4 fused gate_up and down banks, 5, 40 and 150 tokens") {
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, true, true, 5);
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, true, true, 40);
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, true, true, 150);
    }
    TEST_CASE("moe block: sigmoid router, routed scale, gated shared expert") {
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Sigmoid, true);
    }
    TEST_CASE("moe block, Gemma 4 routing: its own router input, per-expert scales, GeGLU experts") {
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, false, false, 7, true);
        xs::test::moe::moe_block(cpu_env(), xs::blocks::Router::Softmax, false, true, 40, true);
    }
}
