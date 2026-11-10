// SignMatMul and the shortlist head on cpu-ref, against fp64 references (shortlist_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "shortlist_cases.h"

namespace {

xs::test::shortlist::Env cpu_env() {
    xs::test::shortlist::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref shortlist ops") {
    TEST_CASE("sign matmul: every row's one-bit score") { xs::test::shortlist::sign_matmul_shapes(cpu_env()); }
    TEST_CASE("shortlist head: the best rows by sketch, their exact logits, the head's pick") {
        xs::test::shortlist::shortlist_head_picks(cpu_env());
    }
}
