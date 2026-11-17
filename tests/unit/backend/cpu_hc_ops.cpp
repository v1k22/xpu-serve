// The hyper-connection ops on cpu-ref, against fp64 references (hc_ops_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "hc_ops_cases.h"

namespace {

xs::test::hc::Env cpu_env() {
    xs::test::hc::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        static const auto backend = xs::make_cpu_ref_backend();
        return backend->compile(g, w, {});
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref hyper-connection ops") {
    TEST_CASE("mhc_route: 2 and 4 streams, a bias or none, 1, 3 and 20 iterations, large logits") {
        xs::test::hc::mhc_route(cpu_env());
    }
    TEST_CASE("stream_mix: read, mix, mix with the branch's update; one token and several") {
        xs::test::hc::stream_mix(cpu_env(), 1);
        xs::test::hc::stream_mix(cpu_env(), 5);
    }
}
