// SignMatMul and the shortlist head on the GPU backend under test (XS_DEVICE_TEST_DEVICE), against fp64 references
// (shortlist_cases.h).
#include "device_test_support.h"
#include "shortlist_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::shortlist::Env device_env() {
    xs::test::shortlist::Env env;
    env.compile = [](const xs::graph::Graph& g, const xs::WeightStore& w) {
        return dev::backend().compile(g, w, dev::options());
    };
    env.atol = 2e-3;  // x is rounded to 12 bits of its largest magnitude
    env.rtol = dev::tolerance({1e-5, 1e-4}, {4e-3, 4e-3}).rtol;
    return env;
}

}  // namespace

TEST_SUITE("device shortlist ops" * doctest::skip(!dev::enabled())) {
    TEST_CASE("sign matmul: every row's one-bit score") { xs::test::shortlist::sign_matmul_shapes(device_env()); }
    TEST_CASE("shortlist head: the best rows by sketch, their exact logits, the head's pick") {
        xs::test::shortlist::shortlist_head_picks(device_env());
    }
}
