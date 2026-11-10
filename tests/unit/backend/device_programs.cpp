// Sessions of several programs on the GPU backend under test (XS_DEVICE_TEST_DEVICE): shared slots, KvRead, truncate
// inside a short step (session_cases.h).
#include "device_test_support.h"
#include "session_cases.h"

namespace dev = xs::test::device;

namespace {

xs::test::session::Env device_env() {
    xs::test::session::Env env;
    env.compile = [](std::span<const xs::graph::Graph* const> programs, const xs::WeightStore& w, int64_t rewind) {
        xs::SessionOptions o = dev::options();
        o.rewind_tokens = rewind;
        return dev::backend().compile(programs, w, o);
    };
    const dev::Tolerance tol = dev::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
    env.atol = tol.atol;
    env.rtol = tol.rtol;
    return env;
}

}  // namespace

TEST_SUITE("device programs" * doctest::skip(!dev::enabled())) {
    TEST_CASE("truncate inside a short step equals never running the dropped tokens") {
        xs::test::session::truncate_inside_step(device_env());
    }
    TEST_CASE("a run asked for some outputs: just those, the state advanced all the same") {
        xs::test::session::requested_outputs(device_env());
    }
    TEST_CASE("programs share KV slots by label; KvRead attends without appending") {
        xs::test::session::shared_kv_slot(device_env());
    }
}
