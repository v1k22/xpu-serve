// Sessions of several programs on cpu-ref: shared slots, KvRead, truncate inside a short step (session_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "session_cases.h"

namespace {

xs::test::session::Env cpu_env() {
    xs::test::session::Env env;
    env.compile = [](std::span<const xs::graph::Graph* const> programs, const xs::WeightStore& w, int64_t rewind) {
        static const auto backend = xs::make_cpu_ref_backend();
        xs::SessionOptions o;
        o.precision = xs::DType::F32;
        o.rewind_tokens = rewind;
        return backend->compile(programs, w, o);
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref programs") {
    TEST_CASE("truncate inside a short step equals never running the dropped tokens") {
        xs::test::session::truncate_inside_step(cpu_env());
    }
    TEST_CASE("a run asked for some outputs: just those, the state advanced all the same") {
        xs::test::session::requested_outputs(cpu_env());
    }
    TEST_CASE("programs share KV slots by label; KvRead attends without appending") {
        xs::test::session::shared_kv_slot(cpu_env());
    }
}
