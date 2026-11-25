// Sessions of several programs on cpu-ref: shared slots, KvRead, truncate inside a short step (session_cases.h).
#include "backend/cpu_ref/cpu_backend.h"
#include "session_cases.h"

namespace {

// `kv`: how the KV cache stores its keys and values (f32 by default).
xs::test::session::Env cpu_env(const xs::quant::KvFormats& kv = {}) {
    xs::test::session::Env env;
    env.compile = [kv](std::span<const xs::graph::Graph* const> programs, const xs::WeightStore& w, int64_t rewind,
                       int64_t sequences) {
        static const auto backend = xs::make_cpu_ref_backend();
        xs::SessionOptions o;
        o.precision = xs::DType::F32;
        o.rewind_tokens = rewind;
        o.sequences = sequences;
        o.kv = kv;
        return backend->compile(programs, w, o);
    };
    return env;
}

}  // namespace

TEST_SUITE("cpu-ref programs") {
    TEST_CASE("truncate inside a short step equals never running the dropped tokens") {
        xs::test::session::truncate_inside_step(cpu_env());
    }
    TEST_CASE("the last step undone whole equals never running it") {
        xs::test::session::undo_last_step(cpu_env());
    }
    TEST_CASE("a windowed KV slot beside conv state: every output as over all the tokens kept") {
        xs::test::session::windowed_slot(cpu_env());
    }
    TEST_CASE("a run asked for some outputs: just those, the state advanced all the same") {
        xs::test::session::requested_outputs(cpu_env());
    }
    TEST_CASE("programs share KV slots by label; KvRead attends without appending") {
        xs::test::session::shared_kv_slot(cpu_env());
    }
    TEST_CASE("sequences: three in one session run in turns, each as a session of its own; reset touches one") {
        xs::test::session::sequences(cpu_env());
    }
    TEST_CASE("segments: steps of several sequences at once give what each sequence's own steps give") {
        xs::test::session::segments(cpu_env());
    }
    TEST_CASE("segments over an int8 and a TurboQuant KV cache, and int8 keys with TurboQuant values") {
        for (const char* kv : {"i8", "turbo4", "i8:turbo3"}) {
            CAPTURE(kv);
            xs::test::session::segments(cpu_env(xs::quant::parse_kv_formats(kv)), 128);
        }
    }
}
