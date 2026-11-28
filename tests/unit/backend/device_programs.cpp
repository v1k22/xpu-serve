// Sessions of several programs on the GPU backend under test (XS_DEVICE_TEST_DEVICE): shared slots, KvRead, truncate
// inside a short step (session_cases.h).
#include "device_test_support.h"
#include "session_cases.h"

namespace dev = xs::test::device;

namespace {

// `kv`: how the KV cache stores its keys and values (the precision's float by default). `window_kv`:
// SessionOptions::window_kv.
xs::test::session::Env device_env(const xs::quant::KvFormats& kv = {}, bool window_kv = false) {
    xs::test::session::Env env;
    env.compile = [kv, window_kv](std::span<const xs::graph::Graph* const> programs, const xs::WeightStore& w,
                                  int64_t rewind, int64_t sequences) {
        xs::SessionOptions o = dev::options();
        o.rewind_tokens = rewind;
        o.sequences = sequences;
        o.kv = kv;
        o.window_kv = window_kv;
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
    TEST_CASE("the last step undone whole equals never running it") {
        xs::test::session::undo_last_step(device_env());
    }
    TEST_CASE("a windowed KV slot beside conv state: every output as over all the tokens kept") {
        xs::test::session::windowed_slot(device_env());
    }
    TEST_CASE("a run asked for some outputs: just those, the state advanced all the same") {
        xs::test::session::requested_outputs(device_env());
    }
    TEST_CASE("programs share KV slots by label; KvRead attends without appending") {
        xs::test::session::shared_kv_slot(device_env());
    }
    TEST_CASE("sequences: three in one session run in turns, each as a session of its own; reset touches one") {
        xs::test::session::sequences(device_env());
    }
    TEST_CASE("segments: steps of several sequences at once give what each sequence's own steps give") {
        xs::test::session::segments(device_env());
    }
    TEST_CASE("a snapshot of a sequence's state, restored after steps past its sliding window") {
        xs::test::session::state_snapshot(device_env());
        for (const char* kv : {"i8", "turbo4"}) {
            CAPTURE(kv);
            xs::test::session::state_snapshot(device_env(xs::quant::parse_kv_formats(kv)), 128);
        }
    }
    TEST_CASE("a prefix copied from another sequence, with its snapshot") {
        for (const int64_t window : {0, 6}) {
            CAPTURE(window);
            xs::test::session::copy_prefix(device_env(), window);
            for (const char* kv : {"i8", "turbo4"}) {
                CAPTURE(kv);
                xs::test::session::copy_prefix(device_env(xs::quant::parse_kv_formats(kv)), window, 128);
            }
        }
    }
    TEST_CASE("segments cut back inside each sequence's segment: speculative rounds of several sequences") {
        xs::test::session::segments_truncate(device_env());
        for (const char* kv : {"i8", "turbo4"}) {
            CAPTURE(kv);
            xs::test::session::segments_truncate(device_env(xs::quant::parse_kv_formats(kv)), 128);
        }
    }
    TEST_CASE("attention alone keeping its window alone (window_kv): steps and cuts, snapshots, copies, segments") {
        namespace s = xs::test::session;
        using Layer = s::AttentionLayer;
        const s::Env env = device_env({}, true);
        s::windowed_slot<Layer>(env);
        s::state_snapshot<Layer>(env);
        for (const int64_t window : {0, 6}) s::copy_prefix<Layer>(env, window);
        s::sequences<Layer>(env);
        s::segments<Layer>(env);
        s::segments_truncate<Layer>(env);
        for (const char* kv : {"i8", "turbo4"}) {
            CAPTURE(kv);
            const s::Env quantized = device_env(xs::quant::parse_kv_formats(kv), true);
            s::state_snapshot<Layer>(quantized, 128);
            s::copy_prefix<Layer>(quantized, 6, 128);
            s::segments_truncate<Layer>(quantized, 128);
        }
    }
    TEST_CASE("attention alone keeping its window alone cuts back no further than that window; keeping all, anywhere") {
        const xs::test::session::AttentionLayer layer(6);
        const xs::graph::Graph* programs[] = {&layer.graph};
        xs::test::TensorRng rng(3);
        for (const bool window_kv : {true, false}) {
            CAPTURE(window_kv);
            const auto session = device_env({}, window_kv).compile(programs, layer.weights, 4, 1);
            xs::TensorMap out;
            // far past the window and the rows a slot holds beyond it (its buffers then slid the window back)
            for (int i = 0; i < 20; ++i) session->run(xs::test::session::AttentionLayer::inputs(rng, 50), out);
            if (window_kv) {
                CHECK_THROWS_AS(session->truncate(5), xs::Error);
                CHECK(session->num_cached() == 1000);
                session->truncate(990);  // inside the last step
                CHECK(session->num_cached() == 990);
            } else {
                session->truncate(5);
                CHECK(session->num_cached() == 5);
            }
        }
    }
    TEST_CASE("segments over an int8 and a TurboQuant KV cache, and int8 keys with TurboQuant values") {
        for (const char* kv : {"i8", "turbo4", "i8:turbo3"}) {
            CAPTURE(kv);
            xs::test::session::segments(device_env(xs::quant::parse_kv_formats(kv)), 128);
        }
    }
}
