// What a session reserves at compile time with SessionOptions::max_step_tokens: the KV cache of the whole context and
// the largest step's memory, or a refusal naming them when they do not fit the device.
#include <doctest/doctest.h>

#include <string>

#include "device_test_support.h"

using namespace xs;
using namespace xs::graph;
namespace dev = xs::test::device;

namespace {

Graph cache_graph(int64_t window) {
    Graph g;
    const int slot = g.add_kv_slot(8, 128, window, "layer0");
    const Value q = g.input("q", DType::F32, {kT, 8, 128});
    const auto [k_all, v_all] = g.kv_cache(slot, g.input("k", DType::F32, {kT, 8, 128}),
                                           g.input("v", DType::F32, {kT, 8, 128}));
    AttentionAttrs attrs;
    attrs.scale = 0.088f;
    attrs.sliding_window = window;
    g.output("out", g.attention(q, k_all, v_all, attrs));
    return g;
}

}  // namespace

TEST_SUITE("device memory" * doctest::skip(!dev::enabled())) {
    TEST_CASE("a context whose KV cache cannot fit the device is refused at compile time, naming what fits") {
        SessionOptions o = dev::options(int64_t{1} << 24);  // 16M tokens of 4 KiB (f16): 64 GiB of keys and values
        o.max_step_tokens = 256;
        try {
            dev::backend().compile(cache_graph(0), test::MemoryWeights{}, o);
            FAIL("compiled");
        } catch (const Error& e) {
            const std::string what = e.what();
            CAPTURE(what);
            CHECK(what.find("does not fit") != std::string::npos);
            CHECK(what.find(dev::is_f16() ? "4096 bytes a token" : "8192 bytes a token") != std::string::npos);
            CHECK(what.find("tokens fit") != std::string::npos);
        }
        // Without max_step_tokens nothing is reserved: the cache grows as steps need it.
        o.max_step_tokens = 0;
        CHECK_NOTHROW(dev::backend().compile(cache_graph(0), test::MemoryWeights{}, o));
    }

    TEST_CASE("a context that fits is reserved and runs, with and without a sliding window") {
        test::TensorRng rng(3);
        for (const int64_t window : {int64_t{0}, int64_t{64}}) {
            CAPTURE(window);
            SessionOptions o = dev::options(65536);
            o.max_step_tokens = 32;
            const auto session = dev::backend().compile(cache_graph(window), test::MemoryWeights{}, o);
            for (const int64_t t : {int64_t{32}, int64_t{1}}) {
                TensorMap out;
                session->run({{"q", rng.uniform(DType::F32, {t, 8, 128})},
                              {"k", rng.uniform(DType::F32, {t, 8, 128})},
                              {"v", rng.uniform(DType::F32, {t, 8, 128})}},
                             out);
                CHECK(out.at("out").shape() == Dims{t, 8, 128});
            }
            CHECK(session->num_cached() == 33);
        }
    }
}
