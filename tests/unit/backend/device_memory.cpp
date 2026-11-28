// What a session reserves at compile time with SessionOptions::max_step_tokens: the KV cache of the whole context and
// the largest step's memory, or a refusal naming them when they do not fit the device.
#include <doctest/doctest.h>

#include <string>

#include "device_test_support.h"
#if XS_HAVE_SYCL
#include "backend/sycl/sycl_backend.h"
#endif

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
        o.max_step_tokens = {256};
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
        o.max_step_tokens.clear();
        CHECK_NOTHROW(dev::backend().compile(cache_graph(0), test::MemoryWeights{}, o));
    }

    TEST_CASE("a context that fits is reserved and runs, with and without a sliding window") {
        test::TensorRng rng(3);
        for (const int64_t window : {int64_t{0}, int64_t{64}}) {
            CAPTURE(window);
            SessionOptions o = dev::options(65536);
            o.max_step_tokens = {32};
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

#if XS_HAVE_SYCL
    TEST_CASE("a sliding window kept alone stays in the memory reserved for it, chunks and decode far past it") {
        // Two sessions reserved at compile time over the same steps: one keeping its window alone (it slides the
        // window back to the front of its buffers many times), one keeping every key. Neither allocates past its
        // reservation; their outputs agree.
        test::TensorRng rng(5);
        const auto compile = [](bool window_kv) {
            SessionOptions o = dev::options(4096);
            o.max_step_tokens = {64};
            o.window_kv = window_kv;
            return dev::backend().compile(cache_graph(48), test::MemoryWeights{}, o);
        };
        const auto windowed = compile(true);
        const auto full = compile(false);
        const size_t reserved = sycl_device_bytes();
        const dev::Tolerance tol = dev::tolerance({2e-5, 2e-5}, {5e-3, 5e-3});
        int64_t cached = 0;
        for (int i = 0; i < 48; ++i) {
            const int64_t t = i % 4 == 0 ? 64 : (i % 4 == 1 ? 37 : 1);
            const TensorMap in = {{"q", rng.uniform(DType::F32, {t, 8, 128})},
                                  {"k", rng.uniform(DType::F32, {t, 8, 128})},
                                  {"v", rng.uniform(DType::F32, {t, 8, 128})}};
            TensorMap got, want;
            windowed->run(in, got);
            full->run(in, want);
            cached += t;
            CAPTURE(i);
            CHECK(test::all_close(got.at("out"), want.at("out"), tol.atol, tol.rtol));
            CHECK(sycl_device_bytes() == reserved);
        }
        CHECK(windowed->num_cached() == cached);
    }
#endif
}
