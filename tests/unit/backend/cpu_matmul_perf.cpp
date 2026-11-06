// Throughput guard for the cpu-ref matmul: a decode-shaped 1536 -> 6144 projection (T = 1), so slow regressions show.
#include <doctest/doctest.h>

#include <chrono>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"

using namespace xs;
using namespace xs::graph;

TEST_SUITE("cpu-ref perf") {
    TEST_CASE("matmul 1536 -> 6144 at T = 1") {
        constexpr int64_t kIn = 1536;
        constexpr int64_t kOut = 6144;
        constexpr int kRuns = 50;
        test::TensorRng rng(31);
        test::MemoryWeights w;
        w.add("w", rng.uniform(DType::BF16, {kOut, kIn}));
        Graph g;
        g.output("out", g.matmul(g.input("x", DType::F32, {kT, kIn}), g.weight(w, "w")));
        const auto backend = make_cpu_ref_backend();
        const auto session = backend->compile(g, w, {});
        const TensorMap inputs{{"x", rng.uniform(DType::F32, {1, kIn})}};
        TensorMap out;
        session->run(inputs, out);  // warm-up
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kRuns; ++i)
            session->run(inputs, out);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / kRuns;
        const double gb_per_s = static_cast<double>(kIn * kOut * sizeof(float)) / seconds / 1e9;
        MESSAGE("cpu-ref matmul 1x1536 @ 6144x1536^T: ", seconds * 1e3, " ms/run, ", gb_per_s, " GB/s of f32 weights");
#ifdef NDEBUG
        CHECK(seconds < 0.02);  // lenient: a 2B-parameter decode is ~210 of these
#endif
    }
}
