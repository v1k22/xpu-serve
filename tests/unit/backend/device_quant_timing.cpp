// Quantized matmul timing by rows: a chain of MLP-shaped q4 matmuls at M = 1..8 rows (a decode step and a speculative
// verify) through the GEMV and its matrix-engine version (backend option gemv=vector / xmx), reported as weight bytes
// streamed per second. Opt-in: XS_DEVICE_BENCH=1; XS_BENCH_GROUP sets the group size (default 32), XS_BENCH_INTER the
// wide side of the matmuls (default 17408: an MLP; smaller for a model's projections, whose launches are short).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <string>

#include "device_test_support.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/timer.h"

using namespace xs;
using graph::Graph;
using graph::kT;
using graph::Value;
namespace dev = xs::test::device;

namespace {

bool bench_enabled() {
    const char* v = std::getenv("XS_DEVICE_BENCH");
    return v != nullptr && std::string(v) == "1";
}

}  // namespace

TEST_SUITE("device timing" * doctest::skip(!dev::enabled())) {
    TEST_CASE("quantized matmul chain by rows" * doctest::skip(!bench_enabled())) {
        const char* g_env = std::getenv("XS_BENCH_GROUP");
        const int64_t group = g_env != nullptr ? std::atoll(g_env) : 32;
        const char* inter_env = std::getenv("XS_BENCH_INTER");
        constexpr int64_t kHidden = 5120;
        const int64_t kInter = inter_env != nullptr ? std::atoll(inter_env) : 17408;
        constexpr int kPairs = 8;  // up then down: 16 matmuls, about 0.9 GB of q4 weights at the default sizes
        test::TensorRng rng(5);
        test::MemoryWeights w;
        size_t bytes = 0;
        for (int i = 0; i < kPairs; ++i) {
            for (const auto& [name, n, k] : {std::tuple<std::string, int64_t, int64_t>{"up", kInter, kHidden},
                                             std::tuple<std::string, int64_t, int64_t>{"down", kHidden, kInter}}) {
                const HostTensor dense = rng.uniform(DType::F32, {n, k}, 1.0f / std::sqrt(static_cast<float>(k)));
                quant::Quantized q = quant::quantize(dense.view(), DType::Q4, group);
                bytes += static_cast<size_t>(n * k / 2 + 2 * n * (k / group) * 2);
                w.add_quantized(name + std::to_string(i), std::move(q));
            }
        }
        Graph g;
        Value h = g.input("x", DType::F32, {kT, kHidden});
        for (int i = 0; i < kPairs; ++i) {
            h = g.matmul(g.matmul(h, g.weight(w, "up" + std::to_string(i))), g.weight(w, "down" + std::to_string(i)));
        }
        g.output("out", h);
        for (const char* path : {"vector", "xmx"}) {
            SessionOptions o = dev::options();
            o.extra["gemv"] = path;
            const auto session = dev::backend().compile(g, w, o);
            std::string line = str_cat("device-timing q4 g", group, " matmul chain 5120 x ", kInter, " (", bytes / 1e9, " GB), gemv=", path,
                                       " on ", dev::device(), ":");
            for (int64_t m = 1; m <= 8; ++m) {
                const HostTensor x = rng.uniform(DType::F32, {m, kHidden});
                TensorMap in{{"x", x}};
                TensorMap out;
                for (int i = 0; i < 3; ++i) session->run(in, out);
                constexpr int kRuns = 20;
                Timer timer;
                for (int i = 0; i < kRuns; ++i) session->run(in, out);
                const double ms = timer.elapsed_ms() / kRuns;
                line += str_cat(" M", m, " ", ms, " ms (", static_cast<double>(bytes) / (ms * 1e6), " GB/s);");
            }
            MESSAGE(line);
        }
    }
}
