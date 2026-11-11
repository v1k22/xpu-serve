// Quantized matmul timing by rows: a chain of MLP-shaped quantized matmuls at each of a list of row counts, through
// every path that takes them: up to 8 rows (a decode step, a speculative verify) the GEMV and its matrix-engine version
// (backend option gemv=vector / xmx); past them (a long verify, a short prompt tail, a policy's action tokens) the
// few-row kernel and dequantize + GEMM (few_rows=64 / 0). Reported as weight bytes streamed per second.
// Opt-in: XS_DEVICE_BENCH=1. XS_BENCH_FORMAT is q4 (default) or q8, XS_BENCH_GROUP the group size (default 32),
// XS_BENCH_HIDDEN and XS_BENCH_INTER the two sides of the matmuls (default 5120 and 17408: a large model's MLP),
// XS_BENCH_ROWS the row counts (default 1,2,3,4,5,6,7,8), XS_BENCH_PAIRS the chain's length in up / down pairs
// (default 8; two lengths tell a launch's fixed cost from a run's).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
        const auto env = [](const char* name, const char* def) {
            const char* v = std::getenv(name);
            return std::string(v != nullptr && *v != 0 ? v : def);
        };
        const int64_t group = std::stoll(env("XS_BENCH_GROUP", "32"));
        const int64_t hidden = std::stoll(env("XS_BENCH_HIDDEN", "5120"));
        const int64_t inter = std::stoll(env("XS_BENCH_INTER", "17408"));
        const std::string format_name = env("XS_BENCH_FORMAT", "q4");
        REQUIRE((format_name == "q4" || format_name == "q8"));
        const DType format = format_name == "q4" ? DType::Q4 : DType::Q8;
        std::vector<int64_t> rows;
        for (std::stringstream list(env("XS_BENCH_ROWS", "1,2,3,4,5,6,7,8")); list.good();) {
            std::string item;
            std::getline(list, item, ',');
            rows.push_back(std::stoll(item));
        }
        // up then down: by default 16 matmuls, about 0.9 GB of q4 weights at the default sizes
        const int kPairs = std::stoi(env("XS_BENCH_PAIRS", "8"));
        test::TensorRng rng(5);
        test::MemoryWeights w;
        size_t bytes = 0;
        for (int i = 0; i < kPairs; ++i) {
            for (const auto& [name, n, k] : {std::tuple<std::string, int64_t, int64_t>{"up", inter, hidden},
                                             std::tuple<std::string, int64_t, int64_t>{"down", hidden, inter}}) {
                const HostTensor dense = rng.uniform(DType::F32, {n, k}, 1.0f / std::sqrt(static_cast<float>(k)));
                bytes += static_cast<size_t>(format == DType::Q4 ? n * k / 2 + 2 * n * (k / group) * 2
                                                                 : n * k + n * (k / group) * 2);
                w.add_quantized(name + std::to_string(i), quant::quantize(dense.view(), format, group));
            }
        }
        Graph g;
        Value h = g.input("x", DType::F32, {kT, hidden});
        for (int i = 0; i < kPairs; ++i) {
            h = g.matmul(g.matmul(h, g.weight(w, "up" + std::to_string(i))), g.weight(w, "down" + std::to_string(i)));
        }
        g.output("out", h);
        const bool few = std::any_of(rows.begin(), rows.end(), [](int64_t m) { return m > 8; });
        const bool decode = std::any_of(rows.begin(), rows.end(), [](int64_t m) { return m <= 8; });
        std::vector<std::pair<std::string, std::string>> paths;  // backend option, value
        if (decode) paths.insert(paths.end(), {{"gemv", "vector"}, {"gemv", "xmx"}});
        if (few) paths.insert(paths.end(), {{"few_rows", "0"}, {"few_rows", "64"}});
        for (const auto& [option, value] : paths) {
            SessionOptions o = dev::options();
            o.extra[option] = value;
            const auto session = dev::backend().compile(g, w, o);
            std::string line = str_cat("device-timing ", format_name, " g", group, " matmul chain ", hidden, " x ", inter, " (",
                                       static_cast<double>(bytes) / 1e9, " GB), ", option, "=", value, " on ", dev::device(),
                                       ":");
            for (const int64_t m : rows) {
                if ((option == "gemv") != (m <= 8)) continue;
                const HostTensor x = rng.uniform(DType::F32, {m, hidden});
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
