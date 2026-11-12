// GGUF block format matmul timing: a chain of MLP-shaped matmuls whose weights are in one block format, at each of a
// list of row counts, beside the same chain over a pack's q4 and q8. Reported as weights read per second (what a
// decode step's speed follows) and as bytes of the weight streamed per second.
// Opt-in: XS_DEVICE_BENCH=1. XS_BENCH_BLOCK lists formats by GGUF name (default: every format the kernels read, and
// q4, q8), XS_BENCH_HIDDEN and XS_BENCH_INTER are the two sides of the matmuls (default 2560 and 9216),
// XS_BENCH_ROWS the row counts (default 1,4,8), XS_BENCH_PAIRS the chain's length in up / down pairs (default 8).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
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

std::vector<std::string> list_of(const std::string& text) {
    std::vector<std::string> out;
    for (std::stringstream list(text); list.good();) {
        std::string item;
        std::getline(list, item, ',');
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

}  // namespace

TEST_SUITE("device timing" * doctest::skip(!dev::enabled())) {
    TEST_CASE("block format matmul chain by rows" * doctest::skip(!bench_enabled())) {
        const auto env = [](const char* name, const char* def) {
            const char* v = std::getenv(name);
            return std::string(v != nullptr && *v != 0 ? v : def);
        };
        const int64_t hidden = std::stoll(env("XS_BENCH_HIDDEN", "2560"));
        const int64_t inter = std::stoll(env("XS_BENCH_INTER", "9216"));
        const int pairs = std::stoi(env("XS_BENCH_PAIRS", "8"));
        const std::vector<std::string> formats =
            list_of(env("XS_BENCH_BLOCK",
                        "q4,q8,Q4_0,Q4_1,Q8_0,Q2_K,Q3_K,Q4_K,Q5_K,Q6_K,IQ4_NL,IQ4_XS,IQ3_XXS,IQ3_S,IQ2_XXS,IQ2_S"));
        std::vector<int64_t> rows;
        for (const std::string& item : list_of(env("XS_BENCH_ROWS", "1,4,8")))
            rows.push_back(std::stoll(item));
        for (const std::string& name : formats) {
            test::TensorRng rng(5);
            test::MemoryWeights w;
            double bytes = 0;
            double weights = 0;
            const auto block = block_format_from_name(name);
            REQUIRE((block || name == "q4" || name == "q8"));
            for (int i = 0; i < pairs; ++i) {
                for (const auto& [side, n, k] : {std::tuple<std::string, int64_t, int64_t>{"up", inter, hidden},
                                                 std::tuple<std::string, int64_t, int64_t>{"down", hidden, inter}}) {
                    const std::string tensor = side + std::to_string(i);
                    weights += static_cast<double>(n * k);
                    if (block) {
                        const int64_t count = n * k / block_layout(*block).weights;
                        w.add_block(tensor, *block, {n, k},
                                    test::random_blocks(*block, count, static_cast<uint32_t>(7 + i)));
                        bytes += static_cast<double>(block_storage_bytes(*block, n * k));
                    } else {
                        const HostTensor dense =
                            rng.uniform(DType::F32, {n, k}, 1.0f / std::sqrt(static_cast<float>(k)));
                        const DType format = name == "q4" ? DType::Q4 : DType::Q8;
                        w.add_quantized(tensor, quant::quantize(dense.view(), format, 32));
                        bytes += static_cast<double>(format == DType::Q4 ? n * k / 2 + n * (k / 32) * 4
                                                                         : n * k + n * (k / 32) * 2);
                    }
                }
            }
            Graph g;
            Value h = g.input("x", DType::F32, {kT, hidden});
            for (int i = 0; i < pairs; ++i) {
                h = g.matmul(g.matmul(h, g.weight(w, "up" + std::to_string(i))),
                             g.weight(w, "down" + std::to_string(i)));
            }
            g.output("out", h);
            const auto session = dev::backend().compile(g, w, dev::options());
            std::string line = str_cat("device-timing ", name, " matmul chain ", hidden, " x ", inter, " (",
                                       weights / 1e6, " M weights, ", bytes / 1e6, " MB, ", 8 * bytes / weights,
                                       " bits a weight) on ", dev::device(), ":");
            for (const int64_t m : rows) {
                // random blocks decode to values that overflow down a chain; the timing does not depend on them
                const HostTensor x(DType::F32, {m, hidden});
                TensorMap in{{"x", x}};
                TensorMap out;
                for (int i = 0; i < 3; ++i)
                    session->run(in, out);
                constexpr int kRuns = 20;
                Timer timer;
                for (int i = 0; i < kRuns; ++i)
                    session->run(in, out);
                const double ms = timer.elapsed_ms() / kRuns;
                line += str_cat(" M", m, " ", ms, " ms (", weights / (ms * 1e6), " G weights/s, ", bytes / (ms * 1e6),
                                " GB/s);");
            }
            MESSAGE(line);
        }
    }
}
