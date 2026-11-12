// The GPU backend with weights in GGUF block formats: MatMul at every row count the GEMV's variants take and past
// them (the weight written out for a GEMM), a residual folded into that GEMM, and Gather, against fp64 over the values
// the host decodes (quant/block_decode.h). Blocks are random bytes, so every code and scale field takes every value.
// The reference option block=f16 (weights decoded as they are uploaded) is held to the same values.

#include <doctest/doctest.h>

#include <cmath>
#include <string>
#include <vector>

#include "device_test_support.h"
#include "reference_ops.h"

using namespace xs;
using namespace xs::graph;
namespace dev = xs::test::device;
using test::Tensor64;
using test::to_f64;

namespace {

constexpr BlockFormat kFormats[] = {
    BlockFormat::Q4_0,    BlockFormat::Q4_1,  BlockFormat::Q8_0,    BlockFormat::Q2_K,   BlockFormat::Q3_K,
    BlockFormat::Q4_K,    BlockFormat::Q5_K,  BlockFormat::Q6_K,    BlockFormat::IQ4_NL, BlockFormat::IQ4_XS,
    BlockFormat::IQ3_XXS, BlockFormat::IQ3_S, BlockFormat::IQ2_XXS, BlockFormat::IQ2_S,
};

std::vector<uint8_t> blocks(BlockFormat f, const Dims& shape, uint32_t seed) {
    return test::random_blocks(f, numel(shape) / block_layout(f).weights, seed);
}

// The formats the backend reads through a pack's q8 / q4 kernels, whose matrix-engine GEMV (from four rows) reads x
// in f16.
bool pack_kernels(BlockFormat f) {
    return f == BlockFormat::Q4_0 || f == BlockFormat::Q4_1 || f == BlockFormat::Q8_0;
}

double max_abs(const Tensor64& t) {
    double m = 0;
    for (const double v : t.data)
        m = std::max(m, std::abs(v));
    return m;
}

}  // namespace

TEST_SUITE("device ops") {
    TEST_CASE("matmul and gather on GGUF block format weights against fp64 of the decoded values" *
              doctest::skip(!dev::enabled())) {
        test::TensorRng rng(71);
        // 768 columns: 24 units of 32, so the lanes of a sub-group take one or two; 37 rows fill no sub-group pair
        for (const auto& [n, k] : {std::pair<int64_t, int64_t>{37, 768}, std::pair<int64_t, int64_t>{64, 2560}}) {
            for (const BlockFormat f : kFormats) {
                CAPTURE(block_format_name(f));
                CAPTURE(k);
                test::MemoryWeights w;
                w.add_block("w", f, {n, k}, blocks(f, {n, k}, 11));
                w.add_block("table", f, {21, k}, blocks(f, {21, k}, 12));
                const Tensor64 w64 = to_f64(w.tensor("w"));
                const double w_max = max_abs(w64);
                for (const char* mode : {"auto", "f16"}) {
                    CAPTURE(mode);
                    SessionOptions o = dev::options();
                    o.extra["block"] = mode;
                    for (const int64_t m : {1, 2, 3, 5, 8, 12, 70}) {
                        CAPTURE(m);
                        const HostTensor x = rng.uniform(DType::F32, {m, k});
                        Graph g;
                        g.output("out", g.matmul(g.input("x", DType::F32, {m, k}), g.weight(w, "w")));
                        const TensorMap out = dev::run(g, w, {{"x", x}}, o);
                        // up to 8 rows the GEMV reads the exact values; past them (and with block=f16) the weight is
                        // rounded to the session's precision first
                        const bool exact = m <= (pack_kernels(f) ? 3 : 8) && std::string(mode) == "auto";
                        const double scale = w_max * std::sqrt(static_cast<double>(k));
                        const dev::Tolerance tol = exact ? dev::Tolerance{2e-5 * scale, 2e-5}
                                                         : dev::tolerance({2e-5 * scale, 2e-5}, {3e-3 * scale, 3e-3});
                        CHECK(dev::close(str_cat("matmul ", block_format_name(f)), out.at("out"),
                                         test::ref::matmul(to_f64(x), w64), tol));
                    }
                    {
                        const std::vector<int32_t> ids{20, 0, 7, 7, 13};
                        Graph g;
                        g.output("rows", g.gather(g.weight(w, "table"), g.input("ids", DType::I32, {5})));
                        const TensorMap out = dev::run(g, w, {{"ids", test::make_tensor(DType::I32, {5}, ids)}}, o);
                        const HostTensor& table = w.tensor("table");
                        Tensor64 want;
                        want.shape = {5, k};
                        for (const int32_t id : ids) {
                            for (int64_t c = 0; c < k; ++c)
                                want.data.push_back(table.data<float>()[id * k + c]);
                        }
                        const bool exact = std::string(mode) == "auto";
                        CHECK(dev::close(str_cat("gather ", block_format_name(f)), out.at("rows"), want,
                                         exact ? dev::Tolerance{0, 2e-6} : dev::tolerance({0, 2e-6}, {0, 1e-3})));
                    }
                }
            }
        }
    }

    TEST_CASE("a residual folded into the GEMM of a GGUF block format weight" * doctest::skip(!dev::enabled())) {
        test::TensorRng rng(72);
        const int64_t n = 64, k = 512, m = 40;
        for (const BlockFormat f : {BlockFormat::Q5_K, BlockFormat::IQ3_S, BlockFormat::Q4_0}) {
            CAPTURE(block_format_name(f));
            test::MemoryWeights w;
            w.add_block("w", f, {n, k}, blocks(f, {n, k}, 21));
            const HostTensor x = rng.uniform(DType::F32, {m, k});
            const HostTensor a = rng.uniform(DType::F32, {m, n});
            const HostTensor b = rng.uniform(DType::F32, {m, n});
            Graph g;
            const Value h = g.add(g.input("a", DType::F32, {m, n}), g.input("b", DType::F32, {m, n}));
            g.output("out", g.add(h, g.matmul(g.input("x", DType::F32, {m, k}), g.weight(w, "w"))));
            const TensorMap out = dev::run(g, w, {{"x", x}, {"a", a}, {"b", b}});
            const auto plus = [](double p, double q) { return p + q; };
            const Tensor64 w64 = to_f64(w.tensor("w"));
            const Tensor64 want = test::ref::broadcast(test::ref::broadcast(to_f64(a), to_f64(b), plus),
                                                       test::ref::matmul(to_f64(x), w64), plus);
            const double scale = max_abs(w64) * std::sqrt(static_cast<double>(k));
            CHECK(dev::close("residual block gemm", out.at("out"), want,
                             dev::tolerance({2e-5 * scale, 2e-5}, {3e-3 * scale, 3e-3})));
        }
    }
}
