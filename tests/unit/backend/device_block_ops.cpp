// The GPU backend with weights in GGUF block formats: MatMul at every row count the GEMV's variants take and past
// them (the weight written out for a GEMM), a residual folded into that GEMM, Gather, and ExpertMatMul over a bank
// (few pairs, pairs grouped by expert, grouped GEMMs), against fp64 over the values the host decodes
// (quant/block_decode.h). Blocks are random bytes, so every code and scale field takes every value.
// The reference option block=f16 (weights decoded as they are uploaded) is held to the same values, and so are both
// GEMVs at every row count they take (gemv=vector, gemv=xmx).

#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
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
    BlockFormat::Q4_0,    BlockFormat::Q4_1,  BlockFormat::Q5_0,    BlockFormat::Q5_1,   BlockFormat::Q8_0,
    BlockFormat::Q2_K,    BlockFormat::Q3_K,  BlockFormat::Q4_K,    BlockFormat::Q5_K,   BlockFormat::Q6_K,
    BlockFormat::IQ1_S,   BlockFormat::IQ1_M, BlockFormat::IQ2_XXS, BlockFormat::IQ2_XS, BlockFormat::IQ2_S,
    BlockFormat::IQ3_XXS, BlockFormat::IQ3_S, BlockFormat::IQ4_NL,  BlockFormat::IQ4_XS, BlockFormat::NVFP4,
    BlockFormat::MXFP4,   BlockFormat::Q2_0,
};
static_assert(sizeof(kFormats) / sizeof(kFormats[0]) == kBlockFormats);

std::vector<uint8_t> blocks(BlockFormat f, const Dims& shape, uint32_t seed) {
    return test::random_blocks(f, numel(shape) / block_layout(f).weights, seed);
}

// The formats the backend reads through a pack's q8 / q4 kernels.
bool pack_kernels(BlockFormat f) {
    return f == BlockFormat::Q4_0 || f == BlockFormat::Q4_1 || f == BlockFormat::Q8_0;
}

// How the weights are uploaded (block=auto: as stored, f16: decoded) and which GEMV reads up to 8 rows.
struct Mode {
    std::string block;
    std::string gemv;
};

// From how many rows a matmul reads x in f16 (the matrix-engine GEMVs: a pack's q4 from four rows, its q8 from three,
// the block formats' from two, all from one with gemv=xmx; and past 8 rows, the GEMM).
int64_t f16_x_from(BlockFormat f, const Mode& mode) {
    if (mode.gemv == "vector") return 9;
    if (mode.gemv == "xmx") return 1;
    if (f == BlockFormat::Q8_0) return 3;
    return pack_kernels(f) ? 4 : 2;
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
        // 768 columns: 24 units of 32, so the lanes of a sub-group take one or two; 37 rows fill no sub-group pair.
        // 1,152: no whole number of 256 columns, which only the formats of narrower blocks can hold.
        for (const auto& [n, k] : {std::pair<int64_t, int64_t>{37, 768}, std::pair<int64_t, int64_t>{64, 2560},
                                   std::pair<int64_t, int64_t>{19, 1152}}) {
            for (const BlockFormat f : kFormats) {
                if (k % block_layout(f).weights != 0) continue;
                CAPTURE(block_format_name(f));
                CAPTURE(k);
                test::MemoryWeights w;
                w.add_block("w", f, {n, k}, blocks(f, {n, k}, 11));
                w.add_block("table", f, {21, k}, blocks(f, {21, k}, 12));
                const Tensor64 w64 = to_f64(w.tensor("w"));
                const double w_max = max_abs(w64);
                for (const Mode& mode : {Mode{"auto", "auto"}, Mode{"f16", "auto"}, Mode{"auto", "vector"},
                                         Mode{"auto", "xmx"}}) {
                    CAPTURE(mode.block);
                    CAPTURE(mode.gemv);
                    SessionOptions o = dev::options();
                    o.extra["block"] = mode.block;
                    o.extra["gemv"] = mode.gemv;
                    // up to 8 rows the GEMVs, 9 to 64 the matrix engines' kernel over tiles of 8 rows (2 or 4 a
                    // pass, 32 rows a pass), beyond the GEMM
                    for (const int64_t m : {1, 2, 3, 5, 8, 9, 12, 16, 32, 38, 64, 70}) {
                        CAPTURE(m);
                        const HostTensor x = rng.uniform(DType::F32, {m, k});
                        Graph g;
                        g.output("out", g.matmul(g.input("x", DType::F32, {m, k}), g.weight(w, "w")));
                        const TensorMap out = dev::run(g, w, {{"x", x}}, o);
                        // the GEMV reads the exact values and x in f32; with block=f16 the weight is rounded to the
                        // session's precision first
                        const bool exact = m < f16_x_from(f, mode) && mode.block == "auto";
                        const double scale = w_max * std::sqrt(static_cast<double>(k));
                        const dev::Tolerance tol = exact ? dev::Tolerance{2e-5 * scale, 2e-5}
                                                         : dev::tolerance({2e-5 * scale, 2e-5}, {3e-3 * scale, 3e-3});
                        CHECK(dev::close(str_cat("matmul ", block_format_name(f)), out.at("out"),
                                         test::ref::matmul(to_f64(x), w64), tol));
                    }
                    if (mode.gemv == "auto") {
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
                        const bool exact = mode.block == "auto";
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

    TEST_CASE("expert_matmul on GGUF block format banks against fp64 of the decoded values" *
              doctest::skip(!dev::enabled())) {
        test::TensorRng rng(73);
        // 18 rows an expert: no whole block of rows; expert 0 is read by ids outside the bank too.
        const int64_t experts = 5, n = 18, slots = 2;
        for (const int64_t k : {768, 256, 1152}) {
            for (const BlockFormat f : kFormats) {
                if (k % block_layout(f).weights != 0) continue;
                CAPTURE(block_format_name(f));
                CAPTURE(k);
                test::MemoryWeights w;
                w.add_block("bank", f, {experts, n, k}, blocks(f, {experts, n, k}, 31));
                const Tensor64 w64 = to_f64(w.tensor("bank"));
                const double scale = max_abs(w64) * std::sqrt(static_cast<double>(k));
                for (const Mode& mode : {Mode{"auto", "auto"}, Mode{"f16", "auto"}, Mode{"auto", "vector"}}) {
                    CAPTURE(mode.block);
                    CAPTURE(mode.gemv);
                    SessionOptions o = dev::options();
                    o.extra["block"] = mode.block;
                    o.extra["gemv"] = mode.gemv;
                    // up to 64 pairs: an expert's pairs as the rows of a GEMV on the matrix engines, 1, 2 or 4 at a
                    // time by the tokens (-3: every pair on one expert, more than the rows of a GEMV), or with
                    // gemv=vector a GEMV a pair; 80: grouped by expert; 280: a GEMM per expert
                    for (const int64_t step : {1, 2, 3, -3, 8, 40, 140}) {
                        CAPTURE(step);
                        const int64_t tokens = std::abs(step);
                        std::vector<int32_t> idx;
                        for (int64_t i = 0; i < tokens * slots; ++i)
                            idx.push_back(static_cast<int32_t>(step < 0 ? 2 : (i * 3 + 1 + i / 7) % experts));
                        idx[1] = static_cast<int32_t>(experts + 2);  // outside the bank: expert 0
                        std::vector<int64_t> idx64(idx.begin(), idx.end());
                        idx64[1] = 0;
                        const HostTensor x2 = rng.uniform(DType::F32, {tokens, k});
                        const HostTensor x3 = rng.uniform(DType::F32, {tokens, slots, k});
                        Graph g;
                        const Value i = g.input("idx", DType::I32, {kT, slots});
                        g.output("shared", g.expert_matmul(g.input("x2", DType::F32, {kT, k}), g.weight(w, "bank"), i));
                        g.output("per_slot",
                                 g.expert_matmul(g.input("x3", DType::F32, {kT, slots, k}), g.weight(w, "bank"), i));
                        const TensorMap out =
                            dev::run(g, w, {{"idx", test::make_tensor(DType::I32, {tokens, slots}, idx)}, {"x2", x2},
                                            {"x3", x3}}, o);
                        // the GEMVs read the exact values, and but for the one on the matrix engines x in f32; a
                        // GEMM, or block=f16, the weight rounded to the session's precision
                        const bool f16_x = mode.gemv == "auto" && tokens * slots <= 64 && !pack_kernels(f);
                        const bool exact = mode.block == "auto" && tokens * slots < 256 && !f16_x;
                        const dev::Tolerance tol = exact ? dev::Tolerance{2e-5 * scale, 2e-5}
                                                         : dev::tolerance({2e-5 * scale, 2e-5}, {3e-3 * scale, 3e-3});
                        CHECK(dev::close(str_cat("expert_matmul ", block_format_name(f)), out.at("shared"),
                                         test::ref::expert_matmul(to_f64(x2), w64, idx64, slots), tol));
                        CHECK(dev::close(str_cat("expert_matmul per slot ", block_format_name(f)), out.at("per_slot"),
                                         test::ref::expert_matmul(to_f64(x3), w64, idx64, slots), tol));
                    }
                }
            }
        }
    }
}
