// The GPU backend with weights in GGUF block formats: MatMul at every row count the GEMV's variants take and past
// them (the weight written out for a GEMM), a residual folded into that GEMM, matmuls of weights in different formats
// that read one x, Gather, and ExpertMatMul over a bank
// (few pairs, pairs grouped by expert, many pairs in passes of an expert's rows or grouped GEMMs), against fp64 over
// the values the host decodes
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

// How the weights are uploaded (block=auto: as stored, f16: decoded), which GEMV reads up to 8 rows, and up to how
// many rows the formats a pack's kernels read take the many-row kernel past the few-row one ("": the device's choice).
struct Mode {
    std::string block;
    std::string gemv;
    std::string tiled_rows = "";
};

// From how many rows a matmul reads x in f16 (the matrix-engine GEMVs: a pack's q8 from three rows, its q4 and the
// block formats' from two, all from one with gemv=xmx; and past 8 rows, the GEMM).
int64_t f16_x_from(BlockFormat f, const Mode& mode) {
    if (mode.gemv == "vector") return 9;
    if (mode.gemv == "xmx") return 1;
    return f == BlockFormat::Q8_0 ? 3 : 2;
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
                                         Mode{"auto", "xmx"}, Mode{"auto", "auto", "1024"}}) {
                    if (!mode.tiled_rows.empty() && !pack_kernels(f)) continue;
                    CAPTURE(mode.block);
                    CAPTURE(mode.gemv);
                    CAPTURE(mode.tiled_rows);
                    SessionOptions o = dev::options();
                    o.extra["block"] = mode.block;
                    o.extra["gemv"] = mode.gemv;
                    if (!mode.tiled_rows.empty()) o.extra["tiled_rows"] = mode.tiled_rows;
                    // up to 8 rows the GEMVs, 9 to 64 the matrix engines' kernel over tiles of 8 rows (2 or 4 a
                    // pass, 32 rows a pass), beyond the GEMM, or with tiled_rows the many-row kernel (work-groups
                    // of 512 rows: one in part, one whole and more)
                    for (const int64_t m : {1, 2, 3, 5, 8, 9, 12, 16, 32, 38, 64, 70, 600}) {
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
                    if (mode.gemv == "auto" && mode.tiled_rows.empty()) {
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

    TEST_CASE("matmuls of Q4_0, Q4_1 and Q8_0 weights that read the same x: each launch one way of reading codes" *
              doctest::skip(!dev::enabled())) {
        test::TensorRng rng(75);
        // Q4_0 has no biases and Q4_1 has them: a launch that took one for the other would read past a weight or
        // drop its biases. a and b (Q4_0) share a launch, the others run alone; `next` reads each output in
        // f16 when the kernels past the GEMV write it so.
        const int64_t k = 512, n = 64;
        using Spec = std::pair<const char*, BlockFormat>;
        const std::vector<Spec> specs{{"a", BlockFormat::Q4_0}, {"c", BlockFormat::Q4_1}, {"b", BlockFormat::Q4_0},
                                      {"e", BlockFormat::Q8_0}, {"d", BlockFormat::Q4_1}};
        test::MemoryWeights w;
        uint32_t seed = 51;
        double w_max = 0;
        for (const auto& [name, f] : specs) {
            w.add_block(name, f, {n, k}, blocks(f, {n, k}, seed++));
            w_max = std::max(w_max, max_abs(to_f64(w.tensor(name))));
        }
        w.add("next", rng.uniform(DType::F32, {16, n}, 1.0f / 8));
        const double scale = w_max * std::sqrt(static_cast<double>(k));
        for (const Mode& mode : {Mode{"auto", "auto"}, Mode{"auto", "vector"}, Mode{"auto", "xmx"},
                                 Mode{"auto", "auto", "1024"}}) {
            CAPTURE(mode.gemv);
            CAPTURE(mode.tiled_rows);
            SessionOptions o = dev::options();
            o.extra["gemv"] = mode.gemv;
            if (!mode.tiled_rows.empty()) o.extra["tiled_rows"] = mode.tiled_rows;
            for (const int64_t m : {1, 2, 5, 8, 12, 40, 64, 200}) {
                CAPTURE(m);
                const HostTensor xv = rng.uniform(DType::F32, {m, k});
                for (const bool chained : {false, true}) {
                    CAPTURE(chained);
                    Graph g;
                    const Value x = g.input("x", DType::F32, {m, k});
                    for (const auto& [name, f] : specs) {
                        const Value y = g.matmul(x, g.weight(w, name));
                        g.output(name, chained ? g.matmul(y, g.weight(w, "next")) : y);
                    }
                    const TensorMap out = dev::run(g, w, {{"x", xv}}, o);
                    for (const auto& [name, f] : specs) {
                        CAPTURE(name);
                        Tensor64 want = test::ref::matmul(to_f64(xv), to_f64(w.tensor(name)));
                        if (chained) want = test::ref::matmul(want, to_f64(w.tensor("next")));
                        const double bound = chained ? scale * std::sqrt(static_cast<double>(n)) : scale;
                        CHECK(dev::close(str_cat("matmul beside other formats ", block_format_name(f)), out.at(name),
                                         want, dev::tolerance({2e-5 * bound, 2e-5}, {3e-3 * bound, 3e-3})));
                    }
                }
            }
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
                    // gemv=vector a GEMV a pair; 80 and 280: an expert's pairs in passes on the matrix engines, or
                    // with gemv=vector (and for the formats a pack's kernels read) grouped by expert and a GEMM per
                    // expert
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
                        const bool f16_x = mode.gemv == "auto" && !pack_kernels(f);
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
    TEST_CASE("expert_matmul over many pairs of GGUF block format banks: passes of every size" *
              doctest::skip(!dev::enabled())) {
        test::TensorRng rng(74);
        // An expert's pairs run 64 rows a pass and the rest in a pass of 8, 16, 32 or 64: 69 pairs are a whole pass
        // and one of 8, 12 one of 16, 31 one of 32, 200 three whole passes and one of 8, 50 one of 64; an expert
        // without pairs has none. 21 rows an expert: no whole group of 16.
        const std::vector<int64_t> counts{69, 12, 31, 0, 200, 50};
        const auto experts = static_cast<int64_t>(counts.size());
        const int64_t n = 21, slots = 2;
        std::vector<int32_t> idx;
        for (int64_t e = 0; e < experts; ++e)
            idx.insert(idx.end(), static_cast<size_t>(counts[static_cast<size_t>(e)]), static_cast<int32_t>(e));
        for (size_t i = idx.size() - 1, state = 12345; i > 0; --i) {  // any fixed order that mixes the experts
            state = state * 1103515245u + 12345u;
            std::swap(idx[i], idx[(state >> 8) % (i + 1)]);
        }
        const std::vector<int64_t> idx64(idx.begin(), idx.end());
        const auto tokens = static_cast<int64_t>(idx.size()) / slots;
        // 1,152 columns: whole blocks of 256 and four units after them
        for (const int64_t k : {768, 256, 1152}) {
            for (const BlockFormat f : kFormats) {
                if (k % block_layout(f).weights != 0) continue;
                CAPTURE(block_format_name(f));
                CAPTURE(k);
                test::MemoryWeights w;
                w.add_block("bank", f, {experts, n, k}, blocks(f, {experts, n, k}, 41));
                const Tensor64 w64 = to_f64(w.tensor("bank"));
                const double scale = max_abs(w64) * std::sqrt(static_cast<double>(k));
                const HostTensor x2 = rng.uniform(DType::F32, {tokens, k});
                const HostTensor x3 = rng.uniform(DType::F32, {tokens, slots, k});
                Graph g;
                const Value i = g.input("idx", DType::I32, {kT, slots});
                g.output("shared", g.expert_matmul(g.input("x2", DType::F32, {kT, k}), g.weight(w, "bank"), i));
                g.output("per_slot",
                         g.expert_matmul(g.input("x3", DType::F32, {kT, slots, k}), g.weight(w, "bank"), i));
                const TensorMap out = dev::run(
                    g, w, {{"idx", test::make_tensor(DType::I32, {tokens, slots}, idx)}, {"x2", x2}, {"x3", x3}});
                const dev::Tolerance tol = dev::tolerance({2e-5 * scale, 2e-5}, {3e-3 * scale, 3e-3});
                CHECK(dev::close(str_cat("expert_matmul passes ", block_format_name(f)), out.at("shared"),
                                 test::ref::expert_matmul(to_f64(x2), w64, idx64, slots), tol));
                CHECK(dev::close(str_cat("expert_matmul passes per slot ", block_format_name(f)), out.at("per_slot"),
                                 test::ref::expert_matmul(to_f64(x3), w64, idx64, slots), tol));
            }
        }
    }
}
