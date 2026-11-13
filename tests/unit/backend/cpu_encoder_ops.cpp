// Kernel contract for the ops vision encoders add (clamp, transpose, axial rope, non-causal attention) and for
// stateless sessions, on the cpu-ref backend.
#include <doctest/doctest.h>

#include <limits>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;

namespace {

constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-5;

HostTensor run_out(const Graph& g, const TensorMap& inputs) {
    const auto backend = make_cpu_ref_backend();
    return test::run_once(*backend, g, test::MemoryWeights{}, inputs).at("out");
}

std::vector<std::vector<int64_t>> pairs_of(const HostTensor& positions) {
    std::vector<std::vector<int64_t>> out;
    const std::vector<int64_t> flat = test::to_ints(positions);
    for (size_t i = 0; i < flat.size(); i += 2)
        out.push_back({flat[i], flat[i + 1]});
    return out;
}

}  // namespace

TEST_SUITE("cpu-ref encoder ops") {
    TEST_CASE("clamp with finite and infinite bounds") {
        test::TensorRng rng(11);
        const HostTensor x = rng.uniform(DType::F32, {4, 33}, 10.0f);
        constexpr float kInf = std::numeric_limits<float>::infinity();
        for (const auto& [lo, hi] : {std::pair{-2.5f, 3.0f}, std::pair{-kInf, 1.0f}, std::pair{-kInf, kInf}}) {
            CAPTURE(lo);
            CAPTURE(hi);
            Graph g;
            g.output("out", g.clamp(g.input("x", DType::F32, {kT, 33}), lo, hi));
            const HostTensor out = run_out(g, {{"x", x}});
            const Tensor64 expected =
                test::ref::map(to_f64(x), [lo, hi](double v) { return std::min<double>(std::max<double>(v, lo), hi); });
            CHECK(test::all_close(out, expected, 0, 0));
        }
    }

    TEST_CASE("transpose of 2-D and 3-D tensors, with T moving, for float and int values") {
        test::TensorRng rng(12);
        const HostTensor x = rng.uniform(DType::F32, {5, 3, 4});
        for (const std::vector<int>& perm : {std::vector<int>{1, 0, 2}, {2, 0, 1}, {0, 2, 1}}) {
            Graph g;
            g.output("out", g.transpose(g.input("x", DType::F32, {kT, 3, 4}), perm));
            CHECK(test::all_close(run_out(g, {{"x", x}}), test::ref::transpose(to_f64(x), perm), 0, 0));
        }
        const HostTensor ids = rng.ints(DType::I32, {6, 2}, 0, 100);
        Graph g;
        g.output("out", g.transpose(g.input("ids", DType::I32, {kT, 2}), {1, 0}));
        const HostTensor out = run_out(g, {{"ids", ids}});
        REQUIRE(out.dtype() == DType::I32);
        CHECK(out.shape() == Dims{2, 6});
        CHECK(out.data<int32_t>()[1] == ids.data<int32_t>()[2]);  // out[0][1] = ids[1][0]
    }

    TEST_CASE("axial rope rotates each section by its own positions") {
        test::TensorRng rng(13);
        const int64_t heads = 3, dim = 64;
        const HostTensor x = rng.uniform(DType::F32, {7, heads, dim});
        const HostTensor positions = rng.ints(DType::I32, {7, 2}, 0, 47);
        std::vector<float> inv_freq(dim / 4);
        for (size_t i = 0; i < inv_freq.size(); ++i)
            inv_freq[i] = std::pow(100.0f, -2.0f * static_cast<float>(i) / (dim / 2));
        Graph g;
        const Value freq = g.constant(test::make_tensor(DType::F32, {dim / 4}, inv_freq));
        g.output("out", g.rope(g.input("x", DType::F32, {kT, heads, dim}), g.input("pos", DType::I32, {kT, 2}), freq));
        const HostTensor out = run_out(g, {{"x", x}, {"pos", positions}});
        const std::vector<double> freq64(inv_freq.begin(), inv_freq.end());
        CHECK(test::all_close(out, test::ref::axial_rope(to_f64(x), pairs_of(positions), freq64), kAtol, kRtol));

        SUBCASE("planted bug: swapping the two position columns must not match") {
            std::vector<std::vector<int64_t>> swapped = pairs_of(positions);
            for (auto& p : swapped)
                std::swap(p[0], p[1]);
            CHECK_FALSE(test::all_close(out, test::ref::axial_rope(to_f64(x), swapped, freq64), kAtol, kRtol));
        }
    }

    TEST_CASE("non-causal attention over the step's own tokens sees every key, with GQA") {
        test::TensorRng rng(14);
        for (const auto& [q_heads, kv_heads] : {std::pair<int64_t, int64_t>{4, 4}, {6, 2}}) {
            CAPTURE(q_heads);
            CAPTURE(kv_heads);
            const int64_t dim = 16;
            const HostTensor q = rng.uniform(DType::F32, {9, q_heads, dim});
            const HostTensor k = rng.uniform(DType::F32, {9, kv_heads, dim});
            const HostTensor v = rng.uniform(DType::F32, {9, kv_heads, dim});
            AttentionAttrs attrs;
            attrs.causal = false;
            attrs.scale = 0.7f;
            Graph g;
            g.output("out", g.attention(g.input("q", DType::F32, {kT, q_heads, dim}),
                                        g.input("k", DType::F32, {kT, kv_heads, dim}),
                                        g.input("v", DType::F32, {kT, kv_heads, dim}), attrs));
            const HostTensor out = run_out(g, {{"q", q}, {"k", k}, {"v", v}});
            test::ref::AttentionSpec spec;
            spec.scale = 0.7;
            spec.causal = false;
            CHECK(test::all_close(out, test::ref::attention(to_f64(q), to_f64(k), to_f64(v), spec), kAtol, kRtol));
            spec.causal = true;  // planted bug: a causal reference must not match
            CHECK_FALSE(
                test::all_close(out, test::ref::attention(to_f64(q), to_f64(k), to_f64(v), spec), kAtol, kRtol));
        }
    }

    TEST_CASE("non-causal attention with a window over the step's own tokens sees as far after a query as before") {
        test::TensorRng rng(16);
        for (const auto& [q_heads, kv_heads] : {std::pair<int64_t, int64_t>{4, 4}, {6, 2}}) {
            for (const int64_t window : {1, 4, 30}) {  // a window wider than the step sees every key
                CAPTURE(q_heads);
                CAPTURE(window);
                const int64_t dim = 16, tokens = 23;
                const HostTensor q = rng.uniform(DType::F32, {tokens, q_heads, dim}, 2.0f);
                const HostTensor k = rng.uniform(DType::F32, {tokens, kv_heads, dim}, 2.0f);
                const HostTensor v = rng.uniform(DType::F32, {tokens, kv_heads, dim});
                AttentionAttrs attrs;
                attrs.causal = false;
                attrs.scale = 0.3f;
                attrs.sliding_window = window;
                Graph g;
                g.output("out", g.attention(g.input("q", DType::F32, {kT, q_heads, dim}),
                                            g.input("k", DType::F32, {kT, kv_heads, dim}),
                                            g.input("v", DType::F32, {kT, kv_heads, dim}), attrs));
                const HostTensor out = run_out(g, {{"q", q}, {"k", k}, {"v", v}});
                test::ref::AttentionSpec spec;
                spec.scale = 0.3;
                spec.sliding_window = window;
                spec.causal = false;
                const auto matches = [&](const test::ref::AttentionSpec& s) {
                    return test::all_close(out, test::ref::attention(to_f64(q), to_f64(k), to_f64(v), s), kAtol,
                                           kRtol);
                };
                CHECK(matches(spec));
                if (window >= tokens) continue;
                test::ref::AttentionSpec planted = spec;  // planted bugs: a window one key wider, or only behind
                planted.window_slack = 1;
                CHECK_FALSE(matches(planted));
                planted = spec;
                planted.one_sided = true;
                CHECK_FALSE(matches(planted));
                planted = spec;
                planted.causal = true;
                if (window > 1) CHECK_FALSE(matches(planted));  // a window of one key is the query's own either way
            }
        }
    }

    TEST_CASE("a graph without KV slots is stateless: runs do not accumulate tokens") {
        Graph g;
        g.output("out", g.scale(g.input("x", DType::F32, {kT, 4}), 2.0f));
        const auto backend = make_cpu_ref_backend();
        SessionOptions options;
        options.max_context = 8;
        const auto session = backend->compile(g, test::MemoryWeights{}, options);
        test::TensorRng rng(15);
        for (int run = 0; run < 3; ++run) {  // 3 x 6 tokens would overflow max_context 8 if they accumulated
            TensorMap out;
            session->run({{"x", rng.uniform(DType::F32, {6, 4})}}, out);
            CHECK(session->num_cached() == 0);
        }
    }

    TEST_CASE("graph refuses a window over static non-causal queries and malformed axial positions") {
        Graph g;
        const Value q = g.input("q", DType::F32, {kT, 2, 8});
        AttentionAttrs attrs;
        attrs.causal = false;
        attrs.sliding_window = 4;
        const Value fixed = g.input("fixed", DType::F32, {6, 2, 8});
        CHECK_THROWS_AS(g.attention(fixed, fixed, fixed, attrs), xs::Error);
        const Value freq = g.constant(HostTensor(DType::F32, {2}));
        CHECK_THROWS_AS(g.rope(q, g.input("pos3", DType::I32, {kT, 3}), freq), xs::Error);  // 8 % 6 != 0
        CHECK_THROWS_AS(g.rope(q, g.input("pos2", DType::I32, {kT, 2}), g.constant(HostTensor(DType::F32, {4}))),
                        xs::Error);  // 2 sections of 8 channels need 2 frequencies
        CHECK_THROWS_AS(g.clamp(q, 1.0f, -1.0f), xs::Error);
        CHECK_THROWS_AS(g.transpose(q, {0, 0, 1}), xs::Error);
    }
}
