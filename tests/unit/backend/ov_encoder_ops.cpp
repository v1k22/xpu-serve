// Kernel contract for the ops vision encoders add (clamp, transpose, axial rope, non-causal attention) and for
// stateless sessions, on the OpenVINO device under test (XS_OV_TEST_DEVICE).
#include <doctest/doctest.h>

#include <limits>

#include "ov_test_support.h"
#include "reference_ops.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace ovt = xs::test::ov;

namespace {

HostTensor run_out(const Graph& g, const TensorMap& inputs) {
    return ovt::run(g, test::MemoryWeights{}, inputs).at("out");
}

std::vector<std::vector<int64_t>> pairs_of(const HostTensor& positions) {
    std::vector<std::vector<int64_t>> out;
    const std::vector<int64_t> flat = test::to_ints(positions);
    for (size_t i = 0; i < flat.size(); i += 2)
        out.push_back({flat[i], flat[i + 1]});
    return out;
}

}  // namespace

TEST_SUITE("openvino encoder ops") {
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
            CHECK(ovt::close("clamp", out, expected, ovt::tolerance({0, 0}, {0, 4.9e-4})));
        }
    }

    TEST_CASE("transpose with T moving, and a matmul against a transposed activation") {
        test::TensorRng rng(12);
        const HostTensor x = rng.uniform(DType::F32, {5, 3, 4});
        for (const std::vector<int>& perm : {std::vector<int>{1, 0, 2}, {2, 0, 1}, {0, 2, 1}}) {
            Graph g;
            g.output("out", g.transpose(g.input("x", DType::F32, {kT, 3, 4}), perm));
            CHECK(ovt::close("transpose", run_out(g, {{"x", x}}), test::ref::transpose(to_f64(x), perm),
                             ovt::tolerance({0, 0}, {0, 4.9e-4})));
        }
        // The pooling pattern of a vision encoder: out [P, D] = pool [P, T] @ h [T, D], spelled matmul(pool, h^T).
        const HostTensor pool = rng.uniform(DType::F32, {6, 9});
        const HostTensor h = rng.uniform(DType::F32, {9, 16});
        Graph g;
        const Value hv = g.input("h", DType::F32, {kT, 16});
        g.output("out", g.matmul(g.input("pool", DType::F32, {6, kT}), g.transpose(hv, {1, 0})));
        const HostTensor out = run_out(g, {{"pool", pool}, {"h", h}});
        const Tensor64 expected = test::ref::matmul(to_f64(pool), test::ref::transpose(to_f64(h), {1, 0}));
        CHECK(ovt::close("pool_matmul", out, expected, ovt::tolerance({1e-5, 1e-5}, {4e-3, 4e-3})));
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
        const ovt::Tolerance tol = ovt::tolerance({1e-5, 1e-5}, {4e-3, 4e-3});
        CHECK(ovt::close("axial_rope", out, test::ref::axial_rope(to_f64(x), pairs_of(positions), freq64), tol));
        std::vector<std::vector<int64_t>> swapped = pairs_of(positions);
        for (auto& p : swapped)
            std::swap(p[0], p[1]);
        CHECK_FALSE(test::all_close(out, test::ref::axial_rope(to_f64(x), swapped, freq64), tol.atol, tol.rtol));
    }

    TEST_CASE("non-causal attention over the step's own tokens, sdpa and matmul paths, with GQA") {
        test::TensorRng rng(14);
        for (const auto& [q_heads, kv_heads] : {std::pair<int64_t, int64_t>{4, 4}, {6, 2}}) {
            for (const bool matmul : {false, true}) {
                CAPTURE(q_heads);
                CAPTURE(kv_heads);
                CAPTURE(matmul);
                const int64_t dim = 64;
                const HostTensor q = rng.uniform(DType::F32, {37, q_heads, dim});
                const HostTensor k = rng.uniform(DType::F32, {37, kv_heads, dim});
                const HostTensor v = rng.uniform(DType::F32, {37, kv_heads, dim});
                AttentionAttrs attrs;
                attrs.causal = false;
                attrs.scale = 0.125f;
                Graph g;
                g.output("out", g.attention(g.input("q", DType::F32, {kT, q_heads, dim}),
                                            g.input("k", DType::F32, {kT, kv_heads, dim}),
                                            g.input("v", DType::F32, {kT, kv_heads, dim}), attrs));
                SessionOptions options = ovt::options();
                options.extra["attention"] = ovt::attention_impls()[matmul ? 1 : 0];
                const HostTensor out = ovt::run(g, test::MemoryWeights{}, {{"q", q}, {"k", k}, {"v", v}}, options)
                                           .at("out");
                test::ref::AttentionSpec spec;
                spec.scale = 0.125;
                spec.causal = false;
                const ovt::Tolerance tol = ovt::tolerance({1e-5, 1e-5}, {4e-3, 4e-3});
                CHECK(ovt::close("attention_bidirectional", out,
                                 test::ref::attention(to_f64(q), to_f64(k), to_f64(v), spec), tol));
                spec.causal = true;
                CHECK_FALSE(test::all_close(out, test::ref::attention(to_f64(q), to_f64(k), to_f64(v), spec), tol.atol,
                                            tol.rtol));
            }
        }
    }

    TEST_CASE("a graph without KV slots is stateless: runs do not accumulate tokens") {
        Graph g;
        g.output("out", g.scale(g.input("x", DType::F32, {kT, 4}), 2.0f));
        const auto session = ovt::backend().compile(g, test::MemoryWeights{}, ovt::options(8));
        test::TensorRng rng(15);
        for (int run = 0; run < 3; ++run) {  // 3 x 6 tokens would overflow max_context 8 if they accumulated
            TensorMap out;
            session->run({{"x", rng.uniform(DType::F32, {6, 4})}}, out);
            CHECK(session->num_cached() == 0);
        }
    }
}
