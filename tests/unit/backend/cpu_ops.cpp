// Kernel contract for the cpu-ref backend: every op against an fp64 reference (backend/reference_ops.h).
#include <doctest/doctest.h>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;

namespace {

// f32 storage of fp64-exact inputs; the remaining error is f32 rounding of the result (and of intermediates).
constexpr double kAtol = 1e-5;
constexpr double kRtol = 1e-5;

HostTensor run_single_output(const Graph& g, const WeightStore& weights, const TensorMap& inputs) {
    const auto backend = make_cpu_ref_backend();
    return test::run_once(*backend, g, weights, inputs).at("out");
}

std::vector<double> to_doubles(const HostTensor& t) {
    return to_f64(t).data;
}

}  // namespace

TEST_SUITE("cpu-ref ops") {
    TEST_CASE("matmul against fp64 for 2-D and 3-D x and every weight dtype") {
        test::TensorRng rng(1);
        struct Case {
            Dims x_shape;
            int64_t n;
            DType w_dtype;
        };
        for (const Case& c : {Case{{1, 7}, 5, DType::F32}, Case{{3, 64}, 33, DType::BF16},
                              Case{{2, 3, 40}, 17, DType::F16}, Case{{5, 1536}, 96, DType::BF16}}) {
            CAPTURE(dims_to_string(c.x_shape));
            test::MemoryWeights w;
            w.add("w", rng.uniform(c.w_dtype, {c.n, c.x_shape.back()}));
            const HostTensor x = rng.uniform(DType::F32, c.x_shape);
            Graph g;
            g.output("out", g.matmul(g.input("x", DType::F32, c.x_shape), g.weight(w, "w")));
            const HostTensor out = run_single_output(g, w, {{"x", x}});
            const Tensor64 expected = test::ref::matmul(to_f64(x), to_f64(w.tensor("w")));
            CHECK(test::all_close(out, expected, kAtol, kRtol));
        }
    }

    TEST_CASE("add and mul broadcast like numpy") {
        test::TensorRng rng(2);
        struct Case {
            Dims a, b;
        };
        for (const Case& c : {Case{{4, 6}, {4, 6}}, Case{{4, 6}, {6}}, Case{{4, 1, 6}, {3, 1}}, Case{{1}, {2, 3}},
                              Case{{2, 1, 5}, {1, 4, 1}}}) {
            CAPTURE(dims_to_string(c.a));
            CAPTURE(dims_to_string(c.b));
            const HostTensor a = rng.uniform(DType::F32, c.a);
            const HostTensor b = rng.uniform(DType::F32, c.b);
            for (const OpKind kind : {OpKind::Add, OpKind::Mul}) {
                Graph g;
                const Value va = g.input("a", DType::F32, c.a);
                const Value vb = g.input("b", DType::F32, c.b);
                g.output("out", kind == OpKind::Add ? g.add(va, vb) : g.mul(va, vb));
                const HostTensor out = run_single_output(g, test::MemoryWeights{}, {{"a", a}, {"b", b}});
                const Tensor64 expected = test::ref::broadcast(
                    to_f64(a), to_f64(b), [kind](double x, double y) { return kind == OpKind::Add ? x + y : x * y; });
                CHECK(test::all_close(out, expected, kAtol, kRtol));
            }
        }
    }

    TEST_CASE("scale, tanh, gelu_tanh and softcap") {
        test::TensorRng rng(3);
        const HostTensor x = rng.uniform(DType::F32, {3, 50}, 8.0f);
        const Tensor64 x64 = to_f64(x);
        struct Case {
            const char* name;
            Value (*build)(Graph&, Value);
            double (*expect)(double);
        };
        const Case cases[] = {
            {"scale", [](Graph& g, Value v) { return g.scale(v, 0.125f); }, [](double v) { return v * 0.125; }},
            {"tanh", [](Graph& g, Value v) { return g.tanh(v); }, [](double v) { return std::tanh(v); }},
            {"gelu_tanh", [](Graph& g, Value v) { return g.gelu_tanh(v); }, test::ref::gelu_tanh},
            {"softcap", [](Graph& g, Value v) { return g.softcap(v, 3.0f); },
             [](double v) { return 3.0 * std::tanh(v / 3.0); }},
        };
        for (const Case& c : cases) {
            CAPTURE(c.name);
            Graph g;
            g.output("out", c.build(g, g.input("x", DType::F32, {3, 50})));
            const HostTensor out = run_single_output(g, test::MemoryWeights{}, {{"x", x}});
            CHECK(test::all_close(out, test::ref::map(x64, c.expect), kAtol, kRtol));
        }
    }

    TEST_CASE("gelu_tanh matches PyTorch values") {
        // torch.nn.functional.gelu(torch.tensor([-3., -1., -0.5, 0., 0.5, 1., 3.], dtype=torch.float64),
        // approximate="tanh")
        const std::vector<double> expected{-0.0036373920817729943, -0.15880800939172324, -0.15428599017206468, 0.0,
                                           0.34571400982793533,    0.8411919906082768,   2.996362607918227};
        Graph g;
        g.output("out", g.gelu_tanh(g.input("x", DType::F32, {7})));
        const HostTensor x = test::make_tensor<float>(DType::F32, {7}, {-3.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 3.f});
        const HostTensor out = run_single_output(g, test::MemoryWeights{}, {{"x", x}});
        CHECK(test::all_close(out, Tensor64{{7}, expected}, 1e-6, 1e-6));
    }

    TEST_CASE("rms_norm with and without weight, no +1 offset") {
        test::TensorRng rng(4);
        test::MemoryWeights w;
        w.add("norm", rng.uniform(DType::BF16, {48}, 2.0f));
        const HostTensor x = rng.uniform(DType::F32, {2, 3, 48}, 4.0f);
        for (const bool with_weight : {false, true}) {
            CAPTURE(with_weight);
            Graph g;
            const Value in = g.input("x", DType::F32, {2, 3, 48});
            g.output("out", with_weight ? g.rms_norm(in, g.weight(w, "norm"), 1e-6f) : g.rms_norm(in, 1e-6f));
            const HostTensor out = run_single_output(g, w, {{"x", x}});
            const Tensor64 weight = to_f64(w.tensor("norm"));
            const Tensor64 expected = test::ref::rms_norm(to_f64(x), with_weight ? &weight : nullptr, 1e-6);
            CHECK(test::all_close(out, expected, kAtol, kRtol));
        }
    }

    TEST_CASE("gather picks table rows by i32 and i64 ids") {
        test::TensorRng rng(5);
        test::MemoryWeights w;
        w.add("embed", rng.uniform(DType::BF16, {11, 6}));
        for (const DType id_type : {DType::I32, DType::I64}) {
            const HostTensor ids = rng.ints(id_type, {5}, 0, 10);
            Graph g;
            g.output("out", g.gather(g.weight(w, "embed"), g.input("ids", id_type, {kT})));
            const HostTensor out = run_single_output(g, w, {{"ids", ids}});
            CHECK(test::all_close(out, test::ref::gather(to_f64(w.tensor("embed")), test::to_ints(ids)), 0, 0));
        }
    }

    TEST_CASE("gather refuses an id outside the table") {
        test::MemoryWeights w;
        w.add("embed", HostTensor(DType::F32, {4, 2}));
        Graph g;
        g.output("out", g.gather(g.weight(w, "embed"), g.input("ids", DType::I32, {kT})));
        const auto backend = make_cpu_ref_backend();
        CHECK_THROWS_WITH_AS(test::run_once(*backend, g, w, {{"ids", test::arange_i32(3, 5)}}),
                             doctest::Contains("outside the table"), xs::Error);
    }

    TEST_CASE("slice with static ranges and the symbolic last-n form") {
        test::TensorRng rng(6);
        const HostTensor x = rng.uniform(DType::F32, {5, 4, 6});
        struct Case {
            int axis;
            int64_t begin, end, b, e;
        };
        for (const Case& c :
             {Case{1, 1, 3, 1, 3}, Case{2, -2, kSliceEnd, 4, 6}, Case{2, 0, -1, 0, 5}, Case{0, -3, kSliceEnd, 2, 5}}) {
            CAPTURE(c.axis);
            CAPTURE(c.begin);
            Graph g;
            g.output("out", g.slice(g.input("x", DType::F32, {kT, 4, 6}), c.axis, c.begin, c.end));
            const HostTensor out = run_single_output(g, test::MemoryWeights{}, {{"x", x}});
            CHECK(test::all_close(out, test::ref::slice(to_f64(x), static_cast<size_t>(c.axis), c.b, c.e), 0, 0));
        }
    }

    TEST_CASE("reshape keeps row-major order and resolves T; slice keeps int dtype") {
        Graph g;
        const Value ids = g.input("ids", DType::I32, {kT});
        g.output("out", g.reshape(g.input("x", DType::F32, {kT, 6}), {kT, 2, 3}));
        g.output("last_id", g.slice(ids, 0, -1));
        test::TensorRng rng(7);
        const HostTensor x = rng.uniform(DType::F32, {4, 6});
        const auto backend = make_cpu_ref_backend();
        const TensorMap out =
            test::run_once(*backend, g, test::MemoryWeights{}, {{"x", x}, {"ids", test::arange_i32(0, 4)}});
        CHECK(out.at("out").shape() == Dims{4, 2, 3});
        CHECK(test::all_close(out.at("out"), Tensor64{{4, 2, 3}, to_doubles(x)}, 0, 0));
        REQUIRE(out.at("last_id").dtype() == DType::I32);
        CHECK(out.at("last_id").data<int32_t>()[0] == 3);
    }

    TEST_CASE("rope rotate-half with zero inv_freq entries and nonzero positions") {
        test::TensorRng rng(8);
        const int64_t heads = 3, dim = 16;
        const HostTensor x = rng.uniform(DType::F32, {4, heads, dim});
        // Gemma's proportional RoPE: the tail pairs get inv_freq 0 and must come out unrotated.
        std::vector<float> inv_freq(dim / 2);
        for (size_t i = 0; i < inv_freq.size(); ++i)
            inv_freq[i] = i < 5 ? std::pow(10000.0f, -2.0f * i / dim) : 0.0f;
        const HostTensor positions = test::make_tensor<int64_t>(DType::I64, {4}, {7, 100, 4095, 12});
        Graph g;
        const Value freq = g.constant(test::make_tensor(DType::F32, {dim / 2}, inv_freq));
        g.output("out", g.rope(g.input("x", DType::F32, {kT, heads, dim}), g.input("pos", DType::I64, {kT}), freq));
        const HostTensor out = run_single_output(g, test::MemoryWeights{}, {{"x", x}, {"pos", positions}});
        const std::vector<double> freq64(inv_freq.begin(), inv_freq.end());
        const Tensor64 expected = test::ref::rope(to_f64(x), test::to_ints(positions), freq64);
        CHECK(test::all_close(out, expected, kAtol, kRtol));
        CHECK(out.data<float>()[dim - 1] == x.data<float>()[dim - 1]);  // inv_freq 0: identity

        SUBCASE("planted bug: an interleaved-layout reference must not match") {
            const Tensor64 wrong =
                test::ref::rope(to_f64(x), test::to_ints(positions), freq64, test::ref::RopeLayout::Interleaved);
            CHECK_FALSE(test::all_close(out, wrong, kAtol, kRtol));
        }
    }

    TEST_CASE("planted bug: a matmul reference against the transposed weight must not match") {
        test::TensorRng rng(9);
        test::MemoryWeights w;
        w.add("w", rng.uniform(DType::F32, {8, 8}));
        const HostTensor x = rng.uniform(DType::F32, {2, 8});
        Graph g;
        g.output("out", g.matmul(g.input("x", DType::F32, {2, 8}), g.weight(w, "w")));
        const HostTensor out = run_single_output(g, w, {{"x", x}});
        Tensor64 transposed = to_f64(w.tensor("w"));
        for (int64_t r = 0; r < 8; ++r) {
            for (int64_t c = 0; c < 8; ++c)
                transposed[r * 8 + c] = to_f64(w.tensor("w"))[c * 8 + r];
        }
        CHECK(test::all_close(out, test::ref::matmul(to_f64(x), to_f64(w.tensor("w"))), kAtol, kRtol));
        CHECK_FALSE(test::all_close(out, test::ref::matmul(to_f64(x), transposed), kAtol, kRtol));
    }
}
