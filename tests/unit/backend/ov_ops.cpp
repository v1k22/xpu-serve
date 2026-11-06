// Kernel contract for the OpenVINO backend: every op against an fp64 reference (reference_ops.h), on the device in
// XS_OV_TEST_DEVICE at its precision.
#include <doctest/doctest.h>

#include "ov_test_support.h"
#include "reference_ops.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace ovt = xs::test::ov;

namespace {

// Elementwise results of O(1) inputs: f32 rounding vs f16 rounding of inputs and output.
const ovt::Tolerance kElementwise = ovt::tolerance({1e-5, 1e-5}, {2e-3, 2e-3});

// Data movement only: exact in f32; f16 activations round each value once (half an f16 ulp).
const ovt::Tolerance kStorage = ovt::tolerance({0, 0}, {0, 4.9e-4});

HostTensor run_out(const Graph& g, const WeightStore& w, const TensorMap& inputs) {
    return ovt::run(g, w, inputs).at("out");
}

}  // namespace

TEST_SUITE("openvino ops") {
    TEST_CASE("matmul against fp64 for every weight dtype, static and dynamic T") {
        test::TensorRng rng(1);
        struct Case {
            Dims x_shape;
            bool dynamic_t;
            int64_t n;
            DType w_dtype;
        };
        for (const Case& c : {Case{{1, 7}, true, 5, DType::F32}, Case{{3, 64}, true, 33, DType::BF16},
                              Case{{2, 3, 40}, false, 17, DType::F16}, Case{{5, 1536}, true, 96, DType::BF16}}) {
            CAPTURE(dims_to_string(c.x_shape));
            CAPTURE(dtype_name(c.w_dtype));
            const int64_t k = c.x_shape.back();
            test::MemoryWeights w;
            // 1/sqrt(K) keeps outputs O(1) so one tolerance fits every K.
            w.add("w", rng.uniform(c.w_dtype, {c.n, k}, 1.0f / std::sqrt(static_cast<float>(k))));
            const HostTensor x = rng.uniform(DType::F32, c.x_shape);
            Dims x_dims = c.x_shape;
            if (c.dynamic_t) x_dims[0] = kT;
            Graph g;
            g.output("out", g.matmul(g.input("x", DType::F32, x_dims), g.weight(w, "w")));
            const HostTensor out = run_out(g, w, {{"x", x}});
            const Tensor64 expected = test::ref::matmul(to_f64(x), to_f64(w.tensor("w")));
            CHECK(ovt::close("matmul", out, expected, ovt::tolerance({1e-5, 1e-4}, {4e-3, 4e-3})));
        }
    }

    TEST_CASE("add and mul broadcast like numpy, including a dynamic T operand") {
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
            Dims a_dims = c.a;
            if (c.a.size() > c.b.size()) a_dims[0] = kT;  // a leading dim b does not have may be dynamic
            for (const OpKind kind : {OpKind::Add, OpKind::Mul}) {
                Graph g;
                const Value va = g.input("a", DType::F32, a_dims);
                const Value vb = g.input("b", DType::F32, c.b);
                g.output("out", kind == OpKind::Add ? g.add(va, vb) : g.mul(va, vb));
                const HostTensor out = run_out(g, test::MemoryWeights{}, {{"a", a}, {"b", b}});
                const Tensor64 expected = test::ref::broadcast(
                    to_f64(a), to_f64(b), [kind](double x, double y) { return kind == OpKind::Add ? x + y : x * y; });
                CHECK(ovt::close(kind == OpKind::Add ? "add" : "mul", out, expected, kElementwise));
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
            double (*ref)(double);
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
            g.output("out", c.build(g, g.input("x", DType::F32, {kT, 50})));
            const HostTensor out = run_out(g, test::MemoryWeights{}, {{"x", x}});
            CHECK(ovt::close(c.name, out, test::ref::map(x64, c.ref), ovt::tolerance({1e-5, 1e-5}, {1e-2, 2e-3})));
        }
    }

    TEST_CASE("rms_norm with and without weight, no +1 offset, and with activations far beyond the f16 range") {
        test::TensorRng rng(4);
        test::MemoryWeights w;
        w.add("norm", rng.uniform(DType::BF16, {1536}, 2.0f));
        for (const float magnitude : {1.0f, 3000.0f}) {
            CAPTURE(magnitude);
            // At 3000, x^2 reaches 9e6: an f16 sum of squares would overflow, so this passes only with f32 inside.
            const HostTensor x = rng.uniform(DType::F32, {4, 1536}, magnitude);
            for (const bool with_weight : {false, true}) {
                CAPTURE(with_weight);
                Graph g;
                const Value in = g.input("x", DType::F32, {kT, 1536});
                g.output("out", with_weight ? g.rms_norm(in, g.weight(w, "norm"), 1e-6f) : g.rms_norm(in, 1e-6f));
                const HostTensor out = run_out(g, w, {{"x", x}});
                const Tensor64 weight = to_f64(w.tensor("norm"));
                const Tensor64 expected = test::ref::rms_norm(to_f64(x), with_weight ? &weight : nullptr, 1e-6);
                CHECK(ovt::close(magnitude > 1 ? "rms_norm_large" : "rms_norm", out, expected,
                                 ovt::tolerance({1e-5, 1e-5}, {4e-3, 4e-3})));
            }
        }
    }

    TEST_CASE("gather picks rows of a bf16 table by i32 and i64 ids") {
        test::TensorRng rng(5);
        test::MemoryWeights w;
        w.add("table", rng.uniform(DType::BF16, {100, 24}));
        for (const DType id_type : {DType::I32, DType::I64}) {
            const HostTensor ids = rng.ints(id_type, {6}, 0, 99);
            Graph g;
            g.output("out", g.gather(g.weight(w, "table"), g.input("ids", id_type, {kT})));
            const HostTensor out = run_out(g, w, {{"ids", ids}});
            // bf16 values are exact in f16 and f32, so a gather must be exact.
            CHECK(ovt::close("gather", out, test::ref::gather(to_f64(w.tensor("table")), test::to_ints(ids)), {0, 0}));
        }
    }

    TEST_CASE("slice with static ranges and the symbolic last-n form; reshape resolves T") {
        test::TensorRng rng(6);
        const HostTensor x = rng.uniform(DType::F32, {5, 4, 8});
        const Tensor64 x64 = to_f64(x);
        struct Case {
            int axis;
            int64_t begin, end;
            int64_t ref_begin, ref_end;
        };
        for (const Case& c :
             {Case{2, 2, 6, 2, 6}, Case{1, -3, kSliceEnd, 1, 4}, Case{0, -2, kSliceEnd, 3, 5}, Case{2, 0, -1, 0, 7}}) {
            CAPTURE(c.axis);
            CAPTURE(c.begin);
            Graph g;
            g.output("out", g.slice(g.input("x", DType::F32, {kT, 4, 8}), c.axis, c.begin, c.end));
            const HostTensor out = run_out(g, test::MemoryWeights{}, {{"x", x}});
            const Tensor64 expected = test::ref::slice(x64, static_cast<size_t>(c.axis), c.ref_begin, c.ref_end);
            CHECK(ovt::close("slice", out, expected, kStorage));
        }
        Graph g;
        const Value flat = g.reshape(g.input("x", DType::F32, {kT, 4, 8}), {kT, 32});
        g.output("out", g.reshape(g.scale(flat, 2.0f), {2, kT, 16}));
        const HostTensor out = run_out(g, test::MemoryWeights{}, {{"x", x}});
        Tensor64 expected = test::ref::map(x64, [](double v) { return 2 * v; });
        expected.shape = {2, 5, 16};
        CHECK(ovt::close("reshape", out, expected, kStorage));
    }

    TEST_CASE("rope rotate-half with zero inv_freq entries, at small and large positions") {
        test::TensorRng rng(7);
        const int64_t heads = 3;
        const int64_t dim = 256;
        std::vector<float> inv_freq(dim / 2);
        for (size_t i = 0; i < inv_freq.size(); ++i) {
            // Gemma-style partial rotary: the upper half of the frequencies is zero (those dims pass through).
            inv_freq[i] = i < inv_freq.size() / 2 ? std::pow(10000.0f, -2.0f * static_cast<float>(i) / dim) : 0.0f;
        }
        for (const int32_t first_pos : {0, 30000}) {
            CAPTURE(first_pos);
            const HostTensor x = rng.uniform(DType::F32, {5, heads, dim});
            const HostTensor pos = test::arange_i32(first_pos, first_pos + 5);
            Graph g;
            const Value freq = g.constant(HostTensor::from_vector(DType::F32, {dim / 2}, inv_freq));
            g.output("out", g.rope(g.input("x", DType::F32, {kT, heads, dim}), g.input("pos", DType::I32, {kT}), freq));
            const HostTensor out = run_out(g, test::MemoryWeights{}, {{"x", x}, {"pos", pos}});
            const std::vector<double> freq64(inv_freq.begin(), inv_freq.end());
            const Tensor64 expected = test::ref::rope(to_f64(x), test::to_ints(pos), freq64);
            // At position 30000 an f32 angle is off by up to ~2e-3 rad (one f32 ulp of the product), as in HF.
            const ovt::Tolerance tol =
                first_pos > 0 ? ovt::tolerance({5e-3, 0}, {6e-3, 4e-3}) : ovt::tolerance({1e-5, 1e-5}, {4e-3, 4e-3});
            CHECK(ovt::close(first_pos > 0 ? "rope_pos30000" : "rope", out, expected, tol));
            const Tensor64 interleaved =
                test::ref::rope(to_f64(x), test::to_ints(pos), freq64, test::ref::RopeLayout::Interleaved);
            CHECK_FALSE(test::all_close(out, interleaved, 4e-3, 4e-3));
        }
    }

    TEST_CASE("kv_cache outputs read back in the logical [L, Hkv, D] layout across steps") {
        test::TensorRng rng(8);
        Graph g;
        const int slot = g.add_kv_slot(2, 8, 0, "layer0");
        const auto [k_all, v_all] =
            g.kv_cache(slot, g.input("k", DType::F32, {kT, 2, 8}), g.input("v", DType::F32, {kT, 2, 8}));
        g.output("k_all", k_all);
        g.output("v_all", g.scale(v_all, 1.0f));
        const auto session = ovt::backend().compile(g, test::MemoryWeights{}, ovt::options());
        std::vector<HostTensor> ks, vs;
        for (const int64_t t : {3, 1, 2}) {
            ks.push_back(rng.uniform(DType::F32, {t, 2, 8}));
            vs.push_back(rng.uniform(DType::F32, {t, 2, 8}));
            TensorMap out;
            session->run({{"k", ks.back()}, {"v", vs.back()}}, out);
            Tensor64 k_expected, v_expected;
            for (size_t s = 0; s < ks.size(); ++s) {
                const Tensor64 k64 = to_f64(ks[s]);
                const Tensor64 v64 = to_f64(vs[s]);
                k_expected.data.insert(k_expected.data.end(), k64.data.begin(), k64.data.end());
                v_expected.data.insert(v_expected.data.end(), v64.data.begin(), v64.data.end());
            }
            k_expected.shape = v_expected.shape = {session->num_cached(), 2, 8};
            CHECK(ovt::close("kv_cache", out.at("k_all"), k_expected, kElementwise));
            CHECK(ovt::close("kv_cache", out.at("v_all"), v_expected, kElementwise));
        }
    }

    TEST_CASE("planted bug: a matmul reference against the transposed weight must not match") {
        test::TensorRng rng(9);
        test::MemoryWeights w;
        w.add("w", rng.uniform(DType::BF16, {16, 16}, 0.25f));
        const HostTensor x = rng.uniform(DType::F32, {3, 16});
        Graph g;
        g.output("out", g.matmul(g.input("x", DType::F32, {kT, 16}), g.weight(w, "w")));
        const HostTensor out = run_out(g, w, {{"x", x}});
        Tensor64 w_t = to_f64(w.tensor("w"));
        for (int64_t i = 0; i < 16; ++i) {
            for (int64_t j = 0; j < i; ++j)
                std::swap(w_t[i * 16 + j], w_t[j * 16 + i]);
        }
        CHECK(test::all_close(out, test::ref::matmul(to_f64(x), to_f64(w.tensor("w"))), 4e-3, 4e-3));
        CHECK_FALSE(test::all_close(out, test::ref::matmul(to_f64(x), w_t), 4e-3, 4e-3));
    }
}
