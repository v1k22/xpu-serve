// Trees of elementwise ops the GPU backend runs as one program (graph/elementwise_groups.h), against cpu-ref, which
// runs each op alone: slices read in place, broadcast leaves, every unary op, values read twice, trees past the
// program's size.
#include <doctest/doctest.h>

#include <functional>

#include "backend/cpu_ref/cpu_backend.h"
#include "device_test_support.h"

using namespace xs;
using xs::test::to_f64;
namespace dev = xs::test::device;

namespace {

const dev::Tolerance kTol = dev::tolerance({1e-5, 1e-5}, {1e-5, 1e-5});

// Runs `build`'s graph at `tokens` on both backends with the same random inputs and compares every output.
void compare(const std::function<void(graph::Graph&)>& build, const std::vector<int64_t>& token_counts) {
    graph::Graph g;
    build(g);
    g.validate();
    test::MemoryWeights weights;
    const auto oracle = make_cpu_ref_backend()->compile(g, weights, SessionOptions{});
    const auto gpu = dev::backend().compile(g, weights, dev::options());
    test::TensorRng rng(7);
    for (const int64_t t : token_counts) {
        CAPTURE(t);
        TensorMap inputs;
        for (const graph::Value& v : g.inputs()) {
            Dims shape = g.type(v).shape;
            for (int64_t& d : shape) d = d == graph::kT ? t : d;
            inputs[std::get<graph::InputAttrs>(g.node(v).attrs).name] = rng.uniform(DType::F32, shape, 2.0f);
        }
        TensorMap expected, actual;
        oracle->run(inputs, expected);
        gpu->run(inputs, actual);
        for (const auto& [name, tensor] : expected) {
            CAPTURE(name);
            CHECK(dev::close("elementwise", actual.at(name), to_f64(tensor), kTol));
        }
    }
}

}  // namespace

TEST_SUITE("device elementwise programs" * doctest::skip(!dev::enabled())) {
    TEST_CASE("a gate's mix of two streams: tanh of two slices of one matmul's output, the stream read twice") {
        compare(
            [](graph::Graph& g) {
                const auto h = g.input("h", DType::F32, {graph::kT, 40});
                const auto m = g.input("m", DType::F32, {graph::kT, 40});
                const auto both = g.input("both", DType::F32, {graph::kT, 80});
                const auto forget = g.tanh(g.slice(both, 1, 0, 40));
                const auto gain = g.tanh(g.slice(both, 1, 40, 80));
                g.output("out", g.add(g.add(h, g.scale(g.mul(forget, h), -1.0f)), g.mul(gain, m)));
            },
            {1, 5});
    }

    TEST_CASE("a softcapped SwiGLU over interleaved and halved gate and up rows, by expert weight") {
        compare(
            [](graph::Graph& g) {
                const auto both = g.input("both", DType::F32, {graph::kT, 2, 24});
                const auto w = g.input("w", DType::F32, {graph::kT, 2, 1});
                const auto pairs = g.reshape(both, {graph::kT, 2, 12, 2});
                const auto gate = g.reshape(g.slice(pairs, 3, 0, 1), {graph::kT, 2, 12});
                const auto up = g.reshape(g.slice(pairs, 3, 1, 2), {graph::kT, 2, 12});
                const auto act = g.mul(g.mul(g.softcap(gate, 3.0f), g.sigmoid(gate)), g.softcap(up, 2.0f));
                g.output("interleaved", g.mul(act, w));
                const auto both2 = g.input("both2", DType::F32, {graph::kT, 2, 24});
                const auto gate2 = g.slice(both2, 2, 0, 12);
                const auto up2 = g.slice(both2, 2, 12, 24);
                g.output("halves", g.mul(g.silu(gate2), g.clamp(up2, -0.5f, 0.75f)));
            },
            {1, 3});
    }

    TEST_CASE("every unary op and Div, with a bias that repeats and a scale held across rows") {
        compare(
            [](graph::Graph& g) {
                const auto x = g.input("x", DType::F32, {graph::kT, 4, 16});
                const auto bias = g.input("bias", DType::F32, {16});
                const auto row = g.input("row", DType::F32, {graph::kT, 4, 1});
                const auto d = g.input("d", DType::F32, {graph::kT, 4, 16});
                const auto biased = g.add(x, bias);
                g.output("gelu_tanh", g.mul(g.gelu_tanh(biased), row));
                g.output("gelu", g.scale(g.gelu(biased), 0.5f));
                g.output("softplus", g.add(g.softplus(x), g.div(row, g.add(g.sigmoid(d), g.sigmoid(x)))));
                g.output("tanh", g.tanh(g.scale(x, 3.0f)));
            },
            {1, 6});
    }

    TEST_CASE("trees past the program's leaves and ops: a sum of twelve values and a chain of forty ops") {
        compare(
            [](graph::Graph& g) {
                graph::Value sum = g.input("x0", DType::F32, {graph::kT, 8});
                for (int i = 1; i < 12; ++i) {
                    sum = g.add(sum, g.input("x" + std::to_string(i), DType::F32, {graph::kT, 8}));
                }
                g.output("sum", sum);
                graph::Value chain = g.input("c", DType::F32, {graph::kT, 8});
                for (int i = 0; i < 40; ++i) chain = i % 2 == 0 ? g.scale(chain, 1.01f) : g.tanh(chain);
                g.output("chain", chain);
            },
            {1, 4});
    }
}
