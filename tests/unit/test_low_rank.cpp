// Low-rank changes to weights (graph/low_rank.h): a program that reads them gives what the same program gives over the
// weights with the change merged into them, on cpu-ref.
#include <doctest/doctest.h>

#include "backend/backend.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "backend/backend_test_util.h"
#include "graph/low_rank.h"

using namespace xs;
using namespace xs::graph;

namespace {

TensorMap run(const Graph& g, const WeightStore& w, const TensorMap& inputs) {
    static const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(g, w, SessionOptions{.precision = DType::F32});
    TensorMap out;
    session->run(inputs, out);
    return out;
}

// w + scale * b @ a, per leading index of a 3-D bank: [.., out, r] @ [.., r, in].
HostTensor merged(const HostTensor& w, const HostTensor& a, const HostTensor& b, float scale) {
    HostTensor m = w.clone();
    const Dims& s = w.shape();
    const int64_t out = s[s.size() - 2], in = s.back(), r = a.shape()[a.shape().size() - 2];
    const int64_t banks = w.numel() / (out * in);
    for (int64_t e = 0; e < banks; ++e) {
        for (int64_t o = 0; o < out; ++o) {
            for (int64_t i = 0; i < in; ++i) {
                double acc = 0;
                for (int64_t k = 0; k < r; ++k)
                    acc += b.data<float>()[(e * out + o) * r + k] * a.data<float>()[(e * r + k) * in + i];
                m.data<float>()[(e * out + o) * in + i] += scale * static_cast<float>(acc);
            }
        }
    }
    return m;
}

}  // namespace

TEST_CASE("low rank: a matmul, an expert bank and an embedding table read with their changes, as if merged") {
    test::TensorRng rng(5);
    const int64_t vocab = 11, width = 8, in = 6, out = 5, experts = 4, r = 2;
    test::MemoryWeights w, m;  // the weights with the changes beside them; the same merged
    const HostTensor lin = rng.uniform(DType::F32, {out, width});
    const HostTensor lin_a = rng.uniform(DType::F32, {r, width}), lin_b = rng.uniform(DType::F32, {out, r});
    const HostTensor bank = rng.uniform(DType::F32, {experts, in, out});
    const HostTensor bank_a = rng.uniform(DType::F32, {experts, r, out});
    const HostTensor bank_b = rng.uniform(DType::F32, {experts, in, r});
    const HostTensor table = rng.uniform(DType::F32, {vocab, width});
    const HostTensor table_a = rng.uniform(DType::F32, {vocab, r}), table_b = rng.uniform(DType::F32, {width, r});
    for (auto [name, t] : {std::pair{"lin", &lin}, {"bank", &bank}, {"table", &table}, {"lin_a", &lin_a},
                           {"lin_b", &lin_b}, {"bank_a", &bank_a}, {"bank_b", &bank_b}, {"table_a", &table_a},
                           {"table_b", &table_b}}) {
        w.add(name, t->clone());
    }
    m.add("lin", merged(lin, lin_a, lin_b, 0.5f));
    m.add("bank", merged(bank, bank_a, bank_b, 2.0f));
    // A table's change is A B^T: row v gains B @ A[v].
    HostTensor table_merged = table.clone();
    for (int64_t v = 0; v < vocab; ++v) {
        for (int64_t c = 0; c < width; ++c) {
            for (int64_t k = 0; k < r; ++k) {
                table_merged.data<float>()[v * width + c] +=
                    table_a.data<float>()[v * r + k] * table_b.data<float>()[c * r + k];
            }
        }
    }
    m.add("table", std::move(table_merged));

    const auto build = [](const WeightStore& store) {
        Graph g;
        const Value ids = g.input("ids", DType::I32, {kT});
        const Value idx = g.input("idx", DType::I32, {kT, 2});
        const Value h = g.gather(g.weight(store, "table"), ids);  // [T, width]
        const Value y = g.matmul(h, g.weight(store, "lin"));      // [T, out]
        g.output("y", y);
        g.output("e", g.expert_matmul(y, g.weight(store, "bank"), idx));  // [T, 2, in]
        return g;
    };
    const LowRanks changes = {{"lin", {"lin_a", "lin_b", 0.5f}}, {"bank", {"bank_a", "bank_b", 2.0f}},
                              {"table", {"table_a", "table_b", 1.0f}}};
    std::set<std::string> used;
    const Graph adapted = with_low_rank(build(w), w, changes, &used);
    CHECK(used == std::set<std::string>{"lin", "bank", "table"});
    TensorMap inputs;
    inputs["ids"] = HostTensor::from_vector(DType::I32, {3}, std::vector<int32_t>{3, 0, 10});
    inputs["idx"] = HostTensor::from_vector(DType::I32, {3, 2}, std::vector<int32_t>{1, 3, 0, 2, 3, 3});
    const TensorMap got = run(adapted, w, inputs);
    const TensorMap want = run(build(m), m, inputs);
    for (const char* name : {"y", "e"}) {
        CAPTURE(name);
        CHECK(test::all_close(got.at(name), want.at(name), 1e-5, 1e-5));
    }
    // Without the changes the graph is the original one.
    CHECK_FALSE(test::all_close(run(build(w), w, inputs).at("y"), want.at("y"), 1e-5, 1e-5));
}

TEST_CASE("low rank: a weight another op reads, and a change whose shapes do not fit, are refused by name") {
    test::TensorRng rng(6);
    test::MemoryWeights w;
    w.add("lin", rng.uniform(DType::F32, {4, 6}));
    w.add("a", rng.uniform(DType::F32, {2, 6}));
    w.add("b", rng.uniform(DType::F32, {4, 2}));
    w.add("bad_b", rng.uniform(DType::F32, {5, 2}));
    Graph g;  // the adapted weight as a matmul's x
    g.output("y", g.matmul(g.weight(w, "lin"), g.weight(w, "a")));
    CHECK_THROWS_WITH(with_low_rank(g, w, {{"lin", {"a", "b", 1.0f}}}, nullptr),
                      doctest::Contains("lin is read by MatMul"));
    Graph h;
    h.output("y", h.matmul(h.input("x", DType::F32, {kT, 6}), h.weight(w, "lin")));
    CHECK_THROWS_WITH(with_low_rank(h, w, {{"lin", {"a", "bad_b", 1.0f}}}, nullptr), doctest::Contains("do not fit"));
    CHECK_NOTHROW(with_low_rank(h, w, {{"lin", {"a", "b", 1.0f}}}, nullptr));
}
