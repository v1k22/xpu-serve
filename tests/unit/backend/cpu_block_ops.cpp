// cpu-ref with weights in GGUF block formats: an op that reads a Block weight gives exactly what it gives with the
// weight's decoded f32 values.
#include <doctest/doctest.h>

#include <cstring>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"

using namespace xs;
using namespace xs::graph;

namespace {

constexpr BlockFormat kFormats[] = {BlockFormat::Q4_0, BlockFormat::Q4_1, BlockFormat::Q5_0, BlockFormat::Q5_1,
                                    BlockFormat::Q8_0};

// `w` with every weight replaced by its decoded f32 value.
test::MemoryWeights decoded(const test::MemoryWeights& w) {
    test::MemoryWeights out;
    for (const std::string& name : w.names())
        out.add(name, w.tensor(name));
    return out;
}

bool same_bits(const HostTensor& a, const HostTensor& b) {
    return a.dtype() == b.dtype() && a.shape() == b.shape() && std::memcmp(a.raw(), b.raw(), a.nbytes()) == 0;
}

}  // namespace

TEST_SUITE("cpu-ref ops") {
    TEST_CASE("matmul, gather and expert_matmul read a block format weight as its decoded values") {
        const auto backend = make_cpu_ref_backend();
        test::TensorRng rng(31);
        for (const BlockFormat f : kFormats) {
            CAPTURE(block_format_name(f));
            const int64_t k = 2 * block_layout(f).weights;
            const auto blocks = [&](Dims shape, uint32_t seed) {
                return test::random_blocks(f, numel(shape) / block_layout(f).weights, seed);
            };
            test::MemoryWeights w;
            w.add_block("w", f, {24, k}, blocks({24, k}, 1));
            w.add_block("table", f, {10, k}, blocks({10, k}, 2));
            w.add_block("bank", f, {3, 8, k}, blocks({3, 8, k}, 3));
            const test::MemoryWeights dense = decoded(w);

            const auto build = [&](const WeightStore& store) {
                Graph g;
                const Value x = g.input("x", DType::F32, {kT, k});
                g.output("mm", g.matmul(x, g.weight(store, "w")));
                g.output("rows", g.gather(g.weight(store, "table"), g.input("ids", DType::I32, {kT})));
                g.output("experts", g.expert_matmul(x, g.weight(store, "bank"), g.input("idx", DType::I32, {kT, 2})));
                return g;
            };
            const TensorMap inputs{{"x", rng.uniform(DType::F32, {5, k})},
                                   {"ids", test::make_tensor(DType::I32, {5}, std::vector<int32_t>{9, 0, 3, 3, 7})},
                                   {"idx", rng.ints(DType::I32, {5, 2}, 0, 2)}};
            const TensorMap got = test::run_once(*backend, build(w), w, inputs);
            const TensorMap want = test::run_once(*backend, build(dense), dense, inputs);
            for (const char* name : {"mm", "rows", "experts"}) {
                CAPTURE(name);
                CHECK(same_bits(got.at(name), want.at(name)));
            }
            // The gathered rows are the table's decoded rows themselves.
            const HostTensor& table = w.tensor("table");
            CHECK(std::memcmp(got.at("rows").data<float>(), table.data<float>() + 9 * k, k * sizeof(float)) == 0);
        }
    }
}
