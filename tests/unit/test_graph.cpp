#include <doctest/doctest.h>

#include "graph/graph.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;

namespace {
class FakeWeights final : public WeightStore {
public:
    void add(const std::string& name, DType dtype, Dims shape) { views_[name] = TensorView{dtype, std::move(shape), nullptr}; }
    const TensorView* find(const std::string& name) const override {
        auto it = views_.find(name);
        return it == views_.end() ? nullptr : &it->second;
    }
    std::vector<std::string> names() const override { return {}; }

private:
    std::map<std::string, TensorView> views_;
};
}  // namespace

TEST_CASE("shape inference through a small attention block") {
    FakeWeights w;
    w.add("embed", DType::BF16, {100, 16});
    w.add("q", DType::BF16, {32, 16});
    w.add("k", DType::BF16, {8, 16});
    Graph g;
    Graph::Scope layer(g, "layers.0");
    const Value ids = g.input("token_ids", DType::I32, {kT});
    const Value pos = g.input("positions", DType::I32, {kT});
    const Value h = g.gather(g.weight(w, "embed"), ids);
    CHECK(g.type(h).shape == Dims{kT, 16});
    const Value q = g.reshape(g.matmul(h, g.weight(w, "q")), {kT, 4, 8});
    const Value k = g.reshape(g.matmul(h, g.weight(w, "k")), {kT, 1, 8});
    const Value inv = g.constant(HostTensor(DType::F32, {4}));
    const Value qr = g.rope(q, pos, inv);
    const int slot = g.add_kv_slot(1, 8, 0, "layers.0");
    const auto [ka, va] = g.kv_cache(slot, g.rope(k, pos, inv), k);
    CHECK(g.type(ka).shape == Dims{kL, 1, 8});
    const Value o = g.attention(qr, ka, va, AttentionAttrs{});
    CHECK(g.type(o).shape == Dims{kT, 4, 8});
    const Value last = g.slice(g.reshape(o, {kT, 32}), 0, -1);
    CHECK(g.type(last).shape == Dims{1, 32});
    g.output("out", last);
    CHECK_NOTHROW(g.validate());
    CHECK(g.node(q).label.rfind("layers.0.", 0) == 0);
    CHECK(g.weight(w, "embed") == Value{2, 0});  // weights are deduplicated
}

TEST_CASE("misuse is refused with a named error") {
    FakeWeights w;
    w.add("q", DType::BF16, {32, 16});
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 12});
    CHECK_THROWS_AS(g.matmul(x, g.weight(w, "q")), xs::Error);
    CHECK_THROWS_AS(g.reshape(x, {kT, 5}), xs::Error);
    CHECK_THROWS_AS(g.slice(x, 0, 0, 2), xs::Error);  // symbolic axis: only "last n"
    CHECK_THROWS_AS(g.input("x", DType::F32, {kT}), xs::Error);
    CHECK_THROWS_AS(g.weight(w, "missing"), xs::Error);
    const int slot = g.add_kv_slot(1, 12, 0, "s");
    (void)slot;
    g.output("x", x);
    CHECK_THROWS_AS(g.validate(), xs::Error);  // slot never appended
}

TEST_CASE("mixture-of-experts ops: shapes and refusals") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 16});
    const Value probs = g.softmax(x);
    CHECK(g.type(probs).shape == (xs::Dims{kT, 16}));
    const auto [values, indices] = g.topk(probs, 4);
    CHECK(g.type(values).shape == (xs::Dims{kT, 4}));
    CHECK(g.type(indices).dtype == DType::I32);
    CHECK(g.type(g.reduce_sum(values, 1)).shape == (xs::Dims{kT}));
    CHECK(g.type(g.div(values, g.reshape(g.reduce_sum(values, 1), {kT, 1}))).shape == (xs::Dims{kT, 4}));
    FakeWeights w;
    w.add("experts", DType::F32, {6, 24, 16});
    const Value up = g.expert_matmul(x, g.weight(w, "experts"), indices);
    CHECK(g.type(up).shape == (xs::Dims{kT, 4, 24}));
    CHECK_THROWS_AS(g.topk(probs, 17), xs::Error);                    // k larger than the axis
    CHECK_THROWS_AS(g.reduce_sum(x, 2), xs::Error);                   // no such axis
    CHECK_THROWS_AS(g.expert_matmul(up, g.weight(w, "experts"), indices), xs::Error);  // inner dims 24 vs 16
}

TEST_CASE("attention that is not causal takes a window only over a step's rows and a cache") {
    Graph g;
    const Value q = g.input("q", DType::F32, {kT, 4, 8});
    const Value k = g.input("k", DType::F32, {kT, 1, 8});
    const auto [ka, va] = g.kv_cache(g.add_kv_slot(1, 8, 16, "block"), k, k);
    CHECK(g.type(g.attention(q, ka, va, AttentionAttrs{1.0f, 16, 0.0f, false})).shape == (xs::Dims{kT, 4, 8}));
    CHECK_THROWS_AS(g.attention(q, k, k, AttentionAttrs{1.0f, 16, 0.0f, false}), xs::Error);
    const Value fixed = g.input("fixed", DType::F32, {3, 4, 8});
    CHECK_THROWS_AS(g.attention(fixed, ka, va, AttentionAttrs{1.0f, 16, 0.0f, false}), xs::Error);
}

TEST_CASE("attention within blocks takes non-causal operands of one static length, a multiple of the block") {
    Graph g;
    AttentionAttrs attrs;
    attrs.causal = false;
    attrs.block = 16;
    const Value q = g.input("q", DType::F32, {48, 4, 8});
    const Value k = g.input("k", DType::F32, {48, 2, 8});
    CHECK(g.type(g.attention(q, k, k, attrs)).shape == (xs::Dims{48, 4, 8}));
    const Value longer = g.input("longer", DType::F32, {64, 2, 8});
    CHECK_THROWS_AS(g.attention(q, longer, longer, attrs), xs::Error);  // as many keys as queries
    attrs.block = 20;
    CHECK_THROWS_AS(g.attention(q, k, k, attrs), xs::Error);  // 48 tokens are no whole number of blocks
    attrs.block = 16;
    attrs.causal = true;
    CHECK_THROWS_AS(g.attention(q, k, k, attrs), xs::Error);
}

TEST_CASE("prev_row keeps the shape; float rows only") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 3, 5});
    CHECK(g.type(g.prev_row(x)).shape == (xs::Dims{kT, 3, 5}));
    CHECK_THROWS_AS(g.prev_row(g.input("ids", DType::I32, {kT})), xs::Error);
}

TEST_CASE("sign matmul: one word of signs per 32 columns, a scale per row") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 64});
    const Value signs = g.constant(xs::HostTensor(DType::I32, {10, 2}));
    const Value scales = g.constant(xs::HostTensor(DType::F32, {10}));
    CHECK(g.type(g.sign_matmul(x, signs, scales)).shape == (xs::Dims{kT, 10}));
    CHECK_THROWS_AS(g.sign_matmul(x, g.constant(xs::HostTensor(DType::I32, {10, 3})), scales), xs::Error);  // 96 columns
    CHECK_THROWS_AS(g.sign_matmul(x, signs, g.constant(xs::HostTensor(DType::F32, {9}))), xs::Error);       // 9 scales
    CHECK_THROWS_AS(g.sign_matmul(x, g.constant(xs::HostTensor(DType::F32, {10, 2})), scales), xs::Error);  // float signs
}
