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
