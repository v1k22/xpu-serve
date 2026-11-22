#include <doctest/doctest.h>

#include "graph/elementwise_groups.h"

using namespace xs;
using namespace xs::graph;

namespace {

// Readers per node, a graph output counted twice (as a backend counts them).
std::vector<int> readers_of(const Graph& g) {
    std::vector<int> readers(g.nodes().size(), 0);
    for (const Node& n : g.nodes()) {
        for (const Value& in : n.inputs) ++readers[static_cast<size_t>(in.node)];
    }
    for (const auto& out : g.outputs()) readers[static_cast<size_t>(out.value.node)] += 2;
    return readers;
}

GroupRules any_rules(int leaves = 8, int ops = 24) {
    GroupRules r;
    r.max_leaves = leaves;
    r.max_ops = ops;
    r.free = [](size_t) { return true; };
    r.readable = [](Value) { return true; };
    return r;
}

std::vector<ElementwiseGroup> groups_of(const Graph& g, const GroupRules& rules = any_rules()) {
    return elementwise_groups(g, readers_of(g), rules);
}

}  // namespace

TEST_CASE("elementwise groups: a gate's mix of two streams is one group over four leaves") {
    Graph g;
    const Value h = g.input("h", DType::F32, {kT, 8});
    const Value m = g.input("m", DType::F32, {kT, 8});
    const Value both = g.input("both", DType::F32, {kT, 16});
    const Value forget = g.tanh(g.slice(both, 1, 0, 8));
    const Value gain = g.tanh(g.slice(both, 1, 8, 16));
    const Value out = g.add(g.add(h, g.scale(g.mul(forget, h), -1.0f)), g.mul(gain, m));
    g.output("out", out);
    const auto groups = groups_of(g);
    REQUIRE(groups.size() == 1);
    const ElementwiseGroup& gr = groups[0];
    CHECK(gr.root == out.node);
    CHECK(gr.ops.size() == 7);
    CHECK(gr.ops.back().node == out.node);
    REQUIRE(gr.leaves.size() == 4);  // both twice (two slices), h once, m
    int slices = 0;
    for (const GroupLeaf& l : gr.leaves) {
        if (l.map == LeafMap::Slice) {
            ++slices;
            CHECK(l.value == both);
        }
    }
    CHECK(slices == 2);
    CHECK(gr.folded.size() == 8);  // six ops and two slices
}

TEST_CASE("elementwise groups: broadcast leaves repeat or hold, a reshape is read in place") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 3, 4});
    const Value per_row = g.input("w", DType::F32, {kT, 3, 1});
    const Value bias = g.input("bias", DType::F32, {4});
    const Value flat = g.input("flat", DType::F32, {kT, 12});
    const Value y = g.add(g.mul(g.add(x, bias), per_row), g.reshape(flat, {kT, 3, 4}));
    g.output("y", y);
    const auto groups = groups_of(g);
    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].leaves.size() == 4);
    CHECK(groups[0].leaves[0].map == LeafMap::Same);
    CHECK(groups[0].leaves[1].map == LeafMap::Period);
    CHECK(groups[0].leaves[2].map == LeafMap::Block);
    CHECK(groups[0].leaves[3].map == LeafMap::Same);
    CHECK(groups[0].leaves[3].value == flat);
}

TEST_CASE("elementwise groups: a value with two readers ends a group; a lone op is no group") {
    Graph g;
    const Value a = g.input("a", DType::F32, {kT, 4});
    const Value b = g.input("b", DType::F32, {kT, 4});
    const Value shared = g.sigmoid(g.add(a, b));  // read twice: a group of its own, then a leaf
    const Value one = g.mul(shared, a);
    const Value two = g.add(g.tanh(shared), b);
    g.output("one", one);
    g.output("two", two);
    g.output("lone", g.scale(a, 2.0f));
    const auto groups = groups_of(g);
    REQUIRE(groups.size() == 2);  // two = tanh + add; shared = add + sigmoid; one and lone run alone
    CHECK(groups[0].root == two.node);
    CHECK(groups[0].ops.size() == 2);
    CHECK(groups[1].root == shared.node);
    CHECK(groups[1].ops.size() == 2);
}

TEST_CASE("elementwise groups: past the limits a group stops at a leaf it cannot take") {
    Graph g;
    std::vector<Value> in;
    for (int i = 0; i < 6; ++i) in.push_back(g.input("x" + std::to_string(i), DType::F32, {kT, 4}));
    Value sum = in[0];
    for (int i = 1; i < 6; ++i) sum = g.add(sum, in[static_cast<size_t>(i)]);
    g.output("sum", sum);
    const auto wide = groups_of(g, any_rules(4, 24));
    // The root's tree needs six leaves: it is refused, and the next root down takes what fits.
    REQUIRE(!wide.empty());
    for (const ElementwiseGroup& gr : wide) CHECK(gr.leaves.size() <= 4);
    const auto narrow = groups_of(g, any_rules(8, 3));
    for (const ElementwiseGroup& gr : narrow) CHECK(gr.ops.size() <= 3);
}

TEST_CASE("elementwise groups: ops other fusions claimed and values the backend cannot read stay out") {
    Graph g;
    const Value a = g.input("a", DType::F32, {kT, 4});
    const Value b = g.input("b", DType::F32, {kT, 4});
    const Value inner = g.mul(a, b);
    const Value out = g.add(g.silu(inner), b);
    g.output("out", out);
    GroupRules rules = any_rules();
    rules.free = [&](size_t i) { return static_cast<int32_t>(i) != inner.node; };
    const auto groups = elementwise_groups(g, readers_of(g), rules);
    REQUIRE(groups.size() == 1);
    CHECK(groups[0].ops.size() == 2);  // silu and add; the claimed mul is a leaf
    rules.readable = [&](Value v) { return v != b; };
    CHECK(elementwise_groups(g, readers_of(g), rules).empty());
}

TEST_CASE("elementwise groups: a slice another node reads through its reshape runs on its own") {
    Graph g;
    const Value both = g.input("both", DType::F32, {kT, 4, 2});
    const Value gate = g.reshape(g.slice(both, 2, 0, 1), {kT, 4});
    const Value y = g.mul(g.softcap(gate, 2.0f), g.tanh(gate));  // both readers of gate inside: the slice folds
    g.output("y", y);
    const Value z = g.scale(g.softcap(gate, 3.0f), 2.0f);
    g.output("z", g.rms_norm(gate, 1e-6f));  // a reader outside every group
    g.output("zz", z);
    const auto groups = groups_of(g);
    for (const ElementwiseGroup& gr : groups) {
        for (const GroupLeaf& l : gr.leaves) CHECK(l.map != LeafMap::Slice);
    }
    Graph h;
    const Value b2 = h.input("both", DType::F32, {kT, 4, 2});
    const Value g2 = h.reshape(h.slice(b2, 2, 0, 1), {kT, 4});
    h.output("y", h.mul(h.softcap(g2, 2.0f), h.tanh(g2)));
    const auto inside = groups_of(h);
    REQUIRE(inside.size() == 1);
    REQUIRE(inside[0].leaves.size() == 1);
    CHECK(inside[0].leaves[0].map == LeafMap::Slice);
}
