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

TEST_CASE("attention that is not causal takes a window only over a step's rows: its own, or a cache") {
    Graph g;
    const Value q = g.input("q", DType::F32, {kT, 4, 8});
    const Value k = g.input("k", DType::F32, {kT, 1, 8});
    const AttentionAttrs window{1.0f, 16, 0.0f, false};
    const auto [ka, va] = g.kv_cache(g.add_kv_slot(1, 8, 16, "block"), k, k);
    CHECK(g.type(g.attention(q, ka, va, window)).shape == (xs::Dims{kT, 4, 8}));
    CHECK(g.type(g.attention(q, k, k, window)).shape == (xs::Dims{kT, 4, 8}));
    const Value fixed = g.input("fixed", DType::F32, {3, 4, 8});
    CHECK_THROWS_AS(g.attention(fixed, ka, va, window), xs::Error);
    CHECK_THROWS_AS(g.attention(q, g.input("kf", DType::F32, {3, 1, 8}), g.input("vf", DType::F32, {3, 1, 8}), window),
                    xs::Error);
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

TEST_CASE("attention in groups takes the step's own q, k and v and an i32 group per token") {
    Graph g;
    AttentionAttrs attrs;
    attrs.causal = false;
    attrs.grouped = true;
    const Value q = g.input("q", DType::F32, {kT, 4, 8});
    const Value k = g.input("k", DType::F32, {kT, 2, 8});
    const Value groups = g.input("groups", DType::I32, {kT});
    const Value out = g.attention_in_groups(q, k, k, groups, attrs);
    CHECK(g.type(out).shape == (xs::Dims{kT, 4, 8}));
    CHECK(g.node(out).inputs.size() == 4);
    CHECK(std::get<AttentionAttrs>(g.node(out).attrs).grouped);
    CHECK_THROWS_AS(g.attention(q, k, k, attrs), xs::Error);  // groups only through attention_in_groups
    CHECK_THROWS_AS(g.attention(q, k, k, g.input("bias", DType::F32, {kT}), attrs), xs::Error);
    CHECK_THROWS_AS(g.attention_in_groups(q, k, k, g.input("f", DType::F32, {kT}), attrs), xs::Error);
    CHECK_THROWS_AS(g.attention_in_groups(q, k, k, g.input("g2", DType::I32, {kT, 1}), attrs), xs::Error);
    const auto [ka, va] = g.kv_cache(g.add_kv_slot(2, 8, 0, "cache"), k, k);
    CHECK_THROWS_AS(g.attention_in_groups(q, ka, va, groups, attrs), xs::Error);  // keys of a cache
    AttentionAttrs other = attrs;
    other.sliding_window = 4;
    CHECK_THROWS_AS(g.attention_in_groups(q, k, k, groups, other), xs::Error);
    other = attrs;
    other.causal = true;
    CHECK_THROWS_AS(g.attention_in_groups(q, k, k, groups, other), xs::Error);
    other = attrs;
    other.grouped = false;
    CHECK_THROWS_AS(g.attention_in_groups(q, k, k, groups, other), xs::Error);
}

TEST_CASE("prev_row keeps the shape; float rows only") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 3, 5});
    CHECK(g.type(g.prev_row(x)).shape == (xs::Dims{kT, 3, 5}));
    CHECK_THROWS_AS(g.prev_row(g.input("ids", DType::I32, {kT})), xs::Error);
}

TEST_CASE("mhc_route: pre, post [T, S] and mix [T, S, S] from [T, 2S + S^2]; stream_mix mixes [P, S] by [S, D]") {
    Graph g;
    const Value z = g.input("z", DType::F32, {kT, 8});
    const xs::graph::MhcMaps maps = g.mhc_route(z, g.input("b", DType::F32, {8}), 2, 20);
    CHECK(g.type(maps.pre).shape == (xs::Dims{kT, 2}));
    CHECK(g.type(maps.post).shape == (xs::Dims{kT, 2}));
    CHECK(g.type(maps.mix).shape == (xs::Dims{kT, 2, 2}));
    CHECK(g.type(g.mhc_route(g.input("z4", DType::F32, {kT, 24}), Value{}, 4, 1).mix).shape == (xs::Dims{kT, 4, 4}));
    CHECK_THROWS_AS(g.mhc_route(z, Value{}, 3, 20), xs::Error);                             // 3 streams take 15
    CHECK_THROWS_AS(g.mhc_route(z, Value{}, 2, 0), xs::Error);                              // no iteration
    CHECK_THROWS_AS(g.mhc_route(z, g.input("b6", DType::F32, {6}), 2, 20), xs::Error);      // a bias of another width
    const Value m = g.input("m", DType::F32, {kT, 2, 2});
    const Value x = g.input("x", DType::F32, {kT, 2, 16});
    CHECK(g.type(g.stream_mix(m, x)).shape == (xs::Dims{kT, 2, 16}));
    const Value pre = g.input("pre", DType::F32, {kT, 1, 2});
    CHECK(g.type(g.stream_mix(pre, x)).shape == (xs::Dims{kT, 1, 16}));
    const Value a = g.input("a", DType::F32, {kT, 2});
    const Value y = g.input("y", DType::F32, {kT, 16});
    CHECK(g.type(g.stream_mix(m, x, a, y)).shape == (xs::Dims{kT, 2, 16}));
    CHECK_THROWS_AS(g.stream_mix(g.input("m3", DType::F32, {kT, 2, 3}), x), xs::Error);   // S differs
    CHECK_THROWS_AS(g.stream_mix(g.input("m4", DType::F32, {4, 2, 2}), x), xs::Error);    // leading dims differ
    CHECK_THROWS_AS(g.stream_mix(m, x, a, g.input("y8", DType::F32, {kT, 8})), xs::Error);  // y of another width
    CHECK_THROWS_AS(g.stream_mix(pre, x, a, y), xs::Error);                                  // a of another P
}

TEST_CASE("conv2d output size: stride, padding per side, dilation; groups must divide both channel counts") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 1, 128, 100});
    xs::graph::Conv2dAttrs s2;
    s2.stride_h = s2.stride_w = 2;
    s2.pad_top = s2.pad_bottom = s2.pad_left = s2.pad_right = 1;
    const Value w = g.constant(xs::HostTensor(DType::F32, {480, 1, 3, 3}));
    CHECK(g.type(g.conv2d(x, w, Value{}, s2)).shape == (xs::Dims{kT, 480, 64, 50}));
    xs::graph::Conv2dAttrs causal;
    causal.dilation_w = 3;
    causal.pad_left = 18;
    const Value line = g.input("line", DType::F32, {2, 6, 1, 50});
    CHECK(g.type(g.conv2d(line, g.constant(xs::HostTensor(DType::F32, {4, 6, 1, 7})), Value{}, causal)).shape ==
          (xs::Dims{2, 4, 1, 50}));
    xs::graph::Conv2dAttrs grouped;
    grouped.groups = 4;
    CHECK_THROWS_AS(g.conv2d(line, g.constant(xs::HostTensor(DType::F32, {4, 6, 1, 7})), Value{}, grouped), xs::Error);
    CHECK_THROWS_AS(g.conv2d(x, w, g.constant(xs::HostTensor(DType::F32, {3})), s2), xs::Error);  // bias of 3
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

TEST_CASE("causal attention with a relative bias takes a bias per head by distance and a sink per head") {
    Graph g;
    const int slot = g.add_kv_slot(2, 32, 0, "kv");
    const Value k = g.input("k", DType::F32, {kT, 2, 32});
    const auto [ka, va] = g.kv_cache(slot, k, k);
    const Value q = g.input("q", DType::F32, {kT, 4, 32});
    const Value rel = g.input("rel", DType::F32, {kT, 4, 16});
    const Value sink = g.input("sink", DType::F32, {kT, 4});
    graph::AttentionAttrs attrs;
    attrs.rel_extent = 16;
    const Value out = g.attention_relative(q, ka, va, rel, sink, attrs);
    CHECK(g.type(out).shape == (Dims{kT, 4, 32}));
    CHECK(g.node(out).inputs.size() == 5);
    CHECK(std::get<graph::AttentionAttrs>(g.node(out).attrs).rel_extent == 16);
    CHECK_THROWS_AS(g.attention(q, ka, va, attrs), xs::Error);  // the bias only through attention_relative
    CHECK_THROWS_AS(g.attention_relative(q, ka, va, g.input("r2", DType::F32, {kT, 4, 8}), sink, attrs), xs::Error);
    CHECK_THROWS_AS(g.attention_relative(q, ka, va, rel, g.input("s2", DType::F32, {kT, 2}), attrs), xs::Error);
    graph::AttentionAttrs other = attrs;
    other.causal = false;
    CHECK_THROWS_AS(g.attention_relative(q, ka, va, rel, sink, other), xs::Error);
    other = attrs;
    other.softcap = 30.0f;
    CHECK_THROWS_AS(g.attention_relative(q, ka, va, rel, sink, other), xs::Error);
    other = attrs;
    other.rel_extent = 0;
    CHECK_THROWS_AS(g.attention_relative(q, ka, va, rel, sink, other), xs::Error);
}

TEST_CASE("l2_clip keeps the shape; dynamic_conv takes taps per token and group over a conv slot") {
    Graph g;
    const Value x = g.input("x", DType::F32, {kT, 3, 16});
    CHECK(g.type(g.l2_clip(x, 2.0f)).shape == (Dims{kT, 3, 16}));
    CHECK_THROWS_AS(g.l2_clip(x, 0.0f), xs::Error);

    const int slot = g.add_conv_slot(48, 4, "conv");
    const Value c = g.input("c", DType::F32, {kT, 48});
    const Value w = g.input("w", DType::F32, {4, 48});
    const Value taps = g.input("taps", DType::F32, {kT, 6, 4});
    CHECK(g.type(g.dynamic_conv(slot, c, w, taps)).shape == (Dims{kT, 48}));
    g.output("out", g.dynamic_conv(slot, c, w, taps));
    CHECK_THROWS_AS(g.validate(), xs::Error);  // the slot is used twice
    const int other = g.add_conv_slot(48, 4, "other");
    CHECK_THROWS_AS(g.dynamic_conv(other, c, g.input("w2", DType::F32, {48, 4}), taps), xs::Error);  // w is [K, C]
    CHECK_THROWS_AS(g.dynamic_conv(other, c, w, g.input("t2", DType::F32, {kT, 5, 4})), xs::Error);  // 5 groups of 48
    CHECK_THROWS_AS(g.dynamic_conv(other, c, w, g.input("t3", DType::F32, {kT, 6, 3})), xs::Error);
}

TEST_CASE("an anchored window is for attention that is not causal, over a cache") {
    Graph g;
    const int slot = g.add_kv_slot(2, 32, 0, "kv");
    const Value k = g.input("k", DType::F32, {kT, 2, 32});
    const auto [ka, va] = g.kv_cache(slot, k, k);
    const Value q = g.input("q", DType::F32, {kT, 4, 32});
    graph::AttentionAttrs attrs;
    attrs.causal = false;
    attrs.sliding_window = 8;
    attrs.anchored = true;
    CHECK(g.type(g.attention(q, ka, va, attrs)).shape == (Dims{kT, 4, 32}));
    graph::AttentionAttrs other = attrs;
    other.causal = true;
    CHECK_THROWS_AS(g.attention(q, ka, va, other), xs::Error);
    other = attrs;
    other.sliding_window = 0;
    CHECK_THROWS_AS(g.attention(q, ka, va, other), xs::Error);
    CHECK_THROWS_AS(g.attention(q, k, k, attrs), xs::Error);  // the step's own keys: no cache to anchor in
}
