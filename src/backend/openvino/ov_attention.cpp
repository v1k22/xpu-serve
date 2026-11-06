#include "backend/openvino/ov_attention.h"

#include <limits>
#include <memory>

#include <openvino/op/ops.hpp>

#include "util/error.h"

namespace xs::ov_backend {

namespace ops = ov::op;

AttentionImpl parse_attention_impl(std::string_view text) {
    if (text == "auto") return AttentionImpl::Auto;
    if (text == "sdpa") return AttentionImpl::Sdpa;
    if (text == "matmul") return AttentionImpl::MatMul;
    fail("attention implementation '", text, "' is unknown (expected auto, sdpa or matmul)");
}

OvOutput attention_mask(const OvOutput& tokens, const OvOutput& cached, int64_t sliding_window,
                        const std::string& label) {
    const std::string name = label + "/mask_w" + std::to_string(sliding_window);
    const OvOutput zero = i64_scalar(0);
    const OvOutput one = i64_scalar(1);
    const OvOutput query_index = std::make_shared<ops::v4::Range>(zero, tokens, one, ov::element::i64);
    const OvOutput key_pos = std::make_shared<ops::v4::Range>(zero, cached, one, ov::element::i64);
    const OvOutput first_query_pos = std::make_shared<ops::v1::Subtract>(cached, tokens);
    const OvOutput query_pos = std::make_shared<ops::v1::Add>(query_index, first_query_pos);
    const OvOutput query_col = std::make_shared<ops::v0::Unsqueeze>(query_pos, i64_scalar(1));
    const OvOutput distance = std::make_shared<ops::v1::Subtract>(query_col, key_pos);  // p_i - j, [T, L]
    OvOutput visible = std::make_shared<ops::v1::GreaterEqual>(distance, zero);
    if (sliding_window > 0) {
        const OvOutput in_window = std::make_shared<ops::v1::Less>(distance, i64_scalar(sliding_window));
        visible = std::make_shared<ops::v1::LogicalAnd>(visible, in_window);
    }
    const auto mask = std::make_shared<ops::v1::Select>(visible, f32_scalar(0.0f),
                                                        f32_scalar(-std::numeric_limits<float>::infinity()));
    return named(mask, name);
}

namespace {

// Grouped-query attention without materializing repeated kv heads: queries are viewed as [1, Hkv, G, T, D] against
// kv [1, Hkv, 1, L, D], so the size-1 group axis broadcasts. With one kv head (or no grouping) plain rank 4 already
// broadcasts. Repeating kv heads with Broadcast instead costs the GPU plugin a kernel build per new L on every step.
struct Grouped {
    OvOutput q, k, v, mask;
    bool rank5 = false;
};

Grouped group_heads(const AttentionOperands& in) {
    Grouped g;
    const int64_t group = in.num_heads / in.num_kv_heads;
    g.rank5 = group > 1 && in.num_kv_heads > 1;
    if (!g.rank5) {
        g.q = in.q;
        g.k = in.k;
        g.v = in.v;
        g.mask = std::make_shared<ops::v0::Unsqueeze>(in.mask, i64_constant({0, 1}));  // [1, 1, T, L]
        return g;
    }
    g.q = std::make_shared<ops::v1::Reshape>(in.q, i64_constant({1, in.num_kv_heads, group, -1, in.head_dim}), false);
    g.k = std::make_shared<ops::v0::Unsqueeze>(in.k, i64_scalar(2));
    g.v = std::make_shared<ops::v0::Unsqueeze>(in.v, i64_scalar(2));
    g.mask = std::make_shared<ops::v0::Unsqueeze>(in.mask, i64_constant({0, 1, 2}));  // [1, 1, 1, T, L]
    return g;
}

OvOutput sdpa(const Grouped& g, float scale) {
    return std::make_shared<ops::v13::ScaledDotProductAttention>(g.q, g.k, g.v, g.mask, f32_scalar(scale), false);
}

// scores -> scale -> softcap -> mask -> softmax -> probs @ v, with everything after the score matmul kept in f32.
OvOutput matmul_attention(const Grouped& g, const graph::AttentionAttrs& attrs, const std::string& label) {
    const auto scores = std::make_shared<ops::v0::MatMul>(g.q, g.k, false, true);
    std::shared_ptr<ov::Node> x = std::make_shared<ops::v1::Multiply>(scores, f32_scalar(attrs.scale));
    keep_in_f32(x);
    if (attrs.softcap > 0.0f) {
        const auto divided = std::make_shared<ops::v1::Divide>(x, f32_scalar(attrs.softcap));
        const auto tanh = std::make_shared<ops::v0::Tanh>(divided);
        x = std::make_shared<ops::v1::Multiply>(tanh, f32_scalar(attrs.softcap));
        keep_in_f32(divided);
        keep_in_f32(tanh);
        keep_in_f32(x);
    }
    const auto masked = std::make_shared<ops::v1::Add>(x, g.mask);
    const auto probs = std::make_shared<ops::v8::Softmax>(masked, -1);
    keep_in_f32(masked);
    keep_in_f32(probs);
    named(probs, label, "probs");
    return std::make_shared<ops::v0::MatMul>(probs, g.v);
}

}  // namespace

OvOutput attention(const AttentionOperands& in, const graph::AttentionAttrs& attrs, AttentionImpl impl,
                   const std::string& label) {
    XS_CHECK(in.num_kv_heads > 0 && in.num_heads % in.num_kv_heads == 0, "attention ", label, ": ", in.num_heads,
             " query heads are not a multiple of ", in.num_kv_heads, " kv heads");
    if (impl == AttentionImpl::Sdpa && attrs.softcap > 0.0f) {
        fail("attention ", label, " has softcap ", attrs.softcap,
             ", which ScaledDotProductAttention cannot express; use attention=auto or matmul");
    }
    const Grouped g = group_heads(in);
    const bool use_matmul = impl == AttentionImpl::MatMul || attrs.softcap > 0.0f;
    OvOutput out = use_matmul ? matmul_attention(g, attrs, label) : sdpa(g, attrs.scale);
    if (g.rank5) out = std::make_shared<ops::v1::Reshape>(out, i64_constant({1, in.num_heads, -1, in.head_dim}), false);
    return named(out.get_node_shared_ptr(), label, use_matmul ? "matmul_attention" : "sdpa");
}

}  // namespace xs::ov_backend
