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

// [1, Hkv, L, D] -> [1, Hq, L, D] by repeating each kv head for its group of query heads. This
// Unsqueeze/Broadcast/Reshape shape is the one the GPU plugin folds back into a grouped-query SDPA.
OvOutput expand_kv_heads(const OvOutput& x, const AttentionOperands& in, const std::string& label) {
    const int64_t group = in.num_heads / in.num_kv_heads;
    if (group == 1) return x;
    const OvOutput unsqueezed = std::make_shared<ops::v0::Unsqueeze>(x, i64_scalar(2));
    const OvOutput repeated = std::make_shared<ops::v3::Broadcast>(unsqueezed, i64_constant({1, 1, group, 1, 1}),
                                                                   ops::BroadcastType::BIDIRECTIONAL);
    return named(std::make_shared<ops::v1::Reshape>(repeated, i64_constant({0, in.num_heads, -1, in.head_dim}), true),
                 label);
}

OvOutput sdpa(const AttentionOperands& in, const OvOutput& k, const OvOutput& v, const OvOutput& mask,
              float scale) {
    return std::make_shared<ops::v13::ScaledDotProductAttention>(in.q, k, v, mask, f32_scalar(scale), false);
}

// scores -> scale -> softcap -> mask -> softmax -> probs @ v, with everything after the score matmul kept in f32.
OvOutput matmul_attention(const AttentionOperands& in, const OvOutput& k, const OvOutput& v, const OvOutput& mask,
                          const graph::AttentionAttrs& attrs, const std::string& label) {
    const auto scores = std::make_shared<ops::v0::MatMul>(in.q, k, false, true);
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
    const auto masked = std::make_shared<ops::v1::Add>(x, mask);
    const auto probs = std::make_shared<ops::v8::Softmax>(masked, -1);
    keep_in_f32(masked);
    keep_in_f32(probs);
    named(probs, label, "probs");
    return std::make_shared<ops::v0::MatMul>(probs, v);
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
    const OvOutput k = expand_kv_heads(in.k, in, label + "/k_expand");
    const OvOutput v = expand_kv_heads(in.v, in, label + "/v_expand");
    const OvOutput mask = std::make_shared<ops::v0::Unsqueeze>(in.mask, i64_constant({0, 1}));  // [1, 1, T, L]
    const bool use_matmul = impl == AttentionImpl::MatMul || attrs.softcap > 0.0f;
    const OvOutput out = use_matmul ? matmul_attention(in, k, v, mask, attrs, label) : sdpa(in, k, v, mask, attrs.scale);
    return named(out.get_node_shared_ptr(), label, use_matmul ? "matmul_attention" : "sdpa");
}

}  // namespace xs::ov_backend
