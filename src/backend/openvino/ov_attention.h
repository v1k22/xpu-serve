#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "backend/openvino/ov_builders.h"
#include "graph/graph.h"

namespace xs::ov_backend {

// How Attention nodes are lowered. Auto picks ScaledDotProductAttention unless the node needs a softcap.
enum class AttentionImpl { Auto, Sdpa, MatMul };

AttentionImpl parse_attention_impl(std::string_view text);

// Additive [T, L] f32 mask from the scalar i64 dims T and L: 0 where key j is visible to query i (p_i = L - T + i,
// j <= p_i and, with a window, p_i - j < window), -inf elsewhere.
OvOutput attention_mask(const OvOutput& tokens, const OvOutput& cached, int64_t sliding_window,
                        const std::string& label);

struct AttentionOperands {
    OvOutput q;     // [1, Hq, T, D]
    OvOutput k;     // [1, Hkv, L, D]
    OvOutput v;     // [1, Hkv, L, D]
    OvOutput mask;  // [T, L]
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
};

// Returns [1, Hq, T, D].
OvOutput attention(const AttentionOperands& in, const graph::AttentionAttrs& attrs, AttentionImpl impl,
                   const std::string& label);

}  // namespace xs::ov_backend
