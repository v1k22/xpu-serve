// ModernBERT's encoder as modeling_modernbert.py computes it (ModernBertEmbeddings, ModernBertEncoderLayer with
// ModernBertAttention and ModernBertMLP, final_norm).

#include "models/common/blocks.h"
#include "models/common/encoder.h"
#include "models/common/torch_nn.h"
#include "models/modern_bert/modern_bert.h"
#include "util/error.h"

namespace xs::modern_bert {

using graph::Graph;
using graph::Value;

Value encode(Graph& g, const WeightStore& w, const Config& c, Value ids, Value positions, const std::string& prefix) {
    const float eps = static_cast<float>(c.norm_eps);
    const int64_t h = c.hidden_size, inter = c.intermediate_size;
    Value x;
    {
        Graph::Scope s(g, "embeddings");
        x = g.gather(g.weight(w, prefix + "embeddings.tok_embeddings.weight"), ids);
        x = torch_nn::layer_norm_without_bias(g, w, x, prefix + "embeddings.norm", eps);
    }
    const auto inv_freq = [&](double theta) {
        return g.constant(blocks::rope_inv_freq(theta, c.head_dim, c.head_dim / 2, c.head_dim / 2));
    };
    const Value global_freq = inv_freq(c.global_rope_theta);
    const Value local_freq = c.local_rope_theta == c.global_rope_theta ? global_freq : inv_freq(c.local_rope_theta);
    const blocks::AttentionSpec global = encoder::attention_spec(c.num_heads, c.head_dim);
    blocks::AttentionSpec local = global;
    // |i - j| <= local_attention / 2, in the graph's terms |i - j| < sliding_window.
    local.attrs.sliding_window = c.local_attention / 2 + 1;
    for (int64_t i = 0; i < c.num_layers; ++i) {
        Graph::Scope s(g, "layers." + std::to_string(i));
        const std::string L = prefix + "layers." + std::to_string(i) + ".";
        const bool is_local = c.local[static_cast<size_t>(i)];
        {
            Graph::Scope a(g, "attn");
            const Value in = i == 0 ? x : torch_nn::layer_norm_without_bias(g, w, x, L + "attn_norm", eps);
            const Value qkv = g.matmul(in, g.weight(w, L + "attn.Wqkv.weight"));  // [T, 3 h]: q, k, v
            XS_CHECK(g.type(qkv).shape.back() == 3 * h, "modern_bert: ", L, "attn.Wqkv.weight is not [3 x ", h,
                     ", ", h, "]");
            const auto part = [&g, qkv, h](int64_t k) {
                return blocks::Linear([&g, qkv, h, k](Value) { return g.slice(qkv, 1, k * h, (k + 1) * h); });
            };
            blocks::AttentionParts parts;
            parts.q = part(0);
            parts.k = part(1);
            parts.v = part(2);
            parts.o = blocks::matmul_by(g, g.weight(w, L + "attn.Wo.weight"));
            x = g.add(x, blocks::attention(g, is_local ? local : global, parts, in, positions,
                                           is_local ? local_freq : global_freq)
                             .out);
        }
        const Value in = torch_nn::layer_norm_without_bias(g, w, x, L + "mlp_norm", eps);
        const Value both = g.matmul(in, g.weight(w, L + "mlp.Wi.weight"));  // [T, 2 inter]: GELU's input, its gate
        XS_CHECK(g.type(both).shape.back() == 2 * inter, "modern_bert: ", L, "mlp.Wi.weight is not [2 x ", inter,
                 ", ", h, "]");
        const auto half = [&g, both, inter](int64_t k) {
            return blocks::Linear([&g, both, inter, k](Value) { return g.slice(both, 1, k * inter, (k + 1) * inter); });
        };
        x = g.add(x, blocks::gated_mlp(g, in, half(0), half(1), blocks::matmul_by(g, g.weight(w, L + "mlp.Wo.weight")),
                                       blocks::Act::Gelu));
    }
    return torch_nn::layer_norm_without_bias(g, w, x, prefix + "final_norm", eps);
}

}  // namespace xs::modern_bert
