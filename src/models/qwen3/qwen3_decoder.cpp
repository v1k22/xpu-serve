#include <cmath>

#include "models/common/blocks.h"
#include "models/qwen3/qwen3.h"
#include "util/error.h"

namespace xs::qwen3 {

using graph::Graph;
using graph::kT;
using graph::Value;

Decoder::Decoder(Config cfg, const WeightStore& weights, std::string prefix, std::string lm_head)
    : cfg_(std::move(cfg)), weights_(weights), prefix_(std::move(prefix)), lm_head_(std::move(lm_head)) {
    XS_CHECK(weights_.contains(prefix_ + "embed_tokens.weight"), "qwen3: no ", prefix_, "embed_tokens.weight");
    if (lm_head_.empty()) lm_head_ = prefix_ + "embed_tokens.weight";
    XS_CHECK(weights_.contains(lm_head_), "qwen3: no output layer ", lm_head_);
}

Decoder::Inputs Decoder::inputs(Graph& g) const {
    return {g.input("token_ids", DType::I32, {kT}), g.input("positions", DType::I32, {kT})};
}

void Decoder::token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const {
    const auto n = static_cast<int64_t>(tokens.size());
    XS_CHECK(n > 0, "qwen3: a step needs at least one token");
    HostTensor ids(DType::I32, {n});
    HostTensor pos(DType::I32, {n});
    for (int64_t i = 0; i < n; ++i) {
        const int32_t t = tokens[static_cast<size_t>(i)];
        if (t < 0 || t >= cfg_.vocab_size) {
            fail("qwen3: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
        }
        ids.data<int32_t>()[i] = t;
        pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
}

Value Decoder::embed(Graph& g, Value ids) const { return g.gather(weight(g, prefix_ + "embed_tokens.weight"), ids); }

Value Decoder::norm(Graph& g, Value x, const std::string& name) const {
    return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.rms_norm_eps));
}

Value Decoder::layers(Graph& g, Value h, Value pos) const {
    const Value freq = g.constant(blocks::rope_inv_freq(cfg_.rope_theta, cfg_.head_dim, cfg_.head_dim / 2,
                                                        cfg_.head_dim / 2));
    const auto proj = [&](const std::string& name) -> blocks::Linear {
        return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
    };
    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string label = "layers." + std::to_string(i);
        const std::string L = prefix_ + label + ".";
        Graph::Scope s(g, label);
        blocks::AttentionSpec spec;
        spec.heads = cfg_.num_heads;
        spec.kv_heads = cfg_.num_kv_heads;
        spec.head_dim = cfg_.head_dim;
        spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
        spec.kv_label = label;
        blocks::AttentionParts parts;
        const std::string A = L + "self_attn.";
        parts.q = proj(A + "q_proj.weight");
        parts.k = proj(A + "k_proj.weight");
        parts.v = proj(A + "v_proj.weight");
        parts.o = proj(A + "o_proj.weight");
        parts.q_norm = [this, &g, A](Value q) { return norm(g, q, A + "q_norm.weight"); };
        parts.k_norm = [this, &g, A](Value k) { return norm(g, k, A + "k_norm.weight"); };
        h = g.add(h, blocks::attention(g, spec, parts, norm(g, h, L + "input_layernorm.weight"), pos, freq).out);
        const Value y = norm(g, h, L + "post_attention_layernorm.weight");
        h = g.add(h, blocks::gated_mlp(g, y, proj(L + "mlp.gate_proj.weight"), proj(L + "mlp.up_proj.weight"),
                                       proj(L + "mlp.down_proj.weight"), blocks::Act::Silu));
    }
    return h;
}

Value Decoder::final_norm(Graph& g, Value h) const { return norm(g, h, prefix_ + "norm.weight"); }

Value Decoder::lm_head(Graph& g) const { return weight(g, lm_head_); }

}  // namespace xs::qwen3
