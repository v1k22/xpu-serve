#include <cmath>

#include "models/common/blocks.h"
#include "models/qwen3/qwen3.h"
#include "util/error.h"

namespace xs::qwen3 {

using graph::Graph;
using graph::kT;
using graph::Value;

Decoder::Decoder(Config cfg, const WeightStore& weights, std::string prefix, std::string lm_head, bool three_axes)
    : cfg_(std::move(cfg)), weights_(weights), prefix_(std::move(prefix)), lm_head_(std::move(lm_head)),
      three_axes_(three_axes) {
    XS_CHECK(!three_axes_ || !cfg_.mrope_axes.empty(), "qwen3: positions on three axes need an interleaved ",
             "mrope_section in the config's RoPE settings");
    XS_CHECK(weights_.contains(prefix_ + "embed_tokens.weight"), "qwen3: no ", prefix_, "embed_tokens.weight");
    if (lm_head_.empty()) lm_head_ = prefix_ + "embed_tokens.weight";
    XS_CHECK(weights_.contains(lm_head_), "qwen3: no output layer ", lm_head_);
}

Decoder::Inputs Decoder::inputs(Graph& g) const {
    const Dims pos = three_axes_ ? Dims{kT, 3} : Dims{kT};
    return {g.input("token_ids", DType::I32, {kT}), g.input("positions", DType::I32, pos)};
}

void Decoder::token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const {
    XS_CHECK(!three_axes_, "qwen3: this decoder reads positions on three axes");
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

void Decoder::token_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                           TensorMap& inputs) const {
    const auto n = static_cast<int64_t>(tokens.size());
    XS_CHECK(three_axes_ && n > 0 && positions.size() == tokens.size(), "qwen3: ", positions.size(),
             " positions for a step of ", n, " tokens on ", three_axes_ ? "three axes" : "one axis");
    HostTensor ids(DType::I32, {n});
    HostTensor pos(DType::I32, {n, 3});
    for (int64_t i = 0; i < n; ++i) {
        const int32_t t = tokens[static_cast<size_t>(i)];
        if (t < 0 || t >= cfg_.vocab_size) {
            fail("qwen3: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
        }
        ids.data<int32_t>()[i] = t;
        std::copy_n(positions[static_cast<size_t>(i)].data(), 3, pos.data<int32_t>() + 3 * i);
    }
    inputs["token_ids"] = std::move(ids);
    inputs["positions"] = std::move(pos);
}

Value Decoder::embed(Graph& g, Value ids) const { return g.gather(weight(g, prefix_ + "embed_tokens.weight"), ids); }

Value Decoder::norm(Graph& g, Value x, const std::string& name) const {
    return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.rms_norm_eps));
}

Value Decoder::layers(Graph& g, Value h, Value pos, const LayerHook& after_layer) const {
    const Value freq = g.constant(blocks::rope_inv_freq(cfg_.rope_theta, cfg_.head_dim, cfg_.head_dim / 2,
                                                        cfg_.head_dim / 2));
    Value axes;
    if (three_axes_) {
        const auto pairs = static_cast<int64_t>(cfg_.mrope_axes.size());
        axes = g.constant(HostTensor::from_vector(DType::I32, {pairs}, cfg_.mrope_axes));
    }
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
        spec.rope_axes = axes;
        blocks::AttentionParts parts;
        const std::string A = L + "self_attn.";
        parts.q = proj(A + "q_proj.weight");
        parts.k = proj(A + "k_proj.weight");
        parts.v = proj(A + "v_proj.weight");
        parts.o = proj(A + "o_proj.weight");
        parts.q_norm = [this, &g, A](Value q) { return norm(g, q, A + "q_norm.weight"); };
        parts.k_norm = [this, &g, A](Value k) { return norm(g, k, A + "k_norm.weight"); };
        h = g.add(h, blocks::attention(g, spec, parts, norm(g, h, L + "input_layernorm.weight"), pos, freq).out);
        h = g.add(h, mlp(g, L, i, norm(g, h, L + "post_attention_layernorm.weight")));
        if (after_layer) h = after_layer(i, h);
    }
    return h;
}

// Qwen3MoeMLP, or Qwen3MoeSparseMoeBlock in a sparse layer: the banks experts.gate_up_proj and experts.down_proj
// (join_expert_banks shows a checkpoint's experts that way).
Value Decoder::mlp(Graph& g, const std::string& L, int64_t index, Value y) const {
    const auto proj = [&](const std::string& name) -> blocks::Linear {
        return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
    };
    if (!cfg_.sparse(index)) {
        return blocks::gated_mlp(g, y, proj(L + "mlp.gate_proj.weight"), proj(L + "mlp.up_proj.weight"),
                                 proj(L + "mlp.down_proj.weight"), blocks::Act::Silu);
    }
    blocks::MoeSpec spec;
    spec.experts = cfg_.num_experts;
    spec.top_k = cfg_.experts_per_token;
    spec.router = blocks::Router::Softmax;
    spec.normalize = cfg_.norm_topk_prob;
    blocks::MoeParts parts;
    parts.router = weight(g, L + "mlp.gate.weight");
    const int64_t e = cfg_.num_experts, hidden = cfg_.hidden_size, inter = cfg_.moe_intermediate_size;
    const auto bank = [&](const std::string& name, const Dims& shape) {
        const Value w = weight(g, L + "mlp.experts." + name);
        XS_CHECK(g.type(w).shape == shape, "qwen3: ", L, "mlp.experts.", name, " has shape ",
                 dims_to_string(g.type(w).shape), ", expected ", dims_to_string(shape));
        return w;
    };
    parts.gate_up = bank("gate_up_proj", {e, 2 * inter, hidden});
    parts.down = bank("down_proj", {e, hidden, inter});
    return blocks::moe(g, y, spec, parts);
}

std::vector<std::string> Decoder::expert_prefixes() const {
    std::vector<std::string> out;
    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        if (cfg_.sparse(i)) out.push_back(prefix_ + "layers." + std::to_string(i) + ".mlp.experts.");
    }
    return out;
}

Value Decoder::final_norm(Graph& g, Value h) const { return norm(g, h, prefix_ + "norm.weight"); }

Value Decoder::lm_head(Graph& g) const { return weight(g, lm_head_); }

}  // namespace xs::qwen3
