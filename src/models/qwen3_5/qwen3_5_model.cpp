#include <algorithm>
#include <cmath>
#include <cstring>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/qwen3_5/qwen3_5.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::qwen3_5 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// Multimodal checkpoints keep the decoder under "model.language_model.", text-only ones under "model.".
std::string probe_prefix(const WeightStore& weights) {
    for (const char* prefix : {"model.language_model.", "model."}) {
        if (weights.contains(std::string(prefix) + "embed_tokens.weight")) return prefix;
    }
    fail("qwen3_5: no embed_tokens.weight under model.language_model. or model.; is this a Qwen3.5 checkpoint?");
}

HostTensor to_f32(const std::string& name, const TensorView& t) {
    HostTensor out(DType::F32, t.shape);
    const auto n = static_cast<size_t>(t.numel());
    switch (t.dtype) {
        case DType::F32: std::memcpy(out.raw(), t.data, n * sizeof(float)); break;
        case DType::BF16: convert_bf16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n); break;
        case DType::F16: convert_f16_to_f32(static_cast<const uint16_t*>(t.data), out.data<float>(), n); break;
        default: fail("qwen3_5: '", name, "' has dtype ", dtype_name(t.dtype));
    }
    return out;
}

class Qwen35Model final : public Model {
public:
    Qwen35Model(Config cfg, ModelInfo info, std::unique_ptr<SafetensorsWeights> weights)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)), prefix_(probe_prefix(*weights_)) {
        build_graph();
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return graph_; }
    const WeightStore& weights() const override { return *weights_; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "qwen3_5: a step needs at least one token");
        XS_CHECK(embeddings.empty(), "qwen3_5: image input is not supported yet");
        HostTensor ids(DType::I32, {n});
        HostTensor pos(DType::I32, {n});
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) fail("qwen3_5: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            ids.data<int32_t>()[i] = t;
            pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
        }
        inputs["token_ids"] = std::move(ids);
        inputs["positions"] = std::move(pos);
    }

private:
    std::string w(const std::string& suffix) const { return prefix_ + suffix; }
    Value weight(const std::string& suffix) { return graph_.weight(*weights_, w(suffix)); }
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }

    // Qwen3_5RMSNorm is zero-centered: x * rsqrt(mean(x^2) + eps) * (1 + w).
    Value norm(Value x, const std::string& suffix) {
        return blocks::rms_norm_one_plus(graph_, x, *weights_, w(suffix), eps());
    }
    // A projection whose weight node is made when the block uses it.
    blocks::Linear proj(const std::string& suffix) {
        return [this, suffix](Value x) { return graph_.matmul(x, weight(suffix)); };
    }

    void build_graph() {
        Graph& g = graph_;
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value embed = weight("embed_tokens.weight");
        Value h = g.gather(embed, ids);

        inv_freq_ = g.constant(
            blocks::rope_inv_freq(cfg_.rope_theta, cfg_.rotary_dim, cfg_.rotary_dim / 2, cfg_.rotary_dim / 2));

        for (int64_t i = 0; i < cfg_.num_layers; ++i) h = decoder_layer(i, h, pos);

        Graph::Scope head(g, "head");
        const Value last = norm(g.slice(h, 0, -1), "norm.weight");
        const Value lm_head = weights_->contains("lm_head.weight") ? g.weight(*weights_, "lm_head.weight") : embed;
        g.output("logits", g.matmul(last, lm_head));
        g.validate();
        log::debug("qwen3_5: step graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv, ",
                   g.conv_slots().size(), " conv and ", g.recurrent_slots().size(), " recurrent slots");
    }

    Value decoder_layer(int64_t i, Value h, Value pos) {
        Graph& g = graph_;
        const std::string L = "layers." + std::to_string(i) + ".";
        Graph::Scope s(g, "layers." + std::to_string(i));
        const Value x = norm(h, L + "input_layernorm.weight");
        const Value mixed = cfg_.layer_kinds[static_cast<size_t>(i)] == LayerKind::Linear ? linear_attention(i, x)
                                                                                           : full_attention(i, x, pos);
        h = g.add(h, mixed);
        const Value y = norm(h, L + "post_attention_layernorm.weight");
        return g.add(h, blocks::gated_mlp(g, y, proj(L + "mlp.gate_proj.weight"), proj(L + "mlp.up_proj.weight"),
                                          proj(L + "mlp.down_proj.weight"), blocks::Act::Silu));
    }

    // Qwen3_5GatedDeltaNet.
    Value linear_attention(int64_t i, Value x) {
        Graph& g = graph_;
        Graph::Scope s(g, "linear_attn");
        const std::string L = "layers." + std::to_string(i) + ".linear_attn.";
        const int64_t hk = cfg_.linear_key_heads, hv = cfg_.linear_value_heads;
        const int64_t dk = cfg_.linear_key_dim, dv = cfg_.linear_value_dim;
        const int64_t key_width = hk * dk, value_width = hv * dv;
        const int64_t conv_width = 2 * key_width + value_width;

        // q, k and v share one projection and one causal conv (with SiLU), then split.
        const Value conv_w = weight(L + "conv1d.weight");
        XS_CHECK(g.type(conv_w).shape == (Dims{conv_width, 1, cfg_.conv_kernel}), "qwen3_5: ", w(L + "conv1d.weight"),
                 " has shape ", dims_to_string(g.type(conv_w).shape));
        const int conv_slot = g.add_conv_slot(conv_width, cfg_.conv_kernel, "layers." + std::to_string(i));
        const Value qkv = g.silu(g.causal_conv(conv_slot, g.matmul(x, weight(L + "in_proj_qkv.weight")),
                                                g.reshape(conv_w, {conv_width, cfg_.conv_kernel})));
        const Value q = g.reshape(g.slice(qkv, 1, 0, key_width), {kT, hk, dk});
        const Value k = g.reshape(g.slice(qkv, 1, key_width, 2 * key_width), {kT, hk, dk});
        const Value v = g.reshape(g.slice(qkv, 1, 2 * key_width, conv_width), {kT, hv, dv});

        // beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), the per-token log decay.
        const Value beta = g.sigmoid(g.matmul(x, weight(L + "in_proj_b.weight")));
        HostTensor neg_a = to_f32(w(L + "A_log"), weights_->get(w(L + "A_log")));
        for (int64_t j = 0; j < neg_a.numel(); ++j) neg_a.data<float>()[j] = -std::exp(neg_a.data<float>()[j]);
        const Value decay = g.mul(g.softplus(g.add(g.matmul(x, weight(L + "in_proj_a.weight")), weight(L + "dt_bias"))),
                                  g.constant(std::move(neg_a)));

        const int slot = g.add_recurrent_slot(hv, dk, dv, "layers." + std::to_string(i));
        Value o = g.gated_delta(slot, q, k, v, decay, beta, 1e-6f);

        // Qwen3_5RMSNormGated: rms_norm(o) * w * silu(z), per value head.
        const Value z = g.reshape(g.matmul(x, weight(L + "in_proj_z.weight")), {kT, hv, dv});
        o = blocks::gated_rms_norm(g, o, weight(L + "norm.weight"), z, eps());
        return g.matmul(g.reshape(o, {kT, value_width}), weight(L + "out_proj.weight"));
    }

    // Qwen3_5Attention: q_proj yields [query | gate] per head; the output is gated by sigmoid(gate).
    Value full_attention(int64_t i, Value x, Value pos) {
        Graph::Scope s(graph_, "self_attn");
        const std::string L = "layers." + std::to_string(i) + ".self_attn.";
        blocks::AttentionSpec spec;
        spec.heads = cfg_.num_heads;
        spec.kv_heads = cfg_.num_kv_heads;
        spec.head_dim = cfg_.head_dim;
        spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
        spec.rotary_dim = cfg_.rotary_dim;
        spec.output_gate = cfg_.attn_output_gate;
        spec.kv_label = "layers." + std::to_string(i);
        blocks::AttentionParts parts;
        parts.q = proj(L + "q_proj.weight");
        parts.k = proj(L + "k_proj.weight");
        parts.v = proj(L + "v_proj.weight");
        parts.o = proj(L + "o_proj.weight");
        parts.q_norm = [this, L](Value q) { return norm(q, L + "q_norm.weight"); };
        parts.k_norm = [this, L](Value k) { return norm(k, L + "k_norm.weight"); };
        return blocks::attention(graph_, spec, parts, x, pos, inv_freq_).out;
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<SafetensorsWeights> weights_;
    std::string prefix_;
    graph::Graph graph_;
    Value inv_freq_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    if (options.vision && config_json.contains("vision_config") && !config_json.at("vision_config").is_null()) {
        log::warn("qwen3_5: the vision tower is not supported yet; loading the text decoder only");
    }

    ModelInfo info;
    info.arch = "qwen3_5";
    info.id = std::filesystem::absolute(dir).lexically_normal().filename().string();
    if (info.id.empty()) info.id = std::filesystem::absolute(dir).lexically_normal().parent_path().filename().string();
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "qwen3_5";

    std::vector<int32_t> gen_eos;
    const bool has_generation_config = std::filesystem::exists(dir / "generation_config.json");
    info.default_sampling = cfg::read_generation_defaults(dir, &gen_eos);
    if (!has_generation_config) {
        // The checkpoint ships no generation_config.json; these are the model card's recommended settings.
        info.default_sampling.temperature = 1.0f;
        info.default_sampling.top_p = 0.95f;
        info.default_sampling.top_k = 20;
    }
    const nlohmann::json& text = config_json.contains("text_config") ? config_json.at("text_config") : config_json;
    std::vector<int32_t> stops = gen_eos;
    for (const nlohmann::json* j : {&config_json, &text}) {
        if (j->contains("eos_token_id")) {
            const auto ids = cfg::get_int_list(*j, "eos_token_id", "config.json");
            stops.insert(stops.end(), ids.begin(), ids.end());
        }
    }
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);  // <|im_end|>: the end of a turn
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "qwen3_5: no end-of-sequence token in config.json or tokenizer_config.json");
    info.stop_token_ids = std::move(stops);

    auto weights = SafetensorsWeights::open_dir(dir);
    return std::make_unique<Qwen35Model>(std::move(cfg), std::move(info), std::move(weights));
}

}  // namespace xs::qwen3_5
