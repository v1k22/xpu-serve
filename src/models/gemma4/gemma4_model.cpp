#include <algorithm>
#include <cmath>
#include <cstring>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs::gemma4 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// Scalars that the reference multiplies in the weight dtype (embedding scales) are rounded the same way here:
// sqrt(1536) = 39.19 becomes 39.25 in bf16, and the checkpoint was trained with that value.
float round_to_dtype(float v, DType dtype) {
    if (dtype == DType::BF16) return bf16_to_f32(f32_to_bf16(v));
    if (dtype == DType::F16) return f16_to_f32(f32_to_f16(v));
    return v;
}

// One inverse frequency per rotary pair (head_dim / 2 entries). Proportional RoPE rotates only the first
// partial_rotary_factor share of the pairs; the rest stay 0 (cos 1, sin 0: identity). Both use head_dim as the
// exponent denominator, as transformers' default and proportional RoPE do.
HostTensor rope_inv_freq(const RopeConfig& rope, int64_t head_dim) {
    const int64_t pairs = head_dim / 2;
    const auto rotated = rope.proportional
                             ? static_cast<int64_t>(std::floor(rope.partial_rotary_factor * static_cast<double>(head_dim) / 2.0))
                             : pairs;
    XS_CHECK(rotated >= 0 && rotated <= pairs, "gemma4: rotary pairs ", rotated, " out of range for head_dim ", head_dim);
    return blocks::rope_inv_freq(rope.theta, head_dim, rotated, pairs);
}

// Checkpoints of Gemma4ForConditionalGeneration keep the decoder under "model.language_model.", text-only ones
// under "model.".
std::string probe_prefix(const WeightStore& weights) {
    for (const char* prefix : {"model.language_model.", "model."}) {
        if (weights.contains(std::string(prefix) + "embed_tokens.weight")) return prefix;
    }
    fail("gemma4: no embed_tokens.weight under model.language_model. or model.; is this a Gemma 4 checkpoint?");
}

class Gemma4Model final : public Model {
public:
    // `vision`: the vision config when the encoder should be loaded (the checkpoint has one and it was asked for).
    Gemma4Model(Config cfg, ModelInfo info, std::unique_ptr<SafetensorsWeights> weights, int32_t pad_token_id,
                const std::optional<VisionConfig>& vision)
        : cfg_(std::move(cfg)),
          info_(std::move(info)),
          weights_(std::move(weights)),
          prefix_(probe_prefix(*weights_)),
          pad_token_id_(pad_token_id) {
        if (cfg_.ple_dim > 0) {
            ple_table_ = &weights_->get(w("embed_tokens_per_layer.weight"));
            XS_CHECK(ple_table_->shape == (Dims{cfg_.ple_vocab_size, cfg_.num_layers * cfg_.ple_dim}),
                     "gemma4: embed_tokens_per_layer has shape ", dims_to_string(ple_table_->shape));
            ple_scale_ = round_to_dtype(std::sqrt(static_cast<float>(cfg_.ple_dim)), cfg_.checkpoint_dtype);
        }
        if (vision) load_vision(*vision);
        build_graph();
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return graph_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<std::string> host_tables() const override {
        if (ple_table_ == nullptr) return {};
        return {w("embed_tokens_per_layer.weight")};
    }

    const VisionEncoder* vision() const override { return vision_.get(); }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "gemma4: a step needs at least one token");
        XS_CHECK(embeddings.empty() || static_cast<int64_t>(embeddings.size()) == n, "gemma4: ", embeddings.size(),
                 " embedding entries for ", n, " tokens");
        XS_CHECK(embeddings.empty() || vision_ != nullptr, "gemma4: embedding rows given, but vision is not loaded");
        HostTensor ids(DType::I32, {n});
        HostTensor pos(DType::I32, {n});
        std::vector<int32_t> ple_ids(tokens.begin(), tokens.end());
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) fail("gemma4: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            ids.data<int32_t>()[i] = t;
            pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
        }
        if (vision_ != nullptr) {
            // Rows replace the scaled token embedding; their per-layer embedding reads the pad token, as the
            // reference does for every multimodal position.
            const int64_t width = cfg_.hidden_size;
            HostTensor media(DType::F32, {n, width});
            HostTensor keep(DType::F32, {n, 1});
            for (int64_t i = 0; i < n; ++i) {
                const float* row = embeddings.empty() ? nullptr : embeddings[static_cast<size_t>(i)];
                keep.data<float>()[i] = row == nullptr ? 1.0f : 0.0f;
                if (row == nullptr) continue;
                std::copy(row, row + width, media.data<float>() + i * width);
                ple_ids[static_cast<size_t>(i)] = pad_token_id_;
            }
            inputs["media"] = std::move(media);
            inputs["media_keep"] = std::move(keep);
        }
        inputs["token_ids"] = std::move(ids);
        inputs["positions"] = std::move(pos);
        if (ple_table_ != nullptr) inputs["ple"] = gather_ple_rows(ple_ids);
    }

private:
    std::string w(const std::string& suffix) const { return prefix_ + suffix; }
    Value weight(const std::string& suffix) { return graph_.weight(*weights_, w(suffix)); }
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }

    // Checkpoints of Gemma4ForConditionalGeneration keep the towers next to the decoder: model.vision_tower.* and
    // model.embed_vision.* beside model.language_model.*.
    void load_vision(const VisionConfig& vc) {
        const std::string vision_prefix = prefix_ == "model.language_model." ? "model." : prefix_;
        if (!weights_->contains(vision_prefix + "vision_tower.patch_embedder.input_proj.weight")) {
            log::warn("gemma4: config.json has a vision_config but the checkpoint has no vision weights; text only");
            return;
        }
        vision_ = make_vision_encoder(vc, *weights_, vision_prefix);
    }

    // The token-identity half of the per-layer embeddings is a 262144 x (layers * 256) table (4.7 GB in bf16). Rows
    // are gathered on the host so the table never has to live in device memory.
    HostTensor gather_ple_rows(std::span<const int32_t> tokens) const {
        const int64_t width = cfg_.num_layers * cfg_.ple_dim;
        HostTensor out(DType::F32, {static_cast<int64_t>(tokens.size()), width});
        for (size_t i = 0; i < tokens.size(); ++i) {
            XS_CHECK(tokens[i] < cfg_.ple_vocab_size, "gemma4: token id ", tokens[i], " outside the PLE vocabulary");
            float* dst = out.data<float>() + static_cast<int64_t>(i) * width;
            if (is_quantized(ple_table_->dtype)) {  // a pack's q8 / q4 table
                quant::dequantize_rows(*ple_table_, tokens[i], 1, dst);
            } else {
                const size_t row_bytes = static_cast<size_t>(width) * dtype_size(ple_table_->dtype);
                const auto* row = static_cast<const std::byte*>(ple_table_->data) + static_cast<size_t>(tokens[i]) * row_bytes;
                to_f32(row, ple_table_->dtype, dst, static_cast<size_t>(width));
            }
            for (int64_t j = 0; j < width; ++j) dst[j] *= ple_scale_;
        }
        return out;
    }

    static void to_f32(const std::byte* src, DType dtype, float* dst, size_t n) {
        switch (dtype) {
            case DType::BF16:
                for (size_t j = 0; j < n; ++j) {
                    uint16_t v;
                    std::memcpy(&v, src + 2 * j, 2);
                    dst[j] = bf16_to_f32(v);
                }
                return;
            case DType::F16:
                for (size_t j = 0; j < n; ++j) {
                    uint16_t v;
                    std::memcpy(&v, src + 2 * j, 2);
                    dst[j] = f16_to_f32(v);
                }
                return;
            case DType::F32: std::memcpy(dst, src, n * sizeof(float)); return;
            default: fail("gemma4: per-layer embedding table has unsupported dtype ", dtype_name(dtype));
        }
    }

    void build_graph() {
        Graph& g = graph_;
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value embed = weight("embed_tokens.weight");
        const float embed_scale = round_to_dtype(std::sqrt(static_cast<float>(cfg_.hidden_size)), cfg_.checkpoint_dtype);
        Value h = g.scale(g.gather(embed, ids), embed_scale);
        if (vision_ != nullptr) {
            // Image soft tokens: keep is 0 and media holds the encoder's row; elsewhere keep is 1 and media 0.
            const Value media = g.input("media", DType::F32, {kT, cfg_.hidden_size});
            const Value keep = g.input("media_keep", DType::F32, {kT, 1});
            h = g.add(g.mul(h, keep), media);
        }

        const Value per_layer = cfg_.ple_dim > 0 ? per_layer_inputs(h) : Value{};
        inv_freq_sliding_ = g.constant(rope_inv_freq(cfg_.rope_sliding, cfg_.head_dim));
        inv_freq_full_ = g.constant(rope_inv_freq(cfg_.rope_full, cfg_.global_head_dim));
        layer_kv_.assign(static_cast<size_t>(cfg_.num_layers), {});

        for (int64_t i = 0; i < cfg_.num_layers; ++i) h = decoder_layer(i, h, pos, per_layer);

        Graph::Scope head(g, "head");
        Value last = g.rms_norm(g.slice(h, 0, -1), weight("norm.weight"), eps());
        const Value lm_head = weights_->contains("lm_head.weight") ? g.weight(*weights_, "lm_head.weight") : embed;
        Value logits = g.matmul(last, lm_head);
        if (cfg_.final_logit_softcap > 0.0) logits = g.softcap(logits, static_cast<float>(cfg_.final_logit_softcap));
        g.output("logits", logits);
        g.validate();
        log::debug("gemma4: step graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv slots");
    }

    // [T, layers, ple_dim]: (RMSNorm(W h / sqrt(hidden)) + token PLE rows) / sqrt(2).
    Value per_layer_inputs(Value h) {
        Graph& g = graph_;
        Graph::Scope s(g, "ple");
        const Value ple = g.input("ple", DType::F32, {kT, cfg_.num_layers * cfg_.ple_dim});
        Value proj = g.matmul(h, weight("per_layer_model_projection.weight"));
        proj = g.scale(proj, 1.0f / std::sqrt(static_cast<float>(cfg_.hidden_size)));
        proj = g.reshape(proj, {kT, cfg_.num_layers, cfg_.ple_dim});
        proj = g.rms_norm(proj, weight("per_layer_projection_norm.weight"), eps());
        const Value tokens = g.reshape(ple, {kT, cfg_.num_layers, cfg_.ple_dim});
        return g.scale(g.add(proj, tokens), static_cast<float>(1.0 / std::sqrt(2.0)));
    }

    Value decoder_layer(int64_t i, Value h, Value pos, Value per_layer) {
        Graph& g = graph_;
        const std::string L = "layers." + std::to_string(i) + ".";
        Graph::Scope s(g, "layers." + std::to_string(i));

        Value attn = attention(i, g.rms_norm(h, weight(L + "input_layernorm.weight"), eps()), pos);
        h = g.add(h, g.rms_norm(attn, weight(L + "post_attention_layernorm.weight"), eps()));

        const Value x = g.rms_norm(h, weight(L + "pre_feedforward_layernorm.weight"), eps());
        const Value mlp = blocks::gated_mlp(g, x, proj(L + "mlp.gate_proj.weight"), proj(L + "mlp.up_proj.weight"),
                                            proj(L + "mlp.down_proj.weight"), blocks::Act::GeluTanh);
        h = g.add(h, g.rms_norm(mlp, weight(L + "post_feedforward_layernorm.weight"), eps()));

        if (per_layer.valid()) {
            Graph::Scope ps(g, "ple");
            const Value p = g.reshape(g.slice(per_layer, 1, i, i + 1), {kT, cfg_.ple_dim});
            Value gate = g.mul(g.gelu_tanh(g.matmul(h, weight(L + "per_layer_input_gate.weight"))), p);
            gate = g.matmul(gate, weight(L + "per_layer_projection.weight"));
            h = g.add(h, g.rms_norm(gate, weight(L + "post_per_layer_input_norm.weight"), eps()));
        }
        return g.mul(h, weight(L + "layer_scalar"));
    }

    // A projection whose weight node is made when a block uses it (KV-shared layers never load k / v).
    blocks::Linear proj(const std::string& suffix) {
        return [this, suffix](Value x) { return graph_.matmul(x, weight(suffix)); };
    }

    Value attention(int64_t i, Value x, Value pos) {
        Graph::Scope s(graph_, "attn");
        const std::string L = "layers." + std::to_string(i) + ".self_attn.";
        const AttnKind kind = cfg_.layer_kinds[static_cast<size_t>(i)];
        blocks::AttentionSpec spec;
        spec.heads = cfg_.num_heads;
        spec.kv_heads = cfg_.num_kv_heads;
        spec.head_dim = cfg_.layer_head_dim(i);
        spec.attrs.scale = 1.0f;  // q_norm / k_norm carry the scale
        spec.attrs.sliding_window = kind == AttnKind::Sliding ? cfg_.sliding_window : 0;
        spec.kv_window = spec.attrs.sliding_window;
        spec.kv_label = "layers." + std::to_string(i);
        blocks::AttentionParts parts;
        parts.q = proj(L + "q_proj.weight");
        parts.k = proj(L + "k_proj.weight");
        parts.v = proj(L + "v_proj.weight");
        parts.o = proj(L + "o_proj.weight");
        parts.q_norm = [this, L](Value q) { return graph_.rms_norm(q, weight(L + "q_norm.weight"), eps()); };
        parts.k_norm = [this, L](Value k) { return graph_.rms_norm(k, weight(L + "k_norm.weight"), eps()); };
        parts.v_norm = [this](Value v) { return graph_.rms_norm(v, eps()); };
        const Value inv_freq = kind == AttnKind::Full ? inv_freq_full_ : inv_freq_sliding_;
        const std::pair<Value, Value>* shared = nullptr;
        if (cfg_.is_kv_shared(i)) {
            shared = &layer_kv_[static_cast<size_t>(cfg_.kv_source_layer(i))];
            XS_CHECK(shared->first.valid(), "gemma4: layer ", i, " shares the KV of a layer that has none");
        }
        const blocks::AttentionOut r = blocks::attention(graph_, spec, parts, x, pos, inv_freq, shared);
        if (shared == nullptr) layer_kv_[static_cast<size_t>(i)] = r.kv;
        return r.out;
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<SafetensorsWeights> weights_;
    std::string prefix_;
    int32_t pad_token_id_ = 0;
    std::unique_ptr<VisionEncoder> vision_;
    graph::Graph graph_;
    const TensorView* ple_table_ = nullptr;
    float ple_scale_ = 1.0f;
    Value inv_freq_sliding_;
    Value inv_freq_full_;
    std::vector<std::pair<Value, Value>> layer_kv_;  // KvCache outputs of each non-shared layer
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    const nlohmann::json& text = config_json.contains("text_config") ? config_json.at("text_config") : config_json;
    const auto pad_token_id = static_cast<int32_t>(cfg::get_int_or(text, "pad_token_id", 0, "config.json"));
    std::optional<VisionConfig> vision;
    if (options.vision) {
        try {
            vision = parse_vision_config(config_json, dir, cfg.hidden_size);
        } catch (const Error& e) {
            log::warn("gemma4: vision disabled, text only: ", e.what());
        }
    }

    ModelInfo info;
    info.arch = "gemma4";
    info.id = std::filesystem::absolute(dir).lexically_normal().filename().string();
    if (info.id.empty()) info.id = std::filesystem::absolute(dir).lexically_normal().parent_path().filename().string();
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "gemma4";
    std::vector<int32_t> gen_eos;
    info.default_sampling = cfg::read_generation_defaults(dir, &gen_eos);
    std::vector<int32_t> stops = cfg::get_int_list(config_json, "eos_token_id", "config.json");
    if (config_json.contains("text_config")) {
        const auto text_eos = cfg::get_int_list(config_json.at("text_config"), "eos_token_id", "config.json text_config");
        stops.insert(stops.end(), text_eos.begin(), text_eos.end());
    }
    stops.insert(stops.end(), gen_eos.begin(), gen_eos.end());
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    info.stop_token_ids = std::move(stops);

    auto weights = SafetensorsWeights::open_dir(dir);
    return std::make_unique<Gemma4Model>(std::move(cfg), std::move(info), std::move(weights), pad_token_id, vision);
}

}  // namespace xs::gemma4
