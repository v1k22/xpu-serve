#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "io/joined_weights.h"
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

// One decoder's graph pieces: its config, where its weights live (<prefix>layers.<i>..., <prefix>norm.weight) and its
// RoPE tables. The target's, or an assistant's.
struct Decoder {
    const Config* cfg = nullptr;
    std::string prefix;
    Value freq_sliding;
    Value freq_full;
};

// The step graph of a Gemma 4 checkpoint and, with an assistant (LoadOptions::drafter), the programs of speculative
// decoding: the target with "hidden" outputs (prefill, verify) and the assistant reading the target's KV (draft).
class Gemma4Model final : public Model, public Speculation {
public:
    // `vision`: the vision config when the encoder should be loaded (the checkpoint has one and it was asked for).
    // `assistant`: the drafter's config and weights (under "assistant." in weights()), built when `speculation`.
    Gemma4Model(Config cfg, ModelInfo info, std::unique_ptr<SafetensorsWeights> target, int32_t pad_token_id,
                const std::optional<VisionConfig>& vision, std::optional<AssistantConfig> assistant,
                std::unique_ptr<SafetensorsWeights> assistant_weights)
        : cfg_(std::move(cfg)),
          info_(std::move(info)),
          prefix_(probe_prefix(*target)),
          pad_token_id_(pad_token_id),
          assistant_(std::move(assistant)) {
        if (cfg_.ple_dim > 0) {
            ple_table_ = &target->get(w("embed_tokens_per_layer.weight"));
            XS_CHECK(ple_table_->shape == (Dims{cfg_.ple_vocab_size, cfg_.num_layers * cfg_.ple_dim}),
                     "gemma4: embed_tokens_per_layer has shape ", dims_to_string(ple_table_->shape));
            ple_scale_ = round_to_dtype(std::sqrt(static_cast<float>(cfg_.ple_dim)), cfg_.checkpoint_dtype);
        }
        embed_scale_ = round_to_dtype(std::sqrt(static_cast<float>(cfg_.hidden_size)), cfg_.checkpoint_dtype);
        if (assistant_) {
            check_assistant(*assistant_weights);
            weights_ = std::make_unique<JoinedWeights>(std::move(target), std::move(assistant_weights), "assistant.");
        } else {
            weights_ = std::move(target);
        }
        if (vision) load_vision(*vision);
        build_target(step_, Head::LastLogits);
        if (assistant_) {
            build_target(prefill_, Head::LastLogitsAllHidden);
            build_target(verify_, Head::AllLogitsAllHidden);
            build_draft(draft_);
        }
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<std::string> host_tables() const override {
        if (ple_table_ == nullptr) return {};
        return {w("embed_tokens_per_layer.weight")};
    }
    const VisionEncoder* vision() const override { return vision_.get(); }
    const Speculation* speculation() const override { return assistant_ ? this : nullptr; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "gemma4: a step needs at least one token");
        XS_CHECK(embeddings.empty() || static_cast<int64_t>(embeddings.size()) == n, "gemma4: ", embeddings.size(),
                 " embedding entries for ", n, " tokens");
        XS_CHECK(embeddings.empty() || vision_ != nullptr, "gemma4: embedding rows given, but vision is not loaded");
        std::vector<int32_t> ple_ids(tokens.begin(), tokens.end());
        token_inputs(tokens, start_pos, 1, inputs);
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
        if (ple_table_ != nullptr) inputs["ple"] = gather_ple_rows(ple_ids);
    }

    const graph::Graph& prefill_graph() const override { return prefill_; }
    const graph::Graph& verify_graph() const override { return verify_; }
    const graph::Graph& draft_graph() const override { return draft_; }
    bool drafter_keeps_state() const override { return false; }

    // Every draft of a round runs at the position of the token it starts from, the one after the hidden row's.
    void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos, int,
                       TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(static_cast<int64_t>(hidden.size()) == n * cfg_.hidden_size, "gemma4: ", n, " draft tokens need ",
                 n * cfg_.hidden_size, " hidden values, got ", hidden.size());
        token_inputs(tokens, start_pos + 1, 0, inputs);
        inputs["hidden"] = HostTensor::from_vector(DType::F32, {n, cfg_.hidden_size},
                                                   std::vector<float>(hidden.begin(), hidden.end()));
    }

private:
    enum class Head {
        LastLogits,           // the step graph
        LastLogitsAllHidden,  // prefill with speculation
        AllLogitsAllHidden,   // verify
    };

    std::string w(const std::string& suffix) const { return prefix_ + suffix; }
    Value weight(Graph& g, const std::string& name) const { return g.weight(*weights_, name); }
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }

    // token_ids and positions start_pos + i * stride.
    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, int64_t stride, TensorMap& inputs) const {
        const auto n = static_cast<int64_t>(tokens.size());
        HostTensor ids(DType::I32, {n});
        HostTensor pos(DType::I32, {n});
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) fail("gemma4: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            ids.data<int32_t>()[i] = t;
            pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i * stride);
        }
        inputs["token_ids"] = std::move(ids);
        inputs["positions"] = std::move(pos);
    }

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

    // The assistant drafts for this target: same vocabulary, its queries fit the target's KV heads, and its
    // inputs and outputs are the target's width.
    void check_assistant(const WeightStore& weights) {
        const AssistantConfig& a = *assistant_;
        const Config& t = a.text;
        XS_CHECK(a.backbone_hidden_size == cfg_.hidden_size && t.vocab_size == cfg_.vocab_size &&
                     t.head_dim == cfg_.head_dim && t.global_head_dim == cfg_.global_head_dim &&
                     t.num_kv_heads == cfg_.num_kv_heads && t.num_global_kv_heads == cfg_.num_global_kv_heads,
                 "gemma4: the assistant (hidden ", a.backbone_hidden_size, ", vocabulary ", t.vocab_size, ", head dims ",
                 t.head_dim, " / ", t.global_head_dim, ", ", t.num_kv_heads, " KV heads) does not fit this model (hidden ",
                 cfg_.hidden_size, ", vocabulary ", cfg_.vocab_size, ", head dims ", cfg_.head_dim, " / ",
                 cfg_.global_head_dim, ", ", cfg_.num_kv_heads, " KV heads)");
        if (!a.ordered_embeddings) return;
        // token_ordering lists the vocabulary cluster by cluster: row c of the table holds centroid c's tokens.
        const TensorView& order = weights.get("masked_embedding.token_ordering");
        const int64_t per = t.vocab_size / a.num_centroids;
        XS_CHECK(order.numel() == t.vocab_size && (order.dtype == DType::I64 || order.dtype == DType::I32),
                 "gemma4: assistant token_ordering is ", dtype_name(order.dtype), dims_to_string(order.shape));
        ordering_ = HostTensor(DType::I32, {a.num_centroids, per});
        for (int64_t i = 0; i < t.vocab_size; ++i) {
            const int64_t id = order.dtype == DType::I64 ? static_cast<const int64_t*>(order.data)[i]
                                                         : static_cast<const int32_t*>(order.data)[i];
            XS_CHECK(id >= 0 && id < t.vocab_size, "gemma4: assistant token_ordering holds ", id);
            ordering_.data<int32_t>()[i] = static_cast<int32_t>(id);
        }
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

    Decoder decoder(Graph& g, const Config& c, std::string prefix) const {
        return {&c, std::move(prefix), g.constant(rope_inv_freq(c.rope_sliding, c.head_dim)),
                g.constant(rope_inv_freq(c.rope_full, c.global_head_dim))};
    }

    Value lm_logits(Graph& g, Value x) const {
        const std::string head = weights_->contains("lm_head.weight") ? "lm_head.weight" : w("embed_tokens.weight");
        Value logits = g.matmul(x, weight(g, head));
        if (cfg_.final_logit_softcap > 0.0) logits = g.softcap(logits, static_cast<float>(cfg_.final_logit_softcap));
        return logits;
    }

    void build_target(Graph& g, Head head) const {
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        Value h = g.scale(g.gather(weight(g, w("embed_tokens.weight")), ids), embed_scale_);
        if (vision_ != nullptr) {
            // Image soft tokens: keep is 0 and media holds the encoder's row; elsewhere keep is 1 and media 0.
            const Value media = g.input("media", DType::F32, {kT, cfg_.hidden_size});
            const Value keep = g.input("media_keep", DType::F32, {kT, 1});
            h = g.add(g.mul(h, keep), media);
        }
        const Value per_layer = cfg_.ple_dim > 0 ? per_layer_inputs(g, h) : Value{};
        const Decoder d = decoder(g, cfg_, prefix_);
        std::vector<std::pair<Value, Value>> layer_kv(static_cast<size_t>(cfg_.num_layers));
        for (int64_t i = 0; i < cfg_.num_layers; ++i) {
            const std::pair<Value, Value>* shared = nullptr;
            if (cfg_.is_kv_shared(i)) {
                shared = &layer_kv[static_cast<size_t>(cfg_.kv_source_layer(i))];
                XS_CHECK(shared->first.valid(), "gemma4: layer ", i, " shares the KV of a layer that has none");
            }
            const LayerOut out = decoder_layer(g, d, i, h, pos, per_layer, shared);
            h = out.h;
            if (shared == nullptr) layer_kv[static_cast<size_t>(i)] = out.kv;
        }

        Graph::Scope scope(g, "head");
        if (head == Head::LastLogits) {
            g.output("logits", lm_logits(g, g.rms_norm(g.slice(h, 0, -1), weight(g, w("norm.weight")), eps())));
        } else {
            const Value hidden = g.rms_norm(h, weight(g, w("norm.weight")), eps());
            g.output("logits", lm_logits(g, head == Head::AllLogitsAllHidden ? hidden : g.slice(hidden, 0, -1)));
            g.output("hidden", hidden);
        }
        g.validate();
        log::debug("gemma4: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv slots");
    }

    // The last layer of `kind` that computes its own KV: what a KV-shared layer of that kind, or the assistant, reads.
    int64_t kv_layer(AttnKind kind) const {
        for (int64_t i = cfg_.num_layers - cfg_.num_kv_shared_layers - 1; i >= 0; --i) {
            if (cfg_.layer_kinds[static_cast<size_t>(i)] == kind) return i;
        }
        fail("gemma4: no layer of each attention kind computes its own KV");
    }

    // Gemma4AssistantForCausalLM over one (token, the target's hidden) pair, reading the target's KV.
    void build_draft(Graph& g) const {
        const AssistantConfig& a = *assistant_;
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value hidden = g.input("hidden", DType::F32, {kT, cfg_.hidden_size});
        Graph::Scope scope(g, "assistant");
        const Value e = g.scale(g.gather(weight(g, w("embed_tokens.weight")), ids), embed_scale_);
        Value x = g.matmul(g.concat({e, hidden}, 1), weight(g, "assistant.pre_projection.weight"));
        std::pair<Value, Value> target_kv[2];
        for (const AttnKind kind : {AttnKind::Sliding, AttnKind::Full}) {
            const int64_t src = kv_layer(kind);
            const int slot = g.add_kv_slot(cfg_.layer_kv_heads(src), cfg_.layer_head_dim(src),
                                           kind == AttnKind::Sliding ? cfg_.sliding_window : 0,
                                           "layers." + std::to_string(src));
            target_kv[static_cast<int>(kind)] = g.kv_read(slot);
        }
        const Decoder d = decoder(g, a.text, "assistant.model.");
        for (int64_t i = 0; i < a.text.num_layers; ++i) {
            const auto kind = static_cast<int>(a.text.layer_kinds[static_cast<size_t>(i)]);
            x = decoder_layer(g, d, i, x, pos, Value{}, &target_kv[kind]).h;
        }
        const Value last = g.rms_norm(g.slice(x, 0, -1), weight(g, "assistant.model.norm.weight"),
                                      static_cast<float>(a.text.rms_norm_eps));
        g.output("hidden", g.matmul(last, weight(g, "assistant.post_projection.weight")));
        const Value table = weight(g, weights_->contains("assistant.lm_head.weight") ? "assistant.lm_head.weight"
                                                                                     : "assistant.model.embed_tokens.weight");
        if (!a.ordered_embeddings) {
            g.output("logits", g.matmul(last, table));
        } else {
            // Gemma4AssistantMaskedEmbedder: the tokens of the top centroids, and their logits.
            const Value centroids = g.matmul(last, weight(g, "assistant.masked_embedding.centroids.weight"));
            const Value top = g.reshape(g.topk(centroids, a.centroid_top_k).second, {a.centroid_top_k});
            const Value candidates = g.reshape(g.gather(g.constant(ordering_), top), {ordering_.numel() /
                                                                                       a.num_centroids * a.centroid_top_k});
            g.output("logits", g.matmul(last, g.gather(table, candidates)));
            g.output("candidates", candidates);
        }
        g.validate();
    }

    // [T, layers, ple_dim]: (RMSNorm(W h / sqrt(hidden)) + token PLE rows) / sqrt(2).
    Value per_layer_inputs(Graph& g, Value h) const {
        Graph::Scope s(g, "ple");
        const Value ple = g.input("ple", DType::F32, {kT, cfg_.num_layers * cfg_.ple_dim});
        Value proj = g.matmul(h, weight(g, w("per_layer_model_projection.weight")));
        proj = g.scale(proj, 1.0f / std::sqrt(static_cast<float>(cfg_.hidden_size)));
        proj = g.reshape(proj, {kT, cfg_.num_layers, cfg_.ple_dim});
        proj = g.rms_norm(proj, weight(g, w("per_layer_projection_norm.weight")), eps());
        const Value tokens = g.reshape(ple, {kT, cfg_.num_layers, cfg_.ple_dim});
        return g.scale(g.add(proj, tokens), static_cast<float>(1.0 / std::sqrt(2.0)));
    }

    struct LayerOut {
        Value h;
        std::pair<Value, Value> kv;  // the k / v its attention read
    };

    // Layer i of decoder d; `shared`: the k / v it reads instead of making its own.
    LayerOut decoder_layer(Graph& g, const Decoder& d, int64_t i, Value h, Value pos, Value per_layer,
                           const std::pair<Value, Value>* shared) const {
        const Config& c = *d.cfg;
        const std::string L = d.prefix + "layers." + std::to_string(i) + ".";
        const auto wt = [&](const std::string& suffix) { return weight(g, L + suffix); };
        const auto proj = [&](const std::string& suffix) -> blocks::Linear {
            return [this, &g, name = L + suffix](Value x) { return g.matmul(x, weight(g, name)); };
        };
        const auto e = static_cast<float>(c.rms_norm_eps);
        Graph::Scope s(g, "layers." + std::to_string(i));

        const AttnKind kind = c.layer_kinds[static_cast<size_t>(i)];
        blocks::AttentionSpec spec;
        spec.heads = c.num_heads;
        spec.kv_heads = c.layer_kv_heads(i);
        spec.head_dim = c.layer_head_dim(i);
        spec.attrs.scale = 1.0f;  // q_norm / k_norm carry the scale
        spec.attrs.sliding_window = kind == AttnKind::Sliding ? c.sliding_window : 0;
        spec.kv_window = spec.attrs.sliding_window;
        spec.kv_label = "layers." + std::to_string(i);
        blocks::AttentionParts parts;
        parts.q = proj("self_attn.q_proj.weight");
        parts.k = proj("self_attn.k_proj.weight");
        if (!(c.k_eq_v && kind == AttnKind::Full)) parts.v = proj("self_attn.v_proj.weight");
        parts.o = proj("self_attn.o_proj.weight");
        parts.q_norm = [&, e](Value q) { return g.rms_norm(q, wt("self_attn.q_norm.weight"), e); };
        parts.k_norm = [&, e](Value k) { return g.rms_norm(k, wt("self_attn.k_norm.weight"), e); };
        parts.v_norm = [&g, e](Value v) { return g.rms_norm(v, e); };
        blocks::AttentionOut attn;
        {
            Graph::Scope as(g, "attn");
            attn = blocks::attention(g, spec, parts, g.rms_norm(h, wt("input_layernorm.weight"), e), pos,
                                     kind == AttnKind::Full ? d.freq_full : d.freq_sliding, shared);
        }
        h = g.add(h, g.rms_norm(attn.out, wt("post_attention_layernorm.weight"), e));

        const Value x = g.rms_norm(h, wt("pre_feedforward_layernorm.weight"), e);
        Value mlp = blocks::gated_mlp(g, x, proj("mlp.gate_proj.weight"), proj("mlp.up_proj.weight"),
                                      proj("mlp.down_proj.weight"), blocks::Act::GeluTanh);
        if (c.num_experts > 0) {
            mlp = g.add(g.rms_norm(mlp, wt("post_feedforward_layernorm_1.weight"), e),
                        g.rms_norm(experts(g, c, L, h), wt("post_feedforward_layernorm_2.weight"), e));
        }
        h = g.add(h, g.rms_norm(mlp, wt("post_feedforward_layernorm.weight"), e));

        if (per_layer.valid()) {
            Graph::Scope ps(g, "ple");
            const Value p = g.reshape(g.slice(per_layer, 1, i, i + 1), {kT, c.ple_dim});
            Value gate = g.mul(g.gelu_tanh(g.matmul(h, wt("per_layer_input_gate.weight"))), p);
            gate = g.matmul(gate, wt("per_layer_projection.weight"));
            h = g.add(h, g.rms_norm(gate, wt("post_per_layer_input_norm.weight"), e));
        }
        return {g.mul(h, wt("layer_scalar")), attn.kv};
    }

    // Gemma4TextRouter + Gemma4TextExperts over the residual h of layer L: the router reads rms_norm(h) * scale /
    // sqrt(hidden), the experts their own norm of h.
    Value experts(Graph& g, const Config& c, const std::string& L, Value h) const {
        Graph::Scope s(g, "experts");
        const auto e = static_cast<float>(c.rms_norm_eps);
        HostTensor router_scale = to_f32_tensor(L + "router.scale");
        const float root = 1.0f / std::sqrt(static_cast<float>(c.hidden_size));
        for (int64_t i = 0; i < router_scale.numel(); ++i) router_scale.data<float>()[i] *= root;
        blocks::MoeSpec spec;
        spec.experts = c.num_experts;
        spec.top_k = c.top_k_experts;
        spec.router = blocks::Router::Softmax;
        spec.normalize = true;
        spec.act = blocks::Act::GeluTanh;
        blocks::MoeParts parts;
        parts.router = weight(g, L + "router.proj.weight");
        parts.router_x = g.rms_norm(h, g.constant(std::move(router_scale)), e);
        parts.expert_scale = weight(g, L + "router.per_expert_scale");
        parts.gate_up = weight(g, L + "experts.gate_up_proj");
        parts.down = weight(g, L + "experts.down_proj");
        const int64_t inter = c.moe_intermediate_size;
        XS_CHECK(g.type(parts.gate_up).shape == (Dims{c.num_experts, 2 * inter, c.hidden_size}) &&
                     g.type(parts.down).shape == (Dims{c.num_experts, c.hidden_size, inter}),
                 "gemma4: ", L, " expert banks have shapes ", dims_to_string(g.type(parts.gate_up).shape), " and ",
                 dims_to_string(g.type(parts.down).shape));
        return blocks::moe(g, g.rms_norm(h, weight(g, L + "pre_feedforward_layernorm_2.weight"), e), spec, parts);
    }

    HostTensor to_f32_tensor(const std::string& name) const {
        const TensorView& v = weights_->get(name);
        XS_CHECK(is_float(v.dtype), "gemma4: '", name, "' has dtype ", dtype_name(v.dtype));
        HostTensor out(DType::F32, v.shape);
        to_f32(static_cast<const std::byte*>(v.data), v.dtype, out.data<float>(), static_cast<size_t>(v.numel()));
        return out;
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    std::string prefix_;
    int32_t pad_token_id_ = 0;
    std::optional<AssistantConfig> assistant_;
    HostTensor ordering_;  // the assistant's tokens per centroid, [centroids, vocab / centroids] i32
    std::unique_ptr<VisionEncoder> vision_;
    graph::Graph step_, prefill_, verify_, draft_;
    const TensorView* ple_table_ = nullptr;
    float ple_scale_ = 1.0f;
    float embed_scale_ = 1.0f;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    const nlohmann::json& text = config_json.contains("text_config") ? config_json.at("text_config") : config_json;
    const auto pad_token_id = static_cast<int32_t>(cfg::get_int_or(text, "pad_token_id", 0, "config.json"));
    std::optional<VisionConfig> vision;
    if (options.vision && cfg.bidirectional_vision) {
        log::warn("gemma4: image tokens attend both ways in this model (use_bidirectional_attention: vision), which is "
                  "not supported yet; text only");
    } else if (options.vision) {
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

    std::optional<AssistantConfig> assistant;
    std::unique_ptr<SafetensorsWeights> assistant_weights;
    if (options.speculation && !options.drafter.empty()) {
        std::ifstream in(options.drafter / "config.json");
        XS_CHECK(in.good(), "gemma4: cannot read ", (options.drafter / "config.json").string());
        assistant = parse_assistant_config(nlohmann::json::parse(in));
        assistant_weights = SafetensorsWeights::open_dir(options.drafter);
    }
    auto weights = SafetensorsWeights::open_dir(dir);
    return std::make_unique<Gemma4Model>(std::move(cfg), std::move(info), std::move(weights), pad_token_id, vision,
                                         std::move(assistant), std::move(assistant_weights));
}

}  // namespace xs::gemma4
