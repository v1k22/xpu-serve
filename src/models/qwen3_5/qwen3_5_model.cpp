#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>

#include "models/common/blocks.h"
#include "io/joined_weights.h"
#include "models/config_util.h"
#include "models/drafters/dflash.h"
#include "models/qwen3_5/qwen3_5.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"
#include "util/timer.h"

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

// Where a block drafter's weights sit beside the target's.
constexpr const char* kDrafterPrefix = "drafter.";

// The programs of a Qwen3.5 checkpoint, built from one set of layer builders: the step graph, and with speculation the
// prefill and verify programs (the same decoder, other outputs) and the drafter's. The drafter is the checkpoint's MTP
// head (vLLM's Qwen3.5 MTP: one decoder layer of the target's kind over fc(cat(norm(embed(t_{i+1})), norm(h_i))),
// then the target's lm_head), or, with a drafter checkpoint beside the model, a block drafter of the DFlash family
// (models/drafters/dflash.h), whose context the prefill and verify programs append.
class Qwen35Model final : public Model, public Speculation, public ChainDrafting {
public:
    // `drafter`: a block drafter's config, its weights already in `weights` under kDrafterPrefix.
    Qwen35Model(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, const LoadOptions& options,
                std::optional<dflash::Config> drafter)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)), prefix_(probe_prefix(*weights_)) {
        build_target(step_, Head::LastLogits);
        if (!options.speculation) return;
        if (drafter) {
            XS_CHECK(drafter->hidden_size == cfg_.hidden_size && drafter->vocab_size == cfg_.vocab_size &&
                         drafter->num_target_layers == cfg_.num_layers,
                     "qwen3_5: the drafter was trained for a target of ", drafter->num_target_layers, " layers of ",
                     drafter->hidden_size, " and ", drafter->vocab_size, " tokens; this model has ", cfg_.num_layers,
                     " of ", cfg_.hidden_size, " and ", cfg_.vocab_size);
            block_ = std::make_unique<dflash::Drafter>(std::move(*drafter), *weights_, kDrafterPrefix, options.drafts);
            build_target(prefill_, Head::LastLogitsAllHidden);
            build_target(verify_, Head::AllLogitsAllHidden);
            block_->build_block(draft_, weight(draft_, prefix_ + "embed_tokens.weight"), lm_head(draft_));
            return;
        }
        if (cfg_.mtp_layers == 0 || !weights_->contains("mtp.fc.weight")) {
            log::warn("qwen3_5: the checkpoint has no MTP head (mtp.*); speculative decoding is unavailable");
            return;
        }
        if (cfg_.mtp_layers != 1) fail("qwen3_5: an MTP head of ", cfg_.mtp_layers, " layers is not supported (1 is)");
        build_target(prefill_, Head::LastLogitsAllHidden);
        build_target(verify_, Head::AllLogitsAllHidden);
        build_mtp(draft_, options.draft_shortlist);
        has_mtp_ = true;
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }
    const Speculation* speculation() const override { return has_mtp_ || block_ ? this : nullptr; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "qwen3_5: image input is not supported yet");
        token_inputs(tokens, start_pos, inputs);
    }

    const graph::Graph& prefill_graph() const override { return prefill_; }
    const graph::Graph& verify_graph() const override { return verify_; }
    const graph::Graph& draft_graph() const override { return draft_; }
    const ChainDrafting* chain() const override { return block_ ? nullptr : this; }
    const BlockDrafting* block() const override { return block_.get(); }
    std::string_view drafter_name() const override { return block_ ? dflash::variant_name(block_->config().variant) : "mtp"; }
    bool drafter_keeps_state() const override { return true; }

    // The pair (h_i, token i + 1) runs at position i, a chained draft one position after the previous one.
    void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos, int depth,
                       TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(static_cast<int64_t>(hidden.size()) == n * cfg_.hidden_size, "qwen3_5: ", n, " MTP tokens need ",
                 n * cfg_.hidden_size, " hidden values, got ", hidden.size());
        token_inputs(tokens, start_pos + depth, inputs);
        inputs["hidden"] = HostTensor::from_vector(DType::F32, {n, cfg_.hidden_size},
                                                   std::vector<float>(hidden.begin(), hidden.end()));
    }

private:
    enum class Head {
        LastLogits,           // the step graph
        LastLogitsAllHidden,  // prefill with speculation
        AllLogitsAllHidden,   // verify
    };

    void token_inputs(std::span<const int32_t> tokens, int64_t start_pos, TensorMap& inputs) const {
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "qwen3_5: a step needs at least one token");
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

    Value weight(Graph& g, const std::string& name) const { return g.weight(*weights_, name); }
    float eps() const { return static_cast<float>(cfg_.rms_norm_eps); }

    // Qwen3_5RMSNorm is zero-centered: x * rsqrt(mean(x^2) + eps) * (1 + w).
    Value norm(Graph& g, Value x, const std::string& name) const {
        return blocks::rms_norm_one_plus(g, x, *weights_, name, eps());
    }
    // A projection whose weight node is made when the block uses it.
    blocks::Linear proj(Graph& g, const std::string& name) const {
        return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
    }
    std::string lm_head_name() const {
        return weights_->contains("lm_head.weight") ? "lm_head.weight" : prefix_ + "embed_tokens.weight";
    }
    Value lm_head(Graph& g) const { return weight(g, lm_head_name()); }
    Value inv_freq(Graph& g) const {
        return g.constant(
            blocks::rope_inv_freq(cfg_.rope_theta, cfg_.rotary_dim, cfg_.rotary_dim / 2, cfg_.rotary_dim / 2));
    }

    void build_target(Graph& g, Head head) const {
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        Value h = g.gather(weight(g, prefix_ + "embed_tokens.weight"), ids);
        const Value freq = inv_freq(g);
        std::vector<Value> after;  // the state after each layer
        for (int64_t i = 0; i < cfg_.num_layers; ++i) {
            const std::string layer = "layers." + std::to_string(i);
            h = decoder_layer(g, prefix_ + layer + ".", layer, cfg_.layer_kinds[static_cast<size_t>(i)], h, pos, freq);
            after.push_back(h);
        }
        if (head == Head::LastLogits) {
            Graph::Scope scope(g, "head");
            const Value logits = g.matmul(norm(g, g.slice(h, 0, -1), prefix_ + "norm.weight"), lm_head(g));
            g.output("logits", logits);
            g.output("argmax", blocks::argmax_rows(g, logits));
        } else {
            const Value hidden = norm(g, h, prefix_ + "norm.weight");
            if (block_ != nullptr) {
                // A block drafter's context: Hugging Face's hidden_states after the layers it names, the last of which
                // is the normed one.
                std::vector<Value> states;
                for (const int64_t id : block_->config().target_layer_ids) {
                    states.push_back(id + 1 < cfg_.num_layers ? after[static_cast<size_t>(id)] : hidden);
                }
                block_->append_context(g, states, pos);
            }
            Graph::Scope scope(g, "head");
            const Value logits =
                g.matmul(head == Head::AllLogitsAllHidden ? hidden : g.slice(hidden, 0, -1), lm_head(g));
            g.output("logits", logits);
            g.output("argmax", blocks::argmax_rows(g, logits));
            if (block_ == nullptr) g.output("hidden", hidden);  // what the MTP head takes
        }
        g.validate();
        log::debug("qwen3_5: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv, ",
                   g.conv_slots().size(), " conv and ", g.recurrent_slots().size(), " recurrent slots");
    }

    // `shortlist` > 0: the drafts pick among that many tokens (blocks::shortlist_head) instead of reading the whole
    // head, which is most of a draft step.
    void build_mtp(Graph& g, int64_t shortlist) const {
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value hidden = g.input("hidden", DType::F32, {kT, cfg_.hidden_size});
        Graph::Scope scope(g, "mtp");
        const Value e = norm(g, g.gather(weight(g, prefix_ + "embed_tokens.weight"), ids),
                             "mtp.pre_fc_norm_embedding.weight");
        const Value x = g.matmul(g.concat({e, norm(g, hidden, "mtp.pre_fc_norm_hidden.weight")}, 1),
                                 weight(g, "mtp.fc.weight"));
        const Value h = decoder_layer(g, "mtp.layers.0.", "mtp.layers.0", LayerKind::Full, x, pos, inv_freq(g));
        const Value last = norm(g, g.slice(h, 0, -1), "mtp.norm.weight");
        if (shortlist > 0 && shortlist < cfg_.vocab_size && cfg_.hidden_size % 32 == 0) {
            Timer timer;
            const quant::SignSketch sketch = quant::sign_sketch(weights_->get(lm_head_name()));
            log::info("qwen3_5: drafts pick among ", shortlist, " of ", cfg_.vocab_size, " tokens (a one-bit sketch of ",
                      "the head, ", sketch.signs.nbytes() >> 20, " MiB, built in ", static_cast<int>(timer.elapsed_ms()),
                      " ms)");
            const blocks::Shortlist s = blocks::shortlist_head(g, last, lm_head(g), sketch, shortlist);
            g.output("logits", s.logits);
            g.output("candidates", s.candidates);
        } else {
            g.output("logits", g.matmul(last, lm_head(g)));
        }
        g.output("hidden", last);
        g.validate();
    }

    // One decoder layer from the weights under `L` (".../layers.<i>."); `label` names its state slots.
    Value decoder_layer(Graph& g, const std::string& L, const std::string& label, LayerKind kind, Value h, Value pos,
                        Value freq) const {
        Graph::Scope s(g, label);
        const Value x = norm(g, h, L + "input_layernorm.weight");
        const Value mixed = kind == LayerKind::Linear ? linear_attention(g, L, label, x)
                                                      : full_attention(g, L, label, x, pos, freq);
        h = g.add(h, mixed);
        const Value y = norm(g, h, L + "post_attention_layernorm.weight");
        if (cfg_.num_experts > 0) return g.add(h, moe(g, L + "mlp.", y));
        return g.add(h, blocks::gated_mlp(g, y, proj(g, L + "mlp.gate_proj.weight"), proj(g, L + "mlp.up_proj.weight"),
                                          proj(g, L + "mlp.down_proj.weight"), blocks::Act::Silu));
    }

    // Qwen3_5MoeSparseMoeBlock.
    Value moe(Graph& g, const std::string& L, Value y) const {
        blocks::MoeSpec spec;
        spec.experts = cfg_.num_experts;
        spec.top_k = cfg_.experts_per_token;
        spec.router = blocks::Router::Softmax;
        spec.normalize = cfg_.norm_topk_prob;
        blocks::MoeParts parts;
        parts.router = weight(g, L + "gate.weight");
        if (!weights_->contains(L + "experts.gate_up_proj")) {
            fail("qwen3_5: ", L, "experts.gate_up_proj is missing; per-expert tensors (experts.<e>.gate_proj) are not ",
                 "supported yet, only the fused expert banks the released checkpoints ship");
        }
        parts.gate_up = weight(g, L + "experts.gate_up_proj");
        parts.down = weight(g, L + "experts.down_proj");
        const int64_t e = cfg_.num_experts, h = cfg_.hidden_size, inter = cfg_.moe_intermediate_size;
        XS_CHECK(g.type(parts.gate_up).shape == (Dims{e, 2 * inter, h}) && g.type(parts.down).shape == (Dims{e, h, inter}),
                 "qwen3_5: ", L, " expert banks have shapes ", dims_to_string(g.type(parts.gate_up).shape), " and ",
                 dims_to_string(g.type(parts.down).shape), ", expected [", e, ", ", 2 * inter, ", ", h, "] and [", e,
                 ", ", h, ", ", inter, "]");
        if (cfg_.shared_expert_intermediate_size > 0) {
            parts.shared_gate = proj(g, L + "shared_expert.gate_proj.weight");
            parts.shared_up = proj(g, L + "shared_expert.up_proj.weight");
            parts.shared_down = proj(g, L + "shared_expert.down_proj.weight");
            parts.shared_expert_gate = weight(g, L + "shared_expert_gate.weight");
        }
        return blocks::moe(g, y, spec, parts);
    }

    // Qwen3_5GatedDeltaNet.
    Value linear_attention(Graph& g, const std::string& layer, const std::string& label, Value x) const {
        Graph::Scope s(g, "linear_attn");
        const std::string L = layer + "linear_attn.";
        const int64_t hk = cfg_.linear_key_heads, hv = cfg_.linear_value_heads;
        const int64_t dk = cfg_.linear_key_dim, dv = cfg_.linear_value_dim;
        const int64_t key_width = hk * dk, value_width = hv * dv;
        const int64_t conv_width = 2 * key_width + value_width;

        // q, k and v share one projection and one causal conv (with SiLU), then split.
        const Value conv_w = weight(g, L + "conv1d.weight");
        XS_CHECK(g.type(conv_w).shape == (Dims{conv_width, 1, cfg_.conv_kernel}), "qwen3_5: ", L, "conv1d.weight has ",
                 "shape ", dims_to_string(g.type(conv_w).shape));
        const int conv_slot = g.add_conv_slot(conv_width, cfg_.conv_kernel, label);
        const Value qkv = g.silu(g.causal_conv(conv_slot, g.matmul(x, weight(g, L + "in_proj_qkv.weight")),
                                                g.reshape(conv_w, {conv_width, cfg_.conv_kernel})));
        const Value q = g.reshape(g.slice(qkv, 1, 0, key_width), {kT, hk, dk});
        const Value k = g.reshape(g.slice(qkv, 1, key_width, 2 * key_width), {kT, hk, dk});
        const Value v = g.reshape(g.slice(qkv, 1, 2 * key_width, conv_width), {kT, hv, dv});

        // beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), the per-token log decay.
        const Value beta = g.sigmoid(g.matmul(x, weight(g, L + "in_proj_b.weight")));
        HostTensor neg_a = to_f32(L + "A_log", weights_->get(L + "A_log"));
        for (int64_t j = 0; j < neg_a.numel(); ++j) neg_a.data<float>()[j] = -std::exp(neg_a.data<float>()[j]);
        const Value decay = g.mul(g.softplus(g.add(g.matmul(x, weight(g, L + "in_proj_a.weight")), weight(g, L + "dt_bias"))),
                                  g.constant(std::move(neg_a)));

        const int slot = g.add_recurrent_slot(hv, dk, dv, label);
        Value o = g.gated_delta(slot, q, k, v, decay, beta, 1e-6f);

        // Qwen3_5RMSNormGated: rms_norm(o) * w * silu(z), per value head.
        const Value z = g.reshape(g.matmul(x, weight(g, L + "in_proj_z.weight")), {kT, hv, dv});
        o = blocks::gated_rms_norm(g, o, weight(g, L + "norm.weight"), z, eps());
        return g.matmul(g.reshape(o, {kT, value_width}), weight(g, L + "out_proj.weight"));
    }

    // Qwen3_5Attention: q_proj yields [query | gate] per head; the output is gated by sigmoid(gate).
    Value full_attention(Graph& g, const std::string& layer, const std::string& label, Value x, Value pos,
                         Value freq) const {
        Graph::Scope s(g, "self_attn");
        const std::string L = layer + "self_attn.";
        blocks::AttentionSpec spec;
        spec.heads = cfg_.num_heads;
        spec.kv_heads = cfg_.num_kv_heads;
        spec.head_dim = cfg_.head_dim;
        spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
        spec.rotary_dim = cfg_.rotary_dim;
        spec.output_gate = cfg_.attn_output_gate;
        spec.kv_label = label;
        blocks::AttentionParts parts;
        parts.q = proj(g, L + "q_proj.weight");
        parts.k = proj(g, L + "k_proj.weight");
        parts.v = proj(g, L + "v_proj.weight");
        parts.o = proj(g, L + "o_proj.weight");
        parts.q_norm = [this, &g, L](Value q) { return norm(g, q, L + "q_norm.weight"); };
        parts.k_norm = [this, &g, L](Value k) { return norm(g, k, L + "k_norm.weight"); };
        return blocks::attention(g, spec, parts, x, pos, freq).out;
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    std::string prefix_;
    graph::Graph step_, prefill_, verify_, draft_;
    bool has_mtp_ = false;
    std::unique_ptr<dflash::Drafter> block_;  // a block drafter instead of the MTP head
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

    std::unique_ptr<WeightStore> weights = SafetensorsWeights::open_dir(dir);
    std::optional<dflash::Config> drafter;
    if (options.speculation && !options.drafter.empty()) {
        std::ifstream in(options.drafter / "config.json");
        XS_CHECK(in.good(), "qwen3_5: cannot read ", (options.drafter / "config.json").string());
        drafter = dflash::parse_config(nlohmann::json::parse(in));
        weights = std::make_unique<JoinedWeights>(std::move(weights), dflash::open_weights(options.drafter), kDrafterPrefix);
    }
    return std::make_unique<Qwen35Model>(std::move(cfg), std::move(info), std::move(weights), options, std::move(drafter));
}

}  // namespace xs::qwen3_5
