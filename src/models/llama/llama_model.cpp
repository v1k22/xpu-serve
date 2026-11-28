// Llama 3 as a language model: the embedding, each layer's attention and SwiGLU MLP (Graph::Scope "layers.<i>"),
// the final norm and the output layer (lm_head, or the embedding table when tied).

#include <algorithm>
#include <cmath>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/llama/llama.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::llama {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

class LlamaModel final : public Model {
public:
    // `batch`: build batch_graph().
    LlamaModel(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, bool batch)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)) {
        XS_CHECK(weights_->contains("model.embed_tokens.weight"), "llama: no model.embed_tokens.weight");
        head_ = cfg_.tie_word_embeddings ? "model.embed_tokens.weight" : "lm_head.weight";
        XS_CHECK(weights_->contains(head_), "llama: no output layer ", head_);
        build(step_, false);
        if (batch) build(batch_, true);
        batched_ = batch;
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }
    const graph::Graph* batch_graph() const override { return batched_ ? &batch_ : nullptr; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "llama: image input is not supported");
        const auto n = static_cast<int64_t>(tokens.size());
        XS_CHECK(n > 0, "llama: a step needs at least one token");
        HostTensor ids(DType::I32, {n});
        HostTensor pos(DType::I32, {n});
        for (int64_t i = 0; i < n; ++i) {
            const int32_t t = tokens[static_cast<size_t>(i)];
            if (t < 0 || t >= cfg_.vocab_size) {
                fail("llama: token id ", t, " outside the vocabulary (", cfg_.vocab_size, ")");
            }
            ids.data<int32_t>()[i] = t;
            pos.data<int32_t>()[i] = static_cast<int32_t>(start_pos + i);
        }
        inputs["token_ids"] = std::move(ids);
        inputs["positions"] = std::move(pos);
    }

private:
    Value weight(Graph& g, const std::string& name) const { return g.weight(*weights_, name); }
    Value norm(Graph& g, Value x, const std::string& name) const {
        return g.rms_norm(x, weight(g, name), static_cast<float>(cfg_.rms_norm_eps));
    }
    blocks::Linear proj(Graph& g, const std::string& name) const {
        return [this, &g, name](Value x) { return g.matmul(x, weight(g, name)); };
    }

    // `all_rows`: logits of every row (batch_graph), else of the last.
    void build(Graph& g, bool all_rows) const {
        if (all_rows) g.allow_segments();
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const int64_t d = cfg_.head_dim;
        const double theta = cfg_.rope_theta;
        const Value freq = g.constant(cfg_.llama3_rope ? blocks::llama3_rope_inv_freq(theta, d, *cfg_.llama3_rope)
                                                       : blocks::rope_inv_freq(theta, d, d / 2, d / 2));
        Value h = g.gather(weight(g, "model.embed_tokens.weight"), ids);
        for (int64_t i = 0; i < cfg_.num_layers; ++i) {
            const std::string label = "layers." + std::to_string(i);
            const std::string L = "model." + label + ".";
            Graph::Scope s(g, label);
            blocks::AttentionSpec spec;
            spec.heads = cfg_.num_heads;
            spec.kv_heads = cfg_.num_kv_heads;
            spec.head_dim = cfg_.head_dim;
            spec.attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg_.head_dim)));
            spec.kv_label = label;
            const std::string A = L + "self_attn.";
            const blocks::AttentionParts parts{.q = proj(g, A + "q_proj.weight"),
                                               .k = proj(g, A + "k_proj.weight"),
                                               .v = proj(g, A + "v_proj.weight"),
                                               .o = proj(g, A + "o_proj.weight")};
            h = g.add(h, blocks::attention(g, spec, parts, norm(g, h, L + "input_layernorm.weight"), pos, freq).out);
            const Value y = norm(g, h, L + "post_attention_layernorm.weight");
            h = g.add(h, blocks::gated_mlp(g, y, proj(g, L + "mlp.gate_proj.weight"), proj(g, L + "mlp.up_proj.weight"),
                                           proj(g, L + "mlp.down_proj.weight"), blocks::Act::Silu));
        }
        Graph::Scope scope(g, "head");
        const Value rows = norm(g, all_rows ? h : g.slice(h, 0, -1), "model.norm.weight");
        const Value logits = g.matmul(rows, weight(g, head_));
        g.output("logits", logits);
        g.output("argmax", blocks::argmax_rows(g, logits));
        g.validate();
        log::debug("llama: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv slots");
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    std::string head_;
    graph::Graph step_, batch_;
    bool batched_ = false;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    if (!options.drafter.empty()) fail("llama: --drafter is not supported");

    ModelInfo info;
    info.arch = "llama";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "llama3";
    // <|eot_id|> ends a turn, <|eom_id|> a message to a tool, <|end_of_text|> the text: the instruct checkpoints list
    // all three in generation_config.json and config.json.
    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    if (config_json.contains("eos_token_id") && !config_json.at("eos_token_id").is_null()) {
        const auto config_eos = cfg::get_int_list(config_json, "eos_token_id", "config.json");
        stops.insert(stops.end(), config_eos.begin(), config_eos.end());
    }
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "llama: no end-of-sequence token in config.json, generation_config.json or ",
             "tokenizer_config.json");
    info.stop_token_ids = std::move(stops);

    return std::make_unique<LlamaModel>(std::move(cfg), std::move(info), SafetensorsWeights::open_dir(dir),
                                        options.batch);
}

}  // namespace xs::llama
