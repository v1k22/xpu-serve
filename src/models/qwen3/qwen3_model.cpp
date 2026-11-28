// Qwen3 as a language model of its own (model_type "qwen3" and "qwen3_moe": Qwen3-0.6B to 32B, Qwen3-30B-A3B,
// Qwen3-235B-A22B): the decoder (qwen3_decoder.cpp), the final norm and the output layer, tied or not. Its turn is
// written in the "qwen3" dialect: <think>...</think>, then content and tool calls as JSON in <tool_call> blocks.

#include <algorithm>

#include "models/common/blocks.h"
#include "models/common/expert_banks.h"
#include "models/config_util.h"
#include "models/qwen3/qwen3.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::qwen3 {
namespace {

using graph::Graph;
using graph::Value;

class Qwen3Model final : public Model {
public:
    // `batch`: build batch_graph().
    Qwen3Model(ModelInfo info, std::unique_ptr<SafetensorsWeights> weights, Decoder decoder, bool batch)
        : info_(std::move(info)), weights_(std::move(weights)), dec_(std::move(decoder)) {
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
        XS_CHECK(embeddings.empty(), "qwen3: image input is not supported");
        dec_.token_inputs(tokens, start_pos, inputs);
    }

private:
    // `all_rows`: logits of every row (batch_graph), else of the last.
    void build(Graph& g, bool all_rows) const {
        if (all_rows) g.allow_segments();
        const Decoder::Inputs in = dec_.inputs(g);
        const Value h = dec_.layers(g, dec_.embed(g, in.ids), in.pos);
        Graph::Scope scope(g, "head");
        const Value rows = dec_.final_norm(g, all_rows ? h : g.slice(h, 0, -1));
        const Value logits = g.matmul(rows, dec_.lm_head(g));
        g.output("logits", logits);
        g.output("argmax", blocks::argmax_rows(g, logits));
        g.validate();
        log::debug("qwen3: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv slots");
    }

    ModelInfo info_;
    std::unique_ptr<SafetensorsWeights> weights_;
    Decoder dec_;
    graph::Graph step_, batch_;
    bool batched_ = false;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    const std::string type = cfg::get_string(config_json, "model_type", "config.json");
    Config cfg = parse_config(config_json, type);
    XS_CHECK(type != "qwen3_moe" || cfg.num_experts > 0, "qwen3_moe: config.json names no num_experts");
    if (!options.drafter.empty()) fail(type, ": --drafter is not supported (no drafter is published for Qwen3)");

    ModelInfo info;
    info.arch = type;
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    info.dialect = "qwen3";
    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    if (config_json.contains("eos_token_id") && !config_json.at("eos_token_id").is_null()) {
        const auto config_eos = cfg::get_int_list(config_json, "eos_token_id", "config.json");
        stops.insert(stops.end(), config_eos.begin(), config_eos.end());
    }
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);  // <|im_end|>: the end of a turn
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), type, ": no end-of-sequence token in config.json, generation_config.json or ",
             "tokenizer_config.json");
    info.stop_token_ids = std::move(stops);

    auto weights = SafetensorsWeights::open_dir(dir);
    const std::string head = cfg.tie_word_embeddings ? "" : "lm_head.weight";
    if (cfg.num_experts > 0) {
        const std::vector<std::string> experts = Decoder(cfg, *weights, "model.", head).expert_prefixes();
        join_expert_banks(*weights, experts, cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size);
    }
    Decoder decoder(cfg, *weights, "model.", head);
    return std::make_unique<Qwen3Model>(std::move(info), std::move(weights), std::move(decoder), options.batch);
}

}  // namespace xs::qwen3
