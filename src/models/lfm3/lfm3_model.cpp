#include <algorithm>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/lfm3/lfm3.h"
#include "models/lfm3/lfm3_decoder.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::lfm3 {
namespace {

using graph::Graph;
using graph::Value;

class Lfm3Model final : public Model {
public:
    Lfm3Model(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)), info_(std::move(info)), weights_(std::move(weights)), dec_(cfg_, *weights_) {
        build(step_);
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return *weights_; }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        XS_CHECK(embeddings.empty(), "lfm3: image input is not supported");
        dec_.token_inputs(tokens, start_pos, inputs);
    }

private:
    void build(Graph& g) const {
        const Value h = dec_.layers(g, dec_.inputs(g));
        Graph::Scope scope(g, "head");
        const Value last = dec_.final_norm(g, g.slice(h, 0, -1));
        const Value logits = g.matmul(last, dec_.embedding(g));  // the head is the embedding table
        g.output("logits", logits);
        g.output("argmax", blocks::argmax_rows(g, logits));
        g.validate();
        log::debug("lfm3: graph has ", g.nodes().size(), " nodes, ", g.kv_slots().size(), " kv and ",
                   g.conv_slots().size(), " conv slots");
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    Decoder dec_;
    graph::Graph step_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& /*options*/) {
    Config cfg = parse_config(config_json);

    ModelInfo info;
    info.arch = "lfm3";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.vocab_size;
    info.hidden_size = cfg.hidden_size;
    info.max_position = cfg.max_position;
    // The chat template writes a turn as Qwen 3.5's does: <think>...</think>, then content and <tool_call> blocks
    // of <function=NAME> with <parameter=KEY> values.
    info.dialect = "qwen3_5";

    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    const auto config_eos = cfg::get_int_list(config_json, "eos_token_id", "config.json");
    stops.insert(stops.end(), config_eos.begin(), config_eos.end());
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);  // <|im_end|>: the end of a turn
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "lfm3: no end-of-sequence token in config.json or generation_config.json");
    info.stop_token_ids = std::move(stops);

    return std::make_unique<Lfm3Model>(std::move(cfg), std::move(info), SafetensorsWeights::open_dir(dir));
}

}  // namespace xs::lfm3
