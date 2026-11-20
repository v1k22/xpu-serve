#include <algorithm>

#include "models/common/blocks.h"
#include "models/common/expert_banks.h"
#include "models/config_util.h"
#include "models/qwen3_omni/qwen3_omni.h"
#include "util/error.h"

namespace xs::qwen3_omni {
namespace {

using graph::Graph;
using graph::Value;

constexpr const char* kDecoder = "thinker.model.";

class Qwen3OmniModel final : public Model {
public:
    Qwen3OmniModel(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, audio::MelParams mel)
        : cfg_(std::move(cfg)),
          info_(std::move(info)),
          weights_(std::move(weights)),
          decoder_(cfg_.text, *weights_, kDecoder, "thinker.lm_head.weight"),
          audio_(qwen3_audio::make_encoder(
              cfg_.audio, *weights_, mel,
              {"thinker.audio_tower.", "thinker.audio_tower.proj1", "thinker.audio_tower.proj2"}, cfg_.audio_token)) {
        const qwen3::Decoder::Inputs in = decoder_.inputs(graph_);
        Value h = blocks::splice_media(graph_, decoder_.embed(graph_, in.ids), cfg_.text.hidden_size);
        h = decoder_.layers(graph_, h, in.pos);
        Graph::Scope scope(graph_, "head");
        const Value last = decoder_.final_norm(graph_, graph_.slice(h, 0, -1));
        const Value logits = graph_.matmul(last, decoder_.lm_head(graph_));
        graph_.output("logits", logits);
        graph_.output("argmax", blocks::argmax_rows(graph_, logits));
        graph_.validate();
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return graph_; }
    const WeightStore& weights() const override { return *weights_; }
    const AudioEncoder* audio() const override { return audio_.get(); }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        decoder_.token_inputs(tokens, start_pos, inputs);
        blocks::media_inputs(embeddings, static_cast<int64_t>(tokens.size()), cfg_.text.hidden_size, inputs);
    }

private:
    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    qwen3::Decoder decoder_;
    std::unique_ptr<AudioEncoder> audio_;
    Graph graph_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    (void)options;  // no drafter, no vision yet
    Config cfg = parse_config(config_json);
    ModelInfo info;
    info.arch = "qwen3_omni_moe";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.text.vocab_size;
    info.hidden_size = cfg.text.hidden_size;
    info.max_position = cfg.text.max_position;
    info.dialect = "qwen3";
    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "qwen3_omni_moe: no end-of-sequence token in generation_config.json or ",
             "tokenizer_config.json");
    info.stop_token_ids = std::move(stops);

    auto weights = SafetensorsWeights::open_dir(dir);
    const std::vector<std::string> experts = qwen3::Decoder(cfg.text, *weights, kDecoder, "").expert_prefixes();
    join_expert_banks(*weights, experts, cfg.text.num_experts, cfg.text.moe_intermediate_size, cfg.text.hidden_size);
    // transformers' processor pads no clip: a clip under the ASR package's half second is read as it is.
    audio::MelParams mel = qwen3_audio::mel_spec(dir);
    mel.min_length = 0;
    return std::make_unique<Qwen3OmniModel>(std::move(cfg), std::move(info), std::move(weights), mel);
}

}  // namespace xs::qwen3_omni
