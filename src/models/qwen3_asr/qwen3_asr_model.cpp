#include <algorithm>

#include "models/common/blocks.h"
#include "models/config_util.h"
#include "models/qwen3_asr/qwen3_asr.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::qwen3_asr {
namespace {

using graph::Graph;
using graph::Value;

class Qwen3AsrModel final : public Model {
public:
    Qwen3AsrModel(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, const std::filesystem::path& dir)
        : cfg_(std::move(cfg)),
          info_(std::move(info)),
          weights_(std::move(weights)),
          decoder_(cfg_.text, *weights_, cfg_.released_layout ? "thinker.model." : "model.language_model.",
                   lm_head_name()),
          audio_(make_audio_tower(cfg_, *weights_, dir)),
          transcription_(make_transcription(cfg_.languages)) {
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
    const Transcription* transcription() const override { return transcription_.get(); }

    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        decoder_.token_inputs(tokens, start_pos, inputs);
        blocks::media_inputs(embeddings, static_cast<int64_t>(tokens.size()), cfg_.text.hidden_size, inputs);
    }

private:
    // The released checkpoint keeps an lm_head even with tied embeddings; a converted one may leave it out.
    std::string lm_head_name() const {
        const std::string head = cfg_.released_layout ? "thinker.lm_head.weight" : "lm_head.weight";
        return weights_->contains(head) ? head : "";
    }

    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    qwen3::Decoder decoder_;
    std::unique_ptr<AudioEncoder> audio_;
    std::unique_ptr<Transcription> transcription_;
    Graph graph_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    (void)options;  // no drafter, no vision
    Config cfg = parse_config(config_json);
    ModelInfo info;
    info.arch = "qwen3_asr";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.text.vocab_size;
    info.hidden_size = cfg.text.hidden_size;
    info.max_position = cfg.text.max_position;
    info.dialect = "plain";  // "language <Name><asr_text><transcript>": the transcription API reads it
    std::vector<int32_t> stops;
    info.default_sampling = cfg::read_generation_defaults(dir, &stops);
    if (const auto eos = cfg::tokenizer_eos_id(dir)) stops.push_back(*eos);
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    XS_CHECK(!stops.empty(), "qwen3_asr: no end-of-sequence token in generation_config.json or tokenizer_config.json");
    info.stop_token_ids = std::move(stops);
    return std::make_unique<Qwen3AsrModel>(std::move(cfg), std::move(info), SafetensorsWeights::open_dir(dir), dir);
}

}  // namespace xs::qwen3_asr
