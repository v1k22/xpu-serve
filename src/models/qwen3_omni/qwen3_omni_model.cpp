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

// A media row: the encoder's row for the token, then (images) the vision tower's deepstack rows, which the decoder adds
// to its first layers' output; an audio row leaves them 0.
int64_t media_width(const Config& cfg) {
    return cfg.text.hidden_size * (1 + (cfg.vision ? static_cast<int64_t>(cfg.vision->deepstack.size()) : 0));
}

class Qwen3OmniModel final : public Model {
public:
    Qwen3OmniModel(Config cfg, ModelInfo info, std::unique_ptr<WeightStore> weights, audio::MelParams mel,
                   std::optional<qwen3_vision::Processor> processor)
        : cfg_(std::move(cfg)),
          info_(std::move(info)),
          weights_(std::move(weights)),
          decoder_(cfg_.text, *weights_, kDecoder, "thinker.lm_head.weight", /*three_axes=*/true),
          audio_(qwen3_audio::make_encoder(
              cfg_.audio, *weights_, mel,
              {"thinker.audio_tower.", "thinker.audio_tower.proj1", "thinker.audio_tower.proj2"}, cfg_.audio_token)) {
        if (cfg_.vision) {
            vision_ = qwen3_vision::make_encoder(*cfg_.vision, *processor, *weights_, "thinker.visual.",
                                                 cfg_.image_token);
        }
        Graph& g = graph_;
        const int64_t hidden = cfg_.text.hidden_size, width = media_width(cfg_);
        const qwen3::Decoder::Inputs in = decoder_.inputs(g);
        const Value media = g.input("media", DType::F32, {graph::kT, width});
        const Value keep = g.input("media_keep", DType::F32, {graph::kT, 1});
        const auto rows = [&](int64_t k) {
            return width == hidden ? media : g.slice(media, 1, k * hidden, (k + 1) * hidden);
        };
        Value h = g.add(g.mul(decoder_.embed(g, in.ids), keep), rows(0));
        const auto deepstack = static_cast<int64_t>(width / hidden - 1);
        h = decoder_.layers(g, h, in.pos, [&](int64_t layer, Value x) {
            return layer < deepstack ? g.add(x, rows(layer + 1)) : x;
        });
        Graph::Scope scope(g, "head");
        const Value last = decoder_.final_norm(g, g.slice(h, 0, -1));
        const Value logits = g.matmul(last, decoder_.lm_head(g));
        g.output("logits", logits);
        g.output("argmax", blocks::argmax_rows(g, logits));
        g.validate();
    }

    const ModelInfo& info() const override { return info_; }
    const graph::Graph& graph() const override { return graph_; }
    const WeightStore& weights() const override { return *weights_; }
    const AudioEncoder* audio() const override { return audio_.get(); }
    const VisionEncoder* vision() const override { return vision_.get(); }

    // One position a token on every axis: text and audio alone.
    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        std::vector<RotaryPosition> positions(tokens.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            const auto p = static_cast<int32_t>(start_pos + static_cast<int64_t>(i));
            positions[i] = {p, p, p};
        }
        prepare_positioned_inputs(tokens, positions, embeddings, inputs);
    }
    void prepare_positioned_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                                   std::span<const float* const> embeddings, TensorMap& inputs) const override {
        decoder_.token_inputs(tokens, positions, inputs);
        blocks::media_inputs(embeddings, static_cast<int64_t>(tokens.size()), media_width(cfg_), inputs);
    }

private:
    Config cfg_;
    ModelInfo info_;
    std::unique_ptr<WeightStore> weights_;
    qwen3::Decoder decoder_;
    std::unique_ptr<AudioEncoder> audio_;
    std::unique_ptr<VisionEncoder> vision_;
    Graph graph_;
};

}  // namespace

std::unique_ptr<Model> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                            const LoadOptions& options) {
    Config cfg = parse_config(config_json);
    if (!options.vision) cfg.vision.reset();
    ModelInfo info;
    info.arch = "qwen3_omni_moe";
    info.id = cfg::model_id(dir);
    info.vocab_size = cfg.text.vocab_size;
    info.hidden_size = cfg.text.hidden_size;
    info.max_position = cfg.text.max_position;
    info.dialect = "qwen3";
    info.media_width = media_width(cfg);
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
    std::optional<qwen3_vision::Processor> processor;
    if (cfg.vision) processor = qwen3_vision::processor_spec(dir, *cfg.vision, options.image_max_pixels);
    return std::make_unique<Qwen3OmniModel>(std::move(cfg), std::move(info), std::move(weights), mel, processor);
}

}  // namespace xs::qwen3_omni
