// Gemma 4 12B's media inputs (Gemma4UnifiedForConditionalGeneration, transformers models/gemma4_unified/): no vision
// or audio tower. An image is cut into merged patches of pooling_kernel x patch_size pixels a side (48), each one soft
// token: Gemma4UnifiedVisionEmbedder's LayerNorm, dense layer, LayerNorm, a position row a side (x, y), LayerNorm,
// then the multimodal embedder's weightless RMSNorm and projection. Audio is cut into frames of samples_per_token
// samples at 16 kHz (40 ms), each one soft token through the same RMSNorm and a projection of its own.

#include <cmath>

#include "audio/resample.h"
#include "image/image.h"
#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "util/error.h"

namespace xs::gemma4 {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr float kLayerNormEps = 1e-5f;  // nn.LayerNorm's default, which the embedder keeps
constexpr int kAudioRate = 16000;

std::vector<int32_t> media_run(int32_t open, int32_t token, int32_t close, int64_t n) {
    std::vector<int32_t> run;
    run.reserve(static_cast<size_t>(n) + 2);
    run.push_back(open);
    run.insert(run.end(), static_cast<size_t>(n), token);
    run.push_back(close);
    return run;
}

class UnifiedVision final : public VisionEncoder {
public:
    UnifiedVision(VisionConfig vc, const WeightStore& weights, std::string prefix)
        : vc_(std::move(vc)), weights_(weights), prefix_(std::move(prefix)) {
        build_graph();
    }

    std::vector<const graph::Graph*> programs() const override { return {&graph_}; }
    int32_t placeholder_token() const override { return vc_.image_token_id; }
    int32_t soft_token() const override { return vc_.image_token_id; }
    std::vector<int32_t> token_run(int64_t num_tokens) const override {
        return media_run(vc_.boi_token_id, vc_.image_token_id, vc_.eoi_token_id, num_tokens);
    }

    // Gemma4UnifiedImageProcessor: the aspect-preserving resize of Gemma 4's processor (sides a multiple of the merged
    // patch), rescale; the merged patches in row-major order, each its pixels (row, column, channel), at (x, y).
    MediaInput prepare(const image::Image& img) const override {
        const auto [height, width] = image_target_size(img.height, img.width, vc_);
        const image::Image resized = image::resize_bicubic(img, width, height);
        const int64_t side = vc_.patch_size * vc_.pooling_kernel;
        const int64_t grid_h = height / side, grid_w = width / side;
        const int64_t tokens = grid_h * grid_w;
        const int64_t dim = side * side * 3;
        HostTensor pixels(DType::F32, {tokens, dim});
        HostTensor positions(DType::I32, {tokens, 2});
        float* px = pixels.data<float>();
        for (int64_t gy = 0; gy < grid_h; ++gy) {
            for (int64_t gx = 0; gx < grid_w; ++gx) {
                const int64_t token = gy * grid_w + gx;
                positions.data<int32_t>()[2 * token] = static_cast<int32_t>(gx);
                positions.data<int32_t>()[2 * token + 1] = static_cast<int32_t>(gy);
                float* out = px + token * dim;
                for (int64_t dy = 0; dy < side; ++dy) {
                    for (int64_t dx = 0; dx < side; ++dx) {
                        for (int c = 0; c < 3; ++c) {
                            *out++ = static_cast<float>(resized.at(gy * side + dy, gx * side + dx, c)) *
                                     vc_.rescale_factor;
                        }
                    }
                }
            }
        }
        MediaInput media;
        media.modality = Modality::Image;
        media.num_tokens = tokens;
        EncoderRun& run = media.runs.emplace_back();
        run.rows = tokens;
        run.inputs.emplace("pixels", std::move(pixels));
        run.inputs.emplace("patch_pos", std::move(positions));
        return media;
    }

private:
    Value weight(const std::string& name) { return graph_.weight(weights_, prefix_ + name); }

    void build_graph() {
        Graph& g = graph_;
        const int64_t side = vc_.patch_size * vc_.pooling_kernel;
        const int64_t hidden = vc_.hidden_size;
        const Value pixels = g.input("pixels", DType::F32, {kT, side * side * 3});
        const Value patch_pos = g.input("patch_pos", DType::I32, {kT, 2});
        const std::string E = "vision_embedder.";
        const auto layer_norm = [&](Value x, const std::string& name) {
            return g.layer_norm(x, weight(E + name + ".weight"), weight(E + name + ".bias"), kLayerNormEps);
        };
        Graph::Scope s(g, "vision_embedder");
        Value h = layer_norm(pixels, "patch_ln1");
        h = g.add(g.matmul(h, weight(E + "patch_dense.weight")), weight(E + "patch_dense.bias"));
        h = layer_norm(h, "patch_ln2");
        const Value table = weight(E + "pos_embedding");  // [positions, 2 (x, y), hidden]
        const Dims& ts = g.type(table).shape;
        const Dims want{vc_.position_embedding_size, 2, hidden};
        XS_CHECK(ts == want, "gemma4 unified vision: pos_embedding has shape ", dims_to_string(ts), ", expected ",
                 dims_to_string(want));
        for (int axis = 0; axis < 2; ++axis) {
            const Value rows =
                g.reshape(g.slice(table, 1, axis, axis + 1), {vc_.position_embedding_size, hidden});
            const Value ids = g.reshape(g.slice(patch_pos, 1, axis, axis + 1), {kT});
            h = g.add(h, g.gather(rows, ids));
        }
        h = layer_norm(h, "pos_norm");
        const Value normed = g.rms_norm(h, static_cast<float>(vc_.rms_norm_eps));
        g.output("embeddings", g.matmul(normed, weight("embed_vision.embedding_projection.weight")));
        g.validate();
    }

    const VisionConfig vc_;
    const WeightStore& weights_;
    const std::string prefix_;
    graph::Graph graph_;
};

class UnifiedAudio final : public AudioEncoder {
public:
    UnifiedAudio(AudioConfig ac, const WeightStore& weights, std::string prefix)
        : ac_(std::move(ac)), weights_(weights), prefix_(std::move(prefix)) {
        Graph& g = graph_;
        Graph::Scope s(g, "embed_audio");
        const Value frames = g.input("frames", DType::F32, {kT, ac_.samples_per_token});
        const Value normed = g.rms_norm(frames, static_cast<float>(ac_.rms_norm_eps));
        const Value projection = g.weight(weights_, prefix_ + "embed_audio.embedding_projection.weight");
        g.output("embeddings", g.matmul(normed, projection));
        g.validate();
    }

    std::vector<const graph::Graph*> programs() const override { return {&graph_}; }
    int32_t placeholder_token() const override { return ac_.audio_token_id; }
    int32_t soft_token() const override { return ac_.audio_token_id; }
    std::vector<int32_t> token_run(int64_t num_tokens) const override {
        return media_run(ac_.boa_token_id, ac_.audio_token_id, ac_.eoa_token_id, num_tokens);
    }

    // Gemma4UnifiedAudioFeatureExtractor: the clip at 16 kHz, zero-padded to whole frames, a frame a soft token.
    MediaInput prepare(const audio::Audio& clip) const override {
        XS_CHECK(clip.sample_rate > 0 && !clip.samples.empty(), "gemma4 audio: an empty clip");
        const std::vector<float> samples = clip.sample_rate == kAudioRate
                                               ? clip.samples
                                               : audio::resample(clip.samples, clip.sample_rate, kAudioRate);
        const int64_t per = ac_.samples_per_token;
        const int64_t tokens = (static_cast<int64_t>(samples.size()) + per - 1) / per;
        HostTensor frames(DType::F32, {tokens, per});
        std::fill(frames.data<float>(), frames.data<float>() + frames.numel(), 0.0f);
        std::copy(samples.begin(), samples.end(), frames.data<float>());
        MediaInput media;
        media.modality = Modality::Audio;
        media.num_tokens = tokens;
        EncoderRun& run = media.runs.emplace_back();
        run.rows = tokens;
        run.inputs.emplace("frames", std::move(frames));
        return media;
    }

private:
    const AudioConfig ac_;
    const WeightStore& weights_;
    const std::string prefix_;
    graph::Graph graph_;
};

int32_t token_or(const nlohmann::json& root, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (root.contains(key) && !root.at(key).is_null()) {
            return static_cast<int32_t>(cfg::get_int(root, key, "config.json"));
        }
    }
    fail("config.json has none of the tokens ", *keys.begin());
}

}  // namespace

std::unique_ptr<VisionEncoder> make_unified_vision_encoder(const VisionConfig& vc, const WeightStore& weights,
                                                           const std::string& prefix) {
    return std::make_unique<UnifiedVision>(vc, weights, prefix);
}

std::optional<AudioConfig> parse_audio_config(const nlohmann::json& config_json, int64_t text_hidden_size) {
    if (!config_json.contains("audio_config") || config_json.at("audio_config").is_null()) return std::nullopt;
    const nlohmann::json& a = config_json.at("audio_config");
    const std::string where = "config.json audio_config";
    const std::string type = cfg::get_string(a, "model_type", where);
    if (type != "gemma4_unified_audio") {
        fail(where, ": model_type '", type, "' is not supported (supported: gemma4_unified_audio)");
    }
    AudioConfig ac;
    ac.samples_per_token = cfg::get_int(a, "audio_embed_dim", where);
    if (a.contains("audio_samples_per_token")) {
        XS_CHECK(cfg::get_int(a, "audio_samples_per_token", where) == ac.samples_per_token, where,
                 ": audio_samples_per_token differs from audio_embed_dim");
    }
    const int64_t out = cfg::get_int_or(a, "output_proj_dims", ac.samples_per_token, where);
    XS_CHECK(out == ac.samples_per_token, where, ": output_proj_dims ", out, " differs from audio_embed_dim ",
             ac.samples_per_token);
    ac.rms_norm_eps = cfg::get_float(a, "rms_norm_eps", where);
    ac.text_hidden_size = text_hidden_size;
    ac.audio_token_id = token_or(config_json, {"audio_token_id"});
    ac.boa_token_id = token_or(config_json, {"boa_token_id"});
    ac.eoa_token_id = token_or(config_json, {"eoa_token_id", "eoa_token_index"});
    XS_CHECK(ac.samples_per_token > 0, where, ": audio_embed_dim must be positive");
    return ac;
}

std::unique_ptr<AudioEncoder> make_audio_encoder(const AudioConfig& ac, const WeightStore& weights,
                                                 const std::string& prefix) {
    return std::make_unique<UnifiedAudio>(ac, weights, prefix);
}

}  // namespace xs::gemma4
