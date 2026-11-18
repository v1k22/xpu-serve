#include <cmath>
#include <limits>

#include "audio/audio.h"
#include "audio/mel.h"
#include "audio/resample.h"
#include "models/common/torch_nn.h"
#include "models/config_util.h"
#include "models/qwen3_asr/qwen3_asr.h"
#include "util/error.h"

namespace xs::qwen3_asr {
namespace {

using graph::Graph;
using graph::Value;

constexpr float kLayerNormEps = 1e-5f;  // nn.LayerNorm's default

// Rows `length` of the encoder's sinusoidal table (SinusoidsPositionEmbedding): [sin(t f_i) | cos(t f_i)] with
// f_i = exp(-i log(10000) / (channels / 2 - 1)).
HostTensor sinusoids(int64_t length, int64_t channels) {
    HostTensor t(DType::F32, {length, channels});
    const int64_t half = channels / 2;
    const double step = std::log(10000.0) / static_cast<double>(half - 1);
    for (int64_t p = 0; p < length; ++p) {
        for (int64_t i = 0; i < half; ++i) {
            const float f = static_cast<float>(std::exp(-step * static_cast<double>(i)));
            const float angle = static_cast<float>(p) * f;
            t.data<float>()[p * channels + i] = std::sin(angle);
            t.data<float>()[p * channels + half + i] = std::cos(angle);
        }
    }
    return t;
}

class AudioTower final : public AudioEncoder {
public:
    AudioTower(const Config& cfg, const WeightStore& weights, audio::MelParams mel)
        : cfg_(cfg.audio), weights_(weights), mel_(mel), audio_token_(cfg.audio_token) {
        tower_ = cfg.released_layout ? "thinker.audio_tower." : "model.audio_tower.";
        proj1_ = cfg.released_layout ? tower_ + "proj1" : "model.multi_modal_projector.linear_1";
        proj2_ = cfg.released_layout ? tower_ + "proj2" : "model.multi_modal_projector.linear_2";
        XS_CHECK(mel_.n_mels == cfg_.mel_bins, "qwen3_asr: the feature extractor makes ", mel_.n_mels,
                 " mel bins; the encoder reads ", cfg_.mel_bins);
        chunk_frames_ = 2 * cfg_.n_window;
        chunks_ = cfg_.n_window_infer / chunk_frames_;
        chunk_rows_ = encoder_rows(cfg_, chunk_frames_);
        build();
    }

    const graph::Graph& graph() const override { return graph_; }
    int32_t placeholder_token() const override { return audio_token_; }
    int32_t soft_token() const override { return audio_token_; }
    std::vector<int32_t> token_run(int64_t n) const override {
        return std::vector<int32_t>(static_cast<size_t>(n), audio_token_);
    }

    // One run per window of n_window_infer frames: its chunks zero-padded to whole chunks and windows (as the
    // reference pads a clip's last chunk), the keys past the clip's rows masked.
    MediaInput prepare(const audio::Audio& clip) const override {
        XS_CHECK(clip.sample_rate > 0 && !clip.samples.empty(), "qwen3_asr: an empty audio clip");
        const std::vector<float> pcm = clip.sample_rate == mel_.sample_rate
                                           ? clip.samples
                                           : audio::resample(clip.samples, clip.sample_rate, mel_.sample_rate);
        const HostTensor mel = audio::log_mel_spectrogram(pcm, mel_);  // [mel bins, frames]
        const int64_t bins = mel.dim(0), frames = mel.dim(1);
        const int64_t window_frames = chunks_ * chunk_frames_;
        MediaInput media;
        media.modality = Modality::Audio;
        for (int64_t first = 0; first < frames; first += window_frames) {
            const int64_t count = std::min(window_frames, frames - first);
            EncoderRun& run = media.runs.emplace_back();
            HostTensor chunks(DType::F32, {chunks_, 1, bins, chunk_frames_});
            for (int64_t b = 0; b < bins; ++b) {
                for (int64_t f = 0; f < count; ++f) {
                    const int64_t chunk = f / chunk_frames_, t = f % chunk_frames_;
                    chunks.data<float>()[(chunk * bins + b) * chunk_frames_ + t] =
                        mel.data<float>()[b * frames + first + f];
                }
            }
            run.rows = encoder_rows(cfg_, count);
            HostTensor key_bias(DType::F32, {chunks_ * chunk_rows_});
            for (int64_t r = 0; r < key_bias.numel(); ++r) {
                key_bias.data<float>()[r] = r < run.rows ? 0.0f : -std::numeric_limits<float>::infinity();
            }
            run.inputs.emplace("mel", std::move(chunks));
            run.inputs.emplace("key_bias", std::move(key_bias));
            media.num_tokens += run.rows;
        }
        return media;
    }

private:
    Value linear(Value x, const std::string& stem) { return torch_nn::linear(graph_, weights_, x, stem); }
    Value layer_norm(Value x, const std::string& stem) {
        return torch_nn::layer_norm(graph_, weights_, x, stem, kLayerNormEps);
    }

    // A window: the chunks' convolutions, conv_out and positions, the layers over its rows, ln_post, the projector.
    void build() {
        Graph& g = graph_;
        const int64_t width = cfg_.d_model, rows = chunks_ * chunk_rows_, head_dim = width / cfg_.heads;
        const Value mel = g.input("mel", DType::F32, {chunks_, 1, cfg_.mel_bins, chunk_frames_});
        const Value key_bias = g.input("key_bias", DType::F32, {rows});
        graph::Conv2dAttrs s2;
        s2.stride_h = s2.stride_w = 2;
        s2.pad_top = s2.pad_bottom = s2.pad_left = s2.pad_right = 1;
        Value x = mel;
        for (const char* conv : {"conv2d1", "conv2d2", "conv2d3"}) {
            Graph::Scope s(g, conv);
            const std::string stem = tower_ + conv;
            x = g.gelu(g.conv2d(x, g.weight(weights_, stem + ".weight"), g.weight(weights_, stem + ".bias"), s2));
        }
        // [chunks, channels, freq, time] -> [chunks, time, channels * freq] -> conv_out, plus positions per chunk
        const Dims conv_shape = g.type(x).shape;
        XS_CHECK(conv_shape[3] == chunk_rows_, "qwen3_asr: a chunk convolves to ", conv_shape[3], " rows, not ",
                 chunk_rows_);
        x = g.reshape(g.transpose(x, {0, 3, 1, 2}), {chunks_, chunk_rows_, conv_shape[1] * conv_shape[2]});
        x = g.matmul(x, g.weight(weights_, tower_ + "conv_out.weight"));
        XS_CHECK(chunk_rows_ <= cfg_.max_positions, "qwen3_asr: ", chunk_rows_, " rows a chunk past the table of ",
                 cfg_.max_positions);
        x = g.reshape(g.add(x, g.constant(sinusoids(chunk_rows_, width))), {rows, width});

        graph::AttentionAttrs attrs;
        attrs.causal = false;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        for (int64_t i = 0; i < cfg_.layers; ++i) {
            const std::string L = tower_ + "layers." + std::to_string(i) + ".";
            Graph::Scope s(g, "layers." + std::to_string(i));
            Value h = layer_norm(x, L + "self_attn_layer_norm");
            const auto heads = [&](const char* proj) {
                return g.reshape(linear(h, L + "self_attn." + proj), {rows, cfg_.heads, head_dim});
            };
            const Value attn = g.attention(heads("q_proj"), heads("k_proj"), heads("v_proj"), key_bias, attrs);
            x = g.add(x, linear(g.reshape(attn, {rows, width}), L + "self_attn.out_proj"));
            h = layer_norm(x, L + "final_layer_norm");
            x = g.add(x, linear(g.gelu(linear(h, L + "fc1")), L + "fc2"));
        }
        x = layer_norm(x, tower_ + "ln_post");
        g.output("embeddings", linear(g.gelu(linear(x, proj1_)), proj2_));
        g.validate();
    }

    AudioConfig cfg_;
    const WeightStore& weights_;
    audio::MelParams mel_;
    int32_t audio_token_;
    std::string tower_, proj1_, proj2_;
    int64_t chunk_frames_ = 0, chunks_ = 0, chunk_rows_ = 0;
    Graph graph_;
};

// preprocessor_config.json's WhisperFeatureExtractor / Qwen3ASRFeatureExtractor settings.
audio::MelParams mel_spec(const std::filesystem::path& dir) {
    audio::MelParams m;
    const std::filesystem::path path = dir / "preprocessor_config.json";
    if (!std::filesystem::exists(path)) return m;  // the extractor's defaults
    const nlohmann::json p = cfg::read_json(path);
    const std::string where = "preprocessor_config.json";
    const std::string type = p.contains("feature_extractor_type") ? cfg::get_string(p, "feature_extractor_type", where)
                                                                  : "WhisperFeatureExtractor";
    if (type != "WhisperFeatureExtractor" && type != "Qwen3ASRFeatureExtractor") {
        fail(where, ": feature_extractor_type '", type, "' is not supported");
    }
    if (p.contains("dither") && cfg::get_float(p, "dither", where) != 0.0) fail(where, ": dither is not supported");
    m.sample_rate = static_cast<int>(cfg::get_int_or(p, "sampling_rate", m.sample_rate, where));
    m.n_fft = static_cast<int>(cfg::get_int_or(p, "n_fft", m.n_fft, where));
    m.hop_length = static_cast<int>(cfg::get_int_or(p, "hop_length", m.hop_length, where));
    m.n_mels = static_cast<int>(cfg::get_int_or(p, "feature_size", m.n_mels, where));
    return m;
}

}  // namespace

std::unique_ptr<AudioEncoder> make_audio_tower(const Config& cfg, const WeightStore& weights,
                                               const std::filesystem::path& dir) {
    return std::make_unique<AudioTower>(cfg, weights, mel_spec(dir));
}

}  // namespace xs::qwen3_asr
