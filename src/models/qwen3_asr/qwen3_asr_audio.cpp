#include <algorithm>
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
// A long clip's windows run this many at a time (program 1), the rest one at a time (program 0): the encoder's cost
// grows with its rows, so a batch only saves the per-run cost, and a short clip would pay for the windows it pads.
constexpr int64_t kBatchWindows = 8;

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
        build(single_, 1);
        build(batch_, kBatchWindows);
    }

    std::vector<const graph::Graph*> programs() const override { return {&single_, &batch_}; }
    int32_t placeholder_token() const override { return audio_token_; }
    int32_t soft_token() const override { return audio_token_; }
    std::vector<int32_t> token_run(int64_t n) const override {
        return std::vector<int32_t>(static_cast<size_t>(n), audio_token_);
    }

    // Windows of n_window_infer frames, kBatchWindows a run while that many remain, then one a run; the clip's last
    // chunk zero-padded (as the reference pads it) and the keys past the clip in its window masked.
    MediaInput prepare(const audio::Audio& clip) const override {
        XS_CHECK(clip.sample_rate > 0 && !clip.samples.empty(), "qwen3_asr: an empty audio clip");
        const std::vector<float> pcm = clip.sample_rate == mel_.sample_rate
                                           ? clip.samples
                                           : audio::resample(clip.samples, clip.sample_rate, mel_.sample_rate);
        const HostTensor mel = audio::log_mel_spectrogram(pcm, mel_);  // [mel bins, frames]
        const int64_t window_frames = chunks_ * chunk_frames_;
        const int64_t windows = (mel.dim(1) + window_frames - 1) / window_frames;
        MediaInput media;
        media.modality = Modality::Audio;
        for (int64_t w = 0; w < windows;) {
            const int64_t count = windows - w >= kBatchWindows ? kBatchWindows : 1;
            media.runs.push_back(windows_run(mel, w, count));
            media.num_tokens += media.runs.back().rows;
            w += count;
        }
        return media;
    }

    // Silence of a batch and one window more: a run of each program.
    MediaInput warm_up_item() const override {
        audio::Audio silence;
        silence.sample_rate = mel_.sample_rate;
        const int64_t frames = (kBatchWindows + 1) * chunks_ * chunk_frames_;
        silence.samples.assign(static_cast<size_t>(frames * mel_.hop_length), 0.0f);
        return prepare(silence);
    }

private:
    // The run of `count` windows from window `first`: their chunks, and a key bias that masks the rows past the clip.
    EncoderRun windows_run(const HostTensor& mel, int64_t first, int64_t count) const {
        const int64_t bins = mel.dim(0), frames = mel.dim(1), window_frames = chunks_ * chunk_frames_;
        const int64_t begin = first * window_frames, used = std::min(count * window_frames, frames - begin);
        EncoderRun run;
        run.program = count == 1 ? 0 : 1;
        HostTensor chunks(DType::F32, {count * chunks_, 1, bins, chunk_frames_});
        for (int64_t b = 0; b < bins; ++b) {
            for (int64_t f = 0; f < used; ++f) {
                const int64_t chunk = f / chunk_frames_, t = f % chunk_frames_;
                chunks.data<float>()[(chunk * bins + b) * chunk_frames_ + t] =
                    mel.data<float>()[b * frames + begin + f];
            }
        }
        run.rows = encoder_rows(cfg_, used);
        HostTensor key_bias(DType::F32, {count * chunks_ * chunk_rows_});
        for (int64_t r = 0; r < key_bias.numel(); ++r) {
            key_bias.data<float>()[r] = r < run.rows ? 0.0f : -std::numeric_limits<float>::infinity();
        }
        run.inputs.emplace("mel", std::move(chunks));
        run.inputs.emplace("key_bias", std::move(key_bias));
        return run;
    }

    Value linear(Graph& g, Value x, const std::string& stem) { return torch_nn::linear(g, weights_, x, stem); }
    Value layer_norm(Graph& g, Value x, const std::string& stem) {
        return torch_nn::layer_norm(g, weights_, x, stem, kLayerNormEps);
    }

    // `windows` windows: the chunks' convolutions, conv_out and positions, the layers over each window's rows (a
    // block of attention), ln_post, the projector.
    void build(Graph& g, int64_t windows) {
        const int64_t width = cfg_.d_model, head_dim = width / cfg_.heads, chunks = windows * chunks_;
        const int64_t rows = chunks * chunk_rows_;
        const Value mel = g.input("mel", DType::F32, {chunks, 1, cfg_.mel_bins, chunk_frames_});
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
        x = g.reshape(g.transpose(x, {0, 3, 1, 2}), {chunks, chunk_rows_, conv_shape[1] * conv_shape[2]});
        x = g.matmul(x, g.weight(weights_, tower_ + "conv_out.weight"));
        XS_CHECK(chunk_rows_ <= cfg_.max_positions, "qwen3_asr: ", chunk_rows_, " rows a chunk past the table of ",
                 cfg_.max_positions);
        x = g.reshape(g.add(x, g.constant(sinusoids(chunk_rows_, width))), {rows, width});

        graph::AttentionAttrs attrs;
        attrs.causal = false;
        attrs.block = chunks_ * chunk_rows_;
        attrs.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        for (int64_t i = 0; i < cfg_.layers; ++i) {
            const std::string L = tower_ + "layers." + std::to_string(i) + ".";
            Graph::Scope s(g, "layers." + std::to_string(i));
            Value h = layer_norm(g, x, L + "self_attn_layer_norm");
            const auto heads = [&](const char* proj) {
                return g.reshape(linear(g, h, L + "self_attn." + proj), {rows, cfg_.heads, head_dim});
            };
            const Value attn = g.attention(heads("q_proj"), heads("k_proj"), heads("v_proj"), key_bias, attrs);
            x = g.add(x, linear(g, g.reshape(attn, {rows, width}), L + "self_attn.out_proj"));
            h = layer_norm(g, x, L + "final_layer_norm");
            x = g.add(x, linear(g, g.gelu(linear(g, h, L + "fc1")), L + "fc2"));
        }
        x = layer_norm(g, x, tower_ + "ln_post");
        g.output("embeddings", linear(g, g.gelu(linear(g, x, proj1_)), proj2_));
        g.validate();
    }

    AudioConfig cfg_;
    const WeightStore& weights_;
    audio::MelParams mel_;
    int32_t audio_token_;
    std::string tower_, proj1_, proj2_;
    int64_t chunk_frames_ = 0, chunks_ = 0, chunk_rows_ = 0;
    Graph single_, batch_;  // programs 0 and 1
};


}  // namespace

// WhisperFeatureExtractor's or Qwen3ASRFeatureExtractor's settings.
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

std::unique_ptr<AudioEncoder> make_audio_tower(const Config& cfg, const WeightStore& weights,
                                               const audio::MelParams& mel) {
    return std::make_unique<AudioTower>(cfg, weights, mel);
}

}  // namespace xs::qwen3_asr
