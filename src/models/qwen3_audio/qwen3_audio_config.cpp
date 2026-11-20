#include "models/config_util.h"
#include "models/qwen3_audio/qwen3_audio.h"
#include "util/error.h"

namespace xs::qwen3_audio {

Config parse_config(const nlohmann::json& a, const std::string& where) {
    const std::string act = a.contains("activation_function") ? cfg::get_string(a, "activation_function", where)
                                                              : "gelu";
    if (act != "gelu") fail(where, ": activation_function '", act, "' is not supported (gelu is)");
    if (cfg::get_bool_or(a, "scale_embedding", false, where)) fail(where, ": scale_embedding is not supported");
    Config c;
    c.d_model = cfg::get_int(a, "d_model", where);
    c.layers = cfg::get_int(a, "encoder_layers", where);
    c.heads = cfg::get_int(a, "encoder_attention_heads", where);
    c.ffn = cfg::get_int(a, "encoder_ffn_dim", where);
    c.mel_bins = cfg::get_int(a, "num_mel_bins", where);
    c.conv_channels = cfg::get_int(a, "downsample_hidden_size", where);
    c.output_dim = cfg::get_int(a, "output_dim", where);
    c.n_window = cfg::get_int(a, "n_window", where);
    c.n_window_infer = cfg::get_int(a, "n_window_infer", where);
    c.max_positions = a.contains("max_source_positions") ? cfg::get_int(a, "max_source_positions", where)
                                                          : cfg::get_int(a, "max_position_embeddings", where);
    XS_CHECK(c.d_model % c.heads == 0, where, ": d_model ", c.d_model, " is not a multiple of ", c.heads, " heads");
    XS_CHECK(c.n_window > 0 && c.n_window_infer % (2 * c.n_window) == 0, where, ": n_window_infer ", c.n_window_infer,
             " is not a whole number of chunks of ", 2 * c.n_window, " frames");
    return c;
}

int64_t encoder_rows(const Config& c, int64_t mel_frames) {
    const int64_t chunk = 2 * c.n_window;
    const auto conv = [](int64_t n) {
        for (int k = 0; k < 3; ++k) n = n > 0 ? (n - 1) / 2 + 1 : 0;
        return n;
    };
    return (mel_frames / chunk) * conv(chunk) + conv(mel_frames % chunk);
}

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

double rows_per_second(const Config& c, const audio::MelParams& mel) {
    return static_cast<double>(encoder_rows(c, 60 * mel.sample_rate / mel.hop_length)) / 60.0;
}

}  // namespace xs::qwen3_audio
