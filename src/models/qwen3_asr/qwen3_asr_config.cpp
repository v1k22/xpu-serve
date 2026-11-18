#include "models/config_util.h"
#include "models/qwen3_asr/qwen3_asr.h"
#include "util/error.h"

namespace xs::qwen3_asr {

Config parse_config(const nlohmann::json& root) {
    Config c;
    c.released_layout = root.contains("thinker_config");
    const nlohmann::json& t = c.released_layout ? root.at("thinker_config") : root;
    const std::string where = c.released_layout ? "config.json thinker_config" : "config.json";
    if (t.contains("classify_num") || root.contains("num_labels")) {
        fail(where, ": this is Qwen3-ForcedAligner (a timestamp classifier), which is not supported yet");
    }
    if (!t.contains("audio_config") || !t.contains("text_config")) fail(where, ": missing audio_config or text_config");
    c.text = qwen3::parse_config(t.at("text_config"), where + " text_config");
    if (root.contains("support_languages")) {
        for (const auto& l : root.at("support_languages")) c.languages.push_back(l.get<std::string>());
    }
    c.audio_token = static_cast<int32_t>(cfg::get_int(t, "audio_token_id", where));

    const nlohmann::json& a = t.at("audio_config");
    const std::string aw = where + " audio_config";
    const std::string act = a.contains("activation_function") ? cfg::get_string(a, "activation_function", aw) : "gelu";
    if (act != "gelu") fail(aw, ": activation_function '", act, "' is not supported (gelu is)");
    if (cfg::get_bool_or(a, "scale_embedding", false, aw)) fail(aw, ": scale_embedding is not supported");
    AudioConfig& ac = c.audio;
    ac.d_model = cfg::get_int(a, "d_model", aw);
    ac.layers = cfg::get_int(a, "encoder_layers", aw);
    ac.heads = cfg::get_int(a, "encoder_attention_heads", aw);
    ac.ffn = cfg::get_int(a, "encoder_ffn_dim", aw);
    ac.mel_bins = cfg::get_int(a, "num_mel_bins", aw);
    ac.conv_channels = cfg::get_int(a, "downsample_hidden_size", aw);
    ac.output_dim = cfg::get_int(a, "output_dim", aw);
    ac.n_window = cfg::get_int(a, "n_window", aw);
    ac.n_window_infer = cfg::get_int(a, "n_window_infer", aw);
    ac.max_positions = a.contains("max_source_positions") ? cfg::get_int(a, "max_source_positions", aw)
                                                           : cfg::get_int(a, "max_position_embeddings", aw);
    XS_CHECK(ac.d_model % ac.heads == 0, aw, ": d_model ", ac.d_model, " is not a multiple of ", ac.heads, " heads");
    XS_CHECK(ac.n_window > 0 && ac.n_window_infer % (2 * ac.n_window) == 0, aw, ": n_window_infer ",
             ac.n_window_infer, " is not a whole number of chunks of ", 2 * ac.n_window, " frames");
    XS_CHECK(ac.output_dim == c.text.hidden_size, aw, ": output_dim ", ac.output_dim, " is not the decoder's width ",
             c.text.hidden_size);
    return c;
}

int64_t encoder_rows(const AudioConfig& a, int64_t mel_frames) {
    const int64_t chunk = 2 * a.n_window;
    const auto conv = [](int64_t n) {
        for (int k = 0; k < 3; ++k) n = n > 0 ? (n - 1) / 2 + 1 : 0;
        return n;
    };
    return (mel_frames / chunk) * conv(chunk) + conv(mel_frames % chunk);
}

}  // namespace xs::qwen3_asr
