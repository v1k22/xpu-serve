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

    c.audio = qwen3_audio::parse_config(t.at("audio_config"), where + " audio_config");
    XS_CHECK(c.audio.output_dim == c.text.hidden_size, where, " audio_config: output_dim ", c.audio.output_dim,
             " is not the decoder's width ", c.text.hidden_size);
    return c;
}

}  // namespace xs::qwen3_asr
