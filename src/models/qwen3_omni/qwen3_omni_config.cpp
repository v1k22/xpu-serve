#include "models/config_util.h"
#include "models/qwen3_omni/qwen3_omni.h"
#include "util/error.h"

namespace xs::qwen3_omni {

Config parse_config(const nlohmann::json& root) {
    if (!root.contains("thinker_config")) fail("config.json: qwen3_omni_moe without a thinker_config");
    const nlohmann::json& t = root.at("thinker_config");
    const std::string where = "config.json thinker_config";
    if (!t.contains("audio_config") || !t.contains("text_config")) fail(where, ": missing audio_config or text_config");
    Config c;
    c.text = qwen3::parse_config(t.at("text_config"), where + " text_config");
    c.audio = qwen3_audio::parse_config(t.at("audio_config"), where + " audio_config");
    c.audio_token = static_cast<int32_t>(cfg::get_int(t, "audio_token_id", where));
    if (t.contains("vision_config")) {
        c.vision = qwen3_vision::parse_config(t.at("vision_config"), where + " vision_config");
        c.image_token = static_cast<int32_t>(cfg::get_int(t, "image_token_id", where));
        XS_CHECK(c.vision->out_hidden == c.text.hidden_size, where, " vision_config: out_hidden_size ",
                 c.vision->out_hidden, " is not the decoder's width ", c.text.hidden_size);
        XS_CHECK(c.vision->deepstack.size() < static_cast<size_t>(c.text.num_layers), where, ": ",
                 c.vision->deepstack.size(), " deepstack rows for ", c.text.num_layers, " decoder layers");
    }
    XS_CHECK(c.audio.output_dim == c.text.hidden_size, where, " audio_config: output_dim ", c.audio.output_dim,
             " is not the decoder's width ", c.text.hidden_size);
    return c;
}

}  // namespace xs::qwen3_omni
