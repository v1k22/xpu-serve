#include "models/config_util.h"
#include "models/pi05/pi05.h"
#include "util/error.h"

namespace xs::pi05 {
namespace {

// openpi gemma.py get_config.
GemmaDims gemma_variant(const std::string& name, const std::string& where) {
    if (name == "gemma_2b") return GemmaDims{2048, 18, 16384, 8, 1, 256};
    if (name == "gemma_300m") return GemmaDims{1024, 18, 4096, 8, 1, 256};
    fail(where, ": Gemma variant '", name, "' is not supported (gemma_2b, gemma_300m)");
}

std::vector<int64_t> feature_shape(const nlohmann::json& feature, const std::string& where) {
    if (!feature.contains("shape") || !feature.at("shape").is_array()) fail(where, ": missing shape");
    return feature.at("shape").get<std::vector<int64_t>>();
}

}  // namespace

Config parse_config(const nlohmann::json& j, const nlohmann::json* preprocessor) {
    const std::string where = "config.json";
    if (cfg::get_string(j, "type", where) != "pi05") fail(where, ": type is not pi05");
    for (const char* flag : {"use_visual_memory", "use_proprioceptive_memory", "use_relative_actions"}) {
        if (cfg::get_bool_or(j, flag, false, where)) fail(where, ": ", flag, " is not supported yet");
    }
    if (cfg::get_int_or(j, "n_obs_steps", 1, where) != 1) fail(where, ": n_obs_steps other than 1 is not supported");

    Config c;
    c.vlm = gemma_variant(cfg::get_string(j, "paligemma_variant", where), where);
    c.expert = gemma_variant(cfg::get_string(j, "action_expert_variant", where), where);
    XS_CHECK(c.vlm.depth == c.expert.depth && c.vlm.head_dim == c.expert.head_dim &&
                 c.vlm.kv_heads == c.expert.kv_heads && c.vlm.heads == c.expert.heads,
             where, ": the action expert must match the VLM's depth and attention layout");
    c.horizon = cfg::get_int(j, "chunk_size", where);
    c.max_action_dim = cfg::get_int(j, "max_action_dim", where);
    c.steps = static_cast<int>(cfg::get_int(j, "num_inference_steps", where));
    c.min_period = cfg::get_float(j, "min_period", where);
    c.max_period = cfg::get_float(j, "max_period", where);
    c.prompt_tokens = cfg::get_int(j, "tokenizer_max_length", where);
    if (j.contains("image_resolution")) {
        const auto res = j.at("image_resolution").get<std::vector<int64_t>>();
        XS_CHECK(res.size() == 2 && res[0] == res[1], where, ": image_resolution must be square");
        c.image_size = res[0];
    }
    XS_CHECK(c.image_size % 14 == 0, where, ": image_resolution ", c.image_size, " is not a multiple of the 14-pixel patch");

    if (!j.contains("input_features") || !j.at("input_features").is_object()) fail(where, ": missing input_features");
    for (const auto& [name, feature] : j.at("input_features").items()) {
        const std::string type = cfg::get_string(feature, "type", where + " input_features." + name);
        const auto shape = feature_shape(feature, where + " input_features." + name);
        if (type == "VISUAL") {
            XS_CHECK(shape.size() == 3 && shape[0] == 3, where, ": camera ", name, " must be [3, H, W]");
            c.cameras.push_back(CameraSpec{name, shape[1], shape[2]});
        } else if (type == "STATE") {
            XS_CHECK(shape.size() == 1, where, ": ", name, " must be a vector");
            c.state_dim = shape[0];
        } else {
            fail(where, ": input feature ", name, " of type ", type, " is not supported");
        }
    }
    XS_CHECK(!c.cameras.empty(), where, ": no camera (VISUAL) input feature");
    XS_CHECK(c.state_dim > 0, where, ": no observation.state input feature");
    if (!j.contains("output_features") || !j.at("output_features").contains("action")) {
        fail(where, ": missing output_features.action");
    }
    const auto action = feature_shape(j.at("output_features").at("action"), where + " output_features.action");
    XS_CHECK(action.size() == 1 && action[0] <= c.max_action_dim, where, ": action must be a vector of at most ",
             c.max_action_dim, " values");
    c.action_dim = action[0];
    XS_CHECK(c.horizon > 0 && c.steps > 0 && c.prompt_tokens > 0, where, ": chunk_size, num_inference_steps and "
             "tokenizer_max_length must be positive");

    if (preprocessor != nullptr && preprocessor->contains("steps")) {
        for (const auto& step : preprocessor->at("steps")) {
            if (step.value("registry_name", "") != "normalizer_processor") continue;
            const auto& features = step.at("config").value("features", nlohmann::json::object());
            if (!features.empty()) {
                fail("policy_preprocessor.json: normalization statistics (", features.size(),
                     " features) are not supported yet; this checkpoint needs them");
            }
        }
    }
    return c;
}

}  // namespace xs::pi05
