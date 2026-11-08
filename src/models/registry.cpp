#include <algorithm>

#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "models/pi05/pi05.h"
#include "models/policy.h"
#include "models/qwen3_5/qwen3_5.h"
#include "models/model.h"
#include "util/error.h"

namespace xs {

std::vector<std::string> supported_model_types() {
    return {"gemma4", "gemma4_text", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"};
}

std::unique_ptr<Model> load_model(const std::filesystem::path& dir, const LoadOptions& options) {
    if (!std::filesystem::is_directory(dir)) fail("model directory '", dir.string(), "' does not exist");
    const auto config = cfg::read_json(dir / "config.json");
    const std::string type = cfg::get_string(config, "model_type", "config.json");
    if (type == "gemma4" || type == "gemma4_text") return gemma4::load(dir, config, options);
    if (type == "qwen3_5" || type == "qwen3_5_text" || type == "qwen3_5_moe" || type == "qwen3_5_moe_text") {
        return qwen3_5::load(dir, config, options);
    }
    std::string list;
    for (const auto& t : supported_model_types()) list += (list.empty() ? "" : ", ") + t;
    fail("model_type '", type, "' is not supported (supported: ", list, ")");
}

ModelKind model_kind(const std::filesystem::path& dir) {
    const auto config = cfg::read_json(dir / "config.json");
    // LeRobot policies name their type under "type"; Hugging Face models under "model_type".
    return config.contains("type") && !config.contains("model_type") ? ModelKind::Policy : ModelKind::Language;
}

std::vector<std::string> supported_policy_types() { return {"pi05"}; }

std::unique_ptr<Policy> load_policy(const std::filesystem::path& dir) {
    if (!std::filesystem::is_directory(dir)) fail("policy directory '", dir.string(), "' does not exist");
    const auto config = cfg::read_json(dir / "config.json");
    const std::string type = cfg::get_string(config, "type", "config.json");
    if (type == "pi05") return pi05::load(dir, config);
    std::string list;
    for (const auto& t : supported_policy_types()) list += (list.empty() ? "" : ", ") + t;
    fail("policy type '", type, "' is not supported (supported: ", list, ")");
}

std::vector<int> Policy::camera_indices(const Observation& obs) const {
    const auto& cams = spec().cameras;
    std::vector<int> out;
    for (const auto& [name, img] : obs.cameras) {
        const auto it = std::find_if(cams.begin(), cams.end(), [&](const CameraSpec& c) { return c.name == name; });
        if (it == cams.end()) {
            std::string known;
            for (const auto& c : cams) known += (known.empty() ? "" : ", ") + c.name;
            fail("camera '", name, "' is not one of the policy's (", known, ")");
        }
        const int index = static_cast<int>(it - cams.begin());
        XS_CHECK(std::find(out.begin(), out.end(), index) == out.end(), "camera '", name, "' appears twice");
        out.push_back(index);
    }
    XS_CHECK(!out.empty(), "an observation needs at least one camera");
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace xs
