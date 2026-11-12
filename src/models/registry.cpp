#include <algorithm>

#include "models/clef/clef.h"
#include "models/config_util.h"
#include "models/decision.h"
#include "models/gemma4/gemma4.h"
#include "models/lfm3/lfm3.h"
#include "models/model.h"
#include "models/pi05/pi05.h"
#include "models/policy.h"
#include "models/pplx_decider/pplx_decider.h"
#include "models/qwen3_5/qwen3_5.h"
#include "util/error.h"

namespace xs {

std::vector<std::string> supported_model_types() {
    return {"gemma4", "gemma4_text", "lfm3", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"};
}

std::unique_ptr<Model> load_model(const std::filesystem::path& dir, const LoadOptions& options) {
    if (!std::filesystem::is_directory(dir)) fail("model directory '", dir.string(), "' does not exist");
    const auto config = cfg::read_json(dir / "config.json");
    const std::string type = cfg::get_string(config, "model_type", "config.json");
    cfg::check_quantization(config);
    if (type == "gemma4" || type == "gemma4_text") return gemma4::load(dir, config, options);
    if (type == "qwen3_5" || type == "qwen3_5_text" || type == "qwen3_5_moe" || type == "qwen3_5_moe_text") {
        return qwen3_5::load(dir, config, options);
    }
    if (type == "lfm3") return lfm3::load(dir, config, options);
    if (config.contains("dflash_config")) {
        fail("'", dir.string(), "' is a block drafter checkpoint: it drafts for the model it was trained with ",
             "(--drafter beside that model; convert --target)");
    }
    std::string list;
    for (const auto& t : supported_model_types()) list += (list.empty() ? "" : ", ") + t;
    fail("model_type '", type, "' is not supported (supported: ", list, ")");
}

ModelKind model_kind(const std::filesystem::path& dir) {
    const auto config = cfg::read_json(dir / "config.json");
    // LeRobot policies name their type under "type"; Hugging Face models under "model_type".
    if (config.contains("type") && !config.contains("model_type")) return ModelKind::Policy;
    // A decision model's config.json is its decoder's: the head beside it tells.
    return clef::is_checkpoint(dir) || pplx_decider::is_checkpoint(dir) ? ModelKind::Decision : ModelKind::Language;
}

const char* question_type_name(QuestionType type) {
    switch (type) {
        case QuestionType::Noul: return "noul";
        case QuestionType::Choice: return "choice";
        case QuestionType::Score: return "score";
    }
    return "?";
}

std::vector<std::string> supported_decision_models() {
    return {"clef", "pplx_decider"};
}

std::unique_ptr<DecisionModel> load_decision_model(const std::filesystem::path& dir) {
    if (!std::filesystem::is_directory(dir)) fail("model directory '", dir.string(), "' does not exist");
    const auto config = cfg::read_json(dir / "config.json");
    cfg::check_quantization(config);
    if (clef::is_checkpoint(dir)) return clef::load(dir, config);
    if (pplx_decider::is_checkpoint(dir)) return pplx_decider::load(dir, config);
    std::string list;
    for (const auto& t : supported_decision_models()) list += (list.empty() ? "" : ", ") + t;
    fail("'", dir.string(), "' is not a decision model this build serves (", list, ")");
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
