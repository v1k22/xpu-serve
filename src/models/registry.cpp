#include "models/config_util.h"
#include "models/gemma4/gemma4.h"
#include "models/model.h"
#include "util/error.h"

namespace xs {

std::vector<std::string> supported_model_types() { return {"gemma4", "gemma4_text"}; }

std::unique_ptr<Model> load_model(const std::filesystem::path& dir) {
    if (!std::filesystem::is_directory(dir)) fail("model directory '", dir.string(), "' does not exist");
    const auto config = cfg::read_json(dir / "config.json");
    const std::string type = cfg::get_string(config, "model_type", "config.json");
    if (type == "gemma4" || type == "gemma4_text") return gemma4::load(dir, config);
    std::string list;
    for (const auto& t : supported_model_types()) list += (list.empty() ? "" : ", ") + t;
    fail("model_type '", type, "' is not supported (supported: ", list, ")");
}

}  // namespace xs
