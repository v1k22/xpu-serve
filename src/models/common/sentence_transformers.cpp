#include "models/common/sentence_transformers.h"

#include <string>
#include <vector>

#include "models/config_util.h"
#include "util/error.h"

namespace xs::st {
namespace {

namespace fs = std::filesystem;
using nlohmann::json;

struct Module {
    std::string kind;  // the class name: "Transformer", "Pooling", ...
    std::string type;  // as modules.json writes it
    std::string path;  // relative to the checkpoint
};

// A module's class is the last part of its type: sentence-transformers 6 writes
// "sentence_transformers.sentence_transformer.modules.pooling.Pooling", older versions
// "sentence_transformers.models.Pooling". A type outside the package is the checkpoint's own code.
std::vector<Module> read_modules(const fs::path& dir) {
    const json modules = cfg::read_json(dir / "modules.json");
    if (!modules.is_array()) fail("modules.json: must be a list of modules");
    std::vector<Module> out;
    for (const json& m : modules) {
        const std::string where = "modules.json entry " + std::to_string(out.size());
        Module module;
        module.type = cfg::get_string(m, "type", where);
        module.path = m.contains("path") && !m.at("path").is_null() ? cfg::get_string(m, "path", where) : "";
        const size_t dot = module.type.rfind('.');
        module.kind = dot == std::string::npos ? module.type : module.type.substr(dot + 1);
        if (module.type.rfind("sentence_transformers.", 0) != 0) {
            fail("modules.json: module '", module.type, "' is the checkpoint's own code, which is not supported");
        }
        if (m.contains("kwargs") && !m.at("kwargs").is_null() && !m.at("kwargs").empty()) {
            fail("modules.json: the ", module.kind, " module's kwargs are not supported");
        }
        out.push_back(std::move(module));
    }
    return out;
}

// sentence-transformers 6 names the mode in "pooling_mode" (a name or a list of names); older checkpoints set one of
// the pooling_mode_* flags, and with none set the mode is mean.
std::vector<std::string> pooling_modes(const json& p, const std::string& where) {
    if (p.contains("pooling_mode") && !p.at("pooling_mode").is_null()) {
        const json& mode = p.at("pooling_mode");
        if (mode.is_string()) return {mode.get<std::string>()};
        std::vector<std::string> modes;
        if (mode.is_array()) {
            for (const json& m : mode) {
                if (!m.is_string()) fail(where, ": pooling_mode must be a name or a list of names");
                modes.push_back(m.get<std::string>());
            }
        }
        if (modes.empty()) fail(where, ": pooling_mode must be a name or a non-empty list of names");
        return modes;
    }
    static const std::pair<const char*, const char*> kFlags[] = {
        {"pooling_mode_cls_token", "cls"},
        {"pooling_mode_max_tokens", "max"},
        {"pooling_mode_mean_tokens", "mean"},
        {"pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"},
        {"pooling_mode_weightedmean_tokens", "weightedmean"},
        {"pooling_mode_lasttoken", "lasttoken"},
    };
    std::vector<std::string> modes;
    for (const auto& [flag, mode] : kFlags) {
        if (cfg::get_bool_or(p, flag, false, where)) modes.emplace_back(mode);
    }
    if (modes.empty()) modes.emplace_back("mean");
    return modes;
}

void read_pooling(const fs::path& path, Pipeline& out) {
    const std::string where = path.string();
    const json p = cfg::read_json(path);
    const char* dimension_key = p.contains("embedding_dimension") ? "embedding_dimension" : "word_embedding_dimension";
    out.dimension = cfg::get_int(p, dimension_key, where);
    const std::vector<std::string> modes = pooling_modes(p, where);
    if (modes.size() > 1) {
        std::string list;
        for (const std::string& m : modes)
            list += (list.empty() ? "" : ", ") + m;
        fail(where, ": several pooling modes concatenated (", list, ") are not supported");
    }
    if (modes[0] == "mean") {
        out.pooling = Pooling::Mean;
    } else if (modes[0] == "cls") {
        out.pooling = Pooling::Cls;
    } else {
        fail(where, ": pooling mode '", modes[0], "' is not supported (supported: mean, cls)");
    }
}

// tokenizer.json's normalizer lowercases: BertNormalizer with lowercase, Lowercase, or a Sequence holding either.
bool lowercases(const json& normalizer) {
    if (!normalizer.is_object()) return false;
    const std::string type = normalizer.value("type", "");
    if (type == "Lowercase") return true;
    if (type == "BertNormalizer") return normalizer.value("lowercase", false);
    if (type == "Sequence" && normalizer.contains("normalizers") && normalizer.at("normalizers").is_array()) {
        for (const json& n : normalizer.at("normalizers")) {
            if (lowercases(n)) return true;
        }
    }
    return false;
}

void read_transformer(const fs::path& dir, Pipeline& out) {
    const fs::path path = dir / "sentence_bert_config.json";
    if (!fs::exists(path)) return;
    const std::string where = "sentence_bert_config.json";
    const json c = cfg::read_json(path);
    if (c.contains("max_seq_length") && !c.at("max_seq_length").is_null()) {
        out.max_seq_length = cfg::get_int(c, "max_seq_length", where);
        XS_CHECK(*out.max_seq_length > 0, where, ": max_seq_length must be positive, got ", *out.max_seq_length);
    }
    if (c.contains("transformer_task") && !c.at("transformer_task").is_null()) {
        const std::string task = cfg::get_string(c, "transformer_task", where);
        if (task != "feature-extraction") fail(where, ": transformer_task '", task, "' is not supported");
    }
    // sentence-transformers lowercases the text first when the tokenizer would not; this tokenizer must.
    if (cfg::get_bool_or(c, "do_lower_case", false, where)) {
        const json tokenizer = cfg::read_json(dir / "tokenizer.json");
        if (!tokenizer.is_object() || !lowercases(tokenizer.value("normalizer", json()))) {
            fail(where, ": do_lower_case with a tokenizer that does not lowercase is not supported");
        }
    }
}

// A default prompt is text sentence-transformers puts before every input; nothing here would.
void check_model_config(const fs::path& dir) {
    const fs::path path = dir / "config_sentence_transformers.json";
    if (!fs::exists(path)) return;
    const std::string where = "config_sentence_transformers.json";
    const json c = cfg::read_json(path);
    if (c.contains("model_type") && !c.at("model_type").is_null()) {
        const std::string type = cfg::get_string(c, "model_type", where);
        if (type != "SentenceTransformer") fail(where, ": model_type '", type, "' is not supported");
    }
    if (c.contains("default_prompt_name") && !c.at("default_prompt_name").is_null()) {
        fail(where, ": a default prompt ('", cfg::get_string(c, "default_prompt_name", where),
             "') is not supported: put the prompt's text in the input");
    }
}

}  // namespace

bool has_pooling(const fs::path& dir) {
    if (!fs::exists(dir / "modules.json")) return false;
    const json modules = cfg::read_json(dir / "modules.json");
    if (!modules.is_array()) return false;
    for (const json& m : modules) {
        if (!m.is_object() || !m.contains("type") || !m.at("type").is_string()) continue;
        const std::string type = m.at("type").get<std::string>();
        if (type.size() >= 8 && type.compare(type.size() - 8, 8, ".Pooling") == 0) return true;
    }
    return false;
}

Pipeline read_pipeline(const fs::path& dir) {
    const std::vector<Module> modules = read_modules(dir);
    std::string kinds;
    for (const Module& m : modules)
        kinds += (kinds.empty() ? "" : ", ") + m.kind;
    const bool layout = (modules.size() == 2 || modules.size() == 3) && modules[0].kind == "Transformer" &&
                        modules[1].kind == "Pooling" && (modules.size() == 2 || modules[2].kind == "Normalize");
    if (!layout) {
        fail("modules.json: the modules ", kinds, " are not supported (supported: Transformer, Pooling, and Normalize ",
             "after them)");
    }
    if (!modules[0].path.empty()) {
        fail("modules.json: a Transformer module in a folder of its own ('", modules[0].path, "') is not supported");
    }
    Pipeline out;
    read_pooling(dir / modules[1].path / "config.json", out);
    if (modules.size() == 3) {
        out.normalize = true;
        const fs::path path = dir / modules[2].path / "config.json";
        if (!modules[2].path.empty() && fs::exists(path)) {
            const json c = cfg::read_json(path);
            const std::string input = c.contains("module_input_name") && !c.at("module_input_name").is_null()
                                          ? cfg::get_string(c, "module_input_name", path.string())
                                          : "sentence_embedding";
            if (input != "sentence_embedding") {
                fail(path.string(), ": a Normalize module of '", input, "' is not supported (supported: ",
                     "sentence_embedding)");
            }
        }
    }
    read_transformer(dir, out);
    check_model_config(dir);
    return out;
}

}  // namespace xs::st
