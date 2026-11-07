#include "models/config_util.h"

#include <fstream>

#include "util/error.h"

namespace xs::cfg {
namespace {
const nlohmann::json& field(const nlohmann::json& obj, const char* key, const std::string& where) {
    if (!obj.is_object() || !obj.contains(key) || obj.at(key).is_null()) fail(where, ": missing field '", key, "'");
    return obj.at(key);
}
}  // namespace

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) fail("cannot read '", path.string(), "'");
    try {
        return nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        fail("'", path.string(), "' is not valid JSON: ", e.what());
    }
}

int64_t get_int(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_number_integer()) fail(where, ": field '", key, "' must be an integer");
    return v.get<int64_t>();
}

double get_float(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_number()) fail(where, ": field '", key, "' must be a number");
    return v.get<double>();
}

std::string get_string(const nlohmann::json& obj, const char* key, const std::string& where) {
    const auto& v = field(obj, key, where);
    if (!v.is_string()) fail(where, ": field '", key, "' must be a string");
    return v.get<std::string>();
}

bool get_bool_or(const nlohmann::json& obj, const char* key, bool fallback, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return fallback;
    if (!obj.at(key).is_boolean()) fail(where, ": field '", key, "' must be a boolean");
    return obj.at(key).get<bool>();
}

int64_t get_int_or(const nlohmann::json& obj, const char* key, int64_t fallback, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return fallback;
    return get_int(obj, key, where);
}

std::vector<int32_t> get_int_list(const nlohmann::json& obj, const char* key, const std::string& where) {
    if (!obj.contains(key) || obj.at(key).is_null()) return {};
    const auto& v = obj.at(key);
    if (v.is_number_integer()) return {v.get<int32_t>()};
    if (!v.is_array()) fail(where, ": field '", key, "' must be an integer or a list of integers");
    std::vector<int32_t> out;
    for (const auto& e : v) {
        if (!e.is_number_integer()) fail(where, ": field '", key, "' must contain only integers");
        out.push_back(e.get<int32_t>());
    }
    return out;
}

SamplingParams read_generation_defaults(const std::filesystem::path& model_dir, std::vector<int32_t>* eos_ids) {
    SamplingParams p;
    const auto path = model_dir / "generation_config.json";
    if (!std::filesystem::exists(path)) return p;
    const auto j = read_json(path);
    const std::string where = "generation_config.json";
    const bool do_sample = get_bool_or(j, "do_sample", false, where);
    if (!do_sample) {
        p.temperature = 0.0f;
    } else if (j.contains("temperature") && j["temperature"].is_number()) {
        p.temperature = j["temperature"].get<float>();
    }
    if (j.contains("top_k") && j["top_k"].is_number_integer()) p.top_k = j["top_k"].get<int>();
    if (j.contains("top_p") && j["top_p"].is_number()) p.top_p = j["top_p"].get<float>();
    if (j.contains("min_p") && j["min_p"].is_number()) p.min_p = j["min_p"].get<float>();
    if (j.contains("repetition_penalty") && j["repetition_penalty"].is_number()) {
        p.repetition_penalty = j["repetition_penalty"].get<float>();
    }
    if (eos_ids != nullptr) *eos_ids = get_int_list(j, "eos_token_id", where);
    return p;
}

std::optional<int32_t> tokenizer_eos_id(const std::filesystem::path& model_dir) {
    const auto path = model_dir / "tokenizer_config.json";
    if (!std::filesystem::exists(path)) return std::nullopt;
    const auto j = read_json(path);
    if (!j.contains("eos_token")) return std::nullopt;
    const nlohmann::json& eos = j.at("eos_token");
    const std::string text = eos.is_string() ? eos.get<std::string>()
                             : eos.is_object() && eos.contains("content") ? eos.at("content").get<std::string>()
                                                                          : std::string();
    if (text.empty() || !j.contains("added_tokens_decoder") || !j.at("added_tokens_decoder").is_object()) {
        return std::nullopt;
    }
    for (const auto& [id, token] : j.at("added_tokens_decoder").items()) {
        if (token.is_object() && token.value("content", "") == text) return static_cast<int32_t>(std::stol(id));
    }
    return std::nullopt;
}

}  // namespace xs::cfg
