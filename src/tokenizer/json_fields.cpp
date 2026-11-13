#include "tokenizer/json_fields.h"

#include "util/error.h"

namespace xs::tok {

std::string component_type(const nlohmann::json& component, std::string_view what) {
    if (!component.is_object() || !component.contains("type") || !component.at("type").is_string()) {
        fail("tokenizer.json ", what, " has no 'type'");
    }
    return component.at("type").get<std::string>();
}

void refuse_component(std::string_view what, std::string_view type) {
    fail("tokenizer.json ", what, " '", type, "' is not supported");
}

std::string string_pattern(const nlohmann::json& component, std::string_view what) {
    const std::string type = component_type(component, what);
    if (!component.contains("pattern") || !component.at("pattern").is_object()) {
        fail("tokenizer.json ", what, " '", type, "' has no 'pattern'");
    }
    const auto& pattern = component.at("pattern");
    if (pattern.contains("Regex")) fail("tokenizer.json ", what, " '", type, "' with a Regex pattern is not supported");
    if (!pattern.contains("String") || !pattern.at("String").is_string()) {
        fail("tokenizer.json ", what, " '", type, "' has an unrecognized pattern");
    }
    std::string text = pattern.at("String").get<std::string>();
    if (text.empty()) fail("tokenizer.json ", what, " '", type, "' has an empty pattern");
    return text;
}

std::optional<std::string> regex_pattern(const nlohmann::json& component, std::string_view what) {
    if (!component.contains("pattern") || !component.at("pattern").is_object()) return std::nullopt;
    const auto& pattern = component.at("pattern");
    if (!pattern.contains("Regex")) return std::nullopt;
    if (!pattern.at("Regex").is_string()) fail("tokenizer.json ", what, ": 'Regex' pattern must be a string");
    return pattern.at("Regex").get<std::string>();
}

std::string string_field(const nlohmann::json& component, const char* key, std::string_view what) {
    if (!component.contains(key) || !component.at(key).is_string()) {
        fail("tokenizer.json ", what, ": field '", key, "' must be a string");
    }
    return component.at(key).get<std::string>();
}

bool bool_field(const nlohmann::json& component, const char* key, std::string_view what) {
    if (!component.contains(key) || !component.at(key).is_boolean()) {
        fail("tokenizer.json ", what, ": field '", key, "' must be a boolean");
    }
    return component.at(key).get<bool>();
}

bool bool_field_or(const nlohmann::json& component, const char* key, bool fallback, std::string_view what) {
    if (!component.contains(key) || component.at(key).is_null()) return fallback;
    return bool_field(component, key, what);
}

}  // namespace xs::tok
