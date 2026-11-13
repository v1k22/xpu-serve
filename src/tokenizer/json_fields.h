#pragma once

// Shared readers for tokenizer.json components. `what` names the component kind in errors ("normalizer", ...).

#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace xs::tok {

// The component's "type" field.
std::string component_type(const nlohmann::json& component, std::string_view what);

[[noreturn]] void refuse_component(std::string_view what, std::string_view type);

// A {"String": "..."} pattern; regex patterns are refused by name.
std::string string_pattern(const nlohmann::json& component, std::string_view what);

// The text of a {"Regex": "..."} pattern; nullopt for any other pattern.
std::optional<std::string> regex_pattern(const nlohmann::json& component, std::string_view what);

std::string string_field(const nlohmann::json& component, const char* key, std::string_view what);
bool bool_field(const nlohmann::json& component, const char* key, std::string_view what);
bool bool_field_or(const nlohmann::json& component, const char* key, bool fallback, std::string_view what);

}  // namespace xs::tok
