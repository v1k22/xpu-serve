#pragma once

// Checked readers for config.json. Configs are untrusted input: a missing or mistyped field is a named error.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/types.h"

namespace xs::cfg {

nlohmann::json read_json(const std::filesystem::path& path);

// Field accessors: `where` names the object in error messages (e.g. "config.json text_config").
int64_t get_int(const nlohmann::json& obj, const char* key, const std::string& where);
double get_float(const nlohmann::json& obj, const char* key, const std::string& where);
std::string get_string(const nlohmann::json& obj, const char* key, const std::string& where);
bool get_bool_or(const nlohmann::json& obj, const char* key, bool fallback, const std::string& where);
// A field that is absent or null counts as `fallback`.
int64_t get_int_or(const nlohmann::json& obj, const char* key, int64_t fallback, const std::string& where);
// An int or a list of ints (e.g. eos_token_id).
std::vector<int32_t> get_int_list(const nlohmann::json& obj, const char* key, const std::string& where);

// A model's display name: the last component of its directory.
std::string model_id(const std::filesystem::path& model_dir);

// Sampling defaults from generation_config.json (missing file or fields keep SamplingParams defaults).
SamplingParams read_generation_defaults(const std::filesystem::path& model_dir, std::vector<int32_t>* eos_ids);

// The id of tokenizer_config.json's eos_token, looked up in its added_tokens_decoder; nullopt when either is absent.
std::optional<int32_t> tokenizer_eos_id(const std::filesystem::path& model_dir);

}  // namespace xs::cfg
