#include <algorithm>
#include <filesystem>

#include "models/config_util.h"
#include "models/laya/laya.h"
#include "util/error.h"

namespace xs::laya {
namespace {

namespace fs = std::filesystem;

const char* const kBuckets[] = {"2", "3-5", "6-10", "11+"};

// A special token as tokenizer_config.json names it: its text, or an object with that text as "content".
std::string special_token(const nlohmann::json& tokenizer, const char* key, const std::string& where) {
    XS_CHECK(tokenizer.contains(key), where, " names no ", key);
    const nlohmann::json& t = tokenizer.at(key);
    if (t.is_object()) return cfg::get_string(t, "content", where + " " + key);
    XS_CHECK(t.is_string() && !t.get_ref<const std::string&>().empty(), where, ": ", key, " is not a token");
    return t.get<std::string>();
}

// rl_agent_config.json: what Laya's system_one reads (the rest is training's).
Policy laya_policy(const fs::path& dir, int64_t& head_layers) {
    const std::string where = "rl_agent_config.json";
    const nlohmann::json j = cfg::read_json(dir / where);
    if (const fs::path root = dir / "config.json"; fs::exists(root)) {
        const nlohmann::json c = cfg::read_json(root);
        if (c.contains("model_type") && c.at("model_type") != "laya") {
            fail("config.json: model_type ", c.at("model_type").dump(), " is not Laya's (laya), beside ", where);
        }
    }
    Policy p;
    p.family = "laya";
    head_layers = cfg::get_int(j, "head_layers", where);
    p.max_tokens = cfg::get_int(j, "max_len", where);
    p.head_tokens = cfg::get_int(j, "head_max_len", where);
    if (j.contains("temperature")) {
        const nlohmann::json& t = j.at("temperature");
        XS_CHECK(t.is_array() && t.size() == 3, where, ": temperature is not one value per question type (choice, ",
                 "score, noul)");
        for (size_t i = 0; i < 3; ++i) {
            XS_CHECK(t[i].is_number(), where, ": temperature[", i, "] is not a number");
            p.temperature[i] = t[i].get<double>();
        }
    }
    if (j.contains("temperature_by_options")) {
        const nlohmann::json& t = j.at("temperature_by_options");
        XS_CHECK(t.is_object(), where, ": temperature_by_options is not an object");
        for (auto it = t.begin(); it != t.end(); ++it) {
            const std::string& key = it.key();
            const size_t colon = key.find(':');
            const std::string type = key.substr(0, colon);
            const std::string bucket = colon == std::string::npos ? "" : key.substr(colon + 1);
            bool known = type == "choice" || type == "score" || type == "noul";
            known = known && std::find(std::begin(kBuckets), std::end(kBuckets), bucket) != std::end(kBuckets);
            if (!known) fail(where, ": temperature_by_options key '", key, "' is not a question type and option count");
            XS_CHECK(it.value().is_number(), where, ": temperature_by_options '", key, "' is not a number");
            p.temperature_by_options[key] = it.value().get<double>();
        }
    }
    for (const double t : p.temperature)
        XS_CHECK(t > 0, where, ": a temperature of ", t);
    for (const auto& [key, t] : p.temperature_by_options)
        XS_CHECK(t > 0, where, ": temperature_by_options '", key, "' is ", t);
    return p;
}

// Julia's config.json (the checkpoint's manifest), julia_config.json and inference-policy.json.
Policy julia_policy(const fs::path& dir, int64_t& head_layers) {
    if (const fs::path root = dir / "config.json"; fs::exists(root)) {
        const nlohmann::json c = cfg::read_json(root);
        const std::string where = "config.json";
        if (cfg::get_string(c, "architecture", where) != "JuliaDecisionModel") {
            fail(where, ": architecture '", c.at("architecture").get<std::string>(), "' is not JuliaDecisionModel");
        }
        if (cfg::get_int_or(c, "format_version", 1, where) != 1) fail(where, ": format_version is not 1");
        // The files the reference reads, where it reads them.
        for (const auto& [key, file] : {std::pair{"julia_config_file", "julia_config.json"},
                                        std::pair{"encoder_config_file", "encoder/config.json"},
                                        std::pair{"weights_file", "model.safetensors"},
                                        std::pair{"tokenizer_directory", "tokenizer"}}) {
            if (c.contains(key) && cfg::get_string(c, key, where) != file) {
                fail(where, ": ", key, " '", c.at(key).get<std::string>(), "' is not supported (", file, " is)");
            }
        }
    }
    std::string where = "julia_config.json";
    const nlohmann::json j = cfg::read_json(dir / where);
    if (cfg::get_int(j, "format_version", where) != 1) fail(where, ": format_version is not 1");
    if (cfg::get_string(j, "architecture", where) != "JuliaDecisionModel") {
        fail(where, ": architecture '", j.at("architecture").get<std::string>(), "' is not JuliaDecisionModel");
    }
    head_layers = cfg::get_int(j, "head_layers", where);

    where = "inference-policy.json";
    XS_CHECK(fs::exists(dir / where), "julia: ", (dir / where).string(), " is missing (the checkpoint's limits and ",
             "encoding)");
    const nlohmann::json policy = cfg::read_json(dir / where);
    if (policy.contains("calibration") && !policy.at("calibration").is_null()) {
        fail(where, ": a calibration is not supported");
    }
    Policy p;
    p.family = "julia";
    p.max_tokens = cfg::get_int(policy, "max_length", where);
    p.head_tokens = cfg::get_int(policy, "head_length", where);
    p.strict = cfg::get_bool_or(policy, "strict_encoding", false, where);
    p.descriptions_only = true;
    p.room_for_state = true;
    // julia/data.py sequence(): the question's budget must leave room for the state.
    XS_CHECK(p.head_tokens + 4 < p.max_tokens, where, ": max_length ", p.max_tokens,
             " leaves no room beyond the question's ", p.head_tokens);
    return p;
}

}  // namespace

bool is_checkpoint(const fs::path& dir) {
    return fs::exists(dir / "encoder" / "config.json") &&
           (fs::exists(dir / "rl_agent_config.json") || fs::exists(dir / "julia_config.json"));
}

Config read_config(const fs::path& dir) {
    const bool julia = fs::exists(dir / "julia_config.json");
    if (julia && fs::exists(dir / "rl_agent_config.json")) {
        fail("'", dir.string(), "' holds both Laya's rl_agent_config.json and Julia's julia_config.json");
    }
    Config c;
    c.policy = julia ? julia_policy(dir, c.head_layers) : laya_policy(dir, c.head_layers);
    c.encoder = modern_bert::parse_config(cfg::read_json(dir / "encoder" / "config.json"), "encoder/config.json");
    const Policy& p = c.policy;
    XS_CHECK(c.head_layers >= 0, c.policy.family, ": head_layers ", c.head_layers);
    XS_CHECK(p.head_tokens > 0 && p.max_tokens > p.head_tokens && p.max_tokens <= c.encoder.max_positions,
             p.family, ": a sequence of ", p.max_tokens, " tokens with a question of ", p.head_tokens,
             " does not fit the encoder's ", c.encoder.max_positions, " positions");
    const std::string where = "tokenizer/tokenizer_config.json";
    const nlohmann::json tokenizer = cfg::read_json(dir / where);
    c.cls = special_token(tokenizer, "cls_token", where);
    c.sep = special_token(tokenizer, "sep_token", where);
    c.mask = special_token(tokenizer, "mask_token", where);
    return c;
}

std::string temperature_bucket(QuestionType type, size_t options) {
    const char* size = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
    return str_cat(question_type_name(type), ":", size);
}

}  // namespace xs::laya
