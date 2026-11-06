#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace xs::chat {

// A model's own Jinja chat template, rendered natively.
class ChatTemplate {
public:
    // Reads <dir>/chat_template.jinja, else the "chat_template" field of <dir>/tokenizer_config.json. Special-token
    // strings (bos_token, eos_token, ...) come from tokenizer_config.json.
    static std::unique_ptr<ChatTemplate> load(const std::filesystem::path& model_dir);
    virtual ~ChatTemplate() = default;

    // messages: OpenAI-style array [{"role": ..., "content": ...}, ...] (content may be a string or a list of parts).
    // tools: array of OpenAI tool definitions, or null. extra: additional template variables (e.g.
    // {"enable_thinking": true}). Throws xs::Error with the template engine's message on failure.
    virtual std::string render(const nlohmann::json& messages, const nlohmann::json& tools, bool add_generation_prompt,
                               const nlohmann::json& extra = nlohmann::json::object()) const = 0;
};

}  // namespace xs::chat
