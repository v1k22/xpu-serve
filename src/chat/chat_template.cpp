#include "chat/chat_template.h"

#include <fstream>
#include <sstream>
#include <utility>

#include <lexer.h>
#include <parser.h>
#include <runtime.h>
#include <value.h>

#include "util/error.h"

namespace xs::chat {
namespace {

namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot read ", path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

nlohmann::json read_tokenizer_config(const fs::path& path) {
    if (!fs::exists(path)) return nlohmann::json::object();
    try {
        return nlohmann::json::parse(read_file(path));
    } catch (const nlohmann::json::exception& e) {
        fail(path.string(), " is not valid JSON: ", e.what());
    }
}

std::string read_template_source(const fs::path& model_dir, const nlohmann::json& tokenizer_config) {
    const fs::path jinja_path = model_dir / "chat_template.jinja";
    if (fs::exists(jinja_path)) return read_file(jinja_path);
    const auto it = tokenizer_config.find("chat_template");
    if (it != tokenizer_config.end() && it->is_string()) return it->get<std::string>();
    fail("no chat template in ", model_dir.string(),
         " (expected chat_template.jinja or a string \"chat_template\" in tokenizer_config.json)");
}

// Hugging Face passes every named special token (bos_token, eos_token, image_token, ...) to the template. A token is
// either a string or an AddedToken object {"content": ...}.
ordered_json special_tokens_of(const nlohmann::json& tokenizer_config) {
    constexpr std::string_view suffix = "_token";
    ordered_json tokens = ordered_json::object();
    for (const auto& [key, value] : tokenizer_config.items()) {
        if (key.size() <= suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        if (value.is_string()) {
            tokens[key] = value;
        } else if (value.is_object() && value.contains("content") && value["content"].is_string()) {
            tokens[key] = value["content"];
        }
    }
    return tokens;
}

// OpenAI clients send tool_calls[].function.arguments as JSON text, while Hugging Face templates expect a mapping;
// like vLLM and llama.cpp, text that parses to a JSON object is replaced by that object. Anything else is left as is.
nlohmann::json with_parsed_tool_arguments(nlohmann::json messages) {
    for (auto& message : messages) {
        if (!message.is_object() || !message.contains("tool_calls") || !message["tool_calls"].is_array()) continue;
        for (auto& call : message["tool_calls"]) {
            if (!call.is_object() || !call.contains("function") || !call["function"].is_object()) continue;
            nlohmann::json& function = call["function"];
            if (!function.contains("arguments") || !function["arguments"].is_string()) continue;
            nlohmann::json parsed = nlohmann::json::parse(function["arguments"].get<std::string>(), nullptr, false);
            if (parsed.is_object()) function["arguments"] = std::move(parsed);
        }
    }
    return messages;
}

class JinjaChatTemplate final : public ChatTemplate {
public:
    JinjaChatTemplate(std::string source, jinja::program program, ordered_json special_tokens)
        : source_(std::move(source)), program_(std::move(program)), special_tokens_(std::move(special_tokens)) {}

    std::string render(const nlohmann::json& messages, const nlohmann::json& tools, bool add_generation_prompt,
                       const nlohmann::json& extra) const override {
        XS_CHECK(messages.is_array(), "chat messages must be a JSON array, got ", messages.type_name());
        XS_CHECK(tools.is_null() || tools.is_array(), "chat tools must be a JSON array or null, got ",
                 tools.type_name());
        XS_CHECK(extra.is_object(), "chat template variables must be a JSON object, got ", extra.type_name());
        try {
            jinja::context ctx(source_);
            jinja::global_from_json(ctx, variables(messages, tools, add_generation_prompt, extra), true);
            jinja::runtime rt(ctx);
            return jinja::runtime::gather_string_parts(rt.execute(program_))->as_string().str();
        } catch (const xs::Error&) {
            throw;
        } catch (const std::exception& e) {
            const std::string_view what = e.what();
            fail("chat template render failed: ", what.empty() ? "the template engine gave no message" : what);
        }
    }

private:
    // Same variables, in the same precedence, as transformers' apply_chat_template: special tokens, then the
    // conversation, then caller kwargs (which override special tokens).
    ordered_json variables(const nlohmann::json& messages, const nlohmann::json& tools, bool add_generation_prompt,
                           const nlohmann::json& extra) const {
        ordered_json vars = special_tokens_;
        vars["messages"] = ordered_json(with_parsed_tool_arguments(messages));
        vars["tools"] = ordered_json(tools);
        vars["documents"] = nullptr;
        vars["add_generation_prompt"] = add_generation_prompt;
        for (const auto& [key, value] : extra.items())
            vars[key] = ordered_json(value);
        return vars;
    }

    std::string source_;
    jinja::program program_;
    ordered_json special_tokens_;
};

}  // namespace

std::unique_ptr<ChatTemplate> ChatTemplate::load(const fs::path& model_dir) {
    const nlohmann::json tokenizer_config = read_tokenizer_config(model_dir / "tokenizer_config.json");
    std::string source = read_template_source(model_dir, tokenizer_config);
    try {
        jinja::lexer lexer;
        jinja::program program = jinja::parse_from_tokens(lexer.tokenize(source));
        return std::make_unique<JinjaChatTemplate>(std::move(source), std::move(program),
                                                   special_tokens_of(tokenizer_config));
    } catch (const std::exception& e) {
        fail("chat template in ", model_dir.string(), " does not parse: ", e.what());
    }
}

}  // namespace xs::chat
