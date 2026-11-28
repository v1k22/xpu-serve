#include "server/anthropic_request.h"

#include <algorithm>
#include <limits>
#include <string>

#include "chat/output_parser.h"
#include "grammar/json_schema.h"
#include "server/request_fields.h"
#include "util/base64.h"

namespace xs::server {
namespace {

using json = RequestJson;
using fields::reject;

// A text block's text, or a block list's text blocks joined (a system prompt, a tool result).
std::string text_of(const json& v, const std::string& param) {
    if (v.is_string()) return v.get<std::string>();
    if (!v.is_array()) reject(param, str_cat("'", param, "' must be a string or an array of text blocks"));
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        const json* type = v[i].is_object() ? fields::find(v[i], "type") : nullptr;
        const json* text = v[i].is_object() ? fields::find(v[i], "text") : nullptr;
        if (type == nullptr || *type != "text" || text == nullptr || !text->is_string()) {
            reject(str_cat(param, "[", i, "]"), str_cat("'", param, "[", i, "]' must be a text block here"));
        }
        out += text->get<std::string>();
    }
    return out;
}

// An image block's bytes: {"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}.
std::vector<uint8_t> image_bytes(const json& block, const std::string& param) {
    const json* source = fields::find(block, "source");
    const json* type = source != nullptr && source->is_object() ? fields::find(*source, "type") : nullptr;
    if (type == nullptr || !type->is_string()) reject(param, str_cat("'", param, ".source' must be an image source"));
    if (*type != "base64") {
        reject(param, str_cat("'", param, ".source.type' ", type->dump(), ": send the image as base64; the server ",
                              "does not fetch images"));
    }
    const json* data = fields::find(*source, "data");
    if (data == nullptr || !data->is_string()) reject(param, str_cat("'", param, ".source.data' must be base64"));
    auto bytes = base64_decode(data->get_ref<const std::string&>());
    if (!bytes.has_value()) reject(param, str_cat("'", param, ".source.data' holds invalid base64"));
    return std::move(*bytes);
}

// One message of the request as the OpenAI-shaped messages a template reads: a user message's tool results first
// (each a tool message), then its text and images; an assistant's text, thinking and tool calls in one message.
void add_message(const json& m, size_t index, json& out, std::vector<MediaPart>& media) {
    const std::string param = str_cat("messages[", index, "]");
    if (!m.is_object()) reject(param, str_cat("'", param, "' must be an object"));
    const json* role = fields::find(m, "role");
    if (role == nullptr || (*role != "user" && *role != "assistant")) {
        reject(param + ".role", str_cat("'", param, ".role' must be \"user\" or \"assistant\""));
    }
    const json* content = fields::find(m, "content");
    if (content == nullptr) reject(param + ".content", str_cat("'", param, ".content' is a required field"));
    if (content->is_string()) {
        out.push_back(json{{"role", *role}, {"content", *content}});
        return;
    }
    if (!content->is_array()) reject(param + ".content", str_cat("'", param, ".content' must be a string or blocks"));
    json parts = json::array();
    std::string text, thinking;
    json calls = json::array();
    for (size_t j = 0; j < content->size(); ++j) {
        const json& block = (*content)[j];
        const std::string bp = str_cat(param, ".content[", j, "]");
        const json* type = block.is_object() ? fields::find(block, "type") : nullptr;
        if (type == nullptr || !type->is_string()) reject(bp, str_cat("'", bp, "' must be a block with a 'type'"));
        const std::string t = type->get<std::string>();
        const json* field_text = fields::find(block, "text");
        if (t == "text") {
            if (field_text == nullptr || !field_text->is_string()) {
                reject(bp, str_cat("'", bp, ".text' must be a string"));
            }
            parts.push_back(json{{"type", "text"}, {"text", *field_text}});
            text += field_text->get<std::string>();
        } else if (t == "image" && *role == "user") {
            media.push_back(MediaPart{bp, Modality::Image, image_bytes(block, bp)});
            parts.push_back(json{{"type", "image"}});
        } else if (t == "tool_result" && *role == "user") {
            const json* id = fields::find(block, "tool_use_id");
            if (id == nullptr || !id->is_string()) reject(bp, str_cat("'", bp, ".tool_use_id' must be a string"));
            const json* result = fields::find(block, "content");
            std::string body = result == nullptr ? "" : text_of(*result, bp + ".content");
            if (fields::boolean(block, "is_error", bp + ".is_error")) body = "Error: " + body;
            out.push_back(json{{"role", "tool"}, {"tool_call_id", *id}, {"content", body}});
        } else if (t == "thinking" && *role == "assistant") {
            const json* th = fields::find(block, "thinking");
            if (th == nullptr || !th->is_string()) reject(bp, str_cat("'", bp, ".thinking' must be a string"));
            thinking += th->get<std::string>();
        } else if (t == "redacted_thinking" && *role == "assistant") {
            continue;  // what it held is not readable text
        } else if (t == "tool_use" && *role == "assistant") {
            const json* id = fields::find(block, "id");
            const json* name = fields::find(block, "name");
            const json* input = fields::find(block, "input");
            if (id == nullptr || !id->is_string() || name == nullptr || !name->is_string()) {
                reject(bp, str_cat("'", bp, "' needs a string 'id' and 'name'"));
            }
            const json arguments = input != nullptr ? *input : json::object();
            calls.push_back(json{{"id", *id},
                                 {"type", "function"},
                                 {"function", {{"name", *name}, {"arguments", arguments}}}});
        } else {
            reject(bp, str_cat("'", bp, "': a ", *role, " message's '", t, "' block is not supported"));
        }
    }
    if (*role == "assistant") {
        json a{{"role", "assistant"}, {"content", text}};
        if (!thinking.empty()) a["reasoning_content"] = thinking;
        if (!calls.empty()) a["tool_calls"] = std::move(calls);
        out.push_back(std::move(a));
    } else if (!parts.empty()) {
        // Text alone as a string, as most templates expect; with an image, the parts.
        const bool media_part =
            std::any_of(parts.begin(), parts.end(), [](const json& p) { return p["type"] != "text"; });
        out.push_back(json{{"role", "user"}, {"content", media_part ? parts : json(text)}});
    }
}

json parse_tools(const json& body) {
    const json* v = fields::find(body, "tools");
    if (v == nullptr) return nullptr;
    if (!v->is_array()) reject("tools", "'tools' must be an array");
    json tools = json::array();
    for (size_t i = 0; i < v->size(); ++i) {
        const json& t = (*v)[i];
        const std::string param = str_cat("tools[", i, "]");
        const json* type = t.is_object() ? fields::find(t, "type") : nullptr;
        if (type != nullptr && *type != "custom") {
            reject(param, str_cat("'", param, "': the server tool ", type->dump(), " is not supported"));
        }
        const json* name = t.is_object() ? fields::find(t, "name") : nullptr;
        const json* schema = t.is_object() ? fields::find(t, "input_schema") : nullptr;
        if (name == nullptr || !name->is_string() || name->get<std::string>().empty() || schema == nullptr ||
            !schema->is_object()) {
            reject(param, str_cat("'", param, "' needs a string 'name' and an object 'input_schema'"));
        }
        json function{{"name", *name}, {"parameters", *schema}};
        if (const json* d = fields::find(t, "description")) function["description"] = *d;
        if (fields::boolean(t, "strict", param + ".strict")) function["strict"] = true;
        tools.push_back(json{{"type", "function"}, {"function", std::move(function)}});
    }
    return tools.empty() ? json(nullptr) : tools;
}

// tool_choice {"type": "auto" | "any" | "tool" (with "name") | "none", "disable_parallel_tool_use"}: any is OpenAI's
// required, tool a named function. Without tools, none.
ToolChoice parse_tool_choice(const json& body, const json& tools, bool& parallel) {
    ToolChoice c;
    c.mode = tools.is_null() ? ToolChoice::Mode::none : ToolChoice::Mode::automatic;
    const json* choice = fields::object(body, "tool_choice");
    if (choice == nullptr) return c;
    const json* type = fields::find(*choice, "type");
    const std::string t = type != nullptr && type->is_string() ? type->get<std::string>() : "";
    if (t == "none") {
        c.mode = ToolChoice::Mode::none;
    } else if (t == "any") {
        c.mode = ToolChoice::Mode::required;
    } else if (t == "tool") {
        const json* name = fields::find(*choice, "name");
        if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
            reject("tool_choice.name", "'tool_choice.name' is required when its type is 'tool'");
        }
        c.mode = ToolChoice::Mode::named;
        c.name = name->get<std::string>();
    } else if (t != "auto") {
        reject("tool_choice.type", "'tool_choice.type' must be \"auto\", \"any\", \"tool\" or \"none\"");
    }
    parallel = !fields::boolean(*choice, "disable_parallel_tool_use", "tool_choice.disable_parallel_tool_use");
    if (c.mode == ToolChoice::Mode::none || tools.is_null()) {
        if (c.mode == ToolChoice::Mode::required || c.mode == ToolChoice::Mode::named) {
            reject("tool_choice", "'tool_choice' of type 'any' or 'tool' needs 'tools'");
        }
        c.mode = ToolChoice::Mode::none;
        return c;
    }
    if (c.mode == ToolChoice::Mode::named &&
        std::none_of(tools.begin(), tools.end(), [&](const json& tool) { return tool["function"]["name"] == c.name; })) {
        reject("tool_choice.name", str_cat("the tool '", c.name, "' in 'tool_choice' is none of the 'tools'"));
    }
    return c;
}

// output_config: format {"type": "json_schema", "schema"} (Anthropic's JSON outputs) and effort (the template's
// reasoning_effort). The deprecated beta field output_format is refused by name, as Anthropic's API does without
// its beta header.
void parse_output_config(const json& body, ChatRequest& r) {
    if (fields::find(body, "output_format") != nullptr) {
        reject("output_format", "'output_format' is the deprecated beta field: use 'output_config.format'");
    }
    const json* config = fields::object(body, "output_config");
    if (config == nullptr) return;
    for (const auto& [key, value] : config->items()) {
        if (key != "format" && key != "effort" && !value.is_null()) {
            reject("output_config." + key, str_cat("'output_config.", key, "' is not supported"));
        }
    }
    if (const json* effort = fields::find(*config, "effort")) {
        if (!effort->is_string()) reject("output_config.effort", "'output_config.effort' must be a string");
        if (!r.template_kwargs.contains("reasoning_effort")) r.template_kwargs["reasoning_effort"] = *effort;
    }
    const json* format = fields::find(*config, "format");
    if (format == nullptr) return;
    if (!format->is_object()) reject("output_config.format", "'output_config.format' must be an object");
    const json* type = fields::find(*format, "type");
    if (type == nullptr || *type != "json_schema") {
        reject("output_config.format.type", "'output_config.format.type' must be \"json_schema\"");
    }
    const json* schema = fields::find(*format, "schema");
    if (schema == nullptr) reject("output_config.format.schema", "'output_config.format.schema' is required");
    try {
        r.options.format = grammar::json_schema_grammar(*schema);
        r.options.format_key = "json_schema:" + schema->dump();
    } catch (const grammar::GrammarError& e) {
        const std::string at = e.path().empty() ? "output_config.format.schema" : "output_config.format.schema." + e.path();
        reject(at, str_cat("'", at, "' ", e.what()));
    }
    const ToolChoice::Mode mode = r.tool_choice.mode;
    if (mode == ToolChoice::Mode::required || mode == ToolChoice::Mode::named) {
        reject("output_config.format", "'output_config.format' is not supported beside a 'tool_choice' that forces a call");
    }
}

}  // namespace

ChatRequest parse_messages_request(const json& body, const SamplingParams& defaults, std::string_view dialect,
                                   bool count_only) {
    ChatRequest r;
    r.messages = json::array();
    if (const json* system = fields::find(body, "system")) {
        r.messages.push_back(json{{"role", "system"}, {"content", text_of(*system, "system")}});
    }
    const json* messages = fields::find(body, "messages");
    if (messages == nullptr || !messages->is_array() || messages->empty()) {
        reject("messages", "'messages' must be a non-empty array");
    }
    for (size_t i = 0; i < messages->size(); ++i) add_message((*messages)[i], i, r.messages, r.media);
    r.tools = parse_tools(body);
    r.tool_choice = parse_tool_choice(body, r.tools, r.parallel_tool_calls);
    if (r.tool_choice.mode == ToolChoice::Mode::none) r.tools = nullptr;
    r.template_kwargs = json::object();
    if (const json* thinking = fields::object(body, "thinking")) {
        const json* type = fields::find(*thinking, "type");
        if (type != nullptr && *type == "enabled") {
            r.template_kwargs["enable_thinking"] = true;
        } else if (type != nullptr && *type == "disabled") {
            r.template_kwargs = chat::direct_answer(dialect).kwargs;
        } else {
            reject("thinking.type", "'thinking.type' must be \"enabled\" or \"disabled\"");
        }
    }
    GenerationOptions& o = r.options;
    o.sampling = parse_sampling(body, defaults);
    o.max_tokens = fields::integer(body, "max_tokens", 1, std::numeric_limits<int32_t>::max());
    if (!o.max_tokens.has_value() && !count_only) reject("max_tokens", "'max_tokens' is a required field");
    if (const json* stops = fields::find(body, "stop_sequences")) {
        if (!stops->is_array()) reject("stop_sequences", "'stop_sequences' must be an array of strings");
        for (const json& s : *stops) {
            if (!s.is_string()) reject("stop_sequences", "'stop_sequences' must be an array of strings");
            o.stop.push_back(s.get<std::string>());
        }
    }
    o.stream = fields::boolean(body, "stream", "stream");
    parse_output_config(body, r);
    return r;
}

}  // namespace xs::server
