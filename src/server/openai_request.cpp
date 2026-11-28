#include "server/openai_request.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>

#include "grammar/json_schema.h"
#include "grammar/text_grammars.h"
#include "server/request_fields.h"
#include "util/base64.h"
#include "util/error.h"

namespace xs::server {
namespace {

using json = RequestJson;

using fields::reject;

}  // namespace

SamplingParams parse_sampling(const json& body, const SamplingParams& defaults) {
    SamplingParams s = defaults;
    s.temperature = fields::number(body, "temperature", s.temperature, 0.0, 100.0);
    s.top_p = fields::number(body, "top_p", s.top_p, 0.0, 1.0);
    s.min_p = fields::number(body, "min_p", s.min_p, 0.0, 1.0);
    s.presence_penalty = fields::number(body, "presence_penalty", s.presence_penalty, -2.0, 2.0);
    s.frequency_penalty = fields::number(body, "frequency_penalty", s.frequency_penalty, -2.0, 2.0);
    s.repetition_penalty = fields::number(body, "repetition_penalty", s.repetition_penalty, 0.01, 100.0);
    // -1 is how vLLM and llama.cpp spell "off"; our off is 0.
    if (const auto top_k = fields::integer(body, "top_k", -1, std::numeric_limits<int32_t>::max())) {
        s.top_k = *top_k < 0 ? 0 : int(*top_k);
    }
    if (const auto seed = fields::integer(body, "seed", 0, std::numeric_limits<int64_t>::max())) {
        s.seed = uint64_t(*seed);
    }
    return s;
}

namespace {

std::optional<int> parse_max_tokens(const json& body) {
    // max_completion_tokens is the newer spelling and wins when both are sent.
    for (const char* name : {"max_completion_tokens", "max_tokens"}) {
        if (const auto n = fields::integer(body, name, 1, std::numeric_limits<int32_t>::max())) return int(*n);
    }
    return std::nullopt;
}

std::vector<std::string> parse_stop(const json& body) {
    const json* v = fields::find(body, "stop");
    if (v == nullptr) return {};
    if (v->is_string()) return {v->get<std::string>()};
    if (!v->is_array()) reject("stop", "'stop' must be a string or an array of strings");
    std::vector<std::string> stops;
    for (const json& s : *v) {
        if (!s.is_string()) reject("stop", "'stop' must be a string or an array of strings");
        stops.push_back(s.get<std::string>());
    }
    return stops;
}

// n, as vLLM takes it: 1 to kMaxChoices, more than one only when sampling (a greedy reply would repeat itself).
// best_of and beam search are refused by name.
int parse_choices(const json& body, const SamplingParams& sampling) {
    for (const char* name : {"best_of", "use_beam_search"}) {
        if (fields::find(body, name) != nullptr) reject(name, str_cat("'", name, "' is not supported"));
    }
    const auto n = fields::integer(body, "n", 1, kMaxChoices);
    if (!n.has_value() || *n == 1) return 1;
    if (sampling.temperature < 1e-5f) {
        reject("n", str_cat("n must be 1 when using greedy sampling, got ", *n));
    }
    return static_cast<int>(*n);
}

// A schema compiled, or refused at its keyword's path below `param`.
OutputFormat schema_format(const json& schema, const std::string& param) {
    try {
        return {grammar::json_schema_grammar(schema), "json_schema:" + schema.dump()};
    } catch (const grammar::GrammarError& e) {
        const std::string at = e.path().empty() ? param : str_cat(param, ".", e.path());
        reject(at, str_cat("'", at, "' ", e.what()));
    }
}

OutputFormat any_object_format() {
    static const std::shared_ptr<const grammar::Grammar> any_object = grammar::json_object_grammar();
    return {any_object, "json_object"};
}

// response_format: {"type": "text"}, {"type": "json_object"}, or {"type": "json_schema", "json_schema": {"name",
// "schema", "strict", "description"}}. A schema the grammar compiler does not take is refused at the keyword's path.
OutputFormat parse_response_format(const json& body) {
    const json* v = fields::object(body, "response_format");
    if (v == nullptr) return {};
    const json* type = fields::find(*v, "type");
    if (type == nullptr || !type->is_string()) {
        reject("response_format.type", "'response_format.type' must be a string");
    }
    const std::string t = type->get<std::string>();
    if (t == "text") return {};
    if (t == "json_object") return any_object_format();
    if (t != "json_schema") {
        reject("response_format.type",
               str_cat("unknown response_format type '", t, "' (text, json_object and json_schema are known)"));
    }
    const json* spec = fields::find(*v, "json_schema");
    if (spec == nullptr || !spec->is_object()) {
        reject("response_format.json_schema", "'response_format.json_schema' must be an object holding the 'schema'");
    }
    for (const char* text : {"name", "description"}) {
        const json* field = fields::find(*spec, text);
        if (field != nullptr && !field->is_string()) {
            const std::string param = str_cat("response_format.json_schema.", text);
            reject(param, str_cat("'", param, "' must be a string"));
        }
    }
    fields::boolean(*spec, "strict", "response_format.json_schema.strict");
    const json* schema = fields::find(*spec, "schema");
    constexpr std::string_view kSchema = "response_format.json_schema.schema";
    if (schema == nullptr) reject(std::string(kSchema), str_cat("'", kSchema, "' is required"));
    return schema_format(*schema, std::string(kSchema));
}

// A grammar in Lark's syntax, which vLLM tells from EBNF by the absence of `::=` (comments aside).
bool lark_like(std::string_view grammar) {
    size_t at = 0;
    while (at < grammar.size()) {
        const size_t end = std::min(grammar.find('\n', at), grammar.size());
        std::string_view line = grammar.substr(at, end - at);
        for (const std::string_view comment : {"#", "//"}) {
            if (const size_t c = line.find(comment); c != std::string_view::npos) line = line.substr(0, c);
        }
        if (line.find("::=") != std::string_view::npos) return false;
        at = end + 1;
    }
    return true;
}

// vLLM's structured_outputs: exactly one of json (a schema, or its text), regex, choice, grammar (EBNF, xgrammar's
// dialect) and json_object. Its other constraints and options are refused by name.
OutputFormat parse_structured_outputs(const json& body) {
    const json* v = fields::object(body, "structured_outputs");
    if (v == nullptr) return {};
    const std::set<std::string> kinds = {"json", "regex", "choice", "grammar", "json_object"};
    std::vector<std::string> given;
    for (const auto& [key, value] : v->items()) {
        if (value.is_null()) continue;
        const std::string param = "structured_outputs." + key;
        if (key == "structural_tag") reject(param, str_cat("'", param, "' is not supported"));
        if (!kinds.contains(key)) {
            reject(param, str_cat("'", param, "' is not supported (json, regex, choice, grammar and json_object are)"));
        }
        given.push_back(key);
    }
    if (given.size() != 1) {
        reject("structured_outputs", "'structured_outputs' must hold exactly one of json, regex, choice, grammar and "
                                     "json_object");
    }
    const std::string& kind = given.front();
    const std::string param = "structured_outputs." + kind;
    const json& value = v->at(kind);
    const auto compiled = [&](const std::function<std::shared_ptr<const grammar::Grammar>()>& compile,
                              std::string key) -> OutputFormat {
        try {
            return {compile(), std::move(key)};
        } catch (const grammar::GrammarError& e) {
            reject(param, str_cat("'", param, "' ", e.what(), e.path().empty() ? "" : str_cat(" (", e.path(), ")")));
        }
    };
    if (kind == "json") {
        if (value.is_string()) {
            const json schema = json::parse(value.get<std::string>(), nullptr, /*allow_exceptions=*/false);
            if (schema.is_discarded()) reject(param, str_cat("'", param, "' holds a string that is not JSON"));
            return schema_format(schema, param);
        }
        return schema_format(value, param);
    }
    if (kind == "json_object") {
        if (!value.is_boolean() || !value.get<bool>()) reject(param, str_cat("'", param, "' must be true if set"));
        return any_object_format();
    }
    if (kind == "regex") {
        if (!value.is_string()) reject(param, str_cat("'", param, "' must be a string"));
        const std::string pattern = value.get<std::string>();
        return compiled([&] { return grammar::regex_grammar(pattern); }, "regex:" + pattern);
    }
    if (kind == "choice") {
        if (!value.is_array() || value.empty()) reject(param, str_cat("'", param, "' must be a non-empty array of strings"));
        std::vector<std::string> options;
        for (const json& o : value) {
            if (!o.is_string()) reject(param, str_cat("'", param, "' must be a non-empty array of strings"));
            options.push_back(o.get<std::string>());
        }
        return compiled([&] { return grammar::choice_grammar(options); }, "choice:" + value.dump());
    }
    // grammar
    if (!value.is_string() || value.get<std::string>().empty()) {
        reject(param, str_cat("'", param, "' must be a non-empty string"));
    }
    const std::string text = value.get<std::string>();
    if (lark_like(text)) {
        reject(param, str_cat("'", param, "' reads as a Lark grammar (no '::='), which is not supported: write EBNF, ",
                              "rules as name ::= body with a 'root' rule"));
    }
    return compiled([&] { return grammar::ebnf_grammar(text); }, "grammar:" + text);
}

// The constraint fields vLLM removed (they now do nothing there): refused, pointing to structured_outputs.
void refuse_guided_fields(const json& body) {
    for (const char* name : {"guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_whitespace_pattern",
                             "guided_decoding_backend"}) {
        if (fields::find(body, name) != nullptr) {
            reject(name, str_cat("'", name, "' is not supported: use 'structured_outputs' (vLLM removed the guided_ fields)"));
        }
    }
}

// Fields of vLLM's log-probability extensions, which this server does not take.
void refuse_logprob_extensions(const json& body) {
    for (const char* name : {"prompt_logprobs", "logprob_token_ids", "return_tokens_as_token_ids"}) {
        if (fields::find(body, name) != nullptr) {
            reject(name, str_cat("'", name, "' is not supported: only the generated tokens' log-probabilities are"));
        }
    }
}

// Chat's logprobs (a boolean) and top_logprobs (0 to kMaxLogprobs alternatives a token, with logprobs on).
int parse_chat_logprobs(const json& body) {
    refuse_logprob_extensions(body);
    const bool on = fields::boolean(body, "logprobs", "logprobs");
    const auto top = fields::integer(body, "top_logprobs", 0, kMaxLogprobs);
    if (top.has_value() && *top > 0 && !on) {
        reject("top_logprobs", "when using 'top_logprobs', 'logprobs' must be set to true");
    }
    return on ? static_cast<int>(top.value_or(0)) : -1;
}

// A text completion's logprobs: 0 to kMaxLogprobs alternatives a token; absent: none. echo is refused.
int parse_text_logprobs(const json& body) {
    refuse_logprob_extensions(body);
    if (fields::boolean(body, "echo", "echo")) reject("echo", "'echo' is not supported: the reply holds the completion only");
    const auto n = fields::integer(body, "logprobs", 0, kMaxLogprobs);
    return n.has_value() ? static_cast<int>(*n) : -1;
}

GenerationOptions parse_generation_options(const json& body, const SamplingParams& defaults) {
    GenerationOptions o;
    o.sampling = parse_sampling(body, defaults);
    o.choices = parse_choices(body, o.sampling);
    o.max_tokens = parse_max_tokens(body);
    o.stop = parse_stop(body);
    o.stream = fields::boolean(body, "stream", "stream");
    if (const json* so = fields::object(body, "stream_options")) {
        o.include_usage = fields::boolean(*so, "include_usage", "stream_options.include_usage");
    }
    refuse_guided_fields(body);
    OutputFormat format = parse_response_format(body);
    OutputFormat structured = parse_structured_outputs(body);
    if (format.grammar && structured.grammar) {
        reject("structured_outputs", "give 'response_format' or 'structured_outputs', not both");
    }
    if (structured.grammar) format = std::move(structured);
    o.format = std::move(format.grammar);
    o.format_key = std::move(format.key);
    return o;
}

// The bytes of an OpenAI image_url part: "data:image/<type>;base64,<payload>". Remote URLs are never fetched.
std::vector<uint8_t> image_url_bytes(const json& part, const std::string& param) {
    const json* image_url = fields::find(part, "image_url");
    const json* url = image_url != nullptr && image_url->is_object() ? fields::find(*image_url, "url") : image_url;
    if (url == nullptr || !url->is_string()) {
        reject(param, str_cat("'", param, ".image_url' must be a string or an object with a string 'url'"));
    }
    const std::string& text = url->get_ref<const std::string&>();
    if (text.rfind("data:", 0) != 0) {
        reject(param, str_cat("'", param, "' must be a data: URL (data:image/png;base64,...); the server does not ",
                              "fetch remote images"));
    }
    const size_t comma = text.find(',');
    const std::string_view header = std::string_view(text).substr(5, comma == std::string::npos ? 0 : comma - 5);
    const bool base64 = header.size() >= 7 && header.substr(header.size() - 7) == ";base64";
    if (comma == std::string::npos || !base64 || header.rfind("image/", 0) != 0) {
        reject(param, str_cat("'", param, "' must be a base64 image data: URL (data:image/<type>;base64,...)"));
    }
    auto bytes = base64_decode(std::string_view(text).substr(comma + 1));
    if (!bytes.has_value()) reject(param, str_cat("'", param, "' holds invalid base64"));
    return std::move(*bytes);
}

// The bytes of an OpenAI input_audio part: {"input_audio": {"data": "<base64 of the file>", "format": "wav"}}.
std::vector<uint8_t> input_audio_bytes(const json& part, const std::string& param) {
    const json* audio = fields::find(part, "input_audio");
    if (audio == nullptr || !audio->is_object()) {
        reject(param, str_cat("'", param, ".input_audio' must be an object with 'data' and 'format'"));
    }
    const json* data = fields::find(*audio, "data");
    const json* format = fields::find(*audio, "format");
    if (data == nullptr || !data->is_string()) {
        reject(param, str_cat("'", param, ".input_audio.data' must be a base64 string"));
    }
    if (format != nullptr) {
        const std::string f = format->is_string() ? format->get<std::string>() : "";
        if (f != "wav" && f != "mp3" && f != "flac") {
            reject(param, str_cat("'", param, ".input_audio.format' must be wav, mp3 or flac"));
        }
    }
    auto bytes = base64_decode(data->get_ref<const std::string&>());
    if (!bytes.has_value()) reject(param, str_cat("'", param, ".input_audio.data' holds invalid base64"));
    return std::move(*bytes);
}

// An assistant message's thinking, as `reasoning` (vLLM's field) or `reasoning_content` (its earlier one): a string,
// `reasoning` winning when both are given. As in vLLM, the template sees it under both names, and another role's
// message under neither.
void reasoning_of(json& m, const std::string& param) {
    std::optional<std::string> text;
    for (const char* key : {"reasoning_content", "reasoning"}) {
        const json* v = fields::find(m, key);
        if (v == nullptr || v->is_null()) continue;
        if (!v->is_string()) reject(param + "." + key, str_cat("'", param, ".", key, "' must be a string"));
        text = v->get<std::string>();
    }
    if (!text) return;
    if (m.at("role") != "assistant") {
        m.erase("reasoning");
        m.erase("reasoning_content");
        return;
    }
    m["reasoning"] = *text;
    m["reasoning_content"] = *text;
}

// Validates messages and takes the media items out of their content parts.
json parse_messages(const json& body, std::vector<MediaPart>& media) {
    const json* v = fields::find(body, "messages");
    if (v == nullptr) reject("messages", "'messages' is a required field");
    if (!v->is_array() || v->empty()) reject("messages", "'messages' must be a non-empty array");
    json messages = *v;
    for (size_t i = 0; i < messages.size(); ++i) {
        json& m = messages[i];
        const std::string param = str_cat("messages[", i, "]");
        if (!m.is_object()) reject(param, str_cat("'", param, "' must be an object"));
        const json* role = fields::find(m, "role");
        if (role == nullptr || !role->is_string()) {
            reject(param + ".role", str_cat("'", param, ".role' must be a string"));
        }
        reasoning_of(m, param);
        auto content = m.find("content");
        if (content == m.end() || !content->is_array()) continue;
        for (size_t j = 0; j < content->size(); ++j) {
            json& part = (*content)[j];
            const std::string part_param = str_cat(param, ".content[", j, "]");
            const json* type = part.is_object() ? fields::find(part, "type") : nullptr;
            if (type == nullptr || !type->is_string()) continue;
            const std::string& t = type->get_ref<const std::string&>();
            if (t == "image_url") {
                media.push_back(MediaPart{part_param, Modality::Image, image_url_bytes(part, part_param)});
                part = json{{"type", "image"}};  // what templates need, without megabytes of base64
            } else if (t == "input_audio") {
                media.push_back(MediaPart{part_param, Modality::Audio, input_audio_bytes(part, part_param)});
                part = json{{"type", "audio"}};
            } else if (t == "image") {
                reject(part_param, str_cat("'", part_param, "': send images as {\"type\": \"image_url\", ",
                                           "\"image_url\": {\"url\": \"data:image/...;base64,...\"}}"));
            } else if (t == "audio" || t == "audio_url") {
                reject(part_param, str_cat("'", part_param, "': send audio as {\"type\": \"input_audio\", ",
                                           "\"input_audio\": {\"data\": \"<base64>\", \"format\": \"wav\"}}"));
            } else if (t == "video" || t == "video_url" || t == "file") {
                reject(part_param, str_cat("'", part_param, "': content of type '", t, "' is not supported yet"));
            }
        }
    }
    return messages;
}

// tools: an array of {"type": "function", "function": {"name", "description", "parameters", "strict"}}.
json parse_tools(const json& body) {
    const json* v = fields::find(body, "tools");
    if (v == nullptr) return nullptr;
    if (!v->is_array()) reject("tools", "'tools' must be an array");
    for (size_t i = 0; i < v->size(); ++i) {
        const std::string param = str_cat("tools[", i, "]");
        const json& tool = (*v)[i];
        if (!tool.is_object()) reject(param, str_cat("'", param, "' must be an object"));
        const json* type = fields::find(tool, "type");
        if (type != nullptr && *type != "function") {
            reject(param + ".type", str_cat("'", param, ".type' must be \"function\""));
        }
        const json* function = fields::find(tool, "function");
        if (function == nullptr || !function->is_object()) {
            reject(param + ".function", str_cat("'", param, ".function' must be an object"));
        }
        const json* name = fields::find(*function, "name");
        if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
            reject(param + ".function.name", str_cat("'", param, ".function.name' must be a non-empty string"));
        }
        fields::boolean(*function, "strict", param + ".function.strict");
        const json* parameters = fields::find(*function, "parameters");
        if (parameters != nullptr && !parameters->is_object()) {
            reject(param + ".function.parameters", str_cat("'", param, ".function.parameters' must be an object"));
        }
    }
    return v->empty() ? json(nullptr) : *v;
}

// tool_choice: "none", "auto", "required" or {"type": "function", "function": {"name": ...}}, resolved as vLLM does:
// with tools and no tool_choice, auto; without tools, none (anything but none or auto is then refused).
ToolChoice parse_tool_choice(const json& body, const json& tools) {
    const json* v = fields::find(body, "tool_choice");
    ToolChoice c;
    c.mode = tools.is_null() ? ToolChoice::Mode::none : ToolChoice::Mode::automatic;
    if (v == nullptr) return c;
    if (v->is_string()) {
        const std::string& s = v->get_ref<const std::string&>();
        if (s == "none" || (s == "auto" && tools.is_null())) {
            c.mode = ToolChoice::Mode::none;
            return c;
        }
        if (s == "auto") return c;
        if (s != "required") {
            reject("tool_choice", str_cat("invalid value for 'tool_choice': '", s, "'; it must be \"none\", \"auto\", ",
                                          "\"required\" or a named function"));
        }
        c.mode = ToolChoice::Mode::required;
    } else if (v->is_object()) {
        const json* type = fields::find(*v, "type");
        if (type == nullptr || *type != "function") {
            reject("tool_choice", "'tool_choice' as an object must be {\"type\": \"function\", \"function\": {\"name\": ...}}");
        }
        const json* function = fields::find(*v, "function");
        if (function == nullptr || !function->is_object()) {
            reject("tool_choice.function", "'tool_choice.function' must be an object with the function's 'name'");
        }
        const json* name = fields::find(*function, "name");
        if (name == nullptr || !name->is_string() || name->get<std::string>().empty()) {
            reject("tool_choice.function.name", "'tool_choice.function.name' must be a non-empty string");
        }
        c.mode = ToolChoice::Mode::named;
        c.name = name->get<std::string>();
    } else {
        reject("tool_choice", "'tool_choice' must be a string or an object");
    }
    if (tools.is_null()) reject("tool_choice", "when using 'tool_choice', 'tools' must be set");
    if (c.mode == ToolChoice::Mode::named &&
        std::none_of(tools.begin(), tools.end(), [&](const json& t) { return t["function"]["name"] == c.name; })) {
        reject("tool_choice", str_cat("the tool '", c.name, "' in 'tool_choice' is none of the 'tools'"));
    }
    return c;
}

std::string parse_prompt(const json& body) {
    const json* v = fields::find(body, "prompt");
    if (v == nullptr) reject("prompt", "'prompt' is a required field");
    if (v->is_string()) return v->get<std::string>();
    if (v->is_array() && v->size() == 1 && (*v)[0].is_string()) return (*v)[0].get<std::string>();
    reject("prompt", "'prompt' must be a string (or an array holding one string)");
}

}  // namespace

json parse_json_body(std::string_view body) {
    json parsed = json::parse(body, nullptr, /*allow_exceptions=*/false);
    if (parsed.is_discarded()) reject("", "the request body is not valid JSON");
    if (!parsed.is_object()) reject("", "the request body must be a JSON object");
    return parsed;
}

ChatRequest parse_chat_request(const json& body, const SamplingParams& defaults) {
    ChatRequest r;
    r.messages = parse_messages(body, r.media);
    r.tools = parse_tools(body);
    r.tool_choice = parse_tool_choice(body, r.tools);
    if (r.tool_choice.mode == ToolChoice::Mode::none) r.tools = nullptr;  // tool_choice none: the model sees no tools
    r.parallel_tool_calls = !fields::find(body, "parallel_tool_calls") ||
                            fields::boolean(body, "parallel_tool_calls", "parallel_tool_calls");
    const json* kwargs = fields::object(body, "chat_template_kwargs");
    r.template_kwargs = kwargs != nullptr ? *kwargs : json::object();
    // OpenAI's reasoning_effort reaches the template under its own name (Inkling 2's reads it), unless the kwargs
    // already set it.
    if (body.contains("reasoning_effort") && !body.at("reasoning_effort").is_null()) {
        if (!body.at("reasoning_effort").is_string()) reject("reasoning_effort", "reasoning_effort must be a string");
        if (!r.template_kwargs.contains("reasoning_effort")) {
            r.template_kwargs["reasoning_effort"] = body.at("reasoning_effort");
        }
    }
    r.options = parse_generation_options(body, defaults);
    r.options.logprobs = parse_chat_logprobs(body);
    const bool forced = r.tool_choice.mode == ToolChoice::Mode::required || r.tool_choice.mode == ToolChoice::Mode::named;
    if (r.options.format && forced) {
        reject("response_format", "'response_format' is not supported beside a 'tool_choice' that forces a call: the "
                                  "reply is the call");
    }
    return r;
}

CompletionRequest parse_completion_request(const json& body, const SamplingParams& defaults) {
    CompletionRequest r;
    r.prompt = parse_prompt(body);
    r.options = parse_generation_options(body, defaults);
    r.options.logprobs = parse_text_logprobs(body);
    return r;
}

}  // namespace xs::server
