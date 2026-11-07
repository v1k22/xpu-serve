#include "server/openai_request.h"

#include <cstdint>
#include <limits>

#include "util/base64.h"
#include "util/error.h"

namespace xs::server {
namespace {

using json = RequestJson;

[[noreturn]] void reject(const std::string& param, const std::string& message) { throw InvalidRequest(message, param); }

// The field's value, or nullptr when it is absent or null.
const json* find_field(const json& object, const char* name) {
    const auto it = object.find(name);
    return it == object.end() || it->is_null() ? nullptr : &*it;
}

float number_field(const json& body, const char* name, float fallback, double min, double max) {
    const json* v = find_field(body, name);
    if (v == nullptr) return fallback;
    if (!v->is_number() || v->get<double>() < min || v->get<double>() > max) {
        reject(name, str_cat("'", name, "' must be a number between ", min, " and ", max));
    }
    return v->get<float>();
}

std::optional<int64_t> integer_field(const json& body, const char* name, int64_t min, int64_t max) {
    const json* v = find_field(body, name);
    if (v == nullptr) return std::nullopt;
    // Parsed JSON stores non-negative integers as unsigned, so both representations are range-checked.
    bool in_range = false;
    if (v->is_number_unsigned()) {
        const uint64_t u = v->get<uint64_t>();
        in_range = u <= uint64_t(max) && (min <= 0 || u >= uint64_t(min));
    } else if (v->is_number_integer()) {
        in_range = v->get<int64_t>() >= min && v->get<int64_t>() <= max;
    }
    if (!in_range) reject(name, str_cat("'", name, "' must be an integer between ", min, " and ", max));
    return v->get<int64_t>();
}

bool bool_field(const json& object, const char* name, const std::string& param) {
    const json* v = find_field(object, name);
    if (v == nullptr) return false;
    if (!v->is_boolean()) reject(param, str_cat("'", param, "' must be a boolean"));
    return v->get<bool>();
}

const json* object_field(const json& body, const char* name) {
    const json* v = find_field(body, name);
    if (v != nullptr && !v->is_object()) reject(name, str_cat("'", name, "' must be an object"));
    return v;
}

SamplingParams parse_sampling(const json& body, const SamplingParams& defaults) {
    SamplingParams s = defaults;
    s.temperature = number_field(body, "temperature", s.temperature, 0.0, 100.0);
    s.top_p = number_field(body, "top_p", s.top_p, 0.0, 1.0);
    s.min_p = number_field(body, "min_p", s.min_p, 0.0, 1.0);
    s.presence_penalty = number_field(body, "presence_penalty", s.presence_penalty, -2.0, 2.0);
    s.frequency_penalty = number_field(body, "frequency_penalty", s.frequency_penalty, -2.0, 2.0);
    s.repetition_penalty = number_field(body, "repetition_penalty", s.repetition_penalty, 0.01, 100.0);
    // -1 is how vLLM and llama.cpp spell "off"; our off is 0.
    if (const auto top_k = integer_field(body, "top_k", -1, std::numeric_limits<int32_t>::max())) {
        s.top_k = *top_k < 0 ? 0 : int(*top_k);
    }
    if (const auto seed = integer_field(body, "seed", 0, std::numeric_limits<int64_t>::max())) s.seed = uint64_t(*seed);
    return s;
}

std::optional<int> parse_max_tokens(const json& body) {
    // max_completion_tokens is the newer spelling and wins when both are sent.
    for (const char* name : {"max_completion_tokens", "max_tokens"}) {
        if (const auto n = integer_field(body, name, 1, std::numeric_limits<int32_t>::max())) return int(*n);
    }
    return std::nullopt;
}

std::vector<std::string> parse_stop(const json& body) {
    const json* v = find_field(body, "stop");
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

void check_single_choice(const json& body) {
    const auto n = integer_field(body, "n", 1, std::numeric_limits<int32_t>::max());
    if (n.has_value() && *n != 1) reject("n", "'n' must be 1: only one choice per request is supported");
}

GenerationOptions parse_generation_options(const json& body, const SamplingParams& defaults) {
    check_single_choice(body);
    GenerationOptions o;
    o.sampling = parse_sampling(body, defaults);
    o.max_tokens = parse_max_tokens(body);
    o.stop = parse_stop(body);
    o.stream = bool_field(body, "stream", "stream");
    if (const json* so = object_field(body, "stream_options")) {
        o.include_usage = bool_field(*so, "include_usage", "stream_options.include_usage");
    }
    return o;
}

// The bytes of an OpenAI image_url part: "data:image/<type>;base64,<payload>". Remote URLs are never fetched.
std::vector<uint8_t> image_url_bytes(const json& part, const std::string& param) {
    const json* image_url = find_field(part, "image_url");
    const json* url = image_url != nullptr && image_url->is_object() ? find_field(*image_url, "url") : image_url;
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

// Validates messages and takes the images out of their content parts.
json parse_messages(const json& body, std::vector<ImagePart>& images) {
    const json* v = find_field(body, "messages");
    if (v == nullptr) reject("messages", "'messages' is a required field");
    if (!v->is_array() || v->empty()) reject("messages", "'messages' must be a non-empty array");
    json messages = *v;
    for (size_t i = 0; i < messages.size(); ++i) {
        json& m = messages[i];
        const std::string param = str_cat("messages[", i, "]");
        if (!m.is_object()) reject(param, str_cat("'", param, "' must be an object"));
        const json* role = find_field(m, "role");
        if (role == nullptr || !role->is_string()) {
            reject(param + ".role", str_cat("'", param, ".role' must be a string"));
        }
        auto content = m.find("content");
        if (content == m.end() || !content->is_array()) continue;
        for (size_t j = 0; j < content->size(); ++j) {
            json& part = (*content)[j];
            const std::string part_param = str_cat(param, ".content[", j, "]");
            const json* type = part.is_object() ? find_field(part, "type") : nullptr;
            if (type == nullptr || !type->is_string()) continue;
            const std::string& t = type->get_ref<const std::string&>();
            if (t == "image_url") {
                images.push_back(ImagePart{part_param, image_url_bytes(part, part_param)});
                part = json{{"type", "image"}};  // what templates need, without megabytes of base64
            } else if (t == "image") {
                reject(part_param, str_cat("'", part_param, "': send images as {\"type\": \"image_url\", ",
                                           "\"image_url\": {\"url\": \"data:image/...;base64,...\"}}"));
            } else if (t == "input_audio" || t == "audio" || t == "video" || t == "video_url" || t == "file") {
                reject(part_param, str_cat("'", part_param, "': content of type '", t, "' is not supported yet"));
            }
        }
    }
    return messages;
}

bool tools_disabled(const json& body) {
    const json* v = find_field(body, "tool_choice");
    if (v == nullptr) return false;
    if (v->is_string() && v->get<std::string>() == "auto") return false;
    if (v->is_string() && v->get<std::string>() == "none") return true;
    if (v->is_object() || (v->is_string() && v->get<std::string>() == "required")) {
        reject("tool_choice", "'tool_choice' only supports \"auto\" and \"none\" for now");
    }
    reject("tool_choice", "'tool_choice' must be \"auto\" or \"none\"");
}

json parse_tools(const json& body) {
    const json* v = find_field(body, "tools");
    if (v != nullptr && !v->is_array()) reject("tools", "'tools' must be an array");
    for (size_t i = 0; v != nullptr && i < v->size(); ++i) {
        if (!(*v)[i].is_object()) reject("tools", str_cat("'tools[", i, "]' must be an object"));
    }
    if (tools_disabled(body) || v == nullptr || v->empty()) return nullptr;
    return *v;
}

void check_response_format(const json& body) {
    const json* v = object_field(body, "response_format");
    if (v == nullptr) return;
    const json* type = find_field(*v, "type");
    if (type == nullptr || !type->is_string()) {
        reject("response_format.type", "'response_format.type' must be a string");
    }
    const std::string t = type->get<std::string>();
    if (t == "text") return;
    if (t == "json_object" || t == "json_schema") {
        reject("response_format", str_cat("response_format type '", t, "' is not supported yet"));
    }
    reject("response_format.type", str_cat("unknown response_format type '", t, "'"));
}

std::string parse_prompt(const json& body) {
    const json* v = find_field(body, "prompt");
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
    r.messages = parse_messages(body, r.images);
    r.tools = parse_tools(body);
    check_response_format(body);
    const json* kwargs = object_field(body, "chat_template_kwargs");
    r.template_kwargs = kwargs != nullptr ? *kwargs : json::object();
    r.options = parse_generation_options(body, defaults);
    return r;
}

CompletionRequest parse_completion_request(const json& body, const SamplingParams& defaults) {
    CompletionRequest r;
    r.prompt = parse_prompt(body);
    r.options = parse_generation_options(body, defaults);
    return r;
}

}  // namespace xs::server
