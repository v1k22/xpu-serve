// Llama 3 output dialect (Llama 3.1 to 3.3's JSON tool calls; vLLM's llama3_json parser). A reply is text, or tool
// calls: when the request offers tools, a reply whose first character is "{", or the text after <|python_tag|>, is a
// sequence of JSON objects {"name": NAME, "parameters": {...}} ("arguments" read alike) separated by ";", as the chat
// template writes a call. Held until the reply ends: if it is not such a sequence, all of it is content. No thinking.

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

namespace xs::chat {
namespace {

using ordered_json = nlohmann::ordered_json;

constexpr std::string_view kPythonTag = "<|python_tag|>";
constexpr std::string_view kSpace = " \t\r\n";

// The pieces of `text` between top-level ";" (outside JSON strings and brackets).
std::vector<std::string_view> split_calls(std::string_view text) {
    std::vector<std::string_view> out;
    size_t start = 0;
    int depth = 0;
    bool in_string = false, escaped = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '{' || c == '[') {
            ++depth;
        } else if (c == '}' || c == ']') {
            --depth;
        } else if (c == ';' && depth == 0) {
            out.push_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(text.substr(start));
    return out;
}

std::string_view trim(std::string_view s) {
    const size_t first = s.find_first_not_of(kSpace);
    if (first == std::string_view::npos) return {};
    return s.substr(first, s.find_last_not_of(kSpace) - first + 1);
}

// The calls `text` holds, or nothing when any piece is not one.
std::optional<std::vector<ToolCall>> parse_calls(std::string_view text, const std::string& id_prefix) {
    std::vector<ToolCall> calls;
    const std::vector<std::string_view> pieces = split_calls(text);
    for (size_t i = 0; i < pieces.size(); ++i) {
        const std::string_view piece = trim(pieces[i]);
        if (piece.empty() && i + 1 == pieces.size() && i > 0) break;  // a trailing ";"
        const ordered_json call = ordered_json::parse(piece, nullptr, /*allow_exceptions=*/false);
        if (call.is_discarded() || !call.is_object() || !call.contains("name") || !call.at("name").is_string()) {
            return std::nullopt;
        }
        const char* key = call.contains("parameters") ? "parameters" : "arguments";
        if (!call.contains(key)) return std::nullopt;
        ordered_json arguments = call.at(key);
        if (arguments.is_string()) {
            arguments = ordered_json::parse(arguments.get<std::string>(), nullptr, /*allow_exceptions=*/false);
        }
        if (!arguments.is_object()) return std::nullopt;
        calls.push_back({id_prefix + std::to_string(calls.size()), call.at("name").get<std::string>(),
                         arguments.dump(-1, ' ', false, ordered_json::error_handler_t::replace)});
    }
    if (calls.empty()) return std::nullopt;
    return calls;
}

class Llama3Parser final : public OutputParser {
public:
    explicit Llama3Parser(bool tools) : tools_(tools), id_prefix_(call_id_prefix()) {}

    ParsedDelta feed(std::string_view text) override {
        ParsedDelta delta;
        if (!tools_) {
            delta.content.assign(text);
            return delta;
        }
        if (calls_) {
            body_ += text;
            return delta;
        }
        pending_ += text;
        if (at_start_ && !pending_.empty()) {  // vLLM reads the first text as it comes, spaces and all
            at_start_ = false;
            if (pending_.front() == '{') {
                calls_ = true;
                body_ = std::move(pending_);
                pending_.clear();
                return delta;
            }
        }
        if (const size_t tag = pending_.find(kPythonTag); tag != std::string::npos) {
            delta.content = pending_.substr(0, tag);
            body_ = pending_.substr(tag + kPythonTag.size());
            pending_.clear();
            calls_ = tagged_ = true;
            return delta;
        }
        // Hold back an end that may be the tag's start.
        size_t keep = 0;
        for (size_t n = std::min(pending_.size(), kPythonTag.size() - 1); n > 0 && keep == 0; --n) {
            if (kPythonTag.starts_with(std::string_view(pending_).substr(pending_.size() - n))) keep = n;
        }
        delta.content = pending_.substr(0, pending_.size() - keep);
        pending_.erase(0, pending_.size() - keep);
        return delta;
    }

    ParsedDelta finish() override {
        ParsedDelta delta;
        if (!calls_) {
            delta.content = std::move(pending_);
            pending_.clear();
            return delta;
        }
        if (auto calls = parse_calls(body_, id_prefix_)) {
            delta.tool_calls = std::move(*calls);
        } else {
            delta.content = (tagged_ ? std::string(kPythonTag) : std::string()) + body_;
        }
        body_.clear();
        calls_ = false;
        return delta;
    }

private:
    const bool tools_;
    const std::string id_prefix_;
    bool at_start_ = true;
    bool calls_ = false;   // inside the calls: held to the end
    bool tagged_ = false;  // opened by <|python_tag|>
    std::string pending_;  // content not yet passed on (a possible tag's start)
    std::string body_;     // the calls' text
};

}  // namespace

std::unique_ptr<OutputParser> make_llama3_parser(bool tools) { return std::make_unique<Llama3Parser>(tools); }

}  // namespace xs::chat
