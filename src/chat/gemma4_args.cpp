#include "chat/gemma4_args.h"

#include <nlohmann/json.hpp>

namespace xs::chat::gemma4 {
namespace {

constexpr int kMaxDepth = 64;  // bounds recursion on adversarial nesting

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// JSON number grammar: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
bool is_json_number(std::string_view s) {
    size_t i = 0;
    auto digits = [&] {
        const size_t start = i;
        while (i < s.size() && is_digit(s[i]))
            ++i;
        return i > start;
    };
    if (i < s.size() && s[i] == '-') ++i;
    if (i < s.size() && s[i] == '0') {
        ++i;
    } else if (!digits()) {
        return false;
    }
    if (i < s.size() && s[i] == '.') {
        ++i;
        if (!digits()) return false;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
        if (!digits()) return false;
    }
    return i == s.size();
}

bool is_json_literal(std::string_view s) {
    return s == "true" || s == "false" || s == "null" || is_json_number(s);
}

std::string quote(std::string_view s) {
    return nlohmann::json(std::string(s)).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// Recursive-descent reader that appends the JSON rendering of what it reads to `out_`.
class ArgsReader {
public:
    explicit ArgsReader(std::string_view text) : text_(text) {}

    std::optional<std::string> read_document() {
        skip_space();
        if (!object(0)) return std::nullopt;
        skip_space();
        if (pos_ != text_.size()) return std::nullopt;
        return std::move(out_);
    }

private:
    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return text_[pos_]; }
    bool starts_with(std::string_view s) const { return text_.substr(pos_).starts_with(s); }
    void skip_space() {
        while (!at_end() && is_space(peek()))
            ++pos_;
    }
    bool consume(char c) {
        if (at_end() || peek() != c) return false;
        ++pos_;
        return true;
    }

    bool object(int depth) {
        if (depth > kMaxDepth || !consume('{')) return false;
        out_ += '{';
        skip_space();
        if (consume('}')) {
            out_ += '}';
            return true;
        }
        for (bool first = true;; first = false) {
            if (!first) out_ += ',';
            skip_space();
            const auto k = key();
            if (!k) return false;
            out_ += quote(*k) + ':';
            skip_space();
            if (!consume(':')) return false;
            skip_space();
            if (!value(depth)) return false;
            skip_space();
            if (consume('}')) {
                out_ += '}';
                return true;
            }
            if (!consume(',')) return false;
        }
    }

    bool array(int depth) {
        if (depth > kMaxDepth || !consume('[')) return false;
        out_ += '[';
        skip_space();
        if (consume(']')) {
            out_ += ']';
            return true;
        }
        for (bool first = true;; first = false) {
            if (!first) out_ += ',';
            skip_space();
            if (!value(depth)) return false;
            skip_space();
            if (consume(']')) {
                out_ += ']';
                return true;
            }
            if (!consume(',')) return false;
        }
    }

    bool value(int depth) {
        if (at_end()) return false;
        if (peek() == '{') return object(depth + 1);
        if (peek() == '[') return array(depth + 1);
        if (opens_string()) {
            const auto s = quoted_string();
            if (s) out_ += quote(*s);
            return s.has_value();
        }
        return bare_value();
    }

    std::optional<std::string> key() {
        if (opens_string()) return quoted_string();
        const size_t start = pos_;
        while (!at_end() && peek() != ':' && !is_space(peek()) &&
               std::string_view("{}[],\"").find(peek()) == std::string_view::npos) {
            ++pos_;
        }
        if (pos_ == start) return std::nullopt;
        return std::string(text_.substr(start, pos_ - start));
    }

    bool opens_string() const { return starts_with(kStringDelim) || (!at_end() && peek() == '"'); }

    // <|"|>raw text<|"|> or a JSON "string"; nullopt when neither is here, complete and well-formed.
    std::optional<std::string> quoted_string() {
        if (starts_with(kStringDelim)) {
            const size_t start = pos_ + kStringDelim.size();
            const size_t end = text_.find(kStringDelim, start);
            if (end == std::string_view::npos) return std::nullopt;
            pos_ = end + kStringDelim.size();
            return std::string(text_.substr(start, end - start));
        }
        if (at_end() || peek() != '"') return std::nullopt;
        size_t end = pos_ + 1;
        while (end < text_.size() && text_[end] != '"')
            end += text_[end] == '\\' ? 2 : 1;
        if (end >= text_.size()) return std::nullopt;
        try {
            auto s = nlohmann::json::parse(text_.substr(pos_, end + 1 - pos_)).get<std::string>();
            pos_ = end + 1;
            return s;
        } catch (const nlohmann::json::exception&) {
            return std::nullopt;
        }
    }

    // Numbers, booleans and null are written bare; any other bare run up to the next separator is a string.
    bool bare_value() {
        const size_t start = pos_;
        while (!at_end() && std::string_view(",}]").find(peek()) == std::string_view::npos)
            ++pos_;
        std::string_view token = text_.substr(start, pos_ - start);
        while (!token.empty() && is_space(token.back()))
            token.remove_suffix(1);
        if (token.empty()) return false;
        out_ += is_json_literal(token) ? std::string(token) : quote(token);
        return true;
    }

    std::string_view text_;
    size_t pos_ = 0;
    std::string out_;
};

}  // namespace

std::optional<std::string> args_to_json(std::string_view text) {
    return ArgsReader(text).read_document();
}

}  // namespace xs::chat::gemma4
