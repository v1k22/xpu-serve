#include "tokenizer/added_vocabulary.h"

#include <algorithm>

#include "tokenizer/json_fields.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "added_tokens entry";

// Rust's char::is_alphanumeric plus '_'. Non-ASCII is approximated as "word" unless it is whitespace, which errs
// toward not matching single_word tokens next to non-ASCII punctuation.
bool is_word_char(char32_t c) {
    if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    return !utf8::is_whitespace(c);
}

bool ends_with_word(std::string_view text) {
    return !text.empty() && is_word_char(utf8::decode_at(text, utf8::previous_char_start(text, text.size())));
}

bool starts_with_word(std::string_view text) { return !text.empty() && is_word_char(utf8::decode_at(text, 0)); }

AddedToken parse_token(const nlohmann::json& entry) {
    if (!entry.is_object() || !entry.contains("id") || !entry.at("id").is_number_integer()) {
        fail("tokenizer.json added_tokens entry has no integer 'id'");
    }
    AddedToken token;
    token.id = entry.at("id").get<int32_t>();
    token.content = string_field(entry, "content", kWhat);
    if (token.content.empty()) fail("tokenizer.json added token ", token.id, " has empty content");
    token.special = bool_field_or(entry, "special", false, kWhat);
    token.lstrip = bool_field_or(entry, "lstrip", false, kWhat);
    token.rstrip = bool_field_or(entry, "rstrip", false, kWhat);
    token.single_word = bool_field_or(entry, "single_word", false, kWhat);
    token.normalized = bool_field_or(entry, "normalized", !token.special, kWhat);
    return token;
}
// Applies the token's flags to a raw match, widening it for lstrip/rstrip. Mirrors Hugging Face
// AddedVocabulary::find_matches: a rejected match still consumes its text (matching resumes after it).
bool accept_match(const AddedToken& token, std::string_view text, size_t segment_start, bool parse_special,
                  size_t& begin, size_t& end) {
    if (token.special && !parse_special) return false;
    if (token.single_word) {
        if (ends_with_word(text.substr(0, begin)) || starts_with_word(text.substr(end))) return false;
    }
    if (token.lstrip) {
        begin = std::max(utf8::trim_whitespace_right(text.substr(0, begin)).size(), segment_start);
    }
    if (token.rstrip) end = text.size() - utf8::trim_whitespace_left(text.substr(end)).size();
    return true;
}
}  // namespace

AddedVocabulary::AddedVocabulary(const nlohmann::json& added_tokens) {
    if (!added_tokens.is_null() && !added_tokens.is_array()) fail("tokenizer.json 'added_tokens' must be a list");
    if (added_tokens.is_array()) {
        for (const auto& entry : added_tokens) tokens_.push_back(parse_token(entry));
    }
    for (size_t i = 0; i < tokens_.size(); ++i) {
        const AddedToken& token = tokens_[i];
        has_normalized_ = has_normalized_ || token.normalized;
        by_first_byte_[token.normalized][static_cast<uint8_t>(token.content[0])].push_back(i);
    }
    for (auto& pass : by_first_byte_) {
        for (auto& bucket : pass) {
            std::stable_sort(bucket.begin(), bucket.end(), [this](size_t a, size_t b) {
                return tokens_[a].content.size() > tokens_[b].content.size();
            });
        }
    }
}

std::optional<AddedVocabulary::Match> AddedVocabulary::next_match(std::string_view text, size_t from,
                                                                  bool normalized_pass) const {
    const auto& buckets = by_first_byte_[normalized_pass];
    for (size_t pos = from; pos < text.size(); ++pos) {
        for (const size_t index : buckets[static_cast<uint8_t>(text[pos])]) {
            const std::string& content = tokens_[index].content;
            if (text.compare(pos, content.size(), content) == 0) {
                return Match{pos, pos + content.size(), &tokens_[index]};
            }
        }
    }
    return std::nullopt;
}

std::vector<TextSegment> AddedVocabulary::split(std::string_view text, bool normalized_pass, bool parse_special) const {
    std::vector<TextSegment> segments;
    size_t segment_start = 0;
    for (size_t from = 0;;) {
        std::optional<Match> match = next_match(text, from, normalized_pass);
        if (!match) break;
        from = match->end;
        if (!accept_match(*match->token, text, segment_start, parse_special, match->begin, match->end)) continue;
        if (match->begin > segment_start) {
            segments.push_back({text.substr(segment_start, match->begin - segment_start), std::nullopt});
        }
        segments.push_back({text.substr(match->begin, match->end - match->begin), match->token->id});
        segment_start = match->end;
        from = match->end;
    }
    if (segment_start < text.size()) segments.push_back({text.substr(segment_start), std::nullopt});
    return segments;
}

}  // namespace xs::tok
