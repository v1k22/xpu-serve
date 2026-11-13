#pragma once

// tokenizer.json "added_tokens": tokens matched verbatim in the text before the model sees it (chat markers, <bos>).

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class Normalizer;

struct AddedToken {
    int32_t id = 0;
    std::string content;
    bool special = false;
    bool lstrip = false;       // the match also swallows whitespace on its left
    bool rstrip = false;       // ... and on its right
    bool single_word = false;  // only matches when not inside a word
    bool normalized = false;   // matched in normalized text instead of the raw input
    // The text matched, and the token's text (decoding included): content, through the normalizer when `normalized`,
    // as Hugging Face does.
    std::string pattern;
};

// A piece of input: either an added token or plain text for the normalizer and model.
struct TextSegment {
    std::string_view text;
    std::optional<int32_t> token;
};

class AddedVocabulary {
public:
    // `normalizer` (may be null) is the tokenizer's: it rewrites the pattern of each normalized token.
    AddedVocabulary(const nlohmann::json& added_tokens, const Normalizer* normalizer);

    // Splits `text` on the added tokens whose `normalized` flag equals `normalized_pass` (Hugging Face matches the
    // others on raw text first). Special tokens match only with parse_special. Leftmost-longest, non-overlapping.
    std::vector<TextSegment> split(std::string_view text, bool normalized_pass, bool parse_special) const;

    const std::vector<AddedToken>& tokens() const { return tokens_; }
    bool has_normalized_tokens() const { return has_normalized_; }

private:
    struct Match {
        size_t begin;
        size_t end;
        const AddedToken* token;
    };
    std::optional<Match> next_match(std::string_view text, size_t from, bool normalized_pass) const;

    std::vector<AddedToken> tokens_;
    // Per pass (raw, normalized) and first byte: token indices, longest content first.
    std::array<std::array<std::vector<size_t>, 256>, 2> by_first_byte_;
    bool has_normalized_ = false;
};

}  // namespace xs::tok
