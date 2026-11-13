#include "tokenizer/wordpiece_model.h"

#include <string>

#include "tokenizer/json_fields.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "model";
}  // namespace

WordPieceModel::WordPieceModel(const nlohmann::json& config) : vocab_(config, "WordPiece") {
    const std::string unk = string_field(config, "unk_token", kWhat);
    const auto unk_id = vocab_.id(unk);
    if (!unk_id) fail("tokenizer.json WordPiece unk_token '", unk, "' is not in the vocab");
    unk_id_ = *unk_id;
    if (!config.contains("max_input_chars_per_word") || !config.at("max_input_chars_per_word").is_number_unsigned()) {
        fail("tokenizer.json model: field 'max_input_chars_per_word' must be a non-negative integer");
    }
    max_input_chars_per_word_ = config.at("max_input_chars_per_word").get<size_t>();
    const std::string prefix = string_field(config, "continuing_subword_prefix", kWhat);
    for (const auto& [piece, piece_id] : vocab_.entries()) {
        if (piece.starts_with(prefix)) continuations_.emplace(piece.substr(prefix.size()), piece_id);
    }
}

// Hugging Face's WordPiece::tokenize: greedy longest match first, shortening the candidate a character at a time.
void WordPieceModel::tokenize(std::string_view word, std::vector<int32_t>& out) const {
    size_t chars = 0;
    for (size_t pos = 0; pos < word.size(); pos += utf8::char_length(word, pos)) ++chars;
    if (chars > max_input_chars_per_word_) {
        out.push_back(unk_id_);
        return;
    }
    const size_t first = out.size();
    for (size_t start = 0; start < word.size();) {
        const auto& pieces = start == 0 ? vocab_.entries() : continuations_;
        size_t end = word.size();
        auto match = pieces.end();
        for (; end > start; end = utf8::previous_char_start(word, end)) {
            match = pieces.find(word.substr(start, end - start));
            if (match != pieces.end()) break;
        }
        if (match == pieces.end()) {
            out.resize(first);
            out.push_back(unk_id_);
            return;
        }
        out.push_back(match->second);
        start = end;
    }
}

}  // namespace xs::tok
