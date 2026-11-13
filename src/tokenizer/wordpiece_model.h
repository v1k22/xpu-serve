#pragma once

// The tokenizer.json "WordPiece" model (BERT): each pre-tokenized word becomes its longest vocab prefix, then the
// longest continuation piece ("##" + text) of what follows, and so on; a word with no such split, or longer than
// max_input_chars_per_word characters, becomes the unk token.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/model.h"

namespace xs::tok {

class WordPieceModel final : public Model {
public:
    explicit WordPieceModel(const nlohmann::json& config);

    void tokenize(std::string_view word, std::vector<int32_t>& out) const override;
    std::optional<int32_t> id(std::string_view piece) const override { return vocab_.id(piece); }
    std::optional<std::string_view> piece(int32_t id) const override { return vocab_.piece(id); }
    int32_t id_limit() const override { return vocab_.id_limit(); }

private:
    Vocab vocab_;
    // The vocab pieces that start with the continuing-subword prefix, keyed by the text after it.
    std::unordered_map<std::string_view, int32_t> continuations_;
    int32_t unk_id_ = 0;
    size_t max_input_chars_per_word_ = 0;
};

}  // namespace xs::tok
