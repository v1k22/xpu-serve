#pragma once

// tokenizer.json decoders: Replace (string pattern), ByteFallback, ByteLevel, Fuse, Strip, WordPiece and Sequence. Like
// Hugging Face, each decoder maps the list of token strings to a new list; the final text is their concatenation.

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

// What the tokens are read as: a whole text (invalid UTF-8 replaced, its ends stripped where a Strip says so, a
// WordPiece first token without its space), or pieces in the middle of a text, each with the bytes it adds there as
// they are (a byte-level or byte-fallback piece may hold part of a character).
enum class DecodeAs { text, pieces };

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual void decode(std::vector<std::string>& tokens, DecodeAs as) const = 0;
};

// Returns nullptr for a null decoder; refuses unsupported types by name.
std::unique_ptr<Decoder> make_decoder(const nlohmann::json& config);

}  // namespace xs::tok
