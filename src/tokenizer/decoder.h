#pragma once

// tokenizer.json decoders: Replace (string pattern), ByteFallback, ByteLevel, Fuse, Strip, WordPiece and Sequence. Like
// Hugging Face, each decoder maps the list of token strings to a new list; the final text is their concatenation.

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual void decode(std::vector<std::string>& tokens) const = 0;
};

// Returns nullptr for a null decoder; refuses unsupported types by name.
std::unique_ptr<Decoder> make_decoder(const nlohmann::json& config);

}  // namespace xs::tok
