#pragma once

// The tokenizer.json model: the vocabulary, and the algorithm that turns one pre-tokenized piece into ids. BPE
// (bpe_model.h) and WordPiece (wordpiece_model.h); others are refused by name.

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

// A model's "vocab" object: piece -> id, and back.
class Vocab {
public:
    // Reads config["vocab"]; `model` names the model type in errors ("BPE").
    Vocab(const nlohmann::json& config, std::string_view model);
    Vocab(const Vocab&) = delete;
    Vocab& operator=(const Vocab&) = delete;

    std::optional<int32_t> id(std::string_view piece) const;
    // The piece of `id`, or nullopt for an id with no vocab entry.
    std::optional<std::string_view> piece(int32_t id) const;
    // One past the largest id.
    int32_t id_limit() const { return static_cast<int32_t>(pieces_.size()); }
    // Every piece and its id; the views stay valid as long as the Vocab.
    const std::unordered_map<std::string_view, int32_t>& entries() const { return ids_; }

private:
    std::vector<char> storage_;  // all pieces, back to back; the views below point into it
    std::vector<std::optional<std::string_view>> pieces_;
    std::unordered_map<std::string_view, int32_t> ids_;
};

class Model {
public:
    virtual ~Model() = default;
    // Appends the ids of one pre-tokenized piece to `out`.
    virtual void tokenize(std::string_view piece, std::vector<int32_t>& out) const = 0;
    virtual std::optional<int32_t> id(std::string_view piece) const = 0;
    // The piece of `id`, or nullopt for an id with no vocab entry.
    virtual std::optional<std::string_view> piece(int32_t id) const = 0;
    // One past the largest vocab id.
    virtual int32_t id_limit() const = 0;
};

// `byte_level`: the pre-tokenizer ends with ByteLevel, so pieces are raw bytes for the model to map through the
// byte-level alphabet.
std::unique_ptr<Model> make_model(const nlohmann::json& config, bool byte_level);

}  // namespace xs::tok
