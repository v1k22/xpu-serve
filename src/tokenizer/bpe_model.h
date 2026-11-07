#pragma once

// The tokenizer.json "BPE" model: vocab, ranked merges, byte fallback and unk handling. Pieces are already normalized
// and pre-tokenized. For byte-level BPE the model starts from one symbol per byte (its byte-level character).

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class BpeModel {
public:
    explicit BpeModel(const nlohmann::json& config);
    BpeModel(const BpeModel&) = delete;
    BpeModel& operator=(const BpeModel&) = delete;

    // From now on, pieces are raw bytes to map through the byte-level alphabet (a ByteLevel pre-tokenizer).
    void use_byte_level_alphabet();

    // Appends the ids of one pre-tokenized piece to `out`.
    void tokenize(std::string_view piece, std::vector<int32_t>& out) const;

    std::optional<int32_t> id(std::string_view piece) const;
    // The piece of `id`, or nullopt for an id with no vocab entry.
    std::optional<std::string_view> piece(int32_t id) const;
    // One past the largest vocab id.
    int32_t id_limit() const { return static_cast<int32_t>(pieces_.size()); }

private:
    struct Merge {
        int32_t rank;
        int32_t new_id;
    };
    struct Symbol {
        int32_t id;
        int32_t prev;
        int32_t next;
        bool merged_away;
    };

    void load_vocab(const nlohmann::json& vocab);
    void load_merges(const nlohmann::json& merges);
    void add_merge(std::string_view left, std::string_view right, int32_t rank);
    bool has_byte_tokens(std::string_view ch) const;
    std::vector<Symbol> initial_symbols(std::string_view piece) const;
    void merge_all(std::vector<Symbol>& symbols) const;
    const Merge* find_merge(int32_t left, int32_t right) const;

    std::vector<char> storage_;  // all vocab pieces, back to back; views below point into it
    std::vector<std::optional<std::string_view>> pieces_;
    std::unordered_map<std::string_view, int32_t> ids_;
    std::unordered_map<uint64_t, Merge> merges_;  // key: left id << 32 | right id
    std::array<int32_t, 256> byte_ids_{};          // "<0xXX>" ids for byte fallback; -1 when absent
    std::array<int32_t, 256> byte_level_ids_{};    // ids of the byte-level characters
    bool byte_level_ = false;
    std::optional<int32_t> unk_id_;
    bool byte_fallback_ = false;
    bool fuse_unk_ = false;
    bool ignore_merges_ = false;
};

}  // namespace xs::tok
