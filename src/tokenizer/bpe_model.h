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

#include "tokenizer/model.h"

namespace xs::tok {

class BpeModel final : public Model {
public:
    // `byte_level`: pieces are raw bytes to map through the byte-level alphabet (a ByteLevel pre-tokenizer).
    BpeModel(const nlohmann::json& config, bool byte_level);

    void tokenize(std::string_view piece, std::vector<int32_t>& out) const override;
    std::optional<int32_t> id(std::string_view piece) const override { return vocab_.id(piece); }
    std::optional<std::string_view> piece(int32_t id) const override { return vocab_.piece(id); }
    int32_t id_limit() const override { return vocab_.id_limit(); }

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

    void use_byte_level_alphabet();
    void load_merges(const nlohmann::json& merges);
    void add_merge(std::string_view left, std::string_view right, int32_t rank);
    bool has_byte_tokens(std::string_view ch) const;
    std::vector<Symbol> initial_symbols(std::string_view piece) const;
    void merge_all(std::vector<Symbol>& symbols) const;
    const Merge* find_merge(int32_t left, int32_t right) const;

    Vocab vocab_;
    std::unordered_map<uint64_t, Merge> merges_;  // key: left id << 32 | right id
    std::array<int32_t, 256> byte_ids_{};          // "<0xXX>" ids for byte fallback; -1 when absent
    std::array<int32_t, 256> byte_level_ids_{};    // ids of the byte-level characters; -1 when absent
    bool byte_level_ = false;
    std::optional<int32_t> unk_id_;
    bool byte_fallback_ = false;
    bool fuse_unk_ = false;
    bool ignore_merges_ = false;
};

}  // namespace xs::tok
