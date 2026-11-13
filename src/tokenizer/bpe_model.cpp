#include "tokenizer/bpe_model.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <string>

#include "tokenizer/byte_level.h"
#include "tokenizer/json_fields.h"
#include "tokenizer/utf8.h"
#include "util/error.h"

namespace xs::tok {
namespace {
constexpr std::string_view kWhat = "model";

uint64_t pair_key(int32_t left, int32_t right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
}

void refuse_option(const nlohmann::json& config, const char* key) {
    if (!config.contains(key) || config.at(key).is_null()) return;
    const auto& v = config.at(key);
    const bool is_default = (v.is_string() && v.get<std::string>().empty()) || (v.is_number() && v.get<double>() == 0);
    if (!is_default) fail("tokenizer.json BPE option '", key, "' is not supported");
}

// One pending merge of the symbol at `pos` with its right neighbour.
struct MergeCandidate {
    int32_t rank;
    int32_t pos;
    int32_t new_id;
};
// std::priority_queue pops the largest; this makes it pop the lowest rank, then the leftmost position.
struct LaterCandidate {
    bool operator()(const MergeCandidate& a, const MergeCandidate& b) const {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
    }
};
}  // namespace

BpeModel::BpeModel(const nlohmann::json& config, bool byte_level) : vocab_(config, "BPE") {
    for (const char* key : {"dropout", "continuing_subword_prefix", "end_of_word_suffix"}) refuse_option(config, key);
    if (!config.contains("merges") || !config.at("merges").is_array()) fail("tokenizer.json BPE model has no 'merges'");
    load_merges(config.at("merges"));

    byte_fallback_ = bool_field_or(config, "byte_fallback", false, kWhat);
    fuse_unk_ = bool_field_or(config, "fuse_unk", false, kWhat);
    ignore_merges_ = bool_field_or(config, "ignore_merges", false, kWhat);
    if (config.contains("unk_token") && !config.at("unk_token").is_null()) {
        const std::string unk = string_field(config, "unk_token", kWhat);
        unk_id_ = id(unk);
        if (!unk_id_) fail("tokenizer.json BPE unk_token '", unk, "' is not in the vocab");
    }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        byte_ids_[b] = id(name).value_or(-1);
    }
    if (byte_level) use_byte_level_alphabet();
}

void BpeModel::load_merges(const nlohmann::json& merges) {
    merges_.reserve(merges.size());
    int32_t rank = 0;
    for (const auto& merge : merges) {
        if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string()) {
            add_merge(merge[0].get_ref<const std::string&>(), merge[1].get_ref<const std::string&>(), rank);
        } else if (merge.is_string()) {
            const std::string_view text = merge.get_ref<const std::string&>();
            const size_t space = text.find(' ');
            if (space == std::string_view::npos || text.find(' ', space + 1) != std::string_view::npos) {
                fail("tokenizer.json BPE merge '", text, "' is not two space-separated pieces");
            }
            add_merge(text.substr(0, space), text.substr(space + 1), rank);
        } else {
            fail("tokenizer.json BPE merge #", rank, " is neither a [left, right] pair nor a string");
        }
        ++rank;
    }
}

void BpeModel::add_merge(std::string_view left, std::string_view right, int32_t rank) {
    const auto left_id = id(left);
    const auto right_id = id(right);
    std::string merged(left);
    merged.append(right);
    const auto new_id = id(merged);
    if (!left_id || !right_id || !new_id) {
        fail("tokenizer.json BPE merge ['", left, "', '", right, "'] uses a piece that is not in the vocab");
    }
    // Like Hugging Face, a duplicated pair keeps its last rank.
    merges_.insert_or_assign(pair_key(*left_id, *right_id), Merge{rank, *new_id});
}

void BpeModel::use_byte_level_alphabet() {
    byte_level_ = true;
    for (int b = 0; b < 256; ++b) {
        const auto symbol_id = id(byte_level::byte_piece(static_cast<uint8_t>(b)));
        if (!symbol_id) fail("tokenizer.json BPE vocab has no byte-level character for byte ", b);
        byte_level_ids_[b] = *symbol_id;
    }
}

bool BpeModel::has_byte_tokens(std::string_view ch) const {
    return std::all_of(ch.begin(), ch.end(), [this](char b) { return byte_ids_[static_cast<uint8_t>(b)] >= 0; });
}

const BpeModel::Merge* BpeModel::find_merge(int32_t left, int32_t right) const {
    const auto it = merges_.find(pair_key(left, right));
    return it == merges_.end() ? nullptr : &it->second;
}

void BpeModel::tokenize(std::string_view piece, std::vector<int32_t>& out) const {
    if (piece.empty()) return;
    if (ignore_merges_) {
        std::string mapped;
        if (byte_level_) {
            for (const char b : piece) mapped += byte_level::byte_piece(static_cast<uint8_t>(b));
        }
        if (const auto whole = id(byte_level_ ? std::string_view(mapped) : piece)) {
            out.push_back(*whole);
            return;
        }
    }
    std::vector<Symbol> symbols = initial_symbols(piece);
    merge_all(symbols);
    for (int32_t pos = symbols.empty() ? -1 : 0; pos >= 0; pos = symbols[pos].next) out.push_back(symbols[pos].id);
}

// One symbol per character; unknown characters become byte tokens, or unk. Mirrors Hugging Face BPE::merge_word,
// including its quirk that a pending (fused) unk is only emitted at the next in-vocab character or at the end.
std::vector<BpeModel::Symbol> BpeModel::initial_symbols(std::string_view piece) const {
    std::vector<Symbol> symbols;
    symbols.reserve(piece.size());
    auto add = [&symbols](int32_t symbol_id) {
        const auto pos = static_cast<int32_t>(symbols.size());
        symbols.push_back({symbol_id, pos - 1, pos + 1, false});
    };
    if (byte_level_) {
        for (const char b : piece) add(byte_level_ids_[static_cast<uint8_t>(b)]);
        if (!symbols.empty()) symbols.back().next = -1;
        return symbols;
    }
    bool unk_pending = false;
    for (size_t pos = 0; pos < piece.size();) {
        const size_t len = utf8::char_length(piece, pos);
        const std::string_view ch = piece.substr(pos, len);
        pos += len;
        if (const auto ch_id = id(ch)) {
            if (unk_pending) add(*unk_id_);
            unk_pending = false;
            add(*ch_id);
            continue;
        }
        if (byte_fallback_ && has_byte_tokens(ch)) {
            for (const char b : ch) add(byte_ids_[static_cast<uint8_t>(b)]);
            continue;
        }
        if (!unk_id_) continue;  // no unk token: Hugging Face drops the character
        if (unk_pending && !fuse_unk_) add(*unk_id_);
        unk_pending = true;
    }
    if (unk_pending) add(*unk_id_);
    if (!symbols.empty()) symbols.back().next = -1;
    return symbols;
}

// Applies merges lowest rank first, leftmost first, in O(n log n): a priority queue of candidate pairs over a doubly
// linked symbol list. Port of Hugging Face tokenizers' Word::merge_all (Apache-2.0), stale-entry check included.
void BpeModel::merge_all(std::vector<Symbol>& symbols) const {
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, LaterCandidate> queue;
    auto push_candidate = [&](int32_t pos) {
        const int32_t next = symbols[pos].next;
        if (next < 0) return;
        if (const Merge* m = find_merge(symbols[pos].id, symbols[next].id)) queue.push({m->rank, pos, m->new_id});
    };
    for (int32_t pos = 0; pos + 1 < static_cast<int32_t>(symbols.size()); ++pos) push_candidate(pos);

    while (!queue.empty()) {
        const MergeCandidate top = queue.top();
        queue.pop();
        Symbol& left = symbols[top.pos];
        if (left.merged_away || left.next < 0) continue;
        const Symbol& right = symbols[left.next];
        const Merge* current = find_merge(left.id, right.id);
        if (current == nullptr || current->new_id != top.new_id) continue;  // stale: a neighbour changed since

        const int32_t right_pos = left.next;
        left.id = top.new_id;
        left.next = right.next;
        symbols[right_pos].merged_away = true;
        if (left.next >= 0) symbols[left.next].prev = top.pos;
        if (left.prev >= 0) push_candidate(left.prev);
        push_candidate(top.pos);
    }
}

}  // namespace xs::tok
