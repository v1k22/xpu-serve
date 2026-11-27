#pragma once

// A model's tokens as a grammar sees them, built once per model: each token's bytes, which ones a grammar may write,
// and the trie of their byte strings that a matcher walks to find the tokens a state allows.

#include <bit>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xs::grammar {

// A set of token ids, one bit each.
class TokenSet {
public:
    TokenSet() = default;
    explicit TokenSet(int32_t size) : size_(size), words_((static_cast<size_t>(size) + 63) / 64, 0) {}
    int32_t size() const { return size_; }
    void set(int32_t id) { words_[static_cast<size_t>(id) >> 6] |= uint64_t{1} << (id & 63); }
    bool has(int32_t id) const { return (words_[static_cast<size_t>(id) >> 6] >> (id & 63)) & 1; }
    void clear() { std::fill(words_.begin(), words_.end(), 0); }
    int32_t count() const {
        int32_t n = 0;
        for (const uint64_t w : words_) n += std::popcount(w);
        return n;
    }
    const std::vector<uint64_t>& words() const { return words_; }
    uint64_t* data() { return words_.data(); }

private:
    int32_t size_ = 0;
    std::vector<uint64_t> words_;
};

class Vocabulary {
public:
    struct Token {
        std::string bytes;  // what the token adds to the output
        bool marker = false;  // a control token (an added one, e.g. <|im_end|>, <think>): a grammar never writes it
    };
    // `tokens` by id.
    explicit Vocabulary(std::vector<Token> tokens);

    int32_t size() const { return static_cast<int32_t>(offsets_.size() - 1); }
    std::string_view bytes(int32_t id) const {
        return std::string_view(bytes_).substr(offsets_[id], offsets_[id + 1] - offsets_[id]);
    }
    // A grammar may write it: a token of the model's vocabulary with bytes.
    bool writable(int32_t id) const { return !marker_[id] && offsets_[id + 1] > offsets_[id]; }

    // The byte strings of the writable tokens as a trie, its nodes in depth-first order (each node one byte below its
    // parent, children in byte order). Node i is at depth depth[i] (1 below the root), its subtree ends before node
    // end[i], and token[i] is the token whose bytes end there (-1: none); the others with the same bytes as a token
    // are listed apart (same_bytes: that token, then one of them).
    struct Trie {
        std::vector<uint8_t> byte;
        std::vector<uint16_t> depth;
        std::vector<uint32_t> end;
        std::vector<int32_t> token;
        std::vector<std::pair<int32_t, int32_t>> same_bytes;
        int max_depth = 0;
        size_t nodes() const { return byte.size(); }
    };
    const Trie& trie() const { return trie_; }

private:
    std::string bytes_;  // every token's bytes, one after another
    std::vector<uint32_t> offsets_;
    std::vector<uint8_t> marker_;
    Trie trie_;
};

}  // namespace xs::grammar
