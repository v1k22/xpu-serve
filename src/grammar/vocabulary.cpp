#include "grammar/vocabulary.h"

#include <algorithm>
#include <numeric>

#include "util/error.h"

namespace xs::grammar {

Vocabulary::Vocabulary(std::vector<Token> tokens) {
    offsets_.reserve(tokens.size() + 1);
    marker_.reserve(tokens.size());
    offsets_.push_back(0);
    for (const Token& t : tokens) {
        bytes_ += t.bytes;
        XS_CHECK(bytes_.size() <= UINT32_MAX, "grammar: a vocabulary of more than 4 GiB of bytes");
        offsets_.push_back(static_cast<uint32_t>(bytes_.size()));
        marker_.push_back(t.marker ? 1 : 0);
        if (t.marker && !t.bytes.empty()) markers_.emplace(t.bytes, static_cast<int32_t>(marker_.size() - 1));
    }

    // Sorted by bytes (std::string_view compares them unsigned, as memcmp does), the tokens come in the trie's
    // depth-first order, and a token's node is always the last one made when it comes.
    std::vector<int32_t> order;
    for (int32_t id = 0; id < size(); ++id) {
        if (writable(id)) order.push_back(id);
    }
    std::stable_sort(order.begin(), order.end(), [this](int32_t a, int32_t b) { return bytes(a) < bytes(b); });
    Trie& t = trie_;
    std::vector<uint32_t> path;  // the nodes from the root down to the last token's
    std::string_view previous;
    for (const int32_t id : order) {
        const std::string_view b = bytes(id);
        XS_CHECK(b.size() <= UINT16_MAX, "grammar: a token of ", b.size(), " bytes");
        const size_t common = static_cast<size_t>(
            std::mismatch(previous.begin(), previous.end(), b.begin(), b.end()).first - previous.begin());
        for (; path.size() > common; path.pop_back()) t.end[path.back()] = static_cast<uint32_t>(t.nodes());
        for (size_t d = common; d < b.size(); ++d) {
            path.push_back(static_cast<uint32_t>(t.nodes()));
            t.byte.push_back(static_cast<uint8_t>(b[d]));
            t.depth.push_back(static_cast<uint16_t>(d + 1));
            t.end.push_back(0);
            t.token.push_back(-1);
        }
        int32_t& at = t.token[path.back()];
        if (at < 0) {
            at = id;
        } else {
            t.same_bytes.emplace_back(at, id);
        }
        t.max_depth = std::max(t.max_depth, static_cast<int>(b.size()));
        previous = b;
    }
    for (; !path.empty(); path.pop_back()) t.end[path.back()] = static_cast<uint32_t>(t.nodes());
}

std::optional<int32_t> Vocabulary::marker(std::string_view text) const {
    const auto it = markers_.find(std::string(text));
    if (it == markers_.end()) return std::nullopt;
    return it->second;
}

}  // namespace xs::grammar
