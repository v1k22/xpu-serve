#include "engine/spec/ngram_drafter.h"

#include <algorithm>

#include "util/error.h"

namespace xs::spec {

std::vector<int32_t> lookup_drafts(std::span<const int32_t> tokens, const NgramOptions& n, int k) {
    const auto total = static_cast<int64_t>(tokens.size());
    if (total < n.min || k <= 0) return {};
    // Read backwards from the last token, the sequence is a text whose prefixes are the suffixes looked up: the prefix
    // function of that text (Knuth-Morris-Pratt), its matches capped at `max`, finds at each position the longest of
    // them that ends there. The longest overall wins, and of those the one read last (the earliest in the sequence).
    const auto at = [&](int64_t i) { return tokens[static_cast<size_t>(total - 1 - i)]; };
    const auto same = [&](int64_t a, int64_t b) { return at(a) >= 0 && at(a) == at(b); };
    std::vector<int64_t> prefix(static_cast<size_t>(n.max), 0);
    int64_t longest = 0, position = 0, matched = 0;
    for (int64_t i = 1; i < total;) {
        if (same(matched, i)) {
            ++matched;
            if (matched >= longest) {
                longest = matched;
                position = i;
            }
            if (i < n.max) prefix[static_cast<size_t>(i)] = matched;
            if (matched == n.max) matched = prefix[static_cast<size_t>(n.max - 1)];
            ++i;
        } else if (matched > 0) {
            matched = prefix[static_cast<size_t>(matched - 1)];
        } else {
            ++i;
        }
    }
    if (longest < n.min) return {};
    // The match starts at total - 1 - position; what followed it comes after its `longest` tokens.
    const int64_t first = total - 1 - position + longest;
    std::vector<int32_t> drafts;
    for (int64_t i = first; i < total && static_cast<int64_t>(drafts.size()) < k; ++i) {
        if (tokens[static_cast<size_t>(i)] < 0) break;
        drafts.push_back(tokens[static_cast<size_t>(i)]);
    }
    return drafts;
}

NgramDrafter::NgramDrafter(NgramOptions n, int max_drafts) : n_(n), max_drafts_(max_drafts) {
    XS_CHECK(n_.min >= 1 && n_.max >= n_.min, "spec: n-grams of ", n_.min, " to ", n_.max, " tokens");
    XS_CHECK(max_drafts_ >= 1, "spec: prompt lookup of ", max_drafts_, " drafts a round");
}

void NgramDrafter::add(const TensorMap&, int64_t keep, int64_t start, std::span<const int32_t> tokens) {
    XS_CHECK(keep >= 0 && keep <= static_cast<int64_t>(tokens.size()) && start >= 0, "spec: ", keep, " of ",
             tokens.size(), " tokens kept at ", start);
    tokens_.resize(static_cast<size_t>(start), -1);  // positions before `start` it was never given stay unknown
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.begin() + keep);
}

std::vector<int32_t> NgramDrafter::propose(int32_t pending, int k) {
    tokens_.push_back(pending);
    std::vector<int32_t> drafts = lookup_drafts(tokens_, n_, std::min(k, max_drafts_));
    tokens_.pop_back();  // the target is fed it with the drafts, and adds what it keeps
    return drafts;
}

void NgramDrafter::truncate(int64_t n) {
    if (n < static_cast<int64_t>(tokens_.size())) tokens_.resize(static_cast<size_t>(std::max<int64_t>(n, 0)));
}

void NgramDrafter::adopt(std::span<const int32_t> tokens) {
    tokens_.assign(tokens.begin(), tokens.end());
}

}  // namespace xs::spec
