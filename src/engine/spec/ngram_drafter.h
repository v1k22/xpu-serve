#pragma once

// Prompt lookup decoding (an n-gram drafter): drafts copied from the sequence itself, with no model of their own. The
// last n tokens (the pending one among them) are looked up earlier in the sequence, prompt and output alike, and the
// tokens that followed them there are the drafts. It suits output that repeats its input: code edits, extraction,
// summaries that quote. The target verifies the drafts as it would a drafter's, with a program that gives every row's
// logits (the model's batch graph), so any language model can speculate this way.

#include <cstdint>
#include <span>
#include <vector>

#include "engine/spec/drafter.h"

namespace xs::spec {

// The n-grams looked up: the longest suffix of the sequence of `min` to `max` tokens that occurred before.
struct NgramOptions {
    int min = 5;
    int max = 5;
};

// The drafts after `tokens` (the sequence, its last token the pending one): the longest suffix of `min` to `max` tokens
// that also occurs earlier, at its first occurrence, and up to `k` of the tokens that follow it there, never past the
// end of `tokens`. Empty when no such suffix occurs. Negative tokens (positions the drafter does not know) match none.
// (vLLM's GPU speculator takes the latest occurrence; its earlier CPU proposer took the first, which drafts longer
// copies: docs/decisions/0034.)
std::vector<int32_t> lookup_drafts(std::span<const int32_t> tokens, const NgramOptions& n, int k);

class NgramDrafter final : public Drafter {
public:
    NgramDrafter(NgramOptions n, int max_drafts);

    std::span<const std::string> target_outputs() const override { return {}; }
    int max_drafts() const override { return max_drafts_; }
    void add(const TensorMap& outputs, int64_t keep, int64_t start, std::span<const int32_t> tokens) override;
    void flush() override {}
    // Fewer than `k` drafts (none) when the sequence offers fewer.
    std::vector<int32_t> propose(int32_t pending, int k) override;
    void truncate(int64_t n) override;
    void reset() override { tokens_.clear(); }
    void adopt(std::span<const int32_t> tokens) override;

private:
    NgramOptions n_;
    int max_drafts_;
    std::vector<int32_t> tokens_;  // the sequence's tokens by position; -1 where unknown
};

}  // namespace xs::spec
