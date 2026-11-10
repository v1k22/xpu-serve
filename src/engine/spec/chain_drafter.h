#pragma once

// A chain drafter's runner (ChainDrafting: an MTP head, an assistant): feeds the drafter the target's hidden states
// and proposes a chain of drafts, one run each.

#include <cstdint>
#include <span>
#include <vector>

#include "engine/spec/drafter.h"

namespace xs::spec {

// Pairs (the target's hidden row of position i, the token at i + 1) wait here until the drafter takes them: a row's
// token is known once the target is fed position i + 1, or when a round proposes after it.
class ChainDrafter final : public Drafter {
public:
    ChainDrafter(const Speculation& spec, Session& session);

    std::span<const std::string> target_outputs() const override;
    int max_drafts() const override;
    // `outputs` holds "hidden" ([rows, width]); tokens[0] completes the pair of the row before `start`.
    void add(const TensorMap& outputs, int64_t keep, int64_t start, std::span<const int32_t> tokens) override;
    // A drafter that keeps state takes every complete pair now.
    void flush() override;
    std::vector<int32_t> propose(int32_t pending, int k) override;
    void truncate(int64_t n) override;
    void reset() override;

private:
    int64_t rows() const;
    // Runs the drafter over the first `n` waiting pairs and drops them; returns its outputs.
    void run_pairs(int64_t n, TensorMap& out);
    int32_t pick(const TensorMap& out) const;

    const ChainDrafting& chain_;
    Session& session_;
    int64_t width_ = 0;            // hidden values per row
    int64_t start_ = 0;            // position of the first waiting row
    std::vector<float> hidden_;    // waiting rows
    std::vector<int32_t> next_;    // tokens of their pairs: one per row, or one fewer while the last row's is unknown
    TensorMap inputs_;
};

}  // namespace xs::spec
