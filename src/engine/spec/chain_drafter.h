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
    ChainDrafter* as_chain() override { return this; }

    // A round's drafts in steps, so that the runs of several sequences' drafters go as one (the engine's batched
    // rounds, Speculation::batch_draft_graph); propose() is these steps over the draft program alone. begin(): the first
    // run's inputs (the waiting pairs, `pending` completing the last) and their rows. first(): its draft, from row
    // `row` of that run's outputs (the last pair's). chained() and next(): the inputs and the draft of the chained run
    // at `depth` (1, 2, ...). end(): the drafter's own state cut back to before the round. The drafter's sequence must
    // be the session's current one at begin() and end().
    int64_t begin(int32_t pending, TensorMap& inputs);
    int32_t first(const TensorMap& out, int64_t row);
    void chained(int depth, TensorMap& inputs) const;
    int32_t next(const TensorMap& out, int64_t row);
    void end();

private:
    int64_t rows() const;
    // Runs the drafter over the first `n` waiting pairs and drops them; returns its outputs.
    void run_pairs(int64_t n, TensorMap& out);
    // Drops the first `n` waiting pairs (the drafter ran them).
    void drop(int64_t n);
    // The draft of row `row` of a run's outputs ("logits" [rows, n] or [1, n]; "candidates" [rows, n] or [n]), and the
    // row's hidden, which the next chained run takes.
    int32_t take(const TensorMap& out, int64_t row);

    const ChainDrafting& chain_;
    Session& session_;
    int64_t width_ = 0;            // hidden values per row
    int64_t start_ = 0;            // position of the first waiting row
    std::vector<float> hidden_;    // waiting rows
    std::vector<int32_t> next_;    // tokens of their pairs: one per row, or one fewer while the last row's is unknown
    TensorMap inputs_;
    // A round in steps: the position of its last pair, the drafter's cached tokens before it, its last draft and that
    // draft's hidden row.
    int64_t position_ = 0, kept_ = 0;
    int32_t last_ = 0;
    std::vector<float> last_hidden_;
};

}  // namespace xs::spec
