#pragma once

// The drafter side of speculative decoding (DESIGN.md 10): feeds a model's drafter the target's hidden states and
// proposes chains of drafts. The engine owns the loop: verify, acceptance, rollback.

#include <cstdint>
#include <span>
#include <vector>

#include "backend/backend.h"
#include "models/model.h"

namespace xs::spec {

// The programs of a speculating session, in the order the engine compiles them.
enum Program : size_t { kStep = 0, kPrefill = 1, kVerify = 2, kDraft = 3 };

// Pairs (the target's hidden row of position i, the token at i + 1) wait here until the drafter takes them: a row's
// token is known once the target is fed position i + 1, or when a round proposes after it.
class DraftRunner {
public:
    DraftRunner(const Speculation& spec, Session& session);

    // The target ran `tokens` at positions start, start + 1, ... and produced `hidden` ([rows, width]); its first
    // `keep` rows count (a verify keeps the accepted ones). tokens[0] completes the pair of the row before `start`.
    void add(const HostTensor& hidden, int64_t keep, int64_t start, std::span<const int32_t> tokens);
    // A drafter that keeps state takes every complete pair now (prefill: bounds what waits here).
    void flush();
    // Drafts `k` tokens after `pending`, the token after the last row added.
    std::vector<int32_t> propose(int32_t pending, int k);
    // The target kept its first `n` positions.
    void truncate(int64_t n);
    void reset();

private:
    int64_t rows() const;
    // Runs the drafter over the first `n` waiting pairs and drops them; returns its outputs.
    void run_pairs(int64_t n, TensorMap& out);
    int32_t pick(const TensorMap& out) const;

    const Speculation& spec_;
    Session& session_;
    int64_t width_ = 0;            // hidden values per row
    int64_t start_ = 0;            // position of the first waiting row
    std::vector<float> hidden_;    // waiting rows
    std::vector<int32_t> next_;    // tokens of their pairs: one per row, or one fewer while the last row's is unknown
    TensorMap inputs_;
};

}  // namespace xs::spec
