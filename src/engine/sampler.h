#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "engine/types.h"

namespace xs {

// Turns one row of logits into a token. Penalties look at `history` (prompt + generated so far for the repetition
// penalty, generated only for presence/frequency, as in OpenAI's definition).
class Sampler {
public:
    explicit Sampler(const SamplingParams& params);

    int32_t sample(std::span<float> logits, std::span<const int32_t> prompt, std::span<const int32_t> generated);

    // Speculative decoding: checks a drafter's `draft` (proposed with probability 1) against these logits by
    // rejection sampling. Returns `draft` with probability p(draft), else a token drawn from p without it, so the
    // result is distributed exactly as sample()'s (greedy: the argmax, accepted when it equals the draft).
    int32_t verify(std::span<float> logits, std::span<const int32_t> prompt, std::span<const int32_t> generated,
                   int32_t draft);

    // False when sample() and verify() return the largest logit's index whatever the history (greedy, no penalties):
    // a row's argmax then stands for its logits.
    bool needs_logits() const {
        return !greedy() || params_.repetition_penalty != 1.0f || params_.presence_penalty != 0.0f ||
               params_.frequency_penalty != 0.0f;
    }

private:
    void apply_penalties(std::span<float> logits, std::span<const int32_t> prompt,
                         std::span<const int32_t> generated) const;
    bool greedy() const { return params_.temperature <= 0.0f || params_.top_k == 1; }
    // The candidates sample() draws from: order_[i] with weight weights_[i] (unnormalized).
    void distribution(std::span<const float> logits);

    SamplingParams params_;
    std::mt19937_64 rng_;
    std::vector<int32_t> order_;   // scratch: candidate token ids
    std::vector<double> weights_;  // scratch: their weights
};

}  // namespace xs
