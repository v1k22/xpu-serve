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

private:
    void apply_penalties(std::span<float> logits, std::span<const int32_t> prompt,
                         std::span<const int32_t> generated) const;

    SamplingParams params_;
    std::mt19937_64 rng_;
    std::vector<int32_t> order_;  // scratch: candidate token ids
};

}  // namespace xs
