// Sampler: speculative verification (rejection sampling against a point-mass draft) draws tokens with the target's
// exact distribution, whatever the draft.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

#include "engine/sampler.h"

namespace xs {
namespace {

const std::vector<float> kLogits = {2.0f, 1.5f, 1.4f, 0.3f, -0.2f, 0.9f, -1.0f, 0.0f};

// softmax(logits / temperature) over the top_k, cut to the smallest prefix reaching top_p, renormalized.
std::map<int32_t, double> distribution(const SamplingParams& p) {
    std::vector<int32_t> order(kLogits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [](int32_t a, int32_t b) { return kLogits[a] > kLogits[b]; });
    order.resize(static_cast<size_t>(p.top_k));
    std::vector<double> w;
    for (int32_t t : order) w.push_back(std::exp((kLogits[t] - kLogits[order[0]]) / p.temperature));
    const double total = std::accumulate(w.begin(), w.end(), 0.0);
    double cumulative = 0, kept = 0;
    size_t keep = 0;
    while (keep < w.size() && cumulative < p.top_p) cumulative += w[keep++] / total;
    for (size_t i = 0; i < keep; ++i) kept += w[i];
    std::map<int32_t, double> out;
    for (size_t i = 0; i < keep; ++i) out[order[i]] = w[i] / kept;
    return out;
}

}  // namespace

TEST_CASE("verify draws the target's distribution whatever the draft (rejection sampling)") {
    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 5;
    params.top_p = 0.95f;
    params.seed = 11;
    const std::map<int32_t, double> want = distribution(params);
    REQUIRE(want.size() >= 3);
    constexpr int kTrials = 200000;
    for (const int32_t draft : {0, 5, 6}) {  // the likeliest token, a less likely one, one outside the candidates
        CAPTURE(draft);
        Sampler sampler(params);
        std::map<int32_t, int> counts;
        for (int i = 0; i < kTrials; ++i) {
            std::vector<float> logits = kLogits;
            ++counts[sampler.verify(logits, {}, {}, draft)];
        }
        for (const auto& [token, count] : counts) CHECK(want.count(token) == 1);
        for (const auto& [token, p] : want) {
            CAPTURE(token);
            const double freq = static_cast<double>(counts[token]) / kTrials;
            CHECK(std::abs(freq - p) < 5.0 * std::sqrt(p * (1 - p) / kTrials) + 1e-4);
        }
    }
}

TEST_CASE("greedy verify returns the argmax: a draft is accepted only when it is the argmax") {
    SamplingParams params;
    params.temperature = 0.0f;
    Sampler sampler(params);
    for (const int32_t draft : {0, 3}) {
        std::vector<float> logits = kLogits;
        CHECK(sampler.verify(logits, {}, {}, draft) == 0);
    }
}

}  // namespace xs
