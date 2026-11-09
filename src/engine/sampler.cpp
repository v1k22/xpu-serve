#include "engine/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace xs {
namespace {
uint64_t make_seed(const SamplingParams& p) { return p.seed ? *p.seed : std::random_device{}(); }

int32_t argmax(std::span<const float> logits) {
    return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
}  // namespace

Sampler::Sampler(const SamplingParams& params) : params_(params), rng_(make_seed(params)) {}

void Sampler::apply_penalties(std::span<float> logits, std::span<const int32_t> prompt,
                              std::span<const int32_t> generated) const {
    const auto vocab = static_cast<int32_t>(logits.size());
    if (params_.repetition_penalty != 1.0f) {
        auto penalize = [&](int32_t t) {
            if (t < 0 || t >= vocab) return;
            float& l = logits[static_cast<size_t>(t)];
            l = l > 0 ? l / params_.repetition_penalty : l * params_.repetition_penalty;
        };
        std::vector<bool> seen(static_cast<size_t>(vocab), false);
        for (auto list : {prompt, generated}) {
            for (int32_t t : list) {
                if (t >= 0 && t < vocab && !seen[static_cast<size_t>(t)]) {
                    seen[static_cast<size_t>(t)] = true;
                    penalize(t);
                }
            }
        }
    }
    if (params_.presence_penalty != 0.0f || params_.frequency_penalty != 0.0f) {
        std::unordered_map<int32_t, int> counts;
        for (int32_t t : generated) ++counts[t];
        for (const auto& [t, c] : counts) {
            if (t < 0 || t >= vocab) continue;
            logits[static_cast<size_t>(t)] -= params_.presence_penalty + params_.frequency_penalty * static_cast<float>(c);
        }
    }
}

int32_t Sampler::sample(std::span<float> logits, std::span<const int32_t> prompt, std::span<const int32_t> generated) {
    apply_penalties(logits, prompt, generated);
    if (greedy()) return argmax(logits);
    distribution(logits);
    std::discrete_distribution<size_t> pick(weights_.begin(), weights_.end());
    return order_[pick(rng_)];
}

int32_t Sampler::verify(std::span<float> logits, std::span<const int32_t> prompt, std::span<const int32_t> generated,
                        int32_t draft) {
    apply_penalties(logits, prompt, generated);
    if (greedy()) return argmax(logits);
    distribution(logits);
    const double total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    const auto it = std::find(order_.begin(), order_.end(), draft);
    if (it != order_.end()) {
        double& w = weights_[static_cast<size_t>(it - order_.begin())];
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * total < w) return draft;
        w = 0.0;  // the residual max(p - q, 0) of a point-mass draft: p without the draft
    }
    std::discrete_distribution<size_t> pick(weights_.begin(), weights_.end());
    return order_[pick(rng_)];
}

void Sampler::distribution(std::span<const float> logits) {
    // Candidates sorted by logit, cut by top-k first (cheap partial sort), then top-p / min-p on the softmax.
    const size_t vocab = logits.size();
    const size_t k = params_.top_k > 0 ? std::min(vocab, static_cast<size_t>(params_.top_k)) : vocab;
    order_.resize(vocab);
    std::iota(order_.begin(), order_.end(), 0);
    auto by_logit = [&](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; };
    std::partial_sort(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(k), order_.end(), by_logit);
    order_.resize(k);

    const float max_logit = logits[static_cast<size_t>(order_[0])];
    weights_.resize(k);
    double total = 0.0;
    for (size_t i = 0; i < k; ++i) {
        weights_[i] =
            std::exp((static_cast<double>(logits[static_cast<size_t>(order_[i])]) - max_logit) / params_.temperature);
        total += weights_[i];
    }
    size_t keep = k;
    if (params_.min_p > 0.0f) {
        const double floor = params_.min_p * weights_[0];  // weights_[0] is the max
        while (keep > 1 && weights_[keep - 1] < floor) --keep;
    }
    if (params_.top_p < 1.0f) {
        double cumulative = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            cumulative += weights_[i] / total;
            if (cumulative >= params_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    order_.resize(keep);
    weights_.resize(keep);
}

}  // namespace xs
