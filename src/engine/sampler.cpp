#include "engine/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "util/error.h"

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
    return draw(distribution(logits));
}

int32_t Sampler::verify(std::span<float> logits, std::span<const int32_t> prompt, std::span<const int32_t> generated,
                        int32_t draft) {
    apply_penalties(logits, prompt, generated);
    if (greedy()) return argmax(logits);
    double total = distribution(logits);
    const auto it = std::find(order_.begin(), order_.end(), draft);
    if (it != order_.end()) {
        double& w = weights_[static_cast<size_t>(it - order_.begin())];
        if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) * total < w) return draft;
        total -= w;
        w = 0.0;  // the residual max(p - q, 0) of a point-mass draft: p without the draft
    }
    return draw(total);
}

int32_t Sampler::draw(double total) {
    double left = std::uniform_real_distribution<double>(0.0, total)(rng_);
    size_t last = 0;  // the last candidate with any weight: where rounding in `total` lands
    for (size_t i = 0; i < weights_.size(); ++i) {
        if (weights_[i] <= 0.0) continue;
        if (left < weights_[i]) return order_[i];
        left -= weights_[i];
        last = i;
    }
    return order_[last];
}

double Sampler::distribution(std::span<const float> logits) {
    const size_t vocab = logits.size();
    const size_t k = params_.top_k > 0 ? std::min(vocab, static_cast<size_t>(params_.top_k)) : vocab;
    const auto weight = [&](size_t token, float max_logit) {
        return std::exp((static_cast<double>(logits[token]) - max_logit) / params_.temperature);
    };
    order_.resize(vocab);
    std::iota(order_.begin(), order_.end(), 0);
    double total = 0.0;
    if (k == vocab && params_.top_p >= 1.0f) {
        // No cut by rank: the tokens in vocabulary order (those within min_p of the likeliest), with no sort of a
        // vocabulary of a few hundred thousand.
        const float max_logit = *std::max_element(logits.begin(), logits.end());
        weights_.resize(vocab);
        size_t kept = 0;
        for (size_t t = 0; t < vocab; ++t) {
            const double w = weight(t, max_logit);
            if (w < params_.min_p) continue;  // the likeliest token's weight is 1
            order_[kept] = static_cast<int32_t>(t);
            weights_[kept++] = w;
            total += w;
        }
        order_.resize(kept);
        weights_.resize(kept);
        return total;
    }
    // Candidates sorted by logit, cut by top-k first (a partial sort), then top-p / min-p on the softmax. Without a
    // top-k the softmax is over the vocabulary, and the sort goes only as far as top-p reaches.
    auto by_logit = [&](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; };
    size_t sorted = k < vocab ? k : std::min<size_t>(vocab, 256);
    std::partial_sort(order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(sorted), order_.end(), by_logit);
    order_.resize(k);
    const float max_logit = logits[static_cast<size_t>(order_[0])];
    for (size_t i = 0; i < k; ++i) total += weight(static_cast<size_t>(order_[i]), max_logit);
    weights_.clear();
    double kept = 0.0;
    for (size_t i = 0; i < k; ++i) {
        if (i == sorted) {  // top-p reaches past what is sorted: the next candidates, four times as far
            sorted = std::min(k, 4 * sorted);
            std::partial_sort(order_.begin() + static_cast<std::ptrdiff_t>(i),
                              order_.begin() + static_cast<std::ptrdiff_t>(sorted), order_.end(), by_logit);
        }
        const double w = weight(static_cast<size_t>(order_[i]), max_logit);
        if (i > 0 && w < params_.min_p) break;
        weights_.push_back(w);
        kept += w;
        if (params_.top_p < 1.0f && kept / total >= params_.top_p) break;
    }
    order_.resize(weights_.size());
    return kept;
}

TokenLogprobs token_logprobs(std::span<const float> logits, int32_t token, int top) {
    XS_CHECK(token >= 0 && static_cast<size_t>(token) < logits.size(), "logprobs of token ", token, " of ",
             logits.size());
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float l : logits) sum += std::exp(static_cast<double>(l) - max_logit);
    const double log_norm = max_logit + std::log(sum);
    const auto logprob = [&](size_t t) { return static_cast<float>(static_cast<double>(logits[t]) - log_norm); };
    TokenLogprobs out;
    out.logprob = logprob(static_cast<size_t>(token));
    // The `top` best by logit (then by id), kept in a heap whose root is the worst of them.
    const auto better = [&](int32_t a, int32_t b) {
        const float la = logits[static_cast<size_t>(a)], lb = logits[static_cast<size_t>(b)];
        return la > lb || (la == lb && a < b);
    };
    const size_t k = std::min(logits.size(), static_cast<size_t>(std::max(top, 0)));
    std::vector<int32_t> best;
    best.reserve(k + 1);
    for (size_t t = 0; t < logits.size() && k > 0; ++t) {
        const auto id = static_cast<int32_t>(t);
        if (best.size() == k && !better(id, best.front())) continue;
        best.push_back(id);
        std::push_heap(best.begin(), best.end(), better);
        if (best.size() > k) {
            std::pop_heap(best.begin(), best.end(), better);
            best.pop_back();
        }
    }
    std::sort(best.begin(), best.end(), better);
    for (const int32_t id : best) out.top.emplace_back(id, logprob(static_cast<size_t>(id)));
    return out;
}

}  // namespace xs
