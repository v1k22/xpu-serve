// Log-probabilities of generated tokens (GenerateRequest::logprobs): a row's log-softmax at the token and its most
// likely tokens; in the engine, one entry a token emitted, the row's as the output constraint left it and before the
// sampler's penalties, the same whether drafts were verified or not (a chain drafter, prompt lookup), and the greedy
// tokens unchanged by asking for them.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "engine/sampler.h"
#include "tiny_engine.h"

namespace xs {

using test::greedy;
using test::make_engine;
using test::make_lookup_engine;

TEST_CASE("a row's log-probabilities: log-softmax at the token, the most likely tokens first, ties by id") {
    const std::vector<float> row{1.0f, 3.0f, 2.0f, 3.0f, -std::numeric_limits<float>::infinity()};
    double norm = 0.0;
    for (const float l : row) norm += std::exp(static_cast<double>(l));
    const auto lp = [&](float l) { return static_cast<float>(l - std::log(norm)); };
    const TokenLogprobs t = token_logprobs(row, 2, 3);
    CHECK(t.logprob == doctest::Approx(lp(2.0f)));
    REQUIRE(t.top.size() == 3);
    CHECK(t.top[0].first == 1);  // 3.0 at ids 1 and 3: the lower id first
    CHECK(t.top[1].first == 3);
    CHECK(t.top[2].first == 2);
    CHECK(t.top[0].second == doctest::Approx(lp(3.0f)));
    CHECK(token_logprobs(row, 4, 0).logprob == -std::numeric_limits<float>::infinity());  // masked
    CHECK(token_logprobs(row, 4, 0).top.empty());
    CHECK(token_logprobs(row, 0, 9).top.size() == row.size());  // at most the whole row
    CHECK_THROWS_AS(token_logprobs(row, 5, 1), xs::Error);
}

namespace {

// Each emitted token's log-probabilities, in order.
struct Recorded {
    std::vector<int32_t> tokens;
    std::vector<TokenLogprobs> logprobs;
};

Recorded run(Engine& engine, const GenerateRequest& r) {
    Recorded out;
    const GenerateResult result = engine.generate(r, [&](int32_t token, const TokenLogprobs* lp) {
        out.tokens.push_back(token);
        if (lp != nullptr) out.logprobs.push_back(*lp);
        return true;
    });
    REQUIRE(result.error == "");
    CHECK(result.tokens == out.tokens);
    return out;
}

void same_logprobs(const Recorded& a, const Recorded& b) {
    REQUIRE(a.tokens == b.tokens);
    REQUIRE(a.logprobs.size() == b.logprobs.size());
    for (size_t i = 0; i < a.logprobs.size(); ++i) {
        CAPTURE(i);
        CHECK(a.logprobs[i].logprob == doctest::Approx(b.logprobs[i].logprob).epsilon(1e-4));
        REQUIRE(a.logprobs[i].top.size() == b.logprobs[i].top.size());
        for (size_t j = 0; j < a.logprobs[i].top.size(); ++j) {
            CHECK(a.logprobs[i].top[j].first == b.logprobs[i].top[j].first);
            CHECK(a.logprobs[i].top[j].second == doctest::Approx(b.logprobs[i].top[j].second).epsilon(1e-4));
        }
    }
}

}  // namespace

TEST_CASE("log-probabilities in the engine: one a token emitted, the same with drafts verified, the tokens unchanged") {
    GenerateRequest r = greedy({1, 7, 3, 12, 5, 9, 0, 14, 2, 11}, 30);
    const Recorded without = run(*make_engine(0, 4, /*argmax=*/true), r);
    CHECK(without.logprobs.empty());  // not asked for
    r.logprobs = 3;
    const Recorded plain = run(*make_engine(0, 4, true), r);
    CHECK(plain.tokens == without.tokens);  // reading the logits instead of the argmax: the same tokens
    REQUIRE(plain.logprobs.size() == plain.tokens.size());
    for (size_t i = 0; i < plain.tokens.size(); ++i) {
        CAPTURE(i);
        REQUIRE(plain.logprobs[i].top.size() == 3);
        CHECK(plain.logprobs[i].top[0].first == plain.tokens[i]);  // greedy: the token is the most likely
        CHECK(plain.logprobs[i].logprob == plain.logprobs[i].top[0].second);
        CHECK(plain.logprobs[i].top[1].second <= plain.logprobs[i].top[0].second);
    }
    same_logprobs(plain, run(*make_engine(3, 4), r));                     // a chain drafter's rounds
    same_logprobs(plain, run(*make_engine(2, 4, false, true), r));        // a block drafter's
    same_logprobs(plain, run(*make_lookup_engine(4, 4, {1, 3}), r));      // prompt lookup's
    // Batched: the rows of a step of several requests, a chain drafter's rounds among them.
    for (const int k : {0, 2}) {
        CAPTURE(k);
        const auto batched = make_engine(k, 4, true, false, 2, true);
        Recorded got;
        const auto h = batched->start(r, [&](int32_t t, const TokenLogprobs* lp) {
            got.tokens.push_back(t);
            if (lp != nullptr) got.logprobs.push_back(*lp);
            return true;
        });
        const auto other = batched->start(greedy({4, 4, 9}, 12), {});
        REQUIRE(h.has_value());
        REQUIRE(other.has_value());
        std::vector<Engine::Handle> live{*h, *other};
        while (!live.empty()) {
            std::vector<Engine::Handle> decoding, next;
            for (const Engine::Handle x : live) {
                if (batched->decoding(x)) {
                    decoding.push_back(x);
                } else if (batched->advance(x)) {
                    batched->finish(x);
                } else {
                    next.push_back(x);
                }
            }
            const std::vector<bool> done = batched->advance_decoding(decoding);
            for (size_t i = 0; i < decoding.size(); ++i) {
                if (done[i]) {
                    batched->finish(decoding[i]);
                } else {
                    next.push_back(decoding[i]);
                }
            }
            live = next;
        }
        same_logprobs(plain, got);
    }
}

TEST_CASE("log-probabilities read the row before the penalties, and after the constraint") {
    GenerateRequest r = greedy({6, 2, 13, 1, 4}, 20);
    r.logprobs = 2;
    const Recorded free = run(*make_engine(0, 8), r);
    r.sampling.repetition_penalty = 1.5f;
    const Recorded penalized = run(*make_engine(0, 8), r);
    // The penalty changes which token comes, never the log-probabilities of the row the model gave for it.
    REQUIRE_FALSE(penalized.tokens.empty());
    CHECK(penalized.logprobs[0].logprob == doctest::Approx(free.logprobs[0].logprob));
    bool differs = false;
    for (size_t i = 0; i < std::min(free.tokens.size(), penalized.tokens.size()); ++i) {
        differs = differs || free.tokens[i] != penalized.tokens[i];
    }
    CHECK(differs);
    for (const TokenLogprobs& lp : penalized.logprobs) CHECK(lp.logprob <= 0.0f);
}

TEST_CASE("log-probabilities of a request out of range are refused") {
    GenerateRequest r = greedy({1, 2}, 4);
    r.logprobs = static_cast<int>(test::TinySpecModel::kVocab) + 1;
    CHECK(make_engine(0, 4)->generate(r, {}).finish == FinishReason::Error);
}

}  // namespace xs
