// Adaptive draft counts (EngineOptions::adaptive_k, engine/spec/draft_length.h): the acceptance a request's rounds
// show, the count with the most tokens a millisecond for measured costs, and in the engine the same greedy tokens as
// with a fixed count, its rounds' drafts moving with acceptance.
#include <doctest/doctest.h>

#include "engine/spec/draft_length.h"
#include "tiny_engine.h"

namespace xs {

using spec::Acceptance;
using spec::DraftLength;
using test::greedy;
using test::make_engine;
using test::make_lookup_engine;

TEST_CASE("a request's acceptance: draft by draft, recent rounds counting most, unreached drafts like the one before") {
    Acceptance a(0.5);
    CHECK(a.rate(1) == doctest::Approx(0.5));
    CHECK(a.tokens(2) == doctest::Approx(1.75));
    for (int i = 0; i < 40; ++i) a.round(4, 4);
    CHECK(a.rate(1) > 0.9);
    CHECK(a.rate(4) > 0.9);
    for (int i = 0; i < 40; ++i) a.round(4, 0);  // the first draft rejected each round: one reached, none accepted
    CHECK(a.rate(1) < 0.1);
    CHECK(a.rate(1) > 0.0);  // never certain
    // A chain drafter whose first draft is mostly right and its second mostly wrong.
    Acceptance b(0.5);
    for (int i = 0; i < 60; ++i) b.round(3, i % 10 == 0 ? 0 : 1);
    CHECK(b.rate(1) == doctest::Approx(0.9).epsilon(0.05));
    CHECK(b.rate(2) < 0.2);
    CHECK(b.rate(3) < 0.3);  // never reached: the draft before it, pulled toward its own (unseen) prior
    CHECK(b.tokens(1) == doctest::Approx(1.0 + b.rate(1)));
    const double before = b.rate(1);
    b.round(0, 0);  // a round without drafts shows nothing
    CHECK(b.rate(1) == before);
    CHECK_THROWS_AS(b.round(2, 3), xs::Error);
}

TEST_CASE("draft counts: the most tokens a millisecond, costs measured past each count's first round") {
    const auto flat = [](double rate) { return Acceptance(rate); };
    const auto measure = [](DraftLength& d, int k, double draft_ms, double verify_ms) {
        for (int i = 0; i < 20; ++i) d.record(k, draft_ms, verify_ms);
    };
    DraftLength d(6);
    CHECK(d.best(flat(0.8), 6) == 1);  // nothing measured: one draft first
    d.record(1, 0.1, 90.0);           // a count's first round pays a new step's setup: not counted
    CHECK(d.cost(1) == 0.0);
    d.record(1, 0.1, 10.0);
    CHECK(d.cost(1) == doctest::Approx(10.1));
    CHECK(d.cost(4) == doctest::Approx(10.1));  // past the counts measured: as the largest, until another is
    CHECK(d.best(flat(0.8), 6) == 6);           // so the most drafts are tried next
    CHECK(d.best(flat(0.8), 3) == 3);           // within the limit
    // A verify that costs the same up to 4 drafts and doubles past them (rows past the fast path).
    for (int k = 2; k <= 6; ++k) measure(d, k, 0.1 * k, k <= 4 ? 10.0 : 20.0);
    CHECK(d.cost(5) == doctest::Approx(20.5));
    CHECK(d.best(flat(0.9), 6) == 4);   // 4.1 tokens / 10.4 ms beats 5.3 / 20.5
    CHECK(d.best(flat(0.2), 6) == 2);   // little accepted: a second draft costs 0.1 ms and pays, a third not
    CHECK(d.best(flat(0.97), 6) == 4);  // even near-certain drafts do not pay for twice the verify
    // Between counts measured, on their line; past them on the line of the last two, never falling.
    DraftLength line(7);
    measure(line, 1, 3.0, 10.0);
    measure(line, 3, 9.0, 10.0);
    CHECK(line.cost(2) == doctest::Approx(16.0));
    CHECK(line.cost(5) == doctest::Approx(25.0));
    // A drafter whose every draft costs as much as a verify row would (chain drafters): fewer drafts.
    DraftLength chain(4);
    for (int k = 1; k <= 4; ++k) measure(chain, k, 3.0 * k, 10.0);
    CHECK(chain.best(flat(0.5), 4) == 1);
    CHECK(chain.best(flat(0.95), 4) == 4);
    // A drafter right on its first draft and wrong after: one draft, however cheap the rest.
    Acceptance first_only(0.5);
    for (int i = 0; i < 60; ++i) first_only.round(3, 1);
    CHECK(chain.best(first_only, 4) == 1);
}

TEST_CASE("adaptive draft counts in the engine: the greedy tokens, rounds of varying drafts") {
    struct Drafting {
        bool lookup, block, batch;
    };
    for (const Drafting d : {Drafting{false, false, false}, Drafting{false, true, false}, Drafting{true, false, false},
                             Drafting{true, false, true}, Drafting{false, false, true}}) {
        CAPTURE(d.lookup);
        CAPTURE(d.block);
        CAPTURE(d.batch);
        const auto plain = make_engine(0, 8);
        const int k = 4;
        const auto adaptive = d.lookup ? make_lookup_engine(k, 8, {1, 3}, false, 2, d.batch, 64, true)
                                       : make_engine(k, 8, false, d.block, 2, d.batch, 64, true);
        GenerateStats stats;
        for (const auto& prompt : {std::vector<int32_t>{1, 7, 3, 12, 5, 9, 0, 14, 2, 11}, std::vector<int32_t>{6, 2, 13},
                                   std::vector<int32_t>{4, 4, 4, 4, 9, 1}}) {
            const GenerateRequest r = greedy(prompt, 60);
            const GenerateResult want = plain->generate(r, {});
            const GenerateResult got = adaptive->generate(r, {});
            CHECK(got.error == "");
            CHECK(got.tokens == want.tokens);
            stats.spec_rounds += got.stats.spec_rounds;
            stats.spec_drafted += got.stats.spec_drafted;
        }
        CHECK(stats.spec_rounds > 0);
        if (!d.lookup) CHECK(stats.spec_drafted < stats.spec_rounds * k);  // some rounds drafted fewer than k
    }
}

}  // namespace xs
