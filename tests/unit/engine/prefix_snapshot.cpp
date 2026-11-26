// A model with conv and recurrent state reuses a prompt's prefix from its sequence's snapshot (decision 0029), on
// cpu-ref over the tiny model with every kind of state: a turn whose prompt departs from the cached one after the
// point its predecessor named (GenerateRequest::snapshot_at) starts from there and gives the tokens it gets from the
// start; one departing before that point, or after a request that named none, starts over; of two free sequences a
// request takes the one it can use the most of; and one whose own sequence offers less takes another's prefix as a
// copy, the other going on unchanged. Without a drafter and with one of each kind.
#include <doctest/doctest.h>

#include <vector>

#include "tiny_engine.h"

namespace xs {

using test::greedy;
using test::make_engine;

namespace {

GenerateRequest turn(std::vector<int32_t> prompt, int max_new, int64_t snapshot_at) {
    GenerateRequest r = greedy(std::move(prompt), max_new);
    r.snapshot_at = snapshot_at;
    return r;
}

struct Drafting {
    int spec_k;
    bool block;
};
constexpr Drafting kDraftings[] = {{0, false}, {2, false}, {3, true}};

}  // namespace

TEST_CASE("a prompt that departs from the cache after the snapshot point starts from the snapshot") {
    for (const Drafting d : kDraftings) {
        for (const int64_t chunk : {int64_t{3}, int64_t{64}}) {
            CAPTURE(d.spec_k);
            CAPTURE(chunk);
            const auto fresh = [&](const GenerateRequest& r) {
                return make_engine(d.spec_k, chunk, false, d.block)->generate(r, {});
            };
            const auto engine = make_engine(d.spec_k, chunk, false, d.block);
            // A conversation: each turn's messages, then a generation prompt (13, 14) the next turn renders otherwise.
            const GenerateRequest a = turn({1, 2, 3, 4, 5, 6, 7, 8, 13, 14}, 6, 8);
            CHECK(engine->generate(a, {}).tokens == fresh(a).tokens);
            const GenerateRequest b = turn({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 7, 12);
            const GenerateResult got_b = engine->generate(b, {});
            CHECK(got_b.stats.cached_tokens == 8);
            CHECK(got_b.tokens == fresh(b).tokens);
            const GenerateRequest c = turn({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 2, 13, 14}, 5, 14);
            const GenerateResult got_c = engine->generate(c, {});
            CHECK(got_c.stats.cached_tokens == 12);
            CHECK(got_c.tokens == fresh(c).tokens);
            // The same prompt again (a regenerated reply): from the snapshot, before the generation prompt.
            const GenerateResult again = engine->generate(c, {});
            CHECK(again.stats.cached_tokens == 14);
            CHECK(again.tokens == got_c.tokens);
            // Departing before the snapshot point: from the start.
            const GenerateRequest e = turn({1, 2, 3, 4, 5, 0, 0, 13, 14}, 4, 7);
            const GenerateResult got_e = engine->generate(e, {});
            CHECK(got_e.stats.cached_tokens == 0);
            CHECK(got_e.tokens == fresh(e).tokens);
            // After a request that named no point, a prompt departing inside the cache starts over.
            const GenerateRequest f = turn({4, 4, 4, 4, 4, 4, 13, 14}, 4, 0);
            CHECK(engine->generate(f, {}).stats.cached_tokens == 0);
            const GenerateRequest g = turn({4, 4, 4, 4, 4, 4, 9, 13, 14}, 4, 0);
            const GenerateResult got_g = engine->generate(g, {});
            CHECK(got_g.stats.cached_tokens == 0);
            CHECK(got_g.tokens == fresh(g).tokens);
        }
    }
}

TEST_CASE("of two free sequences a request takes the one whose prefix it can use, a snapshot's included") {
    for (const Drafting d : kDraftings) {
        CAPTURE(d.spec_k);
        const auto engine = make_engine(d.spec_k, 64, false, d.block, 2);
        // a: a conversation that saved its state after its 8 tokens of messages; o: a prompt that shares 10 tokens
        // with the next one, but no snapshot, so they are of no use to a prompt departing inside its cache.
        const GenerateRequest a = turn({1, 2, 3, 4, 5, 6, 7, 8, 13, 14}, 4, 8);
        const GenerateRequest o = turn({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0}, 4, 0);
        const auto ha = engine->start(a, {});
        const auto ho = engine->start(o, {});
        REQUIRE(ha.has_value());
        REQUIRE(ho.has_value());
        bool done_a = false, done_o = false;
        while (!done_a || !done_o) {
            if (!done_a) done_a = engine->advance(*ha);
            if (!done_o) done_o = engine->advance(*ho);
        }
        engine->finish(*ha);
        engine->finish(*ho);
        const GenerateRequest b = turn({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 13, 14}, 5, 11);
        const GenerateResult got_b = engine->generate(b, {});
        CHECK(got_b.stats.cached_tokens == 8);
        CHECK(got_b.tokens == make_engine(d.spec_k, 64, false, d.block)->generate(b, {}).tokens);
    }
}

TEST_CASE("a request takes a busy sequence's prefix as a copy when its own offers less") {
    for (const int shared : {80, 40}) {  // a conversation's messages; 40 tokens are not worth a copy (kCopyGain)
        std::vector<int32_t> history;
        for (int i = 0; i < shared; ++i) history.push_back(static_cast<int32_t>((i * 5 + 1) % 13));
        const auto with = [&](std::vector<int32_t> tail) {
            std::vector<int32_t> p = history;
            p.insert(p.end(), tail.begin(), tail.end());
            return p;
        };
        for (const Drafting d : kDraftings) {
            CAPTURE(shared);
            CAPTURE(d.spec_k);
            const auto fresh = [&](const GenerateRequest& r) {
                return make_engine(d.spec_k, 32, false, d.block)->generate(r, {});
            };
            const auto engine = make_engine(d.spec_k, 32, false, d.block, 2);
            const GenerateRequest a = turn(with({13, 14}), 4, shared);
            const GenerateResult got_a = engine->generate(a, {});
            // e extends a's whole cache, so it takes a's sequence (keeping its snapshot), and is still decoding when
            // c comes; c shares a's messages, the free sequence nothing.
            std::vector<int32_t> e_prompt = a.prompt;
            e_prompt.insert(e_prompt.end(), got_a.tokens.begin(), got_a.tokens.end());
            e_prompt.push_back(5);
            const GenerateRequest e = turn(e_prompt, 8, 0);
            const auto he = engine->start(e, {});
            REQUIRE(he.has_value());
            while (!engine->decoding(*he)) REQUIRE_FALSE(engine->advance(*he));
            REQUIRE_FALSE(engine->advance(*he));
            const GenerateRequest c = turn(with({9, 10, 11, 13, 14}), 6, shared + 3);
            const auto hc = engine->start(c, {});
            REQUIRE(hc.has_value());
            bool done_c = false, done_e = false;
            while (!done_c || !done_e) {
                if (!done_c) done_c = engine->advance(*hc);
                if (!done_e) done_e = engine->advance(*he);
            }
            const GenerateResult got_c = engine->finish(*hc);
            CHECK(got_c.stats.cached_tokens == (shared >= 64 ? shared : 0));
            CHECK(got_c.tokens == fresh(c).tokens);
            CHECK(engine->finish(*he).tokens == fresh(e).tokens);
        }
    }
}

}  // namespace xs
