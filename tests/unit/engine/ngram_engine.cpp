// Prompt lookup (engine/spec/ngram_drafter.h): the drafts it finds in a sequence (the longest suffix at its first
// occurrence, as vLLM's CPU proposer found them); and in the engine, on cpu-ref over a model with every kind of state, greedy output that equals greedy output
// without drafts, alone and batched, through prefix reuse, a prefix copied from another sequence, stop tokens and
// rounds cut short. The tiny model's output cycles, so its own text repeats and lookups find drafts in it.
#include <doctest/doctest.h>

#include <vector>

#include "engine/spec/ngram_drafter.h"
#include "tiny_engine.h"

namespace xs {

using spec::lookup_drafts;
using spec::NgramOptions;
using test::greedy;
using test::make_engine;
using test::make_lookup_engine;

using Tokens = std::vector<int32_t>;

TEST_CASE("prompt lookup: the longest suffix that occurred before, at its first occurrence, and what followed it") {
    // The suffix 7 8 occurs twice before: the first occurrence's continuation.
    CHECK(lookup_drafts(Tokens{1, 7, 8, 2, 3, 7, 8, 4, 5, 9, 7, 8}, {2, 2}, 3) == Tokens{2, 3, 7});
    // The longest suffix wins over shorter ones that occur earlier: 6 7 8 over 7 8.
    CHECK(lookup_drafts(Tokens{7, 8, 1, 6, 7, 8, 2, 9, 1, 7, 8, 6, 7, 8}, {1, 3}, 2) == Tokens{2, 9});
    // ... up to max: past it, all matches of max tokens tie and the first wins.
    CHECK(lookup_drafts(Tokens{7, 8, 1, 6, 7, 8, 2, 9, 1, 7, 8, 6, 7, 8}, {1, 2}, 2) == Tokens{1, 6});
    // Shorter than min: nothing; no occurrence at all: nothing.
    CHECK(lookup_drafts(Tokens{7, 8, 1, 6, 7, 8, 2, 9, 6, 7, 8}, {4, 5}, 2).empty());
    CHECK(lookup_drafts(Tokens{1, 2, 3, 4, 5}, {1, 3}, 2).empty());
    CHECK(lookup_drafts(Tokens{5}, {1, 3}, 2).empty());
    // Never past the end of the sequence: a match near the end gives what is left.
    CHECK(lookup_drafts(Tokens{1, 2, 3, 1, 2}, {2, 2}, 5) == Tokens{3, 1, 2});
    // A run of one token: its first occurrence, drafts through the run.
    CHECK(lookup_drafts(Tokens{4, 4, 4, 4, 4}, {2, 2}, 8) == Tokens{4, 4, 4});
    // k caps the drafts.
    CHECK(lookup_drafts(Tokens{1, 2, 3, 4, 5, 6, 1, 2}, {2, 2}, 3) == Tokens{3, 4, 5});
    CHECK(lookup_drafts(Tokens{1, 2, 3, 4, 5, 6, 1, 2}, {2, 2}, 0).empty());
    // An unknown position (-1) matches nothing, not even another, and ends the drafts.
    CHECK(lookup_drafts(Tokens{-1, -1, 9, -1, -1}, {2, 2}, 3).empty());
    CHECK(lookup_drafts(Tokens{1, 2, 3, -1, 5, 1, 2}, {2, 2}, 4) == Tokens{3});
    CHECK(lookup_drafts(Tokens{1, 2, -1, 1, 2, 7, 8, 1, 2}, {2, 2}, 4).empty());  // the first occurrence's
    CHECK(lookup_drafts(Tokens{1, 2, 7, 8, -1, 1, 2, -1, 1, 2}, {2, 2}, 4) == Tokens{7, 8});
}

TEST_CASE("prompt lookup's drafter: the sequence it was given, cut back, adopted, and the pending token") {
    spec::NgramDrafter d({2, 2}, 4);
    const Tokens prompt{1, 2, 3, 4, 5, 1, 2, 3};
    d.add({}, 8, 0, prompt);
    CHECK(d.propose(4, 4) == Tokens{5, 1, 2, 3});  // ... 3 4 after the pending 4
    // A verify of 4 tokens from position 8 that keeps 2 of them.
    d.add({}, 2, 8, Tokens{4, 5, 9, 9});
    CHECK(d.propose(1, 9) == Tokens{2, 3, 4, 5});  // max_drafts caps k
    d.truncate(3);
    CHECK(d.propose(9, 3).empty());  // 3 9 never occurred
    // A gap: positions it was never given are unknown.
    d.add({}, 2, 6, Tokens{1, 2});
    CHECK(d.propose(3, 4).empty());
    d.reset();
    CHECK(d.propose(3, 4).empty());
    // Tokens copied from another sequence, a media item's position among them: 7 8 at 1, then 5, then the unknown.
    d.adopt(Tokens{6, 7, 8, 5, -1, 6, 7});
    CHECK(d.propose(8, 3) == Tokens{5});
}

TEST_CASE("prompt lookup in the engine emits exactly the greedy tokens, its drafts found in the text") {
    for (const int k : {1, 3, 7}) {
        for (const int64_t chunk : {int64_t{3}, int64_t{64}}) {
            for (const bool argmax : {false, true}) {
                CAPTURE(k);
                CAPTURE(chunk);
                CAPTURE(argmax);
                const auto plain = make_engine(0, chunk, argmax);
                const auto lookup = make_lookup_engine(k, chunk, {2, 3}, argmax);
                GenerateStats stats;
                const auto same = [&](const GenerateRequest& req) {
                    const GenerateResult want = plain->generate(req, {});
                    const GenerateResult got = lookup->generate(req, {});
                    REQUIRE(want.finish != FinishReason::Error);
                    CHECK(got.error == "");
                    CHECK(got.tokens == want.tokens);
                    CHECK(got.finish == want.finish);
                    CHECK(got.stats.cached_tokens == want.stats.cached_tokens);
                    stats.spec_rounds += got.stats.spec_rounds;
                    stats.spec_drafted += got.stats.spec_drafted;
                    stats.spec_accepted += got.stats.spec_accepted;
                    return want.tokens;
                };
                const Tokens first = {1, 7, 3, 12, 5, 9, 0, 14, 2, 11};
                const Tokens out = same(greedy(first, 40));
                Tokens extended = first;  // the next turn extends the cached sequence
                extended.insert(extended.end(), out.begin(), out.end());
                extended.insert(extended.end(), {4, 8, 15});
                const Tokens out2 = same(greedy(extended, 30));
                same(greedy({6, 2, 13}, 7));  // a new sequence; a round cut short by max_new_tokens
                same(greedy({6, 2, 13, 1}, 30, {out2.at(9), out2.at(17)}));  // stop tokens, possibly inside a round
                CHECK(stats.spec_accepted > 0);
                CHECK(stats.spec_drafted <= stats.spec_rounds * k);
            }
        }
    }
}

TEST_CASE("prompt lookup with requests in flight: verified as one step, each as it runs alone") {
    const GenerateRequest a = greedy({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 30);
    const GenerateRequest b = greedy({3, 1, 4, 1, 5, 9, 2, 6}, 25);
    GenerateRequest c = greedy({8, 8, 2}, 21);
    c.sampling.temperature = 0.9f;
    c.sampling.seed = 3;
    const NgramOptions n{1, 4};
    for (const bool batch : {true, false}) {
        CAPTURE(batch);
        std::vector<GenerateResult> alone;
        for (const GenerateRequest* r : std::initializer_list<const GenerateRequest*>{&a, &b, &c}) {
            alone.push_back(make_lookup_engine(4, 3, n)->generate(*r, {}));
        }
        const auto engine = make_lookup_engine(4, 3, n, false, 3, batch, 12);
        std::vector<Engine::Handle> handles{*engine->start(a, {}), *engine->start(b, {}), *engine->start(c, {})};
        for (const Engine::Handle h : handles) {
            while (!engine->decoding(h)) REQUIRE_FALSE(engine->advance(h));
        }
        std::vector<GenerateResult> got(3);
        std::vector<bool> finished(3, false);
        while (std::find(finished.begin(), finished.end(), false) != finished.end()) {
            std::vector<Engine::Handle> step;
            std::vector<size_t> at;
            for (size_t i = 0; i < 3; ++i) {
                if (finished[i]) continue;
                step.push_back(handles[i]);
                at.push_back(i);
            }
            const std::vector<bool> done = engine->advance_decoding(step);
            for (size_t j = 0; j < at.size(); ++j) {
                if (!done[j]) continue;
                got[at[j]] = engine->finish(handles[at[j]]);
                finished[at[j]] = true;
            }
        }
        for (size_t i = 0; i < 3; ++i) {
            CAPTURE(i);
            CHECK(got[i].error == "");
            CHECK(got[i].tokens == alone[i].tokens);
            if (i < 2) CHECK(got[i].tokens == make_engine(0, 3)->generate(i == 0 ? a : b, {}).tokens);
            CHECK(got[i].stats.spec_accepted > 0);
        }
    }
}

TEST_CASE("prompt lookup after a prefix copied from another sequence: it looks up the copied tokens too") {
    // A request on sequence 0 saves its state where its messages end (90 tokens); the next turn of that conversation
    // holds sequence 0 while a second request, sharing the 90 tokens, copies them to sequence 1 (Engine::take_prefix)
    // and decodes there, its drafter holding the copied tokens (NgramDrafter::adopt) beside its own.
    Tokens shared;
    for (int i = 0; i < 90; ++i) shared.push_back(static_cast<int32_t>((7 * i + 3) % 16));
    GenerateRequest first = greedy(shared, 2);
    first.prompt.push_back(5);
    first.snapshot_at = static_cast<int64_t>(shared.size());
    GenerateRequest second = greedy(shared, 24);
    second.prompt.insert(second.prompt.end(), {9, 9});
    const auto engine = make_lookup_engine(4, 64, {2, 4}, false, 2);
    const GenerateResult got_first = engine->generate(first, {});
    GenerateRequest next_turn = first;
    next_turn.prompt.insert(next_turn.prompt.end(), got_first.tokens.begin(), got_first.tokens.end());
    next_turn.prompt.push_back(3);
    REQUIRE(engine->start(next_turn, {}).has_value());  // on sequence 0, which holds its conversation
    const auto h = engine->start(second, {});
    REQUIRE(h.has_value());
    while (!engine->advance(*h)) {
    }
    const GenerateResult got = engine->finish(*h);
    CHECK(got.error == "");
    CHECK(got.stats.cached_tokens == static_cast<int>(shared.size()));  // copied
    CHECK(got.tokens == make_engine(0, 64)->generate(second, {}).tokens);
    CHECK(got.stats.spec_accepted > 0);
}

}  // namespace xs
