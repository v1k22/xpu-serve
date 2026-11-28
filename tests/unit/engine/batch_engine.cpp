// Decode steps of the requests in flight run as one (EngineOptions::batch), on cpu-ref over a model with every kind of
// state: each request emits the tokens and finish it gets alone, greedy or sampled, reading each row's argmax or the
// logits, through stop tokens, token budgets, prefill chunks between decode steps and a request starting late; and
// speculating with a chain or a block drafter, the rounds of all verified as one step (a chain drafter's drafts of all
// run as one run a depth too).
#include <doctest/doctest.h>

#include <vector>

#include "tiny_engine.h"

namespace xs {

using test::greedy;
using test::make_engine;

TEST_CASE("decode steps of several requests run as one: each request's tokens and finish as it runs alone") {
    GenerateRequest a = greedy({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 9);
    GenerateRequest b = greedy({3, 1, 4, 1, 5, 9, 2, 6}, 14);
    GenerateRequest c = greedy({8, 8, 2}, 11);
    c.sampling.temperature = 0.9f;
    c.sampling.seed = 3;
    struct Drafting {
        int spec_k;
        bool block;
    };
    for (const Drafting d : {Drafting{0, false}, Drafting{2, false}, Drafting{3, true}})
    for (const bool argmax : {false, true}) {
        CAPTURE(d.spec_k);
        CAPTURE(d.block);
        // b stops at its sixth token, wherever that comes first.
        b.stop_token_ids.clear();
        b.stop_token_ids = {make_engine(0, 3, argmax)->generate(b, {}).tokens.at(5)};
        std::vector<GenerateResult> alone;
        for (const GenerateRequest* r : {&a, &b, &c}) {
            alone.push_back(make_engine(d.spec_k, 3, argmax, d.block)->generate(*r, {}));
        }
        for (const bool batch : {true, false}) {
            CAPTURE(argmax);
            CAPTURE(batch);
            const auto engine = make_engine(d.spec_k, 3, argmax, d.block, 3, batch);
            CHECK(engine->options().batch == batch);
            // A chain drafter's drafts of all run as one run a depth (its batch draft program).
            CHECK(engine->options().batch_draft == (batch && d.spec_k > 0 && !d.block));
            std::vector<Engine::Handle> handles{*engine->start(a, {}), *engine->start(b, {})};
            std::vector<GenerateResult> got(3);
            std::vector<bool> finished(3, false);
            bool c_started = false;
            int turns = 0;
            while (std::find(finished.begin(), finished.end(), false) != finished.end()) {
                // A prefill chunk of each request still reading its prompt, then a step of all that decode.
                std::vector<size_t> decoding;
                for (size_t i = 0; i < handles.size(); ++i) {
                    if (finished[i]) continue;
                    if (!engine->decoding(handles[i])) {
                        if (engine->advance(handles[i])) {
                            got[i] = engine->finish(handles[i]);
                            finished[i] = true;
                        }
                    } else {
                        decoding.push_back(i);
                    }
                }
                std::vector<Engine::Handle> step;
                for (const size_t i : decoding) step.push_back(handles[i]);
                const std::vector<bool> done = engine->advance_decoding(step);
                for (size_t j = 0; j < decoding.size(); ++j) {
                    if (!done[j]) continue;
                    got[decoding[j]] = engine->finish(handles[decoding[j]]);
                    finished[decoding[j]] = true;
                }
                if (!c_started && ++turns == 5) {  // c starts late, while a and b decode
                    handles.push_back(*engine->start(c, {}));
                    c_started = true;
                }
            }
            for (size_t i = 0; i < 3; ++i) {
                CAPTURE(i);
                CHECK(got[i].tokens == alone[i].tokens);
                CHECK(got[i].finish == alone[i].finish);
                if (d.spec_k > 0) {  // its rounds ran, with the drafts and acceptances it has alone
                    CHECK(got[i].stats.spec_rounds == alone[i].stats.spec_rounds);
                    CHECK(got[i].stats.spec_accepted == alone[i].stats.spec_accepted);
                    CHECK(got[i].stats.spec_accepted > 0);
                }
            }
            CHECK(got[1].finish == FinishReason::Stop);
            CHECK(engine->has_free_sequence());
        }
    }
}

TEST_CASE("batched speculative rounds share a verify's rows: fewer drafts, the same tokens") {
    // Three requests over a verify of 4 rows: no drafts while all three decode, one each while two do, then a
    // request's own spec_k.
    const GenerateRequest a = greedy({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 12), b = greedy({3, 1, 4, 1, 5}, 6),
                          c = greedy({8, 8, 2}, 9);
    const auto engine = make_engine(3, 64, false, false, 3, true, 4);
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
    const GenerateRequest* rs[] = {&a, &b, &c};
    int drafted = 0, drafted_alone = 0;
    for (size_t i = 0; i < 3; ++i) {
        CAPTURE(i);
        const GenerateResult alone = make_engine(3, 64)->generate(*rs[i], {});
        CHECK(got[i].tokens == alone.tokens);
        drafted += got[i].stats.spec_drafted;
        drafted_alone += alone.stats.spec_drafted;
    }
    CHECK(drafted > 0);  // once fewer requests decode
    CHECK(drafted < drafted_alone);
}

}  // namespace xs
