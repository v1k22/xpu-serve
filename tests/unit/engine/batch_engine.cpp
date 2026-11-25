// Decode steps of the requests in flight run as one (EngineOptions::batch), on cpu-ref over a model with every kind of
// state: each request emits the tokens and finish it gets alone, greedy or sampled, reading each row's argmax or the
// logits, through stop tokens, token budgets, prefill chunks between decode steps and a request starting late.
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
    for (const bool argmax : {false, true}) {
        // b stops at its sixth token, wherever that comes first.
        b.stop_token_ids.clear();
        b.stop_token_ids = {make_engine(0, 3, argmax)->generate(b, {}).tokens.at(5)};
        std::vector<GenerateResult> alone;
        for (const GenerateRequest* r : {&a, &b, &c}) alone.push_back(make_engine(0, 3, argmax)->generate(*r, {}));
        for (const bool batch : {true, false}) {
            CAPTURE(argmax);
            CAPTURE(batch);
            const auto engine = make_engine(0, 3, argmax, false, 3, batch);
            CHECK(engine->options().batch == batch);
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
            }
            CHECK(got[1].finish == FinishReason::Stop);
            CHECK(engine->has_free_sequence());
        }
    }
}

}  // namespace xs
