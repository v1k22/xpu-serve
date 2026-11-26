// Speculative decoding in the engine, on cpu-ref: greedy output with a drafter equals greedy output without one, over
// a model with every kind of state (conv, gated delta, KV) and a drafter with its own KV that is right only part of
// the time, through prefix reuse, chunked prefill, stop tokens and rounds cut short. And the same tokens when the
// target programs offer each row's argmax and the engine reads that instead of the logits. And with a block drafter:
// one run proposes a round's drafts from state the target's own programs append to.
#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <numeric>

#include "tiny_engine.h"

namespace xs {

using test::greedy;
using test::make_engine;
using test::TinySpecModel;
using graph::Graph;
using graph::kT;
using graph::Value;

TEST_CASE("requests in flight together on sequences of their own: each as it runs alone, and its prefix kept") {
    for (const int k : {0, 2}) {
        CAPTURE(k);
        const GenerateRequest a = greedy({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 9);
        const GenerateRequest b = greedy({3, 1, 4, 1, 5, 9, 2, 6}, 12);
        const GenerateResult alone_a = make_engine(k, 3)->generate(a, {});
        const GenerateResult alone_b = make_engine(k, 3)->generate(b, {});
        const auto engine = make_engine(k, 3, false, false, 2);
        const auto ha = engine->start(a, {});
        const auto hb = engine->start(b, {});
        REQUIRE(ha.has_value());
        REQUIRE(hb.has_value());
        CHECK_FALSE(engine->has_free_sequence());
        CHECK_FALSE(engine->start(a, {}).has_value());  // both sequences are busy
        bool done_a = false, done_b = false;
        while (!done_a || !done_b) {  // a's prefill chunks and decode steps between b's
            if (!done_a) done_a = engine->advance(*ha);
            if (!done_b) done_b = engine->advance(*hb);
        }
        const GenerateResult got_a = engine->finish(*ha);
        const GenerateResult got_b = engine->finish(*hb);
        CHECK(got_a.tokens == alone_a.tokens);
        CHECK(got_b.tokens == alone_b.tokens);
        CHECK(engine->has_free_sequence());
        // A conversation's next turn finds the sequence that holds it (the model's conv state cannot be cut back, so
        // the turn extends all that is cached).
        const auto next_turn = [](const GenerateRequest& r, const GenerateResult& got) {
            GenerateRequest t = r;
            t.prompt.insert(t.prompt.end(), got.tokens.begin(), got.tokens.end());
            t.prompt.push_back(7);
            return t;
        };
        const GenerateRequest b2 = next_turn(b, got_b), a2 = next_turn(a, got_a);
        CHECK(engine->generate(b2, {}).stats.cached_tokens > static_cast<int>(b.prompt.size()));
        const GenerateResult again_a = engine->generate(a2, {});
        CHECK(again_a.stats.cached_tokens > static_cast<int>(a.prompt.size()));
        CHECK(again_a.tokens == make_engine(k, 3)->generate(a2, {}).tokens);
    }
}

TEST_CASE("speculative decoding emits exactly the greedy tokens") {
    for (const int k : {1, 2, 4}) {
        for (const int64_t chunk : {int64_t{3}, int64_t{64}}) {
            CAPTURE(k);
            CAPTURE(chunk);
            const auto plain = make_engine(0, chunk);
            const auto spec = make_engine(k, chunk);
            GenerateStats stats;
            const auto same = [&](const GenerateRequest& req) {
                const GenerateResult want = plain->generate(req, {});
                const GenerateResult got = spec->generate(req, {});
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
            const std::vector<int32_t> first = {1, 7, 3, 12, 5, 9, 0, 14, 2, 11};
            const std::vector<int32_t> out = same(greedy(first, 40));
            std::vector<int32_t> extended = first;  // the next turn extends the cached sequence
            extended.insert(extended.end(), out.begin(), out.end());
            extended.insert(extended.end(), {4, 8, 15});
            const std::vector<int32_t> out2 = same(greedy(extended, 30));
            same(greedy({6, 2, 13}, 7));  // a new sequence; a round cut short by max_new_tokens
            same(greedy({6, 2, 13, 1}, 30, {out2.at(9), out2.at(17)}));  // stop tokens, possibly inside a round
            CHECK(stats.spec_accepted > 0);                    // drafts are accepted ...
            CHECK(stats.spec_accepted < stats.spec_drafted);   // ... and rejected
        }
    }
}

TEST_CASE("a block drafter: one run proposes a round's drafts, and the greedy tokens are the same") {
    for (const int k : {1, 2, 4}) {
        for (const int64_t chunk : {int64_t{3}, int64_t{64}}) {
            CAPTURE(k);
            CAPTURE(chunk);
            const auto plain = make_engine(0, chunk);
            const auto spec = make_engine(k, chunk, k == 2, true);
            GenerateStats stats;
            const auto same = [&](const GenerateRequest& req) {
                const GenerateResult want = plain->generate(req, {});
                const GenerateResult got = spec->generate(req, {});
                REQUIRE(want.finish != FinishReason::Error);
                CHECK(got.error == "");
                CHECK(got.tokens == want.tokens);
                CHECK(got.finish == want.finish);
                stats.spec_drafted += got.stats.spec_drafted;
                stats.spec_accepted += got.stats.spec_accepted;
                return want.tokens;
            };
            const std::vector<int32_t> first = {1, 7, 3, 12, 5, 9, 0, 14, 2, 11};
            const std::vector<int32_t> out = same(greedy(first, 40));
            std::vector<int32_t> extended = first;  // the next turn extends the cached sequence
            extended.insert(extended.end(), out.begin(), out.end());
            extended.insert(extended.end(), {4, 8, 15});
            const std::vector<int32_t> out2 = same(greedy(extended, 30));
            same(greedy({6, 2, 13}, 7));  // a new sequence; a round cut short by max_new_tokens
            same(greedy({6, 2, 13, 1}, 30, {out2.at(9), out2.at(17)}));
            CHECK(stats.spec_accepted > 0);
            CHECK(stats.spec_accepted < stats.spec_drafted);
        }
    }
    CHECK_THROWS_AS(make_engine(5, 64, false, true), xs::Error);  // more drafts a round than the block holds
}

TEST_CASE("a block's drafts: as given, or the path of the best score of each row through the candidates") {
    TensorMap given;
    given["drafts"] = test::make_tensor(DType::I32, {3}, std::vector<int32_t>{9, 4, 7});
    CHECK(spec::block_drafts(given) == std::vector<int32_t>{9, 4, 7});
    // three drafts, two candidates each; scores[i][j][k]: the draft before took its candidate j
    TensorMap paths;
    paths["candidates"] = test::make_tensor(DType::I32, {3, 2}, std::vector<int32_t>{10, 11, 20, 21, 30, 31});
    paths["scores"] = test::make_tensor(DType::F32, {3, 2, 2},
                                        std::vector<float>{0.1f, 0.9f, /* unused row */ 5.0f, 0.0f,  // draft 0: 11
                                                           1.0f, 0.0f, /* after 11 */ 0.2f, 0.3f,     // draft 1: 21
                                                           0.0f, 1.0f, /* after 21 */ 0.7f, 0.7f});   // tie: the first
    CHECK(spec::block_drafts(paths) == std::vector<int32_t>{11, 21, 30});
}

TEST_CASE("programs that offer each row's argmax: greedy decoding reads it, sampling still reads the logits") {
    for (const int k : {0, 3}) {
        CAPTURE(k);
        const auto by_logits = make_engine(k, 5);
        const auto by_argmax = make_engine(k, 5, true);
        const auto same = [&](const GenerateRequest& req) {
            const GenerateResult want = by_logits->generate(req, {});
            const GenerateResult got = by_argmax->generate(req, {});
            REQUIRE(want.finish != FinishReason::Error);
            CHECK(got.error == "");
            CHECK(got.tokens == want.tokens);
            CHECK(got.stats.spec_accepted == want.stats.spec_accepted);
        };
        same(greedy({1, 7, 3, 12, 5, 9, 0, 14, 2, 11}, 40));
        GenerateRequest sampled = greedy({6, 2, 13, 1}, 30);  // a seeded sampler draws the same tokens from the logits
        sampled.sampling.temperature = 0.9f;
        sampled.sampling.seed = 11;
        same(sampled);
        GenerateRequest penalized = greedy({6, 2, 13, 1, 4}, 30);  // greedy, but the penalty needs the logits
        penalized.sampling.repetition_penalty = 1.3f;
        same(penalized);
    }
}

namespace {

// A block drafter that asks for a row of the target's last token (BlockDrafting::target_rows), over a session that
// only records what the draft program is given.
struct RowDrafting final : BlockDrafting, Speculation {
    std::vector<std::string> rows{"context"};
    Graph none;
    int64_t block_drafts() const override { return 3; }
    void prepare_block(int32_t token, int64_t, TensorMap& inputs) const override {
        inputs["token_ids"] = HostTensor::from_vector(DType::I32, {1}, std::vector<int32_t>{token});
    }
    std::span<const std::string> target_rows() const override { return rows; }
    const Graph& prefill_graph() const override { return none; }
    const Graph& verify_graph() const override { return none; }
    const Graph& draft_graph() const override { return none; }
    const BlockDrafting* block() const override { return this; }
    std::string_view drafter_name() const override { return "rows"; }
};

struct RecordingSession final : Session {
    int64_t cached = 0;
    TensorMap given;  // the draft program's inputs, last run
    void run(size_t program, const TensorMap& inputs, TensorMap& outputs, std::span<const std::string>) override {
        REQUIRE(program == spec::kDraft);
        given.clear();
        for (const auto& [name, t] : inputs) given[name] = t.clone();
        outputs["drafts"] = HostTensor::from_vector(DType::I32, {3}, std::vector<int32_t>{4, 5, 6});
        cached += 4;  // the block's rows, which the runner cuts back
    }
    int64_t num_cached(size_t) const override { return cached; }
    void reset() override { cached = 0; }
    void truncate(size_t, int64_t n) override { cached = n; }
    StateSnapshot save_state(size_t) const override { return {cached, {}}; }
    void restore_state(size_t, const StateSnapshot& s) override { cached = s.tokens; }
    void use(int64_t sequence) override { REQUIRE(sequence == 0); }
    int64_t current_sequence() const override { return 0; }
};

// What a target's run of `rows` tokens gives the drafter: row r holds r + `first` in every column.
TensorMap target_rows(int64_t rows, float first) {
    HostTensor t(DType::F32, {rows, 2});
    for (int64_t r = 0; r < rows; ++r) t.data<float>()[2 * r] = t.data<float>()[2 * r + 1] = first + static_cast<float>(r);
    TensorMap out;
    out["context"] = std::move(t);
    return out;
}

}  // namespace

TEST_CASE("a block drafter that reads the target's last token: the row of the last token kept, after every step") {
    const RowDrafting drafting;
    RecordingSession session;
    const auto drafter = spec::make_drafter(drafting, session);
    CHECK(drafter->target_outputs().size() == 1);
    const auto context = [&] { return session.given.at("context").data<float>()[0]; };
    const std::vector<int32_t> five(5, 1), four(4, 1);

    // A prefill of 5 tokens that gives a row per token, then one that gives its last token's alone.
    drafter->add(target_rows(5, 10.0f), 5, 0, five);
    session.cached = 5;
    CHECK(drafter->propose(7, 3) == std::vector<int32_t>{4, 5, 6});
    CHECK(session.given.at("context").shape() == Dims{1, 2});
    CHECK(context() == 14.0f);
    CHECK(session.cached == 5);  // the block's rows are cut back
    drafter->add(target_rows(1, 30.0f), 5, 5, five);
    session.cached = 10;
    drafter->propose(7, 2);
    CHECK(context() == 30.0f);

    // A verify of the pending token and 3 drafts that keeps 2 rows: the second row's, not the last's.
    drafter->add(target_rows(4, 40.0f), 2, 10, four);
    session.cached = 12;
    drafter->propose(7, 3);
    CHECK(context() == 41.0f);
    // One that keeps them all, and one that keeps the pending token alone.
    drafter->add(target_rows(4, 50.0f), 4, 12, four);
    session.cached = 16;
    drafter->propose(7, 3);
    CHECK(context() == 53.0f);
    drafter->add(target_rows(4, 60.0f), 1, 16, four);
    session.cached = 17;
    drafter->propose(7, 3);
    CHECK(context() == 60.0f);

    // The sequence cut back to another token: its row is gone, and a draft before the target runs again is refused.
    drafter->truncate(17);  // where it already is: the row still holds
    drafter->propose(7, 3);
    drafter->truncate(9);
    session.cached = 9;
    CHECK_THROWS_AS(drafter->propose(7, 3), xs::Error);
    drafter->add(target_rows(1, 70.0f), 3, 9, std::vector<int32_t>(3, 1));
    session.cached = 12;
    drafter->propose(7, 3);
    CHECK(context() == 70.0f);
    // A run without the row, or one that gives a row per token of another count, is refused by name.
    CHECK_THROWS_WITH_AS(drafter->add(TensorMap{}, 4, 12, four), doctest::Contains("'context'"), xs::Error);
    CHECK_THROWS_WITH_AS(drafter->add(target_rows(3, 0.0f), 2, 12, four), doctest::Contains("'context'"), xs::Error);
    // A last row alone cannot serve a step cut inside: which token it belongs to is not the kept one.
    CHECK_THROWS_WITH_AS(drafter->add(target_rows(1, 0.0f), 2, 12, four), doctest::Contains("'context'"), xs::Error);
}

}  // namespace xs
