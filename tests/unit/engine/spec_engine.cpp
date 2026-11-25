// Speculative decoding in the engine, on cpu-ref: greedy output with a drafter equals greedy output without one, over
// a model with every kind of state (conv, gated delta, KV) and a drafter with its own KV that is right only part of
// the time, through prefix reuse, chunked prefill, stop tokens and rounds cut short. And the same tokens when the
// target programs offer each row's argmax and the engine reads that instead of the logits. And with a block drafter:
// one run proposes a round's drafts from state the target's own programs append to.
#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <numeric>

#include "../backend/backend_test_util.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "engine/engine.h"
#include "engine/spec/block_drafter.h"
#include "models/common/blocks.h"
#include "util/error.h"

namespace xs {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// The next token is mostly f(t) = (5 t + 3) mod V: lm_head's row f(t) is t's embedding. A conv + gated delta layer
// and an attention layer perturb the hidden state enough to change it sometimes. The drafter predicts f of its token
// (but f + 1 after every third token) from the embedding, perturbed a little by the target's hidden and by attention
// over its own KV.
class TinySpecModel final : public Model, public Speculation, public ChainDrafting, public BlockDrafting {
public:
    static constexpr int64_t kVocab = 16, kHidden = 24, kHeads = 2, kDim = 8, kKernel = 4;
    static constexpr float kMix = 4.0f;        // how much the target's layers move the hidden state
    static constexpr float kDraftMix = 0.15f;  // how much the drafter's
    static constexpr int64_t kBlockDrafts = 4;   // of the block drafter, after the pending token

    // `argmax`: the target programs also output each row's argmax (what greedy decoding reads instead of the logits).
    // `block`: the drafter proposes a block a run instead of a token (BlockDrafting): the target's prefill and verify
    // programs append a context row per token to its KV slot, and its program attends over them and its own block.
    // It is handed its guesses, right except after every third token, and gives them back unless its state is off.
    explicit TinySpecModel(bool argmax = false, bool block = false) : argmax_(argmax), block_(block) {
        test::TensorRng rng(7);
        const HostTensor embed = rng.uniform(DType::F32, {kVocab, kHidden});
        HostTensor head(DType::F32, {kVocab, kHidden}), draft_head(DType::F32, {kVocab, kHidden});
        std::fill_n(draft_head.data<float>(), draft_head.numel(), 0.0f);
        for (int64_t t = 0; t < kVocab; ++t) {
            const int64_t next = (5 * t + 3) % kVocab;
            const float* e = embed.data<float>() + t * kHidden;
            std::copy_n(e, kHidden, head.data<float>() + next * kHidden);
            // The drafter is wrong after every third token.
            float* row = draft_head.data<float>() + (t % 3 == 0 ? (next + 1) % kVocab : next) * kHidden;
            for (int64_t c = 0; c < kHidden; ++c) row[c] += e[c];
        }
        weights_.add("embed", embed);
        weights_.add("lm_head", head);
        weights_.add("draft.head", draft_head);
        const int64_t width = 3 * kHeads * kDim;
        const auto add = [&](const std::string& name, Dims shape) {
            weights_.add(name, rng.uniform(DType::F32, shape, 1.0f / std::sqrt(static_cast<float>(shape.back()))));
        };
        add("in", {width, kHidden});
        add("conv", {width, kKernel});
        add("decay", {kHeads, kHidden});
        add("beta", {kHeads, kHidden});
        add("out", {kHidden, kHeads * kDim});
        for (const std::string p : {"attn.", "draft."}) {
            add(p + "q", {kHeads * kDim, kHidden});
            add(p + "k", {kDim, kHidden});
            add(p + "v", {kDim, kHidden});
            add(p + "o", {kHidden, kHeads * kDim});
        }
        add("draft.hidden", {kHidden, kHidden});
        info_.arch = "tiny";
        info_.id = "tiny-spec";
        info_.vocab_size = kVocab;
        info_.hidden_size = kHidden;
        info_.max_position = 4096;
        build_target(step_, false, false);
        build_target(prefill_, false, true);
        build_target(verify_, true, true);
        block_ ? build_block() : build_draft();
    }

    const ModelInfo& info() const override { return info_; }
    const Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return weights_; }
    const Speculation* speculation() const override { return this; }
    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const>,
                        TensorMap& inputs) const override {
        const auto n = static_cast<int64_t>(tokens.size());
        std::vector<int32_t> pos(static_cast<size_t>(n));
        std::iota(pos.begin(), pos.end(), static_cast<int32_t>(start_pos));
        inputs["token_ids"] = HostTensor::from_vector(DType::I32, {n}, std::vector<int32_t>(tokens.begin(), tokens.end()));
        inputs["positions"] = HostTensor::from_vector(DType::I32, {n}, pos);
    }

    const Graph& prefill_graph() const override { return prefill_; }
    const Graph& verify_graph() const override { return verify_; }
    const Graph& draft_graph() const override { return draft_; }
    const ChainDrafting* chain() const override { return block_ ? nullptr : this; }
    const BlockDrafting* block() const override { return block_ ? this : nullptr; }
    std::string_view drafter_name() const override { return block_ ? "tiny-block" : "tiny"; }
    int64_t block_drafts() const override { return kBlockDrafts; }
    void prepare_block(int32_t token, int64_t position, TensorMap& inputs) const override {
        std::vector<int32_t> ids{token};
        for (int64_t j = 0; j < kBlockDrafts; ++j) {
            const int32_t t = ids.back();
            ids.push_back(static_cast<int32_t>((5 * t + 3 + (t % 3 == 0 ? 1 : 0)) % kVocab));
        }
        prepare_inputs(ids, position, {}, inputs);
    }
    bool drafter_keeps_state() const override { return true; }
    void prepare_draft(std::span<const int32_t> tokens, std::span<const float> hidden, int64_t start_pos, int depth,
                       TensorMap& inputs) const override {
        prepare_inputs(tokens, start_pos + depth, {}, inputs);
        inputs["hidden"] = HostTensor::from_vector(DType::F32, {static_cast<int64_t>(tokens.size()), kHidden},
                                                   std::vector<float>(hidden.begin(), hidden.end()));
    }

private:
    Value freq(Graph& g) const { return g.constant(test::make_tensor(DType::F32, {kDim / 2}, std::vector<float>{1.0f, 0.1f, 0.01f, 0.001f})); }

    // Attention over its own KV slot (`label`), added to h at `mix`.
    Value attention(Graph& g, const std::string& p, const std::string& label, Value h, Value pos, float mix) const {
        const Value x = g.rms_norm(h, 1e-6f);
        const auto w = [&](const char* n) { return g.weight(weights_, p + n); };
        const Value q = g.rope(g.reshape(g.matmul(x, w("q")), {kT, kHeads, kDim}), pos, freq(g));
        const Value k = g.rope(g.reshape(g.matmul(x, w("k")), {kT, 1, kDim}), pos, freq(g));
        const auto [k_all, v_all] =
            g.kv_cache(g.add_kv_slot(1, kDim, 0, label), k, g.reshape(g.matmul(x, w("v")), {kT, 1, kDim}));
        const Value a = g.attention(q, k_all, v_all, {1.0f / std::sqrt(float{kDim})});
        return g.add(h, g.scale(g.matmul(g.reshape(a, {kT, kHeads * kDim}), w("o")), mix));
    }

    void build_target(Graph& g, bool all_logits, bool hidden_out) const {
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        Value h = g.gather(g.weight(weights_, "embed"), ids);
        const Value x = g.rms_norm(h, 1e-6f);
        const int64_t kd = kHeads * kDim;
        const Value qkv = g.silu(g.causal_conv(g.add_conv_slot(3 * kd, kKernel, "conv"),
                                                g.matmul(x, g.weight(weights_, "in")), g.weight(weights_, "conv")));
        const auto part = [&](int64_t i) { return g.reshape(g.slice(qkv, 1, i * kd, (i + 1) * kd), {kT, kHeads, kDim}); };
        const Value decay = g.scale(g.sigmoid(g.matmul(x, g.weight(weights_, "decay"))), -1.0f);
        const Value beta = g.sigmoid(g.matmul(x, g.weight(weights_, "beta")));
        const Value o = g.gated_delta(g.add_recurrent_slot(kHeads, kDim, kDim, "delta"), part(0), part(1), part(2),
                                      decay, beta, 1e-6f);
        h = g.add(h, g.scale(g.matmul(g.reshape(o, {kT, kd}), g.weight(weights_, "out")), kMix));
        h = attention(g, "attn.", "attn", h, pos, kMix);
        const Value hidden = g.rms_norm(h, 1e-6f);
        const Value logits = g.matmul(all_logits ? hidden : g.slice(hidden, 0, -1), g.weight(weights_, "lm_head"));
        g.output("logits", logits);
        if (argmax_) g.output("argmax", blocks::argmax_rows(g, logits));
        if (hidden_out && !block_) g.output("hidden", hidden);
        if (hidden_out && block_) {  // the block drafter's context: a key and a value per token, in its own slot
            const auto row = [&](const char* w) {
                return g.reshape(g.matmul(hidden, g.weight(weights_, std::string("draft.") + w)), {kT, 1, kDim});
            };
            g.kv_cache(g.add_kv_slot(1, kDim, 0, "draft"), g.rope(row("k"), pos, freq(g)), row("v"));
        }
    }

    void build_block() {
        Graph& g = draft_;
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value h = attention(g, "draft.", "draft", g.gather(g.weight(weights_, "embed"), ids), pos, kDraftMix);
        const Value rows = g.rms_norm(g.slice(h, 0, -kBlockDrafts), 1e-6f);
        g.output("drafts", blocks::argmax_rows(g, g.matmul(rows, g.weight(weights_, "embed"))));
    }

    void build_draft() {
        Graph& g = draft_;
        const Value ids = g.input("token_ids", DType::I32, {kT});
        const Value pos = g.input("positions", DType::I32, {kT});
        const Value hidden = g.input("hidden", DType::F32, {kT, kHidden});
        Value h = g.add(g.gather(g.weight(weights_, "embed"), ids),
                        g.scale(g.matmul(hidden, g.weight(weights_, "draft.hidden")), kDraftMix));
        h = attention(g, "draft.", "draft", h, pos, kDraftMix);
        const Value last = g.rms_norm(g.slice(h, 0, -1), 1e-6f);
        g.output("logits", g.matmul(last, g.weight(weights_, "draft.head")));
        g.output("hidden", last);
    }

    bool argmax_ = false;
    bool block_ = false;
    test::MemoryWeights weights_;
    ModelInfo info_;
    Graph step_, prefill_, verify_, draft_;
};

std::unique_ptr<Engine> make_engine(int spec_k, int64_t prefill_chunk, bool argmax = false, bool block = false,
                                    int64_t sequences = 1) {
    static const auto backend = make_cpu_ref_backend();
    auto model = std::make_unique<TinySpecModel>(argmax, block);
    std::vector<const Graph*> programs{&model->graph()};
    SessionOptions options;
    options.precision = DType::F32;
    options.max_context = 256;
    options.sequences = sequences;
    if (spec_k > 0) {
        programs.insert(programs.end(), {&model->prefill_graph(), &model->verify_graph(), &model->draft_graph()});
        options.rewind_tokens = spec_k + 1;
    }
    auto session = backend->compile(programs, model->weights(), options);
    EngineOptions engine;
    engine.max_context = options.max_context;
    engine.prefill_chunk = prefill_chunk;
    engine.spec_k = spec_k;
    engine.sequences = sequences;
    return std::make_unique<Engine>(std::move(model), std::move(session), EncoderSessions{}, engine);
}

GenerateRequest greedy(std::vector<int32_t> prompt, int max_new, std::vector<int32_t> stops = {}) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    r.sampling.temperature = 0.0f;
    r.max_new_tokens = max_new;
    r.stop_token_ids = std::move(stops);
    return r;
}

}  // namespace

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
