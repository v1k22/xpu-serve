// Speculative decoding in the engine, on cpu-ref: greedy output with a drafter equals greedy output without one, over
// a model with every kind of state (conv, gated delta, KV) and a drafter with its own KV that is right only part of
// the time, through prefix reuse, chunked prefill, stop tokens and rounds cut short.
#include <doctest/doctest.h>

#include <cmath>
#include <memory>
#include <numeric>

#include "../backend/backend_test_util.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "engine/engine.h"

namespace xs {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// The next token is mostly f(t) = (5 t + 3) mod V: lm_head's row f(t) is t's embedding. A conv + gated delta layer
// and an attention layer perturb the hidden state enough to change it sometimes. The drafter predicts f of its token
// (but f + 1 after every third token) from the embedding, perturbed a little by the target's hidden and by attention
// over its own KV.
class TinySpecModel final : public Model, public Speculation {
public:
    static constexpr int64_t kVocab = 16, kHidden = 24, kHeads = 2, kDim = 8, kKernel = 4;
    static constexpr float kMix = 4.0f;        // how much the target's layers move the hidden state
    static constexpr float kDraftMix = 0.15f;  // how much the drafter's

    TinySpecModel() {
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
        build_draft();
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
        g.output("logits", g.matmul(all_logits ? hidden : g.slice(hidden, 0, -1), g.weight(weights_, "lm_head")));
        if (hidden_out) g.output("hidden", hidden);
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

    test::MemoryWeights weights_;
    ModelInfo info_;
    Graph step_, prefill_, verify_, draft_;
};

std::unique_ptr<Engine> make_engine(int spec_k, int64_t prefill_chunk) {
    static const auto backend = make_cpu_ref_backend();
    auto model = std::make_unique<TinySpecModel>();
    std::vector<const Graph*> programs{&model->graph()};
    SessionOptions options;
    options.precision = DType::F32;
    options.max_context = 256;
    if (spec_k > 0) {
        programs.insert(programs.end(), {&model->prefill_graph(), &model->verify_graph(), &model->draft_graph()});
        options.rewind_tokens = spec_k + 1;
    }
    auto session = backend->compile(programs, model->weights(), options);
    EngineOptions engine;
    engine.max_context = options.max_context;
    engine.prefill_chunk = prefill_chunk;
    engine.spec_k = spec_k;
    return std::make_unique<Engine>(std::move(model), std::move(session), nullptr, engine);
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

}  // namespace xs
