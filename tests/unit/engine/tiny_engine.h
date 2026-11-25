#pragma once

// A tiny model with every kind of state (conv, gated delta, KV) for the engine's tests, with a drafter of its own
// (a chain or a block drafter) and a batch graph, and an engine over it on cpu-ref.

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

namespace xs::test {

using graph::Graph;
using graph::kT;
using graph::Value;

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
    // `batch`: a batch graph too (Model::batch_graph), and a verify graph that allows segments.
    explicit TinySpecModel(bool argmax = false, bool block = false, bool batch = false)
        : argmax_(argmax), block_(block), batch_(batch) {
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
        if (batch_) {
            batch_graph_.allow_segments();
            build_target(batch_graph_, true, false);
            batch_graph_.validate();
        }
        build_target(prefill_, false, true);
        if (batch_) verify_.allow_segments();  // speculating, the rounds of several requests verify as one step
        build_target(verify_, true, true);
        verify_.validate();
        block_ ? build_block() : build_draft();
    }

    const ModelInfo& info() const override { return info_; }
    const Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return weights_; }
    const Speculation* speculation() const override { return this; }
    const Graph* batch_graph() const override { return batch_ ? &batch_graph_ : nullptr; }
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
    bool batch_ = false;
    test::MemoryWeights weights_;
    ModelInfo info_;
    Graph step_, batch_graph_, prefill_, verify_, draft_;
};

// `batch`: the decode steps of the sequences in flight run as one (EngineOptions::batch): speculating, their
// verifies, of up to `verify_rows` rows (EngineOptions::verify_rows).
inline std::unique_ptr<Engine> make_engine(int spec_k, int64_t prefill_chunk, bool argmax = false, bool block = false,
                                           int64_t sequences = 1, bool batch = false, int64_t verify_rows = 64) {
    static const auto backend = make_cpu_ref_backend();
    auto model = std::make_unique<TinySpecModel>(argmax, block, batch);
    std::vector<const Graph*> programs{&model->graph()};
    if (batch && spec_k == 0) programs.push_back(model->batch_graph());
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
    engine.batch = batch;
    engine.verify_rows = verify_rows;
    return std::make_unique<Engine>(std::move(model), std::move(session), EncoderSessions{}, engine);
}

inline GenerateRequest greedy(std::vector<int32_t> prompt, int max_new, std::vector<int32_t> stops = {}) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    r.sampling.temperature = 0.0f;
    r.max_new_tokens = max_new;
    r.stop_token_ids = std::move(stops);
    return r;
}

}  // namespace xs::test
