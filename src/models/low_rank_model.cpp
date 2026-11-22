#include "models/low_rank_model.h"

#include <set>

#include "io/joined_weights.h"
#include "util/error.h"

namespace xs {
namespace {

// Where the changes' tensors are among the model's weights.
constexpr const char* kPrefix = "low_rank.";

class LowRankModel final : public Model, public Speculation {
public:
    LowRankModel(std::unique_ptr<Model> model, std::unique_ptr<WeightStore> tensors, const graph::LowRanks& changes)
        : model_(std::move(model)), weights_(model_->weights(), std::move(tensors), kPrefix) {
        graph::LowRanks named = changes;
        for (auto& [weight, c] : named) {
            c.a = kPrefix + c.a;
            c.b = kPrefix + c.b;
        }
        for (const std::string& table : model_->host_tables()) {
            XS_CHECK(named.count(table) == 0, "adapter: ", table, " is read on the host, where no change reaches it");
        }
        std::set<std::string> used;
        step_ = graph::with_low_rank(model_->graph(), weights_, named, &used);
        if (const Speculation* s = model_->speculation()) {
            prefill_ = graph::with_low_rank(s->prefill_graph(), weights_, named, &used);
            verify_ = graph::with_low_rank(s->verify_graph(), weights_, named, &used);
            draft_ = graph::with_low_rank(s->draft_graph(), weights_, named, &used);
        }
        for (const auto& [weight, c] : named) {
            XS_CHECK(used.count(weight) != 0, "adapter: it changes ", weight, ", which no program of ",
                     model_->info().id, " reads");
        }
    }

    const ModelInfo& info() const override { return model_->info(); }
    const graph::Graph& graph() const override { return step_; }
    const WeightStore& weights() const override { return weights_; }
    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> embeddings,
                        TensorMap& inputs) const override {
        model_->prepare_inputs(tokens, start_pos, embeddings, inputs);
    }
    void prepare_positioned_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                                   std::span<const float* const> embeddings, TensorMap& inputs) const override {
        model_->prepare_positioned_inputs(tokens, positions, embeddings, inputs);
    }
    const VisionEncoder* vision() const override { return model_->vision(); }
    const AudioEncoder* audio() const override { return model_->audio(); }
    const Transcription* transcription() const override { return model_->transcription(); }
    std::vector<std::string> host_tables() const override { return model_->host_tables(); }
    const Speculation* speculation() const override { return model_->speculation() ? this : nullptr; }

    const graph::Graph& prefill_graph() const override { return prefill_; }
    const graph::Graph& verify_graph() const override { return verify_; }
    const graph::Graph& draft_graph() const override { return draft_; }
    const ChainDrafting* chain() const override { return model_->speculation()->chain(); }
    const BlockDrafting* block() const override { return model_->speculation()->block(); }
    std::string_view drafter_name() const override { return model_->speculation()->drafter_name(); }
    std::string_view drafter_weights() const override { return model_->speculation()->drafter_weights(); }

private:
    std::unique_ptr<Model> model_;
    JoinedWeights weights_;
    graph::Graph step_, prefill_, verify_, draft_;
};

}  // namespace

std::unique_ptr<Model> with_low_rank(std::unique_ptr<Model> model, std::unique_ptr<WeightStore> tensors,
                                     const graph::LowRanks& changes) {
    return std::make_unique<LowRankModel>(std::move(model), std::move(tensors), changes);
}

}  // namespace xs
