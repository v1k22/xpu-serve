// Rotary positions in the engine (MediaInput::positions, Model::prepare_positioned_inputs), on cpu-ref: text takes
// one position a token; an image's tokens take their offsets from the image's first position; the token after it
// resumes one past the image's largest; generated tokens continue from there (transformers' get_rope_index and its
// rope deltas); a prompt that extends the cached one feeds only its new tokens, at the same positions; a model with one
// position a token refuses media that bring positions of their own.
#include <doctest/doctest.h>

#include <memory>
#include <vector>

#include "../backend/backend_test_util.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "engine/engine.h"
#include "models/common/blocks.h"
#include "util/error.h"

namespace xs {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr int64_t kVocab = 12, kHidden = 8;
constexpr int32_t kImageToken = 11;

// Encoder rows are its input, as they are.
class PassThroughVision final : public VisionEncoder {
public:
    PassThroughVision() {
        g_.output("embeddings", g_.input("rows", DType::F32, {kT, kHidden}));
        g_.validate();
    }
    std::vector<const Graph*> programs() const override { return {&g_}; }
    int32_t placeholder_token() const override { return kImageToken; }
    int32_t soft_token() const override { return kImageToken; }
    std::vector<int32_t> token_run(int64_t n) const override {
        return std::vector<int32_t>(static_cast<size_t>(n), kImageToken);
    }
    MediaInput prepare(const image::Image&) const override { return {}; }
    MediaInput warm_up_item() const override { return {}; }

private:
    Graph g_;
};

// An embedding, a KV slot (the engine counts cached tokens through it) and a head; it records every position it is
// handed. `three_axes`: it reads positions on three axes; else one a token, through Model's default.
class PositionModel final : public Model {
public:
    explicit PositionModel(bool three_axes) : three_axes_(three_axes) {
        test::TensorRng rng(3);
        weights_.add("embed", rng.uniform(DType::F32, {kVocab, kHidden}));
        weights_.add("head", rng.uniform(DType::F32, {kVocab, kHidden}));
        info_.arch = "positions";
        info_.id = "positions";
        info_.vocab_size = kVocab;
        info_.hidden_size = kHidden;
        info_.max_position = 4096;
        const Value ids = g_.input("token_ids", DType::I32, {kT});
        Value h = blocks::splice_media(g_, g_.gather(g_.weight(weights_, "embed"), ids), kHidden);
        const Value kv = g_.reshape(h, {kT, 1, kHidden});
        const auto [k_all, v_all] = g_.kv_cache(g_.add_kv_slot(1, kHidden, 0, "kv"), kv, kv);
        graph::AttentionAttrs attrs;
        attrs.scale = 0.1f;
        h = g_.add(h, g_.reshape(g_.attention(g_.reshape(h, {kT, 1, kHidden}), k_all, v_all, attrs), {kT, kHidden}));
        g_.output("logits", g_.matmul(g_.slice(h, 0, -1), g_.weight(weights_, "head")));
        g_.validate();
    }

    const ModelInfo& info() const override { return info_; }
    const Graph& graph() const override { return g_; }
    const WeightStore& weights() const override { return weights_; }
    const VisionEncoder* vision() const override { return &vision_; }
    void prepare_inputs(std::span<const int32_t> tokens, int64_t start_pos, std::span<const float* const> rows,
                        TensorMap& inputs) const override {
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto p = static_cast<int32_t>(start_pos + static_cast<int64_t>(i));
            seen.push_back({p, p, p});
        }
        fill(tokens, rows, inputs);
    }
    void prepare_positioned_inputs(std::span<const int32_t> tokens, std::span<const RotaryPosition> positions,
                                   std::span<const float* const> rows, TensorMap& inputs) const override {
        if (!three_axes_) return Model::prepare_positioned_inputs(tokens, positions, rows, inputs);
        seen.insert(seen.end(), positions.begin(), positions.end());
        fill(tokens, rows, inputs);
    }

    mutable std::vector<RotaryPosition> seen;  // every position fed, in order

private:
    void fill(std::span<const int32_t> tokens, std::span<const float* const> rows, TensorMap& inputs) const {
        const auto n = static_cast<int64_t>(tokens.size());
        inputs["token_ids"] =
            HostTensor::from_vector(DType::I32, {n}, std::vector<int32_t>(tokens.begin(), tokens.end()));
        blocks::media_inputs(rows, n, kHidden, inputs);
    }

    bool three_axes_;
    test::MemoryWeights weights_;
    ModelInfo info_;
    PassThroughVision vision_;
    Graph g_;
};

std::unique_ptr<Engine> engine_for(std::unique_ptr<PositionModel> model, PositionModel*& handle) {
    handle = model.get();
    const auto backend = make_cpu_ref_backend();
    SessionOptions options;
    options.max_context = 256;
    auto session = backend->compile(model->graph(), model->weights(), options);
    EncoderSessions encoders;
    encoders[static_cast<size_t>(Modality::Image)] =
        backend->compile(model->vision()->programs(), model->weights(), options);
    EngineOptions eo;
    eo.max_context = 256;
    eo.prefill_chunk = 4;  // the image straddles chunks
    return std::make_unique<Engine>(std::move(model), std::move(session), std::move(encoders), eo);
}

// A 2 x 3 image: its tokens at (0, row, column).
MediaInput image(bool with_positions) {
    MediaInput m;
    m.modality = Modality::Image;
    m.key = 42;
    m.num_tokens = 6;
    EncoderRun& run = m.runs.emplace_back();
    run.rows = 6;
    run.inputs["rows"] = test::TensorRng(9).uniform(DType::F32, {6, kHidden});
    if (with_positions) {
        for (int32_t r = 0; r < 2; ++r) {
            for (int32_t c = 0; c < 3; ++c) m.positions.push_back({0, r, c});
        }
    }
    return m;
}

GenerateRequest request(std::vector<int32_t> prompt, int max_new, bool with_positions = true) {
    GenerateRequest r;
    r.prompt = std::move(prompt);
    r.media.push_back(image(with_positions));
    r.sampling.temperature = 0.0f;
    r.max_new_tokens = max_new;
    return r;
}

std::vector<int32_t> prompt_with_image(std::vector<int32_t> after) {
    std::vector<int32_t> p{1, 2};
    p.insert(p.end(), 6, kImageToken);
    p.insert(p.end(), after.begin(), after.end());
    return p;
}

}  // namespace

TEST_SUITE("engine rotary positions") {
    TEST_CASE("text, an image at its offsets, text one past the image's largest, then generated tokens") {
        PositionModel* model = nullptr;
        auto engine = engine_for(std::make_unique<PositionModel>(true), model);
        const GenerateResult r = engine->generate(request(prompt_with_image({3, 4}), 3), {});
        REQUIRE_MESSAGE(r.finish != FinishReason::Error, r.error);
        const std::vector<RotaryPosition> expected = {
            {0, 0, 0}, {1, 1, 1},                                     // text
            {2, 2, 2}, {2, 2, 3}, {2, 2, 4}, {2, 3, 2}, {2, 3, 3}, {2, 3, 4},  // the image from 2: (0, row, column)
            {5, 5, 5}, {6, 6, 6},                                     // 2 + max(0, 1, 2) + 1
            {7, 7, 7}, {8, 8, 8}};                                    // generated: two fed after the first
        CHECK(model->seen == expected);

        SUBCASE("a prompt that extends the cached one feeds only its new tokens, at the same positions") {
            model->seen.clear();
            std::vector<int32_t> longer = prompt_with_image({3, 4});
            longer.insert(longer.end(), r.tokens.begin(), r.tokens.begin() + 2);
            longer.push_back(5);
            const GenerateResult again = engine->generate(request(longer, 1), {});
            REQUIRE_MESSAGE(again.finish != FinishReason::Error, again.error);
            CHECK(again.stats.cached_tokens == 12);
            CHECK(model->seen == std::vector<RotaryPosition>{{9, 9, 9}});
        }
    }

    TEST_CASE("an image without positions takes one a token, as text") {
        PositionModel* model = nullptr;
        auto engine = engine_for(std::make_unique<PositionModel>(true), model);
        const GenerateResult r = engine->generate(request(prompt_with_image({3}), 1, /*with_positions=*/false), {});
        REQUIRE_MESSAGE(r.finish != FinishReason::Error, r.error);
        REQUIRE(model->seen.size() == 9);
        for (int32_t i = 0; i < 9; ++i) CHECK(model->seen[static_cast<size_t>(i)] == RotaryPosition{i, i, i});
    }

    TEST_CASE("a model with one position a token refuses media that bring positions; without them it runs") {
        PositionModel* model = nullptr;
        auto engine = engine_for(std::make_unique<PositionModel>(false), model);
        const GenerateResult refused = engine->generate(request(prompt_with_image({3}), 1), {});
        CHECK(refused.finish == FinishReason::Error);
        CHECK(refused.error.find("one position a token") != std::string::npos);
        const GenerateResult ok = engine->generate(request(prompt_with_image({3}), 1, /*with_positions=*/false), {});
        CHECK(ok.finish != FinishReason::Error);
    }
}

}  // namespace xs
