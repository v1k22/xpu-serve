#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <span>
#include <vector>

#include "backend/backend.h"
#include "engine/types.h"
#include "models/model.h"

namespace xs {

struct EngineOptions {
    int64_t max_context = 8192;  // tokens per sequence (prompt + generated)
    int64_t prefill_chunk = 1024;
    size_t encoded_media_cache = 16;  // images whose encoder output stays cached (LRU)
};

// Runs generations for one sequence at a time on one backend session. Not thread-safe: one caller (the engine
// thread) owns it. Consecutive requests that share a prompt prefix reuse the cached KV; an image's soft tokens count
// as shared only when the image's pixels are the same.
class Engine {
public:
    // `vision_session` runs the model's vision encoder graph; null when the model has none (or it is not loaded).
    Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, std::unique_ptr<Session> vision_session,
           EngineOptions options);

    GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token);

    const Model& model() const { return *model_; }
    const EngineOptions& options() const { return options_; }
    void set_prefill_chunk(int64_t tokens);
    bool accepts_images() const { return vision_session_ != nullptr; }

private:
    // What a cached position holds: the token id, or for an image's soft token the image's hash and the row.
    using PositionKey = uint64_t;

    // Feeds `tokens` (appended after the cached ones) and returns the logits of the last one.
    std::vector<float> forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                               std::span<const float* const> embeddings);
    // Makes the session hold a prefix of the prompt and returns how many prompt positions are already cached.
    int64_t reuse_prefix(const std::vector<PositionKey>& prompt_keys);
    // The encoder output of one image ([rows, hidden] f32), from the cache or a vision run.
    std::shared_ptr<const HostTensor> encode(const MediaInput& media, GenerateStats& stats);

    std::unique_ptr<Model> model_;
    std::unique_ptr<Session> session_;
    std::unique_ptr<Session> vision_session_;
    EngineOptions options_;
    std::vector<PositionKey> cached_;  // positions whose KV the session holds, in order
    std::list<std::pair<uint64_t, std::shared_ptr<const HostTensor>>> encoded_;  // most recent first
    TensorMap inputs_;
    TensorMap outputs_;
};

}  // namespace xs
