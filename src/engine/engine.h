#pragma once

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
};

// Runs generations for one sequence at a time on one backend session. Not thread-safe: one caller (the engine
// thread) owns it. Consecutive requests that share a prompt prefix reuse the cached KV.
class Engine {
public:
    Engine(std::unique_ptr<Model> model, std::unique_ptr<Session> session, EngineOptions options);

    GenerateResult generate(const GenerateRequest& request, const TokenCallback& on_token);

    const Model& model() const { return *model_; }
    const EngineOptions& options() const { return options_; }

private:
    // Feeds `tokens` (appended after the cached ones) and returns the logits of the last one.
    std::vector<float> forward(std::span<const int32_t> tokens);
    // Makes the session hold a prefix of `prompt` and returns how many prompt tokens are already cached.
    int64_t reuse_prefix(const std::vector<int32_t>& prompt);

    std::unique_ptr<Model> model_;
    std::unique_ptr<Session> session_;
    EngineOptions options_;
    std::vector<int32_t> cached_;  // tokens whose KV the session holds, in order
    TensorMap inputs_;
    TensorMap outputs_;
};

}  // namespace xs
