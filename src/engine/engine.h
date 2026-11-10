#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <span>
#include <vector>

#include "backend/backend.h"
#include "engine/spec/draft_runner.h"
#include "engine/types.h"
#include "models/model.h"

namespace xs {

class Sampler;

struct EngineOptions {
    int64_t max_context = 8192;  // tokens per sequence (prompt + generated)
    int64_t prefill_chunk = 1024;
    size_t encoded_media_cache = 16;  // images whose encoder output stays cached (LRU)
    // Speculative decoding with the model's drafter: drafts per round, 0 = off. The session must hold the programs
    // of spec::Program (the model's Speculation) and rewind steps of spec_k + 1 tokens.
    int spec_k = 0;
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

    // Runs a target program over `tokens` (appended after the cached ones); the outputs the round reads are in
    // outputs_: each row's logits, or with argmax_only_ its argmax alone, and the hidden rows a drafter takes.
    void run_target(size_t program, std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                    std::span<const float* const> embeddings);
    // Feeds `tokens` (appended after the cached ones); target_token(sampler, 0, ...) is then the token after them.
    void forward(std::span<const int32_t> tokens, std::span<const PositionKey> keys,
                 std::span<const float* const> embeddings);
    // The target's token after row `row` of the last run: sampled from the row's logits (checked against `draft`
    // when there is one, Sampler::verify), or the row's argmax when that is all the sampler reads (argmax_only_).
    int32_t target_token(Sampler& sampler, int64_t row, std::span<const int32_t> prompt,
                         std::span<const int32_t> generated, const int32_t* draft);
    // One speculative round after `token` (emitted, not yet cached): `k` drafts, one verify of them all, acceptance
    // and rollback. Returns the accepted drafts and the target's next token.
    std::vector<int32_t> spec_round(int32_t token, int k, Sampler& sampler, std::span<const int32_t> prompt,
                                    std::span<const int32_t> generated, GenerateStats& stats);
    // Drops the sequence: every slot of the session, and what waits for the drafter.
    void reset_sequence();
    // Makes the session hold a prefix of the prompt and returns how many prompt positions are already cached.
    int64_t reuse_prefix(const std::vector<PositionKey>& prompt_keys);
    // The encoder output of one image ([rows, hidden] f32), from the cache or a vision run.
    std::shared_ptr<const HostTensor> encode(const MediaInput& media, GenerateStats& stats);

    std::unique_ptr<Model> model_;
    std::unique_ptr<Session> session_;
    std::unique_ptr<Session> vision_session_;
    EngineOptions options_;
    std::unique_ptr<spec::DraftRunner> drafter_;  // speculating
    std::vector<PositionKey> cached_;  // positions whose KV the session holds, in order
    std::list<std::pair<uint64_t, std::shared_ptr<const HostTensor>>> encoded_;  // most recent first
    TensorMap inputs_;
    TensorMap outputs_;
    bool has_argmax_ = false;   // every target program outputs "argmax" beside "logits"
    bool argmax_only_ = false;  // this request's sampler reads no more than a row's argmax, and the programs give it
};

}  // namespace xs
