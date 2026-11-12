#pragma once

// pplx-decider (perplexity-ai/pplx-decider-flash-v1.2): a Qwen3.5 decoder read as a joint decider. Reference: the
// checkpoint's own package (source/src/pplx_decider: protocol.py, model.py), protocol version 3,
// "joint-question-inline-readout-v1".
//
// The request is one prompt through the chat template: the state, then each question with its options under answer
// codes (A, B, ..., AA, ...) and a marker token. The decoder's full-attention layers see the whole prompt, not just
// what came before (its Gated DeltaNet layers stay causal), so a prompt is one step and nothing of it can be cached.
// A linear read-out, applied to the final hidden state at each question's marker, scores the codes; the scores of
// the question's own options, divided by the checkpoint's calibrated temperature, go through a softmax. A request
// longer than the limit is split by question, each part keeping the whole state. Images are not read (the vision
// tower is not loaded).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/decision.h"

namespace xs::pplx_decider {

inline constexpr int64_t kMaxOptions = 255;

// decision_config.json and joint_config.json, and what the prompt needs of tokenizer_config.json.
struct Config {
    double temperature = 1.0;              // fitted on held-out data; the logits are divided by it once
    std::vector<std::string> codes;        // the answer code of option 0, 1, ...
    std::string marker;                    // the token a question's read-out sits at
    std::vector<std::string> neutralized;  // special tokens that user text must not produce, longest first
};
// Throws xs::Error naming a protocol, attention mode or pooling this code does not implement.
Config parse_config(const nlohmann::json& decision, const nlohmann::json& joint, const nlohmann::json& tokenizer);

// One sequence of a request: all its questions, or a part of them when the whole is too long.
struct Encoded {
    std::vector<int32_t> ids;
    std::vector<size_t> questions;  // indices into the request's questions, in the sequence's order
    std::vector<int64_t> markers;   // each one's marker position
};
// The request as sequences of at most max_tokens. Throws DecisionRefused for a question the model does not take, or
// one that does not fit with the state.
std::vector<Encoded> plan(const DecisionRequest& request, const PromptTools& tools, const Config& config,
                          int64_t max_tokens);

bool is_checkpoint(const std::filesystem::path& dir);  // a directory with a joint read-out
std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json);

}  // namespace xs::pplx_decider
