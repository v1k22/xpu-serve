#pragma once

// Clef (Cloudflare/clef-flash): a Qwen3.5 decoder with a joint schema head. Reference: the checkpoint's
// joint_schema_model.py (encode_record, JointSchemaHead, systemone).
//
// The request is one prompt: the state, then every question with its allowed options. The decoder reads it causally.
// The head takes the decoder's final hidden states: each option's query (the mean state over the option's tokens,
// the mean output-layer row of those tokens, its question's mean state) attends to the whole sequence in evidence
// layers; each question's field (its mean state, a summary of its routed options, the last token's state) goes
// through decoder layers that attend to the other fields and to the sequence; an option's logit is a prior (cosine
// of its output-layer rows with the question) plus a gated score of its routed state against its question's field.
// A softmax per question gives the probabilities. Images and video are not read (the vision tower is not loaded).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/decision.h"

namespace xs::clef {

// joint_head_config.json.
struct HeadConfig {
    int64_t hidden_size = 0;     // the decoder's width
    int64_t width = 0;           // the head's
    int64_t routing_layers = 0;  // evidence layers over the option queries
    int64_t layers = 0;          // decoder layers over the question fields
    int64_t heads = 0;
    int64_t feedforward = 0;
};
HeadConfig parse_head_config(const nlohmann::json& j);

// A request as the model reads it: the prompt's tokens and where each question's instruction and options sit.
struct Span {
    int64_t begin = 0;
    int64_t end = 0;  // exclusive
};
struct EncodedQuestion {
    Span instruction;
    std::vector<Span> options;  // in the prompt's order
    std::vector<int> option;    // the DecisionQuestion::options index of each
};
struct EncodedRecord {
    std::vector<int32_t> ids;
    std::vector<EncodedQuestion> questions;
};
// Throws DecisionRefused when the prompt is longer than max_tokens (nothing is truncated).
EncodedRecord encode_record(const DecisionRequest& request, const PromptTools& tools, int64_t max_tokens);

bool is_checkpoint(const std::filesystem::path& dir);  // a directory with a joint schema head
std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json);

}  // namespace xs::clef
