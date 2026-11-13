#pragma once

// Laya (convaiinnovations/laya: its root model and the typed-decisions and multilingual folders) and Julia-1
// (SupersonicLabs/Julia-1): one decision model on a ModernBERT encoder, read at a marker per option. References:
// Laya's rl_common.py (build_sequence, DecisionModel) and rl_agent_api.py (system_one); Julia's julia/data.py
// (sequence), julia/typed.py (predict_typed), julia/model.py and inference-policy.json.
//
// Each question is a sequence of its own: [CLS] "<type> question: <instructions>" [SEP], then per option the mask
// token and the option's text (at most 48 tokens), then [SEP] state [SEP], each piece tokenized alone. The encoder
// reads it whole (models/modern_bert); the question type's embedding is added to every row; two
// nn.TransformerEncoderLayer (norm first, ReLU, no positions) run over the whole sequence; a scorer (LayerNorm,
// Linear, GELU, Linear) gives each mask row a logit, and a softmax over the question's options its probabilities.
//
// The two differ as data (Policy): Laya writes an option as "name: description", "level i: text" or "false: ..."
// and cuts what does not fit (an option's text, the question, the state) as its reference does, then divides the
// logits by a temperature fitted per type and option count; Julia-1 writes the description alone, takes 2 to 20
// options and string instructions, and refuses what does not fit or spells its mask token (strict_encoding), with
// a plain softmax.

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "models/decision.h"
#include "models/modern_bert/modern_bert.h"

namespace xs::laya {

inline constexpr int64_t kOptionTokens = 48;  // an option's text, at most

struct Policy {
    std::string family;              // "laya" or "julia"
    int64_t max_tokens = 0;          // a sequence's tokens at most (Laya's max_len, Julia's max_length)
    int64_t head_tokens = 0;         // the question's and options' budget (head_max_len, head_length)
    bool descriptions_only = false;  // Julia: an option is its description; Laya: "name: description" and so on
    bool strict = false;             // Julia's strict_encoding: refuse what Laya would cut
    // Julia: a question and options that leave the state no token are refused; Laya cuts the whole sequence to
    // max_tokens instead, and refuses only options past it.
    bool room_for_state = false;
    // Laya: the logits of a question of type t (choice 0, score 1, noul 2) are divided by temperature[t], or by
    // the entry of temperature_by_options for its type and option count ("choice:3-5"). Julia: 1.
    std::vector<double> temperature{1.0, 1.0, 1.0};
    std::map<std::string, double> temperature_by_options;
};

struct Config {
    modern_bert::Config encoder;
    int64_t head_layers = 0;
    Policy policy;
    // tokenizer_config.json's cls_token, sep_token and mask_token: the sequence's pieces besides text.
    std::string cls, sep, mask;
};
// The checkpoint in `dir`: Laya's rl_agent_config.json, or Julia's config.json, julia_config.json and
// inference-policy.json; encoder/config.json; tokenizer/tokenizer_config.json. Refuses by name what it does not
// implement.
Config read_config(const std::filesystem::path& dir);

// One question as the model reads it.
struct Encoded {
    std::vector<int32_t> ids;
    std::vector<int64_t> markers;  // the rows of the question's options, in the prompt's order (a noul: false, true)
    int64_t state_tokens = 0;      // the state's tokens
    int64_t state_kept = 0;        // those the sequence holds (fewer: Laya cut it to fit)
};
// The sequence of `question` of `request`, at most `max_tokens` long. Throws DecisionRefused for what the policy
// refuses.
Encoded encode_question(const DecisionRequest& request, const DecisionQuestion& question, const PromptTools& tools,
                        const Config& config, int64_t max_tokens);

// Laya's temperature bucket of a question with `options` options: "<type>:<2|3-5|6-10|11+>".
std::string temperature_bucket(QuestionType type, size_t options);

// A Laya or Julia-1 checkpoint: encoder/config.json beside rl_agent_config.json or julia_config.json (as llama.cpp's
// converter tells them).
bool is_checkpoint(const std::filesystem::path& dir);
std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir);

}  // namespace xs::laya
