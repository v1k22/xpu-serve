#pragma once

// d1 (LiquidAI/d1-2B): the text decoder of an LFM3-VL checkpoint read as a decision model. Reference: the checkpoint's
// own code (prompt.py: the prompt, the option codes and the read-out; runner.py and attention.py: a request's
// sequences and what their full-attention layers see; calibration.py: the temperature).
//
// A question is a ChatML user turn, the state then the question, closed by the assistant header and the empty thought
// the template writes ("<think></think>"); the answer is read at that last token: each option's best logit among its
// tokens (a choice's code, with and without a space before it; a noul's yes and no forms; a score's digit), divided
// by the calibrated temperature, through a softmax. The program gives the final hidden state there; the host takes
// the logits from the rows of the output layer (the embedding table) those tokens need.
//
// Up to d1_set_size questions are packed after one state, each opened by "QUESTION:" again and closed by its own
// answer slot ("set" attention). On the full-attention layers the state sees every token and each question sees the
// state and itself (attention in groups: the state is group 0, the k-th question group k); the convolution and
// sliding-window layers stay causal over the packed sequence. A lone question reads its whole prompt both ways.
// Images, video and spoken questions are not read (the vision tower is not loaded).

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "models/decision.h"

namespace xs::d1 {

struct Config {
    int code_scheme = 1;   // d1_codes: the codes of more than 26 options, 00, 01, ... (1) or AA, AB, ... (2)
    int64_t set_size = 4;  // d1_set_size: the most questions packed after one state, as the checkpoint was trained
    std::string bos;       // what a prompt starts with: tokenizer_config.json's bos_token
    // calibration.json: a question bucket's temperature (temperatures), else `temperature` (default; 1 without the
    // file).
    std::map<std::string, double> temperatures;
    double temperature = 1.0;
};

// `config`: config.json; `calibration`: calibration.json, or null when the checkpoint has none; `tokenizer`:
// tokenizer_config.json. Throws xs::Error naming an attention mode, code scheme or d1_ key this code does not
// implement.
Config parse_config(const nlohmann::json& config, const nlohmann::json& calibration, const nlohmann::json& tokenizer);

// The temperature of a question of `options` options: its bucket's (noul, score:<levels>, or choice:small up to 3
// options, choice:mid up to 8, choice:large up to 26, choice:inventory), else the default.
double temperature(const Config& config, QuestionType type, size_t options);

// The prompt's pieces, as prompt.py writes them.
// A text with U+200B after the first character of every special token in it, so that it encodes to none of them.
std::string plain(std::string_view text, const std::vector<std::string>& special_tokens);
// The codes of a choice's options by their names: the names when every one is a single letter, A to Z for up to 26,
// past that the scheme's.
std::vector<std::string> option_codes(const std::vector<std::string>& names, int scheme);

// One forward pass: the state, then questions packed after it.
struct Sequence {
    std::vector<int32_t> ids;
    int64_t state = 0;              // the state's tokens: those before the first question's own
    std::vector<size_t> questions;  // indices into the request's questions, in the sequence's order
    std::vector<int64_t> answers;   // each one's answer position: its last token
    // Each token's group: 0 for the state, k + 1 for the tokens of the sequence's k-th question.
    std::vector<int32_t> groups() const;
};

// A request laid out as the model reads it.
struct Plan {
    std::vector<Sequence> sequences;
    // Each question's tokens per option, in DecisionQuestion::options order: the option's score is the best of their
    // logits.
    std::vector<std::vector<std::vector<int32_t>>> option_tokens;
};
// Sequences of up to `set_size` questions. Throws DecisionRefused for a question the model does not take, or a
// sequence longer than max_tokens.
Plan plan(const DecisionRequest& request, const PromptTools& tools, const Config& config, int64_t set_size,
          int64_t max_tokens);

bool is_checkpoint(const std::filesystem::path& dir);  // config.json's architectures: D1Model
// `options.set_size`: questions packed after one state (1 reads each alone); unset: d1_set_size.
std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                                    const DecisionLoadOptions& options = {});

}  // namespace xs::d1
