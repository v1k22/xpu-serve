#pragma once

// A language model read as a decision model: the options of a question get a letter each, the question goes through
// the model's chat template as a user message, and the scores of those letters at the first output position are the
// answer. One prompt per question, no generation. This is how OpenJev (openjev/openjev, a Qwen3.8-27B tuned to carry
// the decision at that position) is served; reference: the checkpoint's helper/shim.py, whose prompt, letters,
// calibration and many-option composition are reproduced here.
//
// Questions of a request share the state at the start of their prompts: the first prompt is read whole, each later
// one from the end of that shared part (the tail read before is undone, Session::truncate), and a request about the
// state of the request before it starts there too.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "models/decision.h"
#include "models/model.h"

namespace xs::letter_readout {

inline constexpr size_t kLetters = 52;  // A to Z, then a to z: the options one prompt can hold

struct Settings {
    double temperature = 1.0;       // a choice's or a score's letter scores are divided by it before the softmax
    double noul_temperature = 1.0;  // P(true) = sigmoid(logit(p of the yes option) / noul_temperature + noul_bias)
    double noul_bias = 0.0;
    bool python_instructions = false;  // instructions given as JSON are written as Python's str() writes them
};

// The settings a model's published numbers were measured with, by name ("openjev"); nullopt for an unknown name.
std::optional<Settings> preset(std::string_view name);
std::vector<std::string> preset_names();

// One read-out's prompt, for a state and a question as text: the user message the chat template wraps.
std::string user_message(const std::string& state, const std::string& instructions,
                         const std::vector<std::pair<std::string, std::string>>& options);

std::unique_ptr<DecisionModel> make(std::unique_ptr<Model> model, const Settings& settings);

}  // namespace xs::letter_readout
