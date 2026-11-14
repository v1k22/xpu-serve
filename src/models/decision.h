#pragma once

// A decision model: a state and typed questions in, a probability for every allowed option of every question out
// (the System One API, POST /v1/systemone). It generates no text: it reads the request as one prompt, or as one per
// question, and a head turns hidden states into a score per option.
//
// The model code lays out the prompt, builds the programs and says which to run with what (DecisionRun). Tokenizing,
// the chat template and the device belong to the runtime (src/decision/), as for a robot policy.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"
#include "util/error.h"

namespace xs {

// The request keeps the client's key order: models write states, instructions and criteria into the prompt as their
// reference code does, and that order is part of the prompt.
using Json = nlohmann::ordered_json;

enum class QuestionType : uint8_t { Noul, Choice, Score };
const char* question_type_name(QuestionType type);  // "noul", "choice", "score"

struct DecisionOption {
    std::string id;  // choice: the option's name; score: "0", "1", ...; noul: "true", "false"
    // A string, an object or an array, or null as the request gave it (a choice may name an option without
    // describing it). nullopt: a noul outcome the request says nothing of.
    std::optional<Json> description;
};

struct DecisionQuestion {
    std::string id;
    QuestionType type = QuestionType::Choice;
    Json instructions;  // a string, an object or an array; null when the request gives none
    std::vector<DecisionOption>
        options;  // choice: in the request's order; score: lowest level first; noul: true, false
};

struct DecisionRequest {
    Json state;                               // a string, an object or an array
    std::vector<DecisionQuestion> questions;  // in the request's order; at least one
};

// A request the model cannot take as it is; the message is for the client. `code` names the reason
// ("max_tokens_exceeded"), or is empty.
class DecisionRefused : public Error {
public:
    DecisionRefused(const std::string& message, std::string code) : Error(message), code_(std::move(code)) {}
    const std::string& code() const { return code_; }

private:
    std::string code_;
};

// What a model writes its prompt with: the checkpoint's tokenizer and chat template, held by the runtime.
class PromptTools {
public:
    virtual ~PromptTools() = default;
    // Token ids of `text`; special tokens written in it become their ids. Nothing is added at either end.
    virtual std::vector<int32_t> encode(std::string_view text) const = 0;
    // The id of the token whose vocabulary piece is exactly `piece`.
    virtual std::optional<int32_t> token_id(std::string_view piece) const = 0;
    // The text of every special token (the tokenizer's added tokens marked special).
    virtual std::vector<std::string> special_tokens() const = 0;
    // The checkpoint's chat template over OpenAI-style `messages`, with the generation prompt and thinking off. Only a
    // model whose spec asks for the template (DecisionSpec::chat_template) may call it.
    virtual std::string chat(const Json& messages) const = 0;
};

// One request in flight. The runtime asks for a step, runs it, and hands its outputs back with the request for the
// next one, until there is none.
class DecisionRun {
public:
    struct Step {
        size_t program = 0;                // index in DecisionModel::programs()
        TensorMap inputs;                  // every input of that program
        std::vector<std::string> outputs;  // the outputs to bring back
        // Tokens of the sequence's state to keep before the step runs: all of them (-1), none (0: a new sequence),
        // or a prefix that Session::truncate can cut back to (the tokens before the last step: that step undone).
        int64_t keep = -1;
    };

    virtual ~DecisionRun() = default;
    // `outputs`: what the step before gave (empty before the first). nullopt: the run is over.
    virtual std::optional<Step> next(TensorMap& outputs) = 0;
    // After the last step: each question's probabilities over its options, in DecisionQuestion::options order.
    virtual std::vector<std::vector<double>> probabilities() const = 0;
    // The token sequences the request is read as (one, or one per question or part of the questions): those known
    // so far, which is all of them once the run is over.
    virtual const std::vector<std::vector<int32_t>>& sequences() const = 0;
};

struct DecisionSpec {
    std::string id;             // display name, the checkpoint directory name by default
    std::string arch;           // e.g. "clef"
    int64_t max_tokens = 0;     // the longest sequence the model reads
    int64_t rewind_tokens = 0;  // SessionOptions::rewind_tokens the model's steps need (DecisionRun::Step::keep)
    // The folder of the checkpoint that holds its tokenizer (tokenizer.json, tokenizer_config.json) and chat template;
    // empty: the checkpoint's own directory.
    std::filesystem::path tokenizer_dir;
    bool chat_template = true;  // the model writes its prompts through the checkpoint's chat template
};

// What the session a run's steps go to can take, and what it holds as the run starts.
struct DecisionLimits {
    int64_t max_tokens = 0;    // the longest sequence (never more than DecisionSpec::max_tokens)
    int64_t chunk_tokens = 0;  // tokens a step should feed at most, for a model that can read its prompt in pieces
    // Tokens of state left by the run before (0 after a reset or a failed run): a model that reads one request's
    // prompt on top of the last one's prefix checks its own record of that prompt against this count.
    int64_t held_tokens = 0;
};

class DecisionModel {
public:
    virtual ~DecisionModel() = default;
    virtual const DecisionSpec& spec() const = 0;
    virtual const WeightStore& weights() const = 0;
    // The programs of one session, in the order DecisionRun::Step::program counts them.
    virtual std::vector<const graph::Graph*> programs() const = 0;
    // Weights the model reads on the host one row at a time (they appear in no program, so convert asks for them
    // here; they may be Q8 / Q4 in a pack).
    virtual std::vector<std::string> host_tables() const { return {}; }
    // Lays out the request's prompt and starts its run. Throws DecisionRefused for a request the model cannot take.
    virtual std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                               const DecisionLimits& limits) const = 0;
};

// How a model reads requests where its reference code leaves the caller a choice the System One API has no field for.
struct DecisionLoadOptions {
    // Questions read together after one state, by a model that packs them (d1); unset: the checkpoint's own number.
    std::optional<int64_t> set_size;
};

// Loads the decision model in `dir`: a checkpoint that model_kind() calls ModelKind::Decision. Throws xs::Error
// naming what is not supported, also an option the model has no use for.
std::unique_ptr<DecisionModel> load_decision_model(const std::filesystem::path& dir,
                                                   const DecisionLoadOptions& options = {});
std::vector<std::string> supported_decision_models();

}  // namespace xs
