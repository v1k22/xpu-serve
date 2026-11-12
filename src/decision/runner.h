#pragma once

// The decision runtime: one decision model on one device. It holds the checkpoint's tokenizer and chat template (what
// the model writes its prompt with), compiles the model's programs into one session and runs a request's steps as the
// model asks for them. Requests run one at a time.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "decision/answers.h"
#include "models/decision.h"

namespace xs {

struct DecisionRunnerOptions {
    std::string backend = "sycl";
    SessionOptions session;       // max_context bounds a request's sequences
    int64_t chunk_tokens = 1024;  // DecisionLimits::chunk_tokens
    std::string model_id;         // empty = the model directory name
};

struct DecisionTiming {
    double prompt_ms = 0;  // host: the prompt's layout and tokens
    double device_ms = 0;  // the programs, with input upload and output download
    double total_ms = 0;   // those and the host work between steps
};

struct DecisionResult {
    std::vector<DecisionAnswer> answers;  // one per question, in the request's order
    int64_t input_tokens = 0;             // tokens of every sequence the request was read as
    int sequences = 0;
    int steps = 0;  // program runs
    DecisionTiming timing;
};

class DecisionRunner {
public:
    // `model`: a model already loaded (a language model read as a decision model); null loads the one in `dir`.
    static std::unique_ptr<DecisionRunner> create(const std::filesystem::path& dir,
                                                  const DecisionRunnerOptions& options,
                                                  std::unique_ptr<DecisionModel> model = nullptr);
    ~DecisionRunner();
    DecisionRunner(const DecisionRunner&) = delete;
    DecisionRunner& operator=(const DecisionRunner&) = delete;

    const DecisionSpec& spec() const { return spec_; }
    int64_t max_tokens() const { return limits_.max_tokens; }
    // Throws DecisionRefused for a request the model cannot take.
    DecisionResult decide(const DecisionRequest& request);
    // The token sequences `request` is read as, without running it (tests, token counts).
    std::vector<std::vector<int32_t>> tokens(const DecisionRequest& request) const;

private:
    class Tools;
    DecisionRunner();

    DecisionSpec spec_;
    DecisionLimits limits_;
    std::unique_ptr<DecisionModel> model_;
    std::unique_ptr<Tools> tools_;
    std::unique_ptr<Session> session_;
    std::vector<size_t> writers_;  // the programs that write state: what a step's `keep` cuts back
};

// A small request of every question type: for warm-up and timing.
DecisionRequest sample_decision_request();

}  // namespace xs
