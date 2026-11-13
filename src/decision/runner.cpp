#include "decision/runner.h"

#include <algorithm>

#include "chat/chat_template.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"
#include "util/log.h"
#include "util/timer.h"

namespace xs {

class DecisionRunner::Tools final : public PromptTools {
public:
    // The tokenizer in `dir`, and its chat template when `chat_template`.
    Tools(const std::filesystem::path& dir, bool chat_template)
        : tokenizer_(tok::Tokenizer::load(dir)), template_(chat_template ? chat::ChatTemplate::load(dir) : nullptr) {
        for (int32_t id = 0; id < tokenizer_->vocab_size(); ++id) {
            if (tokenizer_->is_special(id)) special_.push_back(tokenizer_->token_text(id));
        }
    }

    std::vector<int32_t> encode(std::string_view text) const override { return tokenizer_->encode(text); }
    std::optional<int32_t> token_id(std::string_view piece) const override { return tokenizer_->token_id(piece); }
    std::vector<std::string> special_tokens() const override { return special_; }
    std::string chat(const Json& messages) const override {
        XS_CHECK(template_ != nullptr, "this decision model writes its prompts without a chat template");
        return template_->render(messages, Json(nullptr), true, Json{{"enable_thinking", false}});
    }

private:
    std::unique_ptr<tok::Tokenizer> tokenizer_;
    std::unique_ptr<chat::ChatTemplate> template_;
    std::vector<std::string> special_;
};

DecisionRunner::DecisionRunner() = default;
DecisionRunner::~DecisionRunner() = default;

std::unique_ptr<DecisionRunner> DecisionRunner::create(const std::filesystem::path& dir,
                                                       const DecisionRunnerOptions& options,
                                                       std::unique_ptr<DecisionModel> model) {
    std::unique_ptr<DecisionRunner> r(new DecisionRunner());
    Timer timer;
    r->model_ = model ? std::move(model) : load_decision_model(dir);
    r->spec_ = r->model_->spec();
    r->tools_ = std::make_unique<Tools>(r->spec_.tokenizer_dir.empty() ? dir : dir / r->spec_.tokenizer_dir,
                                        r->spec_.chat_template);
    if (!options.model_id.empty()) r->spec_.id = options.model_id;
    const std::vector<const graph::Graph*> programs = r->model_->programs();
    XS_CHECK(!programs.empty(), "decision model ", r->spec_.id, " has no program");
    size_t nodes = 0;
    for (size_t i = 0; i < programs.size(); ++i) {
        nodes += programs[i]->nodes().size();
        const bool reads = std::any_of(programs[i]->nodes().begin(), programs[i]->nodes().end(),
                                       [](const graph::Node& n) { return n.kind == graph::OpKind::KvRead; });
        if (programs[i]->stateful() && !reads) r->writers_.push_back(i);
    }
    SessionOptions session = options.session;
    if (session.max_context > r->spec_.max_tokens) {
        log::warn("context ", session.max_context, " exceeds the model's ", r->spec_.max_tokens, "; using ",
                  r->spec_.max_tokens);
        session.max_context = r->spec_.max_tokens;
    }
    session.rewind_tokens = r->spec_.rewind_tokens;
    r->limits_.max_tokens = session.max_context;
    r->limits_.chunk_tokens = options.chunk_tokens;
    XS_CHECK(r->limits_.chunk_tokens > 0, "a step must feed at least one token, got ", options.chunk_tokens);
    log::info("decision model ", r->spec_.id, " (", r->spec_.arch, "): ", programs.size(), " programs of ", nodes,
              " nodes; requests of up to ", r->limits_.max_tokens, " tokens");
    auto backend = make_backend(options.backend);
    Timer compile_timer;
    r->session_ = backend->compile(programs, r->model_->weights(), session);
    log::info("compiled on ", options.backend, " ", session.device.empty() ? "(default device)" : session.device,
              " in ", static_cast<int>(compile_timer.elapsed_ms()), " ms");
    // The first run of each program pays the backend's kernel setup and memory plans: pay it here.
    Timer warm;
    r->decide(sample_decision_request());
    log::info("warmed up in ", static_cast<int>(warm.elapsed_ms()), " ms; ready in ",
              static_cast<int>(timer.elapsed_ms()), " ms");
    return r;
}

DecisionLimits DecisionRunner::current_limits() const {
    DecisionLimits limits = limits_;
    limits.held_tokens = writers_.empty() ? 0 : session_->num_cached(writers_.front());
    return limits;
}

std::vector<std::vector<int32_t>> DecisionRunner::tokens(const DecisionRequest& request) const {
    return model_->start(request, *tools_, current_limits())->sequences();
}

DecisionResult DecisionRunner::decide(const DecisionRequest& request) {
    XS_CHECK(!request.questions.empty(), "a decision request needs at least one question");
    DecisionResult result;
    Timer timer;
    const std::unique_ptr<DecisionRun> run = model_->start(request, *tools_, current_limits());
    result.timing.prompt_ms = timer.elapsed_ms();
    TensorMap outputs;
    try {
        for (;;) {
            std::optional<DecisionRun::Step> step = run->next(outputs);
            if (!step) break;
            if (step->keep == 0) {
                session_->reset();
            } else if (step->keep > 0) {
                for (const size_t program : writers_) {
                    if (session_->num_cached(program) > step->keep) session_->truncate(program, step->keep);
                }
            }
            Timer device;
            outputs.clear();
            session_->run(step->program, step->inputs, outputs, step->outputs);
            const double ms = device.elapsed_ms();
            log::debug("decision: step ", result.steps, ", program ", step->program, ": ", ms, " ms");
            result.timing.device_ms += ms;
            ++result.steps;
        }
    } catch (...) {
        session_->reset();  // a half-read request must not be the next one's prefix
        throw;
    }
    const std::vector<std::vector<double>> probabilities = run->probabilities();
    XS_CHECK(probabilities.size() == request.questions.size(), "the model answered ", probabilities.size(), " of ",
             request.questions.size(), " questions");
    for (size_t q = 0; q < probabilities.size(); ++q) {
        XS_CHECK(probabilities[q].size() == request.questions[q].options.size(), "question '", request.questions[q].id,
                 "' has ", request.questions[q].options.size(), " options, the model scored ", probabilities[q].size());
        result.answers.push_back(make_answer(request.questions[q].type, probabilities[q]));
    }
    result.sequences = run->sequences();
    for (const auto& sequence : result.sequences)
        result.input_tokens += static_cast<int64_t>(sequence.size());
    result.timing.total_ms = timer.elapsed_ms();
    return result;
}

DecisionRequest sample_decision_request() {
    DecisionRequest r;
    r.state = "Customer message: I was charged twice for my order last week and nobody has replied.";
    DecisionQuestion route;
    route.id = "route";
    route.type = QuestionType::Choice;
    route.instructions = "Which team should handle this?";
    route.options = {{"billing", Json("Charges and refunds")},
                     {"shipping", Json("Delivery and tracking")},
                     {"technical", Json("Bugs and outages")}};
    DecisionQuestion angry;
    angry.id = "angry";
    angry.type = QuestionType::Noul;
    angry.instructions = "Is the customer angry?";
    angry.options = {{"true", std::nullopt}, {"false", std::nullopt}};
    DecisionQuestion urgency;
    urgency.id = "urgency";
    urgency.type = QuestionType::Score;
    urgency.instructions = "How urgent is this?";
    urgency.options = {{"0", Json("can wait")}, {"1", Json("this week")}, {"2", Json("today")}};
    r.questions = {std::move(route), std::move(angry), std::move(urgency)};
    return r;
}

}  // namespace xs
