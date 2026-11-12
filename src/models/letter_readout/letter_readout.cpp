#include "models/letter_readout/letter_readout.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>

#include "models/common/py_text.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::letter_readout {
namespace {

// The options of one prompt as the helper names them: (key, description).
using Options = std::vector<std::pair<std::string, std::string>>;

char letter(size_t i) {
    return static_cast<char>(i < 26 ? 'A' + i : 'a' + (i - 26));
}

// A description as the prompt shows it: nothing for null, a string as it is, anything else as json.dumps writes it.
std::string describe(const std::optional<Json>& v) {
    if (!v || v->is_null()) return "";
    return v->is_string() ? v->get<std::string>() : py::json_dumps(*v);
}

// The state as text: a string as it is, an object as json.dumps writes it, an array as Python's str() does.
std::string state_text(const Json& state) {
    if (state.is_object()) {
        for (const char* key : {"screenshot", "image"}) {
            if (!state.contains(key) || !state[key].is_string()) continue;
            const std::string& v = state[key].get_ref<const std::string&>();
            if (v.rfind("data:image", 0) == 0 || v.size() > 2000) {
                throw DecisionRefused(str_cat("the state's '", key, "' is an image; this server reads no images"),
                                      "unsupported");
            }
        }
        return py::json_dumps(state);
    }
    return py::str(state);
}

size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    return static_cast<size_t>(std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first - a.begin());
}

// What the session holds after the last step a run of this model asked for.
struct Held {
    std::vector<int32_t> tokens;
    int64_t base = 0;  // where the last step began: what Session::truncate can go back to
};

// One prompt and the probabilities read from it.
struct Readout {
    size_t question = 0;
    std::vector<size_t> options;  // the question's prompt options it lists (all of them, or a part of many)
    std::vector<int32_t> ids;
    std::vector<double> p;  // one per listed option, once read
    bool winners = false;   // the pass over the best option of each part
};

// A question as the helper asks it.
struct Asked {
    std::string instructions;
    Options options;
    std::vector<std::vector<double>> parts;  // many options: each part's probabilities, in order
    std::vector<size_t> part_sizes;
};

class LetterModel;

class LetterRun final : public DecisionRun {
public:
    LetterRun(const LetterModel& model, const DecisionRequest& request, const PromptTools& tools,
              const DecisionLimits& limits);

    std::optional<Step> next(TensorMap& outputs) override;
    std::vector<std::vector<double>> probabilities() const override { return probabilities_; }
    const std::vector<std::vector<int32_t>>& sequences() const override { return sequences_; }

private:
    Readout make_readout(size_t question, std::vector<size_t> options, bool winners);
    void take(const HostTensor& logits);
    void finish(size_t question, std::vector<double> p);
    Step feed();

    const LetterModel& model_;
    const DecisionRequest& request_;
    const PromptTools& tools_;
    DecisionLimits limits_;
    std::string state_;
    std::vector<Asked> asked_;
    std::deque<Readout> queue_;
    std::optional<Readout> current_;  // the read-out whose prompt is being fed
    size_t fed_ = 0;                  // tokens of it the session holds
    size_t shared_ = 0;               // where its prompt stops being the other prompts' too (0: fed in one go)
    std::vector<int32_t> common_;     // the start every prompt of the request has: the chat header and the state
    std::vector<std::vector<int32_t>> sequences_;
    std::vector<std::vector<double>> probabilities_;
};

class LetterModel final : public DecisionModel {
public:
    LetterModel(std::unique_ptr<Model> model, const Settings& settings)
        : model_(std::move(model)), settings_(settings) {
        XS_CHECK(settings_.temperature > 0 && settings_.noul_temperature > 0 && std::isfinite(settings_.noul_bias),
                 "letter read-out: temperatures must be positive");
        const ModelInfo& info = model_->info();
        spec_.id = info.id;
        spec_.arch = "letter_readout";
        spec_.max_tokens = info.max_position;
        const auto& outputs = model_->graph().outputs();
        const auto gives = [&](const char* name) {
            return std::any_of(outputs.begin(), outputs.end(), [&](const auto& o) { return o.name == name; });
        };
        XS_CHECK(gives("logits"), "letter read-out: model ", info.id, " gives no logits");
        filler_ = gives("argmax") ? "argmax" : "logits";
    }

    const DecisionSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return model_->weights(); }
    std::vector<const graph::Graph*> programs() const override { return {&model_->graph()}; }

    std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                       const DecisionLimits& limits) const override {
        if (static_cast<int64_t>(held_.tokens.size()) != limits.held_tokens) held_ = Held{};  // the session moved on
        return std::make_unique<LetterRun>(*this, request, tools, limits);
    }

    const Model& model() const { return *model_; }
    const Settings& settings() const { return settings_; }
    const std::string& filler_output() const { return filler_; }
    Held& held() const { return held_; }
    // The token of each letter; the tokenizer must give every letter one token of its own.
    const std::vector<int32_t>& letter_ids(const PromptTools& tools) const {
        if (letters_.empty()) {
            for (size_t i = 0; i < kLetters; ++i) {
                const std::vector<int32_t> ids = tools.encode(std::string(1, letter(i)));
                XS_CHECK(ids.size() == 1 && std::find(letters_.begin(), letters_.end(), ids[0]) == letters_.end(),
                         "letter read-out: the letter '", letter(i), "' is not one token of its own");
                letters_.push_back(ids[0]);
            }
        }
        return letters_;
    }

private:
    std::unique_ptr<Model> model_;
    Settings settings_;
    DecisionSpec spec_;
    std::string filler_;
    // One request runs at a time on the model's one session: what that session holds, and the letters' tokens.
    mutable Held held_;
    mutable std::vector<int32_t> letters_;
};

LetterRun::LetterRun(const LetterModel& model, const DecisionRequest& request, const PromptTools& tools,
                     const DecisionLimits& limits)
    : model_(model), request_(request), tools_(tools), limits_(limits), probabilities_(request.questions.size()) {
    limits_.max_tokens = std::min(limits_.max_tokens, model.spec().max_tokens);
    state_ = state_text(request.state);
    for (size_t i = 0; i < request.questions.size(); ++i) {
        const DecisionQuestion& q = request.questions[i];
        const auto refuse = [&](const char* why) {
            throw DecisionRefused(str_cat("question '", q.id, "': ", why), "unsupported_question");
        };
        if (q.instructions.is_null()) refuse("this model needs instructions");
        Asked a;
        a.instructions = q.instructions.is_string()             ? q.instructions.get<std::string>()
                         : model.settings().python_instructions ? py::str(q.instructions)
                                                                : py::json_dumps(q.instructions);
        if (q.type == QuestionType::Noul) {
            // The yes option first: the answer's order (true, false) is the prompt's.
            const std::string yes = describe(q.options[0].description), no = describe(q.options[1].description);
            a.options = {{"yes", yes.empty() ? "The statement is true." : yes},
                         {"no", no.empty() ? "The statement is false." : no}};
        } else {
            if (q.type == QuestionType::Score) {
                if (q.options.size() < 2) refuse("the model scores on at least two ordered levels");
                a.instructions += " Rate along the ordered levels below (lowest first).";
            }
            for (const DecisionOption& o : q.options)
                a.options.emplace_back(o.id, describe(o.description));
        }
        asked_.push_back(std::move(a));
    }
    // One prompt a question; a question of more options than there are letters in near-equal parts, then a pass over
    // each part's best.
    for (size_t i = 0; i < asked_.size(); ++i) {
        const size_t n = asked_[i].options.size();
        const size_t parts = (n + kLetters - 1) / kLetters, size = (n + parts - 1) / parts;
        for (size_t first = 0; first < n; first += size) {
            std::vector<size_t> options;
            for (size_t o = first; o < std::min(n, first + size); ++o)
                options.push_back(o);
            asked_[i].part_sizes.push_back(options.size());
            queue_.push_back(make_readout(i, std::move(options), false));
        }
    }
    // The start the request's prompts share: found against the other questions' prompts, or, for a single question,
    // against one made for the purpose (the next request may ask about the same state).
    size_t shared = queue_[0].ids.size();
    if (queue_.size() == 1) {
        const Json other = Json::array({Json{{"role", "user"}, {"content", user_message(state_, "?", {{"a", ""}})}}});
        shared = common_prefix(queue_[0].ids, tools_.encode(tools_.chat(other)));
    }
    for (size_t i = 1; i < queue_.size(); ++i)
        shared = std::min(shared, common_prefix(queue_[0].ids, queue_[i].ids));
    common_.assign(queue_[0].ids.begin(), queue_[0].ids.begin() + static_cast<std::ptrdiff_t>(shared));
}

Readout LetterRun::make_readout(size_t question, std::vector<size_t> options, bool winners) {
    Readout r;
    r.question = question;
    r.winners = winners;
    Options listed;
    for (const size_t o : options)
        listed.push_back(asked_[question].options[o]);
    r.options = std::move(options);
    const Json messages =
        Json::array({Json{{"role", "user"}, {"content", user_message(state_, asked_[question].instructions, listed)}}});
    r.ids = tools_.encode(tools_.chat(messages));
    if (static_cast<int64_t>(r.ids.size()) > limits_.max_tokens) {
        throw DecisionRefused(str_cat("question '", request_.questions[question].id, "' needs ", r.ids.size(),
                                      " tokens with the state; the limit is ", limits_.max_tokens),
                              "max_tokens_exceeded");
    }
    sequences_.push_back(r.ids);
    return r;
}

std::optional<DecisionRun::Step> LetterRun::next(TensorMap& outputs) {
    if (current_ && fed_ == current_->ids.size()) {
        take(outputs.at("logits"));
        current_.reset();
    }
    if (!current_) {
        if (queue_.empty()) return std::nullopt;
        current_ = std::move(queue_.front());
        queue_.pop_front();
        fed_ = std::numeric_limits<size_t>::max();  // not started
    }
    return feed();
}

// The next step of the current prompt. Its first step decides what of the session's state stays: all of it when the
// prompt goes on from there, the tokens before the last step when the prompt shares those, nothing otherwise.
DecisionRun::Step LetterRun::feed() {
    Held& held = model_.held();
    const std::vector<int32_t>& ids = current_->ids;
    Step step;
    step.program = 0;
    if (fed_ == std::numeric_limits<size_t>::max()) {
        const size_t same = std::min(common_prefix(ids, held.tokens), ids.size() - 1);  // the last token must run
        if (same > 0 && same == held.tokens.size()) {
            fed_ = same;
        } else if (held.base > 0 && same >= static_cast<size_t>(held.base)) {
            fed_ = static_cast<size_t>(held.base);
            step.keep = held.base;
        } else {
            fed_ = 0;
            step.keep = 0;
        }
        held.tokens.resize(fed_);
        const bool starts_alike = common_.size() < ids.size() && common_prefix(ids, common_) == common_.size();
        shared_ = starts_alike && common_.size() > fed_ ? common_.size() : 0;
    }
    // Up to the shared part's end in steps, then the question's own part in one, so that the next question can undo it.
    const auto chunk = static_cast<size_t>(std::max<int64_t>(limits_.chunk_tokens, 1));
    const size_t stop = fed_ < shared_ ? shared_ : ids.size();
    const size_t count = std::min(chunk, stop - fed_);
    held.base = static_cast<int64_t>(fed_);
    model_.model().prepare_inputs(std::span<const int32_t>(ids).subspan(fed_, count), static_cast<int64_t>(fed_), {},
                                  step.inputs);
    fed_ += count;
    held.tokens.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(fed_));
    // Only a prompt's last step is read; a step before it asks for the cheapest output there is.
    step.outputs = {fed_ == ids.size() ? "logits" : model_.filler_output()};
    return step;
}

// The letters' scores over the temperature through a softmax: the listed options' probabilities.
void LetterRun::take(const HostTensor& logits) {
    const std::vector<int32_t>& letters = model_.letter_ids(tools_);
    const int64_t vocab = model_.model().info().vocab_size;
    XS_CHECK(logits.dtype() == DType::F32 && logits.numel() == vocab, "letter read-out: logits are ",
             dtype_name(logits.dtype()), dims_to_string(logits.shape()));
    Readout& r = *current_;
    r.p.resize(r.options.size());
    double top = -std::numeric_limits<double>::infinity(), total = 0;
    for (size_t i = 0; i < r.p.size(); ++i) {
        top = std::max(top, r.p[i] = logits.data<float>()[letters[i]] / model_.settings().temperature);
    }
    for (double& v : r.p)
        total += (v = std::exp(v - top));
    for (double& v : r.p)
        v /= total;

    Asked& a = asked_[r.question];
    if (a.part_sizes.size() == 1) return finish(r.question, std::move(r.p));
    if (!r.winners) {
        a.parts.push_back(std::move(r.p));
        if (a.parts.size() < a.part_sizes.size()) return;
        // Every part is read: its best options, one each, go into one more prompt, after the prompts already planned.
        std::vector<size_t> best;
        size_t first = 0;
        for (size_t part = 0; part < a.parts.size(); ++part) {
            const auto& p = a.parts[part];
            best.push_back(first + static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin()));
            first += p.size();
        }
        queue_.push_back(make_readout(r.question, std::move(best), true));
        return;
    }
    // p(option) is its part's share (the winners' pass) times its probability in the part over the part winner's:
    // each winner carries its part's mass on the winners' scale.
    std::vector<double> all;
    double sum = 0;
    for (size_t part = 0; part < a.parts.size(); ++part) {
        const auto& p = a.parts[part];
        const double winner = *std::max_element(p.begin(), p.end());
        for (const double v : p)
            sum += all.emplace_back(r.p[part] * v / winner);
    }
    for (double& v : all)
        v /= sum;
    finish(r.question, std::move(all));
}

void LetterRun::finish(size_t question, std::vector<double> p) {
    if (request_.questions[question].type == QuestionType::Noul) {
        // The yes option's probability, recalibrated: sigmoid(logit(p) / noul_temperature + noul_bias).
        const Settings& s = model_.settings();
        const double yes = std::clamp(p[0], 1e-4, 1 - 1e-4);
        const double z = std::log(yes / (1 - yes)) / s.noul_temperature + s.noul_bias;
        p[0] = 1 / (1 + std::exp(-z));
        p[1] = 1 - p[0];
    }
    probabilities_[question] = std::move(p);
}

}  // namespace

std::string user_message(const std::string& state, const std::string& instructions, const Options& options) {
    std::string lines;
    for (size_t i = 0; i < options.size(); ++i) {
        lines += str_cat(i > 0 ? "\n" : "", "[", letter(i), "] ", options[i].first, ": ", options[i].second);
    }
    return str_cat("State:\n", state, "\n\nQuestion: ", instructions, "\nOptions:\n", lines,
                   "\n\nAnswer with the letter of the best option only.");
}

std::optional<Settings> preset(std::string_view name) {
    // openjev/openjev's serving guide (serve/SERVE.md): READOUT_T, READOUT_NOUL_T, READOUT_NOUL_BIAS,
    // READOUT_INSTR_STYLE=pyrepr.
    if (name == "openjev") return Settings{0.85, 1.829074, 0.0, true};
    if (name == "letters") return Settings{};
    return std::nullopt;
}

std::vector<std::string> preset_names() {
    return {"letters", "openjev"};
}

std::unique_ptr<DecisionModel> make(std::unique_ptr<Model> model, const Settings& settings) {
    return std::make_unique<LetterModel>(std::move(model), settings);
}

}  // namespace xs::letter_readout
