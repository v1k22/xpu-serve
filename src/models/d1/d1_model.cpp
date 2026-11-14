#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "models/config_util.h"
#include "models/d1/d1.h"
#include "models/lfm3/lfm3.h"
#include "models/lfm3/lfm3_decoder.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::d1 {
namespace {

using graph::Graph;
using graph::Value;

// The d1_ keys of config.json: those this code reads, and those that concern only what it does not serve (pictures,
// numbers read on a grid, boxes).
const std::set<std::string> kKeys = {"d1_codes",   "d1_attention", "d1_set_size", "d1_prompt",
                                     "d1_numbers", "d1_box_edges", "d1_vision_max_pixels"};

class D1Model;

// One step per sequence: the whole sequence, then the hidden state at each of its questions' answer positions.
class D1Run final : public DecisionRun {
public:
    D1Run(const D1Model& model, const DecisionRequest& request, Plan plan)
        : model_(model), request_(request), plan_(std::move(plan)), probabilities_(request.questions.size()) {
        for (const Sequence& s : plan_.sequences)
            sequences_.push_back(s.ids);
    }

    std::optional<Step> next(TensorMap& outputs) override;
    std::vector<std::vector<double>> probabilities() const override { return probabilities_; }
    const std::vector<std::vector<int32_t>>& sequences() const override { return sequences_; }

private:
    void take(const Sequence& s, const HostTensor& hidden);

    const D1Model& model_;
    const DecisionRequest& request_;
    Plan plan_;
    std::vector<std::vector<int32_t>> sequences_;
    std::vector<std::vector<double>> probabilities_;
    size_t next_ = 0;  // the sequence the next step reads
};

class D1Model final : public DecisionModel {
public:
    D1Model(lfm3::Config text, Config config, DecisionSpec spec, std::unique_ptr<WeightStore> weights,
            int64_t set_size)
        : text_(std::move(text)),
          config_(std::move(config)),
          spec_(std::move(spec)),
          weights_(std::move(weights)),
          set_size_(set_size),
          dec_(text_, *weights_),
          embedding_(weights_->get(dec_.embedding_name())) {
        XS_CHECK(set_size_ >= 1, "d1: a set size of ", set_size_, "; it is at least 1 (each question alone)");
        XS_CHECK(embedding_.shape == (Dims{text_.vocab_size, text_.hidden_size}), "d1: ", dec_.embedding_name(),
                 " is ", dims_to_string(embedding_.shape), ", expected [", text_.vocab_size, ", ", text_.hidden_size,
                 "]");
        const lfm3::Decoder::Inputs in = dec_.inputs(graph_, lfm3::Decoder::Attention::Grouped);
        const Value h = dec_.layers(graph_, in);
        const Value rows = graph_.input("answer_rows", DType::I32, {set_size_});
        graph_.output("hidden", dec_.final_norm(graph_, graph_.gather(h, rows)));
        graph_.validate();
        log::debug("d1: a program of ", graph_.nodes().size(), " nodes; up to ", set_size_,
                   " questions after a state; temperature ", config_.temperature);
    }

    const DecisionSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<const graph::Graph*> programs() const override { return {&graph_}; }

    std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                       const DecisionLimits& limits) const override {
        return std::make_unique<D1Run>(
            *this, request, plan(request, tools, config_, set_size_, std::min(limits.max_tokens, spec_.max_tokens)));
    }

    const lfm3::Decoder& decoder() const { return dec_; }
    const Config& config() const { return config_; }
    int64_t width() const { return text_.hidden_size; }
    // Questions a sequence holds at most: the rows of the program's "answer_rows".
    int64_t set_size() const { return set_size_; }

    // The output layer's logits of `tokens` for a final hidden state: the rows of the embedding table it is tied to.
    std::vector<double> logits(const float* hidden, const std::vector<int32_t>& tokens) const {
        std::vector<float> row(static_cast<size_t>(text_.hidden_size));
        std::vector<double> out;
        for (const int32_t t : tokens) {
            XS_CHECK(t >= 0 && t < text_.vocab_size, "d1: token ", t, " outside the vocabulary");
            quant::read_rows(embedding_, t, 1, row.data());
            double dot = 0;
            for (size_t d = 0; d < row.size(); ++d)
                dot += static_cast<double>(hidden[d]) * row[d];
            out.push_back(dot);
        }
        return out;
    }

private:
    lfm3::Config text_;
    Config config_;
    DecisionSpec spec_;
    std::unique_ptr<WeightStore> weights_;
    int64_t set_size_ = 0;
    lfm3::Decoder dec_;
    const TensorView& embedding_;
    Graph graph_;
};

std::optional<DecisionRun::Step> D1Run::next(TensorMap& outputs) {
    if (next_ > 0) take(plan_.sequences[next_ - 1], outputs.at("hidden"));
    if (next_ == plan_.sequences.size()) return std::nullopt;
    const Sequence& s = plan_.sequences[next_++];
    Step step;
    step.program = 0;
    step.outputs = {"hidden"};
    step.keep = 0;  // every sequence is read from its start
    model_.decoder().token_inputs(s.ids, 0, step.inputs);
    const std::vector<int32_t> groups = s.groups();
    step.inputs["groups"] = HostTensor::from_vector(DType::I32, {static_cast<int64_t>(groups.size())}, groups);
    HostTensor rows(DType::I32, {model_.set_size()});
    for (int64_t r = 0; r < model_.set_size(); ++r) {  // a sequence of fewer questions repeats its last
        rows.data<int32_t>()[r] =
            static_cast<int32_t>(s.answers[std::min(static_cast<size_t>(r), s.answers.size() - 1)]);
    }
    step.inputs["answer_rows"] = std::move(rows);
    return step;
}

// Each option's best logit among its tokens, over the question's temperature, through a softmax.
void D1Run::take(const Sequence& s, const HostTensor& hidden) {
    const int64_t width = model_.width();
    XS_CHECK(hidden.dtype() == DType::F32 && hidden.shape() == (Dims{model_.set_size(), width}),
             "d1: the hidden states are ", dtype_name(hidden.dtype()), dims_to_string(hidden.shape()), ", expected [",
             model_.set_size(), ", ", width, "]");
    for (size_t i = 0; i < s.questions.size(); ++i) {
        const size_t q = s.questions[i];
        const DecisionQuestion& question = request_.questions[q];
        const double t = temperature(model_.config(), question.type, question.options.size());
        std::vector<double> p;
        double top = -std::numeric_limits<double>::infinity(), total = 0;
        for (const std::vector<int32_t>& forms : plan_.option_tokens[q]) {
            const std::vector<double> l = model_.logits(hidden.data<float>() + static_cast<int64_t>(i) * width, forms);
            top = std::max(top, p.emplace_back(*std::max_element(l.begin(), l.end()) / t));
        }
        for (double& v : p)
            total += (v = std::exp(v - top));
        for (double& v : p)
            v /= total;
        probabilities_[q] = std::move(p);
    }
}

std::string bucket(QuestionType type, size_t k) {
    if (type == QuestionType::Noul) return "noul";
    if (type == QuestionType::Score) return str_cat("score:", k);
    return k <= 3 ? "choice:small" : k <= 8 ? "choice:mid" : k <= 26 ? "choice:large" : "choice:inventory";
}

}  // namespace

double temperature(const Config& config, QuestionType type, size_t options) {
    const auto it = config.temperatures.find(bucket(type, options));
    return it == config.temperatures.end() ? config.temperature : it->second;
}

Config parse_config(const nlohmann::json& config, const nlohmann::json& calibration, const nlohmann::json& tokenizer) {
    const std::string where = "config.json";
    for (auto it = config.begin(); it != config.end(); ++it) {
        if (it.key().starts_with("d1_") && !kKeys.contains(it.key())) {
            fail("d1: config.json key '", it.key(), "' is not supported");
        }
    }
    const std::string attention =
        config.contains("d1_attention") ? cfg::get_string(config, "d1_attention", where) : "causal";
    if (attention != "set") fail("d1: d1_attention '", attention, "' is not supported (set is)");
    Config c;
    const int64_t scheme = cfg::get_int_or(config, "d1_codes", 1, where);
    if (scheme != 1 && scheme != 2) fail("d1: d1_codes ", scheme, " is not supported (1 and 2 are)");
    c.code_scheme = static_cast<int>(scheme);
    c.set_size = cfg::get_int_or(config, "d1_set_size", 4, where);
    XS_CHECK(c.set_size >= 2 && c.set_size <= 128, "d1: d1_set_size ", c.set_size, " is outside 2 to 128");
    const int64_t prompt = cfg::get_int_or(config, "d1_prompt", 1, where);
    if (prompt != 1 && prompt != 2) fail("d1: d1_prompt ", prompt, " is not supported (1 and 2 are)");

    if (tokenizer.contains("bos_token")) {
        const nlohmann::json& bos = tokenizer.at("bos_token");
        if (bos.is_string()) c.bos = bos.get<std::string>();
        if (bos.is_object()) c.bos = cfg::get_string(bos, "content", "tokenizer_config.json bos_token");
    }

    if (!calibration.is_null()) {
        const std::string cwhere = "calibration.json";
        if (!calibration.contains("temperatures") || !calibration.at("temperatures").is_object()) {
            fail("d1: calibration.json has no temperatures");
        }
        const auto positive = [](double t) { return std::isfinite(t) && t > 0; };
        for (auto it = calibration.at("temperatures").begin(); it != calibration.at("temperatures").end(); ++it) {
            XS_CHECK(it.value().is_number() && positive(it.value().get<double>()), "d1: calibration.json temperature '",
                     it.key(), "' is ", it.value().dump());
            c.temperatures[it.key()] = it.value().get<double>();
        }
        if (calibration.contains("default")) c.temperature = cfg::get_float(calibration, "default", cwhere);
        XS_CHECK(positive(c.temperature), "d1: calibration.json default temperature ", c.temperature);
    }
    return c;
}

bool is_checkpoint(const std::filesystem::path& dir) {
    const nlohmann::json config = cfg::read_json(dir / "config.json");
    return config.contains("architectures") && config.at("architectures") == nlohmann::json::array({"D1Model"});
}

std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json,
                                    const DecisionLoadOptions& options) {
    lfm3::Config text = lfm3::parse_config(lfm3::text_config(config_json));
    const std::filesystem::path calibration = dir / "calibration.json";
    Config config = parse_config(config_json,
                                 std::filesystem::exists(calibration) ? cfg::read_json(calibration) : nlohmann::json(),
                                 cfg::read_json(dir / "tokenizer_config.json"));
    DecisionSpec spec;
    spec.id = cfg::model_id(dir);
    spec.arch = "d1";
    spec.max_tokens = text.max_position;
    spec.chat_template = false;  // the prompt is ChatML written as prompt.py writes it
    const int64_t set_size = options.set_size.value_or(config.set_size);
    return std::make_unique<D1Model>(std::move(text), std::move(config), std::move(spec),
                                     SafetensorsWeights::open_dir(dir), set_size);
}

}  // namespace xs::d1
