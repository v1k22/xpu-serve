#include <algorithm>
#include <cmath>
#include <limits>

#include "io/joined_weights.h"
#include "models/config_util.h"
#include "models/pplx_decider/pplx_decider.h"
#include "models/qwen3_5/qwen3_5_decoder.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::pplx_decider {
namespace {

using graph::Graph;
using graph::Value;

// Where the read-out sits beside the decoder's weights.
constexpr const char* kReadout = "readout.";

class PplxModel;

// One step per sequence: the whole prompt, then the read-out rows at its markers.
class PplxRun final : public DecisionRun {
public:
    PplxRun(const PplxModel& model, const DecisionRequest& request, std::vector<Encoded> plan)
        : model_(model), request_(request), plan_(std::move(plan)), probabilities_(request.questions.size()) {
        for (const Encoded& e : plan_)
            sequences_.push_back(e.ids);
    }

    std::optional<Step> next(TensorMap& outputs) override;
    std::vector<std::vector<double>> probabilities() const override { return probabilities_; }
    const std::vector<std::vector<int32_t>>& sequences() const override { return sequences_; }

private:
    void take(const Encoded& e, const HostTensor& readout);

    const PplxModel& model_;
    const DecisionRequest& request_;
    std::vector<Encoded> plan_;
    std::vector<std::vector<int32_t>> sequences_;
    std::vector<std::vector<double>> probabilities_;
    size_t next_ = 0;  // the sequence the next step reads
};

class PplxModel final : public DecisionModel {
public:
    PplxModel(qwen3_5::Config cfg, Config decision, DecisionSpec spec, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)),
          decision_(std::move(decision)),
          spec_(std::move(spec)),
          weights_(std::move(weights)),
          dec_(cfg_, *weights_) {
        const qwen3_5::Decoder::Inputs in = dec_.inputs(graph_);
        // Full-attention layers over the whole prompt; the read-out of every row (a marker's row is picked on the
        // host).
        const Value hidden =
            dec_.final_norm(graph_, dec_.layers(graph_, in, qwen3_5::Decoder::Attention::WholeStep).back());
        const Value readout = graph_.weight(*weights_, std::string(kReadout) + "weight");
        XS_CHECK(graph_.type(readout).shape == (Dims{static_cast<int64_t>(decision_.codes.size()), cfg_.hidden_size}),
                 "pplx_decider: the read-out is ", dims_to_string(graph_.type(readout).shape), ", expected [",
                 decision_.codes.size(), ", ", cfg_.hidden_size, "]");
        graph_.output("readout", graph_.matmul(hidden, readout));
        graph_.validate();
        log::debug("pplx_decider: a program of ", graph_.nodes().size(), " nodes; temperature ", decision_.temperature);
    }

    const DecisionSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<const graph::Graph*> programs() const override { return {&graph_}; }

    std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                       const DecisionLimits& limits) const override {
        return std::make_unique<PplxRun>(
            *this, request, plan(request, tools, decision_, std::min(limits.max_tokens, spec_.max_tokens)));
    }

    const qwen3_5::Decoder& decoder() const { return dec_; }
    const Config& decision() const { return decision_; }

private:
    qwen3_5::Config cfg_;
    Config decision_;
    DecisionSpec spec_;
    std::unique_ptr<WeightStore> weights_;
    qwen3_5::Decoder dec_;
    Graph graph_;
};

std::optional<DecisionRun::Step> PplxRun::next(TensorMap& outputs) {
    if (next_ > 0) take(plan_[next_ - 1], outputs.at("readout"));
    if (next_ == plan_.size()) return std::nullopt;
    Step step;
    step.program = 0;
    step.outputs = {"readout"};
    step.keep = 0;  // every sequence is read from its start
    model_.decoder().token_inputs(plan_[next_++].ids, 0, step.inputs);
    return step;
}

// Each question's scores of its own options at its marker, over the temperature, through a softmax.
void PplxRun::take(const Encoded& e, const HostTensor& readout) {
    const auto codes = static_cast<int64_t>(model_.decision().codes.size());
    XS_CHECK(readout.dtype() == DType::F32 && readout.shape() == (Dims{static_cast<int64_t>(e.ids.size()), codes}),
             "pplx_decider: the read-out is ", dtype_name(readout.dtype()), dims_to_string(readout.shape()),
             ", expected [", e.ids.size(), ", ", codes, "]");
    for (size_t i = 0; i < e.questions.size(); ++i) {
        const DecisionQuestion& q = request_.questions[e.questions[i]];
        const float* row = readout.data<float>() + e.markers[i] * codes;
        const size_t n = q.options.size();
        std::vector<double> p(n);
        double top = -std::numeric_limits<double>::infinity(), total = 0;
        for (size_t o = 0; o < n; ++o)
            top = std::max(top, p[o] = row[o] / model_.decision().temperature);
        for (double& v : p)
            total += (v = std::exp(v - top));
        for (double& v : p)
            v /= total;
        // The prompt lists a noul's "false" first; the answer's order is true, false.
        if (q.type == QuestionType::Noul) std::swap(p[0], p[1]);
        probabilities_[e.questions[i]] = std::move(p);
    }
}

}  // namespace

Config parse_config(const nlohmann::json& decision, const nlohmann::json& joint, const nlohmann::json& tokenizer) {
    Config c;
    const std::string where = "joint_config.json";
    const int64_t version = cfg::get_int(joint, "version", where);
    const std::string layout = cfg::get_string(joint, "template", where);
    if (version != 3 || layout != "joint-question-inline-readout-v1") {
        fail("pplx_decider: protocol version ", version, " ('", layout, "') is not supported (version 3, ",
             "joint-question-inline-readout-v1, is)");
    }
    if (cfg::get_int_or(joint, "readout_offset", 0, where) != 0)
        fail("pplx_decider: a read-out offset is not supported");
    c.marker = cfg::get_string(joint, "marker", where);

    const std::string dwhere = "decision_config.json";
    const std::string mode = cfg::get_string(decision, "attention_mode", dwhere);
    if (mode != "noncausal_full_attention") fail("pplx_decider: attention_mode '", mode, "' is not supported");
    if (decision.contains("pooling") && decision.at("pooling") != "last") {
        fail("pplx_decider: pooling '", decision.at("pooling").dump(), "' is not supported (a read-out per marker is)");
    }
    c.temperature = cfg::get_float(decision, "temperature", dwhere);
    XS_CHECK(std::isfinite(c.temperature) && c.temperature > 0, "pplx_decider: temperature ", c.temperature);
    XS_CHECK(decision.contains("codes") && decision.at("codes").is_array(),
             "pplx_decider: decision_config.json has no codes");
    c.codes = decision.at("codes").get<std::vector<std::string>>();
    XS_CHECK(!c.codes.empty(), "pplx_decider: no answer codes");

    // tokenizer.all_special_tokens: the named special tokens of tokenizer_config.json. The chat markers are not among
    // them in these checkpoints, and the reference leaves them alone too.
    std::vector<std::string> special{c.marker};
    const auto add = [&](const nlohmann::json& v) {
        if (v.is_string() && std::find(special.begin(), special.end(), v.get<std::string>()) == special.end()) {
            special.push_back(v.get<std::string>());
        }
    };
    for (auto it = tokenizer.begin(); it != tokenizer.end(); ++it) {
        if (it.key().ends_with("_token")) add(it.value());
        if (it.key() == "model_specific_special_tokens" || it.key() == "extra_special_tokens") {
            for (const auto& v : it.value())
                add(v);
        }
        if (it.key() == "additional_special_tokens" && it.value().is_array()) {
            for (const auto& v : it.value())
                add(v);
        }
    }
    std::stable_sort(special.begin(), special.end(),
                     [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    c.neutralized = std::move(special);
    return c;
}

bool is_checkpoint(const std::filesystem::path& dir) {
    return std::filesystem::exists(dir / "joint_config.json") && std::filesystem::exists(dir / "decision_config.json");
}

std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json) {
    qwen3_5::Config cfg = qwen3_5::parse_config(config_json);
    XS_CHECK(cfg.num_experts == 0, "pplx_decider: a mixture-of-experts decoder is not supported");
    Config decision =
        parse_config(cfg::read_json(dir / "decision_config.json"), cfg::read_json(dir / "joint_config.json"),
                     cfg::read_json(dir / "tokenizer_config.json"));
    DecisionSpec spec;
    spec.id = cfg::model_id(dir);
    spec.arch = "pplx_decider";
    spec.max_tokens = cfg.max_position;
    // A checkpoint keeps the read-out in a file of its own; a pack holds it beside the decoder.
    std::unique_ptr<WeightStore> weights = SafetensorsWeights::open_dir(dir);
    if (!weights->contains(std::string(kReadout) + "weight")) {
        const std::filesystem::path readout = dir / "readout.safetensors";
        XS_CHECK(std::filesystem::exists(readout), "pplx_decider: ", readout.string(), " is missing");
        weights = std::make_unique<JoinedWeights>(std::move(weights), SafetensorsWeights::open_file(readout), kReadout);
    }
    return std::make_unique<PplxModel>(std::move(cfg), std::move(decision), std::move(spec), std::move(weights));
}

}  // namespace xs::pplx_decider
