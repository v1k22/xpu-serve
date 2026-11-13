#include <cmath>
#include <limits>

#include "models/common/blocks.h"
#include "models/common/encoder.h"
#include "models/common/torch_nn.h"
#include "models/config_util.h"
#include "models/laya/laya.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::laya {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

constexpr float kHeadNormEps = 1e-5f;  // nn.TransformerEncoderLayer's and nn.LayerNorm's default

// The references' QTYPES: a question type's row of type_emb.
int32_t type_row(QuestionType type) {
    switch (type) {
        case QuestionType::Choice: return 0;
        case QuestionType::Score: return 1;
        case QuestionType::Noul: return 2;
    }
    fail("laya: unknown question type");
}

class LayaModel;

// One step per question: its sequence whole, then the scores at its markers.
class LayaRun final : public DecisionRun {
public:
    LayaRun(const LayaModel& model, const DecisionRequest& request, std::vector<Encoded> encoded)
        : model_(model), request_(request), encoded_(std::move(encoded)), probabilities_(request.questions.size()) {
        for (const Encoded& e : encoded_)
            sequences_.push_back(e.ids);
    }

    std::optional<Step> next(TensorMap& outputs) override;
    std::vector<std::vector<double>> probabilities() const override { return probabilities_; }
    const std::vector<std::vector<int32_t>>& sequences() const override { return sequences_; }

private:
    void take(size_t question, const HostTensor& scores);

    const LayaModel& model_;
    const DecisionRequest& request_;
    std::vector<Encoded> encoded_;
    std::vector<std::vector<int32_t>> sequences_;
    std::vector<std::vector<double>> probabilities_;
    size_t next_ = 0;  // the question the next step reads
};

class LayaModel final : public DecisionModel {
public:
    LayaModel(Config cfg, DecisionSpec spec, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)), spec_(std::move(spec)), weights_(std::move(weights)) {
        const int64_t d = cfg_.encoder.hidden_size;
        heads_ = std::max<int64_t>(1, d / 64);  // the reference's nhead
        XS_CHECK(d % heads_ == 0, spec_.arch, ": the head's width ", d, " is not a multiple of its ", heads_, " heads");
        const Value ids = graph_.input("ids", DType::I32, {kT});
        const Value positions = graph_.input("positions", DType::I32, {kT});
        const Value type = graph_.input("type", DType::I32, {1});
        Value h = modern_bert::encode(graph_, *weights_, cfg_.encoder, ids, positions, "encoder.");
        const Value type_emb = graph_.weight(*weights_, "type_emb.weight");
        XS_CHECK(graph_.type(type_emb).shape == (Dims{3, d}), spec_.arch, ": type_emb.weight is ",
                 dims_to_string(graph_.type(type_emb).shape), ", expected [3, ", d, "]");
        h = graph_.add(h, graph_.gather(type_emb, type));  // the question type's row, on every token
        for (int64_t i = 0; i < cfg_.head_layers; ++i)
            h = head_layer(graph_, h, "head.layers." + std::to_string(i) + ".");
        graph_.output("scores", graph_.reshape(scorer(graph_, h), {kT}));
        graph_.validate();
        log::debug(spec_.arch, ": a program of ", graph_.nodes().size(), " nodes");
    }

    const DecisionSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<const graph::Graph*> programs() const override { return {&graph_}; }

    std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                       const DecisionLimits& limits) const override {
        const int64_t max_tokens = std::min(limits.max_tokens, spec_.max_tokens);
        std::vector<Encoded> encoded;
        for (const DecisionQuestion& q : request.questions)
            encoded.push_back(encode_question(request, q, tools, cfg_, max_tokens));
        return std::make_unique<LayaRun>(*this, request, std::move(encoded));
    }

    // Inputs of the step that reads `e` as a question of `type`.
    TensorMap inputs(const Encoded& e, QuestionType type) const {
        const auto tokens = static_cast<int64_t>(e.ids.size());
        HostTensor ids(DType::I32, {tokens}), positions(DType::I32, {tokens}), row(DType::I32, {1});
        for (int64_t t = 0; t < tokens; ++t) {
            ids.data<int32_t>()[t] = e.ids[static_cast<size_t>(t)];
            positions.data<int32_t>()[t] = static_cast<int32_t>(t);
        }
        row.data<int32_t>()[0] = type_row(type);
        TensorMap out;
        out.emplace("ids", std::move(ids));
        out.emplace("positions", std::move(positions));
        out.emplace("type", std::move(row));
        return out;
    }

    // The divisor of a question's logits: Laya's temperature for its type and option count, Julia's 1.
    double temperature(QuestionType type, size_t options) const {
        const Policy& p = cfg_.policy;
        const auto it = p.temperature_by_options.find(temperature_bucket(type, options));
        return it != p.temperature_by_options.end() ? it->second : p.temperature[static_cast<size_t>(type_row(type))];
    }

private:
    // nn.TransformerEncoderLayer(d, heads, 4 d, norm_first=True), ReLU, over the whole sequence and no positions.
    Value head_layer(Graph& g, Value x, const std::string& L) const {
        Graph::Scope scope(g, L.substr(0, L.size() - 1));
        const WeightStore& w = *weights_;
        const int64_t d = cfg_.encoder.hidden_size;
        const Value in = torch_nn::layer_norm(g, w, x, L + "norm1", kHeadNormEps);
        // nn.MultiheadAttention: one in_proj [3 d, d] with its bias, the query rows first, then key, then value.
        const Value qkv = g.add(g.matmul(in, g.weight(w, L + "self_attn.in_proj_weight")),
                                g.weight(w, L + "self_attn.in_proj_bias"));
        XS_CHECK(g.type(qkv).shape.back() == 3 * d, spec_.arch, ": ", L, "self_attn.in_proj_weight is not [3 x ", d,
                 ", ", d, "]");
        const auto part = [&g, qkv, d](int64_t k) {
            return blocks::Linear([&g, qkv, d, k](Value) { return g.slice(qkv, 1, k * d, (k + 1) * d); });
        };
        blocks::AttentionParts parts;
        parts.q = part(0);
        parts.k = part(1);
        parts.v = part(2);
        parts.o = [&g, &w, L](Value v) { return torch_nn::linear(g, w, v, L + "self_attn.out_proj"); };
        const blocks::AttentionSpec spec = encoder::attention_spec(heads_, d / heads_);
        x = g.add(x, blocks::attention(g, spec, parts, in, Value{}, Value{}).out);
        const float inf = std::numeric_limits<float>::infinity();
        const Value hidden =
            g.clamp(torch_nn::linear(g, w, torch_nn::layer_norm(g, w, x, L + "norm2", kHeadNormEps), L + "linear1"),
                    0.0f, inf);  // ReLU
        return g.add(x, torch_nn::linear(g, w, hidden, L + "linear2"));
    }

    // scorer = Sequential(LayerNorm, Linear(d, d), GELU, Linear(d, 1)): a logit per row, [T, 1].
    Value scorer(Graph& g, Value x) const {
        Graph::Scope scope(g, "scorer");
        const WeightStore& w = *weights_;
        const Value normed = torch_nn::layer_norm(g, w, x, "scorer.0", kHeadNormEps);
        const Value h = g.gelu(torch_nn::linear(g, w, normed, "scorer.1"));
        return torch_nn::linear(g, w, h, "scorer.3");
    }

    Config cfg_;
    DecisionSpec spec_;
    std::unique_ptr<WeightStore> weights_;
    int64_t heads_ = 1;
    Graph graph_;
};

std::optional<DecisionRun::Step> LayaRun::next(TensorMap& outputs) {
    if (next_ > 0) take(next_ - 1, outputs.at("scores"));
    if (next_ == encoded_.size()) return std::nullopt;
    Step step;
    step.program = 0;
    step.outputs = {"scores"};
    step.keep = 0;  // every sequence stands alone
    step.inputs = model_.inputs(encoded_[next_], request_.questions[next_].type);
    ++next_;
    return step;
}

// The logits at the question's markers over its temperature, through a softmax.
void LayaRun::take(size_t question, const HostTensor& scores) {
    const Encoded& e = encoded_[question];
    const DecisionQuestion& q = request_.questions[question];
    XS_CHECK(scores.dtype() == DType::F32 && scores.shape() == (Dims{static_cast<int64_t>(e.ids.size())}),
             "laya: the scores are ", dtype_name(scores.dtype()), dims_to_string(scores.shape()), ", expected [",
             e.ids.size(), "]");
    const double temperature = model_.temperature(q.type, e.markers.size());
    std::vector<double> p(e.markers.size());
    double top = -std::numeric_limits<double>::infinity(), total = 0;
    for (size_t o = 0; o < p.size(); ++o)
        top = std::max(top, p[o] = scores.data<float>()[e.markers[o]] / temperature);
    for (double& v : p)
        total += (v = std::exp(v - top));
    for (double& v : p)
        v /= total;
    // The prompt lists a noul's "false" first; the answer's order is true, false.
    if (q.type == QuestionType::Noul) std::swap(p[0], p[1]);
    probabilities_[question] = std::move(p);
}

}  // namespace

std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir) {
    Config cfg = read_config(dir);
    DecisionSpec spec;
    spec.id = cfg::model_id(dir);
    spec.arch = cfg.policy.family;
    spec.max_tokens = cfg.policy.max_tokens;
    spec.tokenizer_dir = "tokenizer";
    spec.chat_template = false;
    std::unique_ptr<WeightStore> weights = SafetensorsWeights::open_dir(dir);
    return std::make_unique<LayaModel>(std::move(cfg), std::move(spec), std::move(weights));
}

}  // namespace xs::laya
