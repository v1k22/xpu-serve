#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

#include "io/joined_weights.h"
#include "models/clef/clef.h"
#include "models/config_util.h"
#include "models/qwen3_5/qwen3_5_decoder.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::clef {
namespace {

using graph::Graph;
using graph::kT;
using graph::Value;

// Where the head's weights sit beside the decoder's.
constexpr const char* kHead = "joint_head.";
constexpr float kLayerNormEps = 1e-5f;  // torch.nn.LayerNorm's default

// The programs of one session.
enum Program : size_t { kEncode, kRoute, kFields, kScore };

// joint_head.safetensors as the head's programs read it: torch stores the three projections of an attention as one
// in_proj_weight [3 W, W] and in_proj_bias [3 W] (query rows first, then key, then value), and a program needs them one
// at a time (the key and value projections of the sequence, the query projection of the options).
class HeadWeights final : public WeightStore {
public:
    explicit HeadWeights(const std::filesystem::path& path) : file_(path) {
        for (const auto& [name, view] : file_.tensors()) {
            const bool weight = name.ends_with(".in_proj_weight"), bias = name.ends_with(".in_proj_bias");
            if (!weight && !bias) {
                views_[name] = view;
                continue;
            }
            XS_CHECK(view.data != nullptr && view.shape.size() == (weight ? 2u : 1u) && view.shape[0] % 3 == 0,
                     "clef: ", name, " has shape ", dims_to_string(view.shape), " and dtype ", dtype_name(view.dtype));
            const int64_t rows = view.shape[0] / 3;
            const size_t row_bytes = dtype_size(view.dtype) * static_cast<size_t>(weight ? view.shape[1] : 1);
            const std::string stem = name.substr(0, name.rfind('.') + 1);
            const char* part[] = {"q_proj", "k_proj", "v_proj"};
            for (int64_t i = 0; i < 3; ++i) {
                TensorView piece = view;
                piece.shape[0] = rows;
                piece.data = static_cast<const uint8_t*>(view.data) + static_cast<size_t>(i * rows) * row_bytes;
                views_[stem + part[i] + (weight ? ".weight" : ".bias")] = piece;
            }
        }
    }

    const TensorView* find(const std::string& name) const override {
        const auto it = views_.find(name);
        return it == views_.end() ? nullptr : &it->second;
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> out;
        for (const auto& [name, view] : views_)
            out.push_back(name);
        return out;
    }

private:
    SafetensorsFile file_;
    std::map<std::string, TensorView> views_;
};

float scalar(const WeightStore& weights, const std::string& name) {
    const TensorView& v = weights.get(name);
    XS_CHECK(v.numel() == 1 && is_float(v.dtype), "clef: ", name, " is not a float scalar");
    return HostTensor::borrow(v.dtype, v.shape, const_cast<void*>(v.data)).to_f32()[0];
}

// The row of the head's type embedding.
int32_t type_id(QuestionType type) {
    switch (type) {
        case QuestionType::Noul:
            return 0;
        case QuestionType::Choice:
            return 1;
        case QuestionType::Score:
            return 2;
    }
    fail("clef: unknown question type");
}

double dot(const float* a, const float* b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; ++i)
        s += static_cast<double>(a[i]) * b[i];
    return s;
}

// The mean of rows [span.begin, span.end) of the sequence, accumulated a step at a time.
struct SpanMean {
    Span span;
    std::vector<double> sum;

    void add(const float* rows, int64_t first, int64_t count, int64_t width) {
        const int64_t begin = std::max(span.begin, first), end = std::min(span.end, first + count);
        if (begin >= end) return;
        if (sum.empty()) sum.assign(static_cast<size_t>(width), 0.0);
        for (int64_t r = begin; r < end; ++r) {
            const float* row = rows + (r - first) * width;
            for (int64_t c = 0; c < width; ++c)
                sum[static_cast<size_t>(c)] += row[c];
        }
    }
    void mean(float* out, int64_t width) const {
        XS_CHECK(static_cast<int64_t>(sum.size()) == width && span.end > span.begin, "clef: an empty span");
        const double n = static_cast<double>(span.end - span.begin);
        for (int64_t c = 0; c < width; ++c)
            out[c] = static_cast<float>(sum[static_cast<size_t>(c)] / n);
    }
};

class ClefModel;

class ClefRun final : public DecisionRun {
public:
    ClefRun(const ClefModel& model, const DecisionRequest& request, EncodedRecord record, int64_t chunk);

    std::optional<Step> next(TensorMap& outputs) override;
    std::vector<std::vector<double>> probabilities() const override { return probabilities_; }
    const std::vector<std::vector<int32_t>>& sequences() const override { return sequences_; }

private:
    enum class Stage { Encode, Route, Fields, Score, Done };

    Step encode_step();
    void take_hidden(const HostTensor& hidden);
    Step route_step();
    Step fields_step(const TensorMap& outputs);
    Step score_step(const TensorMap& outputs);
    void finish(const TensorMap& outputs);

    const ClefModel& model_;
    EncodedRecord record_;
    std::vector<std::vector<int32_t>> sequences_;  // the record's ids, as DecisionRun reports them
    std::vector<int32_t> types_;                   // each question's type id (noul 0, choice 1, score 2)
    int64_t chunk_;
    Stage stage_ = Stage::Encode;
    int64_t fed_ = 0;   // tokens encoded so far
    int64_t step_ = 0;  // tokens of the encode step in flight
    int64_t options_ = 0;
    std::vector<int64_t> first_option_;  // each question's first row among all options
    std::vector<SpanMean> question_means_, option_means_;
    HostTensor question_, option_context_, lexical_;  // [Q, hidden], [O, hidden], [O, hidden]
    std::vector<float> global_;                       // the last token's state
    HostTensor routed_, option_, fields_;             // [O, W], [O, W] (normed), [Q, W]
    std::vector<std::vector<double>> probabilities_;
};

class ClefModel final : public DecisionModel {
public:
    ClefModel(qwen3_5::Config cfg, HeadConfig head, DecisionSpec spec, std::unique_ptr<WeightStore> weights)
        : cfg_(std::move(cfg)),
          head_(head),
          spec_(std::move(spec)),
          weights_(std::move(weights)),
          dec_(cfg_, *weights_) {
        XS_CHECK(head_.hidden_size == cfg_.hidden_size, "clef: the head reads states of ", head_.hidden_size,
                 ", the decoder's are ", cfg_.hidden_size);
        XS_CHECK(head_.width % head_.heads == 0, "clef: head width ", head_.width, " is not a multiple of its ",
                 head_.heads, " heads");
        build_encode(encode_);
        build_route(route_);
        build_fields(fields_);
        build_score(score_);
        prior_scale_ = std::exp(std::min(scalar(*weights_, name("prior_logit_scale")), std::log(100.0f)));
        joint_scale_ = std::exp(std::min(scalar(*weights_, name("joint_logit_scale")), std::log(100.0f)));
        residual_gate_ = 1.0f / (1.0f + std::exp(-scalar(*weights_, name("residual_gate"))));
        log::debug("clef: programs of ", encode_.nodes().size(), ", ", route_.nodes().size(), ", ",
                   fields_.nodes().size(), " and ", score_.nodes().size(), " nodes");
    }

    const DecisionSpec& spec() const override { return spec_; }
    const WeightStore& weights() const override { return *weights_; }
    std::vector<const graph::Graph*> programs() const override { return {&encode_, &route_, &fields_, &score_}; }
    std::vector<std::string> host_tables() const override { return {dec_.lm_head_name()}; }  // lexical()

    std::unique_ptr<DecisionRun> start(const DecisionRequest& request, const PromptTools& tools,
                                       const DecisionLimits& limits) const override {
        EncodedRecord record = encode_record(request, tools, std::min(limits.max_tokens, spec_.max_tokens));
        return std::make_unique<ClefRun>(*this, request, std::move(record), limits.chunk_tokens);
    }

    const qwen3_5::Decoder& decoder() const { return dec_; }
    const HeadConfig& head() const { return head_; }
    float prior_scale() const { return prior_scale_; }
    float joint_scale() const { return joint_scale_; }
    float residual_gate() const { return residual_gate_; }
    // The mean row of the output layer over `ids`: an option's lexical vector.
    void lexical(std::span<const int32_t> ids, float* out) const {
        const TensorView& table = weights_->get(dec_.lm_head_name());
        const int64_t width = cfg_.hidden_size;
        std::vector<float> row(static_cast<size_t>(width));
        std::vector<double> sum(static_cast<size_t>(width), 0.0);
        for (const int32_t id : ids) {
            quant::read_rows(table, id, 1, row.data());
            for (int64_t c = 0; c < width; ++c)
                sum[static_cast<size_t>(c)] += row[static_cast<size_t>(c)];
        }
        for (int64_t c = 0; c < width; ++c)
            out[c] = static_cast<float>(sum[static_cast<size_t>(c)] / static_cast<double>(ids.size()));
    }

private:
    static std::string name(const std::string& tensor) { return kHead + tensor; }
    Value weight(Graph& g, const std::string& tensor) const { return g.weight(*weights_, name(tensor)); }
    // torch.nn.Linear under `stem`: x @ weight^T + bias (when it has one).
    Value linear(Graph& g, Value x, const std::string& stem) const {
        const Value y = g.matmul(x, weight(g, stem + ".weight"));
        return weights_->contains(name(stem + ".bias")) ? g.add(y, weight(g, stem + ".bias")) : y;
    }
    Value layer_norm(Graph& g, Value x, const std::string& stem) const {
        return g.layer_norm(x, weight(g, stem + ".weight"), weight(g, stem + ".bias"), kLayerNormEps);
    }
    int64_t head_dim() const { return head_.width / head_.heads; }
    Value split_heads(Graph& g, Value x) const { return g.reshape(x, {kT, head_.heads, head_dim()}); }
    graph::AttentionAttrs attention_attrs() const {
        graph::AttentionAttrs a;
        a.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim())));
        a.causal = false;
        return a;
    }
    // The key and value projections of the sequence for the attention under `stem`, appended to a slot of its own.
    void append_memory(Graph& g, Value memory, const std::string& stem) const {
        const int slot = g.add_kv_slot(head_.heads, head_dim(), 0, name(stem));
        g.kv_cache(slot, split_heads(g, linear(g, memory, stem + ".k_proj")),
                   split_heads(g, linear(g, memory, stem + ".v_proj")));
    }
    // torch.nn.MultiheadAttention under `stem` from queries x over the sequence a program of the session encoded.
    Value attend_memory(Graph& g, Value x, const std::string& stem) const {
        const int slot = g.add_kv_slot(head_.heads, head_dim(), 0, name(stem));
        const auto [k, v] = g.kv_read(slot);
        const Value a = g.attention(split_heads(g, linear(g, x, stem + ".q_proj")), k, v, attention_attrs());
        return linear(g, g.reshape(a, {kT, head_.width}), stem + ".out_proj");
    }
    Value feedforward(Graph& g, Value x, const std::string& first, const std::string& second) const {
        return linear(g, g.gelu(linear(g, x, first)), second);
    }

    // The decoder over the prompt: the head's view of each token ("hidden"), and what its attentions read of the
    // sequence, kept for the programs after.
    void build_encode(Graph& g) const {
        const qwen3_5::Decoder::Inputs in = dec_.inputs(g);
        const Value states = dec_.final_norm(g, dec_.layers(g, in).back());
        Graph::Scope scope(g, "joint_head");
        const Value hidden = layer_norm(g, states, "hidden_norm");
        g.output("hidden", hidden);
        const Value memory = g.matmul(hidden, weight(g, "memory_projection.weight"));
        for (int64_t i = 0; i < head_.routing_layers; ++i) {
            const std::string L = "evidence_layers." + std::to_string(i);
            append_memory(g, layer_norm(g, memory, L + ".memory_norm"), L + ".attention");
        }
        for (int64_t i = 0; i < head_.layers; ++i) {
            append_memory(g, memory, "layers." + std::to_string(i) + ".multihead_attn");
        }
        g.validate();
    }

    // One row per option of the request: its query through the evidence layers. "base_field" is the option's
    // question projected as a field (the same row for every option of a question).
    void build_route(Graph& g) const {
        const Dims rows{kT, head_.hidden_size};
        const Value context = g.input("option_context", DType::F32, rows);
        const Value lexical = g.input("option_lexical", DType::F32, rows);
        const Value question = g.input("option_question", DType::F32, rows);
        Graph::Scope scope(g, "joint_head");
        Value x = g.add(g.add(g.matmul(context, weight(g, "option_context_projection.weight")),
                              g.matmul(lexical, weight(g, "option_lexical_projection.weight"))),
                        g.matmul(question, weight(g, "option_question_projection.weight")));
        for (int64_t i = 0; i < head_.routing_layers; ++i) {
            const std::string L = "evidence_layers." + std::to_string(i);
            Graph::Scope layer(g, L);
            x = g.add(x, attend_memory(g, layer_norm(g, x, L + ".query_norm"), L + ".attention"));
            x = g.add(x, feedforward(g, layer_norm(g, x, L + ".feedforward_norm"), L + ".feedforward.0",
                                     L + ".feedforward.3"));
        }
        g.output("routed", x);
        g.output("option", layer_norm(g, x, "option_norm"));
        g.output("base_field", g.matmul(question, weight(g, "question_projection.weight")));
        g.validate();
    }

    // One row per question: its field through the decoder layers (torch.nn.TransformerDecoderLayer, norm first).
    void build_fields(Graph& g) const {
        const Value base = g.input("base_field", DType::F32, {kT, head_.width});
        const Value summary = g.input("option_summary", DType::F32, {kT, head_.width});
        const Value global = g.input("global", DType::F32, {kT, head_.hidden_size});
        const Value types = g.input("type_ids", DType::I32, {kT});
        Graph::Scope scope(g, "joint_head");
        Value x = g.add(g.add(base, layer_norm(g, summary, "option_summary_norm")),
                        g.add(g.matmul(global, weight(g, "global_projection.weight")),
                              g.gather(weight(g, "type_embedding.weight"), types)));
        for (int64_t i = 0; i < head_.layers; ++i) {
            const std::string L = "layers." + std::to_string(i);
            Graph::Scope layer(g, L);
            const Value n = layer_norm(g, x, L + ".norm1");
            const std::string self = L + ".self_attn";
            const Value a = g.attention(split_heads(g, linear(g, n, self + ".q_proj")),
                                        split_heads(g, linear(g, n, self + ".k_proj")),
                                        split_heads(g, linear(g, n, self + ".v_proj")), attention_attrs());
            x = g.add(x, linear(g, g.reshape(a, {kT, head_.width}), self + ".out_proj"));
            x = g.add(x, attend_memory(g, layer_norm(g, x, L + ".norm2"), L + ".multihead_attn"));
            x = g.add(x, feedforward(g, layer_norm(g, x, L + ".norm3"), L + ".linear1", L + ".linear2"));
        }
        g.output("fields", layer_norm(g, x, "field_norm"));
        g.validate();
    }

    // One row per option: the residual score of its routed state against its question's field.
    void build_score(Graph& g) const {
        const Value field = g.input("field", DType::F32, {kT, head_.width});
        const Value option = g.input("option", DType::F32, {kT, head_.width});
        Graph::Scope scope(g, "joint_head");
        const float inf = std::numeric_limits<float>::infinity();
        const Value diff = g.add(field, g.scale(option, -1.0f));
        const Value distance =
            g.add(g.clamp(diff, 0.0f, inf), g.clamp(g.scale(diff, -1.0f), 0.0f, inf));  // |field - option|
        const Value features = g.concat({field, option, g.mul(field, option), distance}, 1);
        g.output("residual", feedforward(g, features, "residual_scorer.0", "residual_scorer.3"));
        g.validate();
    }

    qwen3_5::Config cfg_;
    HeadConfig head_;
    DecisionSpec spec_;
    std::unique_ptr<WeightStore> weights_;
    qwen3_5::Decoder dec_;
    Graph encode_, route_, fields_, score_;
    float prior_scale_ = 1.0f, joint_scale_ = 1.0f, residual_gate_ = 0.5f;
};

ClefRun::ClefRun(const ClefModel& model, const DecisionRequest& request, EncodedRecord record, int64_t chunk)
    : model_(model), record_(std::move(record)), chunk_(std::max<int64_t>(chunk, 1)) {
    sequences_.push_back(record_.ids);
    const int64_t hidden = model_.head().hidden_size;
    for (size_t q = 0; q < record_.questions.size(); ++q) {
        const EncodedQuestion& e = record_.questions[q];
        types_.push_back(type_id(request.questions[q].type));
        first_option_.push_back(options_);
        question_means_.push_back({e.instruction, {}});
        for (const Span& s : e.options)
            option_means_.push_back({s, {}});
        options_ += static_cast<int64_t>(e.options.size());
    }
    // An option's lexical vector needs no state: the mean output-layer row of its tokens.
    lexical_ = HostTensor(DType::F32, {options_, hidden});
    int64_t row = 0;
    for (const EncodedQuestion& e : record_.questions) {
        for (const Span& s : e.options) {
            model_.lexical(std::span<const int32_t>(record_.ids)
                               .subspan(static_cast<size_t>(s.begin), static_cast<size_t>(s.end - s.begin)),
                           lexical_.data<float>() + row++ * hidden);
        }
    }
}

std::optional<DecisionRun::Step> ClefRun::next(TensorMap& outputs) {
    switch (stage_) {
        case Stage::Encode:
            if (step_ > 0) take_hidden(outputs.at("hidden"));
            if (fed_ < static_cast<int64_t>(record_.ids.size())) return encode_step();
            stage_ = Stage::Route;
            return route_step();
        case Stage::Route:
            stage_ = Stage::Fields;
            return fields_step(outputs);
        case Stage::Fields:
            stage_ = Stage::Score;
            return score_step(outputs);
        case Stage::Score:
            finish(outputs);
            stage_ = Stage::Done;
            return std::nullopt;
        case Stage::Done:
            break;
    }
    return std::nullopt;
}

DecisionRun::Step ClefRun::encode_step() {
    Step step;
    step.program = kEncode;
    step.outputs = {"hidden"};
    step.keep = fed_ == 0 ? 0 : -1;
    step_ = std::min<int64_t>(chunk_, static_cast<int64_t>(record_.ids.size()) - fed_);
    model_.decoder().token_inputs(
        std::span<const int32_t>(record_.ids).subspan(static_cast<size_t>(fed_), static_cast<size_t>(step_)), fed_,
        step.inputs);
    return step;
}

void ClefRun::take_hidden(const HostTensor& hidden) {
    const int64_t width = model_.head().hidden_size;
    XS_CHECK(hidden.dtype() == DType::F32 && hidden.shape() == (Dims{step_, width}), "clef: hidden output is ",
             dtype_name(hidden.dtype()), dims_to_string(hidden.shape()), ", expected [", step_, ", ", width, "]");
    const float* rows = hidden.data<float>();
    for (SpanMean& m : question_means_)
        m.add(rows, fed_, step_, width);
    for (SpanMean& m : option_means_)
        m.add(rows, fed_, step_, width);
    fed_ += step_;
    if (fed_ == static_cast<int64_t>(record_.ids.size()))
        global_.assign(rows + (step_ - 1) * width, rows + step_ * width);
    step_ = 0;
}

DecisionRun::Step ClefRun::route_step() {
    const int64_t width = model_.head().hidden_size;
    const auto questions = static_cast<int64_t>(record_.questions.size());
    question_ = HostTensor(DType::F32, {questions, width});
    for (int64_t q = 0; q < questions; ++q)
        question_means_[static_cast<size_t>(q)].mean(question_.data<float>() + q * width, width);
    option_context_ = HostTensor(DType::F32, {options_, width});
    HostTensor option_question(DType::F32, {options_, width});
    for (int64_t q = 0; q < questions; ++q) {
        const auto count = static_cast<int64_t>(record_.questions[static_cast<size_t>(q)].options.size());
        for (int64_t o = first_option_[static_cast<size_t>(q)]; o < first_option_[static_cast<size_t>(q)] + count;
             ++o) {
            option_means_[static_cast<size_t>(o)].mean(option_context_.data<float>() + o * width, width);
            std::copy_n(question_.data<float>() + q * width, width, option_question.data<float>() + o * width);
        }
    }
    Step step;
    step.program = kRoute;
    step.outputs = {"routed", "option", "base_field"};
    step.inputs["option_context"] = option_context_.clone();
    step.inputs["option_lexical"] = lexical_.clone();
    step.inputs["option_question"] = std::move(option_question);
    return step;
}

// Each question's summary of its options: their routed states weighted by a softmax of their match with the field.
DecisionRun::Step ClefRun::fields_step(const TensorMap& outputs) {
    const int64_t width = model_.head().width, hidden = model_.head().hidden_size;
    const auto questions = static_cast<int64_t>(record_.questions.size());
    routed_ = outputs.at("routed").clone();
    option_ = outputs.at("option").clone();
    const HostTensor& base = outputs.at("base_field");
    XS_CHECK(routed_.shape() == (Dims{options_, width}) && option_.shape() == routed_.shape() &&
                 base.shape() == routed_.shape(),
             "clef: the route program gave ", dims_to_string(routed_.shape()), ", expected [", options_, ", ", width,
             "]");
    Step step;
    step.program = kFields;
    step.outputs = {"fields"};
    HostTensor base_field(DType::F32, {questions, width}), summary(DType::F32, {questions, width});
    HostTensor global(DType::F32, {questions, hidden}), types(DType::I32, {questions});
    for (int64_t q = 0; q < questions; ++q) {
        const int64_t first = first_option_[static_cast<size_t>(q)];
        const auto count = static_cast<int64_t>(record_.questions[static_cast<size_t>(q)].options.size());
        const float* field = base.data<float>() + first * width;
        std::copy_n(field, width, base_field.data<float>() + q * width);
        std::vector<double> w(static_cast<size_t>(count));
        double top = -std::numeric_limits<double>::infinity();
        for (int64_t o = 0; o < count; ++o) {
            w[static_cast<size_t>(o)] =
                dot(routed_.data<float>() + (first + o) * width, field, width) / std::sqrt(static_cast<double>(width));
            top = std::max(top, w[static_cast<size_t>(o)]);
        }
        double total = 0;
        for (double& v : w)
            total += (v = std::exp(v - top));
        std::vector<double> sum(static_cast<size_t>(width), 0.0);
        for (int64_t o = 0; o < count; ++o) {
            const float* row = routed_.data<float>() + (first + o) * width;
            const double weight = w[static_cast<size_t>(o)] / total;
            for (int64_t c = 0; c < width; ++c)
                sum[static_cast<size_t>(c)] += weight * row[c];
        }
        for (int64_t c = 0; c < width; ++c)
            summary.data<float>()[q * width + c] = static_cast<float>(sum[static_cast<size_t>(c)]);
        std::copy(global_.begin(), global_.end(), global.data<float>() + q * hidden);
        types.data<int32_t>()[q] = types_[static_cast<size_t>(q)];
    }
    step.inputs["base_field"] = std::move(base_field);
    step.inputs["option_summary"] = std::move(summary);
    step.inputs["global"] = std::move(global);
    step.inputs["type_ids"] = std::move(types);
    return step;
}

DecisionRun::Step ClefRun::score_step(const TensorMap& outputs) {
    const int64_t width = model_.head().width;
    const auto questions = static_cast<int64_t>(record_.questions.size());
    fields_ = outputs.at("fields").clone();
    XS_CHECK(fields_.shape() == (Dims{questions, width}), "clef: the fields program gave ",
             dims_to_string(fields_.shape()));
    HostTensor field(DType::F32, {options_, width});
    for (int64_t q = 0; q < questions; ++q) {
        const auto count = static_cast<int64_t>(record_.questions[static_cast<size_t>(q)].options.size());
        for (int64_t o = first_option_[static_cast<size_t>(q)]; o < first_option_[static_cast<size_t>(q)] + count;
             ++o) {
            std::copy_n(fields_.data<float>() + q * width, width, field.data<float>() + o * width);
        }
    }
    Step step;
    step.program = kScore;
    step.outputs = {"residual"};
    step.inputs["field"] = std::move(field);
    step.inputs["option"] = option_.clone();
    return step;
}

// logit = prior + gate * (joint_scale * cos(field, option) + residual), the prior being the cosine of the option's
// lexical vector with its question's state plus the last token's, times prior_scale. A softmax per question.
void ClefRun::finish(const TensorMap& outputs) {
    const int64_t width = model_.head().width, hidden = model_.head().hidden_size;
    const HostTensor& residual = outputs.at("residual");
    XS_CHECK(residual.numel() == options_, "clef: the score program gave ", dims_to_string(residual.shape()));
    const auto norm = [](const float* x, int64_t n) { return std::sqrt(dot(x, x, n)); };
    for (size_t q = 0; q < record_.questions.size(); ++q) {
        const EncodedQuestion& e = record_.questions[q];
        const int64_t first = first_option_[q];
        std::vector<float> anchor(static_cast<size_t>(hidden));
        const float* question = question_.data<float>() + static_cast<int64_t>(q) * hidden;
        for (int64_t c = 0; c < hidden; ++c)
            anchor[static_cast<size_t>(c)] = question[c] + global_[static_cast<size_t>(c)];
        const double anchor_norm = std::max(norm(anchor.data(), hidden), 1e-12);
        const float* field = fields_.data<float>() + static_cast<int64_t>(q) * width;
        const double field_norm = norm(field, width);
        std::vector<double> logits(e.options.size());
        double top = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < e.options.size(); ++j) {
            const int64_t o = first + static_cast<int64_t>(j);
            const float* lexical = lexical_.data<float>() + o * hidden;
            const double prior = model_.prior_scale() * dot(lexical, anchor.data(), hidden) /
                                 (std::max(norm(lexical, hidden), 1e-12) * anchor_norm);
            const float* option = option_.data<float>() + o * width;
            const double cosine = dot(field, option, width) / std::max(field_norm * norm(option, width), 1e-8);
            const double joint = model_.joint_scale() * cosine + residual.data<float>()[o];
            logits[j] = prior + model_.residual_gate() * joint;
            top = std::max(top, logits[j]);
        }
        double total = 0;
        for (double& v : logits)
            total += (v = std::exp(v - top));
        std::vector<double> p(e.options.size());
        for (size_t j = 0; j < e.options.size(); ++j)
            p[static_cast<size_t>(e.option[j])] = logits[j] / total;
        probabilities_.push_back(std::move(p));
    }
}

}  // namespace

HeadConfig parse_head_config(const nlohmann::json& j) {
    const std::string where = "joint_head_config.json";
    HeadConfig c;
    c.hidden_size = cfg::get_int(j, "hidden_size", where);
    c.width = cfg::get_int(j, "width", where);
    c.routing_layers = cfg::get_int(j, "routing_layers", where);
    c.layers = cfg::get_int(j, "layers", where);
    c.heads = cfg::get_int(j, "heads", where);
    c.feedforward = cfg::get_int(j, "feedforward", where);
    XS_CHECK(
        c.hidden_size > 0 && c.width > 0 && c.routing_layers >= 0 && c.layers >= 0 && c.heads > 0 && c.feedforward > 0,
        "clef: joint_head_config.json has a value out of range");
    return c;
}

bool is_checkpoint(const std::filesystem::path& dir) {
    return std::filesystem::exists(dir / "joint_head_config.json");
}

std::unique_ptr<DecisionModel> load(const std::filesystem::path& dir, const nlohmann::json& config_json) {
    qwen3_5::Config cfg = qwen3_5::parse_config(config_json);
    XS_CHECK(cfg.num_experts == 0, "clef: a mixture-of-experts decoder is not supported");
    const HeadConfig head = parse_head_config(cfg::read_json(dir / "joint_head_config.json"));
    DecisionSpec spec;
    spec.id = cfg::model_id(dir);
    spec.arch = "clef";
    spec.max_tokens = cfg.max_position;
    // A checkpoint keeps the head in a file of its own; a pack holds it beside the decoder, under the names the
    // programs read.
    std::unique_ptr<WeightStore> weights = SafetensorsWeights::open_dir(dir);
    if (!weights->contains(std::string(kHead) + "hidden_norm.weight")) {
        const std::filesystem::path head_file = dir / "joint_head.safetensors";
        XS_CHECK(std::filesystem::exists(head_file), "clef: ", head_file.string(), " is missing");
        weights = std::make_unique<JoinedWeights>(std::move(weights), std::make_unique<HeadWeights>(head_file), kHead);
    }
    return std::make_unique<ClefModel>(std::move(cfg), head, std::move(spec), std::move(weights));
}

}  // namespace xs::clef
