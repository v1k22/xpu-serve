#include "backend/openvino/ov_translate.h"

#include <algorithm>
#include <map>

#include <openvino/op/ops.hpp>
#include <openvino/op/util/variable.hpp>

#include "util/error.h"

namespace xs::ov_backend {

namespace ops = ov::op;
using graph::Node;
using graph::OpKind;
using graph::Value;

namespace {

// A graph value after lowering. KvCache outputs start out only in the attention layout [1, H, L, D]; the logical
// [L, H, D] value is built on first use by a non-Attention op.
struct Lowered {
    OvOutput logical;
    OvOutput heads;
};

bool is_float(const OvOutput& x) {
    return x.get_element_type().is_real();
}

OvOutput as_f32(const OvOutput& x) {
    if (x.get_element_type() == ov::element::f32) return x;
    return std::make_shared<ops::v0::Convert>(x, ov::element::f32);
}

// [T, H, D] -> [1, H, T, D]
OvOutput to_heads(const OvOutput& x, const std::string& label) {
    const OvOutput batched = std::make_shared<ops::v0::Unsqueeze>(x, i64_scalar(0));
    return named(std::make_shared<ops::v1::Transpose>(batched, i64_constant({0, 2, 1, 3})), label, "to_heads");
}

// [1, H, T, D] -> [T, H, D]
OvOutput from_heads(const OvOutput& x, const std::string& label) {
    const OvOutput transposed = std::make_shared<ops::v1::Transpose>(x, i64_constant({0, 2, 1, 3}));
    return named(std::make_shared<ops::v0::Squeeze>(transposed, i64_constant({0})), label, "from_heads");
}

OvOutput dim_scalar(const OvOutput& x, int64_t axis) {
    const OvOutput shape = std::make_shared<ops::v3::ShapeOf>(x, ov::element::i64);
    return std::make_shared<ops::v8::Gather>(shape, i64_scalar(axis), i64_scalar(0));
}

class Translator {
public:
    Translator(const graph::Graph& graph, const WeightStore& weights, const TranslateOptions& options)
        : graph_(graph), weights_(weights), options_(options), values_(graph.nodes().size()) {}

    Translation run();

private:
    void lower(size_t index);
    OvOutput lower_input(const Node& n);
    OvOutput lower_rms_norm(const Node& n, const std::string& label);
    OvOutput lower_reshape(const Node& n, const std::string& label);
    OvOutput lower_slice(const Node& n);
    OvOutput lower_rope(const Node& n, const std::string& label);
    void lower_kv_cache(const Node& n, size_t index, const std::string& label);
    OvOutput lower_attention(const Node& n, const std::string& label);

    OvOutput value(Value v);
    OvOutput float_value(Value v) { return as_f32(value(v)); }
    OvOutput heads_value(Value v, const std::string& label);
    OvOutput symbol(int64_t dim) const;
    OvOutput mask(int64_t sliding_window);

    const graph::Graph& graph_;
    const WeightStore& weights_;
    const TranslateOptions options_;
    std::vector<std::vector<Lowered>> values_;
    ov::ParameterVector parameters_;
    ov::SinkVector sinks_;
    std::vector<InputSpec> input_specs_;
    WeightBytes weight_bytes_;
    OvOutput tokens_;  // scalar i64: T
    OvOutput cached_;  // scalar i64: L
    std::map<int64_t, OvOutput> masks_;
};

Translation Translator::run() {
    graph_.validate();
    for (size_t i = 0; i < graph_.nodes().size(); ++i) {
        const Node& n = graph_.nodes()[i];
        try {
            lower(i);
        } catch (const ov::Exception& e) {
            fail("openvino: lowering node ", i, " (", n.label, ": ", graph::op_name(n.kind), ") failed: ", e.what());
        }
    }
    ov::ResultVector results;
    std::vector<std::string> output_names;
    for (const graph::GraphOutput& out : graph_.outputs()) {
        const OvOutput x = value(out.value);
        const auto result = std::make_shared<ops::v0::Result>(is_float(x) ? as_f32(x) : x);
        result->set_friendly_name(out.name);
        results.push_back(result);
        output_names.push_back(out.name);
    }
    Translation t;
    t.stateful = !graph_.kv_slots().empty();
    t.model = std::make_shared<ov::Model>(results, sinks_, parameters_, "xs_step");
    t.inputs = std::move(input_specs_);
    t.outputs = std::move(output_names);
    t.weight_bytes = weight_bytes_;
    return t;
}

void Translator::lower(size_t index) {
    const Node& n = graph_.nodes()[index];
    const std::string label = n.label.empty() ? str_cat(graph::op_name(n.kind), '_', index) : n.label;
    auto set = [&](const OvOutput& out) { values_[index] = {Lowered{out, {}}}; };
    switch (n.kind) {
        case OpKind::Input:
            set(lower_input(n));
            break;
        case OpKind::Weight: {
            const std::string& name = std::get<graph::WeightAttrs>(n.attrs).name;
            set(make_weight(weights_.get(name), options_.precision, name, weight_bytes_));
            break;
        }
        case OpKind::Constant:
            set(make_constant(std::get<graph::ConstantAttrs>(n.attrs).value, label));
            break;
        case OpKind::Gather:
            set(named(std::make_shared<ops::v8::Gather>(float_value(n.inputs[0]), value(n.inputs[1]), i64_scalar(0)),
                      label));
            break;
        case OpKind::MatMul:
            set(named(
                std::make_shared<ops::v0::MatMul>(float_value(n.inputs[0]), float_value(n.inputs[1]), false, true),
                label));
            break;
        case OpKind::Add:
            set(named(std::make_shared<ops::v1::Add>(float_value(n.inputs[0]), float_value(n.inputs[1])), label));
            break;
        case OpKind::Mul:
            set(named(std::make_shared<ops::v1::Multiply>(float_value(n.inputs[0]), float_value(n.inputs[1])), label));
            break;
        case OpKind::Scale: {
            const float factor = std::get<graph::ScaleAttrs>(n.attrs).factor;
            set(named(std::make_shared<ops::v1::Multiply>(float_value(n.inputs[0]), f32_scalar(factor)), label));
            break;
        }
        case OpKind::Tanh:
            set(named(std::make_shared<ops::v0::Tanh>(float_value(n.inputs[0])), label));
            break;
        case OpKind::GeluTanh:
            set(named(std::make_shared<ops::v7::Gelu>(float_value(n.inputs[0]), ops::GeluApproximationMode::TANH),
                      label));
            break;
        case OpKind::Softcap: {
            const float cap = std::get<graph::SoftcapAttrs>(n.attrs).cap;
            const auto divided = std::make_shared<ops::v1::Divide>(float_value(n.inputs[0]), f32_scalar(cap));
            const auto tanh = std::make_shared<ops::v0::Tanh>(divided);
            set(named(std::make_shared<ops::v1::Multiply>(tanh, f32_scalar(cap)), label));
            break;
        }
        case OpKind::Clamp: {
            const auto& a = std::get<graph::ClampAttrs>(n.attrs);
            set(named(std::make_shared<ops::v0::Clamp>(float_value(n.inputs[0]), static_cast<double>(a.lo),
                                                       static_cast<double>(a.hi)),
                      label));
            break;
        }
        case OpKind::RmsNorm:
            set(lower_rms_norm(n, label));
            break;
        case OpKind::Reshape:
            set(lower_reshape(n, label));
            break;
        case OpKind::Slice:
            set(named(lower_slice(n).get_node_shared_ptr(), label));
            break;
        case OpKind::Transpose: {
            const auto& perm = std::get<graph::TransposeAttrs>(n.attrs).perm;
            set(named(std::make_shared<ops::v1::Transpose>(value(n.inputs[0]),
                                                           i64_constant(std::vector<int64_t>(perm.begin(), perm.end()))),
                      label));
            break;
        }
        case OpKind::Rope:
            set(lower_rope(n, label));
            break;
        case OpKind::KvCache:
            lower_kv_cache(n, index, label);
            break;
        case OpKind::Attention:
            set(lower_attention(n, label));
            break;
    }
}

OvOutput Translator::lower_input(const Node& n) {
    const std::string& name = std::get<graph::InputAttrs>(n.attrs).name;
    const graph::TensorType& type = n.outputs[0];
    ov::PartialShape shape;
    for (int64_t d : type.shape) {
        XS_CHECK(d != graph::kL, "graph input '", name, "' uses the symbolic dim L, which only KV state may have");
        shape.push_back(d == graph::kT ? ov::Dimension::dynamic() : ov::Dimension(d));
    }
    const auto param = std::make_shared<ops::v0::Parameter>(to_ov_type(type.dtype), shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    parameters_.push_back(param);
    input_specs_.push_back(InputSpec{name, type.dtype, type.shape});
    for (size_t axis = 0; axis < type.shape.size() && !tokens_.get_node(); ++axis) {
        if (type.shape[axis] == graph::kT) tokens_ = dim_scalar(param, static_cast<int64_t>(axis));
    }
    return param;
}

// x * 1 / sqrt(mean(x^2) + eps) * w, spelled the way the plugins' RMS fusion matches it: the fused GPU kernel
// accumulates in f32 even when activations are f16.
OvOutput Translator::lower_rms_norm(const Node& n, const std::string& label) {
    const OvOutput x = float_value(n.inputs[0]);
    const float eps = std::get<graph::RmsNormAttrs>(n.attrs).eps;
    const auto squared = std::make_shared<ops::v1::Power>(x, f32_scalar(2.0f));
    const auto mean = std::make_shared<ops::v1::ReduceMean>(squared, i64_constant({-1}), true);
    const auto with_eps = std::make_shared<ops::v1::Add>(mean, f32_scalar(eps));
    const auto root = std::make_shared<ops::v0::Sqrt>(with_eps);
    const auto inverse = std::make_shared<ops::v1::Divide>(f32_scalar(1.0f), root);
    const auto normalized = std::make_shared<ops::v1::Multiply>(x, inverse);
    OvOutput weight;
    if (n.inputs.size() > 1) {
        weight = float_value(n.inputs[1]);
    } else {
        const int64_t dim = n.outputs[0].shape.back();
        if (dim < 0) return named(normalized, label);
        weight = std::make_shared<ops::v0::Constant>(ov::element::f32, ov::Shape{static_cast<size_t>(dim)}, 1.0f);
    }
    return named(std::make_shared<ops::v1::Multiply>(weight, normalized), label);
}

OvOutput Translator::lower_reshape(const Node& n, const std::string& label) {
    const Dims& target = std::get<graph::ReshapeAttrs>(n.attrs).shape;
    const auto symbols = std::count_if(target.begin(), target.end(), [](int64_t d) { return d < 0; });
    OvOutput shape;
    if (symbols <= 1) {  // one unknown dim: let Reshape infer it
        std::vector<int64_t> dims(target.begin(), target.end());
        for (int64_t& d : dims)
            d = d < 0 ? -1 : d;
        shape = i64_constant(dims);
    } else {
        ov::OutputVector parts;
        for (int64_t d : target) {
            parts.push_back(d < 0 ? std::make_shared<ops::v0::Unsqueeze>(symbol(d), i64_scalar(0))->output(0)
                                  : i64_constant({d}));
        }
        shape = std::make_shared<ops::v0::Concat>(parts, 0);
    }
    return named(std::make_shared<ops::v1::Reshape>(value(n.inputs[0]), shape, false), label);
}

OvOutput Translator::lower_slice(const Node& n) {
    const auto& a = std::get<graph::SliceAttrs>(n.attrs);
    return std::make_shared<ops::v8::Slice>(value(n.inputs[0]), i64_constant({a.begin}), i64_constant({a.end}),
                                            i64_constant({1}), i64_constant({a.axis}));
}

// NeoX rotate-half in the Hugging Face form: x * cos + rotate_half(x) * sin, per section (one section unless the
// positions are [T, S]). Within a section of width W, rotate_half(x) = [-x2, x1] and the W/2 frequencies repeat
// twice. The angle path stays in f32: positions times inverse frequencies reach tens of thousands of radians, where
// f16 has no fractional bits left. rotate_half is a Gather times a sign rather than a Concat, because the GPU plugin's
// activation scaling mishandles a Concat fed by f32-kept and f16 branches.
OvOutput Translator::lower_rope(const Node& n, const std::string& label) {
    const OvOutput x = float_value(n.inputs[0]);
    const int64_t dim = n.outputs[0].shape[2];
    const Dims& pos_shape = graph_.type(n.inputs[1]).shape;
    const int64_t sections = pos_shape.size() == 2 ? pos_shape[1] : 1;
    const int64_t width = dim / sections;
    const int64_t half = width / 2;

    // angles[t, s, j] = positions[t, s] * freq[j] with freq = [inv_freq, inv_freq], then [T, 1, D] section-major.
    const OvOutput inv_freq = float_value(n.inputs[2]);
    const auto freq = std::make_shared<ops::v0::Concat>(ov::OutputVector{inv_freq, inv_freq}, 0);  // [W]
    const auto positions = std::make_shared<ops::v0::Convert>(value(n.inputs[1]), ov::element::f32);
    const auto positions_3d =
        std::make_shared<ops::v1::Reshape>(positions, i64_constant({-1, sections, 1}), false);  // [T, S, 1]
    const auto angles_3d = std::make_shared<ops::v1::Multiply>(positions_3d, freq);              // [T, S, W]
    const auto angles = std::make_shared<ops::v1::Reshape>(angles_3d, i64_constant({-1, 1, dim}), false);
    const auto cos = std::make_shared<ops::v0::Cos>(angles);
    const auto sin = std::make_shared<ops::v0::Sin>(angles);
    for (const std::shared_ptr<ov::Node>& node :
         {inv_freq.get_node_shared_ptr(), std::shared_ptr<ov::Node>(freq), std::shared_ptr<ov::Node>(positions),
          std::shared_ptr<ov::Node>(positions_3d), std::shared_ptr<ov::Node>(angles_3d),
          std::shared_ptr<ov::Node>(angles), std::shared_ptr<ov::Node>(cos), std::shared_ptr<ov::Node>(sin)}) {
        keep_in_f32(node);
    }
    named(cos, label, "cos");
    named(sin, label, "sin");

    std::vector<int64_t> swap_halves(static_cast<size_t>(dim));
    std::vector<float> sign(static_cast<size_t>(dim));
    for (int64_t c = 0; c < dim; ++c) {
        const int64_t base = c / width * width;
        const int64_t i = c - base;
        swap_halves[static_cast<size_t>(c)] = base + (i + half) % width;
        sign[static_cast<size_t>(c)] = i < half ? -1.0f : 1.0f;
    }
    const auto swapped = std::make_shared<ops::v8::Gather>(x, i64_constant(swap_halves), i64_scalar(-1));
    const auto sign_const =
        std::make_shared<ops::v0::Constant>(ov::element::f32, ov::Shape{static_cast<size_t>(dim)}, sign);
    const auto rotated = std::make_shared<ops::v1::Multiply>(swapped, sign_const);
    const auto x_cos = std::make_shared<ops::v1::Multiply>(x, cos);
    const auto rotated_sin = std::make_shared<ops::v1::Multiply>(rotated, sin);
    return named(std::make_shared<ops::v1::Add>(x_cos, rotated_sin), label);
}

void Translator::lower_kv_cache(const Node& n, size_t index, const std::string& label) {
    const int slot = std::get<graph::KvCacheAttrs>(n.attrs).slot;
    const graph::KvSlot& spec = graph_.kv_slots().at(static_cast<size_t>(slot));
    const ov::PartialShape state_shape{1, spec.num_kv_heads, ov::Dimension::dynamic(), spec.head_dim};
    std::vector<Lowered> outputs;
    for (int which = 0; which < 2; ++which) {
        const std::string part = which == 0 ? "k" : "v";
        const auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{state_shape, ov::element::f32, str_cat("kv", slot, '.', part)});
        const auto past = std::make_shared<ops::v6::ReadValue>(variable);
        named(past, label, part + "_past");
        const OvOutput fresh = to_heads(float_value(n.inputs[static_cast<size_t>(which)]), label + "/" + part);
        const auto all = std::make_shared<ops::v0::Concat>(ov::OutputVector{past, fresh}, 2);
        named(all, label, part + "_all");
        const auto assign = std::make_shared<ops::v6::Assign>(all, variable);
        named(assign, label, part + "_assign");
        sinks_.push_back(assign);
        outputs.push_back(Lowered{{}, all});
    }
    if (!cached_.get_node()) cached_ = dim_scalar(outputs[0].heads, 2);
    values_[index] = std::move(outputs);
}

OvOutput Translator::lower_attention(const Node& n, const std::string& label) {
    const auto& attrs = std::get<graph::AttentionAttrs>(n.attrs);
    const Dims& q_shape = n.outputs[0].shape;
    AttentionOperands in;
    in.q = to_heads(float_value(n.inputs[0]), label + "/q");
    in.k = heads_value(n.inputs[1], label + "/k");
    in.v = heads_value(n.inputs[2], label + "/v");
    if (attrs.causal) in.mask = mask(attrs.sliding_window);
    in.num_heads = q_shape[1];
    in.num_kv_heads = graph_.type(n.inputs[1]).shape[1];
    in.head_dim = q_shape[2];
    return from_heads(attention(in, attrs, options_.attention, label), label);
}

OvOutput Translator::value(Value v) {
    Lowered& slot = values_.at(static_cast<size_t>(v.node)).at(static_cast<size_t>(v.index));
    if (!slot.logical.get_node()) {
        XS_CHECK(slot.heads.get_node(), "graph value of node ", v.node, " used before it was lowered");
        slot.logical = from_heads(slot.heads, graph_.node(v).label + "/logical");
    }
    return slot.logical;
}

OvOutput Translator::heads_value(Value v, const std::string& label) {
    const Lowered& slot = values_.at(static_cast<size_t>(v.node)).at(static_cast<size_t>(v.index));
    if (slot.heads.get_node()) return slot.heads;
    return to_heads(float_value(v), label);
}

OvOutput Translator::symbol(int64_t dim) const {
    const OvOutput& out = dim == graph::kT ? tokens_ : cached_;
    XS_CHECK(out.get_node(), "symbolic dim ", graph::dim_to_string(dim), " is used before any ",
             dim == graph::kT ? "input" : "KvCache node", " defines it");
    return out;
}

OvOutput Translator::mask(int64_t sliding_window) {
    auto it = masks_.find(sliding_window);
    if (it == masks_.end()) {
        it = masks_
                 .emplace(sliding_window,
                          attention_mask(symbol(graph::kT), symbol(graph::kL), sliding_window, "attention"))
                 .first;
    }
    return it->second;
}

}  // namespace

Translation translate(const graph::Graph& graph, const WeightStore& weights, const TranslateOptions& options) {
    return Translator(graph, weights, options).run();
}

}  // namespace xs::ov_backend
