#include "graph/graph.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

#include "util/error.h"

namespace xs::graph {

std::string dim_to_string(int64_t d) {
    if (d == kT) return "T";
    if (d == kL) return "L";
    return std::to_string(d);
}

std::string shape_to_string(const Dims& dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += dim_to_string(dims[i]);
    }
    return s + "]";
}

std::string_view op_name(OpKind kind) {
    switch (kind) {
        case OpKind::Input: return "Input";
        case OpKind::Weight: return "Weight";
        case OpKind::Constant: return "Constant";
        case OpKind::Gather: return "Gather";
        case OpKind::MatMul: return "MatMul";
        case OpKind::Add: return "Add";
        case OpKind::Mul: return "Mul";
        case OpKind::Scale: return "Scale";
        case OpKind::Tanh: return "Tanh";
        case OpKind::GeluTanh: return "GeluTanh";
        case OpKind::Softcap: return "Softcap";
        case OpKind::Sigmoid: return "Sigmoid";
        case OpKind::Silu: return "Silu";
        case OpKind::Softplus: return "Softplus";
        case OpKind::Clamp: return "Clamp";
        case OpKind::RmsNorm: return "RmsNorm";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Slice: return "Slice";
        case OpKind::Transpose: return "Transpose";
        case OpKind::Rope: return "Rope";
        case OpKind::KvCache: return "KvCache";
        case OpKind::KvRead: return "KvRead";
        case OpKind::Attention: return "Attention";
        case OpKind::CausalConv: return "CausalConv";
        case OpKind::GatedDelta: return "GatedDelta";
        case OpKind::Div: return "Div";
        case OpKind::Softmax: return "Softmax";
        case OpKind::TopK: return "TopK";
        case OpKind::ReduceSum: return "ReduceSum";
        case OpKind::ExpertMatMul: return "ExpertMatMul";
        case OpKind::SignMatMul: return "SignMatMul";
        case OpKind::PrevRow: return "PrevRow";
        case OpKind::LayerNorm: return "LayerNorm";
        case OpKind::Concat: return "Concat";
    }
    return "?";
}

namespace {

bool is_symbol(int64_t d) { return d == kT || d == kL; }
bool is_int(DType t) { return t == DType::I32 || t == DType::I64; }

// Static product and multiset of symbols of a shape.
std::pair<int64_t, std::multiset<int64_t>> split_shape(const Dims& dims) {
    int64_t product = 1;
    std::multiset<int64_t> symbols;
    for (int64_t d : dims) {
        if (is_symbol(d)) {
            symbols.insert(d);
        } else {
            product *= d;
        }
    }
    return {product, symbols};
}

}  // namespace

const Node& Graph::node(Value v) const {
    XS_CHECK(v.node >= 0 && static_cast<size_t>(v.node) < nodes_.size(), "invalid value (node ", v.node, ")");
    return nodes_[static_cast<size_t>(v.node)];
}

const TensorType& Graph::type(Value v) const {
    const Node& n = node(v);
    XS_CHECK(v.index >= 0 && static_cast<size_t>(v.index) < n.outputs.size(), "node ", v.node, " has no output ",
             v.index);
    return n.outputs[static_cast<size_t>(v.index)];
}

std::string Graph::where(Value v) const {
    const Node& n = node(v);
    return str_cat("node ", v.node, " (", n.label.empty() ? "-" : n.label, ": ", op_name(n.kind), ")");
}

Value Graph::add_node(OpKind kind, std::vector<Value> inputs, Attrs attrs, std::vector<TensorType> outputs,
                      std::string_view what) {
    for (const Value& in : inputs) (void)type(in);  // validates references
    Node n;
    n.kind = kind;
    n.inputs = std::move(inputs);
    n.attrs = std::move(attrs);
    n.outputs = std::move(outputs);
    n.label = scope_;
    if (!what.empty()) n.label += (n.label.empty() ? "" : ".") + std::string(what);
    nodes_.push_back(std::move(n));
    return Value{static_cast<int32_t>(nodes_.size() - 1), 0};
}

Graph::Scope::Scope(Graph& g, std::string_view name) : g_(g), prev_size_(g.scope_.size()) {
    if (!g_.scope_.empty()) g_.scope_ += '.';
    g_.scope_ += name;
}

Graph::Scope::~Scope() { g_.scope_.resize(prev_size_); }

Value Graph::input(const std::string& name, DType dtype, Dims shape) {
    for (const Node& n : nodes_) {
        if (n.kind == OpKind::Input && std::get<InputAttrs>(n.attrs).name == name) fail("duplicate graph input '", name, "'");
    }
    for (int64_t d : shape) XS_CHECK(d >= 0 || is_symbol(d), "input '", name, "' has invalid dim ", d);
    return add_node(OpKind::Input, {}, InputAttrs{name}, {TensorType{dtype, std::move(shape)}}, name);
}

Value Graph::weight(const WeightStore& store, const std::string& name) {
    for (size_t i = 0; i < nodes_.size(); ++i) {  // one node per weight, so backends upload it once
        const Node& n = nodes_[i];
        if (n.kind == OpKind::Weight && std::get<WeightAttrs>(n.attrs).name == name) {
            return Value{static_cast<int32_t>(i), 0};
        }
    }
    const TensorView& t = store.get(name);
    Node n;
    n.kind = OpKind::Weight;
    n.attrs = WeightAttrs{name};
    n.outputs = {TensorType{t.dtype, t.shape}};
    n.label = name;
    nodes_.push_back(std::move(n));
    return Value{static_cast<int32_t>(nodes_.size() - 1), 0};
}

Value Graph::constant(HostTensor value) {
    XS_CHECK(!value.empty(), "constant without data");
    TensorType t{value.dtype(), value.shape()};
    return add_node(OpKind::Constant, {}, ConstantAttrs{std::move(value)}, {std::move(t)}, "const");
}

Value Graph::gather(Value table, Value ids) {
    const TensorType& tt = type(table);
    const TensorType& it = type(ids);
    XS_CHECK(tt.shape.size() == 2, "gather table must be 2-D, got ", shape_to_string(tt.shape), " at ", where(table));
    XS_CHECK(!it.shape.empty() && is_int(it.dtype), "gather ids must be an integer tensor, got ", dtype_name(it.dtype),
             shape_to_string(it.shape));
    XS_CHECK(!is_int(tt.dtype) || tt.dtype == DType::I32, "an integer gather table must be i32, got ",
             dtype_name(tt.dtype), " at ", where(table));
    const DType out = tt.dtype == DType::I32 ? DType::I32 : DType::F32;
    Dims shape = it.shape;
    shape.push_back(tt.shape[1]);
    return add_node(OpKind::Gather, {table, ids}, {}, {TensorType{out, std::move(shape)}}, "gather");
}

Value Graph::matmul(Value x, Value w) {
    const TensorType& xt = type(x);
    const TensorType& wt = type(w);
    XS_CHECK(wt.shape.size() == 2, "matmul weight must be 2-D [out, in], got ", shape_to_string(wt.shape), " at ",
             where(w));
    XS_CHECK(!xt.shape.empty() && xt.shape.back() == wt.shape[1], "matmul ", shape_to_string(xt.shape), " x ",
             shape_to_string(wt.shape), "^T: inner dims differ (", where(x), ", ", where(w), ")");
    Dims out = xt.shape;
    out.back() = wt.shape[0];
    return add_node(OpKind::MatMul, {x, w}, {}, {TensorType{DType::F32, std::move(out)}}, "matmul");
}

Value Graph::elementwise_binary(OpKind kind, Value a, Value b) {
    const Dims& as = type(a).shape;
    const Dims& bs = type(b).shape;
    const size_t rank = std::max(as.size(), bs.size());
    Dims out(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - as.size() ? 1 : as[i - (rank - as.size())];
        const int64_t db = i < rank - bs.size() ? 1 : bs[i - (rank - bs.size())];
        if (da == db) {
            out[i] = da;
        } else if (da == 1) {
            out[i] = db;
        } else if (db == 1) {
            out[i] = da;
        } else {
            fail(op_name(kind), " cannot broadcast ", shape_to_string(as), " with ", shape_to_string(bs), " (",
                 where(a), ", ", where(b), ")");
        }
    }
    return add_node(kind, {a, b}, {}, {TensorType{DType::F32, std::move(out)}},
                    kind == OpKind::Add ? "add" : kind == OpKind::Mul ? "mul" : "div");
}

Value Graph::add(Value a, Value b) { return elementwise_binary(OpKind::Add, a, b); }
Value Graph::mul(Value a, Value b) { return elementwise_binary(OpKind::Mul, a, b); }
Value Graph::div(Value a, Value b) { return elementwise_binary(OpKind::Div, a, b); }

Value Graph::softmax(Value x) {
    XS_CHECK(!type(x).shape.empty(), "softmax of a scalar at ", where(x));
    return add_node(OpKind::Softmax, {x}, {}, {TensorType{DType::F32, type(x).shape}}, "softmax");
}

std::pair<Value, Value> Graph::topk(Value x, int64_t k) {
    const Dims& s = type(x).shape;
    XS_CHECK(!s.empty() && !is_symbol(s.back()) && k >= 1 && k <= s.back(), "topk ", k, " of ", shape_to_string(s),
             " at ", where(x), ": the last axis must be static and hold at least k values");
    Dims out = s;
    out.back() = k;
    const Value v = add_node(OpKind::TopK, {x}, TopKAttrs{k}, {TensorType{DType::F32, out}, TensorType{DType::I32, out}},
                             "topk");
    return {v, Value{v.node, 1}};
}

Value Graph::reduce_sum(Value x, int axis) {
    const Dims& s = type(x).shape;
    XS_CHECK(axis >= 0 && static_cast<size_t>(axis) < s.size() && s.size() >= 2, "reduce_sum over axis ", axis, " of ",
             shape_to_string(s), " at ", where(x));
    Dims out = s;
    out.erase(out.begin() + axis);
    return add_node(OpKind::ReduceSum, {x}, AxisAttrs{axis}, {TensorType{DType::F32, out}}, "reduce_sum");
}

Value Graph::expert_matmul(Value x, Value w, Value idx) {
    const Dims& xs = type(x).shape;
    const Dims& ws = type(w).shape;
    const Dims& is = type(idx).shape;
    XS_CHECK(ws.size() == 3, "expert_matmul weight must be [E, N, K], got ", shape_to_string(ws), " at ", where(w));
    XS_CHECK(is.size() == 2 && is_int(type(idx).dtype), "expert_matmul indices must be [T, S] int, got ",
             dtype_name(type(idx).dtype), shape_to_string(is));
    XS_CHECK((xs.size() == 2 || (xs.size() == 3 && xs[1] == is[1])) && xs[0] == is[0] && xs.back() == ws[2],
             "expert_matmul x ", shape_to_string(xs), " does not fit indices ", shape_to_string(is), " and weight ",
             shape_to_string(ws));
    return add_node(OpKind::ExpertMatMul, {x, w, idx}, {}, {TensorType{DType::F32, {is[0], is[1], ws[1]}}},
                    "expert_matmul");
}

Value Graph::sign_matmul(Value x, Value signs, Value scales) {
    const Dims& xs = type(x).shape;
    const Dims& ss = type(signs).shape;
    XS_CHECK(ss.size() == 2 && type(signs).dtype == DType::I32, "sign_matmul signs must be [N, K / 32] i32, got ",
             dtype_name(type(signs).dtype), shape_to_string(ss), " at ", where(signs));
    XS_CHECK(type(scales).shape == Dims{ss[0]} && is_float(type(scales).dtype), "sign_matmul scales must be [",
             ss[0], "] floats, got ", dtype_name(type(scales).dtype), shape_to_string(type(scales).shape));
    XS_CHECK(!xs.empty() && xs.back() == ss[1] * 32, "sign_matmul x ", shape_to_string(xs), " does not fit signs ",
             shape_to_string(ss), " (32 columns a word)");
    Dims out = xs;
    out.back() = ss[0];
    return add_node(OpKind::SignMatMul, {x, signs, scales}, {}, {TensorType{DType::F32, std::move(out)}}, "sign_matmul");
}

Value Graph::prev_row(Value x) {
    const TensorType& xt = type(x);
    XS_CHECK(!xt.shape.empty() && is_float(xt.dtype), "prev_row needs float rows, got ", dtype_name(xt.dtype),
             shape_to_string(xt.shape), " at ", where(x));
    return add_node(OpKind::PrevRow, {x}, {}, {TensorType{DType::F32, xt.shape}}, "prev_row");
}

Value Graph::elementwise_unary(OpKind kind, Value x, Attrs attrs) {
    TensorType t{DType::F32, type(x).shape};
    std::string what(op_name(kind));
    std::transform(what.begin(), what.end(), what.begin(), [](unsigned char c) { return std::tolower(c); });
    return add_node(kind, {x}, std::move(attrs), {std::move(t)}, what);
}

Value Graph::scale(Value x, float factor) { return elementwise_unary(OpKind::Scale, x, ScaleAttrs{factor}); }
Value Graph::tanh(Value x) { return elementwise_unary(OpKind::Tanh, x, {}); }
Value Graph::gelu_tanh(Value x) { return elementwise_unary(OpKind::GeluTanh, x, {}); }

Value Graph::softcap(Value x, float cap) {
    XS_CHECK(cap > 0.0f, "softcap must be positive, got ", cap);
    return elementwise_unary(OpKind::Softcap, x, SoftcapAttrs{cap});
}

Value Graph::sigmoid(Value x) { return elementwise_unary(OpKind::Sigmoid, x, {}); }
Value Graph::silu(Value x) { return elementwise_unary(OpKind::Silu, x, {}); }
Value Graph::softplus(Value x) { return elementwise_unary(OpKind::Softplus, x, {}); }

Value Graph::clamp(Value x, float lo, float hi) {
    XS_CHECK(!std::isnan(lo) && !std::isnan(hi) && lo <= hi, "clamp bounds [", lo, ", ", hi, "] are invalid at ",
             where(x));
    return elementwise_unary(OpKind::Clamp, x, ClampAttrs{lo, hi});
}

Value Graph::rms_norm(Value x, float eps) {
    XS_CHECK(!type(x).shape.empty(), "rms_norm of a scalar at ", where(x));
    return add_node(OpKind::RmsNorm, {x}, NormAttrs{eps}, {TensorType{DType::F32, type(x).shape}}, "rms_norm");
}

Value Graph::rms_norm(Value x, Value weight, float eps) {
    const Dims& xs = type(x).shape;
    const Dims& ws = type(weight).shape;
    XS_CHECK(!xs.empty() && ws.size() == 1 && ws[0] == xs.back(), "rms_norm weight ", shape_to_string(ws),
             " does not match the last dim of ", shape_to_string(xs), " (", where(weight), ")");
    return add_node(OpKind::RmsNorm, {x, weight}, NormAttrs{eps}, {TensorType{DType::F32, xs}}, "rms_norm");
}

Value Graph::layer_norm(Value x, Value weight, Value bias, float eps) {
    const Dims& xs = type(x).shape;
    for (const Value p : {weight, bias}) {
        const Dims& ps = type(p).shape;
        XS_CHECK(!xs.empty() && ps.size() == 1 && ps[0] == xs.back(), "layer_norm parameter ", shape_to_string(ps),
                 " does not match the last dim of ", shape_to_string(xs), " (", where(p), ")");
    }
    return add_node(OpKind::LayerNorm, {x, weight, bias}, NormAttrs{eps}, {TensorType{DType::F32, xs}}, "layer_norm");
}

Value Graph::concat(const std::vector<Value>& parts, int axis) {
    XS_CHECK(parts.size() >= 2, "concat needs at least two parts");
    const TensorType& first = type(parts[0]);
    XS_CHECK(axis >= 0 && static_cast<size_t>(axis) < first.shape.size(), "concat axis ", axis, " of ",
             shape_to_string(first.shape), " (", where(parts[0]), ")");
    Dims out = first.shape;
    out[static_cast<size_t>(axis)] = 0;
    for (const Value p : parts) {
        const TensorType& t = type(p);
        XS_CHECK(t.dtype == first.dtype && t.shape.size() == first.shape.size(), "concat parts differ: ",
                 shape_to_string(first.shape), " ", dtype_name(first.dtype), " and ", shape_to_string(t.shape), " ",
                 dtype_name(t.dtype), " (", where(p), ")");
        for (size_t d = 0; d < t.shape.size(); ++d) {
            if (d == static_cast<size_t>(axis)) {
                XS_CHECK(!is_symbol(t.shape[d]), "concat axis ", axis, " must be static, got ", shape_to_string(t.shape),
                         " (", where(p), ")");
                out[d] += t.shape[d];
            } else {
                XS_CHECK(t.shape[d] == first.shape[d], "concat parts differ off the axis: ", shape_to_string(first.shape),
                         " and ", shape_to_string(t.shape), " (", where(p), ")");
            }
        }
    }
    return add_node(OpKind::Concat, parts, AxisAttrs{axis}, {TensorType{first.dtype, out}}, "concat");
}

Value Graph::reshape(Value x, Dims shape) {
    const TensorType& xt = type(x);
    for (int64_t d : shape) XS_CHECK(d >= 0 || is_symbol(d), "reshape target has invalid dim ", d);
    const auto [in_prod, in_syms] = split_shape(xt.shape);
    const auto [out_prod, out_syms] = split_shape(shape);
    XS_CHECK(in_prod == out_prod && in_syms == out_syms, "reshape ", shape_to_string(xt.shape), " -> ",
             shape_to_string(shape), " changes the element count (", where(x), ")");
    return add_node(OpKind::Reshape, {x}, ReshapeAttrs{shape}, {TensorType{xt.dtype, shape}}, "reshape");
}

Value Graph::slice(Value x, int axis, int64_t begin, int64_t end) {
    const TensorType& xt = type(x);
    const int rank = static_cast<int>(xt.shape.size());
    XS_CHECK(axis >= 0 && axis < rank, "slice axis ", axis, " out of range for ", shape_to_string(xt.shape));
    const int64_t dim = xt.shape[static_cast<size_t>(axis)];
    int64_t size = 0;
    if (is_symbol(dim)) {
        XS_CHECK(begin < 0 && end == kSliceEnd, "slice of symbolic axis ", dim_to_string(dim),
                 " supports only the last n elements (begin < 0, end = kSliceEnd) at ", where(x));
        size = -begin;
    } else {
        const int64_t b = begin < 0 ? begin + dim : begin;
        const int64_t e = end == kSliceEnd ? dim : (end < 0 ? end + dim : end);
        XS_CHECK(b >= 0 && b < e && e <= dim, "slice [", begin, ", ", end, ") out of range for dim ", dim, " at ",
                 where(x));
        size = e - b;
    }
    Dims out = xt.shape;
    out[static_cast<size_t>(axis)] = size;
    return add_node(OpKind::Slice, {x}, SliceAttrs{axis, begin, end}, {TensorType{xt.dtype, std::move(out)}},
                    "slice");
}

Value Graph::transpose(Value x, std::vector<int> perm) {
    const TensorType& xt = type(x);
    XS_CHECK(perm.size() == xt.shape.size(), "transpose of ", shape_to_string(xt.shape), " needs ", xt.shape.size(),
             " axes, got ", perm.size(), " at ", where(x));
    std::vector<bool> seen(perm.size(), false);
    Dims out(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        const int axis = perm[i];
        XS_CHECK(axis >= 0 && static_cast<size_t>(axis) < perm.size() && !seen[static_cast<size_t>(axis)],
                 "transpose permutation is not a permutation of 0..", perm.size() - 1, " at ", where(x));
        seen[static_cast<size_t>(axis)] = true;
        out[i] = xt.shape[static_cast<size_t>(axis)];
    }
    return add_node(OpKind::Transpose, {x}, TransposeAttrs{std::move(perm)}, {TensorType{xt.dtype, std::move(out)}},
                    "transpose");
}

Value Graph::rope(Value x, Value positions, Value inv_freq, int64_t rotary_dim) {
    const Dims& xs = type(x).shape;
    const TensorType& pt = type(positions);
    const TensorType& ft = type(inv_freq);
    XS_CHECK(xs.size() == 3 && (xs[0] == kT || xs[0] > 0), "rope input must be [T, heads, head_dim], got ",
             shape_to_string(xs));
    const int64_t rotated = rotary_dim == 0 ? xs[2] : rotary_dim;
    XS_CHECK(rotated > 0 && rotated <= xs[2], "rope rotary_dim ", rotary_dim, " out of range for head_dim ", xs[2]);
    const bool axial = pt.shape.size() == 2;
    XS_CHECK(is_int(pt.dtype) && !pt.shape.empty() && pt.shape[0] == xs[0] &&
                 (pt.shape.size() == 1 || (axial && pt.shape[1] > 0)),
             "rope positions must be [T] or [T, sections] int with the input's T, got ", dtype_name(pt.dtype),
             shape_to_string(pt.shape), " for ", shape_to_string(xs));
    const int64_t sections = axial ? pt.shape[1] : 1;
    XS_CHECK(rotated % (2 * sections) == 0, "rope rotary dim ", rotated, " does not split into ", sections,
             " sections of even width");
    const int64_t pairs = rotated / (2 * sections);
    XS_CHECK(ft.shape == Dims{pairs} && ft.dtype == DType::F32, "rope inv_freq must be [", pairs, "] f32, got ",
             dtype_name(ft.dtype), shape_to_string(ft.shape));
    return add_node(OpKind::Rope, {x, positions, inv_freq}, RopeAttrs{rotary_dim == xs[2] ? 0 : rotary_dim},
                    {TensorType{DType::F32, xs}}, "rope");
}

int Graph::add_kv_slot(int64_t num_kv_heads, int64_t head_dim, int64_t sliding_window, std::string label) {
    XS_CHECK(num_kv_heads > 0 && head_dim > 0 && sliding_window >= 0, "invalid kv slot ", label);
    kv_slots_.push_back(KvSlot{num_kv_heads, head_dim, sliding_window, std::move(label)});
    return static_cast<int>(kv_slots_.size() - 1);
}

std::pair<Value, Value> Graph::kv_cache(int slot, Value k, Value v) {
    XS_CHECK(slot >= 0 && static_cast<size_t>(slot) < kv_slots_.size(), "unknown kv slot ", slot);
    const KvSlot& s = kv_slots_[static_cast<size_t>(slot)];
    const Dims expected{kT, s.num_kv_heads, s.head_dim};
    XS_CHECK(type(k).shape == expected && type(v).shape == expected, "kv_cache slot ", slot, " (", s.label,
             ") expects k, v ", shape_to_string(expected), ", got ", shape_to_string(type(k).shape), " and ",
             shape_to_string(type(v).shape));
    const TensorType all{DType::F32, {kL, s.num_kv_heads, s.head_dim}};
    const Value out = add_node(OpKind::KvCache, {k, v}, KvCacheAttrs{slot}, {all, all}, "kv_cache");
    return {Value{out.node, 0}, Value{out.node, 1}};
}

std::pair<Value, Value> Graph::kv_read(int slot) {
    XS_CHECK(slot >= 0 && static_cast<size_t>(slot) < kv_slots_.size(), "unknown kv slot ", slot);
    const KvSlot& s = kv_slots_[static_cast<size_t>(slot)];
    const TensorType all{DType::F32, {kL, s.num_kv_heads, s.head_dim}};
    const Value out = add_node(OpKind::KvRead, {}, KvCacheAttrs{slot}, {all, all}, "kv_read");
    return {Value{out.node, 0}, Value{out.node, 1}};
}

Value Graph::attention(Value q, Value k_all, Value v_all, const AttentionAttrs& attrs) {
    const Dims& qs = type(q).shape;
    const Dims& ks = type(k_all).shape;
    const Dims& vs = type(v_all).shape;
    XS_CHECK(qs.size() == 3 && (qs[0] == kT || (!attrs.causal && qs[0] > 0)),
             "attention q must be [T, heads, head_dim] (a static T only when not causal), got ", shape_to_string(qs));
    if (attrs.causal) {
        XS_CHECK(ks.size() == 3 && ks[0] == kL && ks == vs,
                 "causal attention k/v must be [L, kv_heads, head_dim] and equal, got ", shape_to_string(ks), " and ",
                 shape_to_string(vs));
    } else {
        XS_CHECK(ks.size() == 3 && ks == vs, "attention k/v must be [N, kv_heads, head_dim] and equal, got ",
                 shape_to_string(ks), " and ", shape_to_string(vs));
        XS_CHECK(attrs.sliding_window == 0 || (qs[0] == kT && ks[0] == kL),
                 "non-causal attention with a sliding window takes q [T, ...] and k/v [L, ...], got ",
                 shape_to_string(qs), " and ", shape_to_string(ks));
    }
    XS_CHECK(ks[2] == qs[2], "attention head_dim differs: q ", qs[2], ", k ", ks[2]);
    XS_CHECK(qs[1] % ks[1] == 0, "attention heads ", qs[1], " not a multiple of kv heads ", ks[1]);
    XS_CHECK(attrs.sliding_window >= 0 && attrs.softcap >= 0.0f, "invalid attention attributes");
    return add_node(OpKind::Attention, {q, k_all, v_all}, attrs, {TensorType{DType::F32, qs}}, "attention");
}

Value Graph::attention(Value q, Value k, Value v, Value key_bias, const AttentionAttrs& attrs) {
    XS_CHECK(!attrs.causal, "attention with a key bias must not be causal");
    const Value out = attention(q, k, v, attrs);
    const TensorType& bt = type(key_bias);
    XS_CHECK(bt.dtype == DType::F32 && bt.shape == (Dims{type(k).shape[0]}), "attention key_bias must be f32 [",
             dim_to_string(type(k).shape[0]), "], got ", shape_to_string(bt.shape), " (", where(key_bias), ")");
    nodes_[static_cast<size_t>(out.node)].inputs.push_back(key_bias);
    return out;
}

int Graph::add_conv_slot(int64_t channels, int64_t kernel, std::string label) {
    XS_CHECK(channels > 0 && kernel > 1, "invalid conv slot ", label);
    conv_slots_.push_back(ConvSlot{channels, kernel, std::move(label)});
    return static_cast<int>(conv_slots_.size() - 1);
}

Value Graph::causal_conv(int slot, Value x, Value w) {
    XS_CHECK(slot >= 0 && static_cast<size_t>(slot) < conv_slots_.size(), "unknown conv slot ", slot);
    const ConvSlot& s = conv_slots_[static_cast<size_t>(slot)];
    XS_CHECK(type(x).shape == (Dims{kT, s.channels}), "causal_conv slot ", slot, " (", s.label, ") expects x [T, ",
             s.channels, "], got ", shape_to_string(type(x).shape));
    XS_CHECK(type(w).shape == (Dims{s.channels, s.kernel}), "causal_conv slot ", slot, " expects w [", s.channels, ", ",
             s.kernel, "], got ", shape_to_string(type(w).shape));
    return add_node(OpKind::CausalConv, {x, w}, StateAttrs{slot, 0.0f}, {TensorType{DType::F32, {kT, s.channels}}},
                    "causal_conv");
}

int Graph::add_recurrent_slot(int64_t heads, int64_t key_dim, int64_t value_dim, std::string label) {
    XS_CHECK(heads > 0 && key_dim > 0 && value_dim > 0, "invalid recurrent slot ", label);
    recurrent_slots_.push_back(RecurrentSlot{heads, key_dim, value_dim, std::move(label)});
    return static_cast<int>(recurrent_slots_.size() - 1);
}

Value Graph::gated_delta(int slot, Value q, Value k, Value v, Value g, Value beta, float qk_l2norm_eps) {
    XS_CHECK(slot >= 0 && static_cast<size_t>(slot) < recurrent_slots_.size(), "unknown recurrent slot ", slot);
    const RecurrentSlot& s = recurrent_slots_[static_cast<size_t>(slot)];
    const Dims& qs = type(q).shape;
    XS_CHECK(qs.size() == 3 && qs[0] == kT && qs[2] == s.key_dim && type(k).shape == qs, "gated_delta slot ", slot, " (",
             s.label, ") expects q, k [T, Hk, ", s.key_dim, "], got ", shape_to_string(qs), " and ",
             shape_to_string(type(k).shape));
    XS_CHECK(type(v).shape == (Dims{kT, s.heads, s.value_dim}), "gated_delta expects v [T, ", s.heads, ", ",
             s.value_dim, "], got ", shape_to_string(type(v).shape));
    XS_CHECK(type(g).shape == (Dims{kT, s.heads}) && type(beta).shape == (Dims{kT, s.heads}),
             "gated_delta expects g and beta [T, ", s.heads, "]");
    XS_CHECK(s.heads % qs[1] == 0, "gated_delta: ", s.heads, " value heads are not a multiple of ", qs[1], " key heads");
    XS_CHECK(qk_l2norm_eps >= 0.0f, "gated_delta: negative l2norm epsilon");
    return add_node(OpKind::GatedDelta, {q, k, v, g, beta}, StateAttrs{slot, qk_l2norm_eps},
                    {TensorType{DType::F32, type(v).shape}}, "gated_delta");
}

void Graph::output(const std::string& name, Value v) {
    (void)type(v);
    for (const auto& o : outputs_) XS_CHECK(o.name != name, "duplicate graph output '", name, "'");
    outputs_.push_back(GraphOutput{name, v});
}

std::vector<Value> Graph::inputs() const {
    std::vector<Value> out;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].kind == OpKind::Input) out.push_back(Value{static_cast<int32_t>(i), 0});
    }
    return out;
}

void Graph::validate() const {
    XS_CHECK(!outputs_.empty(), "graph has no outputs");
    std::vector<int> appended(kv_slots_.size(), 0);
    std::vector<int> convs(conv_slots_.size(), 0);
    std::vector<int> recurrents(recurrent_slots_.size(), 0);
    int reads = 0;
    for (const Node& n : nodes_) {
        if (n.kind == OpKind::KvCache || n.kind == OpKind::KvRead) {
            ++appended[static_cast<size_t>(std::get<KvCacheAttrs>(n.attrs).slot)];
        }
        reads += n.kind == OpKind::KvRead ? 1 : 0;
        if (n.kind == OpKind::CausalConv) ++convs[static_cast<size_t>(std::get<StateAttrs>(n.attrs).slot)];
        if (n.kind == OpKind::GatedDelta) ++recurrents[static_cast<size_t>(std::get<StateAttrs>(n.attrs).slot)];
    }
    for (size_t s = 0; s < appended.size(); ++s) {
        XS_CHECK(appended[s] == 1, "kv slot ", s, " (", kv_slots_[s].label, ") is used ", appended[s],
                 " times; expected exactly once per graph");
    }
    XS_CHECK(reads == 0 || static_cast<size_t>(reads) == kv_slots_.size(),
             "a graph either appends to its kv slots or only reads them (kL is one or the other); ", reads, " of ",
             kv_slots_.size(), " are read");
    for (size_t s = 0; s < convs.size(); ++s) {
        XS_CHECK(convs[s] == 1, "conv slot ", s, " (", conv_slots_[s].label, ") is used ", convs[s], " times");
    }
    for (size_t s = 0; s < recurrents.size(); ++s) {
        XS_CHECK(recurrents[s] == 1, "recurrent slot ", s, " (", recurrent_slots_[s].label, ") is used ", recurrents[s],
                 " times");
    }
}

namespace {

// The attributes of a node as text, and a checksum of a constant's bytes, so two dumps are equal only when the
// graphs compute the same thing.
std::string attrs_to_string(const Attrs& a) {
    std::ostringstream os;
    os.precision(9);
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, ConstantAttrs>) {
                uint64_t h = 1469598103934665603ull;
                const auto* p = static_cast<const unsigned char*>(v.value.raw());
                for (size_t i = 0; i < v.value.nbytes(); ++i) h = (h ^ p[i]) * 1099511628211ull;
                os << "fnv=" << std::hex << h;
            } else if constexpr (std::is_same_v<T, ScaleAttrs>) {
                os << "factor=" << v.factor;
            } else if constexpr (std::is_same_v<T, SoftcapAttrs>) {
                os << "cap=" << v.cap;
            } else if constexpr (std::is_same_v<T, ClampAttrs>) {
                os << "lo=" << v.lo << " hi=" << v.hi;
            } else if constexpr (std::is_same_v<T, NormAttrs>) {
                os << "eps=" << v.eps;
            } else if constexpr (std::is_same_v<T, ReshapeAttrs>) {
                os << "shape=" << shape_to_string(v.shape);
            } else if constexpr (std::is_same_v<T, SliceAttrs>) {
                os << "axis=" << v.axis << " begin=" << v.begin << " end=" << dim_to_string(v.end == kSliceEnd ? -3 : v.end);
            } else if constexpr (std::is_same_v<T, TransposeAttrs>) {
                os << "perm=";
                for (const int p : v.perm) os << p << ',';
            } else if constexpr (std::is_same_v<T, KvCacheAttrs>) {
                os << "slot=" << v.slot;
            } else if constexpr (std::is_same_v<T, RopeAttrs>) {
                os << "rotary_dim=" << v.rotary_dim;
            } else if constexpr (std::is_same_v<T, StateAttrs>) {
                os << "slot=" << v.slot << " l2norm_eps=" << v.qk_l2norm_eps;
            } else if constexpr (std::is_same_v<T, TopKAttrs>) {
                os << "k=" << v.k;
            } else if constexpr (std::is_same_v<T, AxisAttrs>) {
                os << "axis=" << v.axis;
            } else if constexpr (std::is_same_v<T, AttentionAttrs>) {
                os << "scale=" << v.scale << " window=" << v.sliding_window << " softcap=" << v.softcap
                   << " causal=" << v.causal;
            }
        },
        a);
    return os.str();
}

}  // namespace

std::string Graph::dump() const {
    std::ostringstream os;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node& n = nodes_[i];
        os << '%' << i << " = " << op_name(n.kind) << '(';
        for (size_t j = 0; j < n.inputs.size(); ++j) {
            os << (j ? ", " : "") << '%' << n.inputs[j].node;
            if (n.inputs[j].index) os << ':' << n.inputs[j].index;
        }
        os << ')';
        if (const std::string a = attrs_to_string(n.attrs); !a.empty()) os << " {" << a << '}';
        os << " ->";
        for (const auto& t : n.outputs) os << ' ' << dtype_name(t.dtype) << shape_to_string(t.shape);
        if (!n.label.empty()) os << "  # " << n.label;
        os << '\n';
    }
    for (const auto& o : outputs_) os << "output " << o.name << " = %" << o.value.node << '\n';
    for (const auto& s : kv_slots_)
        os << "kv_slot " << s.label << " heads=" << s.num_kv_heads << " dim=" << s.head_dim << " window=" << s.sliding_window << '\n';
    for (const auto& s : conv_slots_) os << "conv_slot " << s.label << " channels=" << s.channels << " kernel=" << s.kernel << '\n';
    for (const auto& s : recurrent_slots_)
        os << "recurrent_slot " << s.label << " heads=" << s.heads << " dims=" << s.key_dim << 'x' << s.value_dim << '\n';
    return os.str();
}

}  // namespace xs::graph
