#include "graph/graph.h"

#include <algorithm>
#include <cctype>
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
        case OpKind::RmsNorm: return "RmsNorm";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Slice: return "Slice";
        case OpKind::Rope: return "Rope";
        case OpKind::KvCache: return "KvCache";
        case OpKind::Attention: return "Attention";
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
    XS_CHECK(it.shape.size() == 1 && is_int(it.dtype), "gather ids must be a 1-D integer tensor, got ",
             dtype_name(it.dtype), shape_to_string(it.shape));
    return add_node(OpKind::Gather, {table, ids}, {}, {TensorType{DType::F32, {it.shape[0], tt.shape[1]}}}, "gather");
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
                    kind == OpKind::Add ? "add" : "mul");
}

Value Graph::add(Value a, Value b) { return elementwise_binary(OpKind::Add, a, b); }
Value Graph::mul(Value a, Value b) { return elementwise_binary(OpKind::Mul, a, b); }

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

Value Graph::rms_norm(Value x, float eps) {
    XS_CHECK(!type(x).shape.empty(), "rms_norm of a scalar at ", where(x));
    return add_node(OpKind::RmsNorm, {x}, RmsNormAttrs{eps}, {TensorType{DType::F32, type(x).shape}}, "rms_norm");
}

Value Graph::rms_norm(Value x, Value weight, float eps) {
    const Dims& xs = type(x).shape;
    const Dims& ws = type(weight).shape;
    XS_CHECK(!xs.empty() && ws.size() == 1 && ws[0] == xs.back(), "rms_norm weight ", shape_to_string(ws),
             " does not match the last dim of ", shape_to_string(xs), " (", where(weight), ")");
    return add_node(OpKind::RmsNorm, {x, weight}, RmsNormAttrs{eps}, {TensorType{DType::F32, xs}}, "rms_norm");
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

Value Graph::rope(Value x, Value positions, Value inv_freq) {
    const Dims& xs = type(x).shape;
    const TensorType& pt = type(positions);
    const TensorType& ft = type(inv_freq);
    XS_CHECK(xs.size() == 3 && xs[0] == kT, "rope input must be [T, heads, head_dim], got ", shape_to_string(xs));
    XS_CHECK(xs[2] % 2 == 0, "rope head_dim must be even, got ", xs[2]);
    XS_CHECK(pt.shape == Dims{kT} && is_int(pt.dtype), "rope positions must be [T] int, got ", dtype_name(pt.dtype),
             shape_to_string(pt.shape));
    XS_CHECK(ft.shape == Dims{xs[2] / 2} && ft.dtype == DType::F32, "rope inv_freq must be [", xs[2] / 2,
             "] f32, got ", dtype_name(ft.dtype), shape_to_string(ft.shape));
    return add_node(OpKind::Rope, {x, positions, inv_freq}, {}, {TensorType{DType::F32, xs}}, "rope");
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

Value Graph::attention(Value q, Value k_all, Value v_all, const AttentionAttrs& attrs) {
    const Dims& qs = type(q).shape;
    const Dims& ks = type(k_all).shape;
    const Dims& vs = type(v_all).shape;
    XS_CHECK(qs.size() == 3 && qs[0] == kT, "attention q must be [T, heads, head_dim], got ", shape_to_string(qs));
    XS_CHECK(ks.size() == 3 && ks[0] == kL && ks == vs, "attention k/v must be [L, kv_heads, head_dim] and equal, got ",
             shape_to_string(ks), " and ", shape_to_string(vs));
    XS_CHECK(ks[2] == qs[2], "attention head_dim differs: q ", qs[2], ", k ", ks[2]);
    XS_CHECK(qs[1] % ks[1] == 0, "attention heads ", qs[1], " not a multiple of kv heads ", ks[1]);
    XS_CHECK(attrs.sliding_window >= 0 && attrs.softcap >= 0.0f, "invalid attention attributes");
    return add_node(OpKind::Attention, {q, k_all, v_all}, attrs, {TensorType{DType::F32, qs}}, "attention");
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
    for (const Node& n : nodes_) {
        if (n.kind == OpKind::KvCache) ++appended[static_cast<size_t>(std::get<KvCacheAttrs>(n.attrs).slot)];
    }
    for (size_t s = 0; s < appended.size(); ++s) {
        XS_CHECK(appended[s] == 1, "kv slot ", s, " (", kv_slots_[s].label, ") is appended ", appended[s],
                 " times; expected exactly once per graph");
    }
}

std::string Graph::dump() const {
    std::ostringstream os;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const Node& n = nodes_[i];
        os << '%' << i << " = " << op_name(n.kind) << '(';
        for (size_t j = 0; j < n.inputs.size(); ++j) {
            os << (j ? ", " : "") << '%' << n.inputs[j].node;
            if (n.inputs[j].index) os << ':' << n.inputs[j].index;
        }
        os << ") ->";
        for (const auto& t : n.outputs) os << ' ' << dtype_name(t.dtype) << shape_to_string(t.shape);
        if (!n.label.empty()) os << "  # " << n.label;
        os << '\n';
    }
    for (const auto& o : outputs_) os << "output " << o.name << " = %" << o.value.node << '\n';
    return os.str();
}

}  // namespace xs::graph
