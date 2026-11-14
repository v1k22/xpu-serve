#include "backend/cpu_ref/cpu_backend.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <thread>
#include <utility>

#include "backend/cpu_ref/ops.h"
#include "backend/cpu_ref/thread_pool.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

namespace xs {

namespace {

using graph::kL;
using graph::kT;
using graph::Node;
using graph::OpKind;
using graph::Value;

std::string cpu_model_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.rfind("model name", 0) != 0) continue;
        const size_t colon = line.find(':');
        if (colon != std::string::npos && colon + 2 <= line.size()) return line.substr(colon + 2);
    }
    return "CPU";
}

HostTensor weight_to_f32(const std::string& name, const TensorView& view) {
    HostTensor out(DType::F32, view.shape);
    const auto n = static_cast<size_t>(view.numel());
    switch (view.dtype) {
        case DType::F32:
            std::memcpy(out.raw(), view.data, n * sizeof(float));
            break;
        case DType::F16:
            convert_f16_to_f32(static_cast<const uint16_t*>(view.data), out.data<float>(), n);
            break;
        case DType::BF16:
            convert_bf16_to_f32(static_cast<const uint16_t*>(view.data), out.data<float>(), n);
            break;
        case DType::Q8:
        case DType::Q4:
        case DType::F8:
        case DType::Block:
            quant::read_rows(view, 0, quant::matrix_rows(view.shape), out.data<float>());
            break;
        default:
            fail("cpu-ref: weight '", name, "' has dtype ", dtype_name(view.dtype),
                 "; supported: f32, f16, bf16, q8, q4, f8, block formats");
    }
    return out;
}

HostTensor as_f32(const HostTensor& t) {
    if (t.dtype() == DType::F32) return t;
    return HostTensor::from_vector(DType::F32, t.shape(), t.to_f32());
}

// Float tensors become f32; integer tensors (ids, positions) keep their dtype.
HostTensor normalize_dtype(const HostTensor& t) {
    return is_float(t.dtype()) ? as_f32(t) : t;
}

void check_options(const SessionOptions& options) {
    XS_CHECK(options.device.empty() || options.device == "CPU", "cpu-ref has only device 'CPU', got '", options.device,
             "'");
    XS_CHECK(options.precision == DType::F32 || options.precision == DType::F16,
             "cpu-ref precision must be f32 or f16, got ", dtype_name(options.precision));
    XS_CHECK(options.max_context > 0, "cpu-ref max_context must be positive, got ", options.max_context);
    if (!options.extra.empty()) fail("cpu-ref does not know backend option '", options.extra.begin()->first, "'");
}

// Sizes of the symbolic dims in one run.
struct Step {
    int64_t tokens = 0;  // kT
    int64_t kv_len = 0;  // kL
};

// The state the programs of a session share, by slot label (backend.h).
struct KvSlotState {
    graph::KvSlot spec;
    std::vector<float> k, v;  // [len, heads, head_dim]
    int64_t len = 0;
};
struct FixedSlotState {  // a conv or recurrent slot
    bool conv = false;
    std::string label;
    size_t elements = 0;
    std::vector<float> state;  // committed
    int64_t tokens = 0;        // tokens it summarizes
    // The last step began at `base` tokens (-1: none since a reset) from the state `before`: truncate can undo it
    // whole. When it had at most rewind_tokens tokens (`replay`), its op's inputs are kept too (conv: x, w;
    // recurrent: q, k, v, g, beta), replayed by truncate for a count inside the step.
    int64_t base = -1;
    bool replay = false;
    std::vector<float> before;
    std::vector<HostTensor> inputs;
    float eps = 0.0f;
};
struct SharedState {
    std::vector<KvSlotState> kv;
    std::vector<FixedSlotState> fixed;
};

// One program (graph) of a session: its leaves, and which shared slots its slots are.
class CpuProgram {
  public:
    CpuProgram(graph::Graph graph, std::vector<HostTensor> leaves, std::shared_ptr<cpu_ref::ThreadPool> pool,
               SharedState& shared, int64_t max_context, quant::KvFormats kv, int64_t rewind_tokens)
        : graph_(std::move(graph)),
          leaves_(std::move(leaves)),
          last_use_(graph_.nodes().size(), 0),
          pool_(std::move(pool)),
          shared_(shared),
          max_context_(max_context),
          kv_format_(kv),
          rewind_tokens_(rewind_tokens) {
        for (const graph::KvSlot& k : graph_.kv_slots()) {
            for (const quant::KvFormat f : {kv_format_.k, kv_format_.v}) {
                XS_CHECK(quant::kv_format_fits(f, k.head_dim), "cpu-ref: KV format ", quant::kv_format_name(f),
                         " needs a head_dim that is a multiple of 128; kv slot '", k.label, "' has ", k.head_dim);
            }
            kv_.push_back(kv_slot(k));
        }
        for (const graph::ConvSlot& c : graph_.conv_slots())
            conv_.push_back(fixed_slot(true, c.label, static_cast<size_t>((c.kernel - 1) * c.channels)));
        for (const graph::RecurrentSlot& r : graph_.recurrent_slots())
            recurrent_.push_back(fixed_slot(false, r.label, static_cast<size_t>(r.heads * r.key_dim * r.value_dim)));
        const auto& nodes = graph_.nodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const Value& in : nodes[i].inputs)
                last_use_[static_cast<size_t>(in.node)] = i;
            reads_kv_ |= nodes[i].kind == OpKind::KvRead;
        }
        for (const auto& out : graph_.outputs())
            last_use_[static_cast<size_t>(out.value.node)] = nodes.size();
    }

    void run(const TensorMap& inputs, TensorMap& outputs, std::span<const std::string> only) {
        const Step step = check_inputs(inputs);
        XS_CHECK(step.kv_len <= max_context_, "cpu-ref: ", step.kv_len, " tokens exceed max_context ", max_context_);
        for (const std::string& name : only) {
            XS_CHECK(std::any_of(graph_.outputs().begin(), graph_.outputs().end(),
                                 [&](const auto& out) { return out.name == name; }),
                     "cpu-ref: the program has no output '", name, "'");
        }
        const auto& nodes = graph_.nodes();
        std::vector<std::vector<HostTensor>> values(nodes.size());
        // Conv and recurrent ops update copies, committed only when the whole step succeeds; a short step keeps its
        // ops' inputs for truncate.
        const bool rewindable = step.tokens <= rewind_tokens_;
        std::vector<std::vector<float>> conv_state, recurrent_state;
        std::vector<std::vector<HostTensor>> conv_inputs(conv_.size()), recurrent_inputs(recurrent_.size());
        for (size_t c : conv_) conv_state.push_back(shared_.fixed[c].state);
        for (size_t r : recurrent_) recurrent_state.push_back(shared_.fixed[r].state);
        const int64_t before = cached();
        try {
            for (size_t i = 0; i < nodes.size(); ++i) {
                values[i] = eval_node(i, values, inputs, step, conv_state, recurrent_state, rewindable ? &conv_inputs : nullptr,
                                      rewindable ? &recurrent_inputs : nullptr);
                check_output_shapes(i, values[i], step);
                for (const Value& in : nodes[i].inputs) {
                    if (last_use_[static_cast<size_t>(in.node)] == i) values[static_cast<size_t>(in.node)].clear();
                }
            }
        } catch (...) {
            for (size_t k : kv_) resize_kv(shared_.kv[k], before);  // a failed step leaves the KV state as it was
            throw;
        }
        for (size_t k : kv_) shared_.kv[k].len = step.kv_len;
        const auto commit = [&](const std::vector<size_t>& slots, std::vector<std::vector<float>>& states,
                                std::vector<std::vector<HostTensor>>& kept) {
            for (size_t j = 0; j < slots.size(); ++j) {
                FixedSlotState& f = shared_.fixed[slots[j]];
                f.base = f.tokens;
                f.replay = rewindable;
                f.before = std::move(f.state);
                f.inputs = rewindable ? std::move(kept[j]) : std::vector<HostTensor>();
                f.state = std::move(states[j]);
                f.tokens += step.tokens;
            }
        };
        commit(conv_, conv_state, conv_inputs);
        commit(recurrent_, recurrent_state, recurrent_inputs);
        for (const auto& out : graph_.outputs()) {
            if (!only.empty() && std::find(only.begin(), only.end(), out.name) == only.end()) continue;
            const HostTensor& t = values[static_cast<size_t>(out.value.node)][static_cast<size_t>(out.value.index)];
            outputs[out.name] = is_leaf(nodes[static_cast<size_t>(out.value.node)].kind) ? t.clone() : t;
        }
    }

    // Tokens in the slots this program writes (they agree after every run and truncate).
    int64_t cached() const {
        if (!kv_.empty() && !reads_kv_) return shared_.kv[kv_[0]].len;
        if (!conv_.empty()) return shared_.fixed[conv_[0]].tokens;
        if (!recurrent_.empty()) return shared_.fixed[recurrent_[0]].tokens;
        return 0;
    }

    void truncate(int64_t n) {
        const int64_t have = cached();
        XS_CHECK(n >= 0 && n <= have, "cannot truncate to ", n, " tokens; ", have, " are cached");
        for (const std::vector<size_t>* slots : {&conv_, &recurrent_}) {  // checked first: a refusal changes nothing
            for (size_t i : *slots) {
                const FixedSlotState& f = shared_.fixed[i];
                XS_CHECK(n == f.tokens || n == 0 || (f.base >= 0 && (n == f.base || (f.replay && n > f.base))),
                         "cannot truncate to ", n, " tokens: conv and recurrent state summarize the whole sequence ",
                         "(the last step can be undone whole; a count inside it only when it had up to rewind_tokens ",
                         "tokens)");
            }
        }
        if (!reads_kv_) {
            for (size_t k : kv_) {
                resize_kv(shared_.kv[k], n);
                shared_.kv[k].len = n;
            }
        }
        for (const std::vector<size_t>* slots : {&conv_, &recurrent_}) {
            for (size_t i : *slots) rewind(shared_.fixed[i], n);
        }
    }

  private:
    static bool is_leaf(OpKind kind) {
        return kind == OpKind::Input || kind == OpKind::Weight || kind == OpKind::Constant;
    }

    size_t kv_slot(const graph::KvSlot& spec) {
        for (size_t i = 0; i < shared_.kv.size(); ++i) {
            const graph::KvSlot& s = shared_.kv[i].spec;
            if (s.label != spec.label) continue;
            XS_CHECK(s.num_kv_heads == spec.num_kv_heads && s.head_dim == spec.head_dim, "kv slot '", spec.label,
                     "' is declared with two shapes");
            return i;
        }
        shared_.kv.push_back(KvSlotState{spec, {}, {}, 0});
        return shared_.kv.size() - 1;
    }

    size_t fixed_slot(bool conv, const std::string& label, size_t elements) {
        for (size_t i = 0; i < shared_.fixed.size(); ++i) {
            const FixedSlotState& f = shared_.fixed[i];
            if (f.conv != conv || f.label != label) continue;
            XS_CHECK(f.elements == elements, "state slot '", label, "' is declared with two sizes");
            return i;
        }
        FixedSlotState f;
        f.conv = conv;
        f.label = label;
        f.elements = elements;
        f.state.assign(elements, 0.0f);
        shared_.fixed.push_back(std::move(f));
        return shared_.fixed.size() - 1;
    }

    // Brings a conv or recurrent slot to `n` tokens: kept, zeroed, or replayed from the state before the last step.
    void rewind(FixedSlotState& f, int64_t n) const {
        if (n == f.tokens) return;
        if (n == 0) {
            std::fill(f.state.begin(), f.state.end(), 0.0f);
            f.tokens = 0;
            f.base = -1;
            return;
        }
        const int64_t m = n - f.base;  // truncate checked base <= n
        f.state = f.before;
        if (m > 0) {
            std::vector<HostTensor> head;
            for (size_t i = 0; i < f.inputs.size(); ++i) {
                const bool per_token = !(f.conv && i == 1);  // a conv's w is not per token
                head.push_back(per_token ? cpu_ref::slice(f.inputs[i], 0, 0, m) : f.inputs[i]);
            }
            if (f.conv) {
                cpu_ref::causal_conv(head[0], head[1], f.state);
            } else {
                cpu_ref::gated_delta(*pool_, head[0], head[1], head[2], head[3], head[4], f.eps, f.state);
            }
        }
        f.tokens = n;
    }

    Dims resolve(const Dims& dims, const Step& step) const {
        Dims out = dims;
        for (int64_t& d : out) {
            if (d == kT) d = step.tokens;
            if (d == kL) d = step.kv_len;
        }
        return out;
    }

    // Validates `inputs` against the graph inputs and returns the step's T and L.
    Step check_inputs(const TensorMap& inputs) const {
        const std::vector<Value> graph_inputs = graph_.inputs();
        for (const auto& [name, tensor] : inputs) {
            const bool known = std::any_of(graph_inputs.begin(), graph_inputs.end(),
                                           [&](const Value& v) { return input_name(v) == name; });
            XS_CHECK(known, "input '", name, "' is not a graph input");
        }
        Step step;
        bool have_tokens = false;
        for (const Value& v : graph_inputs) {
            const std::string& name = input_name(v);
            const auto it = inputs.find(name);
            XS_CHECK(it != inputs.end(), "missing graph input '", name, "'");
            const graph::TensorType& type = graph_.type(v);
            const HostTensor& t = it->second;
            XS_CHECK(t.dtype() == type.dtype, "input '", name, "' must be ", dtype_name(type.dtype), ", got ",
                     dtype_name(t.dtype()));
            XS_CHECK(t.rank() == type.shape.size(), "input '", name, "' must have shape ",
                     graph::shape_to_string(type.shape), ", got ", dims_to_string(t.shape()));
            for (size_t d = 0; d < type.shape.size(); ++d) {
                if (type.shape[d] != kT) continue;
                XS_CHECK(!have_tokens || t.dim(d) == step.tokens, "input '", name, "' has ", t.dim(d),
                         " tokens; another input has ", step.tokens);
                step.tokens = t.dim(d);
                have_tokens = true;
            }
        }
        // A program reading KV slots sees their length; one writing state extends it by T.
        step.kv_len = reads_kv_ ? shared_.kv[kv_[0]].len : cached() + step.tokens;
        for (const Value& v : graph_inputs) {
            const HostTensor& t = inputs.at(input_name(v));
            XS_CHECK(t.shape() == resolve(graph_.type(v).shape, step), "input '", input_name(v), "' must have shape ",
                     graph::shape_to_string(graph_.type(v).shape), " (T=", step.tokens, ", L=", step.kv_len, "), got ",
                     dims_to_string(t.shape()));
        }
        return step;
    }

    const std::string& input_name(Value v) const { return std::get<graph::InputAttrs>(graph_.node(v).attrs).name; }

    void check_output_shapes(size_t i, const std::vector<HostTensor>& produced, const Step& step) const {
        const Node& n = graph_.nodes()[i];
        for (size_t j = 0; j < n.outputs.size(); ++j) {
            const Dims expected = resolve(n.outputs[j].shape, step);
            XS_CHECK(produced[j].shape() == expected, "node ", i, " (", n.label, ": ", graph::op_name(n.kind),
                     ") produced ", dims_to_string(produced[j].shape()), ", expected ", dims_to_string(expected));
        }
    }

    std::vector<HostTensor> eval_node(size_t i, const std::vector<std::vector<HostTensor>>& values,
                                      const TensorMap& inputs, const Step& step,
                                      std::vector<std::vector<float>>& conv_state,
                                      std::vector<std::vector<float>>& recurrent_state,
                                      std::vector<std::vector<HostTensor>>* conv_inputs,
                                      std::vector<std::vector<HostTensor>>* recurrent_inputs) {
        const Node& n = graph_.nodes()[i];
        const auto in = [&](size_t k) -> const HostTensor& {
            const Value& v = n.inputs[k];
            return values[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)];
        };
        const auto f32 = [&](size_t k) { return as_f32(in(k)); };
        switch (n.kind) {
            case OpKind::Input:
                return {normalize_dtype(inputs.at(std::get<graph::InputAttrs>(n.attrs).name))};
            case OpKind::Weight:
            case OpKind::Constant:
                return {leaves_[i]};
            case OpKind::Gather:
                return {cpu_ref::gather(in(0).dtype() == DType::I32 ? in(0) : f32(0), in(1))};
            case OpKind::MatMul:
                return {cpu_ref::matmul(*pool_, f32(0), f32(1))};
            case OpKind::Add:
                return {cpu_ref::add(f32(0), f32(1))};
            case OpKind::Mul:
                return {cpu_ref::mul(f32(0), f32(1))};
            case OpKind::Scale:
                return {cpu_ref::scale(f32(0), std::get<graph::ScaleAttrs>(n.attrs).factor)};
            case OpKind::Tanh:
                return {cpu_ref::tanh(f32(0))};
            case OpKind::GeluTanh:
                return {cpu_ref::gelu_tanh(f32(0))};
            case OpKind::Gelu:
                return {cpu_ref::gelu(f32(0))};
            case OpKind::Softcap:
                return {cpu_ref::softcap(f32(0), std::get<graph::SoftcapAttrs>(n.attrs).cap)};
            case OpKind::Sigmoid:
                return {cpu_ref::sigmoid(f32(0))};
            case OpKind::Silu:
                return {cpu_ref::silu(f32(0))};
            case OpKind::Softplus:
                return {cpu_ref::softplus(f32(0))};
            case OpKind::Clamp: {
                const auto& a = std::get<graph::ClampAttrs>(n.attrs);
                return {cpu_ref::clamp(f32(0), a.lo, a.hi)};
            }
            case OpKind::RmsNorm: {
                const float eps = std::get<graph::NormAttrs>(n.attrs).eps;
                if (n.inputs.size() == 1) return {cpu_ref::rms_norm(f32(0), nullptr, eps)};
                const HostTensor weight = f32(1);
                return {cpu_ref::rms_norm(f32(0), &weight, eps)};
            }
            case OpKind::Reshape:
                return {cpu_ref::reshape(in(0), resolve(std::get<graph::ReshapeAttrs>(n.attrs).shape, step))};
            case OpKind::Slice: {
                const auto& a = std::get<graph::SliceAttrs>(n.attrs);
                return {cpu_ref::slice(in(0), a.axis, a.begin, a.end)};
            }
            case OpKind::Transpose:
                return {cpu_ref::transpose(in(0), std::get<graph::TransposeAttrs>(n.attrs).perm)};
            case OpKind::Rope:
                return {cpu_ref::rope(f32(0), in(1), f32(2), std::get<graph::RopeAttrs>(n.attrs).rotary_dim)};
            case OpKind::CausalConv: {
                const auto slot = static_cast<size_t>(std::get<graph::StateAttrs>(n.attrs).slot);
                if (conv_inputs != nullptr) (*conv_inputs)[slot] = {f32(0), f32(1)};
                return {cpu_ref::causal_conv(f32(0), f32(1), conv_state[slot])};
            }
            case OpKind::GatedDelta: {
                const auto& a = std::get<graph::StateAttrs>(n.attrs);
                const auto slot = static_cast<size_t>(a.slot);
                if (recurrent_inputs != nullptr) {
                    (*recurrent_inputs)[slot] = {f32(0), f32(1), f32(2), f32(3), f32(4)};
                    shared_.fixed[recurrent_[slot]].eps = a.qk_l2norm_eps;
                }
                return {cpu_ref::gated_delta(*pool_, f32(0), f32(1), f32(2), f32(3), f32(4), a.qk_l2norm_eps,
                                             recurrent_state[slot])};
            }
            case OpKind::KvCache: {
                auto [k_all, v_all] = append_kv(std::get<graph::KvCacheAttrs>(n.attrs).slot, f32(0), f32(1), step);
                return {std::move(k_all), std::move(v_all)};
            }
            case OpKind::KvRead: {
                const KvSlotState& s = shared_.kv[kv_[static_cast<size_t>(std::get<graph::KvCacheAttrs>(n.attrs).slot)]];
                const Dims shape{s.len, s.spec.num_kv_heads, s.spec.head_dim};
                return {HostTensor::from_vector(DType::F32, shape, s.k), HostTensor::from_vector(DType::F32, shape, s.v)};
            }
            case OpKind::Attention: {
                const auto& a = std::get<graph::AttentionAttrs>(n.attrs);
                if (a.grouped) return {cpu_ref::attention(*pool_, f32(0), f32(1), f32(2), a, nullptr, &in(3))};
                if (n.inputs.size() == 4) {
                    const HostTensor bias = f32(3);
                    return {cpu_ref::attention(*pool_, f32(0), f32(1), f32(2), a, &bias)};
                }
                return {cpu_ref::attention(*pool_, f32(0), f32(1), f32(2), a)};
            }
            case OpKind::LayerNorm:
                return {cpu_ref::layer_norm(f32(0), f32(1), f32(2), std::get<graph::NormAttrs>(n.attrs).eps)};
            case OpKind::Concat: {
                std::vector<HostTensor> parts;
                for (size_t k = 0; k < n.inputs.size(); ++k)
                    parts.push_back(in(k).dtype() == DType::I32 || in(k).dtype() == DType::I64 ? in(k) : f32(k));
                return {cpu_ref::concat(parts, std::get<graph::AxisAttrs>(n.attrs).axis)};
            }
            case OpKind::Div:
                return {cpu_ref::div(f32(0), f32(1))};
            case OpKind::Softmax:
                return {cpu_ref::softmax(f32(0))};
            case OpKind::TopK: {
                auto [top_values, top_indices] = cpu_ref::topk(f32(0), std::get<graph::TopKAttrs>(n.attrs).k);
                return {std::move(top_values), std::move(top_indices)};
            }
            case OpKind::ReduceSum:
                return {cpu_ref::reduce_sum(f32(0), std::get<graph::AxisAttrs>(n.attrs).axis)};
            case OpKind::ExpertMatMul:
                return {cpu_ref::expert_matmul(*pool_, f32(0), f32(1), in(2))};
            case OpKind::SignMatMul:
                return {cpu_ref::sign_matmul(*pool_, f32(0), in(1), f32(2))};
            case OpKind::PrevRow:
                return {cpu_ref::prev_row(f32(0))};
        }
        fail("cpu-ref: unhandled op ", graph::op_name(n.kind));
    }

    // Appends this step's k, v ([T, Hkv, D]) to the slot and returns the full [L, Hkv, D] tensors.
    std::pair<HostTensor, HostTensor> append_kv(int slot, const HostTensor& k, const HostTensor& v, const Step& step) {
        KvSlotState& state = shared_.kv[kv_[static_cast<size_t>(slot)]];
        const graph::KvSlot& s = state.spec;
        const auto row = static_cast<size_t>(s.num_kv_heads * s.head_dim);
        state.k.resize(static_cast<size_t>(step.kv_len) * row);
        state.v.resize(static_cast<size_t>(step.kv_len) * row);
        const size_t past = static_cast<size_t>(step.kv_len - step.tokens) * row;
        std::copy(k.data<float>(), k.data<float>() + k.numel(), state.k.begin() + static_cast<std::ptrdiff_t>(past));
        std::copy(v.data<float>(), v.data<float>() + v.numel(), state.v.begin() + static_cast<std::ptrdiff_t>(past));
        const int64_t rows = k.numel() / s.head_dim;
        quant::round_trip_kv(kv_format_.k, state.k.data() + past, rows, s.head_dim);
        quant::round_trip_kv(kv_format_.v, state.v.data() + past, rows, s.head_dim);
        const Dims shape{step.kv_len, s.num_kv_heads, s.head_dim};
        return {HostTensor::from_vector(DType::F32, shape, state.k),
                HostTensor::from_vector(DType::F32, shape, state.v)};
    }

    static void resize_kv(KvSlotState& s, int64_t tokens) {
        const auto size = static_cast<size_t>(tokens * s.spec.num_kv_heads * s.spec.head_dim);
        s.k.resize(std::min(size, s.k.size()));
        s.v.resize(std::min(size, s.v.size()));
    }

    const graph::Graph graph_;
    const std::vector<HostTensor> leaves_;  // f32 weights and normalized constants by node index; empty otherwise
    std::vector<size_t> last_use_;          // index of the last node reading each node; outputs live to the end
    const std::shared_ptr<cpu_ref::ThreadPool> pool_;
    SharedState& shared_;
    const int64_t max_context_;
    const quant::KvFormats kv_format_;  // the cache holds each appended row as the value its format stores
    const int64_t rewind_tokens_;
    bool reads_kv_ = false;               // the program reads its KV slots (KvRead) instead of appending
    std::vector<size_t> kv_, conv_, recurrent_;  // the shared slot of each of the graph's slots
};

class CpuSession final : public Session {
  public:
    CpuSession() = default;

    void add(std::unique_ptr<CpuProgram> program) { programs_.push_back(std::move(program)); }
    SharedState& shared() { return shared_; }

    void run(size_t program, const TensorMap& inputs, TensorMap& outputs, std::span<const std::string> only) override {
        at(program).run(inputs, outputs, only);
    }
    int64_t num_cached(size_t program) const override {
        XS_CHECK(program < programs_.size(), "cpu-ref: no program ", program);
        return programs_[program]->cached();
    }
    void reset() override {
        for (KvSlotState& k : shared_.kv) {
            k.k.clear();
            k.v.clear();
            k.len = 0;
        }
        for (FixedSlotState& f : shared_.fixed) {
            std::fill(f.state.begin(), f.state.end(), 0.0f);
            f.tokens = 0;
            f.base = -1;
        }
    }
    void truncate(size_t program, int64_t n) override { at(program).truncate(n); }

  private:
    CpuProgram& at(size_t program) {
        XS_CHECK(program < programs_.size(), "cpu-ref: no program ", program, " (", programs_.size(), " compiled)");
        return *programs_[program];
    }

    SharedState shared_;  // before the programs, which refer to it
    std::vector<std::unique_ptr<CpuProgram>> programs_;
};

class CpuBackend final : public Backend {
  public:
    CpuBackend()
        : pool_(std::make_shared<cpu_ref::ThreadPool>(std::max(1u, std::thread::hardware_concurrency()) - 1)) {}

    std::string_view name() const override { return "cpu-ref"; }

    std::vector<DeviceInfo> devices() const override { return {DeviceInfo{"CPU", cpu_model_name(), 0, false}}; }

    // Computation is always f32 whatever options.precision says: this backend is the accuracy oracle.
    std::unique_ptr<Session> compile(std::span<const graph::Graph* const> programs, const WeightStore& weights,
                                     const SessionOptions& options) override {
        check_options(options);
        XS_CHECK(!programs.empty(), "cpu-ref: compile without a program");
        auto session = std::make_unique<CpuSession>();
        for (const graph::Graph* graph : programs) {
            graph->validate();
            const auto& nodes = graph->nodes();
            std::vector<HostTensor> leaves(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                const Node& n = nodes[i];
                if (n.kind == OpKind::Weight) {
                    const std::string& name = std::get<graph::WeightAttrs>(n.attrs).name;
                    const TensorView& view = weights.get(name);
                    XS_CHECK(view.shape == n.outputs[0].shape, "weight '", name, "' has shape ",
                             dims_to_string(view.shape), " but the graph was built for ",
                             dims_to_string(n.outputs[0].shape));
                    leaves[i] = weight_to_f32(name, view);
                } else if (n.kind == OpKind::Constant) {
                    leaves[i] = normalize_dtype(std::get<graph::ConstantAttrs>(n.attrs).value);
                }
            }
            session->add(std::make_unique<CpuProgram>(*graph, std::move(leaves), pool_, session->shared(),
                                                      options.max_context, options.kv,
                                                      options.rewind_tokens));
        }
        return session;
    }

  private:
    std::shared_ptr<cpu_ref::ThreadPool> pool_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_ref_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace xs
