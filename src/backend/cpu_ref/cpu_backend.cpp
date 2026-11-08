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
            quant::dequantize_rows(view, 0, quant::matrix_rows(view.shape), out.data<float>());
            break;
        default:
            fail("cpu-ref: weight '", name, "' has dtype ", dtype_name(view.dtype), "; supported: f32, f16, bf16, q8, q4");
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
    XS_CHECK(!options.kv_dtype || *options.kv_dtype == DType::I8, "cpu-ref KV cache dtype must be i8 when set, got ",
             dtype_name(*options.kv_dtype));
}

struct KvState {
    std::vector<float> k;
    std::vector<float> v;
};

// Sizes of the symbolic dims in one run.
struct Step {
    int64_t tokens = 0;  // kT
    int64_t kv_len = 0;  // kL
};

class CpuSession final : public Session {
  public:
    CpuSession(graph::Graph graph, std::vector<HostTensor> leaves, std::shared_ptr<cpu_ref::ThreadPool> pool,
               int64_t max_context, bool kv_i8)
        : graph_(std::move(graph)),
          leaves_(std::move(leaves)),
          last_use_(graph_.nodes().size(), 0),
          pool_(std::move(pool)),
          max_context_(max_context),
          kv_i8_(kv_i8),
          kv_(graph_.kv_slots().size()) {
        reset_fixed_state();
        const auto& nodes = graph_.nodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const Value& in : nodes[i].inputs)
                last_use_[static_cast<size_t>(in.node)] = i;
        }
        for (const auto& out : graph_.outputs())
            last_use_[static_cast<size_t>(out.value.node)] = nodes.size();
    }

    void run(const TensorMap& inputs, TensorMap& outputs) override {
        const Step step = check_inputs(inputs);
        XS_CHECK(step.kv_len <= max_context_, "cpu-ref: ", step.kv_len, " tokens exceed max_context ", max_context_);
        const auto& nodes = graph_.nodes();
        std::vector<std::vector<HostTensor>> values(nodes.size());
        // Conv and recurrent ops update copies, committed only when the whole step succeeds.
        pending_conv_ = conv_;
        pending_recurrent_ = recurrent_;
        try {
            for (size_t i = 0; i < nodes.size(); ++i) {
                values[i] = eval_node(i, values, inputs, step);
                check_output_shapes(i, values[i], step);
                for (const Value& in : nodes[i].inputs) {
                    if (last_use_[static_cast<size_t>(in.node)] == i) values[static_cast<size_t>(in.node)].clear();
                }
            }
        } catch (...) {
            resize_kv(cached_);  // a failed step leaves the KV state as it was
            throw;
        }
        if (graph_.stateful()) cached_ = step.kv_len;  // a graph without slots is stateless
        conv_.swap(pending_conv_);
        recurrent_.swap(pending_recurrent_);
        for (const auto& out : graph_.outputs()) {
            const HostTensor& t = values[static_cast<size_t>(out.value.node)][static_cast<size_t>(out.value.index)];
            outputs[out.name] = is_leaf(nodes[static_cast<size_t>(out.value.node)].kind) ? t.clone() : t;
        }
    }

    int64_t num_cached() const override { return cached_; }

    void reset() override { truncate(0); }

    void truncate(int64_t n) override {
        XS_CHECK(n >= 0 && n <= cached_, "cannot truncate to ", n, " tokens; ", cached_, " are cached");
        if (n == cached_) return;
        XS_CHECK(n == 0 || !graph_.has_fixed_state(), "cannot truncate to ", n,
                 " tokens: conv and recurrent state summarize the whole sequence");
        resize_kv(n);
        if (n == 0) reset_fixed_state();
        cached_ = n;
    }

  private:
    static bool is_leaf(OpKind kind) {
        return kind == OpKind::Input || kind == OpKind::Weight || kind == OpKind::Constant;
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
        step.kv_len = cached_ + step.tokens;
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
                                      const TensorMap& inputs, const Step& step) {
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
                return {cpu_ref::gather(f32(0), in(1))};
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
                const float eps = std::get<graph::RmsNormAttrs>(n.attrs).eps;
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
                const auto& a = std::get<graph::StateAttrs>(n.attrs);
                return {cpu_ref::causal_conv(f32(0), f32(1), pending_conv_[static_cast<size_t>(a.slot)])};
            }
            case OpKind::GatedDelta: {
                const auto& a = std::get<graph::StateAttrs>(n.attrs);
                return {cpu_ref::gated_delta(*pool_, f32(0), f32(1), f32(2), f32(3), f32(4), a.qk_l2norm_eps,
                                             pending_recurrent_[static_cast<size_t>(a.slot)])};
            }
            case OpKind::KvCache: {
                auto [k_all, v_all] = append_kv(std::get<graph::KvCacheAttrs>(n.attrs).slot, f32(0), f32(1), step);
                return {std::move(k_all), std::move(v_all)};
            }
            case OpKind::Attention:
                return {cpu_ref::attention(*pool_, f32(0), f32(1), f32(2), std::get<graph::AttentionAttrs>(n.attrs))};
            case OpKind::Div:
                return {cpu_ref::div(f32(0), f32(1))};
            case OpKind::Softmax:
                return {cpu_ref::softmax(f32(0))};
            case OpKind::TopK: {
                auto [top_values, top_indices] = cpu_ref::topk(f32(0), std::get<graph::TopKAttrs>(n.attrs).k);
                return {std::move(top_values), std::move(top_indices)};
            }
            case OpKind::ReduceSum:
                return {cpu_ref::reduce_sum(f32(0), std::get<graph::ReduceAttrs>(n.attrs).axis)};
            case OpKind::ExpertMatMul:
                return {cpu_ref::expert_matmul(*pool_, f32(0), f32(1), in(2))};
        }
        fail("cpu-ref: unhandled op ", graph::op_name(n.kind));
    }

    // Appends this step's k, v ([T, Hkv, D]) to the slot and returns the full [L, Hkv, D] tensors.
    std::pair<HostTensor, HostTensor> append_kv(int slot, const HostTensor& k, const HostTensor& v, const Step& step) {
        const graph::KvSlot& s = graph_.kv_slots()[static_cast<size_t>(slot)];
        const auto row = static_cast<size_t>(s.num_kv_heads * s.head_dim);
        KvState& state = kv_[static_cast<size_t>(slot)];
        state.k.resize(static_cast<size_t>(step.kv_len) * row);
        state.v.resize(static_cast<size_t>(step.kv_len) * row);
        const size_t past = static_cast<size_t>(cached_) * row;
        std::copy(k.data<float>(), k.data<float>() + k.numel(), state.k.begin() + static_cast<std::ptrdiff_t>(past));
        std::copy(v.data<float>(), v.data<float>() + v.numel(), state.v.begin() + static_cast<std::ptrdiff_t>(past));
        if (kv_i8_) {
            const int64_t rows = k.numel() / s.head_dim;
            quant::round_trip_kv_i8(state.k.data() + past, rows, s.head_dim);
            quant::round_trip_kv_i8(state.v.data() + past, rows, s.head_dim);
        }
        const Dims shape{step.kv_len, s.num_kv_heads, s.head_dim};
        return {HostTensor::from_vector(DType::F32, shape, state.k),
                HostTensor::from_vector(DType::F32, shape, state.v)};
    }

    void reset_fixed_state() {
        conv_.clear();
        for (const graph::ConvSlot& c : graph_.conv_slots())
            conv_.emplace_back(static_cast<size_t>((c.kernel - 1) * c.channels), 0.0f);
        recurrent_.clear();
        for (const graph::RecurrentSlot& r : graph_.recurrent_slots())
            recurrent_.emplace_back(static_cast<size_t>(r.heads * r.key_dim * r.value_dim), 0.0f);
    }

    void resize_kv(int64_t tokens) {
        for (size_t slot = 0; slot < kv_.size(); ++slot) {
            const graph::KvSlot& s = graph_.kv_slots()[slot];
            const auto size = static_cast<size_t>(tokens * s.num_kv_heads * s.head_dim);
            kv_[slot].k.resize(std::min(size, kv_[slot].k.size()));
            kv_[slot].v.resize(std::min(size, kv_[slot].v.size()));
        }
    }

    const graph::Graph graph_;
    const std::vector<HostTensor> leaves_;  // f32 weights and normalized constants by node index; empty otherwise
    std::vector<size_t> last_use_;          // index of the last node reading each node; outputs live to the end
    const std::shared_ptr<cpu_ref::ThreadPool> pool_;
    const int64_t max_context_;
    const bool kv_i8_;  // the cache holds each appended row as its int8 KV value
    std::vector<KvState> kv_;
    std::vector<std::vector<float>> conv_, recurrent_;                  // per slot, committed state
    std::vector<std::vector<float>> pending_conv_, pending_recurrent_;  // per slot, the running step's state
    int64_t cached_ = 0;
};

class CpuBackend final : public Backend {
  public:
    CpuBackend()
        : pool_(std::make_shared<cpu_ref::ThreadPool>(std::max(1u, std::thread::hardware_concurrency()) - 1)) {}

    std::string_view name() const override { return "cpu-ref"; }

    std::vector<DeviceInfo> devices() const override { return {DeviceInfo{"CPU", cpu_model_name(), 0, false}}; }

    // Computation is always f32 whatever options.precision says: this backend is the accuracy oracle.
    std::unique_ptr<Session> compile(const graph::Graph& graph, const WeightStore& weights,
                                     const SessionOptions& options) override {
        graph.validate();
        check_options(options);
        const auto& nodes = graph.nodes();
        std::vector<HostTensor> leaves(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind == OpKind::Weight) {
                const std::string& name = std::get<graph::WeightAttrs>(n.attrs).name;
                const TensorView& view = weights.get(name);
                XS_CHECK(view.shape == n.outputs[0].shape, "weight '", name, "' has shape ", dims_to_string(view.shape),
                         " but the graph was built for ", dims_to_string(n.outputs[0].shape));
                leaves[i] = weight_to_f32(name, view);
            } else if (n.kind == OpKind::Constant) {
                leaves[i] = normalize_dtype(std::get<graph::ConstantAttrs>(n.attrs).value);
            }
        }
        return std::make_unique<CpuSession>(graph, std::move(leaves), pool_, options.max_context,
                                            options.kv_dtype == DType::I8);
    }

  private:
    std::shared_ptr<cpu_ref::ThreadPool> pool_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_ref_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace xs
