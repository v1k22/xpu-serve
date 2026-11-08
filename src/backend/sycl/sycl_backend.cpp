#include "backend/sycl/sycl_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "backend/sycl/sycl_kernels.h"
#include "backend/sycl/sycl_matmul.h"
#include "util/error.h"
#include "util/half.h"
#include "util/log.h"

namespace xs {

namespace {

using graph::kL;
using graph::kT;
using graph::Node;
using graph::OpKind;
using graph::Value;
using sycl_be::half;

constexpr size_t kAlign = 256;

size_t align_up(size_t n) {
    return (n + kAlign - 1) & ~(kAlign - 1);
}

std::vector<sycl::device> gpu_devices() {
    std::vector<sycl::device> out;
    for (const sycl::platform& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const sycl::device& d : p.get_devices(sycl::info::device_type::gpu))
            out.push_back(d);
    }
    return out;
}

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::string(v) != "0";
}

// Live USM device allocations of this process (all sessions), for out-of-memory reports.
std::atomic<size_t> g_device_bytes{0};
std::atomic<size_t> g_device_buffers{0};

// One USM device allocation. The owner makes sure the queue no longer uses it before it goes away.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(sycl::queue& q, size_t bytes) : q_(&q), bytes_(bytes) {
        if (bytes == 0) return;
        ptr_ = sycl::malloc_device(bytes, q);
        XS_CHECK(ptr_ != nullptr, "sycl: out of device memory allocating ", bytes, " bytes (", g_device_bytes.load() >> 20,
                 " MiB already in ", g_device_buffers.load(), " buffers)");
        g_device_bytes += bytes;
        ++g_device_buffers;
    }
    DeviceBuffer(DeviceBuffer&& o) noexcept
        : q_(o.q_), ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            q_ = o.q_;
            ptr_ = std::exchange(o.ptr_, nullptr);
            bytes_ = std::exchange(o.bytes_, 0);
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { release(); }

    char* get() const { return static_cast<char*>(ptr_); }
    size_t bytes() const { return bytes_; }

private:
    void release() {
        if (ptr_ != nullptr) {
            sycl::free(ptr_, *q_);
            g_device_bytes -= bytes_;
            --g_device_buffers;
        }
        ptr_ = nullptr;
        bytes_ = 0;
    }
    sycl::queue* q_ = nullptr;
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

// Sizes of the symbolic dims in one run.
struct Step {
    int64_t tokens = 0;  // kT
    int64_t kv_len = 0;  // kL
};

// Where a value lives on the device: an offset in the step's arena, inside a weight or constant (Leaf, by node), or
// in a KV slot's k or v buffer.
enum class Store : uint8_t { Arena, Leaf, KvK, KvV };
struct Loc {
    Store store = Store::Arena;
    int32_t index = 0;  // arena block while planning, leaf node, or KV slot
    int64_t offset = 0;
};

// Memory plan of one step shape: every value's resolved shape and location, and the arena size.
struct Plan {
    std::vector<std::vector<Dims>> shapes;
    std::vector<std::vector<Loc>> locs;
    std::vector<uint8_t> alias;  // the node only names memory another value owns: nothing to run
    size_t arena_bytes = 0;
};

struct KvBuffers {
    DeviceBuffer k, v;
    DeviceBuffer k_scale, v_scale;  // I8: one f32 per (token, KV head)
    int64_t capacity = 0;  // tokens
    int64_t heads = 0;
    int64_t row = 0;       // elements per token: heads * head_dim
    DType dtype = DType::F16;
};

// A scratch buffer that grows on demand (sycl_be::Scratch).
class GrowingScratch final : public sycl_be::Scratch {
public:
    explicit GrowingScratch(sycl::queue& q) : q_(q) {}
    void* get(size_t bytes) override {
        if (bytes > buf_.bytes()) {
            q_.wait();
            buf_ = DeviceBuffer(q_, bytes + bytes / 4);
        }
        return buf_.get();
    }
    void* get_host(size_t) override { fail("sycl: this scratch has no host memory"); }

private:
    sycl::queue& q_;
    DeviceBuffer buf_;
};

// Conv and recurrent state: the step reads buf[cur] and writes buf[1 - cur]; a successful step flips cur, so a
// failed one leaves the state as it was.
struct StateBuffers {
    DeviceBuffer buf[2];
    int cur = 0;
    size_t bytes = 0;
};

bool is_suffix_broadcast(const Dims& b, const Dims& out) {
    if (b.size() > out.size()) return false;
    const size_t pad = out.size() - b.size();
    bool body = false;
    for (size_t d = 0; d < out.size(); ++d) {
        const int64_t bd = d < pad ? 1 : b[d - pad];
        if (!body && bd == 1 && out[d] != 1) continue;
        body = true;
        if (bd != out[d]) return false;
    }
    return true;
}

class SyclSession final : public Session, private sycl_be::Scratch {
public:
    SyclSession(const graph::Graph& graph, const WeightStore& weights, const SessionOptions& options, sycl::queue q,
                const sycl_be::AttentionPlan& attention)
        : graph_(graph),
          q_(std::move(q)),
          attention_(attention),
          precision_(options.precision),
          kv_i8_(options.kv_dtype == DType::I8),
          max_context_(options.max_context),
          profiling_(env_flag("XS_SYCL_PROFILE")),
          sync_(profiling_ || env_flag("XS_SYCL_SYNC")),
          trace_(env_flag("XS_SYCL_TRACE")) {
        // Plans are cached per T. They depend on L too when anything but a KvCache op makes an L-long value or
        // anything but Attention reads one; otherwise L only reaches Attention (and outputs), resolved per run.
        const auto has_l = [](const Dims& d) { return std::find(d.begin(), d.end(), kL) != d.end(); };
        for (const Node& n : graph_.nodes()) {
            if (n.kind != OpKind::KvCache) {
                for (const auto& t : n.outputs)
                    l_dependent_ |= has_l(t.shape);
            }
            if (n.kind != OpKind::Attention) {
                for (const Value& in : n.inputs)
                    l_dependent_ |= has_l(graph_.type(in).shape);
            }
        }
        infer_dtypes();
        check_ops();
        plan_fusions();
        plan_half_values();
        upload_leaves(weights);
        for (const graph::KvSlot& s : graph_.kv_slots()) {
            KvBuffers b;
            b.heads = s.num_kv_heads;
            b.row = s.num_kv_heads * s.head_dim;
            kv_.push_back(std::move(b));
        }
        size_t int8_slots = 0;
        for (size_t i = 0; i < graph_.nodes().size(); ++i) {
            const Node& n = graph_.nodes()[i];
            if (n.kind != OpKind::KvCache) continue;
            kv_[static_cast<size_t>(std::get<graph::KvCacheAttrs>(n.attrs).slot)].dtype = dtype_[i][0];
            int8_slots += dtype_[i][0] == DType::I8 ? 1 : 0;
        }
        if (kv_i8_) log::info("sycl: int8 KV cache for ", int8_slots, " of ", kv_.size(), " slots");
        for (const graph::ConvSlot& c : graph_.conv_slots())
            fixed_state_.push_back(make_state(static_cast<size_t>((c.kernel - 1) * c.channels) * sizeof(float)));
        conv_states_ = fixed_state_.size();
        for (const graph::RecurrentSlot& r : graph_.recurrent_slots())
            fixed_state_.push_back(make_state(static_cast<size_t>(r.heads * r.key_dim * r.value_dim) * sizeof(float)));
        zero_fixed_state();
    }

    ~SyclSession() override {
        try {
            q_.wait();
        } catch (...) {
        }
        if (host_ != nullptr) sycl::free(host_, q_);
        if (profiling_) report_profile();
    }

    void run(const TensorMap& inputs, TensorMap& outputs) override {
        const Step step = check_inputs(inputs);
        XS_CHECK(step.kv_len <= max_context_, "sycl: ", step.kv_len, " tokens exceed max_context ", max_context_);
        std::vector<HostTensor> keep_alive;  // host sources of pending copies
        try {
            ensure_kv_capacity(step.kv_len);
            const Plan& plan = plan_for(step);
            ensure_arena(plan.arena_bytes);
            for (const Value& v : graph_.inputs()) {
                const HostTensor& t = inputs.at(input_name(v));
                const HostTensor& src = keep_alive.emplace_back(
                    is_float(t.dtype()) && t.dtype() != DType::F32
                        ? HostTensor::from_vector(DType::F32, t.shape(), t.to_f32())
                        : t);
                q_.memcpy(ptr(plan, v), src.raw(), src.nbytes());
                if (trace_) {
                    const Loc& l = plan.locs[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)];
                    std::fprintf(stderr, "sycl: input %s %zu bytes at arena offset %lld of %zu (store %d)\n",
                                 input_name(v).c_str(), src.nbytes(), static_cast<long long>(l.offset), arena_.bytes(),
                                 static_cast<int>(l.store));
                    q_.wait_and_throw();
                }
            }
            const auto& nodes = graph_.nodes();
            const auto submit_start = std::chrono::steady_clock::now();
            // The first run of a step shape compiles kernels (JIT, oneMKL): not part of the profile.
            const bool profile_step = profiling_ && !profiled_shapes_.insert(step.tokens).second;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (plan.alias[i] != 0 || nodes[i].kind == OpKind::Input) continue;
                if (profile_step) {
                    timed_launch(i, plan, step);
                } else if (sync_) {
                    try {
                        if (trace_) std::fprintf(stderr, "%s T=%lld L=%lld\n", where(i).c_str(),
                                                 static_cast<long long>(step.tokens), static_cast<long long>(step.kv_len));
                        launch(i, plan, step);
                        q_.wait_and_throw();
                    } catch (const std::exception& e) {
                        fail(where(i), " failed: ", e.what());
                    }
                } else {
                    launch(i, plan, step);
                }
            }
            if (log::level() <= log::Level::Debug) {  // host-bound when submitting takes about as long as the run
                const auto submitted = std::chrono::steady_clock::now();
                q_.wait();
                log::debug("sycl: T=", step.tokens, " ops submitted in ",
                           std::chrono::duration<double, std::milli>(submitted - submit_start).count(), " ms, done after ",
                           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submit_start).count(),
                           " ms");
            }
            std::vector<std::pair<std::string, HostTensor>> results;
            for (const auto& out : graph_.outputs()) {
                const Value v = out.value;
                const DType dt = dtype_[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)];
                HostTensor t(dt, resolve(plan.shapes[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)], step));
                q_.memcpy(t.raw(), ptr(plan, v), t.nbytes());
                results.emplace_back(out.name, std::move(t));
            }
            q_.wait_and_throw();
            for (auto& [name, t] : results)
                outputs[name] = std::move(t);
        } catch (...) {
            try {
                q_.wait();
            } catch (...) {
            }
            throw;
        }
        if (graph_.stateful()) cached_ = step.kv_len;  // a graph without slots is stateless
        for (StateBuffers& s : fixed_state_)
            s.cur = 1 - s.cur;
        if (profiling_ && profiled_shapes_.count(step.tokens) != 0 && profiled_runs_[step.tokens]++ > 0) {
            ++(step.tokens == 1 ? decode_steps_ : prefill_steps_);
        }
    }

    int64_t num_cached() const override { return cached_; }

    void reset() override { truncate(0); }

    void truncate(int64_t n) override {
        XS_CHECK(n >= 0 && n <= cached_, "cannot truncate to ", n, " tokens; ", cached_, " are cached");
        if (n == cached_) return;
        XS_CHECK(n == 0 || !graph_.has_fixed_state(), "cannot truncate to ", n,
                 " tokens: conv and recurrent state summarize the whole sequence");
        if (n == 0) zero_fixed_state();
        cached_ = n;
    }

private:
    // sycl_be::Scratch
    void* get(size_t bytes) override {
        if (bytes > scratch_.bytes()) {
            q_.wait();
            scratch_ = DeviceBuffer(q_, align_up(bytes + bytes / 4));
        }
        return scratch_.get();
    }
    void* get_host(size_t bytes) override {
        if (bytes > host_bytes_) {
            q_.wait();
            if (host_ != nullptr) sycl::free(host_, q_);
            host_bytes_ = align_up(bytes);
            host_ = sycl::malloc_host(host_bytes_, q_);
            XS_CHECK(host_ != nullptr, "sycl: out of pinned host memory allocating ", host_bytes_, " bytes");
        }
        return host_;
    }

    std::string where(size_t i) const {
        const Node& n = graph_.nodes()[i];
        return str_cat("sycl: node ", i, " (", n.label, ": ", graph::op_name(n.kind), ")");
    }

    DType dt(const Value& v) const { return dtype_[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)]; }

    const std::string& input_name(Value v) const { return std::get<graph::InputAttrs>(graph_.node(v).attrs).name; }

    // A weight stays f16 when every op that reads it (through reshapes, slices and transposes) is a matmul weight or
    // a gather table, a KV slot when only attention reads it as k or v; anything else reads f32.
    bool f16_readers_only(size_t node, const std::vector<std::vector<std::pair<size_t, size_t>>>& consumers) const {
        const auto& nodes = graph_.nodes();
        for (const auto& out : graph_.outputs()) {
            if (static_cast<size_t>(out.value.node) == node) return false;
        }
        bool any = false;
        std::vector<size_t> stack{node};
        while (!stack.empty()) {
            const size_t p = stack.back();
            stack.pop_back();
            for (const auto& [c, k] : consumers[p]) {
                const OpKind kind = nodes[c].kind;
                if (kind == OpKind::Reshape || kind == OpKind::Slice || kind == OpKind::Transpose) {
                    stack.push_back(c);
                    continue;
                }
                if (!((kind == OpKind::MatMul && k == 1) || (kind == OpKind::Gather && k == 0) ||
                      (kind == OpKind::ExpertMatMul && k == 1) || (kind == OpKind::Attention && k > 0))) {
                    return false;
                }
                any = true;
            }
        }
        return any;
    }

    // An int8 KV slot needs every reader of its k / v to be an Attention reading them directly (it applies the scales).
    bool attention_reads_only(size_t node, const std::vector<std::vector<std::pair<size_t, size_t>>>& consumers) const {
        for (const auto& out : graph_.outputs()) {
            if (static_cast<size_t>(out.value.node) == node) return false;
        }
        const auto& nodes = graph_.nodes();
        for (const auto& [c, k] : consumers[node]) {
            const int index = nodes[c].inputs[k].index;
            if (nodes[c].kind != OpKind::Attention || static_cast<int>(k) != index + 1) return false;
        }
        return !consumers[node].empty();
    }

    void infer_dtypes() {
        const auto& nodes = graph_.nodes();
        std::vector<std::vector<std::pair<size_t, size_t>>> consumers(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t k = 0; k < nodes[i].inputs.size(); ++k)
                consumers[static_cast<size_t>(nodes[i].inputs[k].node)].emplace_back(i, k);
        }
        const auto device_type = [&](size_t i, DType t) {
            if (is_quantized(t)) return t;  // check_ops makes sure only matmuls and gathers read it
            if (is_float(t)) return DType::F32;
            XS_CHECK(t == DType::I32 || t == DType::I64, where(i), ": dtype ", dtype_name(t), " is not supported");
            return t;
        };
        dtype_.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            dtype_[i].assign(n.outputs.size(), DType::F32);
            switch (n.kind) {
                case OpKind::Input:
                case OpKind::Constant:
                    dtype_[i][0] = device_type(i, n.outputs[0].dtype);
                    break;
                case OpKind::Weight:
                    dtype_[i][0] = device_type(i, n.outputs[0].dtype);
                    if (dtype_[i][0] == DType::F32 && precision_ == DType::F16 && f16_readers_only(i, consumers)) {
                        dtype_[i][0] = DType::F16;
                    }
                    break;
                case OpKind::Reshape:
                case OpKind::Slice:
                case OpKind::Transpose:
                    dtype_[i][0] = dt(n.inputs[0]);
                    break;
                case OpKind::KvCache:
                    if (kv_i8_ && attention_reads_only(i, consumers)) {
                        dtype_[i][0] = dtype_[i][1] = DType::I8;
                    } else {
                        dtype_[i][0] = dtype_[i][1] =
                            precision_ == DType::F16 && f16_readers_only(i, consumers) ? DType::F16 : DType::F32;
                    }
                    break;
                case OpKind::TopK:
                    dtype_[i][1] = DType::I32;
                    break;
                case OpKind::Concat:
                    dtype_[i][0] = dt(n.inputs[0]);
                    break;
                default:
                    break;
            }
        }
    }

    // Operand dtypes and the shapes the kernels support.
    void check_ops() const {
        const auto& nodes = graph_.nodes();
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            const auto need = [&](size_t k, bool ok, const char* what) {
                XS_CHECK(ok, where(i), ": input ", k, " is ", dtype_name(dt(n.inputs[k])), " on the device; needs ", what);
            };
            const auto is_int = [&](size_t k) { return dt(n.inputs[k]) == DType::I32 || dt(n.inputs[k]) == DType::I64; };
            const auto is_f32 = [&](size_t k) { return dt(n.inputs[k]) == DType::F32; };
            const auto is_fp = [&](size_t k) { return dt(n.inputs[k]) == DType::F32 || dt(n.inputs[k]) == DType::F16; };
            // A quantized weight is read straight from its Weight node (no views) by a matmul or a gather.
            const auto is_packed = [&](size_t k) {
                return is_quantized(dt(n.inputs[k])) && graph_.node(n.inputs[k]).kind == OpKind::Weight;
            };
            switch (n.kind) {
                case OpKind::Input:
                case OpKind::Weight:
                case OpKind::Constant:
                    break;
                case OpKind::Reshape:
                case OpKind::Slice:
                case OpKind::Transpose:
                    need(0, !is_quantized(dt(n.inputs[0])), "an unquantized tensor");
                    break;
                case OpKind::Gather:
                    need(0, is_fp(0) || is_packed(0), "f16, f32, q8 or q4");
                    need(1, is_int(1), "i32 or i64");
                    break;
                case OpKind::MatMul: {
                    need(0, is_f32(0), "f32");
                    need(1, is_fp(1) || is_packed(1), "f16, f32, q8 or q4");
                    const int64_t k = graph_.type(n.inputs[1]).shape[1];
                    XS_CHECK(!is_packed(1) || k % 32 == 0, where(i), ": a quantized weight needs a multiple of 32 "
                             "columns, got ", k);
                    break;
                }
                case OpKind::Concat:
                    for (size_t k = 1; k < n.inputs.size(); ++k)
                        need(k, dt(n.inputs[k]) == dt(n.inputs[0]), "the dtype of input 0");
                    need(0, !is_quantized(dt(n.inputs[0])), "an unquantized tensor");
                    break;
                case OpKind::ExpertMatMul: {
                    need(0, is_f32(0), "f32");
                    need(1, is_fp(1) || is_packed(1), "f16, f32, q8 or q4");
                    need(2, is_int(2), "i32 or i64");
                    const int64_t k = graph_.type(n.inputs[1]).shape[2];
                    XS_CHECK(!is_packed(1) || k % 32 == 0, where(i), ": a quantized expert bank needs a multiple of 32 "
                             "columns, got ", k);
                    break;
                }
                case OpKind::Rope:
                    need(0, is_f32(0), "f32");
                    need(1, is_int(1), "i32 or i64");
                    need(2, is_f32(2), "f32");
                    break;
                case OpKind::Attention: {
                    need(0, is_f32(0), "f32");
                    need(1, is_fp(1) || dt(n.inputs[1]) == DType::I8, "f16, f32 or i8");
                    need(2, dt(n.inputs[2]) == dt(n.inputs[1]), "the dtype of k");
                    const int64_t d = graph_.type(n.inputs[0]).shape.back();
                    XS_CHECK(d % 8 == 0 && d <= 512, where(i), ": head_dim ", d,
                             " is not supported (a multiple of 8, at most 512)");
                    if (n.inputs.size() == 4) need(3, is_f32(3), "f32");
                    break;
                }
                case OpKind::GatedDelta: {
                    for (size_t k = 0; k < n.inputs.size(); ++k)
                        need(k, is_f32(k), "f32");
                    const int64_t dk = graph_.type(n.inputs[0]).shape.back();
                    const int64_t dv = graph_.type(n.inputs[2]).shape.back();
                    XS_CHECK(dk % 32 == 0 && dk <= 512 && ((dk / 32) & (dk / 32 - 1)) == 0, where(i), ": key dim ", dk,
                             " is not supported (32 times a power of two, up to 512)");
                    XS_CHECK(dv % 16 == 0 && dv * (dk / 32) <= 1024, where(i), ": value dim ", dv, " with key dim ", dk,
                             " is not supported (value dim a multiple of 16, value dim * key dim / 32 <= 1024)");
                    break;
                }
                default:
                    for (size_t k = 0; k < n.inputs.size(); ++k)
                        need(k, is_f32(k), "f32");
                    break;
            }
        }
        for (const auto& out : graph_.outputs()) {
            const DType d = dt(out.value);
            XS_CHECK(d == DType::F32 || d == DType::I32 || d == DType::I64, "sycl: graph output '", out.name,
                     "' would be ", dtype_name(d), " on the device");
        }
    }

    // Mul(act(x), b) with act in {Silu, GeluTanh, Sigmoid} read only by that Mul and all shapes equal runs as one
    // act_mul; Silu(CausalConv) with the conv read only by the Silu runs inside the conv.
    void plan_fusions() {
        const auto& nodes = graph_.nodes();
        std::vector<int> readers(nodes.size(), 0);
        for (const Node& n : nodes) {
            for (const Value& in : n.inputs)
                ++readers[static_cast<size_t>(in.node)];
        }
        for (const auto& out : graph_.outputs())
            readers[static_cast<size_t>(out.value.node)] += 2;  // never fuse away a graph output
        fused_.assign(nodes.size(), Fused::No);
        act_mul_.assign(nodes.size(), -1);
        conv_silu_.assign(nodes.size(), 0);
        norm_bias_.assign(nodes.size(), -1);
        add_mul_.assign(nodes.size(), -1);
        unary_bias_.assign(nodes.size(), -1);
        add3_.assign(nodes.size(), -1);
        reads_.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
            reads_[i] = nodes[i].inputs;
        const auto act_of = [](OpKind k) -> int {
            switch (k) {
                case OpKind::Silu: return static_cast<int>(sycl_be::UnaryOp::Silu);
                case OpKind::GeluTanh: return static_cast<int>(sycl_be::UnaryOp::GeluTanh);
                case OpKind::Sigmoid: return static_cast<int>(sycl_be::UnaryOp::Sigmoid);
                default: return -1;
            }
        };
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind == OpKind::Mul) {
                const Dims& out = n.outputs[0].shape;
                for (size_t k = 0; k < 2; ++k) {
                    const Value a = n.inputs[k];
                    const Value b = n.inputs[1 - k];
                    const Node& an = graph_.node(a);
                    const int act = act_of(an.kind);
                    const auto u = static_cast<size_t>(a.node);
                    if (act < 0 || readers[u] != 1 || fused_[u] != Fused::No) continue;
                    if (graph_.type(an.inputs[0]).shape != out || graph_.type(b).shape != out) continue;
                    fused_[u] = Fused::Skip;
                    act_mul_[i] = static_cast<int8_t>(act);
                    reads_[i] = {an.inputs[0], b};
                    reads_[u].clear();
                    break;
                }
            } else if (n.kind == OpKind::Silu && fused_[i] == Fused::No) {
                const auto c = static_cast<size_t>(n.inputs[0].node);
                if (nodes[c].kind == OpKind::CausalConv && readers[c] == 1) {
                    conv_silu_[c] = 1;
                    fused_[i] = Fused::Alias;
                }
            }
        }
        // act(Add(x, b)) with b a row vector and the Add read only by act: one act(x + b) (a matmul's bias).
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (act_of(n.kind) < 0 || fused_[i] != Fused::No) continue;
            const auto u = static_cast<size_t>(n.inputs[0].node);
            const Node& add = nodes[u];
            if (add.kind != OpKind::Add || readers[u] != 1 || fused_[u] != Fused::No) continue;
            const Dims& out = n.outputs[0].shape;
            for (size_t k = 0; k < 2; ++k) {
                const Dims& bs = graph_.type(add.inputs[1 - k]).shape;
                if (graph_.type(add.inputs[k]).shape != out || bs.size() != 1 || bs[0] != out.back()) continue;
                fused_[u] = Fused::Skip;
                unary_bias_[i] = static_cast<int32_t>(u);
                reads_[i] = {add.inputs[k], add.inputs[1 - k]};
                reads_[u].clear();
                break;
            }
        }
        // Add(RmsNorm(x, w), c) with c a row vector: one norm with a bias (adaptive norms' shift). Add(a, Mul(b, c))
        // with a and b of the output's shape and c the same or a row vector: one a + b * c (gated residuals).
        // Add(a, Add(b, c)) with c a row vector: one a + (b + c) (a residual and a biased matmul).
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind != OpKind::Add || fused_[i] != Fused::No) continue;
            const Dims& out = n.outputs[0].shape;
            const auto row_vector = [&](Value v) {
                const Dims& s = graph_.type(v).shape;
                return s.size() == 1 && !out.empty() && s[0] == out.back();
            };
            for (size_t k = 0; k < 2; ++k) {
                const Value x = n.inputs[k];
                const Value other = n.inputs[1 - k];
                const auto u = static_cast<size_t>(x.node);
                if (readers[u] != 1 || fused_[u] != Fused::No || graph_.type(x).shape != out) continue;
                if (nodes[u].kind == OpKind::RmsNorm && row_vector(other)) {
                    fused_[u] = Fused::Skip;
                    norm_bias_[i] = static_cast<int32_t>(u);
                    reads_[i] = nodes[u].inputs;
                    reads_[i].push_back(other);
                    reads_[u].clear();
                    break;
                }
                if (nodes[u].kind == OpKind::Add && graph_.type(other).shape == out && norm_bias_[u] < 0 &&
                    add_mul_[u] < 0 && add3_[u] < 0) {
                    const Value b = nodes[u].inputs[0];
                    const Value c = nodes[u].inputs[1];
                    const bool c_row = graph_.type(b).shape == out && row_vector(c);
                    const bool b_row = graph_.type(c).shape == out && row_vector(b);
                    if (!c_row && !b_row) continue;
                    fused_[u] = Fused::Skip;
                    add3_[i] = static_cast<int32_t>(u);
                    reads_[i] = c_row ? std::vector<Value>{other, b, c} : std::vector<Value>{other, c, b};
                    reads_[u].clear();
                    break;
                }
                if (nodes[u].kind == OpKind::Mul && act_mul_[u] < 0 && graph_.type(other).shape == out) {
                    const Value b = nodes[u].inputs[0];
                    const Value c = nodes[u].inputs[1];
                    const bool b_full = graph_.type(b).shape == out;
                    const bool c_full = graph_.type(c).shape == out;
                    if (!(b_full && (c_full || row_vector(c))) && !(c_full && row_vector(b))) continue;
                    fused_[u] = Fused::Skip;
                    add_mul_[i] = static_cast<int32_t>(u);
                    reads_[i] = b_full ? std::vector<Value>{other, b, c} : std::vector<Value>{other, c, b};
                    reads_[u].clear();
                    break;
                }
            }
        }
    }

    // Values that only GEMM-path matmuls (as x) and GEMM-path attention (as q, k, v) read are kept in f16, written so
    // by the kernel that makes them: those GEMMs run on f16 operands anyway, so a conversion pass disappears and the
    // traffic halves. Attention operands produced in f16 (projections, RoPE) round q, k and v to f16 before the
    // scores, the convention of the f16 KV cache. Static shapes only (dynamic ones can take the f32-only GEMV and
    // per-row paths). A fixed point: start from every eligible value, drop any whose readers cannot take f16.
    void plan_half_values() {
        if (precision_ != DType::F16) return;
        const auto& nodes = graph_.nodes();
        const size_t count = nodes.size();
        std::vector<std::vector<std::pair<size_t, size_t>>> readers(count);  // (reader, input index) of output 0
        for (size_t i = 0; i < count; ++i) {
            for (size_t k = 0; k < reads_[i].size(); ++k) {
                if (reads_[i][k].index == 0) readers[static_cast<size_t>(reads_[i][k].node)].emplace_back(i, k);
            }
        }
        std::vector<uint8_t> output(count, 0);
        for (const auto& o : graph_.outputs())
            output[static_cast<size_t>(o.value.node)] = 1;
        const auto static_shape = [](const Dims& d) {
            return std::none_of(d.begin(), d.end(), [](int64_t v) { return v == kT || v == kL; });
        };
        const auto gemm_matmul = [&](size_t m) {  // the matmul runs on the GEMM path with an f16 weight
            const Node& n = nodes[m];
            if (n.kind != OpKind::MatMul || dt(n.inputs[1]) != DType::F16) return false;
            const Dims& x = graph_.type(n.inputs[0]).shape;
            return static_shape(x) && numel(x) / x.back() > sycl_be::kGemvMaxRows;
        };
        const auto gemm_attention = [&](size_t a) {
            const Node& n = nodes[a];
            if (n.kind != OpKind::Attention || attention_.path == sycl_be::AttentionPath::Rows) return false;
            const Dims& q = graph_.type(n.inputs[0]).shape;
            const Dims& k = graph_.type(n.inputs[1]).shape;
            if (!static_shape(q) || !static_shape(k) || dt(n.inputs[1]) == DType::I8) return false;
            sycl_be::AttentionShape sh;
            sh.tokens = q[0];
            sh.head_dim = q[2];
            return sycl_be::attention_matrix_path(sh, attention_.path);
        };
        std::vector<uint8_t> half(count, 0);
        for (size_t i = 0; i < count; ++i) {
            const OpKind k = nodes[i].kind;
            const bool producer = k == OpKind::RmsNorm || k == OpKind::LayerNorm || k == OpKind::Add ||
                                  k == OpKind::Mul || k == OpKind::GeluTanh || k == OpKind::Silu ||
                                  k == OpKind::Rope || k == OpKind::Reshape || k == OpKind::Slice ||
                                  k == OpKind::Transpose || k == OpKind::Concat || (k == OpKind::MatMul && gemm_matmul(i));
            half[i] = producer && fused_[i] == Fused::No && dtype_[i][0] == DType::F32 && output[i] == 0 &&
                      nodes[i].outputs.size() == 1 && static_shape(nodes[i].outputs[0].shape) && !readers[i].empty();
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t i = 0; i < count; ++i) {
                if (half[i] == 0) continue;
                const Node& n = nodes[i];
                bool ok = true;
                if (n.kind == OpKind::Reshape || n.kind == OpKind::Slice || n.kind == OpKind::Transpose) {
                    ok = half[static_cast<size_t>(n.inputs[0].node)];
                }
                if (n.kind == OpKind::Concat) {
                    for (const Value& in : n.inputs)
                        ok = ok && half[static_cast<size_t>(in.node)] != 0;
                }
                for (const auto& [r, k] : readers[i]) {
                    const OpKind rk = nodes[r].kind;
                    bool accept = false;
                    if (rk == OpKind::MatMul) {
                        accept = k == 0 && fused_[r] == Fused::No && gemm_matmul(r);
                    } else if (rk == OpKind::Attention) {
                        const auto kv = [&](size_t idx) { return half[static_cast<size_t>(nodes[r].inputs[idx].node)] != 0; };
                        accept = gemm_attention(r) && (k == 0 || ((k == 1 || k == 2) && kv(1) && kv(2)));
                    } else if (rk == OpKind::Reshape || rk == OpKind::Slice || rk == OpKind::Transpose ||
                               rk == OpKind::Concat) {
                        accept = half[r] != 0;
                    } else if (rk == OpKind::Mul && act_mul_[r] >= 0) {  // a fused activation reads both in one type
                        accept = half[static_cast<size_t>(reads_[r][0].node)] != 0 &&
                                 half[static_cast<size_t>(reads_[r][1].node)] != 0;
                    }
                    ok = ok && accept;
                }
                if (!ok) {
                    half[i] = 0;
                    changed = true;
                }
            }
        }
        size_t halved = 0;
        for (size_t i = 0; i < count; ++i) {
            if (half[i] != 0) {
                dtype_[i][0] = DType::F16;
                ++halved;
            }
        }
        for (size_t i = 0; i < count; ++i) {  // views take their input's dtype
            const OpKind k = nodes[i].kind;
            if (k == OpKind::Reshape || k == OpKind::Slice || k == OpKind::Transpose) dtype_[i][0] = dt(nodes[i].inputs[0]);
        }
        if (halved > 0) log::debug("sycl: ", halved, " values kept in f16 for GEMMs and attention");
    }

    // A pack's quantized weight: codes, scales and (Q4) biases as stored.
    void upload_quantized(size_t node, const TensorView& view) {
        const int64_t cols = view.shape.back();
        const int64_t rows = numel(view.shape) / cols;  // an expert bank [E, N, K] is E * N rows
        const size_t scale_bytes = static_cast<size_t>(rows * (cols / view.quant.group)) * sizeof(uint16_t);
        const auto copy = [&](const void* src, size_t bytes) {
            leaves_.emplace_back(q_, std::max<size_t>(bytes, 1));
            q_.memcpy(leaves_.back().get(), src, bytes).wait();
            return leaves_.back().get();
        };
        sycl_be::QuantWeight w;
        w.q4 = view.dtype == DType::Q4;
        w.group = view.quant.group;
        w.codes = reinterpret_cast<const uint8_t*>(copy(view.data, view.nbytes()));
        w.scales = reinterpret_cast<const half*>(copy(view.quant.scales, scale_bytes));
        if (w.q4) w.biases = reinterpret_cast<const half*>(copy(view.quant.biases, scale_bytes));
        if (view.quant.input_scales != nullptr) {
            w.input_scales = reinterpret_cast<const float*>(
                copy(view.quant.input_scales, static_cast<size_t>(cols) * sizeof(float)));
        }
        leaf_ptr_[node] = const_cast<char*>(reinterpret_cast<const char*>(w.codes));
        leaf_quant_[node] = w;
    }

    DeviceBuffer upload(const std::string& name, const void* data, DType src, const Dims& shape, DType dst) {
        const size_t n = static_cast<size_t>(numel(shape));
        DeviceBuffer buf(q_, std::max<size_t>(n * dtype_size(dst), 1));
        std::vector<uint16_t> h16;
        std::vector<float> h32;
        const void* host = data;
        if (src != dst) {
            const auto* s16 = static_cast<const uint16_t*>(data);
            if (dst == DType::F16) {
                h16.resize(n);
                if (src == DType::BF16) {
                    if (const size_t overflow = count_f16_overflow(s16, n); overflow > 0) {
                        log::warn("weight '", name, "': ", overflow, " bf16 values exceed the f16 range and become inf");
                    }
                    convert_bf16_to_f16(s16, h16.data(), n);
                } else if (src == DType::F32) {
                    convert_f32_to_f16(static_cast<const float*>(data), h16.data(), n);
                } else {
                    fail("sycl: weight '", name, "' has dtype ", dtype_name(src), "; supported: f32, f16, bf16");
                }
                host = h16.data();
            } else {
                h32.resize(n);
                if (src == DType::BF16) {
                    convert_bf16_to_f32(s16, h32.data(), n);
                } else if (src == DType::F16) {
                    convert_f16_to_f32(s16, h32.data(), n);
                } else {
                    fail("sycl: weight '", name, "' has dtype ", dtype_name(src), "; supported: f32, f16, bf16");
                }
                host = h32.data();
            }
        }
        // In pieces: a single 1.6 GB copy from pageable host memory lost the device.
        constexpr size_t kPiece = size_t{256} << 20;
        const size_t total = n * dtype_size(dst);
        for (size_t off = 0; off < total; off += kPiece)
            q_.memcpy(buf.get() + off, static_cast<const char*>(host) + off, std::min(kPiece, total - off));
        q_.wait_and_throw();
        return buf;
    }

    void upload_leaves(const WeightStore& weights) {
        const auto& nodes = graph_.nodes();
        leaf_ptr_.assign(nodes.size(), nullptr);
        leaf_quant_.assign(nodes.size(), sycl_be::QuantWeight{});
        std::map<std::pair<std::string, DType>, size_t> uploaded;  // first node of each weight and device dtype
        size_t bytes = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind == OpKind::Weight) {
                const std::string& name = std::get<graph::WeightAttrs>(n.attrs).name;
                const auto key = std::make_pair(name, dtype_[i][0]);
                if (const auto it = uploaded.find(key); it != uploaded.end()) {
                    leaf_ptr_[i] = leaf_ptr_[it->second];
                    leaf_quant_[i] = leaf_quant_[it->second];
                    continue;
                }
                const TensorView& view = weights.get(name);
                XS_CHECK(view.shape == n.outputs[0].shape, "weight '", name, "' has shape ", dims_to_string(view.shape),
                         " but the graph was built for ", dims_to_string(n.outputs[0].shape));
                uploaded[key] = i;
                const size_t before = leaves_.size();
                if (is_quantized(view.dtype)) {
                    upload_quantized(i, view);
                } else {
                    leaves_.push_back(upload(name, view.data, view.dtype, view.shape, dtype_[i][0]));
                    leaf_ptr_[i] = leaves_.back().get();
                }
                for (size_t b = before; b < leaves_.size(); ++b)
                    bytes += leaves_[b].bytes();
            } else if (n.kind == OpKind::Constant) {
                const HostTensor& value = std::get<graph::ConstantAttrs>(n.attrs).value;
                leaves_.push_back(upload(n.label, value.raw(), value.dtype(), value.shape(), dtype_[i][0]));
                leaf_ptr_[i] = leaves_.back().get();
            }
        }
        log::info("sycl: ", uploaded.size(), " weights, ", bytes >> 20, " MiB on ",
                  q_.get_device().get_info<sycl::info::device::name>());
    }

    StateBuffers make_state(size_t bytes) {
        StateBuffers s;
        s.bytes = bytes;
        s.buf[0] = DeviceBuffer(q_, bytes);
        s.buf[1] = DeviceBuffer(q_, bytes);
        return s;
    }

    void zero_fixed_state() {
        for (StateBuffers& s : fixed_state_)
            q_.memset(s.buf[s.cur].get(), 0, s.bytes);
        q_.wait();
    }

    void ensure_kv_capacity(int64_t tokens) {
        for (KvBuffers& b : kv_) {
            if (tokens <= b.capacity) continue;
            const size_t esize = dtype_size(b.dtype);
            const int64_t cap = std::min(max_context_, std::max({tokens, 2 * b.capacity, int64_t{1024}}));
            const auto grow = [&](DeviceBuffer& buf, size_t per_token) {
                DeviceBuffer bigger(q_, static_cast<size_t>(cap) * per_token);
                if (cached_ > 0) q_.memcpy(bigger.get(), buf.get(), static_cast<size_t>(cached_) * per_token);
                return bigger;
            };
            DeviceBuffer k = grow(b.k, static_cast<size_t>(b.row) * esize);
            DeviceBuffer v = grow(b.v, static_cast<size_t>(b.row) * esize);
            DeviceBuffer ks, vs;
            if (b.dtype == DType::I8) {
                ks = grow(b.k_scale, static_cast<size_t>(b.heads) * sizeof(float));
                vs = grow(b.v_scale, static_cast<size_t>(b.heads) * sizeof(float));
            }
            q_.wait();
            b.k = std::move(k);
            b.v = std::move(v);
            b.k_scale = std::move(ks);
            b.v_scale = std::move(vs);
            b.capacity = cap;
        }
    }

    void ensure_arena(size_t bytes) {
        if (bytes <= arena_.bytes()) return;
        q_.wait();
        arena_ = DeviceBuffer(q_, align_up(bytes + bytes / 8));
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
            if (!known) fail("input '", name, "' is not an input of this graph");
        }
        Step step;
        bool have_tokens = false;
        for (const Value& v : graph_inputs) {
            const std::string& name = input_name(v);
            const auto it = inputs.find(name);
            if (it == inputs.end()) fail("graph input '", name, "' is missing");
            const graph::TensorType& type = graph_.type(v);
            const HostTensor& t = it->second;
            if (t.dtype() != type.dtype) {
                fail("input '", name, "' has dtype ", dtype_name(t.dtype()), "; the graph expects ",
                     dtype_name(type.dtype));
            }
            if (t.rank() != type.shape.size()) {
                fail("input '", name, "' has shape ", dims_to_string(t.shape()), "; the graph expects ",
                     graph::shape_to_string(type.shape));
            }
            for (size_t d = 0; d < type.shape.size(); ++d) {
                if (type.shape[d] != kT) continue;
                if (have_tokens && t.dim(d) != step.tokens) {
                    fail("input '", name, "' has T = ", t.dim(d), " but another input has T = ", step.tokens);
                }
                step.tokens = t.dim(d);
                have_tokens = true;
            }
        }
        step.kv_len = cached_ + step.tokens;
        for (const Value& v : graph_inputs) {
            const HostTensor& t = inputs.at(input_name(v));
            if (t.shape() != resolve(graph_.type(v).shape, step)) {
                fail("input '", input_name(v), "' has shape ", dims_to_string(t.shape()), "; the graph expects ",
                     graph::shape_to_string(graph_.type(v).shape), " (T=", step.tokens, ", L=", step.kv_len, ")");
            }
        }
        // Gather ids index device memory: check them here, where they are host data.
        for (const Node& n : graph_.nodes()) {
            if (n.kind != OpKind::Gather || graph_.node(n.inputs[1]).kind != OpKind::Input) continue;
            const HostTensor& ids = inputs.at(input_name(n.inputs[1]));
            const int64_t rows = graph_.type(n.inputs[0]).shape[0];
            for (int64_t i = 0; i < ids.numel(); ++i) {
                const int64_t id = ids.dtype() == DType::I32 ? ids.data<int32_t>()[i] : ids.data<int64_t>()[i];
                if (id < 0 || id >= rows) {
                    fail("input '", input_name(n.inputs[1]), "': id ", id, " at index ", i, " is outside the table of ",
                         rows, " rows");
                }
            }
        }
        return step;
    }

    const Plan& plan_for(const Step& step) {
        const std::pair<int64_t, int64_t> key{step.tokens, l_dependent_ ? step.kv_len : 0};
        if (const auto it = plans_.find(key); it != plans_.end()) return it->second;
        if (plans_.size() >= 64) plans_.clear();
        return plans_.emplace(key, build_plan(step)).first->second;
    }

    Plan build_plan(const Step& step) const {
        const auto& nodes = graph_.nodes();
        const size_t count = nodes.size();
        Plan p;
        p.shapes.resize(count);
        p.locs.resize(count);
        p.alias.assign(count, 0);
        std::vector<size_t> block_size;
        std::vector<size_t> block_def;
        std::vector<size_t> block_last;
        std::vector<std::vector<int>> block_of(count);  // arena block behind each value, -1 for none
        for (size_t i = 0; i < count; ++i) {
            const Node& n = nodes[i];
            const size_t outs = n.outputs.size();
            p.shapes[i].resize(outs);
            p.locs[i].resize(outs);
            block_of[i].assign(outs, -1);
            for (size_t j = 0; j < outs; ++j) {
                // KvCache outputs keep a symbolic L unless the plan is per L (see l_dependent_).
                p.shapes[i][j] = n.kind == OpKind::KvCache && !l_dependent_ ? resolve(n.outputs[j].shape, Step{step.tokens, kL})
                                                                            : resolve(n.outputs[j].shape, step);
            }
            const auto new_block = [&](size_t j) {
                const size_t bytes = static_cast<size_t>(numel(p.shapes[i][j])) * dtype_size(dtype_[i][j]);
                block_of[i][j] = static_cast<int>(block_size.size());
                p.locs[i][j] = Loc{Store::Arena, block_of[i][j], 0};
                block_size.push_back(align_up(std::max<size_t>(bytes, 1)));
                // Inputs are all uploaded before the first op runs: live from the start.
                block_def.push_back(n.kind == OpKind::Input ? 0 : i);
                block_last.push_back(i);
            };
            const auto alias_input = [&](int64_t extra_bytes) {
                const Value& in = n.inputs[0];
                p.locs[i][0] = p.locs[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)];
                p.locs[i][0].offset += extra_bytes;
                block_of[i][0] = block_of[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)];
                p.alias[i] = 1;
            };
            if (fused_[i] == Fused::Skip) {
                p.alias[i] = 1;  // runs inside its consumer; no memory of its own
                continue;
            }
            switch (n.kind) {
                case OpKind::Weight:
                case OpKind::Constant:
                    p.locs[i][0] = Loc{Store::Leaf, static_cast<int32_t>(i), 0};
                    p.alias[i] = 1;
                    break;
                case OpKind::Reshape:
                    alias_input(0);
                    break;
                case OpKind::Silu:
                    if (fused_[i] == Fused::Alias) {
                        alias_input(0);
                    } else {
                        new_block(0);
                    }
                    break;
                case OpKind::Slice: {
                    const auto& a = std::get<graph::SliceAttrs>(n.attrs);
                    const Dims& in = p.shapes[static_cast<size_t>(n.inputs[0].node)][static_cast<size_t>(n.inputs[0].index)];
                    int64_t outer = 1;
                    for (int d = 0; d < a.axis; ++d)
                        outer *= in[static_cast<size_t>(d)];
                    if (outer == 1) {
                        const int64_t dim = in[static_cast<size_t>(a.axis)];
                        const int64_t begin = a.begin < 0 ? a.begin + dim : a.begin;
                        const int64_t inner = numel(in) / dim;
                        alias_input(begin * inner * static_cast<int64_t>(dtype_size(dtype_[i][0])));
                    } else {
                        new_block(0);
                    }
                    break;
                }
                case OpKind::KvCache: {
                    const int slot = std::get<graph::KvCacheAttrs>(n.attrs).slot;
                    p.locs[i][0] = Loc{Store::KvK, slot, 0};
                    p.locs[i][1] = Loc{Store::KvV, slot, 0};
                    break;
                }
                default:
                    for (size_t j = 0; j < outs; ++j)
                        new_block(j);
                    break;
            }
            for (const Value& in : reads_[i]) {
                const int b = block_of[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)];
                if (b >= 0) block_last[static_cast<size_t>(b)] = std::max(block_last[static_cast<size_t>(b)], i);
            }
        }
        for (const auto& out : graph_.outputs()) {
            const int b = block_of[static_cast<size_t>(out.value.node)][static_cast<size_t>(out.value.index)];
            if (b >= 0) block_last[static_cast<size_t>(b)] = count;
        }

        // First fit over the blocks live at each node; a block is free again after its last reader.
        std::vector<std::vector<size_t>> starts_at(count), ends_at(count + 1);
        for (size_t b = 0; b < block_size.size(); ++b) {
            starts_at[block_def[b]].push_back(b);
            ends_at[block_last[b]].push_back(b);
        }
        std::vector<size_t> block_offset(block_size.size(), 0);
        std::vector<std::pair<size_t, size_t>> live;  // (offset, size), sorted by offset
        std::vector<size_t> live_block;
        for (size_t i = 0; i < count; ++i) {
            for (const size_t b : starts_at[i]) {
                const size_t size = block_size[b];
                size_t candidate = 0;
                size_t pos = 0;
                for (; pos < live.size(); ++pos) {
                    if (candidate + size <= live[pos].first) break;
                    candidate = std::max(candidate, live[pos].first + live[pos].second);
                }
                live.insert(live.begin() + static_cast<std::ptrdiff_t>(pos), {candidate, size});
                live_block.insert(live_block.begin() + static_cast<std::ptrdiff_t>(pos), b);
                block_offset[b] = candidate;
                p.arena_bytes = std::max(p.arena_bytes, candidate + size);
            }
            for (const size_t b : ends_at[i]) {
                const auto it = std::find(live_block.begin(), live_block.end(), b);
                const auto at = it - live_block.begin();
                live.erase(live.begin() + at);
                live_block.erase(it);
            }
        }
        for (auto& locs : p.locs) {
            for (Loc& l : locs) {
                if (l.store != Store::Arena) continue;
                l.offset += static_cast<int64_t>(block_offset[static_cast<size_t>(l.index)]);
                l.index = 0;
            }
        }
        return p;
    }

    char* ptr(const Plan& p, const Value& v) const {
        const Loc& l = p.locs[static_cast<size_t>(v.node)][static_cast<size_t>(v.index)];
        switch (l.store) {
            case Store::Arena:
                return arena_.get() + l.offset;
            case Store::Leaf:
                return leaf_ptr_[static_cast<size_t>(l.index)] + l.offset;
            case Store::KvK:
                return kv_[static_cast<size_t>(l.index)].k.get() + l.offset;
            case Store::KvV:
                return kv_[static_cast<size_t>(l.index)].v.get() + l.offset;
        }
        return nullptr;
    }

    void launch(size_t i, const Plan& p, const Step& step) {
        const Node& n = graph_.nodes()[i];
        const auto in_ptr = [&](size_t k) { return ptr(p, n.inputs[k]); };
        const auto in_f32 = [&](size_t k) { return reinterpret_cast<const float*>(in_ptr(k)); };
        const auto in_shape = [&](size_t k) -> const Dims& {
            return p.shapes[static_cast<size_t>(n.inputs[k].node)][static_cast<size_t>(n.inputs[k].index)];
        };
        const auto in_dt = [&](size_t k) { return dt(n.inputs[k]); };
        char* out_raw = ptr(p, Value{static_cast<int32_t>(i), 0});
        auto* out = reinterpret_cast<float*>(out_raw);
        const Dims& out_shape = p.shapes[i][0];
        const int64_t out_n = n.kind == OpKind::KvCache ? 0 : numel(out_shape);  // KvCache outputs may keep kL
        using sycl_be::UnaryOp;
        // Kernels that may write f16 (plan_half_values) get the output as float* or half*.
        const auto with_out = [&](auto&& fn) {
            if (dt(Value{static_cast<int32_t>(i), 0}) == DType::F16) {
                fn(reinterpret_cast<half*>(out_raw));
            } else {
                fn(out);
            }
        };
        const auto read_f32 = [&](size_t k) { return reinterpret_cast<const float*>(ptr(p, reads_[i][k])); };
        const auto unary = [&](UnaryOp op, float a = 0.0f, float b = 0.0f) {
            if (unary_bias_[i] >= 0) {  // act(x + bias): op is Silu, GeluTanh or Sigmoid
                const Value bias = reads_[i][1];
                const int64_t period = numel(p.shapes[static_cast<size_t>(bias.node)][static_cast<size_t>(bias.index)]);
                with_out([&](auto* y) { sycl_be::unary_bias(q_, op, read_f32(0), read_f32(1), y, out_n, period); });
                return;
            }
            with_out([&](auto* y) { sycl_be::unary(q_, op, in_f32(0), y, out_n, a, b); });
        };
        switch (n.kind) {
            case OpKind::Input:
            case OpKind::Weight:
            case OpKind::Constant:
            case OpKind::Reshape:
                return;
            case OpKind::Gather: {
                const int64_t width = in_shape(0)[1];
                const int64_t count = numel(in_shape(1));
                if (is_quantized(in_dt(0))) {
                    const sycl_be::QuantWeight& t = leaf_quant_[static_cast<size_t>(n.inputs[0].node)];
                    if (in_dt(1) == DType::I32) {
                        sycl_be::gather_quant(q_, t, reinterpret_cast<const int32_t*>(in_ptr(1)), out, count, width);
                    } else {
                        sycl_be::gather_quant(q_, t, reinterpret_cast<const int64_t*>(in_ptr(1)), out, count, width);
                    }
                    return;
                }
                const bool f16 = in_dt(0) == DType::F16;
                if (in_dt(1) == DType::I32) {
                    const auto* ids = reinterpret_cast<const int32_t*>(in_ptr(1));
                    if (f16) {
                        sycl_be::gather(q_, reinterpret_cast<const half*>(in_ptr(0)), ids, out, count, width);
                    } else {
                        sycl_be::gather(q_, in_f32(0), ids, out, count, width);
                    }
                } else {
                    const auto* ids = reinterpret_cast<const int64_t*>(in_ptr(1));
                    if (f16) {
                        sycl_be::gather(q_, reinterpret_cast<const half*>(in_ptr(0)), ids, out, count, width);
                    } else {
                        sycl_be::gather(q_, in_f32(0), ids, out, count, width);
                    }
                }
                return;
            }
            case OpKind::MatMul: {
                const int64_t rows = in_shape(1)[0];
                const int64_t k = in_shape(1)[1];
                const int64_t m = numel(in_shape(0)) / k;
                if (is_quantized(in_dt(1))) {
                    sycl_be::matmul_quant(q_, *this, in_f32(0), leaf_quant_[static_cast<size_t>(n.inputs[1].node)], out,
                                          m, rows, k, precision_ == DType::F32);
                } else if (dt(Value{static_cast<int32_t>(i), 0}) == DType::F16) {  // plan_half_values: y in f16
                    auto* y = reinterpret_cast<half*>(out_raw);
                    const auto* w = reinterpret_cast<const half*>(in_ptr(1));
                    if (in_dt(0) == DType::F16) {
                        sycl_be::matmul_half_y(q_, *this, reinterpret_cast<const half*>(in_ptr(0)), w, y, m, rows, k);
                    } else {
                        sycl_be::matmul_half_y(q_, *this, in_f32(0), w, y, m, rows, k);
                    }
                } else if (in_dt(0) == DType::F16) {  // plan_half_values: x is already f16 (GEMM path)
                    sycl_be::matmul_half_x(q_, reinterpret_cast<const half*>(in_ptr(0)),
                                           reinterpret_cast<const half*>(in_ptr(1)), out, m, rows, k);
                } else if (in_dt(1) == DType::F16) {
                    sycl_be::matmul(q_, *this, in_f32(0), reinterpret_cast<const half*>(in_ptr(1)), out, m, rows, k);
                } else {
                    sycl_be::matmul(q_, *this, in_f32(0), in_f32(1), out, m, rows, k);
                }
                return;
            }
            case OpKind::Add:
            case OpKind::Mul:
            case OpKind::Div: {
                if (act_mul_[i] >= 0) {
                    const auto act = static_cast<UnaryOp>(act_mul_[i]);
                    with_out([&](auto* y) {
                        if (dt(reads_[i][0]) == DType::F16) {
                            sycl_be::act_mul(q_, act, reinterpret_cast<const half*>(ptr(p, reads_[i][0])),
                                             reinterpret_cast<const half*>(ptr(p, reads_[i][1])), y, out_n);
                        } else {
                            sycl_be::act_mul(q_, act, read_f32(0), read_f32(1), y, out_n);
                        }
                    });
                    return;
                }
                if (add3_[i] >= 0) {
                    const Value c = reads_[i][2];
                    const int64_t period = numel(p.shapes[static_cast<size_t>(c.node)][static_cast<size_t>(c.index)]);
                    with_out([&](auto* y) { sycl_be::add3(q_, read_f32(0), read_f32(1), read_f32(2), y, out_n, period); });
                    return;
                }
                if (norm_bias_[i] >= 0) {
                    const Node& norm = graph_.nodes()[static_cast<size_t>(norm_bias_[i])];
                    const int64_t width = out_shape.back();
                    const bool weighted = norm.inputs.size() == 2;
                    with_out([&](auto* y) {
                        sycl_be::rms_norm(q_, read_f32(0), weighted ? read_f32(1) : nullptr, read_f32(weighted ? 2 : 1), y,
                                          out_n / width, width, std::get<graph::NormAttrs>(norm.attrs).eps);
                    });
                    return;
                }
                if (add_mul_[i] >= 0) {
                    const Value c = reads_[i][2];
                    const int64_t period = numel(p.shapes[static_cast<size_t>(c.node)][static_cast<size_t>(c.index)]);
                    with_out([&](auto* y) { sycl_be::add_mul(q_, read_f32(0), read_f32(1), read_f32(2), y, out_n, period); });
                    return;
                }
                const auto op = n.kind == OpKind::Add   ? sycl_be::BinaryOp::Add
                                : n.kind == OpKind::Mul ? sycl_be::BinaryOp::Mul
                                                        : sycl_be::BinaryOp::Div;
                const Dims& a = in_shape(0);
                const Dims& b = in_shape(1);
                if (a == b) {
                    with_out([&](auto* y) { sycl_be::binary_same(q_, op, in_f32(0), in_f32(1), y, out_n); });
                } else if (a == out_shape && is_suffix_broadcast(b, out_shape)) {
                    with_out([&](auto* y) { sycl_be::binary_period(q_, op, in_f32(0), in_f32(1), y, out_n, numel(b)); });
                } else if (op != sycl_be::BinaryOp::Div && b == out_shape && is_suffix_broadcast(a, out_shape)) {
                    // add, mul commute
                    with_out([&](auto* y) { sycl_be::binary_period(q_, op, in_f32(1), in_f32(0), y, out_n, numel(a)); });
                } else {
                    XS_CHECK(out_shape.size() <= static_cast<size_t>(sycl_be::kMaxRank), where(i), ": rank ",
                             out_shape.size(), " exceeds ", sycl_be::kMaxRank);
                    sycl_be::Broadcast s;
                    s.rank = static_cast<int>(out_shape.size());
                    const auto strides = [&](const Dims& shape, int64_t* dst) {
                        int64_t stride = 1;
                        for (size_t d = shape.size(); d-- > 0;) {
                            const size_t o = d + (out_shape.size() - shape.size());
                            dst[o] = shape[d] == 1 ? 0 : stride;
                            stride *= shape[d];
                        }
                    };
                    for (size_t d = 0; d < out_shape.size(); ++d)
                        s.out[d] = out_shape[d];
                    strides(a, s.stride_a);
                    strides(b, s.stride_b);
                    with_out([&](auto* y) { sycl_be::binary(q_, op, in_f32(0), in_f32(1), y, out_n, s); });
                }
                return;
            }
            case OpKind::Scale:
                return unary(UnaryOp::Scale, std::get<graph::ScaleAttrs>(n.attrs).factor);
            case OpKind::Tanh:
                return unary(UnaryOp::Tanh);
            case OpKind::GeluTanh:
                return unary(UnaryOp::GeluTanh);
            case OpKind::Softcap:
                return unary(UnaryOp::Softcap, std::get<graph::SoftcapAttrs>(n.attrs).cap);
            case OpKind::Sigmoid:
                return unary(UnaryOp::Sigmoid);
            case OpKind::Silu:
                return unary(UnaryOp::Silu);
            case OpKind::Softplus:
                return unary(UnaryOp::Softplus);
            case OpKind::Clamp: {
                const auto& a = std::get<graph::ClampAttrs>(n.attrs);
                return unary(UnaryOp::Clamp, a.lo, a.hi);
            }
            case OpKind::RmsNorm: {
                const int64_t width = out_shape.back();
                const float* w = n.inputs.size() == 2 ? in_f32(1) : nullptr;
                with_out([&](auto* y) {
                    sycl_be::rms_norm(q_, in_f32(0), w, static_cast<const float*>(nullptr), y, out_n / width, width,
                                      std::get<graph::NormAttrs>(n.attrs).eps);
                });
                return;
            }
            case OpKind::Slice: {  // the aliasing case returned earlier: a copy
                const auto& a = std::get<graph::SliceAttrs>(n.attrs);
                const Dims& in = in_shape(0);
                const int64_t dim = in[static_cast<size_t>(a.axis)];
                const int64_t begin = a.begin < 0 ? a.begin + dim : a.begin;
                int64_t outer = 1;
                for (int d = 0; d < a.axis; ++d)
                    outer *= in[static_cast<size_t>(d)];
                const int64_t inner = numel(in) / (outer * dim);
                sycl_be::slice(q_, in_ptr(0), out_raw, dtype_size(in_dt(0)), outer, dim, begin,
                               out_shape[static_cast<size_t>(a.axis)], inner);
                return;
            }
            case OpKind::Transpose: {
                const auto& perm = std::get<graph::TransposeAttrs>(n.attrs).perm;
                const Dims& in = in_shape(0);
                XS_CHECK(in.size() <= static_cast<size_t>(sycl_be::kMaxRank), where(i), ": rank ", in.size(),
                         " exceeds ", sycl_be::kMaxRank);
                std::vector<int64_t> in_strides(in.size(), 1);
                for (size_t d = in.size(); d-- > 1;)
                    in_strides[d - 1] = in_strides[d] * in[d];
                sycl_be::Permute pm;
                pm.rank = static_cast<int>(in.size());
                for (size_t d = 0; d < in.size(); ++d) {
                    pm.out[d] = out_shape[d];
                    pm.stride[d] = in_strides[static_cast<size_t>(perm[d])];
                }
                sycl_be::transpose(q_, in_ptr(0), out_raw, dtype_size(in_dt(0)), out_n, pm);
                return;
            }
            case OpKind::Rope: {
                const Dims& x = in_shape(0);
                const Dims& pos = in_shape(1);
                const int64_t sections = pos.size() == 2 ? pos[1] : 1;
                const int64_t rotary_attr = std::get<graph::RopeAttrs>(n.attrs).rotary_dim;
                const int64_t rotary = rotary_attr == 0 ? x[2] : rotary_attr;
                with_out([&](auto* y) {
                    if (in_dt(1) == DType::I32) {
                        sycl_be::rope(q_, in_f32(0), reinterpret_cast<const int32_t*>(in_ptr(1)), in_f32(2), y, x[0],
                                      x[1], x[2], rotary, sections);
                    } else {
                        sycl_be::rope(q_, in_f32(0), reinterpret_cast<const int64_t*>(in_ptr(1)), in_f32(2), y, x[0],
                                      x[1], x[2], rotary, sections);
                    }
                });
                return;
            }
            case OpKind::KvCache: {
                const int slot = std::get<graph::KvCacheAttrs>(n.attrs).slot;
                KvBuffers& b = kv_[static_cast<size_t>(slot)];
                const int64_t count = step.tokens * b.row;
                const size_t offset = static_cast<size_t>((step.kv_len - step.tokens) * b.row);
                if (b.dtype == DType::I8) {
                    const int64_t rows = step.tokens * b.heads;
                    const int64_t first = (step.kv_len - step.tokens) * b.heads;
                    const int64_t width = b.row / b.heads;
                    sycl_be::quantize_kv_i8(q_, in_f32(0), reinterpret_cast<int8_t*>(b.k.get()) + offset,
                                            reinterpret_cast<float*>(b.k_scale.get()) + first, rows, width);
                    sycl_be::quantize_kv_i8(q_, in_f32(1), reinterpret_cast<int8_t*>(b.v.get()) + offset,
                                            reinterpret_cast<float*>(b.v_scale.get()) + first, rows, width);
                } else if (b.dtype == DType::F16) {
                    auto* k = reinterpret_cast<half*>(b.k.get()) + offset;
                    auto* v = reinterpret_cast<half*>(b.v.get()) + offset;
                    sycl_be::to_half(q_, in_f32(0), k, count);
                    sycl_be::to_half(q_, in_f32(1), v, count);
                } else {
                    q_.memcpy(reinterpret_cast<float*>(b.k.get()) + offset, in_f32(0), static_cast<size_t>(count) * 4);
                    q_.memcpy(reinterpret_cast<float*>(b.v.get()) + offset, in_f32(1), static_cast<size_t>(count) * 4);
                }
                return;
            }
            case OpKind::Attention: {
                const auto& a = std::get<graph::AttentionAttrs>(n.attrs);
                const Dims& qs = in_shape(0);
                const Dims& ks = in_shape(1);
                sycl_be::AttentionShape s;
                s.tokens = qs[0];
                s.q_heads = qs[1];
                s.head_dim = qs[2];
                s.kv_len = ks[0] == kL ? step.kv_len : ks[0];
                s.kv_heads = ks[1];
                s.scale = a.scale;
                s.softcap = a.softcap;
                s.window = a.sliding_window;
                s.causal = a.causal;
                if (n.inputs.size() == 4) s.key_bias = in_f32(3);
                if (in_dt(1) == DType::I8) {
                    const int slot = std::get<graph::KvCacheAttrs>(graph_.node(n.inputs[1]).attrs).slot;
                    const KvBuffers& b = kv_[static_cast<size_t>(slot)];
                    sycl_be::KvI8 kv;
                    kv.k = reinterpret_cast<const int8_t*>(b.k.get());
                    kv.v = reinterpret_cast<const int8_t*>(b.v.get());
                    kv.k_scale = reinterpret_cast<const float*>(b.k_scale.get());
                    kv.v_scale = reinterpret_cast<const float*>(b.v_scale.get());
                    if (precision_ == DType::F16) {
                        sycl_be::attention_i8<half>(q_, *this, kv_scratch_, in_f32(0), kv, out, s, attention_);
                    } else {
                        sycl_be::attention_i8<float>(q_, *this, kv_scratch_, in_f32(0), kv, out, s, attention_);
                    }
                } else {
                    const auto run = [&](const auto* qp) {
                        if (in_dt(1) == DType::F16) {
                            sycl_be::attention(q_, *this, qp, reinterpret_cast<const half*>(in_ptr(1)),
                                               reinterpret_cast<const half*>(in_ptr(2)), out, s, attention_);
                        } else {
                            sycl_be::attention(q_, *this, qp, in_f32(1), in_f32(2), out, s, attention_);
                        }
                    };
                    if (in_dt(0) == DType::F16) {
                        run(reinterpret_cast<const half*>(in_ptr(0)));
                    } else {
                        run(in_f32(0));
                    }
                }
                return;
            }
            case OpKind::LayerNorm: {
                const int64_t width = out_shape.back();
                with_out([&](auto* y) {
                    sycl_be::layer_norm(q_, in_f32(0), in_f32(1), in_f32(2), y, out_n / width, width,
                                        std::get<graph::NormAttrs>(n.attrs).eps);
                });
                return;
            }
            case OpKind::Concat: {
                const int axis = std::get<graph::AxisAttrs>(n.attrs).axis;
                const size_t esize = dtype_size(dt(Value{static_cast<int32_t>(i), 0}));
                int64_t outer = 1;
                for (int d = 0; d < axis; ++d)
                    outer *= out_shape[static_cast<size_t>(d)];
                const size_t out_row = static_cast<size_t>(out_n / outer) * esize;
                size_t offset = 0;
                for (size_t k = 0; k < n.inputs.size(); ++k) {
                    const size_t row = static_cast<size_t>(numel(in_shape(k)) / outer) * esize;
                    if (outer == 1) {
                        q_.memcpy(out_raw + offset, in_ptr(k), row);
                    } else {
                        q_.ext_oneapi_memcpy2d(out_raw + offset, out_row, in_ptr(k), row, row,
                                               static_cast<size_t>(outer));
                    }
                    offset += row;
                }
                return;
            }
            case OpKind::Softmax: {
                const int64_t width = out_shape.back();
                sycl_be::softmax(q_, in_f32(0), out, out_n / width, width);
                return;
            }
            case OpKind::TopK: {
                const int64_t width = in_shape(0).back();
                const int64_t k = std::get<graph::TopKAttrs>(n.attrs).k;
                auto* indices = reinterpret_cast<int32_t*>(ptr(p, Value{static_cast<int32_t>(i), 1}));
                sycl_be::topk(q_, in_f32(0), out, indices, numel(in_shape(0)) / width, width, k);
                return;
            }
            case OpKind::ReduceSum: {
                const int axis = std::get<graph::AxisAttrs>(n.attrs).axis;
                const Dims& in = in_shape(0);
                int64_t outer = 1;
                for (int d = 0; d < axis; ++d)
                    outer *= in[static_cast<size_t>(d)];
                const int64_t dim = in[static_cast<size_t>(axis)];
                sycl_be::reduce_sum(q_, in_f32(0), out, outer, dim, numel(in) / (outer * dim));
                return;
            }
            case OpKind::ExpertMatMul: {
                const Dims& ws = in_shape(1);
                const Dims& is = in_shape(2);
                const bool per_slot = in_shape(0).size() == 3;
                if (is_quantized(in_dt(1))) {
                    const sycl_be::QuantWeight& w = leaf_quant_[static_cast<size_t>(n.inputs[1].node)];
                    const int64_t pairs = is[0] * is[1];
                    if (pairs >= sycl_be::kExpertPairsGemm) {
                        const bool f32 = precision_ == DType::F32;
                        if (in_dt(2) == DType::I32) {
                            sycl_be::expert_matmul_quant_gemm(q_, *this, in_f32(0), per_slot, w,
                                                              reinterpret_cast<const int32_t*>(in_ptr(2)), out, is[0],
                                                              is[1], ws[0], ws[1], ws[2], f32);
                        } else {
                            sycl_be::expert_matmul_quant_gemm(q_, *this, in_f32(0), per_slot, w,
                                                              reinterpret_cast<const int64_t*>(in_ptr(2)), out, is[0],
                                                              is[1], ws[0], ws[1], ws[2], f32);
                        }
                        return;
                    }
                    auto* grouping = pairs > sycl_be::kExpertPairsDirect
                                         ? static_cast<int32_t*>(get(sycl_be::expert_grouping_bytes(pairs, ws[0])))
                                         : nullptr;
                    if (in_dt(2) == DType::I32) {
                        sycl_be::expert_matmul_quant(q_, in_f32(0), per_slot, w, reinterpret_cast<const int32_t*>(in_ptr(2)),
                                                     out, is[0], is[1], ws[0], ws[1], ws[2], grouping);
                    } else {
                        sycl_be::expert_matmul_quant(q_, in_f32(0), per_slot, w, reinterpret_cast<const int64_t*>(in_ptr(2)),
                                                     out, is[0], is[1], ws[0], ws[1], ws[2], grouping);
                    }
                    return;
                }
                const auto run = [&](const auto* w) {
                    if (in_dt(2) == DType::I32) {
                        sycl_be::expert_matmul(q_, in_f32(0), per_slot, w, reinterpret_cast<const int32_t*>(in_ptr(2)),
                                               out, is[0], is[1], ws[0], ws[1], ws[2]);
                    } else {
                        sycl_be::expert_matmul(q_, in_f32(0), per_slot, w, reinterpret_cast<const int64_t*>(in_ptr(2)),
                                               out, is[0], is[1], ws[0], ws[1], ws[2]);
                    }
                };
                if (in_dt(1) == DType::F16) {
                    run(reinterpret_cast<const half*>(in_ptr(1)));
                } else {
                    run(in_f32(1));
                }
                return;
            }
            case OpKind::CausalConv: {
                const int slot = std::get<graph::StateAttrs>(n.attrs).slot;
                StateBuffers& st = fixed_state_[static_cast<size_t>(slot)];
                const Dims& x = in_shape(0);
                sycl_be::causal_conv(q_, in_f32(0), in_f32(1), reinterpret_cast<const float*>(st.buf[st.cur].get()),
                                     reinterpret_cast<float*>(st.buf[1 - st.cur].get()), out, x[0], x[1],
                                     in_shape(1)[1], conv_silu_[i] != 0);
                return;
            }
            case OpKind::GatedDelta: {
                const auto& a = std::get<graph::StateAttrs>(n.attrs);
                StateBuffers& st = fixed_state_[conv_states_ + static_cast<size_t>(a.slot)];
                const Dims& qs = in_shape(0);
                const Dims& vs = in_shape(2);
                sycl_be::gated_delta(q_, in_f32(0), in_f32(1), in_f32(2), in_f32(3), in_f32(4),
                                     reinterpret_cast<const float*>(st.buf[st.cur].get()),
                                     reinterpret_cast<float*>(st.buf[1 - st.cur].get()), out, qs[0], qs[1], vs[1],
                                     qs[2], vs[2], a.qk_l2norm_eps);
                return;
            }
        }
        fail(where(i), ": unhandled op");
    }

    // XS_SYCL_PROFILE=1: each op runs alone and its wall time (launch included) adds to its kind's total.
    void timed_launch(size_t i, const Plan& p, const Step& step) {
        q_.wait();
        const auto start = std::chrono::steady_clock::now();
        launch(i, p, step);
        q_.wait();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const Node& n = graph_.nodes()[i];
        std::string key(graph::op_name(n.kind));
        const auto shape = [&](size_t k) -> const Dims& {
            return p.shapes[static_cast<size_t>(n.inputs[k].node)][static_cast<size_t>(n.inputs[k].index)];
        };
        if (n.kind == OpKind::MatMul) {
            key += str_cat(" ", shape(1)[0], "x", shape(1)[1]);
        } else if (n.kind == OpKind::Attention) {
            key += str_cat(" T=", shape(0)[0] == kT ? step.tokens : shape(0)[0], " N=",
                           shape(1)[0] == kL ? step.kv_len : shape(1)[0], " D=", shape(0)[2]);
        }
        (step.tokens == 1 ? decode_ms_ : prefill_ms_)[key] += ms;
    }

    void report_profile() const {
        const auto report = [](const char* what, const std::map<std::string, double>& ms, int64_t steps) {
            if (steps == 0) return;
            std::vector<std::pair<double, std::string>> rows;
            double total = 0;
            for (const auto& [k, v] : ms) {
                rows.emplace_back(v / static_cast<double>(steps), k);
                total += v;
            }
            std::sort(rows.rbegin(), rows.rend());
            log::info("sycl profile (", what, ", ", steps, " steps): ", total / static_cast<double>(steps), " ms/step");
            for (const auto& [v, k] : rows)
                log::info("  ", k, ": ", v, " ms/step");
        };
        report("decode", decode_ms_, decode_steps_);
        report("prefill", prefill_ms_, prefill_steps_);
    }

    const graph::Graph graph_;
    sycl::queue q_;
    const sycl_be::AttentionPlan attention_;
    const DType precision_;
    const bool kv_i8_;  // KV slots that only attention reads are int8 (SessionOptions::kv_dtype)
    const int64_t max_context_;
    const bool profiling_;
    const bool sync_;  // XS_SYCL_SYNC=1: wait after every op and name the op a device error comes from
    const bool trace_;  // XS_SYCL_TRACE=1 with XS_SYCL_SYNC=1: print every op before it runs
    bool l_dependent_ = false;
    std::vector<std::vector<DType>> dtype_;  // device dtype of every value
    // Fusions (plan_fusions): a fused activation runs inside its single consumer.
    enum class Fused : uint8_t { No, Skip, Alias };  // Skip: not run, no output; Alias: names its input's memory
    std::vector<Fused> fused_;
    std::vector<std::vector<Value>> reads_;  // what each node reads on the device (its inputs, or a fusion's)
    std::vector<int8_t> act_mul_;            // Mul run as act(reads[0]) * reads[1]: the UnaryOp, or -1
    std::vector<uint8_t> conv_silu_;         // CausalConv that applies the SiLU after it
    std::vector<int32_t> norm_bias_;         // Add run as RmsNorm(node) + bias: the norm node, or -1
    std::vector<int32_t> add_mul_;           // Add run as a + b * c with the Mul node folded in, or -1
    std::vector<int32_t> unary_bias_;        // activation run as act(x + bias) with the Add folded in, or -1
    std::vector<int32_t> add3_;              // Add run as a + (b + c) with the inner Add folded in, or -1
    std::vector<DeviceBuffer> leaves_;
    std::vector<char*> leaf_ptr_;  // by node index, weights and constants
    std::vector<sycl_be::QuantWeight> leaf_quant_;  // by node index, quantized weights
    std::vector<KvBuffers> kv_;
    std::vector<StateBuffers> fixed_state_;  // conv slots, then recurrent slots
    size_t conv_states_ = 0;
    std::map<std::pair<int64_t, int64_t>, Plan> plans_;
    DeviceBuffer arena_;
    DeviceBuffer scratch_;
    void* host_ = nullptr;  // get_host()
    size_t host_bytes_ = 0;
    GrowingScratch kv_scratch_{q_};  // the int8 KV cache dequantized for prefill attention
    int64_t cached_ = 0;
    std::map<std::string, double> decode_ms_, prefill_ms_;
    std::set<int64_t> profiled_shapes_;      // step token counts seen (their first run is not profiled)
    std::map<int64_t, int64_t> profiled_runs_;
    int64_t decode_steps_ = 0;
    int64_t prefill_steps_ = 0;
};

class SyclBackend final : public Backend {
public:
    SyclBackend() : devices_(gpu_devices()) {}

    std::string_view name() const override { return "sycl"; }

    std::vector<DeviceInfo> devices() const override {
        std::vector<DeviceInfo> out;
        for (size_t i = 0; i < devices_.size(); ++i) {
            DeviceInfo d;
            d.id = str_cat("GPU.", i);
            d.name = devices_[i].get_info<sycl::info::device::name>();
            d.memory_bytes = devices_[i].get_info<sycl::info::device::global_mem_size>();
            out.push_back(std::move(d));
        }
        return out;
    }

    std::unique_ptr<Session> compile(const graph::Graph& graph, const WeightStore& weights,
                                     const SessionOptions& options) override {
        graph.validate();
        XS_CHECK(options.precision == DType::F16 || options.precision == DType::F32,
                 "sycl session precision must be f16 or f32, got ", dtype_name(options.precision));
        XS_CHECK(options.max_context > 0, "sycl max_context must be positive, got ", options.max_context);
        XS_CHECK(!options.kv_dtype || *options.kv_dtype == DType::I8, "sycl KV cache dtype must be i8 when set, got ",
                 dtype_name(*options.kv_dtype));
        sycl_be::AttentionPlan attention;
        for (const auto& [key, value] : options.extra) {
            if (key != "attention") fail("sycl does not know backend option '", key, "' (known: attention)");
            if (value == "rows") {
                attention.path = sycl_be::AttentionPath::Rows;
            } else if (value == "gemm") {
                attention.path = sycl_be::AttentionPath::Gemm;
            } else if (value == "flash") {
                attention.path = sycl_be::AttentionPath::Flash;
            } else if (value != "auto") {
                fail("sycl backend option attention=", value, " is not one of auto, rows, gemm, flash");
            }
        }
        XS_CHECK(!devices_.empty(), "sycl: no Level Zero GPU found");
        size_t index = 0;
        if (!options.device.empty() && options.device != "GPU") {
            XS_CHECK(options.device.rfind("GPU.", 0) == 0, "sycl devices are GPU.<n>, got '", options.device, "'");
            index = static_cast<size_t>(std::stoul(options.device.substr(4)));
            XS_CHECK(index < devices_.size(), "sycl: no device ", options.device, " (", devices_.size(), " GPUs)");
        }
        auto it = queues_.find(index);
        if (it == queues_.end()) {
            // A context of this device alone: in the platform's default context (every GPU), device allocations of a
            // large model failed at 14 to 18 GiB of a 24 GiB card with two cards present.
            sycl::queue q(sycl::context(devices_[index]), devices_[index],
                          sycl::property_list{sycl::property::queue::in_order()});
            it = queues_.emplace(index, std::move(q)).first;
        }
        attention.flash_device = sycl_be::flash_attention_device(devices_[index]);
        attention.compute_units = devices_[index].get_info<sycl::info::device::max_compute_units>();
        return std::make_unique<SyclSession>(graph, weights, options, it->second, attention);
    }

private:
    std::vector<sycl::device> devices_;
    std::map<size_t, sycl::queue> queues_;  // one in-order queue per device, shared by its sessions
};

}  // namespace

std::unique_ptr<Backend> make_sycl_backend() {
    return std::make_unique<SyclBackend>();
}

}  // namespace xs
