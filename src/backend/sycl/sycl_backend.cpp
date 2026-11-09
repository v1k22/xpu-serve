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
    std::vector<uint8_t> alias;     // the node only names memory another value owns: nothing to run
    std::vector<uint8_t> in_place;  // a fused residual GEMM (mm_add_) writing into its residual's memory
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

// An integrated GPU (it shares the system's memory, so streaming weights bounds its small matmuls).
bool integrated_gpu(const sycl::device& d) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return d.get_info<sycl::info::device::host_unified_memory>();
#pragma clang diagnostic pop
}

// How quantized matmuls run on a device. On an integrated GPU (system memory) the matrix-engine kernels that dequantize
// on the way win: quant_gemm up to kQuantGemmMaxRows rows, quant_gemm_tiled up to kTiledMaxRows (Arc B390, a
// 17408 x 5120 q4 weight at 512 rows: 3.1 ms against 4.0 for dequantize + GEMM); past that oneMKL's GEMM amortizes the
// f16 copy of the weight (1,024 rows: 6.3 against 5.2 ms). On a discrete card's faster memory dequantize + GEMM is as
// fast or faster at every row count (Arc Pro B60: 1.5 ms both at 512 rows, 1.1 against 1.5 at 128).
constexpr int64_t kTiledMaxRows = 512;

sycl_be::QuantMatmulPlan quant_plan(const sycl::device& d, const sycl_be::AttentionPlan& attention, DType precision) {
    sycl_be::QuantMatmulPlan p;
    p.f32 = precision == DType::F32;
    p.compute_units = attention.compute_units;
    if (attention.xmx && integrated_gpu(d)) {
        p.few_rows = true;
        p.tiled_rows = kTiledMaxRows;
    }
    return p;
}

// What the programs of one session share (backend.h): the queue, the weights uploaded so far (by name and device
// dtype), the state slots (by label) with their token counts, and the step memory (programs run one at a time).
struct SharedKv {
    graph::KvSlot spec;
    KvBuffers buf;
    int64_t len = 0;
    bool typed = false;  // a writing program has set buf.dtype
};
struct SharedFixed {  // a conv or recurrent slot
    bool conv = false;
    std::string label;
    StateBuffers state;
    int64_t tokens = 0;
    // The last step, when it had at most rewind_tokens tokens (base >= 0): the state before it is the other buffer
    // (state.buf[1 - state.cur]) and its op's per-token inputs are kept here, replayed by truncate for a count inside
    // the step (conv: x; recurrent: k, v, g, beta).
    int64_t base = -1;
    std::vector<DeviceBuffer> inputs;
    std::vector<int64_t> input_row;  // floats per token of each kept input
    int64_t channels = 0, kernel = 0;                                     // conv
    int64_t k_heads = 0, v_heads = 0, key_dim = 0, value_dim = 0;         // recurrent
    float eps = 0.0f;
};
struct SharedLeaf {
    char* ptr = nullptr;
    sycl_be::QuantWeight quant;
};

class SyclShared final : public sycl_be::Scratch {
public:
    SyclShared(sycl::queue queue, int64_t max_context, int64_t rewind_tokens)
        : q(std::move(queue)), max_context(max_context), rewind_tokens(rewind_tokens) {}
    ~SyclShared() override {
        try {
            q.wait();
        } catch (...) {
        }
        if (host_ != nullptr) sycl::free(host_, q);
    }

    void* get(size_t bytes) override {
        if (bytes > scratch_.bytes()) {
            q.wait();
            scratch_ = DeviceBuffer(q, align_up(bytes + bytes / 4));
        }
        return scratch_.get();
    }
    void* get_host(size_t bytes) override {
        if (bytes > host_bytes_) {
            q.wait();
            if (host_ != nullptr) sycl::free(host_, q);
            host_bytes_ = align_up(bytes);
            host_ = sycl::malloc_host(host_bytes_, q);
            XS_CHECK(host_ != nullptr, "sycl: out of pinned host memory allocating ", host_bytes_, " bytes");
        }
        return host_;
    }

    void ensure_arena(size_t bytes) {
        if (bytes <= arena.bytes()) return;
        q.wait();
        arena = DeviceBuffer(q, align_up(bytes + bytes / 8));
    }

    void reset() {
        for (SharedKv& k : kv) k.len = 0;
        for (SharedFixed& f : fixed) {
            q.memset(f.state.buf[f.state.cur].get(), 0, f.state.bytes);
            f.tokens = 0;
            f.base = -1;
        }
        q.wait();
    }

    sycl::queue q;
    const int64_t max_context;
    const int64_t rewind_tokens;
    std::vector<SharedKv> kv;
    std::vector<SharedFixed> fixed;
    std::map<std::pair<std::string, DType>, SharedLeaf> weights;  // uploaded weights by name and device dtype
    std::vector<DeviceBuffer> weight_buffers;
    DeviceBuffer arena;
    GrowingScratch kv_scratch{q};  // the int8 KV cache dequantized for prefill attention

private:
    DeviceBuffer scratch_;
    void* host_ = nullptr;
    size_t host_bytes_ = 0;
};

// One program (graph) of a session: its plans, fusions and launches over the shared state.
class SyclProgram final : private sycl_be::Scratch {
public:
    SyclProgram(const graph::Graph& graph, const WeightStore& weights, const SessionOptions& options, SyclShared& shared,
                const sycl_be::AttentionPlan& attention, const sycl_be::QuantMatmulPlan& quant)
        : graph_(graph),
          shared_(shared),
          q_(shared.q),
          attention_(attention),
          quant_(quant),
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
            reads_kv_ |= n.kind == OpKind::KvRead;
            if (n.kind != OpKind::KvCache && n.kind != OpKind::KvRead) {
                for (const auto& t : n.outputs)
                    l_dependent_ |= has_l(t.shape);
            }
            if (n.kind != OpKind::Attention) {
                for (const Value& in : n.inputs)
                    l_dependent_ |= has_l(graph_.type(in).shape);
            }
        }
        for (const graph::KvSlot& k : graph_.kv_slots()) kv_slot_.push_back(shared_kv(k));
        for (const graph::ConvSlot& c : graph_.conv_slots()) {
            fixed_slot_.push_back(shared_fixed(true, c.label, static_cast<size_t>((c.kernel - 1) * c.channels)));
        }
        conv_states_ = fixed_slot_.size();
        for (const graph::RecurrentSlot& r : graph_.recurrent_slots()) {
            fixed_slot_.push_back(
                shared_fixed(false, r.label, static_cast<size_t>(r.heads * r.key_dim * r.value_dim)));
        }
        infer_dtypes();
        check_ops();
        plan_fusions();
        plan_half_values();
        upload_leaves(weights);
        finish_mm_parts();
        size_t int8_slots = 0;
        for (size_t i = 0; i < graph_.nodes().size(); ++i) {
            const Node& n = graph_.nodes()[i];
            if (n.kind != OpKind::KvCache) continue;
            SharedKv& slot = shared_.kv[kv_slot_[static_cast<size_t>(std::get<graph::KvCacheAttrs>(n.attrs).slot)]];
            XS_CHECK(!slot.typed || slot.buf.dtype == dtype_[i][0], "sycl: kv slot '", slot.spec.label,
                     "' is written in two dtypes by two programs");
            slot.buf.dtype = dtype_[i][0];
            slot.typed = true;
            int8_slots += dtype_[i][0] == DType::I8 ? 1 : 0;
        }
        if (kv_i8_ && !reads_kv_) log::info("sycl: int8 KV cache for ", int8_slots, " of ", kv_slot_.size(), " slots");
    }

    ~SyclProgram() override {
        if (profiling_) report_profile();
    }

    void run(const TensorMap& inputs, TensorMap& outputs) {
        const Step step = check_inputs(inputs);
        XS_CHECK(step.kv_len <= max_context_, "sycl: ", step.kv_len, " tokens exceed max_context ", max_context_);
        std::vector<HostTensor> keep_alive;  // host sources of pending copies
        rewinding_ = step.tokens <= shared_.rewind_tokens;
        try {
            if (!reads_kv_) ensure_kv_capacity(step.kv_len);
            const Plan& plan = plan_for(step);
            shared_.ensure_arena(plan.arena_bytes);
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
                                 input_name(v).c_str(), src.nbytes(), static_cast<long long>(l.offset), shared_.arena.bytes(),
                                 static_cast<int>(l.store));
                    q_.wait_and_throw();
                }
            }
            const auto& nodes = graph_.nodes();
            const auto submit_start = std::chrono::steady_clock::now();
            // The first run of a step shape compiles kernels (JIT, oneMKL): not part of the profile.
            const bool profile_step = profiling_ && !profiled_shapes_.insert(step.tokens).second;
            if (profile_step) marks_.assign(1, {std::string(), q_.ext_oneapi_submit_barrier()});
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
        if (!reads_kv_) {
            for (size_t k : kv_slot_) shared_.kv[k].len = step.kv_len;
        }
        for (size_t f : fixed_slot_) {
            SharedFixed& st = shared_.fixed[f];
            st.state.cur = 1 - st.state.cur;
            st.base = rewinding_ ? st.tokens : -1;  // rewinding_: the step's inputs went to st.inputs
            st.tokens += step.tokens;
        }
        if (!marks_.empty()) {
            // Device time between consecutive barriers is the op's: no host round trip in the measurement.
            auto& ms = step.tokens == 1 ? decode_ms_ : prefill_ms_;
            uint64_t prev = marks_[0].second.get_profiling_info<sycl::info::event_profiling::command_end>();
            for (size_t m = 1; m < marks_.size(); ++m) {
                const uint64_t end = marks_[m].second.get_profiling_info<sycl::info::event_profiling::command_end>();
                ms[marks_[m].first] += static_cast<double>(end - prev) * 1e-6;
                prev = end;
            }
            marks_.clear();
        }
        if (profiling_ && profiled_shapes_.count(step.tokens) != 0 && profiled_runs_[step.tokens]++ > 0) {
            ++(step.tokens == 1 ? decode_steps_ : prefill_steps_);
        }
    }

    // Tokens in the slots this program writes (they agree after every run and truncate).
    int64_t cached() const {
        if (!kv_slot_.empty() && !reads_kv_) return shared_.kv[kv_slot_[0]].len;
        if (!fixed_slot_.empty()) return shared_.fixed[fixed_slot_[0]].tokens;
        return 0;
    }

    void truncate(int64_t n) {
        const int64_t have = cached();
        XS_CHECK(n >= 0 && n <= have, "cannot truncate to ", n, " tokens; ", have, " are cached");
        for (size_t i : fixed_slot_) {  // checked first: a refusal changes nothing
            const SharedFixed& f = shared_.fixed[i];
            XS_CHECK(n == f.tokens || n == 0 || (f.base >= 0 && n >= f.base), "cannot truncate to ", n,
                     " tokens: conv and recurrent state summarize the whole sequence (only a count inside a last step ",
                     "of up to rewind_tokens tokens is kept)");
        }
        if (!reads_kv_) {
            for (size_t k : kv_slot_) shared_.kv[k].len = n;
        }
        for (size_t f : fixed_slot_) rewind(shared_.fixed[f], n);
        q_.wait_and_throw();
    }

private:
    // Brings a conv or recurrent slot to `n` tokens: kept, zeroed, or replayed from the state before the last step
    // (the other buffer) through the kept inputs, into the current buffer.
    void rewind(SharedFixed& f, int64_t n) {
        StateBuffers& st = f.state;
        if (n == f.tokens) return;
        if (n == 0) {
            q_.memset(st.buf[st.cur].get(), 0, st.bytes);
            f.tokens = 0;
            f.base = -1;
            return;
        }
        const int64_t m = n - f.base;  // truncate checked base <= n
        const auto* before = reinterpret_cast<const float*>(st.buf[1 - st.cur].get());
        auto* now = reinterpret_cast<float*>(st.buf[st.cur].get());
        const auto in = [&](size_t i) { return reinterpret_cast<const float*>(f.inputs[i].get()); };
        if (m == 0) {
            q_.memcpy(now, before, st.bytes);
        } else if (f.conv) {
            // the window is the last kernel - 1 rows of [before; x[0, m)]
            const int64_t past = f.kernel - 1;
            const int64_t from_before = std::max<int64_t>(0, past - m);
            const auto row = static_cast<size_t>(f.channels) * sizeof(float);
            if (from_before > 0) q_.memcpy(now, before + (past - from_before) * f.channels, from_before * row);
            if (past - from_before > 0) {
                q_.memcpy(now + from_before * f.channels, in(0) + (m - (past - from_before)) * f.channels,
                          (past - from_before) * row);
            }
        } else {
            auto* out = static_cast<float*>(get(static_cast<size_t>(m * f.v_heads * f.value_dim) * sizeof(float)));
            sycl_be::gated_delta(q_, in(0), in(0), in(1), in(2), in(3), before, now, out, m, f.k_heads, f.v_heads,
                                 f.key_dim, f.value_dim, f.eps);
        }
        f.tokens = n;
    }

    // Keeps a short step's per-token inputs of a conv or recurrent op for rewind (SharedFixed).
    void keep_inputs(SharedFixed& f, std::initializer_list<std::pair<const float*, int64_t>> inputs, int64_t tokens) {
        if (f.inputs.size() != inputs.size()) {
            f.inputs.clear();
            f.input_row.clear();
            for (const auto& [ptr, row] : inputs) {
                f.inputs.emplace_back(q_, static_cast<size_t>(shared_.rewind_tokens * row) * sizeof(float));
                f.input_row.push_back(row);
            }
        }
        size_t i = 0;
        for (const auto& [ptr, row] : inputs) {
            q_.memcpy(f.inputs[i++].get(), ptr, static_cast<size_t>(tokens * row) * sizeof(float));
        }
    }

    size_t shared_kv(const graph::KvSlot& spec) {
        for (size_t i = 0; i < shared_.kv.size(); ++i) {
            const graph::KvSlot& s = shared_.kv[i].spec;
            if (s.label != spec.label) continue;
            XS_CHECK(s.num_kv_heads == spec.num_kv_heads && s.head_dim == spec.head_dim, "sycl: kv slot '",
                     spec.label, "' is declared with two shapes");
            return i;
        }
        SharedKv k;
        k.spec = spec;
        k.buf.heads = spec.num_kv_heads;
        k.buf.row = spec.num_kv_heads * spec.head_dim;
        shared_.kv.push_back(std::move(k));
        return shared_.kv.size() - 1;
    }

    size_t shared_fixed(bool conv, const std::string& label, size_t elements) {
        const size_t bytes = elements * sizeof(float);
        for (size_t i = 0; i < shared_.fixed.size(); ++i) {
            const SharedFixed& f = shared_.fixed[i];
            if (f.conv != conv || f.label != label) continue;
            XS_CHECK(f.state.bytes == bytes, "sycl: state slot '", label, "' is declared with two sizes");
            return i;
        }
        SharedFixed f;
        f.conv = conv;
        f.label = label;
        f.state = make_state(bytes);
        q_.memset(f.state.buf[0].get(), 0, bytes);
        q_.wait();
        shared_.fixed.push_back(std::move(f));
        return shared_.fixed.size() - 1;
    }

    KvBuffers& kv_buffers(int64_t slot) const { return shared_.kv[kv_slot_[static_cast<size_t>(slot)]].buf; }
    StateBuffers& fixed_state(size_t index) { return shared_.fixed[fixed_slot_[index]].state; }

    // sycl_be::Scratch: the session's
    void* get(size_t bytes) override { return shared_.get(bytes); }
    void* get_host(size_t bytes) override { return shared_.get_host(bytes); }

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
                case OpKind::KvRead: {  // in the dtype its writer stores (a program compiled before this one)
                    const SharedKv& slot = shared_.kv[kv_slot_[static_cast<size_t>(
                        std::get<graph::KvCacheAttrs>(n.attrs).slot)]];
                    XS_CHECK(slot.typed, "sycl: kv slot '", slot.spec.label, "' is read before a program writes it; ",
                             "compile the writer first");
                    dtype_[i][0] = dtype_[i][1] = slot.buf.dtype;
                    break;
                }
                case OpKind::TopK:
                    dtype_[i][1] = DType::I32;
                    break;
                case OpKind::Gather:
                    if (dt(n.inputs[0]) == DType::I32) dtype_[i][0] = DType::I32;
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
                    need(0, is_fp(0) || is_packed(0) || dt(n.inputs[0]) == DType::I32, "f16, f32, q8, q4 or i32");
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
        attn_split_.assign(nodes.size(), -1);
        mm_add_.assign(nodes.size(), -1);
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
        // Add(a, MatMul(x, w)) with an f16 or quantized weight past the GEMV and the matmul read only here: one GEMM
        // adding into the output (beta 1, or a quantized kernel's adding epilogue), which the plan puts in a's memory
        // when nothing reads a afterwards: a residual stream's update without the add's three passes over [T, width].
        // reads_: a, x, w.
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind != OpKind::Add || fused_[i] != Fused::No || norm_bias_[i] >= 0 || add_mul_[i] >= 0 || add3_[i] >= 0)
                continue;
            const Dims& out = n.outputs[0].shape;
            for (size_t k = 0; k < 2; ++k) {
                const auto mm = static_cast<size_t>(n.inputs[1 - k].node);
                const Node& m = nodes[mm];
                if (m.kind != OpKind::MatMul || readers[mm] != 1 || fused_[mm] != Fused::No) continue;
                const DType wt = dt(m.inputs[1]);
                if ((wt != DType::F16 && !is_quantized(wt)) || graph_.type(n.inputs[k]).shape != out) continue;
                const Dims& x = graph_.type(m.inputs[0]).shape;
                if (std::any_of(x.begin(), x.end(), [](int64_t v) { return v == kT || v == kL; }) ||
                    numel(x) / x.back() <= sycl_be::kGemvMaxRows)
                    continue;
                fused_[mm] = Fused::Skip;
                mm_add_[i] = static_cast<int32_t>(mm);
                reads_[i] = {n.inputs[k], m.inputs[0], m.inputs[1]};
                reads_[mm].clear();
                break;
            }
        }
        // Attention(q, Concat(k1, k2), Concat(v1, v2)) on the GEMM path (head_dim past the fused kernel's, more
        // queries than the per-row kernel takes): its GEMMs read the two parts in place, so the concat copies go (a
        // policy's action expert attends to the prefix's keys and its own, every flow step). reads_: q, k1, v1,
        // [bias], k2, v2.
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind != OpKind::Attention || fused_[i] != Fused::No) continue;
            const auto kc = static_cast<size_t>(n.inputs[1].node);
            const auto vc = static_cast<size_t>(n.inputs[2].node);
            const auto two_parts = [&](size_t c) {
                return nodes[c].kind == OpKind::Concat && nodes[c].inputs.size() == 2 && readers[c] == 1 &&
                       fused_[c] == Fused::No && std::get<graph::AxisAttrs>(nodes[c].attrs).axis == 0;
            };
            if (kc == vc || !two_parts(kc) || !two_parts(vc)) continue;
            const Dims& q = graph_.type(n.inputs[0]).shape;
            const int64_t first = graph_.type(nodes[kc].inputs[0]).shape[0];
            if (q.size() != 3 || q[0] == kT || first == kT || first != graph_.type(nodes[vc].inputs[0]).shape[0]) continue;
            sycl_be::AttentionShape sh;
            sh.tokens = q[0];
            sh.head_dim = q[2];
            if (sh.head_dim <= 128 || !sycl_be::attention_matrix_path(sh, attention_.path)) continue;
            fused_[kc] = Fused::Skip;
            fused_[vc] = Fused::Skip;
            reads_[kc].clear();
            reads_[vc].clear();
            reads_[i] = {n.inputs[0], nodes[kc].inputs[0], nodes[vc].inputs[0]};
            if (n.inputs.size() == 4) reads_[i].push_back(n.inputs[3]);
            attn_split_[i] = static_cast<int8_t>(reads_[i].size());
            reads_[i].push_back(nodes[kc].inputs[1]);
            reads_[i].push_back(nodes[vc].inputs[1]);
        }
        // MatMuls with quantized weights of one format that read the same x (an MLP's gate and up, attention's q, k
        // and v): at decode sizes the first runs them all in one GEMV launch, which fills the device where a small
        // one alone does not, each output in its own memory, live from there (build_plan). finish_mm_parts drops
        // the weights the kernel cannot pair once they are on the device.
        mm_parts_.assign(nodes.size(), {});
        mm_leader_.assign(nodes.size(), -1);
        std::map<std::pair<int32_t, int32_t>, size_t> filling;  // x -> the MatMul collecting its siblings
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind != OpKind::MatMul || fused_[i] != Fused::No || !is_quantized(dt(n.inputs[1])) ||
                nodes[static_cast<size_t>(n.inputs[1].node)].kind != OpKind::Weight)
                continue;
            const auto x = std::make_pair(n.inputs[0].node, n.inputs[0].index);
            const auto it = filling.find(x);
            if (it != filling.end() && dt(nodes[it->second].inputs[1]) == dt(n.inputs[1]) &&
                mm_parts_[it->second].size() < static_cast<size_t>(sycl_be::kQuantGemvParts)) {
                mm_parts_[it->second].push_back(static_cast<int32_t>(i));
                mm_leader_[i] = static_cast<int32_t>(it->second);
            } else {
                filling[x] = i;
                mm_parts_[i] = {static_cast<int32_t>(i)};
            }
        }
        for (auto& parts : mm_parts_) {
            if (parts.size() == 1) parts.clear();
        }
    }

    void finish_mm_parts() {
        const auto& nodes = graph_.nodes();
        const auto weight = [&](int32_t m) -> const sycl_be::QuantWeight& {
            return leaf_quant_[static_cast<size_t>(nodes[static_cast<size_t>(m)].inputs[1].node)];
        };
        for (size_t i = 0; i < mm_parts_.size(); ++i) {
            std::vector<int32_t>& parts = mm_parts_[i];
            if (parts.empty()) continue;
            std::vector<int32_t> kept{parts[0]};
            for (size_t j = 1; j < parts.size(); ++j) {
                if (sycl_be::quant_gemv_parts_match(weight(parts[j]), weight(parts[0]))) {
                    kept.push_back(parts[j]);
                } else {
                    mm_leader_[static_cast<size_t>(parts[j])] = -1;
                }
            }
            parts = kept.size() > 1 ? std::move(kept) : std::vector<int32_t>{};
        }
    }

    // Whether MatMul `lead`'s siblings run in its launch this step: an x of at most kGemvMaxRows rows in f32, f32
    // outputs.
    bool mm_parts_run(size_t lead, const Plan& p) const {
        const Node& n = graph_.nodes()[lead];
        const Dims& x = p.shapes[static_cast<size_t>(n.inputs[0].node)][static_cast<size_t>(n.inputs[0].index)];
        if (numel(x) / graph_.type(n.inputs[1]).shape[1] > sycl_be::kGemvMaxRows || dt(n.inputs[0]) != DType::F32)
            return false;
        return std::all_of(mm_parts_[lead].begin(), mm_parts_[lead].end(),
                           [&](int32_t m) { return dt(Value{m, 0}) == DType::F32; });
    }

    // Values that only GEMM-path matmuls and quantized matmuls past the GEMV (as x) and GEMM-path attention (as q, k,
    // v) read are kept in f16, written so by the kernel that makes them: those GEMMs run on f16 operands anyway, so a
    // conversion pass disappears and the traffic halves. Attention operands produced in f16 (projections, RoPE) round
    // q, k and v to f16 before the scores, the convention of the f16 KV cache. Static shapes only (dynamic ones can take the f32-only GEMV and
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
        // a quantized matmul past the GEMV's rows: every path there reads x in f16 (sycl_be::QuantPath)
        const auto quant_matmul = [&](size_t m) {
            const Node& n = nodes[m];
            if (n.kind != OpKind::MatMul || !is_quantized(dt(n.inputs[1]))) return false;
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
                                  k == OpKind::Transpose || k == OpKind::Concat ||
                                  (k == OpKind::MatMul && (gemm_matmul(i) || quant_matmul(i)));
            half[i] = producer && fused_[i] == Fused::No && mm_add_[i] < 0 && dtype_[i][0] == DType::F32 && output[i] == 0 &&
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
                        accept = k == 0 && fused_[r] == Fused::No && (gemm_matmul(r) || quant_matmul(r));
                    } else if (rk == OpKind::Attention && attn_split_[r] >= 0) {  // the four parts in one type
                        const auto part = [&](size_t idx) { return half[static_cast<size_t>(reads_[r][idx].node)] != 0; };
                        const auto s2 = static_cast<size_t>(attn_split_[r]);
                        const bool kv = k == 1 || k == 2 || k == s2 || k == s2 + 1;
                        accept = gemm_attention(r) && (k == 0 || (kv && part(1) && part(2) && part(s2) && part(s2 + 1)));
                    } else if (rk == OpKind::Attention) {
                        const auto kv = [&](size_t idx) { return half[static_cast<size_t>(nodes[r].inputs[idx].node)] != 0; };
                        accept = gemm_attention(r) && (k == 0 || ((k == 1 || k == 2) && kv(1) && kv(2)));
                    } else if (rk == OpKind::Reshape || rk == OpKind::Slice || rk == OpKind::Transpose ||
                               rk == OpKind::Concat) {
                        accept = half[r] != 0;
                    } else if (rk == OpKind::Add && mm_add_[r] >= 0) {  // a fused residual GEMM: x in either type
                        accept = k == 1;
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
            shared_.weight_buffers.emplace_back(q_, std::max<size_t>(bytes, 1));
            q_.memcpy(shared_.weight_buffers.back().get(), src, bytes).wait();
            return shared_.weight_buffers.back().get();
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
        // Weights are the session's (a weight two programs read in one device dtype is uploaded once); constants the
        // program's.
        size_t uploaded = 0;
        size_t bytes = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            const Node& n = nodes[i];
            if (n.kind == OpKind::Weight) {
                const std::string& name = std::get<graph::WeightAttrs>(n.attrs).name;
                const auto key = std::make_pair(name, dtype_[i][0]);
                if (const auto it = shared_.weights.find(key); it != shared_.weights.end()) {
                    leaf_ptr_[i] = it->second.ptr;
                    leaf_quant_[i] = it->second.quant;
                    continue;
                }
                const TensorView& view = weights.get(name);
                XS_CHECK(view.shape == n.outputs[0].shape, "weight '", name, "' has shape ", dims_to_string(view.shape),
                         " but the graph was built for ", dims_to_string(n.outputs[0].shape));
                const size_t before = shared_.weight_buffers.size();
                if (is_quantized(view.dtype)) {
                    upload_quantized(i, view);
                } else {
                    shared_.weight_buffers.push_back(upload(name, view.data, view.dtype, view.shape, dtype_[i][0]));
                    leaf_ptr_[i] = shared_.weight_buffers.back().get();
                }
                shared_.weights[key] = SharedLeaf{leaf_ptr_[i], leaf_quant_[i]};
                ++uploaded;
                for (size_t b = before; b < shared_.weight_buffers.size(); ++b)
                    bytes += shared_.weight_buffers[b].bytes();
            } else if (n.kind == OpKind::Constant) {
                const HostTensor& value = std::get<graph::ConstantAttrs>(n.attrs).value;
                leaves_.push_back(upload(n.label, value.raw(), value.dtype(), value.shape(), dtype_[i][0]));
                leaf_ptr_[i] = leaves_.back().get();
            }
        }
        if (uploaded > 0) log::info("sycl: ", uploaded, " weights, ", bytes >> 20, " MiB on ",
                  q_.get_device().get_info<sycl::info::device::name>());
    }

    StateBuffers make_state(size_t bytes) {
        StateBuffers s;
        s.bytes = bytes;
        s.buf[0] = DeviceBuffer(q_, bytes);
        s.buf[1] = DeviceBuffer(q_, bytes);
        return s;
    }

    void ensure_kv_capacity(int64_t tokens) {
        for (size_t slot : kv_slot_) {
            KvBuffers& b = shared_.kv[slot].buf;
            const int64_t len = shared_.kv[slot].len;
            if (tokens <= b.capacity) continue;
            const size_t esize = dtype_size(b.dtype);
            const int64_t cap = std::min(max_context_, std::max({tokens, 2 * b.capacity, int64_t{1024}}));
            const auto grow = [&](DeviceBuffer& buf, size_t per_token) {
                DeviceBuffer bigger(q_, static_cast<size_t>(cap) * per_token);
                if (len > 0) q_.memcpy(bigger.get(), buf.get(), static_cast<size_t>(len) * per_token);
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
        // A program reading KV slots sees their length; one writing state extends it by T.
        step.kv_len = reads_kv_ ? shared_.kv[kv_slot_[0]].len : cached() + step.tokens;
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
        p.in_place.assign(count, 0);
        // For residual GEMMs (mm_add_): the last reader of each value, and whether a view (reshape, slice) reads it
        // (then its memory may be read later under another name).
        std::vector<std::vector<size_t>> last_reader(count);
        std::vector<std::vector<uint8_t>> viewed(count);
        for (size_t i = 0; i < count; ++i) {
            last_reader[i].assign(nodes[i].outputs.size(), 0);
            viewed[i].assign(nodes[i].outputs.size(), 0);
        }
        for (size_t i = 0; i < count; ++i) {
            for (const Value& in : reads_[i]) {
                auto& last = last_reader[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)];
                last = std::max(last, i);
                if (nodes[i].kind == OpKind::Reshape || nodes[i].kind == OpKind::Slice)
                    viewed[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)] = 1;
            }
        }
        std::vector<uint8_t> graph_output(count, 0);
        for (const auto& o : graph_.outputs())
            graph_output[static_cast<size_t>(o.value.node)] = 1;
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
                p.shapes[i][j] = (n.kind == OpKind::KvCache || n.kind == OpKind::KvRead) && !l_dependent_
                                     ? resolve(n.outputs[j].shape, Step{step.tokens, kL})
                                                                            : resolve(n.outputs[j].shape, step);
            }
            const auto new_block = [&](size_t j) {
                const size_t bytes = static_cast<size_t>(numel(p.shapes[i][j])) * dtype_size(dtype_[i][j]);
                block_of[i][j] = static_cast<int>(block_size.size());
                p.locs[i][j] = Loc{Store::Arena, block_of[i][j], 0};
                block_size.push_back(align_up(std::max<size_t>(bytes, 1)));
                // Inputs are all uploaded before the first op runs: live from the start; a MatMul run in a
                // sibling's launch, from the sibling.
                block_def.push_back(n.kind == OpKind::Input ? 0
                                    : mm_leader_[i] >= 0 ? static_cast<size_t>(mm_leader_[i])
                                                         : i);
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
            if (mm_add_[i] >= 0) {  // in the residual's memory when this is its last reader and it owns that memory
                const Value& a = reads_[i][0];
                const auto an = static_cast<size_t>(a.node);
                const auto ai = static_cast<size_t>(a.index);
                const OpKind ak = nodes[an].kind;
                const bool own = p.alias[an] == 0 && p.locs[an][ai].store == Store::Arena && ak != OpKind::Input &&
                                 block_of[an][ai] >= 0;
                if (own && graph_output[an] == 0 && viewed[an][ai] == 0 && last_reader[an][ai] == i &&
                    dtype_[an][ai] == DType::F32 && dtype_[i][0] == DType::F32 && p.shapes[an][ai] == p.shapes[i][0]) {
                    p.locs[i][0] = p.locs[an][ai];
                    block_of[i][0] = block_of[an][ai];
                    p.in_place[i] = 1;
                } else {
                    new_block(0);
                }
                for (const Value& in : reads_[i]) {
                    const int b = block_of[static_cast<size_t>(in.node)][static_cast<size_t>(in.index)];
                    if (b >= 0) block_last[static_cast<size_t>(b)] = std::max(block_last[static_cast<size_t>(b)], i);
                }
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
                case OpKind::KvCache:
                case OpKind::KvRead: {
                    const int slot = std::get<graph::KvCacheAttrs>(n.attrs).slot;
                    p.locs[i][0] = Loc{Store::KvK, slot, 0};
                    p.locs[i][1] = Loc{Store::KvV, slot, 0};
                    if (n.kind == OpKind::KvRead) p.alias[i] = 1;  // nothing to run: it names the slot's memory
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
                return shared_.arena.get() + l.offset;
            case Store::Leaf:
                return leaf_ptr_[static_cast<size_t>(l.index)] + l.offset;
            case Store::KvK:
                return kv_buffers(l.index).k.get() + l.offset;
            case Store::KvV:
                return kv_buffers(l.index).v.get() + l.offset;
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
        const int64_t out_n = n.kind == OpKind::KvCache || n.kind == OpKind::KvRead ? 0 : numel(out_shape);  // may keep kL
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
        // A value that may be f16 (plan_half_values) as const float* or const half*.
        const auto with_value = [&](const Value& v, auto&& fn) {
            if (dt(v) == DType::F16) {
                fn(reinterpret_cast<const half*>(ptr(p, v)));
            } else {
                fn(reinterpret_cast<const float*>(ptr(p, v)));
            }
        };
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
            case OpKind::KvRead:  // its outputs are the slot's buffers (build_plan)
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
                if (in_dt(0) == DType::I32) {
                    const auto* table = reinterpret_cast<const uint32_t*>(in_ptr(0));
                    auto* rows = reinterpret_cast<uint32_t*>(out);
                    if (in_dt(1) == DType::I32) {
                        sycl_be::gather_bits(q_, table, reinterpret_cast<const int32_t*>(in_ptr(1)), rows, count, width);
                    } else {
                        sycl_be::gather_bits(q_, table, reinterpret_cast<const int64_t*>(in_ptr(1)), rows, count, width);
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
                if (mm_leader_[i] >= 0 && mm_parts_run(static_cast<size_t>(mm_leader_[i]), p)) return;  // ran there
                if (!mm_parts_[i].empty() && mm_parts_run(i, p)) {
                    const float* x = in_f32(0);
                    if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {  // the kernel loads x in float4s
                        auto* aligned = static_cast<float*>(get(static_cast<size_t>(m * k) * sizeof(float)));
                        q_.memcpy(aligned, x, static_cast<size_t>(m * k) * sizeof(float));
                        x = aligned;
                    }
                    sycl_be::QuantGemvPart parts[sycl_be::kQuantGemvParts];
                    const auto count = static_cast<int>(mm_parts_[i].size());
                    for (int j = 0; j < count; ++j) {
                        const auto mj = static_cast<size_t>(mm_parts_[i][static_cast<size_t>(j)]);
                        const Value wj = graph_.nodes()[mj].inputs[1];
                        parts[j] = {leaf_quant_[static_cast<size_t>(wj.node)],
                                    reinterpret_cast<float*>(ptr(p, Value{static_cast<int32_t>(mj), 0})),
                                    graph_.type(wj).shape[0]};
                    }
                    sycl_be::gemv_quant_parts(q_, x, parts, count, m, k);
                    return;
                }
                if (is_quantized(in_dt(1))) {  // plan_half_values: x and y in f16 or f32
                    const sycl_be::QuantWeight& w = leaf_quant_[static_cast<size_t>(n.inputs[1].node)];
                    with_value(n.inputs[0], [&](const auto* x) {
                        with_out([&](auto* y) { sycl_be::matmul_quant(q_, *this, x, w, y, m, rows, k, quant_); });
                    });
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
                if (mm_add_[i] >= 0) {  // out = a + x @ w^T: a GEMM adding into out, which holds a in place or a copy
                    const Value a = reads_[i][0];
                    const Value x = reads_[i][1];
                    const Dims& ws = graph_.type(reads_[i][2]).shape;
                    XS_CHECK(dt(a) == DType::F32 && dt(Value{static_cast<int32_t>(i), 0}) == DType::F32,
                             "sycl: a residual GEMM in f16");
                    const int64_t m = out_n / ws[0];
                    if (p.in_place[i] == 0) q_.memcpy(out, ptr(p, a), static_cast<size_t>(out_n) * sizeof(float));
                    if (is_quantized(dt(reads_[i][2]))) {
                        const sycl_be::QuantWeight& qw = leaf_quant_[static_cast<size_t>(reads_[i][2].node)];
                        with_value(x, [&](const auto* xp) {
                            sycl_be::matmul_quant(q_, *this, xp, qw, out, m, ws[0], ws[1], quant_, true);
                        });
                        return;
                    }
                    const auto* w = reinterpret_cast<const half*>(ptr(p, reads_[i][2]));
                    if (dt(x) == DType::F16) {
                        sycl_be::matmul_accumulate(q_, *this, reinterpret_cast<const half*>(ptr(p, x)), w, out, m, ws[0], ws[1]);
                    } else {
                        sycl_be::matmul_accumulate(q_, *this, reinterpret_cast<const float*>(ptr(p, x)), w, out, m, ws[0], ws[1]);
                    }
                    return;
                }
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
                KvBuffers& b = kv_buffers(slot);
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
                    const KvBuffers& b = kv_buffers(slot);
                    sycl_be::KvI8 kv;
                    kv.k = reinterpret_cast<const int8_t*>(b.k.get());
                    kv.v = reinterpret_cast<const int8_t*>(b.v.get());
                    kv.k_scale = reinterpret_cast<const float*>(b.k_scale.get());
                    kv.v_scale = reinterpret_cast<const float*>(b.v_scale.get());
                    if (precision_ == DType::F16) {
                        sycl_be::attention_i8<half>(q_, *this, shared_.kv_scratch, in_f32(0), kv, out, s, attention_);
                    } else {
                        sycl_be::attention_i8<float>(q_, *this, shared_.kv_scratch, in_f32(0), kv, out, s, attention_);
                    }
                } else {
                    // Keys and values in place, in two parts when the concats were fused away (attn_split_).
                    const int split = attn_split_[i];
                    const Value kv_k = split >= 0 ? reads_[i][1] : n.inputs[1];
                    const Value kv_v = split >= 0 ? reads_[i][2] : n.inputs[2];
                    sycl_be::KvSecond second;
                    if (split >= 0) {
                        second.k = ptr(p, reads_[i][static_cast<size_t>(split)]);
                        second.v = ptr(p, reads_[i][static_cast<size_t>(split) + 1]);
                        second.first_len = graph_.type(kv_k).shape[0];
                    }
                    const auto run = [&](const auto* qp) {
                        if (dt(kv_k) == DType::F16) {
                            sycl_be::attention(q_, *this, qp, reinterpret_cast<const half*>(ptr(p, kv_k)),
                                               reinterpret_cast<const half*>(ptr(p, kv_v)), out, s, attention_, second);
                        } else {
                            sycl_be::attention(q_, *this, qp, reinterpret_cast<const float*>(ptr(p, kv_k)),
                                               reinterpret_cast<const float*>(ptr(p, kv_v)), out, s, attention_, second);
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
                if (n.inputs.size() <= static_cast<size_t>(sycl_be::kConcatParts)) {
                    sycl_be::ConcatParts parts;  // one kernel: a copy per part costs a launch each
                    for (size_t k = 0; k < n.inputs.size(); ++k) {
                        parts.src[parts.count] = in_ptr(k);
                        parts.bytes[parts.count++] = static_cast<size_t>(numel(in_shape(k)) / outer) * esize;
                    }
                    sycl_be::concat(q_, parts, out_raw, outer);
                    return;
                }
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
                const auto slot = static_cast<size_t>(std::get<graph::StateAttrs>(n.attrs).slot);
                SharedFixed& f = shared_.fixed[fixed_slot_[slot]];
                StateBuffers& st = f.state;
                const Dims& x = in_shape(0);
                sycl_be::causal_conv(q_, in_f32(0), in_f32(1), reinterpret_cast<const float*>(st.buf[st.cur].get()),
                                     reinterpret_cast<float*>(st.buf[1 - st.cur].get()), out, x[0], x[1],
                                     in_shape(1)[1], conv_silu_[i] != 0);
                if (rewinding_) {
                    f.channels = x[1];
                    f.kernel = in_shape(1)[1];
                    keep_inputs(f, {{in_f32(0), x[1]}}, x[0]);
                }
                return;
            }
            case OpKind::GatedDelta: {
                const auto& a = std::get<graph::StateAttrs>(n.attrs);
                SharedFixed& f = shared_.fixed[fixed_slot_[conv_states_ + static_cast<size_t>(a.slot)]];
                StateBuffers& st = f.state;
                const Dims& qs = in_shape(0);
                const Dims& vs = in_shape(2);
                sycl_be::gated_delta(q_, in_f32(0), in_f32(1), in_f32(2), in_f32(3), in_f32(4),
                                     reinterpret_cast<const float*>(st.buf[st.cur].get()),
                                     reinterpret_cast<float*>(st.buf[1 - st.cur].get()), out, qs[0], qs[1], vs[1],
                                     qs[2], vs[2], a.qk_l2norm_eps);
                if (rewinding_) {
                    f.k_heads = qs[1];
                    f.v_heads = vs[1];
                    f.key_dim = qs[2];
                    f.value_dim = vs[2];
                    f.eps = a.qk_l2norm_eps;
                    keep_inputs(f, {{in_f32(1), qs[1] * qs[2]}, {in_f32(2), vs[1] * vs[2]}, {in_f32(3), vs[1]},
                                    {in_f32(4), vs[1]}},
                                qs[0]);
                }
                return;
            }
        }
        fail(where(i), ": unhandled op");
    }

    // XS_SYCL_PROFILE=1: op i's device time (between its barrier and the previous one's) adds to its kind's total;
    // the run reads the barriers' timestamps once it is done.
    void timed_launch(size_t i, const Plan& p, const Step& step) {
        launch(i, p, step);
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
        } else {  // elementwise passes: by output shape, dtype and fusion
            const Dims& o = p.shapes[i][0];
            key += " [";
            for (size_t d = 0; d < o.size(); ++d) key += str_cat(d ? "," : "", o[d]);
            key += str_cat("] ", dtype_name(dtype_[i][0]), fused_[i] != Fused::No ? " fused" : "");
        }
        marks_.emplace_back(std::move(key), q_.ext_oneapi_submit_barrier());
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
    SyclShared& shared_;
    sycl::queue& q_;
    const sycl_be::AttentionPlan attention_;
    const sycl_be::QuantMatmulPlan quant_;  // how quantized matmuls run on this device (sycl_be::quant_path)
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
    std::vector<int8_t> attn_split_;         // Attention over two-part keys and values: where k2 is in reads_, or -1
    std::vector<int32_t> mm_add_;            // Add run as a GEMM adding into the residual (the MatMul folded in), or -1
    std::vector<std::vector<int32_t>> mm_parts_;  // MatMul whose siblings run in its GEMV: all of them, itself first
    std::vector<int32_t> mm_leader_;         // MatMul run in a sibling's GEMV (mm_parts_): that sibling, or -1
    std::vector<DeviceBuffer> leaves_;  // constants (weights are the session's: SyclShared)
    std::vector<char*> leaf_ptr_;  // by node index, weights and constants
    std::vector<sycl_be::QuantWeight> leaf_quant_;  // by node index, quantized weights
    std::vector<size_t> kv_slot_;     // the shared KV slot of each of the graph's
    std::vector<size_t> fixed_slot_;  // the shared slot of each conv slot, then each recurrent slot
    size_t conv_states_ = 0;
    bool reads_kv_ = false;   // the program reads its KV slots (KvRead) instead of appending
    bool rewinding_ = false;  // the running step keeps its conv and recurrent inputs (SharedFixed)
    std::map<std::pair<int64_t, int64_t>, Plan> plans_;
    std::map<std::string, double> decode_ms_, prefill_ms_;
    std::vector<std::pair<std::string, sycl::event>> marks_;  // a profiled run's barriers: (op, event)
    std::set<int64_t> profiled_shapes_;      // step token counts seen (their first run is not profiled)
    std::map<int64_t, int64_t> profiled_runs_;
    int64_t decode_steps_ = 0;
    int64_t prefill_steps_ = 0;
};

// A session: the shared state and its programs (backend.h).
class SyclSession final : public Session {
public:
    explicit SyclSession(std::unique_ptr<SyclShared> shared) : shared_(std::move(shared)) {}

    SyclShared& shared() { return *shared_; }
    void add(std::unique_ptr<SyclProgram> program) { programs_.push_back(std::move(program)); }

    void run(size_t program, const TensorMap& inputs, TensorMap& outputs) override {
        at(program).run(inputs, outputs);
    }
    int64_t num_cached(size_t program) const override {
        XS_CHECK(program < programs_.size(), "sycl: no program ", program);
        return programs_[program]->cached();
    }
    void reset() override { shared_->reset(); }
    void truncate(size_t program, int64_t n) override { at(program).truncate(n); }

private:
    SyclProgram& at(size_t program) {
        XS_CHECK(program < programs_.size(), "sycl: no program ", program, " (", programs_.size(), " compiled)");
        return *programs_[program];
    }

    std::unique_ptr<SyclShared> shared_;  // outlives the programs, which refer to it
    std::vector<std::unique_ptr<SyclProgram>> programs_;
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
            d.integrated = integrated_gpu(devices_[i]);
            out.push_back(std::move(d));
        }
        return out;
    }

    std::unique_ptr<Session> compile(std::span<const graph::Graph* const> programs, const WeightStore& weights,
                                     const SessionOptions& options) override {
        XS_CHECK(!programs.empty(), "sycl: compile without a program");
        for (const graph::Graph* graph : programs) graph->validate();
        XS_CHECK(options.rewind_tokens >= 0, "sycl rewind_tokens must not be negative");
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
            sycl::property_list props{sycl::property::queue::in_order()};
            if (env_flag("XS_SYCL_PROFILE")) {
                props = sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
            }
            sycl::queue q(sycl::context(devices_[index]), devices_[index], props);
            it = queues_.emplace(index, std::move(q)).first;
        }
        attention.xmx = sycl_be::xmx_device(devices_[index]);
        attention.compute_units = devices_[index].get_info<sycl::info::device::max_compute_units>();
        auto session = std::make_unique<SyclSession>(
            std::make_unique<SyclShared>(it->second, options.max_context, options.rewind_tokens));
        const sycl_be::QuantMatmulPlan quant = quant_plan(devices_[index], attention, options.precision);
        for (const graph::Graph* graph : programs) {
            session->add(std::make_unique<SyclProgram>(*graph, weights, options, session->shared(), attention, quant));
        }
        return session;
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
