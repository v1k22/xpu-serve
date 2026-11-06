#pragma once

// The step graph: an in-memory description of one engine step (prefill chunk or decode), built by model code at load
// time and executed by a backend. It is not a file format and is never traced from a framework.
//
// Conventions (every backend must implement exactly these semantics; the OpKind comments are the op spec):
//   * Two symbolic dims exist: kT = tokens fed in this step, kL = tokens in the KV cache after this step (past + T).
//     All other dims are static.
//   * Tensors carry no batch dim. One graph run advances one sequence.
//   * Weights are referenced by name from a WeightStore and keep their stored dtype (e.g. bf16); backends decide the
//     compute precision. Computed values are logically f32 (DType::F32); backends may compute in f16 where noted.
//   * Matmul weights use the Hugging Face layout [out, in]: matmul(x, w) = x @ w^T.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs::graph {

inline constexpr int64_t kT = -1;  // tokens in this step
inline constexpr int64_t kL = -2;  // tokens in the KV cache after this step
inline constexpr int64_t kSliceEnd = INT64_MAX;

std::string dim_to_string(int64_t d);
std::string shape_to_string(const Dims& dims);

// Reference to output `index` of node `node`.
struct Value {
    int32_t node = -1;
    int32_t index = 0;
    bool valid() const { return node >= 0; }
    friend bool operator==(const Value&, const Value&) = default;
};

enum class OpKind : uint8_t {
    Input,      // graph input fed by the caller every run
    Weight,     // named tensor from the WeightStore
    Constant,   // small tensor embedded in the graph
    Gather,     // (table [V, D], ids [N] int) -> [N, D]; rows of table
    MatMul,     // (x [..., K], w [N, K]) -> [..., N] = x @ w^T
    Add,        // (a, b) numpy broadcasting
    Mul,        // (a, b) numpy broadcasting
    Scale,      // x * factor
    Tanh,       // tanh(x)
    GeluTanh,   // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    Softcap,    // cap * tanh(x / cap)
    RmsNorm,    // (x [..., D] [, w [D]]) -> x * (mean(x^2) + eps)^-1/2 [* w], computed in f32. No "+1" offset.
    Reshape,    // same element count; target may use kT / kL
    Slice,      // python-style [begin, end) on one axis; a symbolic axis only supports "last n" (begin < 0, end = kSliceEnd)
    Rope,       // (x [T, H, D], positions [T] int, inv_freq [D/2] f32) -> [T, H, D]; rotate-half (NeoX) layout
    KvCache,    // (k [T, Hkv, D], v [T, Hkv, D]) -> (k_all [L, Hkv, D], v_all [L, Hkv, D]); appends to the slot's state
    Attention,  // (q [T, Hq, D], k_all [L, Hkv, D], v_all [L, Hkv, D]) -> [T, Hq, D]; causal, optional window, GQA
};

std::string_view op_name(OpKind kind);

struct InputAttrs {
    std::string name;
};
struct WeightAttrs {
    std::string name;
};
struct ConstantAttrs {
    HostTensor value;
};
struct ScaleAttrs {
    float factor = 1.0f;
};
struct SoftcapAttrs {
    float cap = 0.0f;
};
struct RmsNormAttrs {
    float eps = 1e-6f;
};
struct ReshapeAttrs {
    Dims shape;
};
struct SliceAttrs {
    int axis = 0;
    int64_t begin = 0;
    int64_t end = kSliceEnd;
};
struct KvCacheAttrs {
    int slot = -1;
};
// Query i of the step sits at absolute position p_i = L - T + i; key j sits at position j.
// Key j is visible to query i iff j <= p_i and (sliding_window == 0 or p_i - j < sliding_window).
// scores = scale * q.k (f32); if softcap > 0: scores = softcap * tanh(scores / softcap); softmax in f32.
struct AttentionAttrs {
    float scale = 1.0f;
    int64_t sliding_window = 0;
    float softcap = 0.0f;
};

using Attrs = std::variant<std::monostate, InputAttrs, WeightAttrs, ConstantAttrs, ScaleAttrs, SoftcapAttrs,
                           RmsNormAttrs, ReshapeAttrs, SliceAttrs, KvCacheAttrs, AttentionAttrs>;

struct TensorType {
    DType dtype = DType::F32;
    Dims shape;  // may contain kT / kL
};

struct Node {
    OpKind kind = OpKind::Input;
    std::vector<Value> inputs;
    Attrs attrs;
    std::vector<TensorType> outputs;
    std::string label;  // scope path for debugging and backend node names, e.g. "layers.3.attn.q_proj"
};

// Per-sequence KV state owned by the backend session. A slot is appended by exactly one KvCache node per graph.
struct KvSlot {
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t sliding_window = 0;  // hint: keys older than this are never read (0 = all are)
    std::string label;
};

struct GraphOutput {
    std::string name;
    Value value;
};

class Graph {
public:
    // Leaves.
    Value input(const std::string& name, DType dtype, Dims shape);
    Value weight(const WeightStore& store, const std::string& name);
    Value constant(HostTensor value);

    // Ops. Each checks its operands and infers the output type; misuse throws xs::Error naming the node.
    Value gather(Value table, Value ids);
    Value matmul(Value x, Value w);
    Value add(Value a, Value b);
    Value mul(Value a, Value b);
    Value scale(Value x, float factor);
    Value tanh(Value x);
    Value gelu_tanh(Value x);
    Value softcap(Value x, float cap);
    Value rms_norm(Value x, float eps);
    Value rms_norm(Value x, Value weight, float eps);
    Value reshape(Value x, Dims shape);
    Value slice(Value x, int axis, int64_t begin, int64_t end = kSliceEnd);
    Value rope(Value x, Value positions, Value inv_freq);
    int add_kv_slot(int64_t num_kv_heads, int64_t head_dim, int64_t sliding_window, std::string label);
    std::pair<Value, Value> kv_cache(int slot, Value k, Value v);
    Value attention(Value q, Value k_all, Value v_all, const AttentionAttrs& attrs);

    void output(const std::string& name, Value v);

    // Label scope for the nodes created while it is alive: Graph::Scope s(g, "layers.3");
    class Scope {
    public:
        Scope(Graph& g, std::string_view name);
        ~Scope();
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        Graph& g_;
        size_t prev_size_;
    };

    // Inspection.
    const std::vector<Node>& nodes() const { return nodes_; }
    const Node& node(Value v) const;
    const TensorType& type(Value v) const;
    const std::vector<GraphOutput>& outputs() const { return outputs_; }
    const std::vector<KvSlot>& kv_slots() const { return kv_slots_; }
    std::vector<Value> inputs() const;  // Input nodes in creation order

    // Checks graph-level invariants (outputs exist, every slot appended exactly once, unique names). Throws.
    void validate() const;
    // Human-readable listing, one node per line.
    std::string dump() const;

private:
    Value add_node(OpKind kind, std::vector<Value> inputs, Attrs attrs, std::vector<TensorType> outputs,
                   std::string_view what = {});
    Value elementwise_binary(OpKind kind, Value a, Value b);
    Value elementwise_unary(OpKind kind, Value x, Attrs attrs);
    std::string where(Value v) const;  // "node 12 (layers.3.attn.q_proj: MatMul)"

    std::vector<Node> nodes_;
    std::vector<GraphOutput> outputs_;
    std::vector<KvSlot> kv_slots_;
    std::string scope_;
};

}  // namespace xs::graph
