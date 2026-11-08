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
    Sigmoid,    // 1 / (1 + exp(-x))
    Silu,       // x * sigmoid(x)
    Softplus,   // log(1 + exp(x))
    Clamp,      // min(max(x, lo), hi); either bound may be infinite
    RmsNorm,    // (x [..., D] [, w [D]]) -> x * (mean(x^2) + eps)^-1/2 [* w], computed in f32. No "+1" offset.
    Reshape,    // same element count; target may use kT / kL
    Slice,      // python-style [begin, end) on one axis; a symbolic axis only supports "last n" (begin < 0, end = kSliceEnd)
    Transpose,  // out.shape[i] = x.shape[perm[i]]; any dtype
    Rope,       // (x [T, H, D], positions [T] or [T, S] int, inv_freq [R/(2S)] f32) -> [T, H, D]; see note below
    KvCache,    // (k [T, Hkv, D], v [T, Hkv, D]) -> (k_all [L, Hkv, D], v_all [L, Hkv, D]); appends to the slot's state
    Attention,  // (q [T, Hq, D], k [N, Hkv, D], v [N, Hkv, D]) -> [T, Hq, D]; see AttentionAttrs; GQA
    CausalConv, // (x [T, C], w [C, K]) -> [T, C]; depthwise causal conv over the slot's last K-1 inputs, see below
    GatedDelta, // (q [T, Hk, Dk], k [T, Hk, Dk], v [T, Hv, Dv], g [T, Hv], beta [T, Hv]) -> [T, Hv, Dv]; see below
    Div,        // (a, b) numpy broadcasting: a / b
    Softmax,    // softmax over the last axis, computed in f32
    TopK,       // (x [..., N]) -> (values [..., k] f32, indices [..., k] i32): the k largest along the last axis,
                // largest first; equal values keep the lower index first
    ReduceSum,  // sum over one axis (ReduceAttrs::axis), which is removed; f32
    ExpertMatMul, // (x [T, K] or [T, S, K], w [E, N, K], idx [T, S] int) -> [T, S, N]: out[t, s] = x[t] (or
                  // x[t, s]) @ w[idx[t, s]]^T; every idx in [0, E). The expert matmuls of a mixture of experts
};

// CausalConv: with x' = [the slot's K-1 previous inputs (zeros at the start), x], out[t, c] = sum_j w[c, j] x'[t + j, c];
// the slot then keeps the last K-1 rows of x'.
//
// GatedDelta (gated delta rule, the recurrence of Gated DeltaNet linear attention): v head h reads key head
// h / (Hv / Hk) and owns a state S_h [Dk, Dv] (zeros at the start), kept in the slot across runs. Per token t:
//   q, k = l2norm(q_t), l2norm(k_t) (when qk_l2norm_eps > 0: x / sqrt(sum x^2 + eps)), then q *= 1 / sqrt(Dk);
//   S = S * exp(g_t); S = S + k (beta_t (v_t - S^T k))^T; out_t = S^T q. Computed in f32.

// Rope: rotate-half (NeoX) rotary embedding in the Hugging Face form, out = x * cos + rotate_half(x) * sin, over the
// first R = rotary_dim channels of each head (all D when 0); channels R.. pass through (partial rotary).
// With positions [T] the rotated channels are one section. With positions [T, S] (axial RoPE, e.g. 2-D image patches)
// they split into S contiguous sections of R/S channels; section s rotates by positions[:, s]. Within a section of
// width W, channel c < W/2 pairs with c + W/2 and both rotate by the angle position * inv_freq[c].

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
struct ClampAttrs {
    float lo = 0.0f;
    float hi = 0.0f;
};
struct TransposeAttrs {
    std::vector<int> perm;
};
struct KvCacheAttrs {
    int slot = -1;
};
struct RopeAttrs {
    int64_t rotary_dim = 0;  // 0 = the whole head dim
};
struct TopKAttrs {
    int64_t k = 1;
};
struct ReduceAttrs {
    int axis = 0;
};
struct StateAttrs {  // CausalConv and GatedDelta: the slot that carries the op's state across runs
    int slot = -1;
    float qk_l2norm_eps = 0.0f;  // GatedDelta only; 0 = no l2norm of q and k
};
// Causal (the default): k and v are KvCache outputs [L, Hkv, D]. Query i of the step sits at absolute position
// p_i = L - T + i; key j sits at position j. Key j is visible to query i iff j <= p_i and (sliding_window == 0 or
// p_i - j < sliding_window).
// Not causal (encoders): every key is visible to every query; k and v may have any length N (e.g. kT for
// self-attention over the step's own tokens); sliding_window must be 0.
// scores = scale * q.k (f32); if softcap > 0: scores = softcap * tanh(scores / softcap); softmax in f32.
struct AttentionAttrs {
    float scale = 1.0f;
    int64_t sliding_window = 0;
    float softcap = 0.0f;
    bool causal = true;
};

using Attrs = std::variant<std::monostate, InputAttrs, WeightAttrs, ConstantAttrs, ScaleAttrs, SoftcapAttrs,
                           ClampAttrs, RmsNormAttrs, ReshapeAttrs, SliceAttrs, TransposeAttrs, KvCacheAttrs,
                           RopeAttrs, StateAttrs, AttentionAttrs, TopKAttrs, ReduceAttrs>;

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

// Per-sequence state owned by the backend session. Each slot is used by exactly one node per graph. A graph without
// slots is stateless: every run stands alone (encoders). KV slots grow with the sequence and can be cut back to a
// prefix; conv and recurrent slots have a fixed size and summarize the whole sequence, so they cannot.
struct KvSlot {
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t sliding_window = 0;  // hint: keys older than this are never read (0 = all are)
    std::string label;
};
struct ConvSlot {  // the last kernel - 1 inputs of a CausalConv: [kernel - 1, channels]
    int64_t channels = 0;
    int64_t kernel = 0;
    std::string label;
};
struct RecurrentSlot {  // a GatedDelta state: [heads, key_dim, value_dim], f32
    int64_t heads = 0;
    int64_t key_dim = 0;
    int64_t value_dim = 0;
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
    Value sigmoid(Value x);
    Value silu(Value x);
    Value softplus(Value x);
    Value clamp(Value x, float lo, float hi);
    Value rms_norm(Value x, float eps);
    Value rms_norm(Value x, Value weight, float eps);
    Value reshape(Value x, Dims shape);
    Value slice(Value x, int axis, int64_t begin, int64_t end = kSliceEnd);
    Value transpose(Value x, std::vector<int> perm);
    Value rope(Value x, Value positions, Value inv_freq, int64_t rotary_dim = 0);
    int add_kv_slot(int64_t num_kv_heads, int64_t head_dim, int64_t sliding_window, std::string label);
    std::pair<Value, Value> kv_cache(int slot, Value k, Value v);
    Value attention(Value q, Value k_all, Value v_all, const AttentionAttrs& attrs);
    int add_conv_slot(int64_t channels, int64_t kernel, std::string label);
    Value causal_conv(int slot, Value x, Value w);
    int add_recurrent_slot(int64_t heads, int64_t key_dim, int64_t value_dim, std::string label);
    Value gated_delta(int slot, Value q, Value k, Value v, Value g, Value beta, float qk_l2norm_eps);
    Value div(Value a, Value b);
    Value softmax(Value x);
    std::pair<Value, Value> topk(Value x, int64_t k);  // (values, indices)
    Value reduce_sum(Value x, int axis);
    Value expert_matmul(Value x, Value w, Value idx);

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
    const std::vector<ConvSlot>& conv_slots() const { return conv_slots_; }
    const std::vector<RecurrentSlot>& recurrent_slots() const { return recurrent_slots_; }
    bool stateful() const { return !kv_slots_.empty() || !conv_slots_.empty() || !recurrent_slots_.empty(); }
    // True when some state cannot be cut back to a prefix (conv or recurrent slots).
    bool has_fixed_state() const { return !conv_slots_.empty() || !recurrent_slots_.empty(); }
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
    std::vector<ConvSlot> conv_slots_;
    std::vector<RecurrentSlot> recurrent_slots_;
    std::string scope_;
};

}  // namespace xs::graph
