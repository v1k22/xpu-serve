#pragma once

// Building blocks the model graphs share: activations and gated MLPs, RMSNorm variants, RoPE frequencies and the
// attention block. A block builds graph ops from values and callables the model hands it, so it knows nothing about
// checkpoint names: a model maps its weights, the block does the math once for every model.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "quant/quant.h"
#include "tensor/tensor.h"

namespace xs::blocks {

using graph::Graph;
using graph::Value;

// A projection x [..., in] -> [..., out]: usually a matmul with a weight, sometimes more (clipped linears).
using Linear = std::function<Value(Value)>;
// A per-head transform of [T, heads, head_dim] (a q / k / v norm). Empty: none.
using HeadNorm = std::function<Value(Value)>;

inline Linear matmul_by(Graph& g, Value w) {
    return [&g, w](Value x) { return g.matmul(x, w); };
}

// ---- Activations and MLPs

enum class Act : uint8_t { Silu, GeluTanh, Gelu };  // Gelu: the exact one (erf)
Value activation(Graph& g, Act act, Value x);

// down(act(gate(x)) * up(x)): SwiGLU with Silu, GeGLU with GeluTanh or Gelu.
Value gated_mlp(Graph& g, Value x, const Linear& gate, const Linear& up, const Linear& down, Act act);

// ---- Mixture of experts

enum class Router : uint8_t { Softmax, Sigmoid };

struct MoeSpec {
    int64_t experts = 0;
    int64_t top_k = 0;
    Router router = Router::Softmax;  // expert probabilities from the router logits
    bool normalize = true;            // the k chosen probabilities rescaled to sum to 1
    float routed_scale = 1.0f;        // the weights then times this
    Act act = Act::Silu;
};

struct MoeParts {
    Value router;                     // [E, hidden]
    Value router_x;                   // the router's input when it is not x (invalid: x)
    Value expert_scale;               // [E]: each expert's routing weight times its entry, after normalizing; or invalid
    Value gate, up, down;             // [E, I, hidden], [E, I, hidden], [E, hidden, I]
    Value gate_up;                    // instead of gate and up: [E, 2 I, hidden], gate rows first (one expert matmul)
    Linear shared_gate, shared_up, shared_down;  // a shared expert every token uses; empty: none
    Value shared_expert_gate;         // [1, hidden]: the shared output times sigmoid(x . w); invalid: no gate
};

// x [T, hidden] -> [T, hidden]: the router picks top_k experts per token, each runs a gated MLP, and their outputs
// are summed with the routing weights; plus the (optionally gated) shared expert.
Value moe(Graph& g, Value x, const MoeSpec& s, const MoeParts& p);

// ---- A drafter's head over a shortlist

// A drafter's pick over a large vocabulary without reading its whole head: every row is scored by the head's one-bit
// sketch (Graph::sign_matmul), and the `count` best are multiplied exactly. x [1, hidden], head [V, hidden].
// The candidates [count] (rows of the head) and their logits [1, count]: the outputs "candidates" and "logits" of a
// draft program. A pick the shortlist misses costs a draft, never output: the target verifies it.
struct Shortlist {
    Value logits;
    Value candidates;
};
Shortlist shortlist_head(Graph& g, Value x, Value head, const quant::SignSketch& sketch, int64_t count);

// The index of each row's largest logit (ties to the lower index), [rows] i32: the "argmax" output a target program
// offers beside "logits", so that greedy decoding never brings the logits back from the device.
Value argmax_rows(Graph& g, Value logits);

// ---- RMSNorm variants (plain x * w is Graph::rms_norm)

// x * rsqrt(mean(x^2) + eps) * (1 + w): zero-centered norms (Qwen3.5); `1 + w` is read on the host as a constant.
Value rms_norm_one_plus(Graph& g, Value x, const WeightStore& weights, const std::string& name, float eps);
// rms_norm(x) * w * silu(gate) (Qwen3.5's gated norm), over the last dim.
Value gated_rms_norm(Graph& g, Value x, Value w, Value gate, float eps);

// ---- RoPE

// `length` inverse frequencies for Graph::rope: entry i < pairs is theta^(-2 i / dim), the rest 0 (identity). Plain
// RoPE over the first R channels: (theta, R, R / 2, R / 2). Gemma 4's proportional RoPE keeps the full head layout:
// (theta, head_dim, rotated pairs, head_dim / 2). Axial 2-D RoPE (vision): (theta, head_dim / 2, head_dim / 4, same).
HostTensor rope_inv_freq(double theta, int64_t dim, int64_t pairs, int64_t length);

// ---- Attention

struct AttentionSpec {
    int64_t heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    graph::AttentionAttrs attrs;  // scale, sliding window, softcap, causal
    int64_t rotary_dim = 0;       // Graph::rope over the first rotary_dim channels (0: all)
    bool output_gate = false;     // q projection is [query | gate] per head; the output is scaled by sigmoid(gate)
    bool kv_cache = true;         // decoder: k / v go through a new KV slot; encoder: attend over this step's k / v
    int64_t kv_window = 0;        // the slot's sliding-window hint
    std::string kv_label;
};

struct AttentionParts {
    Linear q, k, v, o;  // v empty: the values are the key projection's output, before k_norm and RoPE (K = V)
    HeadNorm q_norm, k_norm, v_norm;
    Linear head_gate;   // x -> [T, heads]: head h's output is scaled by sigmoid(head_gate(x)[h]) (LFM3); empty: none
};

struct AttentionOut {
    Value out;                    // [T, hidden]: after the o projection
    std::pair<Value, Value> kv;   // the k / v the attention read (to share with later layers)
};

// Self-attention over x [T, hidden]: projections, per-head norms, RoPE with (pos, inv_freq) when inv_freq is valid,
// the KV slot, attention, the output gates and the o projection. With `shared_kv` the layer reads that k / v (an
// earlier layer's cache) instead of projecting its own.
AttentionOut attention(Graph& g, const AttentionSpec& s, const AttentionParts& p, Value x, Value pos, Value inv_freq,
                       const std::pair<Value, Value>* shared_kv = nullptr);

}  // namespace xs::blocks
