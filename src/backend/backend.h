#pragma once

// The backend interface. A backend turns a step graph plus its weights into a Session that runs steps for one
// sequence and owns that sequence's KV state. Model code never talks to a backend; the engine does.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "graph/graph.h"
#include "io/safetensors.h"
#include "tensor/tensor.h"

namespace xs {

struct DeviceInfo {
    std::string id;    // what --device accepts, e.g. "GPU.0", "CPU"
    std::string name;  // human-readable, e.g. "Intel(R) Arc(TM) B390 GPU"
    uint64_t memory_bytes = 0;
    bool integrated = false;
};

struct SessionOptions {
    std::string device;                // empty = backend default
    DType precision = DType::F16;      // compute precision for activations and weights (F16 or F32)
    int64_t max_context = 8192;        // upper bound on tokens per sequence (sizes caches)
    // KV cache storage: empty = the backend's choice for `precision`; I8 = int8 codes with one f32 scale per (token,
    // KV head), rounded as quant::quantize_kv_row_i8.
    std::optional<DType> kv_dtype;
    // Steps of up to this many tokens can be cut back to any point inside them (truncate), conv and recurrent state
    // included: the verify step of speculative decoding (k + 1 tokens). 0: only whole sequences.
    int64_t rewind_tokens = 0;
    std::map<std::string, std::string> extra;  // backend-specific options, refused by name when unknown
};

// The programs of a session run over one sequence. They share the weights (by name) and the state slots: a KV, conv
// or recurrent slot that two programs declare with the same label is one state. Each slot counts the tokens it holds;
// a program extends the slots it writes (KvCache, CausalConv, GatedDelta), which always hold the same count, and may
// read KV slots another program writes (KvRead, without appending; such a program writes no KV slot itself).
class Session {
public:
    virtual ~Session() = default;

    // Runs one step of program `program` (its index in compile's list). `inputs` holds every graph input by name
    // (shapes resolve kT). Fills `outputs` with every graph output by name as freshly allocated host tensors (f32 for
    // float outputs), or with just the outputs named in `only` when it is not empty (a backend then skips what neither
    // they nor the state need: a row's argmax asked for instead of its logits never brings the logits back).
    // Advances the slots the program writes by T tokens; a failed step leaves the state as it was. A program without
    // slots is stateless.
    virtual void run(size_t program, const TensorMap& inputs, TensorMap& outputs, std::span<const std::string> only) = 0;
    void run(size_t program, const TensorMap& inputs, TensorMap& outputs) { run(program, inputs, outputs, {}); }
    void run(const TensorMap& inputs, TensorMap& outputs) { run(0, inputs, outputs, {}); }

    // Tokens in the slots `program` writes (L of its last run, 0 after reset or for a program writing none).
    virtual int64_t num_cached(size_t program) const = 0;
    int64_t num_cached() const { return num_cached(0); }
    // Drops the state of every slot.
    virtual void reset() = 0;
    // Keeps the first `n` tokens of the slots `program` writes (n <= num_cached(program)). Conv and recurrent state
    // (Graph::has_fixed_state) summarize the sequence: it keeps 0 tokens, all of them, or, after a step of at most
    // SessionOptions::rewind_tokens tokens, any count from before that step on; anything else throws.
    virtual void truncate(size_t program, int64_t n) = 0;
    void truncate(int64_t n) { truncate(0, n); }
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string_view name() const = 0;
    virtual std::vector<DeviceInfo> devices() const = 0;
    // Compiles programs over one sequence (Session) with the weights they reference. Throws xs::Error naming anything
    // it cannot do, including two programs declaring one slot label with different shapes.
    virtual std::unique_ptr<Session> compile(std::span<const graph::Graph* const> programs, const WeightStore& weights,
                                             const SessionOptions& options) = 0;
    std::unique_ptr<Session> compile(const graph::Graph& graph, const WeightStore& weights,
                                     const SessionOptions& options) {
        const graph::Graph* programs[] = {&graph};
        return compile(programs, weights, options);
    }
};

// Backends compiled into this binary, in order of preference, e.g. {"sycl", "cpu-ref"}.
std::vector<std::string> available_backends();
// Throws xs::Error listing the available backends when `name` is unknown.
std::unique_ptr<Backend> make_backend(std::string_view name);

}  // namespace xs
