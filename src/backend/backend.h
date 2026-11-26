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
#include "quant/kv_format.h"
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
    // How the KV cache stores its keys and its values (quant/kv_format.h). A backend stores exactly what
    // quant::round_trip_kv gives each appended row.
    quant::KvFormats kv;
    // Steps of up to this many tokens can be cut back to any point inside them (truncate), conv and recurrent state
    // included: the verify step of speculative decoding (k + 1 tokens). 0: a step is kept or dropped whole.
    int64_t rewind_tokens = 0;
    // Per program, in compile's order: the most tokens one step of it feeds (a prefill chunk, a verify). A backend that
    // sizes its device memory reserves at compile time what such steps over a cache of max_context tokens need, and
    // the cache itself, and refuses the context by name when they do not fit. Empty: nothing reserved ahead (memory
    // grows as runs need it); 0, or a program past the list: one that adds nothing to the reservation.
    std::vector<int64_t> max_step_tokens;
    // Sequences the session holds at once, each with state slots of its own (Session::use picks the one that runs);
    // the programs and their step memory are shared.
    int64_t sequences = 1;
    std::map<std::string, std::string> extra;  // backend-specific options, refused by name when unknown
};

// A run's rows that are the next tokens of one sequence (Session::run with segments).
struct Segment {
    int64_t sequence = 0;
    int64_t tokens = 0;
};

// What a sequence's slots held at `tokens` tokens that cutting its KV back to them could not give back later
// (Session::save_state): its conv and recurrent state, and the keys and values of a KV slot that keeps only its
// sliding window. Host memory, in the backend's own layout, a part a slot.
struct StateSnapshot {
    int64_t tokens = 0;
    std::vector<std::vector<uint8_t>> parts;
    size_t bytes() const {
        size_t n = 0;
        for (const std::vector<uint8_t>& p : parts) n += p.size();
        return n;
    }
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
    // One step of a program whose graph allows segments (Graph::allow_segments) over several sequences: the inputs'
    // rows are `segments` one after another, each the next tokens of its sequence (no sequence twice). Gives what
    // running each segment alone on its sequence gives, the outputs' rows in the same order; every sequence advances by
    // its tokens, and a failed step leaves them all as they were. The current sequence stays current. This
    // implementation runs the segments one after another; a backend that runs them as one step overrides it.
    virtual void run_segments(size_t program, std::span<const Segment> segments, const TensorMap& inputs,
                              TensorMap& outputs, std::span<const std::string> only);

    // Tokens in the slots `program` writes (L of its last run, 0 after reset or for a program writing none).
    virtual int64_t num_cached(size_t program) const = 0;
    int64_t num_cached() const { return num_cached(0); }
    // Drops the state of every slot of the current sequence.
    virtual void reset() = 0;
    // Makes `sequence` (0 to SessionOptions::sequences - 1) the one later runs, num_cached, reset and truncate act on;
    // the others keep their state. Sequence 0 is current after compile.
    virtual void use(int64_t sequence) = 0;
    virtual int64_t current_sequence() const = 0;
    // Keeps the first `n` tokens of the slots `program` writes (n <= num_cached(program)). Conv and recurrent state
    // (Graph::has_fixed_state) summarize the sequence: it keeps 0 tokens, all of them, the tokens before the last
    // step (that step undone whole, whatever its size), or, after a step of at most SessionOptions::rewind_tokens
    // tokens, any count from before that step on; anything else throws.
    virtual void truncate(size_t program, int64_t n) = 0;
    void truncate(int64_t n) { truncate(0, n); }
    // A snapshot of the slots `program` writes, of the current sequence as it stands (StateSnapshot).
    virtual StateSnapshot save_state(size_t program) const = 0;
    // Brings the slots `program` writes back to `snapshot` (save_state of the same program): the KV slots cut back to
    // snapshot.tokens, which they must hold, the tokens the snapshot was taken after; the conv and recurrent state
    // and a windowed slot's keys as saved. The current sequence's KV holds those tokens when the snapshot is its own
    // and nothing since cut it below them. Truncate cannot undo the restore (no step to undo into).
    virtual void restore_state(size_t program, const StateSnapshot& snapshot) = 0;
    // Makes the KV slots `program` writes of the current sequence hold the first `tokens` tokens of sequence `from`'s
    // (another sequence, whose slots hold them), in place of their own. A slot that keeps only its window and the conv
    // and recurrent state are left to restore_state of a snapshot `from` took at `tokens`, which must follow when the
    // program has such state.
    virtual void copy_prefix(size_t program, int64_t from, int64_t tokens) = 0;
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
