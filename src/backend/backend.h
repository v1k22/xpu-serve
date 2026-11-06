#pragma once

// The backend interface. A backend turns a step graph plus its weights into a Session that runs steps for one
// sequence and owns that sequence's KV state. Model code never talks to a backend; the engine does.

#include <cstdint>
#include <map>
#include <memory>
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
    std::string cache_dir;             // compiled-model cache, empty = none
    std::map<std::string, std::string> extra;  // backend-specific options, refused by name when unknown
};

class Session {
public:
    virtual ~Session() = default;

    // Runs one step. `inputs` holds every graph input by name (shapes resolve kT). Fills `outputs` with every graph
    // output by name as freshly allocated host tensors (f32 for float outputs). Appends T tokens to the KV state.
    virtual void run(const TensorMap& inputs, TensorMap& outputs) = 0;

    // Tokens currently in the KV state (L of the last run, 0 after reset).
    virtual int64_t num_cached() const = 0;
    // Drops all KV state.
    virtual void reset() = 0;
    // Keeps the first `n` cached tokens (n <= num_cached()).
    virtual void truncate(int64_t n) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual std::string_view name() const = 0;
    virtual std::vector<DeviceInfo> devices() const = 0;
    // Compiles `graph` with the weights it references. Throws xs::Error naming anything it cannot do.
    virtual std::unique_ptr<Session> compile(const graph::Graph& graph, const WeightStore& weights,
                                             const SessionOptions& options) = 0;
};

// Backends compiled into this binary, e.g. {"openvino", "cpu-ref"}.
std::vector<std::string> available_backends();
// Throws xs::Error listing the available backends when `name` is unknown.
std::unique_ptr<Backend> make_backend(std::string_view name);

}  // namespace xs
