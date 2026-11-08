#pragma once

// The policy runtime: one robot policy on one device. It tokenizes the prompt, keeps one compiled program per call
// shape (camera subset, flow steps, padded prompt length), compiled and run at warm-up so requests of that shape pay
// for no kernel compilation or memory planning, and reports where each call's time goes. Calls run one at a time
// (latency first, no batching).

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "models/policy.h"

namespace xs {
namespace tok {
class Tokenizer;
}

struct PolicyRunnerOptions {
    std::string backend = "sycl";
    SessionOptions session;
};

struct ActOptions {
    int steps = 0;                     // 0 = the checkpoint's default
    uint64_t seed = 0;                 // the noise seed when `noise` is null
    const HostTensor* noise = nullptr; // [horizon, noise_dim] f32, for reproducing a reference
};

struct ActTiming {
    double prepare_ms = 0;  // host: prompt, tokenizer, pixels, embeddings
    double device_ms = 0;   // the program, including input upload and output download
    double finish_ms = 0;   // host: actions to the robot's units
    double total_ms = 0;
};

struct ActResult {
    HostTensor actions;  // [horizon, action_dim]
    TensorMap captures;  // XS_CAPTURE=matmul: the "cap.*" outputs (branch exp/manual-quant)
    std::vector<int32_t> prompt_ids;
    ActTiming timing;
};

class PolicyRunner {
public:
    static std::unique_ptr<PolicyRunner> create(const std::filesystem::path& dir, const PolicyRunnerOptions& options);
    ~PolicyRunner();
    PolicyRunner(const PolicyRunner&) = delete;
    PolicyRunner& operator=(const PolicyRunner&) = delete;

    const PolicySpec& spec() const { return policy_->spec(); }
    // Builds and compiles the program for calls like `obs` (its cameras and prompt length) with `steps` flow steps,
    // then runs it `runs` times. Returns the compile time in ms (0 when the program existed).
    double warm_up(const Observation& obs, int steps, int runs = 2);
    ActResult act(const Observation& obs, const ActOptions& options);

private:
    PolicyRunner() = default;
    CallShape shape_of(const Observation& obs, int steps, std::vector<int32_t>* prompt_ids) const;
    Session& program(const CallShape& shape, double* compile_ms = nullptr);

    std::unique_ptr<Policy> policy_;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<tok::Tokenizer> tokenizer_;
    SessionOptions session_;
    std::map<CallShape, std::unique_ptr<Session>> programs_;
};

// An observation with every listed camera (gray-gradient images of the spec's size), a zero state and an
// instruction: for warm-up and benchmarks.
Observation synthetic_observation(const PolicySpec& spec, const std::vector<int>& cameras, const std::string& instruction);

}  // namespace xs
