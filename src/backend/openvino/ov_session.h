#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>

#include "backend/backend.h"
#include "backend/openvino/ov_translate.h"

namespace xs::ov_backend {

// One sequence on one compiled step graph. The KV cache lives in the infer request's variable states.
class OvSession final : public Session {
public:
    OvSession(ov::CompiledModel compiled, std::vector<InputSpec> inputs, std::vector<std::string> outputs,
              int64_t max_context, bool stateful, bool fixed_state);

    ~OvSession() override;
    void run(const TensorMap& inputs, TensorMap& outputs) override;
    // XS_OV_PROFILE: accumulate the plugin's per-node timings and log the costliest op types at destruction.
    void enable_profile_report() { profile_ = true; }
    int64_t num_cached() const override { return num_cached_; }
    void reset() override;
    void truncate(int64_t n) override;

private:
    // Checks names, dtypes and static dims of every input; returns the step's token count T (0 if no input has T).
    int64_t validate_inputs(const TensorMap& inputs) const;
    void bind_inputs(const TensorMap& inputs, int64_t tokens);
    void read_outputs(TensorMap& outputs);
    void zero_state_tables();

    ov::CompiledModel compiled_;
    ov::InferRequest request_;
    std::vector<InputSpec> inputs_;
    std::vector<std::string> outputs_;
    int64_t max_context_ = 0;
    bool stateful_ = true;
    bool fixed_state_ = false;
    int64_t num_cached_ = 0;
    std::vector<ov::Tensor> state_tables_;  // InputRole::StateTable, on the device where the plugin has one
    std::vector<int32_t> index_values_;     // backing store of the paged-op index inputs bound for a step
    bool profile_ = false;
    struct ProfileEntry {
        double us = 0;
        int64_t count = 0;
    };
    std::map<std::string, ProfileEntry> profile_by_type_;  // by exec type / layer type, single-token steps only
    int64_t profiled_steps_ = 0;
    double bind_ms_ = 0, infer_ms_ = 0, read_ms_ = 0;  // host wall time of single-token steps
};

}  // namespace xs::ov_backend
