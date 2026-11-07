#pragma once

#include <cstdint>
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
              int64_t max_context, bool stateful);

    void run(const TensorMap& inputs, TensorMap& outputs) override;
    int64_t num_cached() const override { return num_cached_; }
    void reset() override;
    void truncate(int64_t n) override;

private:
    // Checks names, dtypes and static dims of every input; returns the step's token count T (0 if no input has T).
    int64_t validate_inputs(const TensorMap& inputs) const;
    void bind_inputs(const TensorMap& inputs);
    void read_outputs(TensorMap& outputs);

    ov::CompiledModel compiled_;
    ov::InferRequest request_;
    std::vector<InputSpec> inputs_;
    std::vector<std::string> outputs_;
    int64_t max_context_ = 0;
    bool stateful_ = true;
    int64_t num_cached_ = 0;
};

}  // namespace xs::ov_backend
