#pragma once

#include <memory>

#include "backend/backend.h"

namespace xs {

// OpenVINO backend: translates the step graph into an ov::Model (opset ops, no IR files) and runs it on any
// OpenVINO device (GPU, CPU).
std::unique_ptr<Backend> make_openvino_backend();

}  // namespace xs
