#pragma once

#include <cstddef>
#include <memory>

#include "backend/backend.h"

namespace xs {

// SYCL backend: runs the step graph op by op with its own SYCL kernels (oneMKL GEMMs for long matmuls) on a
// Level Zero GPU. Weights and the KV cache are f16 (or f32 at f32 precision); activations are f32.
std::unique_ptr<Backend> make_sycl_backend();

// Bytes of device memory the backend's buffers hold now, every session's and device's (what a session's reservation,
// SessionOptions::max_step_tokens, is checked against).
size_t sycl_device_bytes();

}  // namespace xs
