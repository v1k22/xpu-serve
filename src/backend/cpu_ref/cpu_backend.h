#pragma once

#include <memory>

#include "backend/backend.h"

namespace xs {

// Reference backend: straightforward f32 loops on the CPU. Slow on purpose; it is the oracle for tests.
std::unique_ptr<Backend> make_cpu_ref_backend();

}  // namespace xs
