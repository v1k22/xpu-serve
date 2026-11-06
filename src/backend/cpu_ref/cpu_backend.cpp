#include "backend/cpu_ref/cpu_backend.h"

#include "util/error.h"

namespace xs {

// Placeholder until the reference backend lands.
std::unique_ptr<Backend> make_cpu_ref_backend() { fail("the cpu-ref backend is not implemented yet"); }

}  // namespace xs
