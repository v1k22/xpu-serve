#include "backend/backend.h"

#include "backend/cpu_ref/cpu_backend.h"
#include "util/error.h"
#ifdef XS_HAVE_SYCL
#include "backend/sycl/sycl_backend.h"
#endif

namespace xs {

// In order of preference: the first is the CLI's default.
std::vector<std::string> available_backends() {
    std::vector<std::string> out;
#ifdef XS_HAVE_SYCL
    out.emplace_back("sycl");
#endif
    out.emplace_back("cpu-ref");
    return out;
}

std::unique_ptr<Backend> make_backend(std::string_view name) {
#ifdef XS_HAVE_SYCL
    if (name == "sycl") return make_sycl_backend();
#endif
    if (name == "cpu-ref") return make_cpu_ref_backend();
    std::string list;
    for (const auto& b : available_backends()) list += (list.empty() ? "" : ", ") + b;
    fail("backend '", name, "' is not available in this build (available: ", list, ")");
}

}  // namespace xs
