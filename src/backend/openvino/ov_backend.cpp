#include "backend/openvino/ov_backend.h"

#include "util/error.h"

namespace xs {

// Placeholder until the OpenVINO backend lands.
std::unique_ptr<Backend> make_openvino_backend() { fail("the openvino backend is not implemented yet"); }

}  // namespace xs
