# OpenVINO developer-API headers

- **Upstream:** [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino), tag `2026.2.1`, Apache-2.0
  (the OpenVINO runtime's own license).
- **Taken, unchanged:** `src/core/dev_api/openvino/op/gated_delta_net.hpp` (`ov::op::internal::GatedDeltaNet`) and
  `src/core/dev_api/openvino/op/paged_gated_delta_net.hpp` (`ov::op::internal::PagedGatedDeltaNet`).
- **Why:** the runtime exports these ops (constructors, type info) and its CPU and GPU plugins run it, but the 2026.2
  archive ships only the public headers. The OpenVINO backend lowers the graph's `GatedDelta` op to the paged op
  (its GPU kernel is the optimized one) or, with `--backend-opt gdn=ref`, to the plain one.
- **Constraint:** the header must match the runtime it links against; `CMakeLists.txt` refuses other OpenVINO
  versions for this target. Update it together with the OpenVINO pin.
