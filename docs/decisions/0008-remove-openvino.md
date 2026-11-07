# 0008: Remove the OpenVINO backend

Date: 2026-11-07 · Status: accepted

The SYCL backend ([0007](0007-sycl-backend.md)) is the GPU backend, and nothing needs OpenVINO any more: NPUs are out
of scope for now, and CPU inference is not a goal. So `src/backend/openvino/`, the vendored OpenVINO dev headers and
the OpenVINO build option are gone; a build needs only oneAPI (icpx and oneMKL), and a plain C++20 compiler still
builds everything except the GPU backend.

What stays:
- The CPU reference backend, as the test oracle.
- The backend-neutral GPU suites, renamed from `ov_*` to `device_*` (ops, attention, encoder ops, recurrent ops,
  sessions, agreement with cpu-ref, decode timing); they run on the SYCL backend when `XS_DEVICE_TEST_DEVICE` is set.
- OpenVINO GenAI as an external speed baseline in benchmarks.

The OpenVINO backend remains in git history (last at commit 2525805) should a CPU or NPU path be wanted later.
