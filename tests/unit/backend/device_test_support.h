#pragma once

// Shared setup for the GPU backend tests (the SYCL backend): they run when XS_DEVICE_TEST_DEVICE names a device
// (e.g. GPU.0) in a build with the SYCL backend, and skip otherwise. Precision f16, or f32 with
// XS_DEVICE_TEST_PRECISION=f32; per-precision tolerances; XS_DEVICE_TEST_REPORT=1 prints
// "device-error <op> max_abs=... max_rel=..." lines.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "backend_test_util.h"
#include "util/log.h"

namespace xs::test::device {

inline std::string device() {
    const char* d = std::getenv("XS_DEVICE_TEST_DEVICE");
    return d != nullptr ? d : "";
}

inline bool enabled() {
    const std::vector<std::string> built = available_backends();
    return !device().empty() && std::find(built.begin(), built.end(), "sycl") != built.end();
}

// The two attention implementations to cover, as values of the backend's "attention" option.
inline std::vector<std::string> attention_impls() {
    return {"rows", "gemm"};
}

inline DType precision() {
    const char* p = std::getenv("XS_DEVICE_TEST_PRECISION");
    return p != nullptr && std::string(p) == "f32" ? DType::F32 : DType::F16;
}
inline bool is_f16() {
    return precision() == DType::F16;
}

inline SessionOptions options(int64_t max_context = 4096) {
    SessionOptions o;
    o.device = device();
    o.precision = precision();
    o.max_context = max_context;
    return o;
}

// One backend for the whole test binary. XS_DEVICE_TEST_LOG sets the log level.
inline Backend& backend() {
    static const std::unique_ptr<Backend> instance = [] {
        if (const char* level = std::getenv("XS_DEVICE_TEST_LOG")) log::set_level(log::parse_level(level));
        return make_backend("sycl");
    }();
    return *instance;
}

struct Tolerance {
    double atol = 0;
    double rtol = 0;
};

// Picks the tolerance for the precision under test.
inline Tolerance tolerance(Tolerance f32, Tolerance f16) {
    return is_f16() ? f16 : f32;
}

// all_close with the error recorded under `op`. max_rel is max |a - e| / max |e| (a per-element ratio is
// meaningless next to zeros).
inline bool close(const std::string& op, const HostTensor& actual, const Tensor64& expected, Tolerance tol) {
    if (actual.shape() != expected.shape) return false;
    const double max_abs = max_abs_diff(actual, expected);
    double scale = 0;
    for (double e : expected.data)
        scale = std::max(scale, std::abs(e));
    const bool ok = all_close(actual, expected, tol.atol, tol.rtol);
    const char* report = std::getenv("XS_DEVICE_TEST_REPORT");
    if (report != nullptr && std::string(report) == "1") {
        const std::string line = str_cat("device-error ", op, " max_abs=", max_abs,
                                         " max_rel=", scale > 0 ? max_abs / scale : 0.0, ok ? "" : " FAIL");
        MESSAGE(line);
    }
    return ok;
}

inline TensorMap run(const graph::Graph& g, const WeightStore& weights, const TensorMap& inputs,
                     const SessionOptions& o = options()) {
    return run_once(backend(), g, weights, inputs, o);
}

}  // namespace xs::test::device
