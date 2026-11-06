#pragma once

// Shared setup for the OpenVINO backend tests: the device under test (XS_OV_TEST_DEVICE, default CPU), its
// precision (f32 on CPU, f16 elsewhere), per-precision tolerances and an optional per-op error report
// (XS_OV_TEST_REPORT=1 prints "ov-error <op> max_abs=... max_rel=..." lines).

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>

#include "backend/openvino/ov_backend.h"
#include "backend_test_util.h"
#include "util/log.h"

namespace xs::test::ov {

inline std::string device() {
    const char* d = std::getenv("XS_OV_TEST_DEVICE");
    return d != nullptr && *d != '\0' ? d : "CPU";
}

inline DType precision() {
    return device() == "CPU" ? DType::F32 : DType::F16;
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

// One backend (one ov::Core) for the whole test binary: creating a Core loads the device plugins. XS_OV_TEST_LOG
// sets the log level (e.g. "debug" lists the ops each compiled model runs).
inline Backend& backend() {
    static const std::unique_ptr<Backend> instance = [] {
        if (const char* level = std::getenv("XS_OV_TEST_LOG")) log::set_level(log::parse_level(level));
        return make_openvino_backend();
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
    const char* report = std::getenv("XS_OV_TEST_REPORT");
    if (report != nullptr && std::string(report) == "1") {
        const std::string line = str_cat("ov-error ", op, " max_abs=", max_abs,
                                         " max_rel=", scale > 0 ? max_abs / scale : 0.0, ok ? "" : " FAIL");
        MESSAGE(line);
    }
    return ok;
}

inline TensorMap run(const graph::Graph& g, const WeightStore& weights, const TensorMap& inputs,
                     const SessionOptions& o = options()) {
    return run_once(backend(), g, weights, inputs, o);
}

}  // namespace xs::test::ov
