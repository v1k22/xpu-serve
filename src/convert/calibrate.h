#pragma once

// AWQ calibration: runs text through a model on a backend and measures, for each requested matmul weight, the mean
// square of every channel of that matmul's input. The model's graph is copied with the inputs as extra outputs, so
// any backend and any model work unchanged.

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "models/model.h"

namespace xs {

struct CalibrationOptions {
    std::string backend;            // empty: the first backend of this build (sycl when built)
    std::string device;             // empty: the backend's default
    int64_t max_tokens = 8192;      // tokens used, at most
    int64_t sample_tokens = 512;    // tokens per sample; each sample starts a fresh sequence
};

// weight name -> mean of x[c]^2 over every calibration token, per input channel c.
std::map<std::string, std::vector<double>> input_channel_mean_squares(const Model& model,
                                                                      const std::vector<int32_t>& tokens,
                                                                      const std::set<std::string>& weights,
                                                                      const CalibrationOptions& options);

// A few thousand tokens of varied text (prose, science, history, code, math, several languages), written for this
// project, used when no calibration file is given.
const std::string& builtin_calibration_text();

}  // namespace xs
