#pragma once

// What the tests that hold one model against another share: a model's logits step after step on a backend, and the
// gap between two such runs.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "backend/backend.h"
#include "models/model.h"

namespace xs::test {

// The logits of a prefill of `prompt` and of one decode step per token of `steps` after it (the state carried).
inline std::vector<std::vector<float>> step_logits(Backend& backend, const Model& model, const SessionOptions& options,
                                                   const std::vector<int32_t>& prompt,
                                                   const std::vector<int32_t>& steps) {
    const auto session = backend.compile(model.graph(), model.weights(), options);
    std::vector<std::vector<int32_t>> feeds{prompt};
    for (const int32_t t : steps)
        feeds.push_back({t});
    std::vector<std::vector<float>> out;
    int64_t pos = 0;
    for (const std::vector<int32_t>& ids : feeds) {
        TensorMap inputs, outputs;
        model.prepare_inputs(ids, pos, {}, inputs);
        session->run(inputs, outputs);
        const HostTensor& l = outputs.at("logits");
        out.emplace_back(l.data<float>(), l.data<float>() + l.numel());
        pos += static_cast<int64_t>(ids.size());
    }
    return out;
}

// The largest difference between two runs' logits, over the largest logit of `b`.
inline double relative_gap(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    double gap = 0.0, scale = 0.0;
    for (size_t s = 0; s < a.size(); ++s) {
        for (size_t i = 0; i < a[s].size(); ++i) {
            gap = std::max(gap, static_cast<double>(std::fabs(a[s][i] - b[s][i])));
            scale = std::max(scale, static_cast<double>(std::fabs(b[s][i])));
        }
    }
    return gap / scale;
}

}  // namespace xs::test
