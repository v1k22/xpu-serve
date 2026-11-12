#pragma once

// What the GGUF import tests share: importing a file, a model's logits and greedy tokens on cpu-ref, a weight's
// values, and the error a changed file gives.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "convert/gguf/gguf_import.h"
#include "io/gguf_test_util.h"
#include "models/model.h"
#include "quant/quant.h"
#include "util/error.h"

namespace xs::test {

inline gguf_import::ImportReport import_into(const std::filesystem::path& gguf, const std::filesystem::path& out) {
    gguf_import::ImportOptions o;
    o.source = gguf;
    o.output = out;
    return gguf_import::import_gguf(o);
}

inline SessionOptions f32_session() {
    SessionOptions so;
    so.precision = DType::F32;
    return so;
}

// The logits of a prefill of `prompt` and of one decode step per token of `steps` after it.
inline std::vector<std::vector<float>> step_logits(const Model& model, const std::vector<int32_t>& prompt,
                                                   const std::vector<int32_t>& steps) {
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(model.graph(), model.weights(), f32_session());
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

// `tokens` greedy tokens after `prompt`.
inline std::vector<int32_t> greedy(const Model& model, const std::vector<int32_t>& prompt, int tokens) {
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(model.graph(), model.weights(), f32_session());
    std::vector<int32_t> out, feed = prompt;
    int64_t pos = 0;
    for (int i = 0; i < tokens; ++i) {
        TensorMap inputs, outputs;
        model.prepare_inputs(feed, pos, {}, inputs);
        session->run(inputs, outputs);
        pos += static_cast<int64_t>(feed.size());
        const HostTensor& l = outputs.at("logits");
        const float* row = l.data<float>();
        const auto best = static_cast<int32_t>(std::max_element(row, row + l.numel()) - row);
        out.push_back(best);
        feed = {best};
    }
    return out;
}

inline std::vector<float> values(const TensorView& v) {
    std::vector<float> out(static_cast<size_t>(v.numel()));
    quant::read_rows(v, 0, quant::matrix_rows(v.shape), out.data());
    return out;
}

// The GGUF `build(dir / "src")` writes, with `change` applied, imported into a fresh directory: the error it gives,
// or "" when it imports.
template <typename Build, typename Change>
std::string refusal(const std::filesystem::path& dir, Build build, Change change) {
    GgufWriter w = build(dir / "src");
    change(w);
    w.write(dir / "variant.gguf");
    std::filesystem::remove_all(dir / "variant");
    try {
        import_into(dir / "variant.gguf", dir / "variant");
    } catch (const Error& e) {
        return e.what();
    }
    return "";
}

}  // namespace xs::test
