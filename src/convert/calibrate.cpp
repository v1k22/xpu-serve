#include "convert/calibrate.h"

#include <algorithm>

#include "backend/backend.h"
#include "graph/graph.h"
#include "util/error.h"
#include "util/log.h"

namespace xs {

std::map<std::string, std::vector<double>> input_channel_mean_squares(const Model& model,
                                                                      const std::vector<int32_t>& tokens,
                                                                      const std::set<std::string>& weights,
                                                                      const CalibrationOptions& o) {
    XS_CHECK(!tokens.empty(), "calibration: no tokens");
    // The model's graph plus one output per distinct input of a matmul that reads one of `weights`.
    graph::Graph g = model.graph();
    std::map<std::string, std::string> tap_of;  // weight -> output
    std::map<std::pair<int32_t, int32_t>, std::string> taps;
    const size_t nodes = g.nodes().size();
    for (size_t i = 0; i < nodes; ++i) {
        const graph::Node n = g.nodes()[i];
        if (n.kind != graph::OpKind::MatMul || g.node(n.inputs[1]).kind != graph::OpKind::Weight) continue;
        const std::string& w = std::get<graph::WeightAttrs>(g.node(n.inputs[1]).attrs).name;
        if (weights.count(w) == 0) continue;
        const auto key = std::make_pair(n.inputs[0].node, n.inputs[0].index);
        auto it = taps.find(key);
        if (it == taps.end()) {
            const std::string name = str_cat("calib.", key.first, ".", key.second);
            g.output(name, n.inputs[0]);
            it = taps.emplace(key, name).first;
        }
        tap_of[w] = it->second;
    }

    const std::string backend_name = o.backend.empty() ? available_backends().front() : o.backend;
    SessionOptions so;
    so.device = o.device;
    so.precision = backend_name == "cpu-ref" ? DType::F32 : DType::F16;
    so.max_context = o.sample_tokens;
    const auto session = make_backend(backend_name)->compile(g, model.weights(), so);

    std::map<std::string, std::vector<double>> sums;
    std::map<std::string, int64_t> rows;
    const int64_t total = std::min<int64_t>(static_cast<int64_t>(tokens.size()), o.max_tokens);
    for (int64_t start = 0; start < total; start += o.sample_tokens) {
        const int64_t count = std::min(o.sample_tokens, total - start);
        const std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + start + count);
        session->reset();
        TensorMap inputs, outputs;
        model.prepare_inputs(ids, 0, {}, inputs);
        session->run(inputs, outputs);
        for (const auto& [key, name] : taps) {
            const HostTensor& t = outputs.at(name);
            const int64_t width = t.shape().back();
            std::vector<double>& s = sums[name];
            s.resize(static_cast<size_t>(width), 0.0);
            const float* x = t.data<float>();
            for (int64_t r = 0; r < t.numel() / width; ++r) {
                for (int64_t c = 0; c < width; ++c) {
                    const double v = x[r * width + c];
                    s[static_cast<size_t>(c)] += v * v;
                }
            }
            rows[name] += t.numel() / width;
        }
    }
    log::info("calibration: ", total, " tokens on ", backend_name, ", ", taps.size(), " matmul inputs");
    std::map<std::string, std::vector<double>> out;
    for (const auto& [w, name] : tap_of) {
        std::vector<double> m = sums.at(name);
        for (double& v : m)
            v /= static_cast<double>(rows.at(name));
        out[w] = std::move(m);
    }
    return out;
}

}  // namespace xs
