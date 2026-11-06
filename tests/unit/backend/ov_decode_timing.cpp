// Decode-step timing of a synthetic 35-layer Gemma-shaped graph (T = 1). Opt-in: XS_OV_BENCH=1.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "ov_test_support.h"
#include "ov_tiny_model.h"
#include "util/timer.h"

using namespace xs;
namespace ovt = xs::test::ov;

namespace {

bool bench_enabled() {
    const char* v = std::getenv("XS_OV_BENCH");
    return v != nullptr && std::string(v) == "1";
}

size_t weight_bytes(const test::MemoryWeights& w) {
    size_t total = 0;
    for (const std::string& name : w.names()) total += w.tensor(name).nbytes();
    return total;
}

void time_decode(const char* what, const ovt::TinyModelSpec& spec) {
    test::TensorRng rng(7);
    test::MemoryWeights weights;
    ovt::add_tiny_weights(spec, weights, rng);
    const graph::Graph g = ovt::tiny_model(spec, weights);
    Timer compile_timer;
    const auto session = ovt::backend().compile(g, weights, ovt::options());
    const double compile_ms = compile_timer.elapsed_ms();

    constexpr int32_t kContext = 128;
    constexpr int32_t kWarmup = 5;
    constexpr int32_t kSteps = 50;
    TensorMap out;
    session->run(ovt::tiny_inputs(spec, 0, kContext), out);
    for (int32_t i = kContext; i < kContext + kWarmup; ++i) session->run(ovt::tiny_inputs(spec, i, i + 1), out);
    Timer timer;
    for (int32_t i = kContext + kWarmup; i < kContext + kWarmup + kSteps; ++i) {
        session->run(ovt::tiny_inputs(spec, i, i + 1), out);
    }
    const double step_ms = timer.elapsed_ms() / kSteps;
    // bf16 weight bytes equal the f16 bytes the device reads per step.
    const double gb_per_s = static_cast<double>(weight_bytes(weights)) / (step_ms * 1e6);
    MESSAGE("ov-timing ", what, " on ", ovt::device(), ": ", step_ms, " ms/step at context ", kContext,
            " (weights ", weight_bytes(weights) / 1e9, " GB, ", gb_per_s, " GB/s), compile ", compile_ms, " ms");
}

}  // namespace

TEST_SUITE("openvino timing") {
    TEST_CASE("decode step of a 35-layer graph, hidden 1536") {
        if (!bench_enabled()) return;
        ovt::TinyModelSpec full;
        full.vocab = 256;
        full.hidden = 1536;
        full.heads = 8;
        full.kv_heads = 1;
        full.head_dim = 256;
        full.ffn = 6144;
        full.layers = 35;
        full.sliding_window = 512;
        full.global_every = 5;
        time_decode("hidden1536_ffn6144", full);

        ovt::TinyModelSpec overhead = full;  // same graph shape, negligible weights: per-step overhead
        overhead.hidden = 64;
        overhead.head_dim = 64;
        overhead.ffn = 64;
        time_decode("hidden64_overhead", overhead);
    }
}
