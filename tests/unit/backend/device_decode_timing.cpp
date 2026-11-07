// Decode-step timing of a synthetic 35-layer Gemma-shaped graph (T = 1). Opt-in: XS_DEVICE_BENCH=1.
#include <doctest/doctest.h>

#include <cstdlib>
#include <string>

#include "device_test_support.h"
#include "tiny_model.h"
#include "util/error.h"
#include "util/timer.h"

using namespace xs;
namespace dev = xs::test::device;

namespace {

bool bench_enabled() {
    const char* v = std::getenv("XS_DEVICE_BENCH");
    return v != nullptr && std::string(v) == "1";
}

size_t weight_bytes(const test::MemoryWeights& w) {
    size_t total = 0;
    for (const std::string& name : w.names())
        total += w.tensor(name).nbytes();
    return total;
}

void time_decode(const char* what, const dev::TinyModelSpec& spec) {
    test::TensorRng rng(7);
    test::MemoryWeights weights;
    dev::add_tiny_weights(spec, weights, rng);
    const graph::Graph g = dev::tiny_model(spec, weights);
    Timer compile_timer;
    const auto session = dev::backend().compile(g, weights, dev::options());
    const double compile_ms = compile_timer.elapsed_ms();

    constexpr int32_t kContext = 128;
    constexpr int32_t kWarmup = 5;
    constexpr int32_t kSteps = 50;
    TensorMap out;
    session->run(dev::tiny_inputs(spec, 0, kContext), out);
    for (int32_t i = kContext; i < kContext + kWarmup; ++i)
        session->run(dev::tiny_inputs(spec, i, i + 1), out);
    Timer timer;
    for (int32_t i = kContext + kWarmup; i < kContext + kWarmup + kSteps; ++i) {
        session->run(dev::tiny_inputs(spec, i, i + 1), out);
    }
    const double step_ms = timer.elapsed_ms() / kSteps;
    // bf16 weight bytes equal the f16 bytes the device reads per step.
    const double gb_per_s = static_cast<double>(weight_bytes(weights)) / (step_ms * 1e6);
    const std::string line =
        str_cat("device-timing ", what, " on ", dev::device(), ": ", step_ms, " ms/step at context ", kContext,
                " (weights ", weight_bytes(weights) / 1e9, " GB, ", gb_per_s, " GB/s), compile ", compile_ms, " ms");
    MESSAGE(line);
}

}  // namespace

TEST_SUITE("device timing" * doctest::skip(!dev::enabled())) {
    TEST_CASE("decode step of a 35-layer graph, hidden 1536") {
        if (!bench_enabled()) return;
        dev::TinyModelSpec full;
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

        dev::TinyModelSpec overhead = full;  // same graph shape, negligible weights: per-step overhead
        overhead.hidden = 64;
        overhead.head_dim = 64;
        overhead.ffn = 64;
        time_decode("hidden64_overhead", overhead);
    }
}
