// Per-op A/B: the same random graphs and steps on the GPU backend and on the cpu-ref oracle.
#include <doctest/doctest.h>

#include "backend/cpu_ref/cpu_backend.h"
#include "device_test_support.h"
#include "tiny_model.h"

using namespace xs;
using xs::test::to_f64;
namespace dev = xs::test::device;

namespace {

const dev::Tolerance kCross = dev::tolerance({1e-4, 1e-4}, {3e-2, 2e-2});

// Runs `steps` (token counts) through both backends and compares every output of every step.
void compare_backends(const dev::TinyModelSpec& spec, const std::vector<int32_t>& steps, uint32_t seed,
                      bool kv_i8 = false) {
    test::TensorRng rng(seed);
    test::MemoryWeights weights;
    dev::add_tiny_weights(spec, weights, rng);
    const graph::Graph g = dev::tiny_model(spec, weights);
    SessionOptions oracle_options;
    SessionOptions gpu_options = dev::options();
    if (kv_i8) oracle_options.kv_dtype = gpu_options.kv_dtype = DType::I8;
    const auto oracle = make_cpu_ref_backend()->compile(g, weights, oracle_options);
    const auto gpu = dev::backend().compile(g, weights, gpu_options);
    int32_t begin = 0;
    for (const int32_t t : steps) {
        CAPTURE(begin);
        const TensorMap inputs = dev::tiny_inputs(spec, begin, begin + t);
        TensorMap expected, actual;
        oracle->run(inputs, expected);
        gpu->run(inputs, actual);
        for (const auto& [name, tensor] : expected) {
            CAPTURE(name);
            CHECK(dev::close("cross_backend", actual.at(name), to_f64(tensor), kCross));
        }
        begin += t;
    }
    CHECK(gpu->num_cached() == oracle->num_cached());
}

}  // namespace

TEST_SUITE("device vs cpu-ref" * doctest::skip(!dev::enabled())) {
    TEST_CASE("tiny decoder: sliding and global layers, GQA, prefill then decode") {
        compare_backends({}, {7, 1, 1, 3}, 1);
    }

    TEST_CASE("tiny decoder: head_dim 256 with attention softcap, MQA") {
        dev::TinyModelSpec spec;
        spec.head_dim = 256;
        spec.heads = 4;
        spec.kv_heads = 1;
        spec.attn_softcap = 50.0f;
        spec.layers = 3;
        compare_backends(spec, {5, 1, 2}, 2);
    }

    TEST_CASE("tiny decoder: head_dim 512, window longer than the prefill") {
        dev::TinyModelSpec spec;
        spec.head_dim = 512;
        spec.heads = 2;
        spec.kv_heads = 1;
        spec.sliding_window = 16;
        spec.logits = false;
        compare_backends(spec, {12, 1, 1, 6}, 3);
    }

    TEST_CASE("tiny decoder with an int8 KV cache on both backends: sliding and global layers, GQA") {
        compare_backends({}, {7, 1, 1, 3}, 4, true);
        dev::TinyModelSpec spec;
        spec.head_dim = 256;
        spec.heads = 4;
        spec.kv_heads = 1;
        spec.sliding_window = 4;
        compare_backends(spec, {9, 1, 2}, 5, true);
    }
}
