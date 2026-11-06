// Per-op A/B: the same random graphs and steps on the OpenVINO backend and on the cpu-ref oracle.
#include <doctest/doctest.h>

#include "backend/cpu_ref/cpu_backend.h"
#include "ov_test_support.h"
#include "ov_tiny_model.h"

using namespace xs;
using xs::test::to_f64;
namespace ovt = xs::test::ov;

namespace {

const ovt::Tolerance kCross = ovt::tolerance({1e-4, 1e-4}, {3e-2, 2e-2});

// Runs `steps` (token counts) through both backends and compares every output of every step.
void compare_backends(const ovt::TinyModelSpec& spec, const std::vector<int32_t>& steps, uint32_t seed) {
    test::TensorRng rng(seed);
    test::MemoryWeights weights;
    ovt::add_tiny_weights(spec, weights, rng);
    const graph::Graph g = ovt::tiny_model(spec, weights);
    const auto oracle = make_cpu_ref_backend()->compile(g, weights, SessionOptions{});
    const auto ov = ovt::backend().compile(g, weights, ovt::options());
    int32_t begin = 0;
    for (const int32_t t : steps) {
        CAPTURE(begin);
        const TensorMap inputs = ovt::tiny_inputs(spec, begin, begin + t);
        TensorMap expected, actual;
        oracle->run(inputs, expected);
        ov->run(inputs, actual);
        for (const auto& [name, tensor] : expected) {
            CAPTURE(name);
            CHECK(ovt::close("cross_backend", actual.at(name), to_f64(tensor), kCross));
        }
        begin += t;
    }
    CHECK(ov->num_cached() == oracle->num_cached());
}

}  // namespace

TEST_SUITE("openvino vs cpu-ref") {
    TEST_CASE("tiny decoder: sliding and global layers, GQA, prefill then decode") {
        compare_backends({}, {7, 1, 1, 3}, 1);
    }

    TEST_CASE("tiny decoder: head_dim 256 with attention softcap, MQA") {
        ovt::TinyModelSpec spec;
        spec.head_dim = 256;
        spec.heads = 4;
        spec.kv_heads = 1;
        spec.attn_softcap = 50.0f;
        spec.layers = 3;
        compare_backends(spec, {5, 1, 2}, 2);
    }

    TEST_CASE("tiny decoder: head_dim 512, window longer than the prefill") {
        ovt::TinyModelSpec spec;
        spec.head_dim = 512;
        spec.heads = 2;
        spec.kv_heads = 1;
        spec.sliding_window = 16;
        spec.logits = false;
        compare_backends(spec, {12, 1, 1, 6}, 3);
    }
}
