// Session contract of the cpu-ref backend on a tiny two-layer transformer: step equivalence, truncate, reset and
// the named refusals.
#include <doctest/doctest.h>

#include <cmath>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend_test_util.h"
#include "util/error.h"

using namespace xs;
using namespace xs::graph;

namespace {

constexpr int64_t kVocab = 32;
constexpr int64_t kHidden = 24;
constexpr int64_t kQHeads = 4;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kHeadDim = 8;
constexpr int64_t kFfn = 40;

test::MemoryWeights tiny_weights() {
    test::TensorRng rng(21);
    test::MemoryWeights w;
    w.add("embed", rng.uniform(DType::BF16, {kVocab, kHidden}));
    w.add("final_norm", rng.uniform(DType::BF16, {kHidden}, 2.0f));
    for (const std::string layer : {"l0.", "l1."}) {
        w.add(layer + "norm", rng.uniform(DType::BF16, {kHidden}, 2.0f));
        w.add(layer + "q", rng.uniform(DType::BF16, {kQHeads * kHeadDim, kHidden}, 0.5f));
        w.add(layer + "k", rng.uniform(DType::BF16, {kKvHeads * kHeadDim, kHidden}, 0.5f));
        w.add(layer + "v", rng.uniform(DType::BF16, {kKvHeads * kHeadDim, kHidden}, 0.5f));
        w.add(layer + "o", rng.uniform(DType::F16, {kHidden, kQHeads * kHeadDim}, 0.5f));
        w.add(layer + "up", rng.uniform(DType::F32, {kFfn, kHidden}, 0.5f));
        w.add(layer + "down", rng.uniform(DType::BF16, {kHidden, kFfn}, 0.5f));
    }
    return w;
}

// Layer 0 uses a sliding window of 4, layer 1 attends globally; logits are softcapped.
Graph tiny_model(const WeightStore& w) {
    Graph g;
    const Value ids = g.input("ids", DType::I32, {kT});
    const Value pos = g.input("positions", DType::I32, {kT});
    std::vector<float> inv_freq(kHeadDim / 2);
    for (size_t i = 0; i < inv_freq.size(); ++i)
        inv_freq[i] = std::pow(100.0f, -2.0f * i / kHeadDim);
    const Value freq = g.constant(test::make_tensor(DType::F32, {kHeadDim / 2}, inv_freq));
    Value h = g.gather(g.weight(w, "embed"), ids);
    for (const auto& [layer, window] :
         {std::pair{std::string("l0"), int64_t{4}}, std::pair{std::string("l1"), int64_t{0}}}) {
        const Graph::Scope scope(g, layer);
        const auto wt = [&](const char* name) { return g.weight(w, layer + "." + name); };
        const Value x = g.rms_norm(h, wt("norm"), 1e-6f);
        const Value q = g.rope(g.reshape(g.matmul(x, wt("q")), {kT, kQHeads, kHeadDim}), pos, freq);
        const Value k = g.rope(g.reshape(g.matmul(x, wt("k")), {kT, kKvHeads, kHeadDim}), pos, freq);
        const Value v = g.reshape(g.matmul(x, wt("v")), {kT, kKvHeads, kHeadDim});
        const auto [k_all, v_all] = g.kv_cache(g.add_kv_slot(kKvHeads, kHeadDim, window, layer), k, v);
        const Value attn = g.attention(q, k_all, v_all, {1.0f / std::sqrt(float{kHeadDim}), window, 0.0f});
        h = g.add(h, g.matmul(g.reshape(attn, {kT, kQHeads * kHeadDim}), wt("o")));
        const Value up = g.matmul(g.rms_norm(h, 1e-6f), wt("up"));
        h = g.add(h, g.matmul(g.mul(g.gelu_tanh(up), up), wt("down")));
    }
    const Value logits =
        g.softcap(g.matmul(g.rms_norm(h, g.weight(w, "final_norm"), 1e-6f), g.weight(w, "embed")), 10.0f);
    g.output("logits", logits);
    g.output("last", g.slice(logits, 0, -1));
    return g;
}

const std::vector<int32_t> kPrompt{3, 17, 4, 4, 29, 0, 8, 31, 12, 5};

TensorMap step_inputs(int32_t begin, int32_t end, int32_t position_shift = 0) {
    const std::vector<int32_t> ids(kPrompt.begin() + begin, kPrompt.begin() + end);
    const auto n = static_cast<int64_t>(ids.size());
    return {{"ids", test::make_tensor(DType::I32, {n}, ids)},
            {"positions", test::arange_i32(begin + position_shift, end + position_shift)}};
}

HostTensor run_step(Session& s, int32_t begin, int32_t end, int32_t position_shift = 0) {
    TensorMap out;
    s.run(step_inputs(begin, end, position_shift), out);
    return out.at("logits");
}

struct Fixture {
    test::MemoryWeights weights = tiny_weights();
    std::unique_ptr<Backend> backend = make_cpu_ref_backend();
    Graph graph = tiny_model(weights);
    std::unique_ptr<Session> session = backend->compile(graph, weights, {});
};

}  // namespace

TEST_SUITE("cpu-ref session") {
    TEST_CASE("step equivalence: N tokens at once equal N-k tokens then k single-token steps") {
        Fixture f;
        const HostTensor all = run_step(*f.session, 0, 10);
        CHECK(f.session->num_cached() == 10);
        for (const int32_t k : {1, 3, 7}) {
            CAPTURE(k);
            f.session->reset();
            run_step(*f.session, 0, 10 - k);
            for (int32_t i = 10 - k; i < 10; ++i) {
                CHECK(test::all_close(run_step(*f.session, i, i + 1), test::rows(all, i, i + 1), 1e-5, 0));
            }
        }

        SUBCASE("planted bug: continuing at shifted positions must not match") {
            f.session->reset();
            run_step(*f.session, 0, 7);
            CHECK_FALSE(test::all_close(run_step(*f.session, 7, 8, 1), test::rows(all, 7, 8), 1e-5, 0));
        }
    }

    TEST_CASE("truncate then re-running the dropped tokens reproduces the outputs") {
        Fixture f;
        const HostTensor all = run_step(*f.session, 0, 10);
        f.session->truncate(6);
        CHECK(f.session->num_cached() == 6);
        CHECK(test::all_close(run_step(*f.session, 6, 10), test::rows(all, 6, 10), 1e-5, 0));
        CHECK(f.session->num_cached() == 10);
        CHECK_THROWS_AS(f.session->truncate(11), xs::Error);
    }

    TEST_CASE("reset clears the state") {
        Fixture f;
        const HostTensor first = run_step(*f.session, 0, 5);
        run_step(*f.session, 5, 10);
        f.session->reset();
        CHECK(f.session->num_cached() == 0);
        CHECK(test::all_close(run_step(*f.session, 0, 5), first, 0, 0));
    }

    TEST_CASE("outputs are fresh f32 tensors and the slice is the last row") {
        Fixture f;
        TensorMap out;
        f.session->run(step_inputs(0, 4), out);
        CHECK(out.at("logits").dtype() == DType::F32);
        CHECK(test::all_close(out.at("last"), test::rows(out.at("logits"), 3, 4), 0, 0));
        out.at("logits").data<float>()[0] = 1e9f;  // writing an output must not disturb the session
        f.session->reset();
        TensorMap again;
        f.session->run(step_inputs(0, 4), again);
        CHECK(again.at("logits").data<float>()[0] != 1e9f);
    }

    TEST_CASE("bad inputs are refused by name and leave the state unchanged") {
        Fixture f;
        run_step(*f.session, 0, 3);
        TensorMap out;
        TensorMap missing = step_inputs(3, 5);
        missing.erase("positions");
        CHECK_THROWS_WITH_AS(f.session->run(missing, out), doctest::Contains("missing graph input 'positions'"),
                             xs::Error);
        TensorMap unknown = step_inputs(3, 5);
        unknown["typo"] = test::arange_i32(0, 2);
        CHECK_THROWS_WITH_AS(f.session->run(unknown, out), doctest::Contains("'typo'"), xs::Error);
        TensorMap wrong_dtype = step_inputs(3, 5);
        wrong_dtype["ids"] = test::make_tensor<int64_t>(DType::I64, {2}, {1, 2});
        CHECK_THROWS_WITH_AS(f.session->run(wrong_dtype, out), doctest::Contains("input 'ids' must be i32"), xs::Error);
        TensorMap mismatched_t = step_inputs(3, 5);
        mismatched_t["positions"] = test::arange_i32(3, 6);
        CHECK_THROWS_WITH_AS(f.session->run(mismatched_t, out), doctest::Contains("tokens"), xs::Error);
        TensorMap bad_id = step_inputs(3, 5);
        bad_id["ids"] = test::make_tensor<int32_t>(DType::I32, {2}, {1, 99});
        CHECK_THROWS_AS(f.session->run(bad_id, out), xs::Error);
        CHECK(f.session->num_cached() == 3);
        const HostTensor all = run_step(*Fixture{}.session, 0, 5);
        CHECK(test::all_close(run_step(*f.session, 3, 5), test::rows(all, 3, 5), 1e-5, 0));
    }

    TEST_CASE("max_context bounds the cache") {
        Fixture f;
        SessionOptions options;
        options.max_context = 6;
        const auto session = f.backend->compile(f.graph, f.weights, options);
        run_step(*session, 0, 5);
        CHECK_THROWS_WITH_AS(run_step(*session, 5, 7), doctest::Contains("exceed max_context 6"), xs::Error);
        CHECK(session->num_cached() == 5);
        CHECK_NOTHROW(run_step(*session, 5, 6));
    }

    TEST_CASE("compile refuses unsupported options and mismatched weights by name") {
        Fixture f;
        SessionOptions gpu;
        gpu.device = "GPU.0";
        CHECK_THROWS_WITH_AS(f.backend->compile(f.graph, f.weights, gpu), doctest::Contains("'GPU.0'"), xs::Error);
        SessionOptions bf16;
        bf16.precision = DType::BF16;
        CHECK_THROWS_WITH_AS(f.backend->compile(f.graph, f.weights, bf16), doctest::Contains("precision"), xs::Error);
        SessionOptions extra;
        extra.extra["num_streams"] = "2";
        CHECK_THROWS_WITH_AS(f.backend->compile(f.graph, f.weights, extra), doctest::Contains("'num_streams'"),
                             xs::Error);
        SessionOptions f32;
        f32.precision = DType::F32;
        CHECK_NOTHROW(f.backend->compile(f.graph, f.weights, f32));

        test::MemoryWeights reshaped = tiny_weights();
        reshaped.add("final_norm", HostTensor(DType::F32, {kHidden + 1}));
        CHECK_THROWS_WITH_AS(f.backend->compile(f.graph, reshaped, {}), doctest::Contains("'final_norm'"), xs::Error);
        test::MemoryWeights int_weights = tiny_weights();
        int_weights.add("final_norm", HostTensor(DType::I32, {kHidden}));
        CHECK_THROWS_WITH_AS(f.backend->compile(f.graph, int_weights, {}), doctest::Contains("dtype i32"), xs::Error);
    }

    TEST_CASE("the backend names itself and its one device") {
        const auto backend = make_backend("cpu-ref");
        CHECK(backend->name() == "cpu-ref");
        const auto devices = backend->devices();
        REQUIRE(devices.size() == 1);
        CHECK(devices[0].id == "CPU");
        CHECK(!devices[0].name.empty());
    }
}
