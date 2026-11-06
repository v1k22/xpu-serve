// Session contract of the OpenVINO backend on a tiny decoder: step equivalence, truncate, reset, input validation,
// max_context, options and devices.
#include <doctest/doctest.h>

#include <filesystem>

#include "ov_test_support.h"
#include "ov_tiny_model.h"
#include "util/error.h"

using namespace xs;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace ovt = xs::test::ov;

namespace {

// Same math through differently shaped kernels (T = 9 vs T = 1): f32 reassociation, or f16 rounding of every
// intermediate over two layers.
const ovt::Tolerance kStep = ovt::tolerance({1e-4, 1e-4}, {3e-2, 2e-2});
constexpr int32_t kPrompt = 9;

struct TinyFixture {
    ovt::TinyModelSpec spec;
    test::MemoryWeights weights;
    graph::Graph graph;

    explicit TinyFixture(ovt::TinyModelSpec s = {}) : spec(s) {
        test::TensorRng rng(42);
        ovt::add_tiny_weights(spec, weights, rng);
        graph = ovt::tiny_model(spec, weights);
    }

    std::unique_ptr<Session> session(SessionOptions o = ovt::options()) const {
        return ovt::backend().compile(graph, weights, o);
    }
};

HostTensor run_logits(Session& s, const ovt::TinyModelSpec& spec, int32_t begin, int32_t end) {
    TensorMap out;
    s.run(ovt::tiny_inputs(spec, begin, end), out);
    return out.at("logits");
}

}  // namespace

TEST_SUITE("openvino session") {
    TEST_CASE("step equivalence: N tokens at once equal N-k tokens then k single-token steps") {
        const TinyFixture f;
        const auto whole = f.session();
        const HostTensor all = run_logits(*whole, f.spec, 0, kPrompt);
        CHECK(whole->num_cached() == kPrompt);
        const auto stepped = f.session();
        const HostTensor head = run_logits(*stepped, f.spec, 0, kPrompt - 3);
        CHECK(ovt::close("session_prefill", head, to_f64(test::rows(all, 0, kPrompt - 3)), kStep));
        for (int32_t i = kPrompt - 3; i < kPrompt; ++i) {
            CAPTURE(i);
            const HostTensor row = run_logits(*stepped, f.spec, i, i + 1);
            CHECK(ovt::close("session_decode", row, to_f64(test::rows(all, i, i + 1)), kStep));
        }
        CHECK(stepped->num_cached() == kPrompt);
    }

    TEST_CASE("truncate then re-running the dropped tokens reproduces the outputs") {
        const TinyFixture f;
        const auto s = f.session();
        const HostTensor all = run_logits(*s, f.spec, 0, kPrompt);
        for (const int32_t keep : {6, 2}) {
            CAPTURE(keep);
            s->truncate(keep);
            CHECK(s->num_cached() == keep);
            const HostTensor redo = run_logits(*s, f.spec, keep, kPrompt);
            CHECK(ovt::close("truncate", redo, to_f64(test::rows(all, keep, kPrompt)), kStep));
        }
        s->truncate(kPrompt);
        CHECK(s->num_cached() == kPrompt);
        CHECK_THROWS_WITH_AS(s->truncate(kPrompt + 1), doctest::Contains("truncate"), Error);
    }

    TEST_CASE("reset clears the state") {
        const TinyFixture f;
        const auto s = f.session();
        const HostTensor first = run_logits(*s, f.spec, 0, kPrompt);
        s->reset();
        CHECK(s->num_cached() == 0);
        const HostTensor again = run_logits(*s, f.spec, 0, kPrompt);
        CHECK(ovt::close("reset", again, to_f64(first), {1e-6, 1e-6}));
        s->truncate(0);
        CHECK(s->num_cached() == 0);
        CHECK(ovt::close("reset", run_logits(*s, f.spec, 0, kPrompt), to_f64(first), {1e-6, 1e-6}));
    }

    TEST_CASE("outputs are fresh f32 host tensors and 'last' is the last row") {
        const TinyFixture f;
        const auto s = f.session();
        TensorMap out;
        s->run(ovt::tiny_inputs(f.spec, 0, 5), out);
        const HostTensor logits = out.at("logits");
        CHECK(logits.dtype() == DType::F32);
        CHECK(logits.shape() == Dims{5, f.spec.vocab});
        CHECK(ovt::close("last", out.at("last"), to_f64(test::rows(logits, 4, 5)), {0, 0}));
        TensorMap next;
        s->run(ovt::tiny_inputs(f.spec, 5, 6), next);
        CHECK(next.at("logits").raw() != logits.raw());
        CHECK(logits.shape() == Dims{5, f.spec.vocab});
    }

    TEST_CASE("bad inputs are refused by name and leave the state unchanged") {
        const TinyFixture f;
        const auto s = f.session();
        const HostTensor reference = run_logits(*f.session(), f.spec, 0, 4);
        run_logits(*s, f.spec, 0, 2);
        TensorMap out;
        TensorMap missing = ovt::tiny_inputs(f.spec, 2, 4);
        missing.erase("pos");
        CHECK_THROWS_WITH_AS(s->run(missing, out), doctest::Contains("'pos' is missing"), Error);
        TensorMap unknown = ovt::tiny_inputs(f.spec, 2, 4);
        unknown["extra"] = test::arange_i32(0, 2);
        CHECK_THROWS_WITH_AS(s->run(unknown, out), doctest::Contains("'extra' is not an input"), Error);
        TensorMap wrong_dtype = ovt::tiny_inputs(f.spec, 2, 4);
        wrong_dtype["ids"] = HostTensor(DType::I64, {2});
        CHECK_THROWS_WITH_AS(s->run(wrong_dtype, out), doctest::Contains("'ids' has dtype i64"), Error);
        TensorMap wrong_rank = ovt::tiny_inputs(f.spec, 2, 4);
        wrong_rank["ids"] = HostTensor(DType::I32, {2, 1});
        CHECK_THROWS_WITH_AS(s->run(wrong_rank, out), doctest::Contains("'ids' has shape [2, 1]"), Error);
        TensorMap mixed_t = ovt::tiny_inputs(f.spec, 2, 4);
        mixed_t["pos"] = test::arange_i32(2, 5);
        CHECK_THROWS_WITH_AS(s->run(mixed_t, out), doctest::Contains("T = 3"), Error);
        CHECK(s->num_cached() == 2);
        const HostTensor rest = run_logits(*s, f.spec, 2, 4);
        CHECK(ovt::close("validation", rest, to_f64(test::rows(reference, 2, 4)), kStep));
    }

    TEST_CASE("max_context bounds the cache") {
        const TinyFixture f;
        const auto s = f.session(ovt::options(8));
        run_logits(*s, f.spec, 0, 6);
        CHECK_THROWS_WITH_AS(run_logits(*s, f.spec, 6, 9), doctest::Contains("max_context 8"), Error);
        CHECK(s->num_cached() == 6);
        run_logits(*s, f.spec, 6, 8);
        CHECK(s->num_cached() == 8);
    }

    TEST_CASE("compile refuses unknown and malformed options by name") {
        const TinyFixture f;
        SessionOptions unknown = ovt::options();
        unknown.extra["num_streams"] = "2";
        CHECK_THROWS_WITH_AS(f.session(unknown), doctest::Contains("'num_streams' is unknown"), Error);
        SessionOptions bad_factor = ovt::options();
        bad_factor.extra["activations_scale_factor"] = "eight";
        CHECK_THROWS_WITH_AS(f.session(bad_factor), doctest::Contains("activations_scale_factor='eight'"), Error);
        SessionOptions bad_impl = ovt::options();
        bad_impl.extra["attention"] = "flash";
        CHECK_THROWS_WITH_AS(f.session(bad_impl), doctest::Contains("'flash'"), Error);
        SessionOptions bf16 = ovt::options();
        bf16.precision = DType::BF16;
        CHECK_THROWS_WITH_AS(f.session(bf16), doctest::Contains("precision must be f16 or f32"), Error);
        SessionOptions no_device = ovt::options();
        no_device.device = "NPU.7";
        CHECK_THROWS_WITH_AS(f.session(no_device), doctest::Contains("NPU.7"), Error);
    }

    TEST_CASE("activations_scale_factor is honoured where the device has it, refused by name elsewhere") {
        const TinyFixture f;
        SessionOptions o = ovt::options();
        o.extra["activations_scale_factor"] = "8";
        if (!ovt::is_f16()) {
            CHECK_THROWS_WITH_AS(f.session(o), doctest::Contains("does not support ACTIVATIONS_SCALE_FACTOR"), Error);
            return;
        }
        const HostTensor plain = run_logits(*f.session(), f.spec, 0, kPrompt);
        CHECK(ovt::close("scaled", run_logits(*f.session(o), f.spec, 0, kPrompt), to_f64(plain), kStep));
    }

    TEST_CASE("cache_dir is used and keeps the results") {
        const TinyFixture f;
        const HostTensor plain = run_logits(*f.session(), f.spec, 0, kPrompt);
        SessionOptions o = ovt::options();
        o.cache_dir = (std::filesystem::temp_directory_path() / "xs_ov_test_cache").string();
        for (int round = 0; round < 2; ++round) {  // the second compile may load from the cache
            CAPTURE(round);
            CHECK(ovt::close("cached", run_logits(*f.session(o), f.spec, 0, kPrompt), to_f64(plain), {1e-6, 1e-6}));
        }
        CHECK_FALSE(std::filesystem::is_empty(o.cache_dir));
        std::filesystem::remove_all(o.cache_dir);
    }

    TEST_CASE("devices lists the device under test with a name and memory") {
        const std::vector<DeviceInfo> devices = ovt::backend().devices();
        REQUIRE_FALSE(devices.empty());
        const std::string family = ovt::device().substr(0, 3);
        bool found = false;
        for (const DeviceInfo& d : devices) {
            CAPTURE(d.id);
            CHECK_FALSE(d.name.empty());
            CHECK(d.memory_bytes > 0);
            found = found || d.id.rfind(family, 0) == 0;
        }
        CHECK(found);
    }
}
