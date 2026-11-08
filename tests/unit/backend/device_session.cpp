// Session contract of the GPU backend on a tiny decoder: step equivalence, truncate, reset, input validation,
// max_context, options and devices.
#include <doctest/doctest.h>

#include <filesystem>

#include "device_test_support.h"
#include "tiny_model.h"
#include "util/error.h"

using namespace xs;
using xs::test::Tensor64;
using xs::test::to_f64;
namespace dev = xs::test::device;

namespace {

// Same math through differently shaped kernels (T = 9 vs T = 1): f32 reassociation, or f16 rounding of every
// intermediate over two layers.
const dev::Tolerance kStep = dev::tolerance({1e-4, 1e-4}, {3e-2, 2e-2});
constexpr int32_t kPrompt = 9;

struct TinyFixture {
    dev::TinyModelSpec spec;
    test::MemoryWeights weights;
    graph::Graph graph;

    explicit TinyFixture(dev::TinyModelSpec s = {}) : spec(s) {
        test::TensorRng rng(42);
        dev::add_tiny_weights(spec, weights, rng);
        graph = dev::tiny_model(spec, weights);
    }

    std::unique_ptr<Session> session(SessionOptions o = dev::options()) const {
        return dev::backend().compile(graph, weights, o);
    }
};

HostTensor run_logits(Session& s, const dev::TinyModelSpec& spec, int32_t begin, int32_t end) {
    TensorMap out;
    s.run(dev::tiny_inputs(spec, begin, end), out);
    return out.at("logits");
}

}  // namespace

TEST_SUITE("device session" * doctest::skip(!dev::enabled())) {
    TEST_CASE("step equivalence: N tokens at once equal N-k tokens then k single-token steps") {
        const TinyFixture f;
        const auto whole = f.session();
        const HostTensor all = run_logits(*whole, f.spec, 0, kPrompt);
        CHECK(whole->num_cached() == kPrompt);
        const auto stepped = f.session();
        const HostTensor head = run_logits(*stepped, f.spec, 0, kPrompt - 3);
        CHECK(dev::close("session_prefill", head, to_f64(test::rows(all, 0, kPrompt - 3)), kStep));
        for (int32_t i = kPrompt - 3; i < kPrompt; ++i) {
            CAPTURE(i);
            const HostTensor row = run_logits(*stepped, f.spec, i, i + 1);
            CHECK(dev::close("session_decode", row, to_f64(test::rows(all, i, i + 1)), kStep));
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
            CHECK(dev::close("truncate", redo, to_f64(test::rows(all, keep, kPrompt)), kStep));
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
        CHECK(dev::close("reset", again, to_f64(first), {1e-6, 1e-6}));
        s->truncate(0);
        CHECK(s->num_cached() == 0);
        CHECK(dev::close("reset", run_logits(*s, f.spec, 0, kPrompt), to_f64(first), {1e-6, 1e-6}));
    }

    TEST_CASE("outputs are fresh f32 host tensors and 'last' is the last row") {
        const TinyFixture f;
        const auto s = f.session();
        TensorMap out;
        s->run(dev::tiny_inputs(f.spec, 0, 5), out);
        const HostTensor logits = out.at("logits");
        CHECK(logits.dtype() == DType::F32);
        CHECK(logits.shape() == Dims{5, f.spec.vocab});
        CHECK(dev::close("last", out.at("last"), to_f64(test::rows(logits, 4, 5)), {0, 0}));
        TensorMap next;
        s->run(dev::tiny_inputs(f.spec, 5, 6), next);
        CHECK(next.at("logits").raw() != logits.raw());
        CHECK(logits.shape() == Dims{5, f.spec.vocab});
    }

    TEST_CASE("bad inputs are refused by name and leave the state unchanged") {
        const TinyFixture f;
        const auto s = f.session();
        const HostTensor reference = run_logits(*f.session(), f.spec, 0, 4);
        run_logits(*s, f.spec, 0, 2);
        TensorMap out;
        TensorMap missing = dev::tiny_inputs(f.spec, 2, 4);
        missing.erase("pos");
        CHECK_THROWS_WITH_AS(s->run(missing, out), doctest::Contains("'pos' is missing"), Error);
        TensorMap unknown = dev::tiny_inputs(f.spec, 2, 4);
        unknown["extra"] = test::arange_i32(0, 2);
        CHECK_THROWS_WITH_AS(s->run(unknown, out), doctest::Contains("'extra' is not an input"), Error);
        TensorMap wrong_dtype = dev::tiny_inputs(f.spec, 2, 4);
        wrong_dtype["ids"] = HostTensor(DType::I64, {2});
        CHECK_THROWS_WITH_AS(s->run(wrong_dtype, out), doctest::Contains("'ids' has dtype i64"), Error);
        TensorMap wrong_rank = dev::tiny_inputs(f.spec, 2, 4);
        wrong_rank["ids"] = HostTensor(DType::I32, {2, 1});
        CHECK_THROWS_WITH_AS(s->run(wrong_rank, out), doctest::Contains("'ids' has shape [2, 1]"), Error);
        TensorMap mixed_t = dev::tiny_inputs(f.spec, 2, 4);
        mixed_t["pos"] = test::arange_i32(2, 5);
        CHECK_THROWS_WITH_AS(s->run(mixed_t, out), doctest::Contains("T = 3"), Error);
        CHECK(s->num_cached() == 2);
        const HostTensor rest = run_logits(*s, f.spec, 2, 4);
        CHECK(dev::close("validation", rest, to_f64(test::rows(reference, 2, 4)), kStep));
    }

    TEST_CASE("max_context bounds the cache") {
        const TinyFixture f;
        const auto s = f.session(dev::options(8));
        run_logits(*s, f.spec, 0, 6);
        CHECK_THROWS_WITH_AS(run_logits(*s, f.spec, 6, 9), doctest::Contains("max_context 8"), Error);
        CHECK(s->num_cached() == 6);
        run_logits(*s, f.spec, 6, 8);
        CHECK(s->num_cached() == 8);
    }

    TEST_CASE("compile refuses unknown and malformed options by name") {
        const TinyFixture f;
        SessionOptions unknown = dev::options();
        unknown.extra["num_streams"] = "2";
        CHECK_THROWS_WITH_AS(f.session(unknown), doctest::Contains("'num_streams'"), Error);
        SessionOptions bad_impl = dev::options();
        bad_impl.extra["attention"] = "flash";
        CHECK_THROWS_WITH_AS(f.session(bad_impl), doctest::Contains("attention=flash"), Error);
        SessionOptions bf16 = dev::options();
        bf16.precision = DType::BF16;
        CHECK_THROWS_WITH_AS(f.session(bf16), doctest::Contains("precision must be f16 or f32"), Error);
        SessionOptions f16_kv = dev::options();
        f16_kv.kv_dtype = DType::F16;
        CHECK_THROWS_WITH_AS(f.session(f16_kv), doctest::Contains("KV cache dtype must be i8"), Error);
        SessionOptions no_device = dev::options();
        no_device.device = "GPU.7";
        CHECK_THROWS_WITH_AS(f.session(no_device), doctest::Contains("GPU.7"), Error);
    }

    TEST_CASE("devices lists the device under test with a name and memory") {
        const std::vector<DeviceInfo> devices = dev::backend().devices();
        REQUIRE_FALSE(devices.empty());
        const std::string family = dev::device().substr(0, 3);
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
