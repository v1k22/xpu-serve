// SenseNova Loop on a tiny checkpoint the test writes (a Qwen3.5 whose middle layers run twice inside two-stream
// hyper-connections): the loop runs (the same weights run once are another model), each run of a layer keeps state of
// its own (a prompt fed in pieces gives the logits of the whole), the GPU agrees with cpu-ref, a pack keeps the
// hyper-connection routers as they are stored, and a block drafter beside it is refused by name. Its parity with the
// checkpoint's own Python code is model_parity.cpp's, on a tiny checkpoint of make_tiny.py --arch sensenova_loop.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "backend/device_test_support.h"
#include "convert/convert.h"
#include "models/model.h"
#include "step_logits.h"
#include "tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
namespace dev = xs::test::device;
using xs::test::relative_gap;
using xs::test::step_logits;
using xs::test::TempDir;
using xs::test::TinyQwen;

namespace {

// Eight layers in two linear / linear / linear / full cycles; layers 2 to 5 run twice.
TinyQwen tiny_loop() {
    TinyQwen m;
    m.layers = {"linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"};
    m.loop_start = 2;
    m.loop_end = 6;
    m.loop_count = 2;
    return m;
}

const std::vector<int32_t> kPrompt{3, 17, 42, 99, 5, 250, 7, 8, 64, 128, 11, 200};
const std::vector<int32_t> kSteps{9, 31, 77};

SessionOptions f32() {
    SessionOptions o;
    o.precision = DType::F32;
    return o;
}

}  // namespace

TEST_SUITE("sensenova loop") {
    TEST_CASE("the loop runs, and each run of a layer keeps state of its own") {
        const TempDir work("xs_sensenova_loop");
        write_tiny_qwen(work.path / "loop", tiny_loop(), 7);
        const auto model = load_model(work.path / "loop");
        CHECK(model->info().arch == "sensenova_loop");
        // 12 runs: layers 0, 1, then 2 to 5 twice, then 6, 7; full attention in layer 3 (twice) and 7
        CHECK(model->graph().kv_slots().size() == 3);
        CHECK(model->graph().conv_slots().size() == 9);
        CHECK(model->graph().recurrent_slots().size() == 9);
        const auto cpu = make_cpu_ref_backend();
        const auto whole = step_logits(*cpu, *model, f32(), kPrompt, kSteps);

        // the prompt in pieces: its first 5 tokens, then one at a time
        std::vector<int32_t> rest(kPrompt.begin() + 5, kPrompt.end());
        rest.insert(rest.end(), kSteps.begin(), kSteps.end());
        const auto pieces = step_logits(*cpu, *model, f32(), {kPrompt.begin(), kPrompt.begin() + 5}, rest);
        const auto first = pieces.end() - static_cast<std::ptrdiff_t>(whole.size());
        const std::vector<std::vector<float>> tail(first, pieces.end());
        CHECK(relative_gap(tail, whole) < 1e-5);

        // planted: the same weights with each layer run once
        TinyQwen once = tiny_loop();
        once.loop_count = 1;
        write_tiny_qwen(work.path / "once", once, 7);
        const auto other = load_model(work.path / "once");
        CHECK(relative_gap(step_logits(*cpu, *other, f32(), kPrompt, kSteps), whole) > 0.05);

        if (dev::enabled()) {
            const auto gpu = step_logits(dev::backend(), *model, dev::options(), kPrompt, kSteps);
            const double gap = relative_gap(gpu, whole);
            MESSAGE("GPU vs cpu-ref: largest logit gap ", gap, " of the largest logit");
            CHECK(gap < (dev::is_f16() ? 2e-2 : 1e-4));
        }
    }

    TEST_CASE("a pack quantizes the layers' matmuls and keeps the hyper-connection routers") {
        const TempDir work("xs_sensenova_loop_pack");
        write_tiny_qwen(work.path / "loop", tiny_loop(), 7);
        ConvertOptions o;
        o.source = work.path / "loop";
        o.output = work.path / "q8";
        o.format = DType::Q8;
        o.group = 32;
        o.min_elements = 0;  // the tiny matrices count; the routers stay all the same
        convert_model(o);
        const auto pack = load_model(o.output);
        const WeightStore& w = pack->weights();
        CHECK(w.get("model.layers.2.mlp.up_proj.weight").dtype == DType::Q8);
        for (const char* name : {"model.layers.2.mhc_attn.weight", "model.layers.5.mhc_ffn.weight"}) {
            CAPTURE(name);
            CHECK(w.get(name).dtype == DType::BF16);
        }
        const auto cpu = make_cpu_ref_backend();
        const auto original = load_model(o.source);
        const double gap = relative_gap(step_logits(*cpu, *pack, f32(), kPrompt, kSteps),
                                        step_logits(*cpu, *original, f32(), kPrompt, kSteps));
        MESSAGE("q8 pack vs the original: largest logit gap ", gap, " of the largest logit");
        CHECK(gap < 0.1);  // random weights: a sanity bound; quality is measured on the real model
    }

    TEST_CASE("a block drafter beside a looped model is refused by name") {
        const TempDir work("xs_sensenova_loop_drafter");
        const TinyQwen target = tiny_loop();
        write_tiny_qwen(work.path / "loop", target, 7);
        xs::test::write_tiny_drafter(work.path / "drafter", xs::test::TinyDrafter{}, target, 8);
        CHECK_THROWS_WITH_AS(
            load_model(work.path / "loop", {.vision = false, .speculation = true, .drafter = work.path / "drafter"}),
            doctest::Contains("loop_settings"), xs::Error);
    }
}
