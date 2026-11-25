// A tiny Inkling 2 GGUF in either converter's layout (tiny_inkling2_gguf.h): it imports, and the two layouts of one
// model import as the same model, on cpu-ref.
#include <doctest/doctest.h>

#include <fstream>

#include <nlohmann/json.hpp>

#include "convert/import_test_util.h"
#include "convert/tiny_inkling2_gguf.h"
#include "model/tiny_checkpoints.h"

namespace fs = std::filesystem;
using namespace xs;

TEST_CASE("tiny inkling2 GGUF: Unsloth's layout and mainline llama.cpp's import as the same model") {
    const test::TempDir work("xs_gguf_inkling2_tiny");
    fs::create_directories(work.path);
    test::tiny_inkling2_gguf(false).write(work.path / "unsloth.gguf");
    test::tiny_inkling2_gguf(true).write(work.path / "mainline.gguf");
    CHECK(test::import_into(work.path / "unsloth.gguf", work.path / "u").architecture == "inkling2");
    CHECK(test::import_into(work.path / "mainline.gguf", work.path / "m").architecture == "inkling2_loop");

    const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "m" / "config.json"));
    CHECK(config["model_type"] == "inkling2_loop");
    CHECK(config["text_config"]["gate_up_rows"] == "halves");
    CHECK(config["text_config"]["local_layer_ids"] == nlohmann::json::array({0, 1, 2, 3, 4, 6, 7}));
    CHECK(config["text_config"]["loop_config"]["num_passes"] == 2);
    CHECK(config == nlohmann::json::parse(std::ifstream(work.path / "u" / "config.json")));

    const auto unsloth = load_model(work.path / "u");
    const auto mainline = load_model(work.path / "m");
    // Mainline's layout keeps each looped layer's message queries; Unsloth's the loop's two tensors.
    CHECK(mainline->weights().contains("model.llm.layers.2.message_queries"));
    CHECK(unsloth->weights().contains("model.llm.middle_cycle_cross_pass_queries"));
    const std::vector<int32_t> prompt = {5, 77, 266, 3, 40, 9, 120};
    const auto reference = test::step_logits(*unsloth, prompt, {11, 12});
    const double gap = test::relative_gap(test::step_logits(*mainline, prompt, {11, 12}), reference);
    MESSAGE("the two layouts' logits: largest difference " << gap << " of the largest logit");
    CHECK(gap < 1e-6);
    CHECK(test::greedy(*mainline, prompt, 6) == test::greedy(*unsloth, prompt, 6));

    // Planted: a looped layer's message queries in another order are another model.
    test::GgufWriter swapped = test::tiny_inkling2_gguf(true);
    swapped.swap_tensors("blk.2.loop_msg_query.weight", "blk.3.loop_msg_query.weight");
    swapped.write(work.path / "swapped.gguf");
    test::import_into(work.path / "swapped.gguf", work.path / "s");
    CHECK(test::relative_gap(test::step_logits(*load_model(work.path / "s"), prompt, {11, 12}), reference) > 1e-3);
}

TEST_CASE("tiny inkling2 GGUF: the loop's message gates as Q8_0 blocks, as AtomicChat's files hold them, run") {
    const test::TempDir work("xs_gguf_inkling2_q8_gates");
    fs::create_directories(work.path);
    test::TinyInkling2 m;
    m.gate_rank = 32;  // a row of whole blocks
    m.gates = test::TinyInkling2::Gates::Q8_0Values;
    test::tiny_inkling2_gguf(true, m).write(work.path / "f32.gguf");
    m.gates = test::TinyInkling2::Gates::Q8_0;
    test::tiny_inkling2_gguf(true, m).write(work.path / "q8.gguf");
    test::import_into(work.path / "f32.gguf", work.path / "f");
    test::import_into(work.path / "q8.gguf", work.path / "q");
    const std::vector<int32_t> prompt = {5, 77, 266, 3, 40, 9, 120};
    const auto reference = test::step_logits(*load_model(work.path / "f"), prompt, {11, 12});
    const double gap = test::relative_gap(test::step_logits(*load_model(work.path / "q"), prompt, {11, 12}), reference);
    MESSAGE("Q8_0 gates against F32 of their values: largest difference " << gap << " of the largest logit");
    CHECK(gap < 1e-5);
}
