// GGUF import end to end, hermetically: a tiny qwen35 GGUF written the way llama.cpp's converter writes one
// (tiny_gguf.h: Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, tiled value heads with 2 key heads and 4 value heads, merged
// input projections in one layer, an MTP head, a byte-level BPE vocabulary) imports as the model its blocks decode
// to: the same weights, logits, greedy tokens and drafts as its twin checkpoint on cpu-ref, the same tokenizer as the
// original; and what the importer does not implement is refused by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "backend/cpu_ref/cpu_backend.h"
#include "chat/chat_template.h"
#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "io/safetensors.h"
#include "models/model.h"
#include "quant/quant.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::import_into;
using xs::test::relative_gap;
using xs::test::TempDir;
using xs::test::values;

namespace {

const std::vector<int32_t> kPrompt{3, 17, 42, 99, 5, 250, 7, 8, 64, 128, 266, 271};
const std::vector<int32_t> kSteps{11, 200, 1};

std::vector<std::vector<float>> step_logits(const Model& model) {
    return xs::test::step_logits(model, kPrompt, kSteps);
}

// The MTP head's logits for three pairs (hidden row, next token), from its own program.
std::vector<float> draft_logits(const Model& model) {
    const Speculation* spec = model.speculation();
    REQUIRE(spec != nullptr);
    REQUIRE(spec->chain() != nullptr);
    const auto backend = make_cpu_ref_backend();
    const auto session = backend->compile(spec->draft_graph(), model.weights(), xs::test::f32_session());
    const int64_t hidden = model.info().hidden_size;
    std::vector<float> rows(static_cast<size_t>(3 * hidden));
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = std::sin(0.37f * static_cast<float>(i));
    TensorMap inputs, outputs;
    spec->chain()->prepare_draft(std::vector<int32_t>{5, 77, 266}, rows, 4, 0, inputs);
    session->run(inputs, outputs);
    const HostTensor& l = outputs.at("logits");
    return {l.data<float>(), l.data<float>() + l.numel()};
}

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// A pack tensor's values in the twin's (grouped) order: the import keeps linear attention's value heads tiled.
std::vector<float> grouped_values(const std::string& name, const TensorView& v) {
    const xs::test::TinyQwen m = xs::test::tiny_gguf_model();
    const int64_t hk = m.lin_k_heads, hv = m.lin_v_heads, d = m.lin_dim;
    const auto back = [&](int axis, int64_t first, int64_t unit) {
        return xs::test::retile(values(v), v.shape, axis, first, unit, hk, hv, false);
    };
    if (ends_with(name, "in_proj_qkv.weight") || ends_with(name, "conv1d.weight")) return back(0, 2 * hk * d, d);
    if (ends_with(name, "in_proj_z.weight")) return back(0, 0, d);
    if (ends_with(name, "in_proj_a.weight") || ends_with(name, "in_proj_b.weight") || ends_with(name, "A_log") ||
        ends_with(name, "dt_bias")) {
        return back(0, 0, 1);
    }
    if (ends_with(name, "out_proj.weight")) return back(1, 0, d);
    return values(v);
}

// The tiny GGUF with `change` applied, imported into a fresh directory: the error it gives, or "" when it imports.
template <typename Change>
std::string refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(dir, [](const fs::path& src) { return xs::test::build_tiny_gguf(src, 11); }, change);
}

}  // namespace

TEST_SUITE("gguf import") {
    TEST_CASE("a tiny qwen35 GGUF imports as the model its blocks decode to") {
        const TempDir work("xs_gguf_import_test");
        xs::test::build_tiny_gguf(work.path, 11).write(work.path / "tiny.gguf");
        const gguf_import::ImportReport report = import_into(work.path / "tiny.gguf", work.path / "pack");
        CHECK(report.architecture == "qwen35");
        CHECK(report.model_name == "Tiny Qwen3.5");
        std::vector<std::string> formats;
        int64_t tensors = 0, weights = 0;
        for (const auto& f : report.formats) {
            formats.push_back(f.format);
            tensors += f.tensors;
            weights += f.weights;
        }
        std::sort(formats.begin(), formats.end());
        CHECK(formats == std::vector<std::string>{"BF16", "F16", "F32", "Q4_0", "Q8_0"});
        CHECK(tensors == report.tensors);
        CHECK(weights == report.weights);
        CHECK(report.output_bytes > fs::file_size(work.path / "pack" / "model.safetensors"));

        // The manifest names each block format tensor with its logical shape; float tensors are plain safetensors.
        const nlohmann::json manifest = nlohmann::json::parse(std::ifstream(work.path / "pack" / kPackManifest));
        CHECK(manifest["pack_version"] == kPackVersion);
        CHECK(manifest["source"]["general.architecture"] == "qwen35");
        CHECK(manifest["source"]["name"] == "tiny.gguf");
        CHECK(manifest["tensors"]["model.layers.0.linear_attn.in_proj_b.weight"] ==
              nlohmann::json{{"format", "Q8_0"}, {"shape", {4, 128}}});
        CHECK_FALSE(manifest["tensors"].contains("model.norm.weight"));
        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "pack" / "config.json"));
        CHECK(config["linear_value_head_order"] == "tiled");
        CHECK(config["rms_norm_eps"].get<double>() == 1e-6);
        CHECK(config["rope_parameters"]["partial_rotary_factor"].get<double>() == 0.25);

        // Every weight of the twin, with the same values (value heads in the file's order).
        const auto pack = load_model(work.path / "pack", LoadOptions{.speculation = true, .draft_shortlist = 0});
        const auto twin = load_model(work.path / "twin", LoadOptions{.speculation = true, .draft_shortlist = 0});
        CHECK(pack->weights().names() == twin->weights().names());
        for (const std::string& name : twin->weights().names()) {
            CAPTURE(name);
            const TensorView& a = pack->weights().get(name);
            REQUIRE(a.shape == twin->weights().get(name).shape);
            CHECK(grouped_values(name, a) == values(twin->weights().get(name)));
        }
        CHECK(pack->weights().get("model.embed_tokens.weight").dtype == DType::Block);
        CHECK(pack->weights().get("model.layers.1.mlp.down_proj.weight").dtype == DType::BF16);
        CHECK(pack->weights().get("model.layers.3.self_attn.o_proj.weight").dtype == DType::F16);
        CHECK(pack->info().stop_token_ids == twin->info().stop_token_ids);

        const auto reference = step_logits(*twin);
        const double gap = relative_gap(step_logits(*pack), reference);
        MESSAGE("imported pack against its twin: largest logit difference " << gap << " of the largest logit");
        CHECK(gap < 1e-6);
        CHECK(xs::test::greedy(*pack, kPrompt, 8) == xs::test::greedy(*twin, kPrompt, 8));
        CHECK(draft_logits(*pack) == draft_logits(*twin));

        // Planted: the same pack read in Hugging Face's head order is another model.
        fs::create_directories(work.path / "grouped");
        for (const auto& e : fs::directory_iterator(work.path / "pack")) {
            fs::copy_file(e.path(), work.path / "grouped" / e.path().filename());
        }
        nlohmann::json grouped = config;
        grouped.erase("linear_value_head_order");
        std::ofstream(work.path / "grouped" / "config.json") << grouped.dump(1);
        CHECK(relative_gap(step_logits(*load_model(work.path / "grouped")), reference) > 1e-2);
    }

    TEST_CASE("the tokenizer written from the GGUF encodes and decodes like the original") {
        const TempDir work("xs_gguf_import_tokenizer_test");
        xs::test::build_tiny_gguf(work.path, 3).write(work.path / "tiny.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        const auto original = tok::Tokenizer::load(work.path / "hf");
        const auto imported = tok::Tokenizer::load(work.path / "pack");
        CHECK(imported->vocab_size() == original->vocab_size());  // the [PAD<id>] placeholders are left out
        CHECK(imported->vocab_size() == 274);
        CHECK(imported->eos_id() == original->eos_id());
        CHECK(imported->bos_id() == original->bos_id());
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            CAPTURE(id);
            CHECK(imported->token_text(id) == original->token_text(id));
            CHECK(imported->is_special(id) == original->is_special(id));
        }
        CHECK_FALSE(imported->is_special(*imported->token_id("<think>")));
        CHECK_FALSE(imported->is_special(*imported->token_id("<|fim_prefix|>")));  // typed control in the file
        CHECK(imported->is_special(*imported->token_id("<|im_end|>")));
        for (const std::string text :
             {"the theater in the evening", "  \n\n  Ünïcödé — 中文 😀 x'll 123",
              "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nok</think>",
              "<tool_call>{\"a\": 1}</tool_call><|fim_prefix|>x<|endoftext|>", "é and é", ""}) {
            CAPTURE(text);
            for (const bool special : {true, false}) {
                const std::vector<int32_t> ids = original->encode(text, special);
                CHECK(imported->encode(text, special) == ids);
                CHECK(imported->decode(ids) == original->decode(ids));
                CHECK(imported->decode(ids, true) == original->decode(ids, true));
            }
        }
    }

    TEST_CASE("the chat template and the stop token come from the file") {
        const TempDir work("xs_gguf_import_template_test");
        xs::test::build_tiny_gguf(work.path, 5).write(work.path / "tiny.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        std::ifstream in(work.path / "pack" / "chat_template.jinja");
        CHECK(std::string(std::istreambuf_iterator<char>(in), {}) == xs::test::kTinyChatTemplate);
        const auto chat = chat::ChatTemplate::load(work.path / "pack");
        const nlohmann::json messages = nlohmann::json::array({{{"role", "user"}, {"content", "Hi"}}});
        CHECK(chat->render(messages, nullptr, true) == "<|im_start|>user\nHi<|im_end|><|im_start|>assistant\n<think>");
        const nlohmann::json tc = nlohmann::json::parse(std::ifstream(work.path / "pack" / "tokenizer_config.json"));
        CHECK(tc["eos_token"] == "<|im_end|>");
        CHECK(tc["pad_token"] == "<|endoftext|>");
        CHECK(tc["bos_token"].is_null());
        CHECK(tc["added_tokens_decoder"]["268"]["special"] == false);
        CHECK(load_model(work.path / "pack")->info().stop_token_ids == std::vector<int32_t>{267});
        CHECK_FALSE(fs::exists(work.path / "pack" / "generation_config.json"));  // the file names no sampling defaults
    }

    TEST_CASE("sampling defaults in the file become generation_config.json") {
        const TempDir work("xs_gguf_import_sampling_test");
        GgufWriter w = xs::test::build_tiny_gguf(work.path, 5);
        w.f32("general.sampling.temp", 0.6f).i32("general.sampling.top_k", 20).f32("general.sampling.top_p", 0.95f);
        w.write(work.path / "tiny.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        const nlohmann::json g = nlohmann::json::parse(std::ifstream(work.path / "pack" / "generation_config.json"));
        CHECK(g == nlohmann::json{{"do_sample", true}, {"temperature", 0.6}, {"top_k", 20}, {"top_p", 0.95}});
        const SamplingParams p = load_model(work.path / "pack")->info().default_sampling;
        CHECK(p.temperature == doctest::Approx(0.6));
        CHECK(p.top_k == 20);
    }

    TEST_CASE("refusals name what is wrong") {
        const TempDir work("xs_gguf_import_refusal_test");
        CHECK(refusal(work.path, [](GgufWriter&) {}).empty());
        const auto architecture = [&](const char* name) {
            return refusal(work.path, [&](GgufWriter& w) {
                w.drop_value("general.architecture").string("general.architecture", name);
            });
        };
        CHECK(architecture("llama").find(
                  "GGUF architecture 'llama' is not supported (supported: qwen35, gemma4, gemma4-assistant)") !=
              std::string::npos);
        CHECK(architecture("qwen35moe").find("'qwen35moe' (Qwen3.5 mixture of experts) is not supported yet") !=
              std::string::npos);
        CHECK(
            refusal(work.path,
                    [](GgufWriter& w) { w.f32_tensor("blk.0.ssm_extra.weight", {128}, std::vector<float>(128, 0.0f)); })
                .find("tensor 'blk.0.ssm_extra.weight' has no counterpart in block 0 (linear attention)") !=
            std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.f32_tensor("rope_freqs.weight", {16}, std::vector<float>(16));
              }).find("tensor 'rope_freqs.weight' has no counterpart in the model") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) { w.drop_tensor("blk.2.ssm_dt.bias"); })
                  .find("no tensor for 'model.layers.2.linear_attn.dt_bias' (block 2, linear attention)") !=
              std::string::npos);
        CHECK(refusal(work.path,
                      [](GgufWriter& w) { w.f32_tensor("blk.0.ssm_beta.weight", {128, 4}, std::vector<float>(512)); })
                  .find("'model.layers.0.linear_attn.in_proj_b.weight' comes from more than one tensor") !=
              std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.u32("qwen35.expert_count", 8);
              }).find("metadata 'qwen35.expert_count' is not read by this importer") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.string("qwen35.rope.scaling.type", "yarn");
              }).find("rope.scaling.type 'yarn' is not supported") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("qwen35.ssm.time_step_rank").u32("qwen35.ssm.time_step_rank", 8);
              }).find("has shape") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("tokenizer.ggml.model").string("tokenizer.ggml.model", "llama");
              }).find("tokenizer.ggml.model 'llama' is not supported (supported: gpt2, gemma4)") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("tokenizer.ggml.pre").string("tokenizer.ggml.pre", "llama-bpe");
              }).find("tokenizer.ggml.pre 'llama-bpe' is not supported (supported: qwen35)") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) { w.boolean("tokenizer.ggml.add_space_prefix", true); })
                  .find("tokenizer.ggml.add_space_prefix is true, which disagrees with the tokenizer 'qwen35'") !=
              std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.boolean("tokenizer.ggml.normalizer.nfc", false);
              }).find("tokenizer.ggml.normalizer.nfc is false") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("tokenizer.chat_template");
              }).find("has no tokenizer.chat_template") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("tokenizer.ggml.eos_token_id");
              }).find("names no tokenizer.ggml.eos_token_id") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.boolean("prism.hadamard.gdn_v_grouped", true);
              }).find("'prism.hadamard.gdn_v_grouped': weights with transforms folded into them") != std::string::npos);
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_value("general.type").string("general.type", "adapter");
              }).find("is general.type 'adapter', not a model") != std::string::npos);
        // A norm weight in a block format: undoing its 1 + w would need its values decoded.
        CHECK(refusal(work.path,
                      [](GgufWriter& w) {
                          std::vector<float> ones(128, 1.0f), decoded(128);
                          w.drop_tensor("output_norm.weight")
                              .tensor("output_norm.weight", {128}, xs::test::kGgmlQ8_0,
                                      xs::test::encode_q8_0(ones.data(), 128, decoded.data()));
                      })
                  .find("'output_norm.weight' is stored as Q8_0, but w = (1 + w) - 1 of a norm weight needs a float") !=
              std::string::npos);
        CHECK(
            refusal(work.path, [](GgufWriter& w) {
                w.drop_tensor("blk.0.ssm_a").f32_tensor("blk.0.ssm_a", {4}, {-1.0f, 0.5f, -2.0f, -3.0f});
            }).find("'blk.0.ssm_a' holds 0.5 at 1, for which A_log = log(-ssm_a) is not finite") != std::string::npos);

        fs::create_directories(work.path / "full");
        std::ofstream(work.path / "full" / "x") << "x";
        xs::test::build_tiny_gguf(work.path / "src2", 11).write(work.path / "ok.gguf");
        CHECK_THROWS_WITH_AS(import_into(work.path / "ok.gguf", work.path / "full"),
                             doctest::Contains("exists and is not empty"), Error);
    }
}
