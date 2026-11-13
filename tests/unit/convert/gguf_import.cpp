// GGUF import end to end, hermetically: a tiny qwen35 GGUF written the way llama.cpp's converter writes one
// (tiny_gguf.h: Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, tiled value heads with 2 key heads and 4 value heads, merged
// input projections in one layer, an MTP head, a byte-level BPE vocabulary) imports as the model its blocks decode
// to: the same weights, logits, greedy tokens and drafts as its twin checkpoint on cpu-ref, the same tokenizer as the
// original; the same model as a file without the head and the head's own file (--mtp) imports as the same pack; and
// what the importer does not implement is refused by name.
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
using xs::test::has;
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

// The tiny GGUF with `change` applied, imported into a fresh directory: the error it gives, or "" when it imports.
template <typename Change>
std::string refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(dir, [](const fs::path& src) { return xs::test::build_tiny_gguf(src, 11); }, change);
}

// The tiny model file and its head's file with `change(model, mtp)` applied, imported with --mtp into a fresh
// directory: the error it gives, or "" when it imports.
template <typename Change>
std::string mtp_refusal(const fs::path& dir, xs::test::TinyMtpFiles files, Change change) {
    change(files.model, files.mtp);
    files.model.write(dir / "model.gguf");
    files.mtp.write(dir / "mtp.gguf");
    fs::remove_all(dir / "variant");
    try {
        import_into(dir / "model.gguf", dir / "variant", dir / "mtp.gguf");
    } catch (const Error& e) {
        return e.what();
    }
    return "";
}

std::string file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
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
            CHECK(xs::test::grouped_values(name, a) == values(twin->weights().get(name)));
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

    TEST_CASE("a model file without its MTP head and the head's own file import as the file that holds both") {
        const TempDir work("xs_gguf_import_mtp_test");
        const GgufWriter whole = xs::test::build_tiny_gguf(work.path / "src", 11);
        whole.write(work.path / "whole.gguf");
        xs::test::TinyMtpFiles split = xs::test::split_tiny_gguf(whole);
        // The pack's tokenizer, chat template and sampling defaults are the model file's, whatever the head's file has.
        split.mtp.string("tokenizer.chat_template", "{{ 'the draft model file' }}").f32("general.sampling.temp", 0.1f);
        split.model.write(work.path / "model.gguf");
        split.mtp.write(work.path / "mtp.gguf");
        const gguf_import::ImportReport one = import_into(work.path / "whole.gguf", work.path / "one");
        const gguf_import::ImportReport two =
            import_into(work.path / "model.gguf", work.path / "two", work.path / "mtp.gguf");

        // The same files, byte for byte, but the manifest, which names its sources.
        std::vector<std::string> names, other_names;
        for (const auto& e : fs::directory_iterator(work.path / "one"))
            names.push_back(e.path().filename().string());
        for (const auto& e : fs::directory_iterator(work.path / "two"))
            other_names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        std::sort(other_names.begin(), other_names.end());
        CHECK(other_names == names);
        CHECK(std::find(names.begin(), names.end(), "generation_config.json") == names.end());
        for (const std::string& name : names) {
            CAPTURE(name);
            if (name == kPackManifest) continue;
            CHECK(file_bytes(work.path / "two" / name) == file_bytes(work.path / "one" / name));
        }
        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "two" / "config.json"));
        CHECK(config["mtp_num_hidden_layers"] == 1);
        CHECK(config["tie_word_embeddings"] == true);  // the head file's own output head is not the model's
        const nlohmann::json m1 = nlohmann::json::parse(std::ifstream(work.path / "one" / kPackManifest));
        const nlohmann::json m2 = nlohmann::json::parse(std::ifstream(work.path / "two" / kPackManifest));
        CHECK(m2["tensors"] == m1["tensors"]);
        CHECK(m2["source"]["name"] == "model.gguf");
        CHECK(m2["source"]["mtp"] == "mtp.gguf");
        CHECK_FALSE(m1["source"].contains("mtp"));

        // The report counts the tensors the import read: the head file's copies are not among them.
        CHECK(two.tensors == one.tensors);
        CHECK(two.weights == one.weights);
        REQUIRE(two.formats.size() == one.formats.size());
        for (size_t i = 0; i < one.formats.size(); ++i) {
            CHECK(two.formats[i].format == one.formats[i].format);
            CHECK(two.formats[i].tensors == one.formats[i].tensors);
            CHECK(two.formats[i].bytes == one.formats[i].bytes);
        }

        // The same drafts on cpu-ref.
        const LoadOptions spec{.speculation = true, .draft_shortlist = 0};
        const xs::test::Drafts reference = xs::test::drafts(*load_model(work.path / "one", spec), kPrompt);
        const xs::test::Drafts ours = xs::test::drafts(*load_model(work.path / "two", spec), kPrompt);
        MESSAGE("drafts: " << ours.tokens[0] << " " << ours.tokens[1] << " " << ours.tokens[2]);
        CHECK(ours.logits == reference.logits);
        CHECK(ours.tokens == reference.tokens);

        // Planted: the head of another model of the same shapes imports, and is another drafter.
        const GgufWriter other = xs::test::build_tiny_gguf(work.path / "src2", 12);
        xs::test::split_tiny_gguf(other).mtp.write(work.path / "other.gguf");
        import_into(work.path / "model.gguf", work.path / "other", work.path / "other.gguf");
        CHECK(file_bytes(work.path / "other" / "model.safetensors") !=
              file_bytes(work.path / "two" / "model.safetensors"));
        CHECK(xs::test::drafts(*load_model(work.path / "other", spec), kPrompt).logits != reference.logits);
    }

    TEST_CASE("--mtp: refusals name what is wrong") {
        const TempDir work("xs_gguf_import_mtp_refusal_test");
        const GgufWriter whole = xs::test::build_tiny_gguf(work.path / "src", 11);
        const xs::test::TinyMtpFiles files = xs::test::split_tiny_gguf(whole);
        const auto refused = [&](auto change) { return mtp_refusal(work.path, files, change); };
        using W = GgufWriter;
        CHECK(refused([](W&, W&) {}).empty());
        // The head file needs no copies of the model's tensors.
        CHECK(refused([](W&, W& mtp) {
                  mtp.drop_tensor("token_embd.weight").drop_tensor("output.weight").drop_tensor("output_norm.weight");
              }).empty());
        // Per-block layer lists: the model's blocks, then the head's.
        CHECK(refused([](W& model, W& mtp) {
                  model.bools("qwen35.attention.recurrent_layers", {true, true, true, false});
                  mtp.bools("qwen35.attention.recurrent_layers", {true, true, true, false, false});
              }).empty());

        CHECK(has(refused([&](W& model, W&) { model = whole; }),
                  "qwen35: --mtp: the model file already has an MTP head (nextn_predict_layers 1)"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_value("general.architecture").string("general.architecture", "gemma4");
                  }),
                  "qwen35: the MTP file is general.architecture 'gemma4', not qwen35"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_value("qwen35.nextn_predict_layers").u32("qwen35.nextn_predict_layers", 2);
                  }),
                  "the MTP file has nextn_predict_layers 2; an MTP file holds one MTP layer"));
        CHECK(has(refused([](W&, W& mtp) { mtp.drop_value("qwen35.block_count").u32("qwen35.block_count", 6); }),
                  "the MTP file has block_count 6, not the model file's 4 blocks and the head"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_value("qwen35.embedding_length").u32("qwen35.embedding_length", 256);
                  }),
                  "the MTP file's qwen35.embedding_length (u32 256) differs from the model file's (u32 128): it is "
                  "the head of another model"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_value("qwen35.attention.layer_norm_rms_epsilon")
                          .f32("qwen35.attention.layer_norm_rms_epsilon", 1e-5f);
                  }),
                  "qwen35.attention.layer_norm_rms_epsilon (f32 1e-05) differs from the model file's (f32 1e-06)"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_value("qwen35.rope.dimension_sections")
                          .i32s("qwen35.rope.dimension_sections", {11, 11, 10, 1});
                  }),
                  "qwen35.rope.dimension_sections (i32 [11, 11, 10, 1]) differs from the model file's (i32 [11, 11, "
                  "10, 0])"));
        CHECK(has(refused([](W&, W& mtp) { mtp.drop_value("qwen35.full_attention_interval"); }),
                  "qwen35.full_attention_interval (absent) differs from the model file's (u32 4)"));
        CHECK(has(refused([](W&, W& mtp) { mtp.u32("qwen35.vocab_size", 288); }),
                  "qwen35.vocab_size (u32 288) differs from the model file's (absent)"));
        CHECK(has(refused([](W& model, W& mtp) {
                      model.bools("qwen35.attention.recurrent_layers", {true, true, true, false});
                      mtp.bools("qwen35.attention.recurrent_layers", {true, true, false, true, false});
                  }),
                  "qwen35.attention.recurrent_layers (an array of 5 bool) differs"));
        CHECK(has(refused([](W&, W& mtp) { mtp.boolean("prism.hadamard.gdn_v_grouped", true); }),
                  "weights with transforms folded into them"));

        CHECK(has(refused([](W&, W& mtp) { mtp.f32_tensor("blk.0.attn_norm.weight", {128}, std::vector<float>(128)); }),
                  "the MTP file's tensor 'blk.0.attn_norm.weight' is neither in the head's block 4 nor one of the "
                  "model's token_embd, output and output_norm"));
        CHECK(has(refused([](W&, W& mtp) { mtp.f32_tensor("rope_freqs.weight", {16}, std::vector<float>(16)); }),
                  "the MTP file's tensor 'rope_freqs.weight' is neither in the head's block 4"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_tensor("output.weight")
                          .f32_tensor("output.weight", {128, 256}, std::vector<float>(32768));
                  }),
                  "the MTP file's 'output.weight' has shape [256, 128]; the model's is [288, 128]"));
        CHECK(has(refused([](W&, W& mtp) { mtp.drop_tensor("blk.4.nextn.eh_proj.weight"); }),
                  "qwen35: the file has no tensor for 'mtp.fc.weight' (block 4, an MTP layer)"));
        CHECK(has(refused([](W&, W& mtp) {
                      mtp.drop_tensor("blk.4.nextn.enorm.weight")
                          .f32_tensor("blk.4.nextn.enorm.weight", {64}, std::vector<float>(64, 1.0f));
                  }),
                  "tensor 'blk.4.nextn.enorm.weight' has shape [64]; the metadata gives [128]"));
        CHECK(has(refused([](W& model, W&) {
                      model.f32_tensor("blk.4.attn_norm.weight", {128}, std::vector<float>(128, 1.0f));
                  }),
                  "qwen35: tensor 'blk.4.attn_norm.weight' is not in one of the model file's 4 blocks"));
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
        // Leading dims of 1 do not count: [n] written as [1, n] is the same tensor.
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_tensor("output_norm.weight").f32_tensor("output_norm.weight", {128, 1}, std::vector<float>(128));
              }).empty());
        const auto architecture = [&](const char* name) {
            return refusal(work.path, [&](GgufWriter& w) {
                w.drop_value("general.architecture").string("general.architecture", name);
            });
        };
        CHECK(architecture("llama").find("GGUF architecture 'llama' is not supported (supported: qwen35, gemma4, "
                                         "gemma4-assistant, clef, modern-bert)") != std::string::npos);
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
              }).find("tokenizer.ggml.pre 'llama-bpe' is not supported (supported: qwen35, modern-bert, mmbert)") !=
              std::string::npos);
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
