// GGUF import of Gemma 4, hermetically: a tiny gemma4 GGUF written the way llama.cpp's converter writes one
// (tiny_gemma4_gguf.h: per-layer embeddings, KV-shared layers with a double-wide MLP, sliding and full-attention layers
// of their own head sizes and KV heads, q / k / v fused in one layer, rope_freqs, Q8_0 / Q4_0 / F16 / BF16 / F32
// tensors, a Gemma-style BPE vocabulary with byte fallback) imports as the model its blocks decode to: the same
// weights, logits and greedy tokens as its twin checkpoint on cpu-ref, the same tokenizer as the original; a tiny
// gemma4-assistant GGUF imports as the drafter its blocks decode to, the same drafts over the imported target; and what
// the importer does not implement is refused by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "chat/chat_template.h"
#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gemma4_gguf.h"
#include "models/model.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::Drafts;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::import_into;
using xs::test::relative_gap;
using xs::test::TempDir;
using xs::test::values;

namespace {

// Longer than the tiny model's sliding window (8), so later steps attend to a window, not the whole prompt.
const std::vector<int32_t> kPrompt{2, 5, 286, 17, 42, 99, 287, 250, 7, 8, 64, 128, 290, 271};
const std::vector<int32_t> kSteps{11, 200, 1, 293};

std::vector<std::vector<float>> step_logits(const Model& model) {
    return xs::test::step_logits(model, kPrompt, kSteps);
}

template <typename Change>
std::string refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(dir, [](const fs::path& src) { return xs::test::build_tiny_gemma4_gguf(src, 7); }, change);
}

template <typename Change>
std::string assistant_refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(
        dir, [](const fs::path& src) { return xs::test::build_tiny_gemma4_assistant_gguf(src, 9); }, change);
}

// The drafter after a prefill of kPrompt: its logits for (the first greedy token, the prompt's last hidden row), and
// the three drafts it chains from them.
Drafts drafts(const Model& model) {
    return xs::test::drafts(model, kPrompt);
}

}  // namespace

TEST_SUITE("gguf import gemma4") {
    TEST_CASE("a tiny gemma4 GGUF imports as the model its blocks decode to") {
        const TempDir work("xs_gguf_gemma4_import_test");
        xs::test::build_tiny_gemma4_gguf(work.path, 7).write(work.path / "tiny.gguf");
        const gguf_import::ImportReport report = import_into(work.path / "tiny.gguf", work.path / "pack");
        CHECK(report.architecture == "gemma4");
        CHECK(report.model_name == "Tiny Gemma 4");
        std::vector<std::string> formats;
        for (const auto& f : report.formats)
            formats.push_back(f.format);
        std::sort(formats.begin(), formats.end());
        CHECK(formats == std::vector<std::string>{"BF16", "F16", "F32", "Q4_0", "Q8_0"});

        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "pack" / "config.json"));
        const nlohmann::json original = nlohmann::json::parse(std::ifstream(work.path / "hf" / "config.json"));
        for (const char* key : {"model_type",
                                "dtype",
                                "vocab_size",
                                "vocab_size_per_layer_input",
                                "hidden_size",
                                "intermediate_size",
                                "use_double_wide_mlp",
                                "num_hidden_layers",
                                "layer_types",
                                "num_attention_heads",
                                "num_key_value_heads",
                                "num_global_key_value_heads",
                                "head_dim",
                                "global_head_dim",
                                "sliding_window",
                                "num_kv_shared_layers",
                                "hidden_size_per_layer_input",
                                "rms_norm_eps",
                                "final_logit_softcapping",
                                "max_position_embeddings",
                                "rope_parameters",
                                "tie_word_embeddings",
                                "hidden_activation",
                                "bos_token_id",
                                "pad_token_id"}) {
            CAPTURE(key);
            CHECK(config[key] == original[key]);
        }
        const nlohmann::json manifest = nlohmann::json::parse(std::ifstream(work.path / "pack" / kPackManifest));
        CHECK(manifest["source"]["general.architecture"] == "gemma4");
        CHECK(manifest["tensors"]["model.layers.0.self_attn.k_proj.weight"] ==
              nlohmann::json{{"format", "Q8_0"}, {"shape", {64, 128}}});  // cut from attn_qkv
        CHECK_FALSE(manifest["tensors"].contains("rope_freqs.weight"));

        // Every weight of the twin, with the same values; rope_freqs is the config's, not a weight.
        const auto pack = load_model(work.path / "pack");
        const auto twin = load_model(work.path / "twin");
        CHECK(pack->weights().names() == twin->weights().names());
        for (const std::string& name : twin->weights().names()) {
            CAPTURE(name);
            const TensorView& a = pack->weights().get(name);
            REQUIRE(a.shape == twin->weights().get(name).shape);
            CHECK(values(a) == values(twin->weights().get(name)));
        }
        CHECK(pack->weights().get("model.embed_tokens_per_layer.weight").dtype == DType::Block);
        CHECK(pack->weights().get("model.per_layer_model_projection.weight").dtype == DType::BF16);
        CHECK(pack->host_tables() == std::vector<std::string>{"model.embed_tokens_per_layer.weight"});
        CHECK(pack->info().stop_token_ids == twin->info().stop_token_ids);
        CHECK(pack->info().stop_token_ids == std::vector<int32_t>{1, 6, 9});

        // The same weights on the same graph: the same logits, to the bit.
        const auto reference = step_logits(*twin);
        const double gap = relative_gap(step_logits(*pack), reference);
        MESSAGE("imported pack against its twin: largest logit difference " << gap << " of the largest logit");
        CHECK(gap == 0.0);
        CHECK(xs::test::greedy(*pack, kPrompt, 8) == xs::test::greedy(*twin, kPrompt, 8));

        // Planted: the full-attention layers' RoPE read from rope_freqs wrong is another model.
        fs::create_directories(work.path / "rope");
        for (const auto& e : fs::directory_iterator(work.path / "pack"))
            fs::copy_file(e.path(), work.path / "rope" / e.path().filename());
        nlohmann::json rope = config;
        rope["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.5;
        std::ofstream(work.path / "rope" / "config.json") << rope.dump(1);
        CHECK(relative_gap(step_logits(*load_model(work.path / "rope")), reference) > 1e-3);
    }

    TEST_CASE("the gemma4 tokenizer written from the GGUF encodes and decodes like the original") {
        const TempDir work("xs_gguf_gemma4_tokenizer_test");
        xs::test::build_tiny_gemma4_gguf(work.path, 3).write(work.path / "tiny.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        const auto original = tok::Tokenizer::load(work.path / "hf");
        const auto imported = tok::Tokenizer::load(work.path / "pack");
        REQUIRE(imported->vocab_size() == original->vocab_size());
        CHECK(imported->eos_id() == original->eos_id());
        CHECK(imported->bos_id() == original->bos_id());
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            CAPTURE(id);
            CHECK(imported->token_text(id) == original->token_text(id));
            CHECK(imported->is_special(id) == original->is_special(id));
        }
        // Typed normal (<eos>) and user-defined (<|channel>) in the file, special in the original.
        CHECK(imported->is_special(1));
        CHECK(imported->is_special(*imported->token_id("<|channel>")));
        const nlohmann::json written = nlohmann::json::parse(std::ifstream(work.path / "pack" / "tokenizer.json"));
        const nlohmann::json hf = nlohmann::json::parse(std::ifstream(work.path / "hf" / "tokenizer.json"));
        CHECK(written["model"]["vocab"] == hf["model"]["vocab"]);  // the added tokens are pieces of it too
        for (const char* part : {"normalizer", "pre_tokenizer", "post_processor", "decoder"}) {
            CAPTURE(part);
            CHECK(written[part] == hf[part]);
        }
        for (const std::string text :
             {"the theater in the evening", "  \n\n  é and 中文 😀 the\n\n\nend ",
              "<|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\nok<channel|>the",
              "<|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|><|tool_response>", "<eos> <bos> <|\"|> <|\"", "", " "}) {
            CAPTURE(text);
            for (const bool special : {true, false}) {
                const std::vector<int32_t> ids = original->encode(text, special);
                CHECK(imported->encode(text, special) == ids);
                CHECK(imported->decode(ids) == original->decode(ids));
                CHECK(imported->decode(ids, true) == original->decode(ids, true));
            }
        }
        CHECK(imported->encode("<|\"|>", false) == std::vector<int32_t>{11});  // spelled by the merges
    }

    TEST_CASE("gemma4: the chat template, stop tokens and sampling defaults come from the file") {
        const TempDir work("xs_gguf_gemma4_template_test");
        GgufWriter w = xs::test::build_tiny_gemma4_gguf(work.path, 5);
        w.f32("general.sampling.temp", 1.0f).i32("general.sampling.top_k", 64).f32("general.sampling.top_p", 0.95f);
        w.write(work.path / "tiny.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        const auto chat = chat::ChatTemplate::load(work.path / "pack");
        const nlohmann::json messages = nlohmann::json::array({{{"role", "user"}, {"content", "Hi"}}});
        CHECK(chat->render(messages, nullptr, true) == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n");
        const nlohmann::json tc = nlohmann::json::parse(std::ifstream(work.path / "pack" / "tokenizer_config.json"));
        CHECK(tc["bos_token"] == "<bos>");
        CHECK(tc["eos_token"] == "<turn|>");
        CHECK(tc["unk_token"] == "<unk>");
        CHECK(tc["pad_token"] == "<pad>");
        CHECK(tc["add_bos_token"] == true);
        CHECK(tc["added_tokens_decoder"].size() == xs::test::kTinyGemmaAdded.size());
        const auto model = load_model(work.path / "pack");
        CHECK(model->info().stop_token_ids == std::vector<int32_t>{1, 6, 9});  // <eos>, <turn|>, <|tool_response>
        CHECK(model->info().default_sampling.top_k == 64);
        CHECK(model->info().default_sampling.temperature == doctest::Approx(1.0));
        CHECK(model->info().dialect == "gemma4");
        CHECK(model->vision() == nullptr);
    }

    TEST_CASE("a tiny gemma4-assistant GGUF imports as the drafter its blocks decode to") {
        const TempDir work("xs_gguf_gemma4_assistant_test");
        xs::test::build_tiny_gemma4_gguf(work.path, 7).write(work.path / "tiny.gguf");
        xs::test::build_tiny_gemma4_assistant_gguf(work.path, 9).write(work.path / "assistant.gguf");
        import_into(work.path / "tiny.gguf", work.path / "pack");
        const gguf_import::ImportReport report = import_into(work.path / "assistant.gguf", work.path / "drafter");
        CHECK(report.architecture == "gemma4-assistant");
        // A drafter is served with the target's tokenizer and chat template: its pack holds none.
        std::vector<std::string> files;
        for (const auto& e : fs::directory_iterator(work.path / "drafter"))
            files.push_back(e.path().filename().string());
        std::sort(files.begin(), files.end());
        CHECK(files == std::vector<std::string>{"config.json", "model.safetensors", kPackManifest});
        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "drafter" / "config.json"));
        const nlohmann::json original =
            nlohmann::json::parse(std::ifstream(work.path / "hf-assistant" / "config.json"));
        for (const char* key :
             {"model_type", "dtype", "backbone_hidden_size", "use_ordered_embeddings", "tie_word_embeddings"}) {
            CAPTURE(key);
            CHECK(config[key] == original[key]);
        }
        for (const auto& [key, value] : original["text_config"].items()) {
            CAPTURE(key);
            CHECK(config["text_config"][key] == value);
        }

        const auto pack =
            load_model(work.path / "pack", LoadOptions{.speculation = true, .drafter = work.path / "drafter"});
        const auto twin =
            load_model(work.path / "twin", LoadOptions{.speculation = true, .drafter = work.path / "twin-assistant"});
        CHECK(pack->weights().names() == twin->weights().names());
        for (const std::string& name : twin->weights().names()) {
            CAPTURE(name);
            CHECK(values(pack->weights().get(name)) == values(twin->weights().get(name)));
        }
        CHECK(pack->weights().get("assistant.pre_projection.weight").dtype == DType::Block);
        const Drafts ours = drafts(*pack), theirs = drafts(*twin);
        CHECK(ours.logits == theirs.logits);
        CHECK(ours.tokens == theirs.tokens);
        MESSAGE("drafts over the imported target: " << ours.tokens[0] << " " << ours.tokens[1] << " "
                                                    << ours.tokens[2]);

        // Planted: the drafter's full-attention RoPE read wrong is another drafter.
        nlohmann::json rope = config;
        rope["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.5;
        std::ofstream(work.path / "drafter" / "config.json") << rope.dump(1);
        const auto planted =
            load_model(work.path / "pack", LoadOptions{.speculation = true, .drafter = work.path / "drafter"});
        CHECK(drafts(*planted).logits != theirs.logits);
    }

    TEST_CASE("gemma4-assistant: refusals name what is wrong") {
        const TempDir work("xs_gguf_gemma4_assistant_refusal_test");
        CHECK(assistant_refusal(work.path, [](GgufWriter&) {}).empty());
        CHECK(has(assistant_refusal(work.path,
                                    [](GgufWriter& w) {
                                        w.drop_value("gemma4-assistant.attention.shared_kv_layers")
                                            .u32("gemma4-assistant.attention.shared_kv_layers", 1);
                                    }),
                  "gemma4-assistant.attention.shared_kv_layers is 1; an assistant's 2 layers all draft over the "
                  "target's KV"));
        CHECK(has(assistant_refusal(work.path,
                                    [](GgufWriter& w) {
                                        w.drop_value("gemma4-assistant.embedding_length_per_layer_input")
                                            .u32("gemma4-assistant.embedding_length_per_layer_input", 32);
                                    }),
                  "gemma4-assistant: embedding_length_per_layer_input is 32"));
        CHECK(has(assistant_refusal(work.path, [](GgufWriter& w) { w.drop_tensor("nextn.post_projection.weight"); }),
                  "no tensor for 'post_projection.weight' (the model's)"));
        // The centroid head llama.cpp's converter leaves out has no place in the pack.
        CHECK(has(assistant_refusal(work.path,
                                    [](GgufWriter& w) {
                                        w.f32_tensor("masked_embd_centroids.weight", {64, 16},
                                                     std::vector<float>(1024));
                                    }),
                  "tensor 'masked_embd_centroids.weight' has no counterpart in the model"));
        CHECK(has(assistant_refusal(work.path, [](GgufWriter& w) { w.u32("gemma4-assistant.expert_count", 4); }),
                  "mixture-of-experts Gemma 4 (26B-A4B) is not supported yet"));
        xs::test::build_tiny_gemma4_assistant_gguf(work.path / "src", 9).write(work.path / "assistant.gguf");
        CHECK_THROWS_WITH_AS(
            import_into(work.path / "assistant.gguf", work.path / "with-mtp", work.path / "assistant.gguf"),
            doctest::Contains("gemma4-assistant: an MTP head in a file of its own (--mtp) is not supported"), Error);
    }

    TEST_CASE("gemma4: refusals name what is wrong") {
        const TempDir work("xs_gguf_gemma4_refusal_test");
        CHECK(refusal(work.path, [](GgufWriter&) {}).empty());
        // A KV-shared layer's own k and v are never read: the file may leave them out.
        CHECK(refusal(work.path, [](GgufWriter& w) {
                  w.drop_tensor("blk.4.attn_k.weight")
                      .drop_tensor("blk.4.attn_v.weight")
                      .drop_tensor("blk.4.attn_k_norm.weight");
              }).empty());
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.drop_tensor("blk.2.attn_k.weight"); }),
                  "gemma4: the file has no tensor for 'model.layers.2.self_attn.k_proj.weight' (block 2, sliding "
                  "attention)"));
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.drop_tensor("blk.3.attn_v.weight"); }),
                  "gemma4: block 3 has no attn_v.weight: values from the keys (attention_k_eq_v) are not supported"));
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.drop_tensor("blk.5.layer_output_scale.weight"); }),
                  "no tensor for 'model.layers.5.layer_scalar' (block 5, full attention, KV-shared)"));
        CHECK(
            has(refusal(work.path,
                        [](GgufWriter& w) { w.f32_tensor("blk.1.attn_extra.weight", {128}, std::vector<float>(128)); }),
                "tensor 'blk.1.attn_extra.weight' has no counterpart in block 1 (sliding attention)"));
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.u32("gemma4.expert_count", 8); }),
                  "gemma4.expert_count: mixture-of-experts Gemma 4 (26B-A4B) is not supported yet"));
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.u32("gemma4.altup.num_inputs", 4); }),
                  "metadata 'gemma4.altup.num_inputs' is not read by this importer"));
        CHECK(has(refusal(work.path,
                          [](GgufWriter& w) {
                              std::vector<float> f(64, 1e30f);
                              std::fill_n(f.begin(), 16, 1.0f);
                              f[40] = 4.0f;
                              w.drop_tensor("rope_freqs.weight").f32_tensor("rope_freqs.weight", {64}, f);
                          }),
                  "rope_freqs.weight holds 4 at 40; the import reads only the converter's proportional RoPE"));
        CHECK(
            has(refusal(work.path,
                        [](GgufWriter& w) {
                            w.drop_value("gemma4.rope.dimension_count_swa").u32("gemma4.rope.dimension_count_swa", 32);
                        }),
                "rope.dimension_count_swa 32 is not the sliding-attention head size 64"));
        // Layers 3 and 2 swapped in kind: llama.cpp would read the shared layers' KV from the wrong layers.
        CHECK(has(refusal(work.path,
                          [](GgufWriter& w) {
                              w.drop_value("gemma4.attention.sliding_window_pattern")
                                  .bools("gemma4.attention.sliding_window_pattern",
                                         {true, true, false, true, true, false})
                                  .drop_value("gemma4.attention.head_count_kv")
                                  .i32s("gemma4.attention.head_count_kv", {1, 1, 2, 1, 1, 2});
                          }),
                  "KV-shared layer 4 reads layer 2's KV in llama.cpp but layer 3's in Gemma 4"));
        CHECK(has(refusal(work.path,
                          [](GgufWriter& w) {
                              w.drop_value("gemma4.feed_forward_length")
                                  .i32s("gemma4.feed_forward_length", {256, 256, 256, 512, 512, 512});
                          }),
                  "feed_forward_length of layer 3 is 512"));
        CHECK(has(refusal(work.path, [](GgufWriter& w) { w.string("tokenizer.ggml.pre", "llama-bpe"); }),
                  "tokenizer.ggml.pre 'llama-bpe' disagrees with tokenizer.ggml.model 'gemma4'"));
        CHECK(has(
            refusal(
                work.path,
                [](GgufWriter& w) {
                    w.drop_value("tokenizer.ggml.add_space_prefix").boolean("tokenizer.ggml.add_space_prefix", true);
                }),
            "tokenizer.ggml.add_space_prefix is true, which disagrees with the tokenizer 'gemma4'"));
        CHECK(has(refusal(work.path,
                          [](GgufWriter& w) {
                              w.drop_value("tokenizer.ggml.unknown_token_id").u32("tokenizer.ggml.unknown_token_id", 4);
                          }),
                  "tokenizer.ggml.unknown_token_id is '<mask>', which disagrees"));
        xs::test::build_tiny_gemma4_gguf(work.path / "src", 7).write(work.path / "tiny.gguf");
        CHECK_THROWS_WITH_AS(import_into(work.path / "tiny.gguf", work.path / "with-mtp", work.path / "tiny.gguf"),
                             doctest::Contains("gemma4: an MTP head in a file of its own (--mtp) is not supported"),
                             Error);
    }
}
