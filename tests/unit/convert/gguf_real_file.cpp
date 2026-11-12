// A real qwen35 GGUF (XS_GGUF_FILE, e.g. unsloth/Qwen3.5-4B-GGUF's Qwen3.5-4B-Q4_K_M.gguf) against the Hugging Face
// checkpoint it was converted from (XS_QWEN35_DIR: its config.json, tokenizer.json, tokenizer_config.json and chat
// template; the weights are not read). The tokenizer written from the file passes the Qwen3.5 tokenizer parity cases
// and matches Hugging Face token by token; config.json gives the model code what the checkpoint's does; the file's
// chat template renders; the imported pack loads. Each case skips unless both variables are set and the file is a
// qwen35 one.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "chat/chat_template.h"
#include "convert/gguf/architecture.h"
#include "convert/gguf/gguf_import.h"
#include "convert/gguf/tokenizer.h"
#include "convert/real_gguf.h"
#include "io/gguf.h"
#include "model/tiny_checkpoints.h"
#include "models/config_util.h"
#include "models/model.h"
#include "models/qwen3_5/qwen3_5.h"
#include "tokenizer/parity_cases.h"
#include "tokenizer/tokenizer.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;

namespace {

std::optional<xs::test::RealGguf> real_file() {
    return xs::test::real_gguf("XS_QWEN35_DIR", "qwen35");
}

}  // namespace

TEST_SUITE("gguf real file") {
    TEST_CASE("tokenizer parity: a real qwen35 GGUF's tokenizer encodes and decodes as Hugging Face's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_real_tokenizer_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        const nlohmann::json fixture = nlohmann::json::parse(
            std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/qwen35_cases.json"));
        MESSAGE("parity cases: " << fixture["encode_cases"].size() << " encode, " << fixture["decode_cases"].size()
                                 << " decode");
        xs::test::check_tokenizer_cases(*imported, fixture);

        // Token by token against the checkpoint's tokenizer: the same text and the same special flag for every id.
        const auto original = tok::Tokenizer::load(real->hf);
        REQUIRE(imported->vocab_size() == original->vocab_size());
        int differ = 0;
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            if (imported->token_text(id) != original->token_text(id) ||
                imported->is_special(id) != original->is_special(id)) {
                ++differ;
            }
        }
        CHECK(differ == 0);

        // The parts the file names only by tokenizer.ggml.pre are the checkpoint's.
        const nlohmann::json written = nlohmann::json::parse(std::ifstream(work.path / "tokenizer.json"));
        const nlohmann::json hf = nlohmann::json::parse(std::ifstream(real->hf / "tokenizer.json"));
        for (const char* part : {"normalizer", "pre_tokenizer", "post_processor", "decoder"}) {
            CAPTURE(part);
            CHECK(written[part] == hf[part]);
        }
        nlohmann::json model = hf["model"];
        model.erase("vocab");
        model.erase("merges");
        nlohmann::json written_model = written["model"];
        CHECK(written_model["merges"] == hf["model"]["merges"]);
        CHECK(written_model["vocab"] == hf["model"]["vocab"]);
        written_model.erase("vocab");
        written_model.erase("merges");
        CHECK(written_model == model);
    }

    TEST_CASE("config: what the model code reads from a real qwen35 GGUF equals the checkpoint's config") {
        const auto real = real_file();
        if (!real) return;
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan =
            gguf_import::find_architecture(file.get_string("general.architecture")).plan(file);
        const nlohmann::json ours = nlohmann::json::parse(plan.config.dump());
        const nlohmann::json theirs = cfg::read_json(real->hf / "config.json");
        const qwen3_5::Config a = qwen3_5::parse_config(ours);
        const qwen3_5::Config b = qwen3_5::parse_config(theirs);
        CHECK(a.vocab_size == b.vocab_size);
        CHECK(a.hidden_size == b.hidden_size);
        CHECK(a.intermediate_size == b.intermediate_size);
        CHECK(a.num_experts == b.num_experts);
        CHECK(a.num_layers == b.num_layers);
        CHECK(static_cast<float>(a.rms_norm_eps) == static_cast<float>(b.rms_norm_eps));
        CHECK(a.rms_norm_eps == b.rms_norm_eps);
        CHECK(a.max_position == b.max_position);
        CHECK(a.layer_kinds == b.layer_kinds);
        CHECK(a.num_heads == b.num_heads);
        CHECK(a.num_kv_heads == b.num_kv_heads);
        CHECK(a.head_dim == b.head_dim);
        CHECK(a.attn_output_gate == b.attn_output_gate);
        CHECK(a.rope_theta == b.rope_theta);
        CHECK(a.rotary_dim == b.rotary_dim);
        CHECK(a.linear_key_heads == b.linear_key_heads);
        CHECK(a.linear_value_heads == b.linear_value_heads);
        CHECK(a.linear_key_dim == b.linear_key_dim);
        CHECK(a.linear_value_dim == b.linear_value_dim);
        CHECK(a.conv_kernel == b.conv_kernel);
        // What differs, by construction: the value heads' order, and an MTP head only when the file holds one.
        CHECK(a.value_head_order == qwen3_5::ValueHeadOrder::Tiled);
        CHECK(b.value_head_order == qwen3_5::ValueHeadOrder::Grouped);
        const int64_t nextn =
            file.find("qwen35.nextn_predict_layers") ? file.get_int("qwen35.nextn_predict_layers") : 0;
        CHECK(a.mtp_layers == nextn);
        if (a.mtp_layers != b.mtp_layers) {
            MESSAGE("mtp_num_hidden_layers: " << a.mtp_layers << " (the file holds no MTP head), the checkpoint's "
                                              << b.mtp_layers);
        }
        CHECK(ours["tie_word_embeddings"] == theirs["text_config"]["tie_word_embeddings"]);

        // Stop tokens: the file names one end-of-sequence token; the checkpoint's config and tokenizer_config name
        // theirs. Every stop token of the pack is one of the checkpoint's.
        const std::vector<int32_t> stops = gguf_import::tokenizer_files(file).stop_ids;
        std::vector<int32_t> hf_stops = cfg::get_int_list(theirs["text_config"], "eos_token_id", "config.json");
        if (const auto eos = cfg::tokenizer_eos_id(real->hf)) hf_stops.push_back(*eos);
        for (const int32_t id : stops) {
            CAPTURE(id);
            CHECK(std::find(hf_stops.begin(), hf_stops.end(), id) != hf_stops.end());
        }
        for (const int32_t id : hf_stops) {
            if (std::find(stops.begin(), stops.end(), id) == stops.end()) {
                MESSAGE("stop token " << id << " of the checkpoint is not named by the file");
            }
        }
    }

    TEST_CASE("the chat template of a real qwen35 GGUF loads and renders") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_real_template_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const std::string source = gguf_import::chat_template(file);
        std::ofstream(work.path / "chat_template.jinja") << source;
        const auto chat = chat::ChatTemplate::load(work.path);
        const nlohmann::json messages = nlohmann::json::array(
            {{{"role", "system"}, {"content", "Be brief."}}, {{"role", "user"}, {"content", "What is 2 + 2?"}}});
        const std::string prompt = chat->render(messages, nullptr, true);
        CHECK(prompt.find("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
                          "<|im_start|>assistant\n") != std::string::npos);
        std::ifstream hf(real->hf / "chat_template.jinja");
        if (hf && std::string(std::istreambuf_iterator<char>(hf), {}) != source) {
            MESSAGE("the file's chat template is not the checkpoint's chat_template.jinja (the publisher changed it)");
        }
    }

    TEST_CASE("a real qwen35 GGUF imports into a pack that loads") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_real_import_test");
        gguf_import::ImportOptions o;
        o.source = real->gguf;
        o.output = work.path / "pack";
        const gguf_import::ImportReport report = gguf_import::import_gguf(o);
        CHECK(report.architecture == "qwen35");
        const auto model = load_model(o.output);
        const gguf::File file(real->gguf);
        CHECK(model->info().vocab_size == file.find_tensor("token_embd.weight")->rows());
        for (const gguf::TensorInfo& t : file.tensors()) {
            if (t.dtype != DType::Block || t.name != "token_embd.weight") continue;
            const TensorView& v = model->weights().get("model.embed_tokens.weight");
            CHECK(v.dtype == DType::Block);
            CHECK(v.block == t.block);
            CHECK(std::memcmp(v.data, t.data, t.nbytes) == 0);
        }
    }
}
