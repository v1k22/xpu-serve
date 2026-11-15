// A real GGUF of the Qwen3.5 family (XS_GGUF_FILE) against the Hugging Face checkpoint it was converted from: a qwen35
// file (e.g. unsloth/Qwen3.5-4B-GGUF's Qwen3.5-4B-Q4_K_M.gguf) with XS_QWEN35_DIR, a qwen35moe file (a mixture of
// experts, e.g. one of Qwen3.6-35B-A3B) with XS_QWEN35MOE_DIR. The directory holds the checkpoint's small files:
// config.json, and tokenizer.json, tokenizer_config.json and chat_template.jinja, which are compared when they are
// there (the weights are not read). The tokenizer written from the file passes the Qwen3.5 tokenizer parity cases and
// matches Hugging Face token by token; config.json gives the model code what the checkpoint's does; the file's chat
// template renders; the imported pack loads, with its drafter when the file holds an MTP head, and holds every block
// format tensor of the file byte for byte (a mixture of experts' gate and up banks apart). Each case skips unless
// XS_GGUF_FILE and the variable of the file's architecture are set.
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

// The file and its checkpoint's directory: XS_QWEN35_DIR for a qwen35 file, XS_QWEN35MOE_DIR for a qwen35moe one.
std::optional<xs::test::RealGguf> real_file() {
    if (auto dense = xs::test::real_gguf("XS_QWEN35_DIR", "qwen35")) return dense;
    return xs::test::real_gguf("XS_QWEN35MOE_DIR", "qwen35moe");
}

// The MTP layers the file holds.
int64_t mtp_layers(const gguf::File& file, const xs::test::RealGguf& real) {
    const std::string key = real.architecture + ".nextn_predict_layers";
    return file.find(key) ? file.get_int(key) : 0;
}

}  // namespace

TEST_SUITE("gguf real file") {
    TEST_CASE("tokenizer parity: a real qwen35 or qwen35moe GGUF's tokenizer encodes and decodes as Hugging Face's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_real_tokenizer_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        nlohmann::json fixture = nlohmann::json::parse(
            std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/qwen35_cases.json"));
        MESSAGE("parity cases: " << fixture["encode_cases"].size() << " encode, " << fixture["decode_cases"].size()
                                 << " decode");
        // The cases are Qwen3.5-4B's, whose tokenizer names no BOS token. A file can name one (to a Qwen3.5 prompt it
        // is a chat template variable, never a token added), and the tokenizer written from it then has the file's.
        if (const gguf::Value* bos = file.find("tokenizer.ggml.bos_token_id")) {
            MESSAGE("the file names a BOS token, " << bos->as_int() << "; the parity cases' tokenizer has none");
            fixture["bos_id"] = bos->as_int();
        }
        xs::test::check_tokenizer_cases(*imported, fixture);

        if (!fs::exists(real->hf / "tokenizer.json")) {
            MESSAGE(real->hf.string() << " has no tokenizer.json: the tokenizer is not compared with the checkpoint's");
            return;
        }
        // Token by token against the checkpoint's tokenizer: the same text and the same special flag for every id.
        const auto original = tok::Tokenizer::load(real->hf);
        CHECK(xs::test::differing_tokens(*imported, *original) == 0);
        if (imported->bos_id() != original->bos_id()) {
            MESSAGE("the file's BOS token is not the one of the checkpoint's tokenizer_config.json");
        }

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

    TEST_CASE("config: what the model code reads from a real qwen35 or qwen35moe GGUF equals the checkpoint's config") {
        const auto real = real_file();
        if (!real) return;
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture(real->architecture).plan(file);
        const nlohmann::json ours = nlohmann::json::parse(plan.config.dump());
        const nlohmann::json theirs = cfg::read_json(real->hf / "config.json");
        const nlohmann::json& text = theirs.contains("text_config") ? theirs["text_config"] : theirs;
        const qwen3_5::Config a = qwen3_5::parse_config(ours);
        const qwen3_5::Config b = qwen3_5::parse_config(theirs);
        xs::test::check_decoder_config(a, b);
        CHECK((a.num_experts > 0) == (real->architecture == "qwen35moe"));
        // An MTP head only when the file holds one.
        CHECK(a.mtp_layers == mtp_layers(file, *real));
        if (a.mtp_layers != b.mtp_layers) {
            MESSAGE("mtp_num_hidden_layers: " << a.mtp_layers << " (the file holds no MTP head), the checkpoint's "
                                              << b.mtp_layers);
        }
        CHECK(ours["tie_word_embeddings"] == text["tie_word_embeddings"]);

        // Stop tokens: the file names one end-of-sequence token; the checkpoint's config (the whole model's and the
        // text model's) and tokenizer_config name theirs. Every stop token of the pack is one of the checkpoint's.
        const std::vector<int32_t> stops = gguf_import::tokenizer_files(file).stop_ids;
        std::vector<int32_t> hf_stops;
        for (const nlohmann::json* j : {&theirs, &text}) {
            if (!j->contains("eos_token_id")) continue;
            const std::vector<int32_t> ids = cfg::get_int_list(*j, "eos_token_id", "config.json");
            hf_stops.insert(hf_stops.end(), ids.begin(), ids.end());
        }
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

    TEST_CASE("the chat template of a real qwen35 or qwen35moe GGUF loads and renders") {
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

    TEST_CASE("a real qwen35 or qwen35moe GGUF imports into a pack that loads") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_real_import_test");
        gguf_import::ImportOptions o;
        o.source = real->gguf;
        o.output = work.path / "pack";
        const gguf_import::ImportReport report = gguf_import::import_gguf(o);
        CHECK(report.architecture == real->architecture);
        // With the drafter's programs, which read the MTP head's weights when the file holds one.
        const auto model = load_model(o.output, LoadOptions{.speculation = true, .draft_shortlist = 0});
        const gguf::File file(real->gguf);
        CHECK(model->info().vocab_size == file.find_tensor("token_embd.weight")->rows());
        CHECK((model->speculation() != nullptr) == (mtp_layers(file, *real) > 0));

        // Every tensor the file stores in a block format: in the pack under the name the model reads, in that format,
        // the file's bytes.
        const WeightStore& weights = model->weights();
        const gguf_import::ImportPlan plan = gguf_import::find_architecture(real->architecture).plan(file);
        int blocks = 0;
        for (const gguf_import::PackTensor& p : plan.tensors) {
            if (p.source->dtype != DType::Block) continue;
            CAPTURE(p.name);
            REQUIRE(weights.contains(p.name));
            const TensorView& v = weights.get(p.name);
            CHECK(v.dtype == DType::Block);
            CHECK(v.block == p.source->block);
            CHECK(v.shape == p.shape);
            const size_t row_bytes = p.source->nbytes / static_cast<size_t>(p.source->rows());
            const auto rows = static_cast<size_t>(numel(p.shape) / p.shape.back());
            CHECK(std::memcmp(v.data, p.source->data + static_cast<size_t>(p.first_row) * row_bytes,
                              rows * row_bytes) == 0);
            ++blocks;
        }
        MESSAGE(blocks << " block format tensors of the file are in the pack byte for byte");

        // A mixture of experts keeps its gate and up banks apart, in every layer and in the MTP head.
        const qwen3_5::Config config = qwen3_5::parse_config(cfg::read_json(o.output / "config.json"));
        if (config.num_experts == 0) return;
        std::vector<std::string> layers;
        for (int64_t i = 0; i < config.num_layers; ++i)
            layers.push_back("model.layers." + std::to_string(i));
        for (int64_t i = 0; i < config.mtp_layers; ++i)
            layers.push_back("mtp.layers." + std::to_string(i));
        const int64_t e = config.num_experts, inter = config.moe_intermediate_size, h = config.hidden_size;
        for (const std::string& layer : layers) {
            CAPTURE(layer);
            const std::string E = layer + ".mlp.experts.";
            CHECK_FALSE(weights.contains(E + "gate_up_proj"));
            CHECK(weights.get(E + "gate_proj").shape == Dims{e, inter, h});
            CHECK(weights.get(E + "up_proj").shape == Dims{e, inter, h});
            CHECK(weights.get(E + "down_proj").shape == Dims{e, h, inter});
            CHECK(weights.get(layer + ".mlp.gate.weight").shape == Dims{e, h});
            CHECK(weights.get(layer + ".mlp.shared_expert_gate.weight").shape == Dims{1, h});
        }
    }
}
