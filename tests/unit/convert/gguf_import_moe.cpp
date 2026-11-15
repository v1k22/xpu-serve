// GGUF import of a Qwen3.5 mixture of experts end to end, hermetically: a tiny qwen35moe GGUF written the way
// llama.cpp's converter writes one (tiny_gguf.h: the router, 3-D gate, up and down banks in Q8_0 / Q4_0 / F16 / BF16
// with gate and up in different formats in some blocks, the shared expert and its 1-D gate, tiled value heads, an MTP
// block with experts of its own) imports as the model its blocks decode to: the banks kept apart byte for byte, the
// same weights, logits, greedy tokens and drafts as its twin checkpoint, whose banks are fused, on cpu-ref; the same
// model as a file without the head and the head's own file (--mtp) imports as the same pack; and what the importer does
// not implement is refused by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "io/gguf.h"
#include "io/safetensors.h"
#include "models/model.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::draft_logits;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::import_into;
using xs::test::relative_gap;
using xs::test::TempDir;
using xs::test::values;

namespace {

const std::vector<int32_t> kPrompt{3, 17, 42, 99, 5, 250, 7, 8, 64, 128, 266, 271};
const std::vector<int32_t> kSteps{11, 200, 1};
const LoadOptions kWithHead{.speculation = true, .draft_shortlist = 0};

std::vector<std::vector<float>> step_logits(const Model& model) {
    return xs::test::step_logits(model, kPrompt, kSteps);
}

// The name of the fused bank `name` (".../experts.gate_up_proj") with its last part replaced by `part`.
std::string beside(const std::string& name, const char* part) {
    return name.substr(0, name.rfind('.') + 1) + part;
}

// A fused bank's values from those of its halves [E, I, hidden]: each expert's gate rows, then its up rows.
std::vector<float> fused(const TensorView& gate, const TensorView& up) {
    const std::vector<float> g = values(gate), u = values(up);
    const auto half = static_cast<std::ptrdiff_t>(gate.numel() / gate.shape[0]);
    std::vector<float> out;
    for (int64_t e = 0; e < gate.shape[0]; ++e) {
        out.insert(out.end(), g.begin() + e * half, g.begin() + (e + 1) * half);
        out.insert(out.end(), u.begin() + e * half, u.begin() + (e + 1) * half);
    }
    return out;
}

// The tiny qwen35moe GGUF with `change` applied, imported into a fresh directory: the error it gives, or "" when it
// imports.
template <typename Change>
std::string refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(dir, [](const fs::path& src) { return xs::test::build_tiny_moe_gguf(src, 11); }, change);
}

std::string file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

}  // namespace

TEST_SUITE("gguf import") {
    TEST_CASE("a tiny qwen35moe GGUF imports as the model its blocks decode to") {
        const TempDir work("xs_gguf_import_moe_test");
        xs::test::build_tiny_moe_gguf(work.path, 11).write(work.path / "tiny.gguf");
        const gguf_import::ImportReport report = import_into(work.path / "tiny.gguf", work.path / "pack");
        CHECK(report.architecture == "qwen35moe");
        CHECK(report.model_name == "Tiny Qwen3.5 MoE");

        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "pack" / "config.json"));
        CHECK(config["architectures"] == nlohmann::json::array({"Qwen3_5MoeForCausalLM"}));
        CHECK(config["model_type"] == "qwen3_5_moe_text");
        CHECK(config["num_experts"] == 8);
        CHECK(config["num_experts_per_tok"] == 2);
        CHECK(config["moe_intermediate_size"] == 64);
        CHECK(config["shared_expert_intermediate_size"] == 96);
        CHECK_FALSE(config.contains("intermediate_size"));
        CHECK(config["mtp_num_hidden_layers"] == 1);
        CHECK(config["linear_value_head_order"] == "tiled");

        // The banks stay apart, each in its file's format, under names of their own; the router is a float tensor.
        const nlohmann::json manifest = nlohmann::json::parse(std::ifstream(work.path / "pack" / kPackManifest));
        CHECK(manifest["source"]["general.architecture"] == "qwen35moe");
        CHECK(manifest["source"]["model_type"] == "qwen3_5_moe_text");
        const std::string E = "model.layers.0.mlp.experts.";
        CHECK(manifest["tensors"][E + "gate_proj"] == nlohmann::json{{"format", "Q4_0"}, {"shape", {8, 64, 128}}});
        CHECK(manifest["tensors"][E + "up_proj"] == nlohmann::json{{"format", "Q8_0"}, {"shape", {8, 64, 128}}});
        CHECK(manifest["tensors"][E + "down_proj"] == nlohmann::json{{"format", "Q8_0"}, {"shape", {8, 128, 64}}});
        CHECK(manifest["tensors"]["mtp.layers.0.mlp.experts.down_proj"] ==
              nlohmann::json{{"format", "Q4_0"}, {"shape", {8, 128, 64}}});
        CHECK_FALSE(manifest["tensors"].contains("model.layers.0.mlp.experts.gate_up_proj"));
        CHECK_FALSE(manifest["tensors"].contains("model.layers.0.mlp.gate.weight"));

        const auto pack = load_model(work.path / "pack", kWithHead);
        const auto twin = load_model(work.path / "twin", kWithHead);
        const WeightStore& ours = pack->weights();
        const gguf::File file(work.path / "tiny.gguf");
        for (const auto& [name, stored] :
             {std::pair{E + "gate_proj", "blk.0.ffn_gate_exps.weight"},
              std::pair{E + "up_proj", "blk.0.ffn_up_exps.weight"},
              std::pair{E + "down_proj", "blk.0.ffn_down_exps.weight"},
              std::pair{std::string("mtp.layers.0.mlp.experts.gate_proj"), "blk.4.ffn_gate_exps.weight"}}) {
            CAPTURE(name);
            const gguf::TensorInfo* t = file.find_tensor(stored);
            REQUIRE(t != nullptr);
            const TensorView& v = ours.get(name);
            CHECK(v.dtype == DType::Block);
            CHECK(v.block == t->block);
            CHECK(v.shape == t->shape());
            CHECK(std::memcmp(v.data, t->data, t->nbytes) == 0);
        }
        CHECK(ours.get("model.layers.2.mlp.experts.gate_proj").dtype == DType::F16);
        CHECK(ours.get("model.layers.1.mlp.experts.down_proj").dtype == DType::BF16);
        CHECK(ours.get("mtp.layers.0.mlp.experts.up_proj").dtype == DType::BF16);
        CHECK(ours.get("model.layers.0.mlp.gate.weight").dtype == DType::F32);
        CHECK(ours.get("model.layers.0.mlp.shared_expert_gate.weight").shape == Dims{1, 128});
        CHECK(ours.get("model.layers.0.mlp.shared_expert.down_proj.weight").shape == Dims{128, 96});

        // Every weight of the twin with the same values (value heads in the file's order): a fused bank of the twin
        // is the pack's two halves, an expert's gate rows before its up rows.
        std::vector<std::string> expected;
        int banks = 0;
        for (const std::string& name : twin->weights().names()) {
            CAPTURE(name);
            const TensorView& b = twin->weights().get(name);
            if (!name.ends_with("experts.gate_up_proj")) {
                expected.push_back(name);
                REQUIRE(ours.contains(name));
                REQUIRE(ours.get(name).shape == b.shape);
                CHECK(xs::test::grouped_values(name, ours.get(name)) == values(b));
                continue;
            }
            ++banks;
            const std::string gate = beside(name, "gate_proj"), up = beside(name, "up_proj");
            expected.insert(expected.end(), {gate, up});
            REQUIRE(ours.contains(gate));
            REQUIRE(ours.contains(up));
            CHECK(ours.get(gate).shape == Dims{8, 64, 128});
            CHECK(ours.get(up).shape == Dims{8, 64, 128});
            CHECK(b.shape == Dims{8, 128, 128});
            CHECK(fused(ours.get(gate), ours.get(up)) == values(b));
        }
        CHECK(banks == 5);  // four layers and the MTP head
        std::sort(expected.begin(), expected.end());
        CHECK(ours.names() == expected);
        CHECK(pack->info().stop_token_ids == twin->info().stop_token_ids);

        const auto reference = step_logits(*twin);
        const double gap = relative_gap(step_logits(*pack), reference);
        MESSAGE("imported pack against its twin: largest logit difference " << gap << " of the largest logit");
        CHECK(gap < 1e-6);
        CHECK(xs::test::greedy(*pack, kPrompt, 8) == xs::test::greedy(*twin, kPrompt, 8));
        // The head alone, on the same hidden rows, and after the model's own prefill, three drafts chained.
        CHECK(draft_logits(*pack) == draft_logits(*twin));
        const xs::test::Drafts drafts = xs::test::drafts(*pack, kPrompt),
                               twin_drafts = xs::test::drafts(*twin, kPrompt);
        MESSAGE("drafts: " << drafts.tokens[0] << " " << drafts.tokens[1] << " " << drafts.tokens[2]);
        CHECK(drafts.tokens == twin_drafts.tokens);
        CHECK(relative_gap({drafts.logits}, {twin_drafts.logits}) < 1e-6);

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

    TEST_CASE("planted: a qwen35moe file whose gate and up banks are exchanged is another model") {
        const TempDir work("xs_gguf_import_moe_swap_test");
        const GgufWriter right = xs::test::build_tiny_moe_gguf(work.path, 11);
        // The file with the two banks of each of blocks [first, last) under each other's name, imported.
        const auto exchanged = [&](const char* name, int first, int last) {
            GgufWriter w = right;
            for (int i = first; i < last; ++i) {
                const std::string b = "blk." + std::to_string(i) + ".";
                w.swap_tensors(b + "ffn_gate_exps.weight", b + "ffn_up_exps.weight");
            }
            w.write(work.path / "variant.gguf");
            import_into(work.path / "variant.gguf", work.path / name);
            return load_model(work.path / name, kWithHead);
        };
        const auto twin = load_model(work.path / "twin", kWithHead);
        const auto reference = step_logits(*twin);
        const std::vector<float> reference_drafts = draft_logits(*twin);
        const auto none = exchanged("none", 0, 0);
        CHECK(relative_gap(step_logits(*none), reference) < 1e-6);
        CHECK(draft_logits(*none) == reference_drafts);
        // The decoder's banks exchanged: other logits. The head's alone: the same logits, other drafts.
        CHECK(relative_gap(step_logits(*exchanged("decoder", 0, 4)), reference) > 1e-2);
        const auto head = exchanged("head", 4, 5);
        CHECK(relative_gap(step_logits(*head), reference) < 1e-6);
        CHECK(relative_gap({draft_logits(*head)}, {reference_drafts}) > 1e-2);
    }

    TEST_CASE("a qwen35moe file without its MTP head and the head's own file import as the file that holds both") {
        const TempDir work("xs_gguf_import_moe_mtp_test");
        const GgufWriter whole = xs::test::build_tiny_moe_gguf(work.path / "src", 11);
        whole.write(work.path / "whole.gguf");
        const xs::test::TinyMtpFiles split = xs::test::split_tiny_gguf(whole, "qwen35moe");
        split.model.write(work.path / "model.gguf");
        split.mtp.write(work.path / "mtp.gguf");
        import_into(work.path / "whole.gguf", work.path / "one");
        import_into(work.path / "model.gguf", work.path / "two", work.path / "mtp.gguf");

        // The same files, byte for byte, but the manifest, which names its sources.
        int files = 0;
        for (const auto& e : fs::directory_iterator(work.path / "one")) {
            const std::string name = e.path().filename().string();
            CAPTURE(name);
            REQUIRE(fs::exists(work.path / "two" / name));
            if (name == kPackManifest) continue;
            CHECK(file_bytes(work.path / "two" / name) == file_bytes(e.path()));
            ++files;
        }
        CHECK(files >= 5);
        const nlohmann::json manifest = nlohmann::json::parse(std::ifstream(work.path / "two" / kPackManifest));
        CHECK(manifest["source"]["mtp"] == "mtp.gguf");
        CHECK(manifest["tensors"].contains("mtp.layers.0.mlp.experts.gate_proj"));

        const xs::test::Drafts reference = xs::test::drafts(*load_model(work.path / "one", kWithHead), kPrompt);
        const xs::test::Drafts ours = xs::test::drafts(*load_model(work.path / "two", kWithHead), kPrompt);
        CHECK(ours.logits == reference.logits);
        CHECK(ours.tokens == reference.tokens);

        // The model file alone imports without a head; the head of a model of other experts is refused.
        import_into(work.path / "model.gguf", work.path / "bare");
        CHECK(load_model(work.path / "bare", kWithHead)->speculation() == nullptr);
        xs::test::TinyMtpFiles other = split;
        other.mtp.drop_value("qwen35moe.expert_used_count").u32("qwen35moe.expert_used_count", 4);
        other.mtp.write(work.path / "other.gguf");
        std::string error;
        try {
            import_into(work.path / "model.gguf", work.path / "other", work.path / "other.gguf");
        } catch (const Error& e) {
            error = e.what();
        }
        CHECK(has(error,
                  "qwen35moe: the MTP file's qwen35moe.expert_used_count (u32 4) differs from the model file's "
                  "(u32 2): it is the head of another model"));
    }

    TEST_CASE("qwen35moe: refusals name what is wrong") {
        const TempDir work("xs_gguf_import_moe_refusal_test");
        using W = GgufWriter;
        const auto refused = [&](auto change) { return refusal(work.path, change); };
        const auto zeros = [](W& w, const char* name, std::vector<int64_t> dims) {
            int64_t n = 1;
            for (const int64_t d : dims)
                n *= d;
            w.drop_tensor(name).f32_tensor(name, std::move(dims), std::vector<float>(static_cast<size_t>(n)));
        };
        CHECK(refused([](W&) {}).empty());
        // The shared expert's gate as the converter itself writes it, [1, hidden].
        CHECK(refused([&](W& w) { zeros(w, "blk.0.ffn_gate_inp_shexp.weight", {128, 1}); }).empty());

        // Metadata: a key the importer does not know, a dense model's key, each expert key missing, counts that
        // disagree.
        CHECK(has(refused([](W& w) { w.f32("qwen35moe.expert_weights_scale", 2.0f); }),
                  "qwen35moe: metadata 'qwen35moe.expert_weights_scale' is not read by this importer"));
        CHECK(has(refused([](W& w) { w.u32("qwen35moe.feed_forward_length", 256); }),
                  "metadata 'qwen35moe.feed_forward_length' is not read by this importer"));
        for (const char* key :
             {"qwen35moe.expert_count", "qwen35moe.expert_used_count", "qwen35moe.expert_feed_forward_length",
              "qwen35moe.expert_shared_feed_forward_length"}) {
            CAPTURE(key);
            CHECK(has(refused([&](W& w) { w.drop_value(key); }), std::string("has no metadata '") + key + "'"));
        }
        CHECK(has(
            refused([](W& w) { w.drop_value("qwen35moe.expert_used_count").u32("qwen35moe.expert_used_count", 9); }),
            "qwen35moe: expert_used_count 9 of expert_count 8"));
        CHECK(has(refused([](W& w) { w.drop_value("qwen35moe.expert_count").u32("qwen35moe.expert_count", 0); }),
                  "qwen35moe: qwen35moe.expert_count is 0"));

        // Tensors: a bank of another shape (in the file, or by the metadata), a missing one, a dense model's MLP, a
        // dense file's merged input projections, a tensor of another kind of file.
        CHECK(has(
            refused([&](W& w) { zeros(w, "blk.0.ffn_up_exps.weight", {128, 32, 8}); }),
            "qwen35moe: tensor 'blk.0.ffn_up_exps.weight' has shape [8, 32, 128]; the metadata gives [8, 64, 128]"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.3.ffn_down_exps.weight", {128, 64, 8}); }),
                  "tensor 'blk.3.ffn_down_exps.weight' has shape [8, 64, 128]; the metadata gives [8, 128, 64]"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.1.ffn_gate_exps.weight", {128, 512}); }),
                  "tensor 'blk.1.ffn_gate_exps.weight' has shape [512, 128]; the metadata gives [8, 64, 128]"));
        CHECK(has(
            refused([](W& w) {
                w.drop_value("qwen35moe.expert_feed_forward_length").u32("qwen35moe.expert_feed_forward_length", 32);
            }),
            "the metadata gives [8, 32, 128]"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.0.ffn_gate_inp.weight", {128, 4}); }),
                  "tensor 'blk.0.ffn_gate_inp.weight' has shape [4, 128]; the metadata gives [8, 128]"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.0.ffn_gate_inp_shexp.weight", {64}); }),
                  "tensor 'blk.0.ffn_gate_inp_shexp.weight' has shape [64]; the metadata gives [1, 128]"));
        CHECK(has(refused([](W& w) { w.drop_tensor("blk.1.ffn_up_exps.weight"); }),
                  "qwen35moe: the file has no tensor for 'model.layers.1.mlp.experts.up_proj' (block 1, linear "
                  "attention)"));
        CHECK(has(refused([](W& w) { w.drop_tensor("blk.4.ffn_gate_exps.weight"); }),
                  "the file has no tensor for 'mtp.layers.0.mlp.experts.gate_proj' (block 4, an MTP layer)"));
        CHECK(
            has(refused([](W& w) { w.drop_tensor("blk.3.ffn_gate_inp_shexp.weight"); }),
                "the file has no tensor for 'model.layers.3.mlp.shared_expert_gate.weight' (block 3, full attention)"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.0.ffn_gate.weight", {128, 256}); }),
                  "qwen35moe: tensor 'blk.0.ffn_gate.weight' has no counterpart in block 0 (linear attention)"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.0.attn_qkv_gate.weight", {128, 384}); }),
                  "tensor 'blk.0.attn_qkv_gate.weight' has no counterpart in block 0 (linear attention)"));
        CHECK(has(refused([&](W& w) { zeros(w, "blk.0.ffn_gate_up_exps.weight", {128, 128, 8}); }),
                  "tensor 'blk.0.ffn_gate_up_exps.weight' has no counterpart in block 0 (linear attention)"));

        // A dense file has no experts, and neither architecture is the other under its name.
        const auto dense = [&](auto change) {
            return xs::test::refusal(
                work.path, [](const fs::path& src) { return xs::test::build_tiny_gguf(src, 11); }, change);
        };
        CHECK(has(dense([&](W& w) { zeros(w, "blk.0.ffn_gate_exps.weight", {128, 64, 8}); }),
                  "qwen35: tensor 'blk.0.ffn_gate_exps.weight' has no counterpart in block 0 (linear attention)"));
        CHECK(has(dense([](W& w) { w.drop_value("general.architecture").string("general.architecture", "qwen35moe"); }),
                  "has no metadata 'qwen35moe.embedding_length'"));
    }
}
