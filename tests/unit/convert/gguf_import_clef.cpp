// GGUF import of Clef end to end, hermetically: a tiny clef GGUF written the way llama.cpp's converter writes one
// (tiny_clef_gguf.h: the qwen35 decoder under clef.*, the head's blocks as dec.blk.<i>, each attention's in_proj as
// three tensors, the scales as the values the head multiplies by, the head's matrices in block formats) imports as the
// Clef checkpoint its blocks decode to: the pack has a converted checkpoint's names and joint_head_config.json, the
// same weights, and answers requests as the checkpoint does on cpu-ref (the GPU as cpu-ref); planted: the query and key
// projections exchanged, the scales stored as the parameters; what the importer does not implement is refused by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <vector>

#include "backend/device_test_support.h"
#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_clef_gguf.h"
#include "io/safetensors.h"
#include "models/model.h"
#include "models/policy.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::import_into;
using xs::test::TempDir;
using xs::test::values;

namespace {

DecisionRequest ticket() {
    DecisionRequest r;
    r.state = Json{{"ticket", 4411}, {"text", "cannot log in after the reset email expired"}, {"vip", true}};
    r.questions = {
        xs::test::choice("team", "Which team?", {{"billing", "payments"}, {"access", "logins"}, {"other", nullptr}}),
        xs::test::noul("urgent", "Is it urgent?"),
        xs::test::score("effort", "How much work?", {"minutes", "hours", "days"})};
    return r;
}

std::vector<std::vector<double>> answers(const DecisionModel& model, Backend& backend, SessionOptions options) {
    const auto session = backend.compile(model.programs(), model.weights(), options);
    return xs::test::decide(model, *session, ticket(), DecisionLimits{4096, 4096});
}

std::vector<std::vector<double>> answers(const fs::path& dir) {
    return answers(*load_decision_model(dir), *make_backend("cpu-ref"), SessionOptions{.precision = DType::F32});
}

double max_diff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    REQUIRE(a.size() == b.size());
    double d = 0;
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t o = 0; o < a[q].size(); ++o)
            d = std::max(d, std::abs(a[q][o] - b[q][o]));
    }
    return d;
}

// The names a model reads: the decoder's and the head's (a checkpoint's directory also shows the head's file under
// its own names, which the model reads as joint_head.*).
std::set<std::string> model_names(const DecisionModel& model) {
    std::set<std::string> out;
    for (const std::string& name : model.weights().names()) {
        if (name.starts_with("model.") || name.starts_with("joint_head.")) out.insert(name);
    }
    return out;
}

// The tiny clef GGUF `variant` imported into `dir`/pack.
fs::path imported(const fs::path& dir, const xs::test::TinyClefVariant& variant = {}) {
    xs::test::build_tiny_clef_gguf(dir, 11, variant).write(dir / "tiny.gguf");
    import_into(dir / "tiny.gguf", dir / "pack");
    return dir / "pack";
}

std::string file_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

template <typename Change>
std::string refusal(const fs::path& dir, Change change) {
    return xs::test::refusal(dir, [](const fs::path& src) { return xs::test::build_tiny_clef_gguf(src, 11); }, change);
}

}  // namespace

TEST_SUITE("gguf import clef") {
    TEST_CASE("a tiny clef GGUF imports as the Clef checkpoint its blocks decode to") {
        const TempDir work("xs_gguf_import_clef_test");
        const fs::path pack = imported(work.path);
        const fs::path twin = work.path / "twin";

        // A decision model's pack: the checkpoint's joint_head_config.json beside the decoder's config.json;
        // llama.cpp's own prompt template is not imported.
        std::set<std::string> files;
        for (const auto& e : fs::directory_iterator(pack))
            files.insert(e.path().filename().string());
        CHECK(files == std::set<std::string>{"chat_template.jinja", "config.json", "joint_head_config.json",
                                             "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                             kPackManifest});
        CHECK(nlohmann::json::parse(std::ifstream(pack / "joint_head_config.json")) ==
              nlohmann::json::parse(std::ifstream(work.path / "hf" / "joint_head_config.json")));
        CHECK(file_bytes(pack / "chat_template.jinja") == xs::test::kTinyChatTemplate);
        CHECK(model_kind(pack) == ModelKind::Decision);
        const auto ours = load_decision_model(pack);
        const auto reference = load_decision_model(twin);
        CHECK(ours->spec().arch == "clef");

        // The twin's names, the head's in_proj apart as q, k and v as in a pack of the checkpoint; the scales as stored
        // in place of the three parameters.
        std::set<std::string> expected = model_names(*reference);
        for (const char* p : {"prior_logit_scale", "joint_logit_scale", "residual_gate"})
            CHECK(expected.erase(std::string("joint_head.") + p) == 1);
        expected.insert("joint_head.scales");
        CHECK(model_names(*ours) == expected);
        for (const std::string& name : model_names(*reference)) {
            if (!ours->weights().contains(name)) continue;
            CAPTURE(name);
            const TensorView& a = ours->weights().get(name);
            REQUIRE(a.shape == reference->weights().get(name).shape);
            CHECK(xs::test::grouped_values(name, a) == values(reference->weights().get(name)));
        }
        // The values the head multiplies by, as the converter computed them from the parameters.
        std::vector<float> scales;
        for (const char* p : {"prior_logit_scale", "joint_logit_scale", "residual_gate"}) {
            const double x = values(reference->weights().get(std::string("joint_head.") + p))[0];
            scales.push_back(static_cast<float>(std::string(p) == "residual_gate"
                                                    ? 1.0 / (1.0 + std::exp(-x))
                                                    : std::exp(std::min(x, std::log(100.0)))));
        }
        CHECK(values(ours->weights().get("joint_head.scales")) == scales);
        // The head's matrices in the file's formats, read by the programs (attention, MLP, projections) and on the
        // host (the output layer's rows of an option's tokens).
        const auto format = [&](const std::string& name) { return ours->weights().get(name).dtype; };
        CHECK(format("joint_head.evidence_layers.0.attention.q_proj.weight") == DType::Block);
        CHECK(format("joint_head.layers.1.multihead_attn.out_proj.weight") == DType::Block);
        CHECK(format("joint_head.memory_projection.weight") == DType::Block);
        CHECK(format("joint_head.residual_scorer.3.weight") == DType::BF16);
        CHECK(format("model.embed_tokens.weight") == DType::Block);
        CHECK(ours->weights().get("joint_head.residual_scorer.3.weight").shape == Dims{1, 64});

        const auto p = answers(twin);
        const auto q = answers(pack);
        MESSAGE("imported pack against its twin: largest probability difference " << max_diff(p, q));
        CHECK(max_diff(p, q) < 1e-6);
        CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

        SUBCASE("the GPU agrees with the reference backend") {
            namespace dev = xs::test::device;
            if (!dev::enabled()) return;
            CHECK(max_diff(q, answers(*ours, dev::backend(), dev::options())) < (dev::is_f16() ? 5e-3 : 1e-4));
        }
    }

    TEST_CASE("clef: the layouts of a file, and planted import errors") {
        const TempDir work("xs_gguf_import_clef_variant_test");
        const fs::path pack = imported(work.path);
        const auto p = answers(work.path / "twin");

        // decision.scorer_out.weight as the converter writes it, [1, width], not 1-D as llama-quantize does.
        const fs::path other = imported(work.path / "2d", {.scorer_out_2d = true});
        CHECK(file_bytes(other / "model.safetensors") == file_bytes(pack / "model.safetensors"));

        // Planted: the query and key projections exchanged, the scales taken for the parameters.
        const double swapped = max_diff(p, answers(imported(work.path / "swapped", {.swap_query_key = true})));
        const double raw = max_diff(p, answers(imported(work.path / "raw", {.raw_scales = true})));
        MESSAGE("largest probability difference: query and key exchanged " << swapped << ", the parameters as scales "
                                                                           << raw);
        CHECK(swapped > 1e-3);
        CHECK(raw > 1e-3);
    }

    TEST_CASE("clef: refusals name what is wrong") {
        const TempDir work("xs_gguf_import_clef_refusal_test");
        const auto refused = [&](auto change) { return refusal(work.path, change); };
        using W = GgufWriter;
        CHECK(refused([](W&) {}).empty());
        CHECK(has(refused([](W& w) { w.drop_value("clef.decision.type").string("clef.decision.type", "lev"); }),
                  "clef: decision.type 'lev' is not supported (supported: clef)"));
        CHECK(has(refused([](W& w) {
                      w.drop_value("clef.attention.layer_norm_epsilon").f32("clef.attention.layer_norm_epsilon", 1e-6f);
                  }),
                  "clef: attention.layer_norm_epsilon 1e-06 is not supported: the head's LayerNorm uses 1e-05"));
        // A key of the namespace that neither the decoder nor the head reads, an MTP head's among them.
        CHECK(has(refused([](W& w) { w.u32("clef.nextn_predict_layers", 0); }),
                  "clef: metadata 'clef.nextn_predict_layers' is not read by this importer"));
        CHECK(has(refused([](W& w) { w.f32("clef.decision.temperature", 1.5f); }),
                  "metadata 'clef.decision.temperature' is not read by this importer"));
        // The decoder is checked as a qwen35 file's is, under clef.*.
        CHECK(has(refused([](W& w) { w.string("clef.rope.scaling.type", "yarn"); }),
                  "clef: clef.rope.scaling.type 'yarn' is not supported"));
        CHECK(has(refused([](W& w) { w.drop_tensor("blk.2.ssm_dt.bias"); }),
                  "clef: the file has no tensor for 'model.layers.2.linear_attn.dt_bias' (block 2, linear attention)"));
        CHECK(has(refused([](W& w) { w.drop_value("clef.decision.head_count").u32("clef.decision.head_count", 3); }),
                  "clef: the decision head's width 64 is not a multiple of its 3 heads"));
        CHECK(has(refused([](W& w) { w.drop_tensor("decision.proj_memory.weight"); }),
                  "clef: the file has no decision.proj_memory.weight, which gives the decision head's width"));
        CHECK(has(refused([](W& w) { w.drop_tensor("dec.blk.0.cross_attn_q.bias"); }),
                  "clef: the file has no tensor for 'joint_head.evidence_layers.0.attention.q_proj.bias'"));
        CHECK(has(refused([](W& w) { w.f32_tensor("dec.blk.4.attn_norm.weight", {64}, std::vector<float>(64)); }),
                  "clef: tensor 'dec.blk.4.attn_norm.weight' has no counterpart in the model"));
        // Fewer evidence layers than the file has: block 1 read as a decoder layer has a tensor it has no name for.
        CHECK(has(refused([](W& w) {
                      w.drop_value("clef.decision.routing_block_count").u32("clef.decision.routing_block_count", 1);
                  }),
                  "has no counterpart in the model"));
        CHECK(has(refused([](W& w) {
                      w.drop_tensor("decision.scales").f32_tensor("decision.scales", {4}, {1.0f, 1.0f, 0.5f, 0.0f});
                  }),
                  "clef: tensor 'decision.scales' has shape [4]; the metadata gives [3]"));
        CHECK(has(refused([](W& w) {
                      w.drop_tensor("dec.blk.0.ffn_up.weight")
                          .f32_tensor("dec.blk.0.ffn_up.weight", {64, 128}, std::vector<float>(64 * 128));
                  }),
                  "; the metadata gives [128]"));  // the MLP's width is read from this tensor

        // A Clef file has no MTP head of its own, nor one apart.
        xs::test::build_tiny_clef_gguf(work.path / "src", 11).write(work.path / "clef.gguf");
        CHECK_THROWS_WITH_AS(import_into(work.path / "clef.gguf", work.path / "mtp", work.path / "clef.gguf"),
                             doctest::Contains("clef: an MTP head in a file of its own (--mtp) is not supported"),
                             Error);
    }
}
