// Block drafters (DFlash, DFlash 2, DSpine) on a tiny random Qwen3.5 written to disk: what the engine and `convert`
// rely on, checked without a reference (their math against the references is block_drafter_parity.cpp): a context
// fed in chunks or cut back after a verify is the same context, a causal block stops after the drafts asked for, a
// drafter converts to a pack of its own, and configs that are not implemented are refused by name.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>

#include "backend/backend_test_util.h"
#include "backend/cpu_ref/cpu_backend.h"
#include "convert/convert.h"
#include "engine/spec/block_drafter.h"
#include "models/drafters/dflash.h"
#include "models/model.h"
#include "tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;
using xs::test::TinyDrafter;
using xs::test::TinyQwen;

namespace {

const char* const kKinds[] = {"dflash", "dflash2", "dspine"};

// A tiny target and one drafter of each kind beside it.
struct Checkpoints {
    TempDir work;
    fs::path target;
    explicit Checkpoints(const std::string& name) : work(name), target(work.path / "target") {
        xs::test::write_tiny_qwen(target, TinyQwen{}, 3);
        for (const char* kind : kKinds) {
            TinyDrafter d;
            d.kind = kind;
            xs::test::write_tiny_drafter(work.path / kind, d, TinyQwen{}, 11);
        }
    }
    fs::path drafter(const std::string& kind) const { return work.path / kind; }
};

// The target with a drafter, compiled on cpu-ref with the speculation programs.
struct Loaded {
    std::unique_ptr<Model> model;
    std::unique_ptr<Session> session;
    const BlockDrafting* block = nullptr;

    Loaded(const fs::path& target, const fs::path& drafter, int64_t drafts = 0) {
        model = load_model(target, {.vision = false, .speculation = true, .drafter = drafter, .drafts = drafts});
        const Speculation* spec = model->speculation();
        REQUIRE(spec != nullptr);
        REQUIRE(spec->block() != nullptr);
        block = spec->block();
        SessionOptions options;
        options.precision = DType::F32;
        options.rewind_tokens = 16;
        const graph::Graph* programs[] = {&model->graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                          &spec->draft_graph()};
        session = make_cpu_ref_backend()->compile(programs, model->weights(), options);
    }

    void feed(size_t program, std::span<const int32_t> tokens) {
        TensorMap inputs, outputs;
        model->prepare_inputs(tokens, session->num_cached(spec::kVerify), {}, inputs);
        session->run(program, inputs, outputs);
    }

    // The block after the tokens fed so far and `pending`.
    TensorMap draft(int32_t pending) {
        const int64_t position = session->num_cached(spec::kDraft);
        TensorMap inputs, outputs;
        block->prepare_block(pending, position, inputs);
        session->run(spec::kDraft, inputs, outputs);
        session->truncate(spec::kDraft, position);
        CHECK(session->num_cached(spec::kDraft) == position);
        return outputs;
    }
};

void check_same(const TensorMap& a, const TensorMap& b) {
    REQUIRE(a.size() == b.size());
    for (const auto& [name, t] : a) {
        CAPTURE(name);
        const HostTensor& o = b.at(name);
        REQUIRE(t.shape() == o.shape());
        if (t.dtype() == DType::I32) {
            CHECK(std::equal(t.data<int32_t>(), t.data<int32_t>() + t.numel(), o.data<int32_t>()));
        } else {
            CHECK(test::all_close(t, test::to_f64(o), 1e-5, 1e-5));
        }
    }
}

const std::vector<int32_t> kPrompt{3, 17, 42, 99, 5, 250, 7, 8, 64, 128, 31, 77, 200, 12, 90};  // longer than the window

}  // namespace

TEST_SUITE("block drafters") {
    TEST_CASE("a block's outputs; a context fed in chunks, or cut back after a verify, is the same context") {
        const Checkpoints files("xs_block_drafters");
        for (const std::string kind : kKinds) {
            CAPTURE(kind);
            Loaded whole(files.target, files.drafter(kind));
            CHECK(whole.model->speculation()->drafter_name() == kind);
            CHECK(whole.model->speculation()->drafter_weights() == "drafter.");
            const int64_t n = whole.block->block_drafts();
            CHECK(n == 5);
            whole.feed(spec::kPrefill, kPrompt);
            CHECK(whole.session->num_cached(spec::kDraft) == static_cast<int64_t>(kPrompt.size()));
            const TensorMap out = whole.draft(9);
            if (kind == "dflash") {
                CHECK(out.at("drafts").shape() == Dims{n});
            } else {
                CHECK(out.at("candidates").shape() == Dims{n, 4});
                CHECK(out.at("scores").shape() == Dims{n, 4, 4});
                for (int64_t i = 0; i < n; ++i) {  // a position's candidates are distinct tokens
                    const int32_t* row = out.at("candidates").data<int32_t>() + i * 4;
                    CHECK(std::set<int32_t>(row, row + 4).size() == 4);
                }
            }
            const std::vector<int32_t> drafts = spec::block_drafts(out);
            CHECK(drafts.size() == static_cast<size_t>(n));
            for (const int32_t t : drafts) CHECK((t >= 0 && t < 256));
            check_same(out, whole.draft(9));  // drafting leaves no trace

            Loaded chunked(files.target, files.drafter(kind));
            chunked.feed(spec::kPrefill, std::span(kPrompt).first(8));
            chunked.feed(spec::kPrefill, std::span(kPrompt).subspan(8));
            check_same(out, chunked.draft(9));

            // a verify of six tokens of which one is kept, then the rest of the prompt
            Loaded verified(files.target, files.drafter(kind));
            verified.feed(spec::kPrefill, std::span(kPrompt).first(10));
            const std::vector<int32_t> step{kPrompt[10], 1, 2, 3, 4, 5};
            verified.feed(spec::kVerify, step);
            verified.session->truncate(spec::kVerify, 11);
            verified.feed(spec::kVerify, std::span(kPrompt).subspan(11));
            check_same(out, verified.draft(9));

            Loaded other(files.target, files.drafter(kind));  // planted: another context is another block
            other.feed(spec::kPrefill, std::span(kPrompt).first(14));
            const TensorMap moved = other.draft(9);
            const char* key = kind == "dflash" ? "drafts" : "scores";
            CHECK(std::memcmp(moved.at(key).data<char>(), out.at(key).data<char>(), out.at(key).nbytes()) != 0);
        }
    }

    TEST_CASE("a causal block stops after the drafts asked for; a block that sees itself whole runs all its rows") {
        const Checkpoints files("xs_block_drafters_rows");
        Loaded whole(files.target, files.drafter("dspine"));
        Loaded few(files.target, files.drafter("dspine"), 2);
        CHECK(few.block->block_drafts() == 2);
        whole.feed(spec::kPrefill, kPrompt);
        few.feed(spec::kPrefill, kPrompt);
        const TensorMap all = whole.draft(9);
        const TensorMap two = few.draft(9);
        REQUIRE(two.at("candidates").shape() == Dims{2, 4});
        CHECK(std::equal(two.at("candidates").data<int32_t>(), two.at("candidates").data<int32_t>() + 8,
                         all.at("candidates").data<int32_t>()));
        for (int64_t i = 0; i < 2 * 4 * 4; ++i) {
            CHECK(two.at("scores").data<float>()[i] == doctest::Approx(all.at("scores").data<float>()[i]).epsilon(1e-5));
        }
        const std::vector<int32_t> a = spec::block_drafts(all);
        CHECK(spec::block_drafts(two) == std::vector<int32_t>(a.begin(), a.begin() + 2));
        CHECK(Loaded(files.target, files.drafter("dspine"), 40).block->block_drafts() == 5);  // no more than its block
        for (const char* kind : {"dflash", "dflash2"}) CHECK(Loaded(files.target, files.drafter(kind), 2).block->block_drafts() == 5);
    }

    TEST_CASE("convert writes a drafter's pack: its matmul weights and tables quantized, close to the original") {
        const Checkpoints files("xs_block_drafters_convert");
        for (const std::string kind : {"dflash2", "dspine"}) {
            CAPTURE(kind);
            Loaded original(files.target, files.drafter(kind));
            original.feed(spec::kPrefill, kPrompt);
            const TensorMap want = original.draft(9);

            ConvertOptions o;
            o.source = files.drafter(kind);
            o.target = files.target;
            o.output = files.work.path / (kind + "-q8");
            o.format = DType::Q8;
            o.group = 32;
            o.min_elements = 0;
            const ConvertReport r = convert_model(o);
            CHECK(r.q4 == 0);
            CHECK(r.q8 > 14);  // fc and seven matmuls a layer, at least
            CHECK(r.output_bytes < r.source_bytes);
            CHECK(fs::exists(o.output / "config.json"));
            const auto manifest = nlohmann::json::parse(std::ifstream(o.output / kPackManifest));
            CHECK(manifest.at("source").at("drafter") == kind);
            CHECK(manifest.at("source").at("name") == kind);

            const auto pack = SafetensorsWeights::open_dir(o.output);
            CHECK(pack->get("fc.weight").dtype == DType::Q8);
            CHECK(pack->get("layers.1.mlp.down_proj.weight").dtype == DType::Q8);
            CHECK(pack->get("layers.0.self_attn.k_proj.weight").dtype == DType::Q8);
            CHECK(pack->get("norm.weight").dtype == DType::BF16);
            CHECK_FALSE(pack->contains("model.embed_tokens.weight"));  // the target's stay the target's
            if (kind == "dspine") {
                CHECK(pack->get("selected_prefix_bridge.codebook").dtype == DType::Q8);
                // the stacks are read one matrix at a time and stay as they are
                CHECK(pack->get("selected_prefix_bridge.feature_write_weights").dtype == DType::BF16);
                CHECK(pack->get("selected_prefix_bridge.early_read_weights").shape == Dims{1, 32, 128});
            } else {
                CHECK(pack->get("candidate_selector.successor_codebook").dtype == DType::Q8);
                CHECK(pack->get("layers.0.mlp_conv.kernel_projection.weight").dtype == DType::Q8);
                CHECK(pack->get("layers.0.mlp_conv.base_kernel").dtype == DType::BF16);
            }

            Loaded packed(files.target, o.output);
            packed.feed(spec::kPrefill, kPrompt);
            const TensorMap got = packed.draft(9);
            // Scores over the candidates both list (a near tie at the edge of the list may flip).
            const HostTensor& c0 = want.at("candidates");
            const HostTensor& c1 = got.at("candidates");
            const int64_t k = c0.dim(1);
            size_t shared = 0;
            double dot = 0, a2 = 0, b2 = 0;
            for (int64_t i = 0; i < c0.dim(0); ++i) {
                for (int64_t a = 0; a < k; ++a) {
                    const int32_t* row = c1.data<int32_t>() + i * k;
                    const auto b = std::find(row, row + k, c0.data<int32_t>()[i * k + a]) - row;
                    if (b == k) continue;
                    ++shared;
                    const double x = want.at("scores").data<float>()[(i * k) * k + a];  // sender 0
                    const double y = got.at("scores").data<float>()[(i * k) * k + b];
                    if (i > 0 && c0.data<int32_t>()[(i - 1) * k] != c1.data<int32_t>()[(i - 1) * k]) continue;
                    dot += x * y;
                    a2 += x * x;
                    b2 += y * y;
                }
            }
            MESSAGE(kind, " q8 pack: ", shared, " of ", c0.numel(), " candidates shared, scores cosine ",
                    dot / std::sqrt(a2 * b2));
            CHECK(shared * 4 >= static_cast<size_t>(c0.numel()) * 3);
            CHECK(dot / std::sqrt(a2 * b2) > 0.99);
        }
    }

    TEST_CASE("convert and load refuse a drafter without its target, another model's drafter, and AWQ") {
        const Checkpoints files("xs_block_drafters_refuse");
        ConvertOptions o;
        o.source = files.drafter("dspine");
        o.output = files.work.path / "out";
        CHECK_THROWS_WITH_AS(convert_model(o), doctest::Contains("block drafter"), xs::Error);
        CHECK_THROWS_WITH_AS(load_model(files.drafter("dflash2")), doctest::Contains("block drafter"), xs::Error);
        o.target = files.target;
        o.method = QuantMethod::Awq;
        CHECK_THROWS_WITH_AS(convert_model(o), doctest::Contains("AWQ"), xs::Error);

        TinyQwen wider;
        wider.hidden = 160;
        xs::test::write_tiny_drafter(files.work.path / "wider", TinyDrafter{}, wider, 5);
        CHECK_THROWS_WITH_AS(
            load_model(files.target, {.vision = false, .speculation = true, .drafter = files.work.path / "wider"}),
            doctest::Contains("trained for a target"), xs::Error);
    }

    TEST_CASE("a drafter config: variants, defaults, and refusals by name") {
        const Checkpoints files("xs_block_drafters_config");
        const auto read = [&](const std::string& kind) {
            return nlohmann::json::parse(std::ifstream(files.drafter(kind) / "config.json"));
        };
        const dflash::Config plain = dflash::parse_config(read("dflash"));
        CHECK(plain.variant == dflash::Variant::DFlash);
        CHECK_FALSE(plain.causal);
        CHECK(plain.sliding_window == 0);
        CHECK(plain.block_size == 6);
        CHECK(plain.target_layer_ids == std::vector<int64_t>{0, 2, 3});
        const dflash::Config two = dflash::parse_config(read("dflash2"));
        CHECK(two.variant == dflash::Variant::DFlash2);
        CHECK_FALSE(two.causal);
        CHECK(two.sliding_window == 12);
        CHECK(two.conv_taps == 2);
        CHECK(two.top_k == 4);
        const dflash::Config spine = dflash::parse_config(read("dspine"));
        CHECK(spine.variant == dflash::Variant::DSpine);
        CHECK(spine.causal);
        CHECK(spine.code_size == 32);
        CHECK(spine.bridge_rank == 16);

        nlohmann::json j = read("dflash");
        j["dflash_config"].erase("target_layer_ids");  // z-lab's default spread over the target's layers
        j["num_target_layers"] = 64;
        j["num_hidden_layers"] = 5;
        j["layer_types"] = std::vector<std::string>(5, "full_attention");
        CHECK(dflash::parse_config(j).target_layer_ids == std::vector<int64_t>{1, 16, 31, 46, 61});

        const auto refused = [](nlohmann::json config, const char* what) {
            CHECK_THROWS_WITH_AS(dflash::parse_config(config), doctest::Contains(what), xs::Error);
        };
        j = read("dflash2");
        j["dflash_config"]["projector_type"] = "mlp";
        refused(j, "projector_type");
        j = read("dflash2");
        j["architectures"] = {"DSparkDraftModel"};
        refused(j, "DSparkDraftModel");
        j = read("dflash2");
        j["sliding_window"] = 4;
        refused(j, "window");
        j = read("dspine");
        j["dflash_config"]["selected_prefix_terminal"] = true;
        refused(j, "terminal");
        j = read("dspine");
        j["dflash_config"]["selected_prefix_feature_layers"] = {2};
        refused(j, "every layer");
        j = read("dspine");
        j["dflash_config"]["selected_prefix_last_write_refine"]["kind"] = "soft_sender";
        refused(j, "soft_sender");
        j = read("dflash");
        j["final_logit_softcapping"] = 30.0;
        refused(j, "final_logit_softcapping");
    }
}
