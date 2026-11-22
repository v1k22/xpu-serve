// A real inkling2 GGUF (XS_GGUF_FILE, e.g. unsloth/Inkling-2-Mini-GGUF's Inkling-2-Mini-UD-Q4_K_XL.gguf) against the
// Hugging Face checkpoint it was converted from (XS_INKLING2_DIR: its config.json and tokenizer.json; the weights are
// not read). The tokenizer written from the file passes the Inkling 2 parity cases and matches the checkpoint's token
// by token; config.json gives the model code what the checkpoint's does, but for the order of the expert banks' rows.
// With the drafter's file (XS_GGUF_DRAFTER_FILE, dspark-Inkling-2-Mini-Q8_0.gguf), the two files plan a pack with the
// checkpoint's mtp_config and every tensor of its mtp.safetensors; with the projector's (XS_GGUF_MMPROJ_FILE,
// mmproj-BF16.gguf), the checkpoint's vision_config and every tensor of its tower. Each case skips unless its
// variables are set.
// The weights are checked by the model parity test on the imported pack (model_parity.cpp), the lossless file's
// against Hugging Face fp32.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>

#include "convert/gguf/architecture.h"
#include "convert/real_gguf.h"
#include "io/gguf.h"
#include "model/tiny_checkpoints.h"
#include "models/drafters/dspark.h"
#include "models/inkling2/inkling2.h"
#include "tokenizer/parity_cases.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;

namespace {

std::optional<xs::test::RealGguf> real_file() {
    const auto real = xs::test::real_gguf("XS_INKLING2_DIR", "inkling2");
    if (real && !fs::exists(real->hf / "config.json")) return std::nullopt;  // the tokenizer alone: other tests'
    return real;
}

nlohmann::json checkpoint_config(const fs::path& dir) {
    return nlohmann::json::parse(std::ifstream(dir / "config.json"));
}

}  // namespace

TEST_SUITE("gguf real file") {
    TEST_CASE("tokenizer parity: a real inkling2 GGUF's tokenizer encodes and decodes as Hugging Face's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_inkling2_real_tokenizer_test");
        xs::test::write_gguf_tokenizer(gguf::File(real->gguf), work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        const auto original = tok::Tokenizer::load(real->hf);
        xs::test::check_tokenizer_cases(
            *imported, nlohmann::json::parse(std::ifstream(fs::path(XS_SOURCE_DIR) /
                                                           "tests/fixtures/tokenizer/inkling2_cases.json")));
        REQUIRE(imported->vocab_size() == original->vocab_size());
        int differ = 0;
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            const std::string text = original->token_text(id);
            const std::vector<int32_t> one{id};
            if (imported->token_text(id) != text || imported->is_special(id) != original->is_special(id) ||
                imported->encode(text) != original->encode(text) || imported->decode(one) != original->decode(one)) {
                if (++differ <= 10) MESSAGE("token " << id << " differs: '" << text << "'");
            }
        }
        MESSAGE(original->vocab_size() << " tokens compared, " << differ << " differ");
        CHECK(differ == 0);
        CHECK(imported->eos_id() == original->eos_id());
    }

    TEST_CASE("a real inkling2 GGUF's config.json reads as the checkpoint's") {
        const auto real = real_file();
        if (!real) return;
        const gguf::File file(real->gguf);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture("inkling2").plan(file);
        const inkling2::Config a = inkling2::parse_config(nlohmann::json::parse(plan.config.dump()));
        const inkling2::Config b = inkling2::parse_config(checkpoint_config(real->hf));
        CHECK(a.vocab_size == b.vocab_size);
        CHECK(a.output_size == b.output_size);
        CHECK(a.hidden_size == b.hidden_size);
        CHECK(a.num_layers == b.num_layers);
        CHECK(a.norm_eps == b.norm_eps);
        CHECK(a.max_position == b.max_position);
        CHECK(a.num_heads == b.num_heads);
        CHECK(a.num_kv_heads == b.num_kv_heads);
        CHECK(a.head_dim == b.head_dim);
        CHECK(a.sliding == b.sliding);
        CHECK(a.sliding_window == b.sliding_window);
        CHECK(a.rel_width == b.rel_width);
        CHECK(a.rel_extent == b.rel_extent);
        CHECK(static_cast<float>(a.qk_clip) == static_cast<float>(b.qk_clip));  // the file stores an f32
        CHECK(a.rel_clip == b.rel_clip);
        CHECK(static_cast<float>(a.log_scale_alpha) == static_cast<float>(b.log_scale_alpha));
        CHECK(a.log_scale_floor == b.log_scale_floor);
        CHECK(a.conv_kernel == b.conv_kernel);
        CHECK(a.block_layers == b.block_layers);
        CHECK(a.experts == b.experts);
        CHECK(a.experts_per_token == b.experts_per_token);
        CHECK(a.expert_width == b.expert_width);
        CHECK(a.route_scale == b.route_scale);
        CHECK(a.gate_softcap == b.gate_softcap);
        CHECK(a.up_softcap == b.up_softcap);
        CHECK(a.logit_softcap == b.logit_softcap);
        CHECK(a.loop.start == b.loop.start);
        CHECK(a.loop.end == b.loop.end);
        CHECK(a.loop.passes == b.loop.passes);
        CHECK(a.loop.gate_rank == b.loop.gate_rank);
        // The converter stores the gate rows of a fused bank, then its up rows; the checkpoint alternates them.
        CHECK(b.gate_up_alternates);
        CHECK_FALSE(a.gate_up_alternates);
    }

    TEST_CASE("a real inkling2 GGUF and its drafter's file plan the checkpoint's drafter") {
        const auto real = real_file();
        const char* d = std::getenv("XS_GGUF_DRAFTER_FILE");
        if (!real || d == nullptr || *d == '\0') return;
        const gguf::File file(real->gguf), drafter(d);
        const gguf_import::ImportPlan plan = gguf_import::find_architecture("inkling2").plan_with_mtp(file, drafter);
        const nlohmann::json hf = checkpoint_config(real->hf);
        const inkling2::Config target = inkling2::parse_config(hf);
        const auto parse = [&](const nlohmann::json& c) {
            return dspark::parse_config(c, target.hidden_size, target.applications());
        };
        const dspark::Config a = parse(nlohmann::json::parse(plan.config.at("mtp_config").dump()));
        const dspark::Config b = parse(hf.at("mtp_config"));
        CHECK(a.hidden_size == b.hidden_size);
        CHECK(a.intermediate_size == b.intermediate_size);
        CHECK(a.num_layers == b.num_layers);
        CHECK(a.num_heads == b.num_heads);
        CHECK(a.num_kv_heads == b.num_kv_heads);
        CHECK(a.head_dim == b.head_dim);
        CHECK(a.block_size == b.block_size);
        CHECK(a.markov_rank == b.markov_rank);
        CHECK(a.windows == b.windows);
        CHECK(a.target_ids == b.target_ids);
        CHECK(a.rms_norm_eps == b.rms_norm_eps);
        CHECK(a.rope_theta == b.rope_theta);
        // The 15 tensors of each drafter layer and the 9 beside them, under the pack's prefix.
        std::set<std::string> names;
        for (const gguf_import::PackTensor& t : plan.tensors) {
            if (t.name.rfind("mtp.", 0) == 0) names.insert(t.name);
        }
        CHECK(static_cast<int64_t>(names.size()) == 15 * a.num_layers + 9);
        for (const char* name : {"mtp.fuse_weight", "mtp.mask_embed", "mtp.markov_head.rnn_proj.weight",
                                 "mtp.layers.0.self_attn.k_proj_ctx.weight", "mtp.layers.2.dflare_norm.weight"}) {
            CAPTURE(name);
            CHECK(names.count(name) == 1);
        }
        // A model file is not a drafter's.
        CHECK_THROWS_WITH_AS(gguf_import::find_architecture("inkling2").plan_with_mtp(file, file),
                             doctest::Contains("is not Inkling 2's drafter"), xs::Error);
    }

    TEST_CASE("a real inkling2 GGUF and its projector file plan the checkpoint's vision tower") {
        const auto real = real_file();
        const char* m = std::getenv("XS_GGUF_MMPROJ_FILE");
        if (!real || m == nullptr || *m == '\0') return;
        const gguf::File file(real->gguf), projector(m);
        const gguf_import::Architecture& arch = gguf_import::find_architecture("inkling2");
        gguf_import::ImportPlan plan = arch.plan(file);
        arch.add_projector(plan, file, projector);
        nlohmann::json config = nlohmann::json::parse(plan.config.dump());
        const inkling2::VisionConfig a = inkling2::parse_vision_config(config);
        const inkling2::VisionConfig b = inkling2::parse_vision_config(checkpoint_config(real->hf));
        CHECK(a.hidden == b.hidden);
        CHECK(a.layers == b.layers);
        CHECK(a.heads == b.heads);
        CHECK(a.intermediate == b.intermediate);
        CHECK(a.patch == b.patch);
        CHECK(a.merge == b.merge);
        CHECK(a.grid == b.grid);
        CHECK(a.projection == b.projection);
        CHECK(a.out_hidden == b.out_hidden);
        CHECK(a.norm_eps == b.norm_eps);
        CHECK(a.rope_theta == b.rope_theta);
        // The 18 tensors of each layer and the 8 beside them, under the checkpoint's names.
        std::set<std::string> names;
        for (const gguf_import::PackTensor& t : plan.tensors) {
            if (t.name.rfind("model.visual.", 0) == 0) names.insert(t.name);
        }
        CHECK(static_cast<int64_t>(names.size()) == 18 * a.layers + 8);
        for (const char* name : {"model.visual.backbone.embeddings.patch_embedding.weight",
                                 "model.visual.backbone.embeddings.cls_position_embedding",
                                 "model.visual.backbone.layers.0.ls_1.gamma", "model.visual.proj_out.weight"}) {
            CAPTURE(name);
            CHECK(names.count(name) == 1);
        }
        // Neither a model file nor a drafter's is a vision tower's.
        gguf_import::ImportPlan again = arch.plan(file);
        CHECK_THROWS_WITH_AS(arch.add_projector(again, file, file),
                             doctest::Contains("is not Inkling 2's vision tower"), xs::Error);
    }
}
