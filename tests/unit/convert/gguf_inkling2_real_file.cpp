// A real inkling2 GGUF (XS_GGUF_FILE, e.g. unsloth/Inkling-2-Mini-GGUF's Inkling-2-Mini-UD-Q4_K_XL.gguf) against the
// Hugging Face checkpoint it was converted from (XS_INKLING2_DIR: its config.json and tokenizer.json; the weights are
// not read). The tokenizer written from the file passes the Inkling 2 parity cases and matches the checkpoint's token
// by token; config.json gives the model code what the checkpoint's does, but for the order of the expert banks' rows.
// Each case skips unless both variables are set.
// The weights are checked by the model parity test on the imported pack (model_parity.cpp), the lossless file's
// against Hugging Face fp32.
#include <doctest/doctest.h>

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
}
