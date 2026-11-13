// GGUF import of the decision models on ModernBERT, hermetically: a tiny modern-bert GGUF of Laya and of Julia-1,
// written the way llama.cpp's converter writes one (tiny_modern_bert_gguf.h: the head's layers as blocks after the
// encoder's, the scorer, the question types' table, Q8_0 / F16 / BF16 / F32 tensors, Laya's temperatures, the prompt
// template the converter writes for each, ModernBERT's and mmBERT's vocabularies in miniature) imports as the
// checkpoint its blocks decode to, laid out as the original: the same configuration, tokenizer, token sequences and
// answers as its twin through the decision runtime on cpu-ref; and what the importer does not implement is refused by
// name.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/gguf/gguf_import.h"
#include "convert/import_test_util.h"
#include "convert/tiny_modern_bert_gguf.h"
#include "decision/runner.h"
#include "models/laya/laya.h"
#include "models/policy.h"
#include "tokenizer/tokenizer.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::has;
using xs::test::import_into;
using xs::test::TempDir;

namespace {

template <typename Change>
std::string refusal(const fs::path& dir, bool julia, Change change) {
    return xs::test::refusal(
        dir, [julia](const fs::path& src) { return xs::test::build_tiny_modern_bert_gguf(src, julia, 31); }, change);
}

// Requests both models take: every question type, the state as text and as JSON, a long state, text that spells the
// other tokenizer's special tokens.
std::vector<DecisionRequest> requests() {
    using xs::test::choice;
    using xs::test::noul;
    using xs::test::score;
    DecisionRequest a;
    a.state = "the cat sat on the mat, then the dog sat on it.";
    a.questions = {choice("who", "who sat on the mat?", {{"cat", "the cat"}, {"dog", "the dog"}, {"rat", "a rat"}}),
                   noul("late", "is the cat late?"), score("rating", "rate the mat", {"bad", "fair", "good"})};
    DecisionRequest b;
    b.state = Json{{"order", {{"items", 3}, {"total", 12.5}}}, {"note", "ich habe [CLS] zweimal bezahlt"}};
    b.questions = {choice("route", "which team handles it?", {{"billing", "charges"}, {"shipping", "delivery"}})};
    DecisionRequest c;
    std::string text;
    for (int i = 0; i < 60; ++i)
        text += "the rat ran in the net and the cat sat on the hat. ";
    c.state = text;
    c.questions = {noul("rat", "did the rat run?")};
    return {a, b, c};
}

}  // namespace

TEST_SUITE("gguf import modern-bert") {
    TEST_CASE("a tiny modern-bert GGUF of Laya or Julia-1 imports as the checkpoint its blocks decode to") {
        for (const bool julia : {false, true}) {
            CAPTURE(julia);
            const TempDir work(julia ? "xs_gguf_julia_import_test" : "xs_gguf_laya_import_test");
            xs::test::build_tiny_modern_bert_gguf(work.path, julia, 31).write(work.path / "tiny.gguf");
            const gguf_import::ImportReport report = import_into(work.path / "tiny.gguf", work.path / "pack");
            CHECK(report.architecture == "modern-bert");
            CHECK(model_kind(work.path / "pack") == ModelKind::Decision);
            CHECK_FALSE(fs::exists(work.path / "pack" / "chat_template.jinja"));

            // The original's layout and configuration.
            const laya::Config a = laya::read_config(work.path / "pack");
            const laya::Config b = laya::read_config(work.path / "twin");
            CHECK(a.policy.family == (julia ? "julia" : "laya"));
            CHECK(a.policy.max_tokens == b.policy.max_tokens);
            CHECK(a.policy.head_tokens == b.policy.head_tokens);
            CHECK(a.policy.strict == b.policy.strict);
            CHECK(a.policy.temperature == b.policy.temperature);
            CHECK(a.policy.temperature_by_options == b.policy.temperature_by_options);
            CHECK(a.head_layers == b.head_layers);
            CHECK(std::tie(a.cls, a.sep, a.mask) == std::tie(b.cls, b.sep, b.mask));
            CHECK(a.encoder.local == b.encoder.local);
            CHECK(a.encoder.local_attention == b.encoder.local_attention);
            CHECK(a.encoder.global_rope_theta == b.encoder.global_rope_theta);
            CHECK(a.encoder.local_rope_theta == b.encoder.local_rope_theta);
            CHECK(a.encoder.norm_eps == b.encoder.norm_eps);
            CHECK(a.encoder.max_positions == b.encoder.max_positions);

            // The tokenizer: the original's, token by token and on text.
            const auto imported = tok::Tokenizer::load(work.path / "pack" / "tokenizer");
            const auto original = tok::Tokenizer::load(work.path / "hf" / "tokenizer");
            REQUIRE(imported->vocab_size() == original->vocab_size());
            for (int32_t id = 0; id < original->vocab_size(); ++id) {
                CAPTURE(id);
                const std::string t = original->token_text(id);
                CHECK(imported->token_text(id) == t);
                CHECK(imported->is_special(id) == original->is_special(id));
                CHECK(imported->encode("a  " + t) == original->encode("a  " + t));
                CHECK(imported->decode(std::vector<int32_t>{id}) == original->decode(std::vector<int32_t>{id}));
            }
            for (const std::string text : {"the cat sat  on the  mat\n\nin ther", "x [MASK] y  [MASK]", "a  <mask> b",
                                           "ich \xc3\xbc\xe2\x80\x94 \xe6\x97\xa5"}) {
                CAPTURE(text);
                CHECK(imported->encode(text) == original->encode(text));
                CHECK(imported->with_special_tokens(imported->encode(text)) ==
                      original->with_special_tokens(original->encode(text)));
            }

            // The same sequences and answers as the twin, through the decision runtime.
            DecisionRunnerOptions o;
            o.backend = "cpu-ref";
            o.session.precision = DType::F32;
            const auto pack = DecisionRunner::create(work.path / "pack", o);
            const auto twin = DecisionRunner::create(work.path / "twin", o);
            for (const DecisionRequest& r : requests()) {
                const DecisionResult ours = pack->decide(r), theirs = twin->decide(r);
                CHECK(ours.sequences == theirs.sequences);
                for (size_t q = 0; q < ours.answers.size(); ++q) {
                    for (size_t i = 0; i < ours.answers[q].probabilities.size(); ++i) {
                        CHECK(ours.answers[q].probabilities[i] ==
                              doctest::Approx(theirs.answers[q].probabilities[i]).epsilon(1e-6));
                    }
                }
            }
        }
    }

    TEST_CASE("modern-bert: what the importer does not implement is refused by name") {
        const TempDir work("xs_gguf_modern_bert_refusals_test");
        const auto with = [](std::string key, std::string value) {
            return [key, value](GgufWriter& w) { w.drop_value(key).string(key, value); };
        };
        CHECK(has(refusal(work.path, false, with("modern-bert.decision.type", "lfm2-d1")),
                  "modern-bert: decision.type 'lfm2-d1' is not supported (decision.type 'laya' is: Laya, Julia-1)"));
        CHECK(has(refusal(work.path, false, [](GgufWriter& w) { w.drop_value("modern-bert.decision.type"); }),
                  "a file without decision.type (an encoder or embedding model)"));
        // A prompt template neither model's: another model's file.
        CHECK(has(refusal(work.path, false, with("tokenizer.chat_template.systemone", "[CLS]{{ state }}[SEP]")),
                  "is neither Laya's nor Julia-1's"));
        // Laya's state budget goes with its published question budgets only.
        CHECK(has(refusal(work.path, false,
                          [](GgufWriter& w) {
                              w.drop_value("modern-bert.decision.max_head_tokens")
                                  .u32("modern-bert.decision.max_head_tokens", 300);
                          }),
                  "no published Laya pairs one with that question budget"));
        CHECK(has(refusal(work.path, true,
                          [](GgufWriter& w) { w.f32("modern-bert.decision.temperature.choice", 2.0f); }),
                  "modern-bert.decision.temperature.choice: Julia-1 has no temperature"));
        CHECK(has(refusal(work.path, false, [](GgufWriter& w) { w.u32("modern-bert.decision.head_count", 4); }),
                  "metadata 'modern-bert.decision.head_count' is not read by this importer"));
        CHECK(has(refusal(work.path, false,
                          [](GgufWriter& w) {
                              w.drop_value("modern-bert.attention.causal")
                                  .boolean("modern-bert.attention.causal", true);
                          }),
                  "causal attention is not ModernBERT's"));
        CHECK(has(refusal(work.path, false, with("modern-bert.hidden_activation", "silu")),
                  "modern-bert.hidden_activation 'silu' is not supported"));
        CHECK(has(refusal(work.path, false,
                          [](GgufWriter& w) {
                              w.drop_value("modern-bert.feed_forward_length")
                                  .i32s("modern-bert.feed_forward_length", {48, 48, 48, 48, 512, 256});
                          }),
                  "block 5 has a feed-forward width of 256"));
        CHECK(has(refusal(work.path, false,
                          [](GgufWriter& w) {
                              w.drop_value("tokenizer.ggml.token_type_count").u32("tokenizer.ggml.token_type_count", 2);
                          }),
                  "tokenizer.ggml.token_type_count must be 3"));
        CHECK(has(refusal(work.path, false,
                          [](GgufWriter& w) {
                              w.f32_tensor("blk.0.attn_norm.weight", {128}, std::vector<float>(128, 1.0f));
                          }),
                  "tensor 'blk.0.attn_norm.weight' has no counterpart in block 0 (an encoder layer)"));
        CHECK(has(refusal(work.path, false, [](GgufWriter& w) { w.drop_tensor("cls.bias"); }),
                  "the file has no tensor for 'scorer.1.bias'"));
        CHECK(has(refusal(work.path, true, with("tokenizer.ggml.metaspace.prepend_scheme", "first")),
                  "tokenizer.ggml.metaspace.prepend_scheme 'first' disagrees with the tokenizer 'mmbert'"));
    }
}
