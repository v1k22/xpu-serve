// A real modern-bert decision GGUF (XS_GGUF_FILE: ggml-org/Laya-GGUF's or ggml-org/Julia-1-GGUF's) against the
// checkpoint it was converted from (XS_LAYA_DIR: convaiinnovations/laya; XS_JULIA_DIR: SupersonicLabs/Julia-1; the one
// the file is a conversion of, by the model the import identifies). The tokenizer written from the file passes the
// checkpoint's tokenizer parity cases and matches it token by token over the whole vocabulary; the configuration the
// import writes is the checkpoint's; the pack reads every request as the checkpoint does, to the token, and answers
// within the file's rounding. Each case skips unless the variables are set and the file is a modern-bert one.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "convert/gguf/architecture.h"
#include "convert/import_test_util.h"
#include "convert/real_gguf.h"
#include "decision/runner.h"
#include "io/gguf.h"
#include "model/tiny_checkpoints.h"
#include "models/config_util.h"
#include "models/laya/laya.h"
#include "server/decision_handlers.h"
#include "tokenizer/parity_cases.h"
#include "tokenizer/tokenizer.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::TempDir;

namespace {

struct RealDecisionGguf {
    fs::path gguf;
    fs::path checkpoint;
    bool julia = false;
    std::string fixture_key;  // the tokenizer parity cases of the checkpoint's tokenizer
};

// XS_GGUF_FILE, a modern-bert file, with the checkpoint it stands for.
std::optional<RealDecisionGguf> real_file() {
    const char* g = std::getenv("XS_GGUF_FILE");
    if (g == nullptr || *g == '\0') return std::nullopt;
    const gguf::File file(g);
    if (file.get_string("general.architecture") != "modern-bert") return std::nullopt;
    const bool julia = gguf_import::modern_bert().plan(file).config.contains("architecture");
    const char* dir = std::getenv(julia ? "XS_JULIA_DIR" : "XS_LAYA_DIR");
    if (dir == nullptr || *dir == '\0') return std::nullopt;
    return RealDecisionGguf{g, dir, julia, julia ? "julia/tokenizer" : "laya/tokenizer"};
}

// Each question's largest probability difference between two answers to one request, and how many of their most
// likely options differ.
struct Gaps {
    std::vector<double> per_question;
    int top_differs = 0;
};
void add_gaps(const DecisionResult& a, const DecisionResult& b, Gaps& gaps) {
    for (size_t q = 0; q < a.answers.size(); ++q) {
        double gap = 0;
        for (size_t o = 0; o < a.answers[q].probabilities.size(); ++o)
            gap = std::max(gap, std::abs(a.answers[q].probabilities[o] - b.answers[q].probabilities[o]));
        gaps.per_question.push_back(gap);
        gaps.top_differs += a.answers[q].best != b.answers[q].best;
    }
}

}  // namespace

TEST_SUITE("gguf real file") {
    TEST_CASE("tokenizer parity: a real modern-bert GGUF's tokenizer encodes and decodes as the checkpoint's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_modern_bert_real_tokenizer_test");
        const gguf::File file(real->gguf);
        xs::test::write_gguf_tokenizer(file, work.path);
        const auto imported = tok::Tokenizer::load(work.path);
        const auto original = tok::Tokenizer::load(real->checkpoint / "tokenizer");

        const nlohmann::json all = nlohmann::json::parse(
            std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/fixtures/tokenizer/laya_julia_cases.json"));
        nlohmann::json fixture = xs::test::cases_of(all["tokenizers_by_folder"], real->fixture_key);
        // ModernBERT's tokenizer_config.json names no bos or eos token; the file names its [CLS] and [SEP] so.
        for (const auto& [key, id] :
             {std::pair{"bos_id", imported->bos_id()}, std::pair{"eos_id", imported->eos_id()}}) {
            if (fixture[key].is_null() && id) {
                MESSAGE(std::string(key) << ": the file names " << *id << " ('" << imported->token_text(*id) << "')");
                fixture[key] = *id;
            }
        }
        MESSAGE("parity cases: " << fixture["encode_cases"].size() << " encode, " << fixture["decode_cases"].size()
                                 << " decode");
        xs::test::check_special_token_cases(*imported, fixture);

        // Token by token against the checkpoint's tokenizer: the same text, special flag, encoding of the text and
        // decoding for every id, and the same text after two spaces (the mask token takes them; a run of spaces, not).
        REQUIRE(imported->vocab_size() == original->vocab_size());
        int differ = 0;
        for (int32_t id = 0; id < original->vocab_size(); ++id) {
            const std::string text = original->token_text(id);
            const std::vector<int32_t> one{id};
            if (imported->token_text(id) != text || imported->is_special(id) != original->is_special(id) ||
                imported->encode(text) != original->encode(text) ||
                imported->encode("a  " + text) != original->encode("a  " + text) ||
                imported->decode(one) != original->decode(one)) {
                if (++differ <= 10) MESSAGE("token " << id << " differs: '" << text << "'");
            }
        }
        MESSAGE(original->vocab_size() << " tokens compared, " << differ << " differ");
        CHECK(differ == 0);

        // The parts the file names only by tokenizer.ggml.pre are the checkpoint's.
        const nlohmann::json written = nlohmann::json::parse(std::ifstream(work.path / "tokenizer.json"));
        const nlohmann::json hf = nlohmann::json::parse(std::ifstream(real->checkpoint / "tokenizer/tokenizer.json"));
        for (const char* part : {"normalizer", "pre_tokenizer", "post_processor", "decoder"}) {
            CAPTURE(part);
            CHECK(written[part] == hf[part]);
        }
        nlohmann::json settings = hf["model"], written_settings = written["model"];
        CHECK(written_settings["merges"].size() == settings["merges"].size());
        for (nlohmann::json* m : {&settings, &written_settings}) {
            m->erase("vocab");
            m->erase("merges");
        }
        CHECK(written_settings == settings);
        // The special tokens the model assembles its sequences from.
        const nlohmann::json config = nlohmann::json::parse(std::ifstream(work.path / "tokenizer_config.json"));
        const nlohmann::json hf_config = cfg::read_json(real->checkpoint / "tokenizer/tokenizer_config.json");
        for (const char* token : {"cls_token", "sep_token", "mask_token", "pad_token", "unk_token"})
            CHECK(config[token] == hf_config[token]);
    }

    TEST_CASE("config: what the model code reads from a real modern-bert GGUF is the checkpoint's") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_modern_bert_real_config_test");
        xs::test::import_into(real->gguf, work.path / "pack");
        const laya::Config a = laya::read_config(work.path / "pack");
        const laya::Config b = laya::read_config(real->checkpoint);
        CHECK(a.policy.family == b.policy.family);
        CHECK(a.policy.max_tokens == b.policy.max_tokens);
        CHECK(a.policy.head_tokens == b.policy.head_tokens);
        CHECK(a.policy.strict == b.policy.strict);
        CHECK(a.policy.descriptions_only == b.policy.descriptions_only);
        CHECK(a.policy.temperature == b.policy.temperature);
        CHECK(a.policy.temperature_by_options == b.policy.temperature_by_options);
        CHECK(a.head_layers == b.head_layers);
        CHECK(a.cls == b.cls);
        CHECK(a.sep == b.sep);
        CHECK(a.mask == b.mask);
        const modern_bert::Config& x = a.encoder;
        const modern_bert::Config& y = b.encoder;
        CHECK(x.hidden_size == y.hidden_size);
        CHECK(x.num_layers == y.num_layers);
        CHECK(x.num_heads == y.num_heads);
        CHECK(x.intermediate_size == y.intermediate_size);
        CHECK(x.vocab_size == y.vocab_size);
        CHECK(x.max_positions == y.max_positions);
        CHECK(x.norm_eps == y.norm_eps);
        CHECK(x.local_attention == y.local_attention);
        CHECK(x.global_rope_theta == y.global_rope_theta);
        CHECK(x.local_rope_theta == y.local_rope_theta);
        CHECK(x.local == y.local);
    }

    TEST_CASE("decisions: the import of a real modern-bert GGUF reads requests as its checkpoint and answers alike") {
        const auto real = real_file();
        if (!real) return;
        const TempDir work("xs_gguf_modern_bert_real_answers_test");
        xs::test::import_into(real->gguf, work.path / "pack");
        DecisionRunnerOptions o;
        o.backend = "cpu-ref";
        o.session.precision = DType::F32;
        const auto pack = DecisionRunner::create(work.path / "pack", o);
        const auto original = DecisionRunner::create(real->checkpoint, o);
        nlohmann::ordered_json requests = nlohmann::ordered_json::array();
        for (const char* file : {"requests.json", "requests_laya_julia.json"}) {
            const nlohmann::ordered_json set =
                nlohmann::ordered_json::parse(std::ifstream(fs::path(XS_SOURCE_DIR) / "tests/parity/decision" / file));
            for (const auto& r : set["requests"]) {
                if (!r["state"].is_string() || r["state"].get<std::string>().size() < 20000) requests.push_back(r);
            }
        }
        REQUIRE(requests.size() > 20);
        // A file's weights are the checkpoint's rounded (bf16 from f16 or f32 drops three or more bits; Q8_0 keeps 8):
        // the answers move, most on questions the model is unsure of.
        const bool q8 = gguf::File(real->gguf).get_int("general.file_type") == 7;
        Gaps gaps;
        int answered = 0, refused = 0;
        for (const auto& r : requests) {
            const DecisionRequest request = server::parse_systemone_request(r);
            std::optional<DecisionResult> theirs;
            try {
                theirs = original->decide(request);
            } catch (const DecisionRefused&) {
                CHECK_THROWS_AS(pack->decide(request), DecisionRefused);
                ++refused;
                continue;
            }
            const DecisionResult ours = pack->decide(request);
            CHECK(ours.sequences == theirs->sequences);
            const size_t first = gaps.per_question.size();
            add_gaps(ours, *theirs, gaps);
            for (size_t q = 0; q < request.questions.size(); ++q) {
                if (gaps.per_question[first + q] > 0.1) {
                    std::string shown;
                    for (size_t i = 0; i < ours.answers[q].probabilities.size(); ++i) {
                        shown += str_cat(" ", request.questions[q].options[i].id, " ",
                                         ours.answers[q].probabilities[i], " (", theirs->answers[q].probabilities[i], ")");
                    }
                    MESSAGE("question '" << request.questions[q].id << "', the pack's (the checkpoint's):" << shown);
                }
            }
            ++answered;
        }
        std::vector<double> sorted = gaps.per_question;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2], worst = sorted.back();
        MESSAGE(real->gguf.filename().string() << ": " << answered << " requests answered, " << refused
                                               << " refused alike; " << sorted.size() << " questions, probability "
                                               << "difference median " << median << ", largest " << worst
                                               << "; most likely option differs on " << gaps.top_differs);
        // A mapping gone wrong moves every answer; rounding moves the few the model is unsure of (Julia-1's Q8_0 file
        // moves a noul whose state spells <bos>, <eos> and <pad> by 0.24).
        CHECK(median < (q8 ? 0.03 : 0.01));
        CHECK(worst < (q8 ? 0.3 : 0.1));
        CHECK(gaps.top_differs <= static_cast<int>(sorted.size()) / 10);
    }
}
