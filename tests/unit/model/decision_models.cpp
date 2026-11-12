// Decision models on tiny checkpoints the test writes: what must hold whatever the weights are. The numbers
// themselves are checked against each model's reference code in decision_parity.cpp.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>

#include "../backend/device_test_support.h"
#include "convert/convert.h"
#include "decision_test_util.h"
#include "models/clef/clef.h"
#include "models/letter_readout/letter_readout.h"
#include "models/model.h"
#include "models/policy.h"
#include "models/pplx_decider/pplx_decider.h"

namespace {

using xs::DecisionLimits;
using xs::DecisionRequest;
using xs::Json;
using xs::test::choice;
using xs::test::noul;
using xs::test::score;

DecisionRequest ticket() {
    DecisionRequest r;
    r.state = Json{{"ticket", 4411}, {"text", "cannot log in after the reset email expired"}, {"vip", true}};
    r.questions = {choice("team", "Which team?", {{"billing", "payments"}, {"access", "logins"}, {"other", nullptr}}),
                   noul("urgent", "Is it urgent?"), score("effort", "How much work?", {"minutes", "hours", "days"})};
    return r;
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

}  // namespace

TEST_SUITE("decision models") {
    TEST_CASE("clef: a checkpoint with a joint schema head is a decision model") {
        const xs::test::TempDir dir("xs_tiny_clef");
        xs::test::write_tiny_clef(dir.path, 5);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Decision);
        const auto model = xs::load_decision_model(dir.path);
        CHECK(model->spec().arch == "clef");
        REQUIRE(model->programs().size() == 4);
        const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                  xs::SessionOptions{.precision = xs::DType::F32});
        const DecisionLimits whole{4096, 4096};

        int steps = 0;
        const auto p = xs::test::decide(*model, *session, ticket(), whole, &steps);
        CHECK(steps == 4);  // the prompt in one step, then the options, the questions, the scores
        REQUIRE(p.size() == 3);
        for (const auto& question : p) {
            double total = 0;
            for (const double v : question)
                total += v;
            CHECK(total == doctest::Approx(1.0));
        }
        // Answers must depend on the weights enough for the checks below to mean something.
        CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

        SUBCASE("the prompt fed in steps gives the same answers") {
            int chunked_steps = 0;
            const auto chunked = xs::test::decide(*model, *session, ticket(), DecisionLimits{4096, 7}, &chunked_steps);
            CHECK(chunked_steps > 20);
            CHECK(max_diff(p, chunked) < 1e-5);
        }
        SUBCASE("a choice's options are read sorted by name: the request's order only moves the answers") {
            DecisionRequest swapped = ticket();
            std::swap(swapped.questions[0].options[0], swapped.questions[0].options[2]);
            const auto q = xs::test::decide(*model, *session, swapped, whole);
            CHECK(q[0][0] == doctest::Approx(p[0][2]).epsilon(1e-6));
            CHECK(q[0][2] == doctest::Approx(p[0][0]).epsilon(1e-6));
            CHECK(max_diff({p[1], p[2]}, {q[1], q[2]}) < 1e-9);
        }
        SUBCASE("questions are decided jointly: another question in the request moves an answer") {
            DecisionRequest alone = ticket();
            alone.questions.resize(1);
            CHECK(max_diff({p[0]}, xs::test::decide(*model, *session, alone, whole)) > 1e-4);
        }
        SUBCASE("a noul outcome described as null is not the outcome left out") {
            const xs::test::ByteTools tools;
            DecisionRequest described = ticket();
            described.questions[1].options[1].description = Json(nullptr);
            const auto a = model->start(ticket(), tools, whole)->sequences();
            const auto b = model->start(described, tools, whole)->sequences();
            CHECK(a[0].size() > b[0].size());  // the model's own wording of "false" is gone
        }
        SUBCASE("a request longer than the limit is refused, not cut") {
            try {
                xs::test::decide(*model, *session, ticket(), DecisionLimits{200, 200});
                FAIL("expected a refusal");
            } catch (const xs::DecisionRefused& e) {
                CHECK(e.code() == "max_tokens_exceeded");
                CHECK(std::string(e.what()).find("the limit is 200") != std::string::npos);
            }
            CHECK(max_diff(p, xs::test::decide(*model, *session, ticket(), whole)) < 1e-9);  // and the next one runs
        }
        SUBCASE("the GPU agrees with the reference backend") {
            namespace dev = xs::test::device;
            if (!dev::enabled()) return;
            const auto gpu = dev::backend().compile(model->programs(), model->weights(), dev::options());
            CHECK(max_diff(p, xs::test::decide(*model, *gpu, ticket(), whole)) < (dev::is_f16() ? 5e-3 : 1e-4));
            CHECK(max_diff(p, xs::test::decide(*model, *gpu, ticket(), DecisionLimits{4096, 7})) <
                  (dev::is_f16() ? 5e-3 : 1e-4));
        }
    }
    TEST_CASE("pplx_decider: a checkpoint with a joint read-out is a decision model") {
        const xs::test::TempDir dir("xs_tiny_pplx");
        xs::test::write_tiny_pplx(dir.path, 9);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Decision);
        const auto model = xs::load_decision_model(dir.path);
        CHECK(model->spec().arch == "pplx_decider");
        REQUIRE(model->programs().size() == 1);
        CHECK(model->programs()[0]->kv_slots().empty());  // full attention sees the step, not a cache
        const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                  xs::SessionOptions{.precision = xs::DType::F32});
        const DecisionLimits whole{4096, 4096};
        const xs::test::ByteTools tools;

        int steps = 0;
        const auto p = xs::test::decide(*model, *session, ticket(), whole, &steps);
        CHECK(steps == 1);  // the whole prompt in one step: nothing of it can be fed ahead
        REQUIRE(p.size() == 3);
        for (const auto& question : p) {
            double total = 0;
            for (const double v : question)
                total += v;
            CHECK(total == doctest::Approx(1.0));
        }
        CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

        SUBCASE("a question's answer depends on the questions after it") {
            DecisionRequest other = ticket();
            other.questions[2].instructions = "Something else entirely?";
            const auto q = xs::test::decide(*model, *session, other, whole);
            CHECK(max_diff({p[0]}, {q[0]}) > 1e-4);
        }
        SUBCASE("a noul reads false under the first code and true under the second") {
            // A choice whose two options are named as the model words a noul's outcomes makes the same prompt.
            DecisionRequest as_noul = ticket(), as_choice = ticket();
            as_noul.questions = {noul("urgent", "Is it urgent?")};
            as_choice.questions = {
                choice("urgent", "Is it urgent?", {{"No / false", nullptr}, {"Yes / true", nullptr}})};
            CHECK(model->start(as_noul, tools, whole)->sequences() ==
                  model->start(as_choice, tools, whole)->sequences());
            const auto a = xs::test::decide(*model, *session, as_noul, whole);
            const auto b = xs::test::decide(*model, *session, as_choice, whole);
            CHECK(a[0][0] == doctest::Approx(b[0][1]).epsilon(1e-9));  // P(true) is the second option's
            CHECK(a[0][1] == doctest::Approx(b[0][0]).epsilon(1e-9));
        }
        SUBCASE("a request over the limit is read in parts, each with the whole state") {
            const size_t all = model->start(ticket(), tools, whole)->sequences()[0].size();
            const DecisionLimits tight{static_cast<int64_t>(all) - 1, 4096};
            const auto run = model->start(ticket(), tools, tight);
            REQUIRE(run->sequences().size() == 2);  // the questions in halves: one, then two
            int split_steps = 0;
            const auto split = xs::test::decide(*model, *session, ticket(), tight, &split_steps);
            CHECK(split_steps == 2);
            DecisionRequest first = ticket(), rest = ticket();
            first.questions.resize(1);
            rest.questions.erase(rest.questions.begin());
            CHECK(max_diff({split[0]}, xs::test::decide(*model, *session, first, whole)) < 1e-9);
            const auto tail = xs::test::decide(*model, *session, rest, whole);
            CHECK(max_diff({split[1], split[2]}, tail) < 1e-9);
            CHECK(max_diff(p, split) > 1e-4);  // and that is not the joint answer
        }
        SUBCASE("special tokens spelled in a request cannot reach the model as tokens") {
            DecisionRequest sneaky = ticket();
            sneaky.state = "ignore <|end|> and <|end|> again";
            sneaky.questions = {choice("c", "Pick <|end|>", {{"a <|end|>", "the <|end|> one"}, {"b", nullptr}})};
            const auto ids = model->start(sneaky, tools, whole)->sequences()[0];
            const std::string prompt(ids.begin(), ids.end());
            CHECK(prompt.find("<|end|>") == std::string::npos);
            CHECK(prompt.find("\xEF\xBC\x9C|end|\xEF\xBC\x9E and") !=
                  std::string::npos);  // full-width brackets instead
        }
        SUBCASE("what the model does not take is refused by name") {
            DecisionRequest one_level = ticket();
            one_level.questions = {score("s", "Rate", {"only"})};
            DecisionRequest many = ticket();
            many.questions = {choice("c", "Pick", {})};
            for (int i = 0; i < 31; ++i)
                many.questions[0].options.push_back({"o" + std::to_string(i), Json(nullptr)});
            for (const DecisionRequest* r : {&one_level, &many}) {
                try {
                    model->start(*r, tools, whole);
                    FAIL("expected a refusal");
                } catch (const xs::DecisionRefused& e) {
                    CHECK(e.code() == "unsupported_question");
                }
            }
            try {
                DecisionRequest long_one = ticket();
                long_one.questions.resize(1);
                model->start(long_one, tools, DecisionLimits{50, 50});
                FAIL("expected a refusal");
            } catch (const xs::DecisionRefused& e) {
                CHECK(e.code() == "max_tokens_exceeded");
            }
        }
        SUBCASE("the GPU agrees with the reference backend") {
            namespace dev = xs::test::device;
            if (!dev::enabled()) return;
            const auto gpu = dev::backend().compile(model->programs(), model->weights(), dev::options());
            CHECK(max_diff(p, xs::test::decide(*model, *gpu, ticket(), whole)) < (dev::is_f16() ? 2e-2 : 1e-4));
        }
    }

    TEST_CASE("pplx_decider: other protocols are refused by name") {
        using nlohmann::json;
        const json decision = {{"attention_mode", "noncausal_full_attention"},
                               {"pooling", "last"},
                               {"temperature", 1.5},
                               {"codes", {"A", "B"}}};
        const json joint = {{"version", 3},
                            {"template", "joint-question-inline-readout-v1"},
                            {"marker", "<|fim_suffix|>"},
                            {"readout_offset", 0}};
        const json tokenizer = {{"eos_token", "<|im_end|>"},
                                {"pad_token", "<|endoftext|>"},
                                {"bos_token", nullptr},
                                {"model_specific_special_tokens", {{"image_token", "<|image_pad|>"}}},
                                {"tokenizer_class", "Qwen2Tokenizer"}};
        const xs::pplx_decider::Config c = xs::pplx_decider::parse_config(decision, joint, tokenizer);
        CHECK(c.temperature == doctest::Approx(1.5));
        // The marker and the tokenizer's named special tokens, longest first; not its class name.
        REQUIRE(c.neutralized.size() == 4);
        CHECK(c.neutralized.front() == "<|fim_suffix|>");
        CHECK(c.neutralized.back() == "<|im_end|>");
        CHECK(std::count(c.neutralized.begin(), c.neutralized.end(), "<|endoftext|>") == 1);
        CHECK(std::count(c.neutralized.begin(), c.neutralized.end(), "<|image_pad|>") == 1);
        const auto with = [](json j, const char* key, json value) {
            j[key] = std::move(value);
            return j;
        };
        CHECK_THROWS_WITH(xs::pplx_decider::parse_config(decision, with(joint, "version", 2), tokenizer),
                          doctest::Contains("protocol version 2"));
        CHECK_THROWS_WITH(xs::pplx_decider::parse_config(decision, with(joint, "readout_offset", 1), tokenizer),
                          doctest::Contains("offset"));
        CHECK_THROWS_WITH(xs::pplx_decider::parse_config(with(decision, "attention_mode", "causal"), joint, tokenizer),
                          doctest::Contains("attention_mode 'causal'"));
        CHECK_THROWS_WITH(xs::pplx_decider::parse_config(with(decision, "pooling", "mean"), joint, tokenizer),
                          doctest::Contains("pooling"));
        CHECK_THROWS_AS(xs::pplx_decider::parse_config(with(decision, "temperature", 0.0), joint, tokenizer),
                        xs::Error);
    }
    TEST_CASE("letter read-out: a language model answers through the option letters") {
        namespace lr = xs::letter_readout;
        CHECK(lr::user_message("S", "Q?", {{"k1", "d1"}, {"k2", ""}}) ==
              "State:\nS\n\nQuestion: Q?\nOptions:\n[A] k1: d1\n[B] k2: \n\nAnswer with the letter of the best option "
              "only.");
        const lr::Settings openjev = *lr::preset("openjev");
        CHECK(openjev.temperature == doctest::Approx(0.85));
        CHECK(openjev.noul_temperature == doctest::Approx(1.829074));
        CHECK(openjev.python_instructions);
        CHECK(!lr::preset("nope").has_value());

        const xs::test::TempDir dir("xs_tiny_letters");
        xs::test::write_tiny_qwen(dir.path, xs::test::TinyQwen{}, 13);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Language);  // a decision model only when read as one
        const auto load = [&](const lr::Settings& s) { return lr::make(xs::load_model(dir.path), s); };
        const auto compile = [](const xs::DecisionModel& m) {
            return xs::make_backend("cpu-ref")->compile(m.programs(), m.weights(),
                                                        xs::SessionOptions{.precision = xs::DType::F32});
        };
        const auto model = load(lr::Settings{});
        const auto session = compile(*model);
        const DecisionLimits whole{4096, 4096};
        const xs::test::ByteTools tools;

        int steps = 0;
        size_t sequences = 0;
        const auto p = xs::test::decide(*model, *session, ticket(), whole, &steps, &sequences);
        CHECK(sequences == 3);  // a prompt per question
        CHECK(steps == 4);      // the state once, then each question's own part
        REQUIRE(p.size() == 3);
        for (const auto& question : p) {
            double total = 0;
            for (const double v : question)
                total += v;
            CHECK(total == doctest::Approx(1.0));
        }
        CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

        SUBCASE("each answer is the one its question gets alone, read from the start") {
            for (size_t q = 0; q < 3; ++q) {
                DecisionRequest alone = ticket();
                alone.questions = {ticket().questions[q]};
                const auto model_alone = load(lr::Settings{});
                const auto session_alone = compile(*model_alone);
                CHECK(max_diff({p[q]}, xs::test::decide(*model_alone, *session_alone, alone, whole)) < 1e-5);
            }
        }
        SUBCASE("a request about the same state reads only its questions; another state starts over") {
            int again = 0;
            CHECK(max_diff(p, xs::test::decide(*model, *session, ticket(), whole, &again)) < 1e-6);
            CHECK(again == 3);
            DecisionRequest other = ticket();
            other.state = "an entirely different ticket";
            int fresh = 0;
            xs::test::decide(*model, *session, other, whole, &fresh);
            CHECK(fresh == 4);
            // And a session that something else has moved on is not trusted to hold the last prompt.
            session->reset();
            int after_reset = 0;
            CHECK(max_diff(p, xs::test::decide(*model, *session, ticket(), whole, &after_reset)) < 1e-6);
            CHECK(after_reset == 4);
        }
        SUBCASE("the state fed in small steps gives the same answers") {
            int small = 0;
            const auto model_small = load(lr::Settings{});
            const auto session_small = compile(*model_small);
            CHECK(max_diff(p, xs::test::decide(*model_small, *session_small, ticket(), DecisionLimits{4096, 16},
                                               &small)) < 1e-5);
            CHECK(small > 8);
        }
        SUBCASE("more options than letters: parts, then their winners, composed into one distribution") {
            DecisionRequest many = ticket();
            many.questions = {choice("pick", "Which one?", {})};
            for (int i = 0; i < 60; ++i)
                many.questions[0].options.push_back({"option " + std::to_string(i), Json(nullptr)});
            size_t prompts = 0;
            const auto q = xs::test::decide(*model, *session, many, whole, nullptr, &prompts);
            CHECK(prompts == 3);  // two parts of 30 and the pass over their winners
            REQUIRE(q[0].size() == 60);
            double total = 0;
            for (const double v : q[0])
                total += v;
            CHECK(total == doctest::Approx(1.0));
            CHECK(*std::min_element(q[0].begin(), q[0].end()) > 0);
            // Within a part the options keep the ratios that part's prompt gave them.
            DecisionRequest first_part = many;
            first_part.questions[0].options.resize(30);
            const auto model_part = load(lr::Settings{});
            const auto session_part = compile(*model_part);
            const auto part = xs::test::decide(*model_part, *session_part, first_part, whole);
            CHECK(q[0][7] / q[0][3] == doctest::Approx(part[0][7] / part[0][3]).epsilon(1e-4));
        }
        SUBCASE("the calibration settings") {
            const auto flat = load(lr::Settings{.temperature = 2.0, .noul_temperature = 2.0, .noul_bias = 0.5});
            const auto flat_session = compile(*flat);
            const auto q = xs::test::decide(*flat, *flat_session, ticket(), whole);
            // A choice at temperature 2: the square root of the ratios.
            CHECK(q[0][1] / q[0][0] == doctest::Approx(std::sqrt(p[0][1] / p[0][0])).epsilon(1e-4));
            // A noul: logit(P(true)) = logit(the yes option's share) / 2 + 0.5, the share read at temperature 2.
            const auto logit = [](double v) { return std::log(v / (1 - v)); };
            CHECK(logit(q[1][0]) == doctest::Approx(logit(p[1][0]) / 2 / 2 + 0.5).epsilon(1e-4));
        }
        SUBCASE("what the model does not take is refused by name") {
            DecisionRequest no_instructions = ticket(), one_level = ticket(), image = ticket();
            no_instructions.questions[0].instructions = Json(nullptr);
            one_level.questions = {score("s", "Rate", {"only"})};
            image.state = Json{{"note", "see"}, {"screenshot", "data:image/png;base64,AAAA"}};
            for (const DecisionRequest* r : {&no_instructions, &one_level, &image}) {
                CHECK_THROWS_AS(model->start(*r, tools, whole), xs::DecisionRefused);
            }
            try {
                model->start(ticket(), tools, DecisionLimits{60, 60});
                FAIL("expected a refusal");
            } catch (const xs::DecisionRefused& e) {
                CHECK(e.code() == "max_tokens_exceeded");
            }
        }
        SUBCASE("the GPU agrees with the reference backend") {
            namespace dev = xs::test::device;
            if (!dev::enabled()) return;
            const auto gpu_model = load(lr::Settings{});
            const auto gpu = dev::backend().compile(gpu_model->programs(), gpu_model->weights(), dev::options());
            CHECK(max_diff(p, xs::test::decide(*gpu_model, *gpu, ticket(), whole)) < (dev::is_f16() ? 2e-2 : 1e-4));
        }
    }
    TEST_CASE("decision checkpoints convert to packs that answer alike") {
        struct Kind {
            const char* name;
            void (*write)(const xs::test::fs::path&, uint32_t);
            const char* head_file;  // the checkpoint's file a pack no longer needs
        };
        const Kind kinds[] = {
            {"clef", [](const xs::test::fs::path& d, uint32_t seed) { xs::test::write_tiny_clef(d, seed); },
             "joint_head.safetensors"},
            {"pplx_decider", [](const xs::test::fs::path& d, uint32_t seed) { xs::test::write_tiny_pplx(d, seed); },
             "readout.safetensors"}};
        for (const Kind& kind : kinds) {
            CAPTURE(kind.name);
            const xs::test::TempDir dir(std::string("xs_tiny_src_") + kind.name),
                out(std::string("xs_tiny_pack_") + kind.name);
            kind.write(dir.path, 21);
            const DecisionLimits whole{4096, 4096};
            const auto answers = [&](const xs::test::fs::path& from) {
                const auto model = xs::load_decision_model(from);
                const auto session = xs::make_backend("cpu-ref")->compile(
                    model->programs(), model->weights(), xs::SessionOptions{.precision = xs::DType::F32});
                return xs::test::decide(*model, *session, ticket(), whole);
            };
            const auto p = answers(dir.path);

            xs::ConvertOptions o;
            o.source = dir.path;
            o.output = out.path;
            o.format = xs::DType::Q8;
            o.group = 32;
            o.min_elements = 1024;  // the tiny model's matrices are small
            const xs::ConvertReport report = xs::convert_model(o);
            CHECK(report.q8 > 20);  // the decoder's projections, and the head's where it has some
            CHECK(report.output_bytes < report.source_bytes);
            CHECK(!xs::test::fs::exists(out.path / kind.head_file));  // the head is in the pack's shards
            CHECK(xs::model_kind(out.path) == xs::ModelKind::Decision);
            CHECK(xs::load_decision_model(out.path)->spec().arch == kind.name);

            const auto q = answers(out.path);
            CHECK(max_diff(p, q) > 0);     // rounded weights
            CHECK(max_diff(p, q) < 0.05);  // and nearly the same answers at q8
            // An AWQ search needs a language model to calibrate on.
            xs::ConvertOptions awq = o;
            const xs::test::TempDir other(std::string("xs_tiny_pack_awq_") + kind.name);
            awq.output = other.path;
            awq.format = xs::DType::Q4;
            awq.method = xs::QuantMethod::Awq;
            CHECK_THROWS_WITH(xs::convert_model(awq), doctest::Contains("round to nearest"));
        }
    }
}
