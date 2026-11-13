// Decision models on tiny checkpoints the test writes: what must hold whatever the weights are. The numbers
// themselves are checked against each model's reference code in decision_parity.cpp.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>

#include "../backend/device_test_support.h"
#include "convert/convert.h"
#include "decision/runner.h"
#include "decision_test_util.h"
#include "embedding_test_util.h"
#include "io/safetensors_writer.h"
#include "models/clef/clef.h"
#include "models/config_util.h"
#include "models/d1/d1.h"
#include "models/laya/laya.h"
#include "models/letter_readout/letter_readout.h"
#include "models/model.h"
#include "models/policy.h"
#include "models/pplx_decider/pplx_decider.h"

namespace {

using xs::DecisionLimits;
using xs::DecisionQuestion;
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

// ticket() with every option described, as Julia-1 needs.
DecisionRequest described_ticket() {
    DecisionRequest r = ticket();
    r.questions[0].options[2].description = Json("anything else");
    return r;
}

// The bytes of a ByteTools sequence.
std::string text_of(const std::vector<int32_t>& ids) {
    std::string out;
    for (const int32_t id : ids)
        out += static_cast<char>(id);
    return out;
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
    TEST_CASE("clef: the head's scales as a GGUF file stores them answer as the parameters they come from") {
        namespace fs = xs::test::fs;
        const xs::test::TempDir dir("xs_tiny_clef_scales");
        xs::test::write_tiny_clef(dir.path / "params", 5);
        const auto answers = [](const fs::path& from) {
            const auto model = xs::load_decision_model(from);
            const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                      xs::SessionOptions{.precision = xs::DType::F32});
            return xs::test::decide(*model, *session, ticket(), DecisionLimits{4096, 4096});
        };
        // The same checkpoint with `scales` (prior, joint, gate) in the head's file; and, when `both`, the parameters
        // beside it.
        const auto with_scales = [&](const fs::path& to, bool both) {
            fs::create_directories(to);
            for (const char* f : {"config.json", "joint_head_config.json", "model.safetensors"})
                fs::copy_file(dir.path / "params" / f, to / f);
            const auto head = xs::SafetensorsWeights::open_file(dir.path / "params" / "joint_head.safetensors");
            std::vector<xs::OutUnit> units;
            std::map<std::string, float> x;
            for (const std::string& name : head->names()) {
                const xs::TensorView& v = head->get(name);
                if (name == "prior_logit_scale" || name == "joint_logit_scale" || name == "residual_gate") {
                    x[name] = xs::HostTensor::borrow(v.dtype, v.shape, const_cast<void*>(v.data)).to_f32()[0];
                    if (!both) continue;
                }
                units.push_back({{{name, v.dtype, v.shape}}, [v](std::ostream& out) {
                                     out.write(static_cast<const char*>(v.data),
                                               static_cast<std::streamsize>(v.nbytes()));
                                 }});
            }
            REQUIRE(x.size() == 3);
            const auto scales = std::make_shared<std::vector<float>>(
                std::vector<float>{std::exp(std::min(x["prior_logit_scale"], std::log(100.0f))),
                                   std::exp(std::min(x["joint_logit_scale"], std::log(100.0f))),
                                   1.0f / (1.0f + std::exp(-x["residual_gate"]))});
            units.push_back({{{"scales", xs::DType::F32, {3}}}, [scales](std::ostream& out) {
                                 out.write(reinterpret_cast<const char*>(scales->data()), 3 * sizeof(float));
                             }});
            xs::write_safetensors(to / "joint_head.safetensors", units);
        };
        with_scales(dir.path / "scales", false);
        CHECK(answers(dir.path / "scales") == answers(dir.path / "params"));
        with_scales(dir.path / "both", true);
        CHECK_THROWS_WITH(xs::load_decision_model(dir.path / "both"),
                          doctest::Contains("clef: the weights hold both joint_head.scales and joint_head."));
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
    TEST_CASE("d1: an LFM3-VL checkpoint named D1Model is a decision model") {
        const xs::test::TempDir dir("xs_tiny_d1");
        xs::test::write_tiny_d1(dir.path, 17);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Decision);
        const auto model = xs::load_decision_model(dir.path);
        CHECK(model->spec().arch == "d1");
        REQUIRE(model->programs().size() == 1);
        // Full-attention layers read the whole step both ways; the sliding layer stays causal within its window.
        int both_ways = 0, windowed = 0;
        for (const xs::graph::Node& n : model->programs()[0]->nodes()) {
            if (n.kind != xs::graph::OpKind::Attention) continue;
            const auto& a = std::get<xs::graph::AttentionAttrs>(n.attrs);
            both_ways += !a.causal && a.sliding_window == 0;
            windowed += a.causal && a.sliding_window == 16;
        }
        CHECK(both_ways == 2);
        CHECK(windowed == 1);
        const auto compile = [](const xs::DecisionModel& m) {
            return xs::make_backend("cpu-ref")->compile(m.programs(), m.weights(),
                                                        xs::SessionOptions{.precision = xs::DType::F32});
        };
        const auto session = compile(*model);
        const DecisionLimits whole{4096, 4096};
        const xs::test::ByteTools tools = xs::test::d1_tools();

        int steps = 0;
        size_t sequences = 0;
        const auto p = xs::test::decide(*model, *session, ticket(), whole, &steps, &sequences, &tools);
        CHECK(sequences == 3);  // each question alone after the state
        CHECK(steps == 3);
        REQUIRE(p.size() == 3);
        for (const auto& question : p) {
            double total = 0;
            for (const double v : question)
                total += v;
            CHECK(total == doctest::Approx(1.0));
        }
        CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

        SUBCASE("the prompt: the state, a question, the assistant's empty thought") {
            const auto run = model->start(ticket(), tools, whole);
            REQUIRE(run->sequences().size() == 3);
            const std::string head =
                "<|startoftext|><|im_start|>user\n{\n  \"ticket\": 4411,\n  \"text\": \"cannot log in after the reset "
                "email expired\",\n  \"vip\": true\n}\n\n\nQUESTION:\n";
            const std::string tail = "<|im_end|>\n<|im_start|>assistant\n<think></think>";
            CHECK(tools.text(run->sequences()[0]) ==
                  head + "Which team?\n\nOptions:\nA payments\nB logins\nC other\n\nReply with the option code only." +
                      tail);
            CHECK(tools.text(run->sequences()[1]) == head + "Is it urgent?\n\nReply with yes or no only." + tail);
            CHECK(tools.text(run->sequences()[2]) ==
                  head + "How much work?\n\n0 minutes\n1 hours\n2 days\n\nReply with a single digit 0-2 only." + tail);
            DecisionRequest text = ticket();
            text.state = "plain text";
            CHECK(tools.text(model->start(text, tools, whole)->sequences()[1]) ==
                  "<|startoftext|><|im_start|>user\nplain text\n\n\nQUESTION:\nIs it urgent?\n\nReply with yes or no "
                  "only." +
                      tail);
        }
        SUBCASE("each answer is its question's alone") {
            for (size_t q = 0; q < 3; ++q) {
                DecisionRequest alone = ticket();
                alone.questions = {ticket().questions[q]};
                CHECK(max_diff({p[q]}, xs::test::decide(*model, *session, alone, whole, nullptr, nullptr, &tools)) <
                      1e-12);
            }
        }
        SUBCASE("a noul's outcomes, and the codes of a choice's options") {
            DecisionRequest r = ticket();
            r.questions = {noul("hw", "Hardware?"), choice("letters", "Which?", {{"x", "first"}, {"Y", nullptr}}),
                           choice("many", "Which one?", {})};
            r.questions[0].options[0].description = Json("Equipment failed.");
            for (int i = 0; i < 54; ++i)
                r.questions[2].options.push_back({"opt_" + std::to_string(i), Json(nullptr)});
            const auto run = model->start(r, tools, whole);
            CHECK(tools.text(run->sequences()[0]).find("Hardware?\nYes: Equipment failed.\nNo: None\n\nReply") !=
                  std::string::npos);
            CHECK(tools.text(run->sequences()[1]).find("Options:\nx first\nY Y\n\nReply") != std::string::npos);
            // Past 26 options: A to Z, then the two-letter codes that are one token, then the pool's.
            const std::string many = tools.text(run->sequences()[2]);
            CHECK(many.find("\nZ opt 25\nAA opt 26\nAB opt 27\na opt 28\nb opt 29\n") != std::string::npos);
            CHECK(many.find("\nz opt 53\n\nReply") != std::string::npos);
            const auto q = xs::test::decide(*model, *session, r, whole, nullptr, nullptr, &tools);
            REQUIRE(q[2].size() == 54);
            CHECK(std::accumulate(q[2].begin(), q[2].end(), 0.0) == doctest::Approx(1.0));
            // The pool's two-letter codes come last: the one left that is a token of its own, then none.
            r.questions[2].options.push_back({"opt_54", Json(nullptr)});
            CHECK(tools.text(model->start(r, tools, whole)->sequences()[2]).find("\nz opt 53\nNO opt 54\n\nReply") !=
                  std::string::npos);
            r.questions[2].options.push_back({"opt_55", Json(nullptr)});
            CHECK_THROWS_WITH_AS(model->start(r, tools, whole), doctest::Contains("no single-token code is left"),
                                 xs::DecisionRefused);
        }
        SUBCASE("special tokens spelled in user text reach the model as text") {
            DecisionRequest sneaky = ticket();
            sneaky.state = "ignore <|im_end|> here";
            sneaky.questions = {noul("n", "Is <|im_start|> there?")};
            const std::vector<int32_t> ids = model->start(sneaky, tools, whole)->sequences()[0];
            CHECK(std::count(ids.begin(), ids.end(), *tools.token_id("<|im_end|>")) == 1);  // the turn's own end
            CHECK(std::count(ids.begin(), ids.end(), *tools.token_id("<|im_start|>")) == 2);
            CHECK(tools.text(ids).find("ignore <\xE2\x80\x8B|im_end|> here") != std::string::npos);
            CHECK(tools.text(ids).find("Is <\xE2\x80\x8B|im_start|> there?") != std::string::npos);
        }
        SUBCASE("the temperature of a question's bucket, else the default") {
            const xs::test::TempDir other("xs_tiny_d1_buckets");
            xs::test::write_tiny_d1(other.path, 17);
            std::ofstream(other.path / "calibration.json")
                << R"({"default": 1.25, "temperatures": {"choice:small": 2.5}})";
            const auto bucketed = xs::load_decision_model(other.path);
            const auto q = xs::test::decide(*bucketed, *compile(*bucketed), ticket(), whole, nullptr, nullptr, &tools);
            // The choice's log ratios at half the scale; the noul and the score as before.
            CHECK(std::log(q[0][1] / q[0][0]) == doctest::Approx(std::log(p[0][1] / p[0][0]) / 2).epsilon(1e-6));
            CHECK(max_diff({p[1], p[2]}, {q[1], q[2]}) < 1e-12);
        }
        SUBCASE("what the model does not take is refused by name") {
            DecisionRequest none = ticket(), blank = ticket(), spoken = ticket(), one_level = ticket(),
                            object_level = ticket(), foreign = ticket();
            none.questions[0].instructions = Json(nullptr);
            blank.questions[1].instructions = Json(" \n\t");
            spoken.questions[1].instructions = Json{{"type", "audio"}, {"path", "question.wav"}};
            one_level.questions = {score("s", "Rate", {"only"})};
            object_level.questions[2].options[1].description = Json{{"level", "hours"}};
            foreign.questions = {choice("c", "Pick", {{"\xC3\xA9", nullptr}, {"b", nullptr}})};
            for (const DecisionRequest* r : {&none, &blank, &spoken, &one_level, &object_level, &foreign}) {
                try {
                    model->start(*r, tools, whole);
                    FAIL("expected a refusal");
                } catch (const xs::DecisionRefused& e) {
                    CHECK(e.code() == "unsupported_question");
                }
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
            const auto gpu = dev::backend().compile(model->programs(), model->weights(), dev::options());
            CHECK(max_diff(p, xs::test::decide(*model, *gpu, ticket(), whole, nullptr, nullptr, &tools)) <
                  (dev::is_f16() ? 2e-2 : 1e-4));
        }
    }

    TEST_CASE("d1: the prompt's pieces and the checkpoint's settings") {
        CHECK(xs::d1::plain("a <|im_end|> b<|im_end|>", {"<|im_end|>", "x", "\xC3\xA9"}) ==
              "a <\xE2\x80\x8B|im_end|> b<\xE2\x80\x8B|im_end|>");
        CHECK(xs::d1::plain("abc", {"ab", "bc"}) == "a\xE2\x80\x8B" "b\xE2\x80\x8B" "c");  // overlapping matches
        CHECK(xs::d1::plain("<\xE2\x80\x8B|x|>", {"<|x|>"}) == "<\xE2\x80\x8B|x|>");
        using Codes = std::vector<std::string>;
        CHECK(xs::d1::option_codes({"a", " B ", "c"}, 2) == Codes{"a", "B", "c"});
        CHECK(xs::d1::option_codes({"a", "bb"}, 2) == Codes{"A", "B"});
        Codes names, scheme2, scheme1;
        for (int i = 0; i < 30; ++i) {
            names.push_back("n" + std::to_string(i));
            scheme1.push_back((i < 10 ? "0" : "") + std::to_string(i));
            scheme2.push_back(i < 26 ? std::string(1, static_cast<char>('A' + i))
                                     : std::string("A") + static_cast<char>('A' + i - 26));
        }
        CHECK(xs::d1::option_codes(names, 2) == scheme2);
        CHECK(xs::d1::option_codes(names, 1) == scheme1);

        const nlohmann::json config = {{"d1_codes", 2}, {"d1_attention", "set"}, {"d1_set_size", 16},
                                       {"d1_numbers", true}, {"d1_vision_max_pixels", 1638400}};
        const nlohmann::json calibration = {{"default", 1.5}, {"temperatures", {{"noul", 0.5}, {"score:3", 2.0}}}};
        const nlohmann::json tokenizer = {{"bos_token", "<|startoftext|>"}};
        const xs::d1::Config c = xs::d1::parse_config(config, calibration, tokenizer);
        CHECK(c.code_scheme == 2);
        CHECK(c.set_size == 16);
        CHECK(c.bos == "<|startoftext|>");
        using xs::QuestionType;
        CHECK(xs::d1::temperature(c, QuestionType::Noul, 2) == 0.5);
        CHECK(xs::d1::temperature(c, QuestionType::Score, 3) == 2.0);
        CHECK(xs::d1::temperature(c, QuestionType::Score, 4) == 1.5);
        CHECK(xs::d1::temperature(c, QuestionType::Choice, 3) == 1.5);
        for (const auto& [k, bucket] : {std::pair<size_t, const char*>{3, "choice:small"}, {4, "choice:mid"},
                                        {8, "choice:mid"}, {9, "choice:large"}, {26, "choice:large"},
                                        {27, "choice:inventory"}}) {
            CAPTURE(bucket);
            const xs::d1::Config one = xs::d1::parse_config(
                config, nlohmann::json{{"temperatures", {{bucket, 3.0}}}}, tokenizer);
            CHECK(xs::d1::temperature(one, QuestionType::Choice, k) == 3.0);
            CHECK(one.temperature == 1.0);
        }
        const xs::d1::Config bare = xs::d1::parse_config(config, nlohmann::json(), nlohmann::json::object());
        CHECK(bare.temperature == 1.0);  // no calibration.json: the logits as they are
        CHECK(bare.bos.empty());

        const auto with = [&](const char* key, nlohmann::json value) {
            nlohmann::json c2 = config;
            c2[key] = std::move(value);
            return c2;
        };
        for (const auto& [bad, names_it] :
             {std::pair<nlohmann::json, const char*>{with("d1_attention", "state"), "state"},
              {with("d1_attention", "causal"), "causal"},
              {with("d1_codes", 3), "d1_codes"},
              {with("d1_set_size", 1), "d1_set_size"},
              {with("d1_prompt", 3), "d1_prompt"},
              {with("d1_new_thing", true), "d1_new_thing"}}) {
            CHECK_THROWS_WITH_AS(xs::d1::parse_config(bad, calibration, tokenizer), doctest::Contains(names_it),
                                 xs::Error);
        }
        CHECK_THROWS_WITH_AS(xs::d1::parse_config(config, nlohmann::json{{"default", 1.0}}, tokenizer),
                             doctest::Contains("temperatures"), xs::Error);

        const xs::test::TempDir dir("xs_tiny_d1_omni");
        xs::test::write_tiny_d1(dir.path, 3);
        nlohmann::json omni = xs::cfg::read_json(dir.path / "config.json");
        omni["model_type"] = "lfm3_omni";
        std::ofstream(dir.path / "config.json") << omni.dump();
        CHECK_THROWS_WITH_AS(xs::load_decision_model(dir.path), doctest::Contains("lfm3_omni"), xs::Error);
    }

    TEST_CASE("laya and julia-1: a sequence per question, each answered as if asked alone") {
        for (const bool julia : {false, true}) {
            CAPTURE(julia);
            const xs::test::TempDir dir(julia ? "xs_tiny_julia" : "xs_tiny_laya");
            xs::test::TinyLaya m;
            m.julia = julia;
            xs::test::write_tiny_laya(dir.path, m, 9);
            CHECK(xs::model_kind(dir.path) == xs::ModelKind::Decision);
            const auto model = xs::load_decision_model(dir.path);
            CHECK(model->spec().arch == (julia ? "julia" : "laya"));
            CHECK(model->spec().max_tokens == 256);
            CHECK(model->spec().tokenizer_dir == "tokenizer");
            CHECK_FALSE(model->spec().chat_template);
            REQUIRE(model->programs().size() == 1);
            CHECK_FALSE(model->programs()[0]->stateful());
            const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                      xs::SessionOptions{.precision = xs::DType::F32});
            const DecisionLimits whole{4096, 4096};
            int steps = 0;
            size_t sequences = 0;
            const auto p = xs::test::decide(*model, *session, described_ticket(), whole, &steps, &sequences);
            CHECK(steps == 3);
            CHECK(sequences == 3);
            REQUIRE(p.size() == 3);
            for (const auto& question : p) {
                double total = 0;
                for (const double v : question)
                    total += v;
                CHECK(total == doctest::Approx(1.0));
            }
            CHECK(*std::max_element(p[0].begin(), p[0].end()) - *std::min_element(p[0].begin(), p[0].end()) > 0.02);

            // Each question alone, and the questions in the other order.
            for (size_t q = 0; q < 3; ++q) {
                DecisionRequest alone = described_ticket();
                alone.questions = {alone.questions[q]};
                CHECK(max_diff({p[q]}, xs::test::decide(*model, *session, alone, whole)) < 1e-9);
            }
            DecisionRequest reversed = described_ticket();
            std::reverse(reversed.questions.begin(), reversed.questions.end());
            const auto r = xs::test::decide(*model, *session, reversed, whole);
            CHECK(max_diff({p[0], p[1], p[2]}, {r[2], r[1], r[0]}) < 1e-9);
            // A noul's prompt lists "false" before "true" (its answer lists true first), in each model's wording.
            const xs::test::ByteTools tools;
            const std::string noul = text_of(model->start(described_ticket(), tools, whole)->sequences()[1]);
            CHECK(noul.find(julia ? "\x03 false\x03 true" : "\x03 false: no, the statement does not hold\x03 true: ") !=
                  std::string::npos);

            namespace dev = xs::test::device;
            if (!dev::enabled()) continue;
            const auto gpu = dev::backend().compile(model->programs(), model->weights(), dev::options());
            CHECK(max_diff(p, xs::test::decide(*model, *gpu, described_ticket(), whole)) <
                  (dev::is_f16() ? 1e-2 : 1e-4));
        }
    }

    TEST_CASE("laya: what does not fit is cut as its reference cuts it; its wording of options and instructions") {
        const xs::test::TempDir dir("xs_tiny_laya_cuts"), wide("xs_tiny_laya_wording");
        xs::test::TinyLaya m;
        m.max_tokens = 160;
        m.head_tokens = 48;
        m.root_config = false;  // as a variant's folder: the model is found by its rl_agent_config.json
        xs::test::write_tiny_laya(dir.path, m, 10);
        CHECK(xs::model_kind(dir.path) == xs::ModelKind::Decision);
        const auto model = xs::load_decision_model(dir.path);
        const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                  xs::SessionOptions{.precision = xs::DType::F32});
        const xs::test::ByteTools tools;
        const DecisionLimits whole{4096, 4096};
        const auto sequence = [&](const DecisionRequest& r) {
            return model->start(r, tools, whole)->sequences()[0];
        };
        DecisionRequest r;
        r.questions = {noul("q", "Is it late?")};

        // A long state keeps the tokens that fit the 160 of the sequence; the answer is that of the shorter state.
        const std::string long_state(300, 'x');
        r.state = long_state;
        const std::vector<int32_t> cut = sequence(r);
        CHECK(cut.size() == 160);
        CHECK(cut.back() == 2);  // [SEP] closes it
        const size_t prefix = text_of(cut).find(std::string(1, '\x02') + "x") + 1;  // the state starts after [SEP]
        DecisionRequest shorter = r;
        shorter.state = std::string(159 - prefix, 'x');
        CHECK(sequence(shorter) == cut);
        const auto long_answers = xs::test::decide(*model, *session, r, whole);
        CHECK(max_diff(long_answers, xs::test::decide(*model, *session, shorter, whole)) < 1e-12);

        // Ten options of 17 tokens do not fit the question budget of 48: each is cut to 4 (its marker and 3 bytes),
        // and the question to 8 bytes.
        std::vector<std::pair<std::string, Json>> options;
        for (int i = 0; i < 10; ++i)
            options.push_back({"option" + std::to_string(i), "number " + std::to_string(i) + " of ten"});
        r.state = "s";
        r.questions = {choice("c", "Which of the ten options fits?", options)};
        std::string expected = "\x01" "choice q\x02";
        for (int i = 0; i < 10; ++i)
            expected += "\x03 op";
        CHECK(text_of(sequence(r)) == expected + "\x02s\x02");

        // Sixty options at four tokens each pass the sequence's 160 tokens: refused, as its reference refuses them.
        options.resize(60, {"o", "x"});
        r.questions = {choice("c", "Pick", options)};
        CHECK_THROWS_AS(sequence(r), xs::DecisionRefused);

        // Names, descriptions and instructions as its reference writes them: json.dumps (ASCII only) for instructions
        // that are not text, str() for descriptions that are not, a name alone without a description; its mask token
        // in user text becomes a space. A model whose question budget holds them whole.
        m.max_tokens = 400;
        m.head_tokens = 120;
        xs::test::write_tiny_laya(wide.path, m, 10);
        const auto wording = xs::load_decision_model(wide.path);
        const auto written = [&](const DecisionRequest& req) { return wording->start(req, tools, whole)->sequences(); };
        DecisionQuestion q = choice("c", "", {{"a", nullptr}, {"b", ""}, {"c", Json{{"k", 1}}}, {"d", "x\x03y"}});
        q.instructions = Json{{"lang", "fran\xc3\xa7" "ais"}};
        r.questions = {q};
        CHECK(text_of(written(r)[0]) ==
              "\x01" "choice question: {\"lang\": \"fran\\u00e7ais\"}\x02"
              "\x03 a\x03 b\x03 c: {'k': 1}\x03 d: x y\x02s\x02");
        DecisionQuestion level = score("s", "Rate", {"low"});
        level.options.push_back({"1", Json::array({"high", 2})});
        q = noul("n", "Odd?");
        q.options[0].description = Json("It is.");
        r.questions = {level, q};
        const auto both = written(r);
        CHECK(text_of(both[0]).find("\x03 level 0: low\x03 level 1: ['high', 2]\x02") != std::string::npos);
        CHECK(text_of(both[1]).find("\x03 false: no, the statement does not hold\x03 true: It is.\x02") !=
              std::string::npos);
        q.options[1].description = Json{{"not", "text"}};  // its reference adds it to a string and fails
        r.questions = {q};
        CHECK_THROWS_WITH_AS(written(r), doctest::Contains("an object or array is not supported"), xs::DecisionRefused);
    }

    TEST_CASE("laya: the temperature of a question's type, or of its type and option count, divides its logits") {
        const xs::test::TempDir plain("xs_tiny_laya_t1"), fitted("xs_tiny_laya_t2");
        xs::test::TinyLaya m;
        xs::test::write_tiny_laya(plain.path, m, 11);
        m.temperature = {2.0, 3.0, 4.0};
        m.temperature_by_options = {{"choice:3-5", 0.5}, {"noul:2", 1.5}};
        xs::test::write_tiny_laya(fitted.path, m, 11);
        const auto answers = [](const xs::test::fs::path& dir) {
            const auto model = xs::load_decision_model(dir);
            const auto session = xs::make_backend("cpu-ref")->compile(model->programs(), model->weights(),
                                                                      xs::SessionOptions{.precision = xs::DType::F32});
            return xs::test::decide(*model, *session, ticket(), DecisionLimits{4096, 4096});
        };
        const auto p = answers(plain.path), q = answers(fitted.path);
        const double temperature[] = {0.5, 1.5, 3.0};  // choice of 3 options, noul, score of 3 levels (by type)
        for (size_t i = 0; i < 3; ++i) {
            CAPTURE(i);
            for (size_t o = 1; o < p[i].size(); ++o)
                CHECK(std::log(q[i][o] / q[i][0]) * temperature[i] == doctest::Approx(std::log(p[i][o] / p[i][0])));
        }
    }

    TEST_CASE("julia-1: what its strict encoding cannot read whole is refused by name") {
        const xs::test::TempDir dir("xs_tiny_julia_refusals");
        xs::test::TinyLaya m;
        m.julia = true;
        xs::test::write_tiny_laya(dir.path, m, 12);
        const auto model = xs::load_decision_model(dir.path);
        const xs::test::ByteTools tools;
        const auto refusal = [&](const DecisionRequest& r) -> std::string {
            try {
                model->start(r, tools, DecisionLimits{4096, 4096});
            } catch (const xs::DecisionRefused& e) {
                return e.what();
            }
            return "accepted";
        };
        const DecisionRequest ok = described_ticket();
        CHECK(refusal(ok) == "accepted");
        DecisionRequest r = ok;
        r.questions[1].instructions = Json(nullptr);
        CHECK(refusal(r) == "question 'urgent': Julia-1 takes its instructions as text");
        r = ok;
        r.questions[0].options.resize(1);
        CHECK(refusal(r) == "question 'team': Julia-1 takes 2 to 20 options; the question has 1");
        r = ok;
        r.questions[0].options[2].description = Json(nullptr);
        CHECK(refusal(r) ==
              "question 'team': Julia-1 reads each option as its description, a non-empty text; 'other' has none");
        r = ok;
        r.questions[1].options[0].description = Json("yes");
        CHECK(refusal(r) ==
              "question 'urgent': Julia-1 takes a noul's criteria for both 'false' and 'true', or neither");
        r = ok;
        r.state = "a \x03 in the state";
        CHECK(refusal(r).find("refuses a request that spells its marker") != std::string::npos);
        r = ok;
        r.questions[0].options[0].description = Json(std::string(60, 'x'));
        CHECK(refusal(r) == "question 'team': Julia-1 reads at most 48 tokens of an option; one has 61");
        r = ok;
        for (xs::DecisionOption& o : r.questions[0].options)
            o.description = Json(std::string(40, 'x'));
        CHECK(refusal(r).find("reads a question and its options in 96 tokens") != std::string::npos);
        r = ok;
        r.state = std::string(300, 's');
        CHECK(refusal(r).find("tokens with the state; the limit is 256") != std::string::npos);
        // Without strict encoding, the state is cut instead.
        const xs::test::TempDir loose("xs_tiny_julia_loose");
        m.strict = false;
        xs::test::write_tiny_laya(loose.path, m, 12);
        const auto cutting = xs::load_decision_model(loose.path);
        CHECK(cutting->start(r, tools, DecisionLimits{4096, 4096})->sequences()[0].size() == 256);
    }

    TEST_CASE("laya and julia-1: the runtime reads their tokenizer from its folder, with no chat template") {
        for (const bool julia : {false, true}) {
            CAPTURE(julia);
            const xs::test::TempDir dir("xs_tiny_laya_runner");
            xs::test::TinyLaya m;
            m.julia = julia;
            xs::test::write_tiny_laya(dir.path, m, 14);
            std::ofstream(dir.path / "tokenizer" / "tokenizer.json") << xs::test::tiny_wordpiece_tokenizer().dump();
            std::ofstream(dir.path / "tokenizer" / "tokenizer_config.json")
                << nlohmann::json{{"cls_token", "[CLS]"}, {"sep_token", "[SEP]"}, {"mask_token", "[MASK]"}}.dump();
            xs::DecisionRunnerOptions o;
            o.backend = "cpu-ref";
            o.session.precision = xs::DType::F32;
            const auto runner = xs::DecisionRunner::create(dir.path, o);  // warms up on the sample request
            DecisionRequest r;
            r.state = "the cat sat on the mat.";
            r.questions = {noul("sat", "the cat sat"), choice("pet", "the pet", {{"cat", "a cat"}, {"dog", "a dog"}})};
            const xs::DecisionResult result = runner->decide(r);
            REQUIRE(result.sequences.size() == 2);
            // [CLS] noul question : the cat sat [SEP] [MASK] false [MASK] true [SEP] the cat ... [SEP]
            CHECK(result.sequences[0].front() == 2);
            CHECK(std::count(result.sequences[0].begin(), result.sequences[0].end(), 4) == 2);
            CHECK(result.sequences[0].back() == 3);
            CHECK(result.answers[1].probabilities.size() == 2);
        }
    }

    TEST_CASE("laya and julia-1: what they do not implement is refused by name") {
        using nlohmann::json;
        const xs::test::TempDir dir("xs_tiny_laya_refusals"), out("xs_tiny_laya_pack");
        const auto load_error = [&]() {
            try {
                xs::load_decision_model(dir.path);
            } catch (const xs::Error& e) {
                return std::string(e.what());
            }
            return std::string("loaded");
        };
        xs::test::TinyLaya m;
        xs::test::write_tiny_laya(dir.path, m, 13);
        std::ofstream(dir.path / "rl_agent_config.json")
            << json{{"head_layers", 2}, {"max_len", 256}, {"head_max_len", 96},
                    {"temperature_by_options", {{"choice:big", 1.0}}}}.dump();
        CHECK(load_error() ==
              "rl_agent_config.json: temperature_by_options key 'choice:big' is not a question type and option count");
        std::ofstream(dir.path / "rl_agent_config.json")
            << json{{"head_layers", 2}, {"max_len", 9000}, {"head_max_len", 96}}.dump();
        CHECK(load_error().find("laya: a sequence of 9000 tokens with a question of 96 does not fit the encoder's 512 "
                                "positions") != std::string::npos);
        // A pack would leave the tokenizer and encoder folders behind.
        std::ofstream(dir.path / "rl_agent_config.json")
            << json{{"head_layers", 2}, {"max_len", 256}, {"head_max_len", 96}}.dump();
        xs::ConvertOptions o;
        o.source = dir.path;
        o.output = out.path;
        o.format = xs::DType::Q8;
        CHECK_THROWS_WITH(xs::convert_model(o), doctest::Contains("a pack does not carry yet"));

        xs::test::fs::remove_all(dir.path);
        m.julia = true;
        xs::test::write_tiny_laya(dir.path, m, 13);
        std::ofstream(dir.path / "inference-policy.json")
            << json{{"max_length", 256}, {"head_length", 96}, {"strict_encoding", true}, {"calibration", {1.5}}}.dump();
        CHECK(load_error() == "inference-policy.json: a calibration is not supported");
        xs::test::fs::remove(dir.path / "inference-policy.json");
        CHECK(load_error().find("inference-policy.json is missing") != std::string::npos);
        std::ofstream(dir.path / "rl_agent_config.json") << json{{"head_layers", 2}}.dump();
        CHECK(load_error().find("holds both Laya's rl_agent_config.json and Julia's julia_config.json") !=
              std::string::npos);
    }

    TEST_CASE("decision checkpoints convert to packs that answer alike") {
        struct Kind {
            const char* name;
            void (*write)(const xs::test::fs::path&, uint32_t);
            const char* head_file;  // the checkpoint's file a pack no longer needs (null: no head of its own)
            xs::test::ByteTools tools;
        };
        const Kind kinds[] = {
            {"clef", [](const xs::test::fs::path& d, uint32_t seed) { xs::test::write_tiny_clef(d, seed); },
             "joint_head.safetensors", xs::test::ByteTools()},
            {"pplx_decider", [](const xs::test::fs::path& d, uint32_t seed) { xs::test::write_tiny_pplx(d, seed); },
             "readout.safetensors", xs::test::ByteTools()},
            {"d1", [](const xs::test::fs::path& d, uint32_t seed) { xs::test::write_tiny_d1(d, seed); }, nullptr,
             xs::test::d1_tools()}};
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
                return xs::test::decide(*model, *session, ticket(), whole, nullptr, nullptr, &kind.tools);
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
            if (kind.head_file != nullptr) {
                CHECK(!xs::test::fs::exists(out.path / kind.head_file));  // the head is in the pack's shards
            }
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
