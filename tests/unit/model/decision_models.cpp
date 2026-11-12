// Decision models on tiny checkpoints the test writes: what must hold whatever the weights are. The numbers
// themselves are checked against each model's reference code in decision_parity.cpp.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>

#include "../backend/device_test_support.h"
#include "decision_test_util.h"
#include "models/clef/clef.h"
#include "models/policy.h"

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
}
