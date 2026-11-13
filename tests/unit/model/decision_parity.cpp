// A decision model against the checkpoint's own Python code (tests/parity/decision/reference.py): each request is
// read as the same tokens and every option gets the same probability, and a request the reference refuses is refused.
// Opt-in:
//
//   XS_TEST_DECISION_DIR   the checkpoint (a released one, or tests/parity/decision/make_tiny.py's)
//   XS_DECISION_FIXTURES   reference.py's output for it
//   XS_PARITY_BACKEND      cpu-ref (default) | sycl
//   XS_PARITY_DEVICE, XS_PARITY_PRECISION (f32 by default)
//   XS_DECISION_CHUNK      tokens a step feeds (default 1024; a small one splits spans across steps)

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

#include "decision/runner.h"
#include "models/letter_readout/letter_readout.h"
#include "models/model.h"
#include "models/policy.h"
#include "parity_util.h"
#include "server/decision_handlers.h"

namespace {

using xs::test::env_or;

bool enabled() {
    return !env_or("XS_TEST_DECISION_DIR", "").empty() && !env_or("XS_DECISION_FIXTURES", "").empty();
}

}  // namespace

TEST_CASE("decision model matches its reference code" * doctest::skip(!enabled())) {
    std::ifstream in(env_or("XS_DECISION_FIXTURES", ""));
    REQUIRE_MESSAGE(in.good(), "cannot read XS_DECISION_FIXTURES");
    const auto fixtures = nlohmann::ordered_json::parse(in);
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    // Probabilities: f32 follows the reference to rounding; f16 weights move a near-even answer by about a point.
    const double tolerance = f32 ? 2e-4 : 2e-2;

    xs::DecisionRunnerOptions options;
    options.backend = env_or("XS_PARITY_BACKEND", "cpu-ref");
    options.session = xs::test::parity_options();
    options.session.max_context = 16384;
    options.chunk_tokens = std::stoll(env_or("XS_DECISION_CHUNK", "1024"));
    const std::string dir = env_or("XS_TEST_DECISION_DIR", "");
    std::unique_ptr<xs::DecisionModel>
        model;  // a language model: read at the option letters, as the fixtures' helper does
    if (xs::model_kind(dir) == xs::ModelKind::Language) {
        model = xs::letter_readout::make(xs::load_model(dir, xs::LoadOptions{.vision = false}),
                                         *xs::letter_readout::preset(fixtures.value("preset", "letters")));
    }
    const auto runner = xs::DecisionRunner::create(dir, options, std::move(model));

    double worst = 0;
    int questions = 0, top_differs = 0, refused = 0;
    for (const auto& c : fixtures["cases"]) {
        const xs::DecisionRequest request = xs::server::parse_systemone_request(c["request"]);
        if (c.contains("refused")) {
            CAPTURE(c["refused"].get<std::string>());
            CHECK_THROWS_AS(runner->decide(request), xs::DecisionRefused);
            ++refused;
            continue;
        }
        const xs::DecisionResult result = runner->decide(request);
        // The same sequences, in whatever order each side read them (a model that reads one prompt per question).
        auto expected_ids = c["input_ids"].get<std::vector<std::vector<int32_t>>>();
        std::vector<std::vector<int32_t>> ids = result.sequences;
        std::sort(expected_ids.begin(), expected_ids.end());
        std::sort(ids.begin(), ids.end());
        REQUIRE(ids.size() == expected_ids.size());
        for (size_t s = 0; s < ids.size(); ++s) {
            CAPTURE(s);
            REQUIRE(ids[s].size() == expected_ids[s].size());
            CHECK(ids[s] == expected_ids[s]);
        }
        REQUIRE(result.answers.size() == request.questions.size());
        for (size_t q = 0; q < request.questions.size(); ++q) {
            const xs::DecisionQuestion& question = request.questions[q];
            const auto& expected = c["probabilities"][question.id];
            REQUIRE(expected.size() == question.options.size());
            std::vector<double> ref;
            for (const xs::DecisionOption& option : question.options)
                ref.push_back(expected[option.id].get<double>());
            const std::vector<double>& ours = result.answers[q].probabilities;
            double diff = 0;
            for (size_t o = 0; o < ref.size(); ++o)
                diff = std::max(diff, std::abs(ours[o] - ref[o]));
            const size_t top = static_cast<size_t>(std::max_element(ref.begin(), ref.end()) - ref.begin());
            CAPTURE(question.id);
            CAPTURE(diff);
            CHECK(diff <= tolerance);
            // The same option on top, unless the reference itself holds two within the tolerance.
            if (result.answers[q].best != top) {
                ++top_differs;
                CHECK(ref[top] - ref[result.answers[q].best] <= 2 * tolerance);
            }
            worst = std::max(worst, diff);
            ++questions;
        }
    }
    MESSAGE(fixtures["kind"].get<std::string>(), ": ", fixtures["cases"].size(), " requests (", refused,
            " refused), ", questions, " questions, max |probability difference| ", worst, ", top option differs on ",
            top_differs);
}
