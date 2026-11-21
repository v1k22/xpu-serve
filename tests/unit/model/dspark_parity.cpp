// Inkling 2's block drafter (DSpark) against its reference rounds (tests/parity/model/dspark_reference.py), for the
// checkpoint in XS_TEST_MODEL_DIR. Opt-in, with the XS_PARITY_* variables of model_parity.cpp; the references are
// <fixtures>/dspark.json.

#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "engine/spec/drafter.h"
#include "models/model.h"
#include "parity_util.h"
#include "test_support.h"

namespace {

using xs::test::env_or;

bool enabled() {
    if (!xs::test::model_dir().has_value()) return false;
    std::ifstream in(*xs::test::model_dir() / "config.json");
    return in.good() && nlohmann::json::parse(in).contains("mtp_config") &&
           std::filesystem::exists(xs::test::model_fixtures() / "dspark.json");
}

}  // namespace

// Each round: the block after the tokens committed so far, through the engine's runner (which hands the head its
// context row), then the target over the pending token and the reference's drafts, keeping what the reference
// counted as accepted (so both sides draft from the same sequence).
TEST_CASE("Inkling 2's drafter matches the reference" * doctest::skip(!enabled())) {
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    std::ifstream mf(xs::test::model_fixtures() / "dspark.json");
    const auto manifest = nlohmann::json::parse(mf);
    const int n = manifest["block_size"].get<int>() - 1;
    auto model = xs::load_model(*xs::test::model_dir(), {.vision = false, .speculation = true, .draft_shortlist = 0});
    const xs::Speculation* spec = model->speculation();
    REQUIRE(spec != nullptr);
    REQUIRE(spec->block() != nullptr);
    REQUIRE(spec->drafter_name() == "dspark");
    REQUIRE(spec->block()->block_drafts() == n);

    xs::SessionOptions options = xs::test::parity_options();
    options.rewind_tokens = n + 1;
    const xs::graph::Graph* programs[] = {&model->graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                          &spec->draft_graph()};
    auto session = xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"))->compile(programs, model->weights(), options);
    auto drafter = xs::spec::make_drafter(*spec, *session);
    // A draft that leads the runner-up by more than this must be ours at f16 too.
    const double clear_lead = 0.5;

    for (const auto& c : manifest["cases"]) {
        const auto ids = c["input_ids"].get<std::vector<int32_t>>();
        session->reset();
        drafter->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(ids, 0, {}, inputs);
        session->run(xs::spec::kPrefill, inputs, outputs);
        drafter->add(outputs, static_cast<int64_t>(ids.size()), 0, ids);
        size_t same = 0, all = 0;
        for (const auto& round : c["rounds"]) {
            const int32_t anchor = round["anchor"].get<int32_t>();
            const auto want = round["drafts"].get<std::vector<int32_t>>();
            const auto leads = round["leads"].get<std::vector<double>>();
            const int64_t position = session->num_cached(xs::spec::kVerify);
            const std::vector<int32_t> drafts = drafter->propose(anchor, n);
            REQUIRE(drafts.size() == want.size());
            for (size_t i = 0; i < drafts.size(); ++i) {
                all += 1;
                same += drafts[i] == want[i];
                // a draft follows the ones before it: past a row that differs, the rest need not agree
                if (drafts[i] != want[i]) {
                    if (f32 || leads[i] > clear_lead) CHECK(drafts[i] == want[i]);
                    break;
                }
            }
            std::vector<int32_t> step{anchor};
            step.insert(step.end(), want.begin(), want.end());
            const int64_t keep = round["accepted"].get<int64_t>() + 1;
            inputs.clear();
            outputs.clear();
            model->prepare_inputs(step, position, {}, inputs);
            session->run(xs::spec::kVerify, inputs, outputs);
            session->truncate(xs::spec::kVerify, position + keep);
            drafter->add(outputs, keep, position, step);
        }
        MESSAGE("dspark ", c["name"].get<std::string>(), ": ", same, " of ", all, " drafts the reference's over ",
                c["rounds"].size(), " rounds");
        CHECK(all > 0);
    }
}
