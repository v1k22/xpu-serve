// A block drafter (DFlash, DFlash 2, DSpine) against its reference rounds
// (tests/parity/model/block_drafter_reference.py), for the target in XS_TEST_MODEL_DIR and the drafter checkpoint in
// XS_TEST_DRAFTER_DIR. Opt-in, with the XS_PARITY_* variables of model_parity.cpp; the references are
// <fixtures>/<drafter name>.json.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "engine/spec/block_drafter.h"
#include "models/model.h"
#include "parity_util.h"
#include "test_support.h"

namespace {

using xs::test::env_or;

bool enabled() {
    const std::string drafter = env_or("XS_TEST_DRAFTER_DIR", "");
    if (!xs::test::model_dir().has_value() || drafter.empty()) return false;
    std::ifstream in(std::filesystem::path(drafter) / "config.json");
    return in.good() && nlohmann::json::parse(in).contains("dflash_config");
}

int64_t index_of(const std::vector<int32_t>& row, int32_t token) {
    const auto it = std::find(row.begin(), row.end(), token);
    return it == row.end() ? -1 : it - row.begin();
}

}  // namespace

// Each round: the block after the tokens committed so far, then the target over the pending token and the
// reference's drafts, keeping what the reference counted as accepted (so both sides draft from the same sequence).
// Candidates are compared as sets and scores over the candidates both sides have, each row of scores up to a positive
// factor (DSpine's reference ranks normalized states, the graph unnormalized ones).
TEST_CASE("a block drafter matches the reference" * doctest::skip(!enabled())) {
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    auto model = xs::load_model(*xs::test::model_dir(), {.vision = false,
                                                         .speculation = true,
                                                         .drafter = env_or("XS_TEST_DRAFTER_DIR", ""),
                                                         .draft_shortlist = 0});
    const xs::Speculation* spec = model->speculation();
    REQUIRE(spec != nullptr);
    REQUIRE(spec->block() != nullptr);
    const std::string name(spec->drafter_name());
    const std::filesystem::path fixtures = xs::test::model_fixtures();
    std::ifstream mf(fixtures / (name + ".json"));
    REQUIRE_MESSAGE(mf.good(), "no reference rounds: ", (fixtures / (name + ".json")).string());
    const auto manifest = nlohmann::json::parse(mf);
    const int64_t n = spec->block()->block_drafts();
    REQUIRE(manifest["block_size"].get<int64_t>() == n + 1);

    xs::SessionOptions options = xs::test::parity_options();
    options.rewind_tokens = static_cast<int>(n) + 1;
    const xs::graph::Graph* programs[] = {&model->graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                          &spec->draft_graph()};
    auto session = xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"))->compile(programs, model->weights(), options);

    // Scores may differ by this much of a row's largest (a quantized KV cache rounds the context's keys and values),
    // and so many of the candidates must be the reference's.
    const bool kv_quantized = options.kv != xs::quant::KvFormats{};
    const double score_tolerance = kv_quantized ? 8e-2 : f32 ? 2e-3 : 3e-2;
    const double shared_candidates = kv_quantized ? 0.85 : f32 ? 1.0 : 0.9;
    for (const auto& c : manifest["cases"]) {
        const auto ids = c["input_ids"].get<std::vector<int32_t>>();
        session->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(ids, 0, {}, inputs);
        session->run(xs::spec::kPrefill, inputs, outputs);
        double worst = 0.0;
        size_t same_drafts = 0, all_drafts = 0, shared = 0, listed = 0;
        for (const auto& round : c["rounds"]) {
            const int32_t anchor = round["anchor"].get<int32_t>();
            const auto want = round["drafts"].get<std::vector<int32_t>>();
            const int64_t position = session->num_cached(xs::spec::kDraft);
            inputs.clear();
            outputs.clear();
            spec->block()->prepare_block(anchor, position, inputs);
            session->run(xs::spec::kDraft, inputs, outputs);
            session->truncate(xs::spec::kDraft, position);
            const std::vector<int32_t> drafts = xs::spec::block_drafts(outputs);
            REQUIRE(drafts.size() == want.size());
            all_drafts += drafts.size();
            for (size_t i = 0; i < drafts.size(); ++i) same_drafts += drafts[i] == want[i];

            if (round.contains("candidates")) {
                const auto ref_cand = round["candidates"].get<std::vector<std::vector<int32_t>>>();
                const std::vector<float> ref = xs::test::read_f32(fixtures / round["scores"].get<std::string>());
                const xs::HostTensor& cand = outputs.at("candidates");
                const xs::HostTensor& scores = outputs.at("scores");
                const int64_t k = cand.dim(1);
                REQUIRE(ref.size() == static_cast<size_t>(n * k * k));
                const auto ours_row = [&](int64_t i) {
                    return std::vector<int32_t>(cand.data<int32_t>() + i * k, cand.data<int32_t>() + (i + 1) * k);
                };
                int64_t from = 0;  // the candidate our path took for the draft before
                for (int64_t i = 0; i < n; ++i) {
                    const std::vector<int32_t> mine = ours_row(i);
                    const std::vector<int32_t>& theirs = ref_cand[static_cast<size_t>(i)];
                    listed += static_cast<size_t>(k);
                    for (const int32_t t : mine) shared += index_of(theirs, t) >= 0;
                    // every sender both sides have: the pending token for the first draft
                    for (int64_t j = 0; j < (i == 0 ? 1 : k); ++j) {
                        const int64_t ref_j = i == 0 ? 0 : index_of(ref_cand[static_cast<size_t>(i - 1)], ours_row(i - 1)[static_cast<size_t>(j)]);
                        if (ref_j < 0) continue;
                        const float* o = scores.data<float>() + (i * k + j) * k;
                        const float* r = ref.data() + (i * k + ref_j) * k;
                        double dot = 0, r2 = 0, largest = 0, best = -1e30;
                        std::vector<std::pair<double, double>> pairs;
                        for (int64_t a = 0; a < k; ++a) {
                            const int64_t b = index_of(theirs, mine[static_cast<size_t>(a)]);
                            if (b < 0) continue;
                            pairs.emplace_back(o[a], r[b]);
                            dot += static_cast<double>(o[a]) * r[b];
                            r2 += static_cast<double>(r[b]) * r[b];
                            largest = std::max(largest, std::fabs(static_cast<double>(r[b])));
                            best = std::max(best, static_cast<double>(r[b]));
                        }
                        if (pairs.size() < 2) continue;
                        const double factor = dot / r2;
                        CHECK(factor > 0.0);
                        if (name != "dspine") CHECK(factor == doctest::Approx(1.0).epsilon(score_tolerance));
                        for (const auto& [mine_score, ref_score] : pairs) {
                            worst = std::max(worst, std::fabs(mine_score - factor * ref_score) / (factor * largest));
                        }
                        // our path's pick, by the reference's own scores: the best of the candidates both list, or
                        // within rounding of it
                        if (j == from) {
                            const int64_t pick = std::max_element(o, o + k) - o;
                            const int64_t ref_pick = index_of(theirs, mine[static_cast<size_t>(pick)]);
                            if (ref_pick >= 0) CHECK(r[ref_pick] >= best - 2 * score_tolerance * largest);
                        }
                    }
                    const float* row = scores.data<float>() + (i * k + from) * k;
                    from = std::max_element(row, row + k) - row;
                }
            } else if (!f32 || kv_quantized) {
                // a clear winner must be ours too
                const auto margins = round["margins"].get<std::vector<double>>();
                for (size_t i = 0; i < drafts.size(); ++i) {
                    if (margins[i] > score_tolerance) CHECK(drafts[i] == want[i]);
                }
            }
            if (f32 && !kv_quantized) CHECK(drafts == want);

            std::vector<int32_t> step{anchor};
            step.insert(step.end(), want.begin(), want.end());
            const int64_t accepted = round["accepted"].get<int64_t>();
            inputs.clear();
            outputs.clear();
            model->prepare_inputs(step, position, {}, inputs);
            session->run(xs::spec::kVerify, inputs, outputs);
            session->truncate(xs::spec::kVerify, position + 1 + accepted);
        }
        MESSAGE(name, " ", c["name"].get<std::string>(), ": ", same_drafts, " of ", all_drafts, " drafts the reference's, ",
                shared, " of ", listed, " candidates shared, scores off by ", worst, " of a row's largest");
        CHECK(worst <= score_tolerance);
        CHECK(static_cast<double>(shared) >= shared_candidates * static_cast<double>(listed));
    }
}
