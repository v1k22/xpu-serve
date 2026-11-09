// End-to-end parity of a model's step graph (and its MTP drafter, when the fixtures have one) against Hugging Face
// reference outputs (tests/parity/model/hf_reference.py), for whichever model XS_TEST_MODEL_DIR holds. Opt-in: needs
// XS_TEST_MODEL_DIR and the generated fixtures.
//
//   XS_PARITY_BACKEND   cpu-ref (default) | sycl
//   XS_PARITY_DEVICE    backend device, e.g. GPU
//   XS_PARITY_PRECISION f32 (default) | f16
//   XS_PARITY_KV_DTYPE  auto (default) | i8 (the int8 KV cache)
//   XS_PARITY_FIXTURES  default: <source>/tests/fixtures/generated/<model_type of the model>
//   XS_TEST_DRAFTER_DIR the drafter checkpoint the fixtures were made with (hf_reference.py --assistant)

#include <doctest/doctest.h>

#include <cmath>
#include <numeric>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "engine/engine.h"
#include "engine/spec/draft_runner.h"
#include "models/model.h"
#include "parity_util.h"
#include "test_support.h"

namespace {

using xs::test::compare;
using xs::test::Compare;
using xs::test::env_or;
using xs::test::read_f32;

std::filesystem::path fixtures() {
    std::string model_type = "unknown";
    if (const auto dir = xs::test::model_dir()) {
        std::ifstream in(*dir / "config.json");
        if (in.good()) model_type = nlohmann::json::parse(in).value("model_type", model_type);
    }
    return env_or("XS_PARITY_FIXTURES", std::string(XS_SOURCE_DIR) + "/tests/fixtures/generated/" + model_type);
}

bool enabled() { return xs::test::model_dir().has_value() && std::filesystem::exists(fixtures() / "manifest.json"); }

xs::SessionOptions parity_options() {
    xs::SessionOptions options;
    options.device = env_or("XS_PARITY_DEVICE", "");
    options.precision = env_or("XS_PARITY_PRECISION", "f32") == "f32" ? xs::DType::F32 : xs::DType::F16;
    options.max_context = 4096;
    if (env_or("XS_PARITY_KV_DTYPE", "auto") == "i8") options.kv_dtype = xs::DType::I8;
    return options;
}

}  // namespace

TEST_CASE("model matches Hugging Face logits and greedy output" * doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const std::string backend_name = env_or("XS_PARITY_BACKEND", "cpu-ref");
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.9999 : 0.995;
    const size_t greedy_must_match = f32 ? 32 : 8;

    auto model = xs::load_model(*xs::test::model_dir());
    auto session = xs::make_backend(backend_name)->compile(model->graph(), model->weights(), parity_options());

    // Prompt logits straight from the session.
    for (const auto& c : manifest["cases"]) {
        const auto ids = c["input_ids"].get<std::vector<int32_t>>();
        session->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(ids, 0, {}, inputs);
        session->run(inputs, outputs);
        const auto& logits = outputs.at("logits");
        const std::vector<float> ours(logits.data<float>(), logits.data<float>() + logits.numel());
        const Compare r = compare(ours, read_f32(fixtures() / c["last_logits"].get<std::string>()));
        MESSAGE(c["name"].get<std::string>(), ": ", ids.size(), " tokens, cosine ", r.cosine, ", max |diff| ",
                r.max_abs, ", argmax ", r.argmax_ours, " vs ", r.argmax_ref);
        CHECK(r.cosine >= min_cosine);
        CHECK(r.argmax_ours == r.argmax_ref);
    }

    // Greedy continuation through the engine (prefill + one-token decode steps). No stop tokens: the reference
    // continuation includes the stop token when one was produced, so it is compared like any other token.
    xs::EngineOptions eo;
    eo.max_context = 4096;
    xs::Engine engine(std::move(model), std::move(session), nullptr, eo);
    for (const auto& c : manifest["cases"]) {
        const auto ref = c["greedy"].get<std::vector<int32_t>>();
        xs::GenerateRequest req;
        req.prompt = c["input_ids"].get<std::vector<int32_t>>();
        req.sampling.temperature = 0.0f;
        req.max_new_tokens = static_cast<int>(ref.size());
        const xs::GenerateResult r = engine.generate(req, {});
        REQUIRE_MESSAGE(r.finish != xs::FinishReason::Error, r.error);
        size_t same = 0;
        while (same < ref.size() && same < r.tokens.size() && ref[same] == r.tokens[same]) ++same;
        MESSAGE(c["name"].get<std::string>(), ": greedy matches ", same, "/", ref.size(), " tokens; decode ",
                r.stats.decode_tok_per_s(), " tok/s");
        CHECK(same >= std::min(greedy_must_match, ref.size()));
    }
}

// The drafter after each prompt's pairs and the first greedy token: its logits, then the drafts it chains.
TEST_CASE("the drafter matches the reference (MTP head or assistant)" * doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    if (!manifest["cases"].at(0).contains("mtp_logits")) return;  // a model without a drafter
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    auto model = xs::load_model(*xs::test::model_dir(),
                                {.vision = false, .speculation = true, .drafter = env_or("XS_TEST_DRAFTER_DIR", "")});
    const xs::Speculation* spec = model->speculation();
    REQUIRE(spec != nullptr);
    xs::SessionOptions options = parity_options();
    options.rewind_tokens = 4;
    const xs::graph::Graph* programs[] = {&model->graph(), &spec->prefill_graph(), &spec->verify_graph(),
                                          &spec->draft_graph()};
    auto session = xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"))->compile(programs, model->weights(), options);
    xs::spec::DraftRunner drafter(*spec, *session);
    for (const auto& c : manifest["cases"]) {
        const auto ids = c["input_ids"].get<std::vector<int32_t>>();
        const int32_t first = c["greedy"].at(0).get<int32_t>();
        session->reset();
        drafter.reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(ids, 0, {}, inputs);
        session->run(xs::spec::kPrefill, inputs, outputs);
        const xs::HostTensor hidden = std::move(outputs.at("hidden"));

        // The pairs the drafter takes after the prompt: all of them, or (without state) the last alone.
        std::vector<int32_t> next(ids.begin() + 1, ids.end());
        next.push_back(first);
        std::span<const float> rows(hidden.data<float>(), static_cast<size_t>(hidden.numel()));
        int64_t start = 0;
        if (!spec->drafter_keeps_state()) {
            start = static_cast<int64_t>(ids.size()) - 1;
            next = {first};
            rows = rows.last(static_cast<size_t>(hidden.dim(1)));
        }
        inputs.clear();
        spec->prepare_draft(next, rows, start, 0, inputs);
        session->run(xs::spec::kDraft, inputs, outputs);
        // Our logits and the reference's over the same tokens: the vocabulary, or the candidates of an ordered head
        // that the reference also scored (a centroid at the top-k boundary can flip on rounding).
        const auto& logits = outputs.at("logits");
        const std::vector<float> ref_all = read_f32(fixtures() / c["mtp_logits"].get<std::string>());
        std::vector<float> ours(logits.data<float>(), logits.data<float>() + logits.numel());
        std::vector<float> ref = ref_all;
        std::vector<int32_t> token_of(ours.size());
        std::iota(token_of.begin(), token_of.end(), 0);
        if (outputs.count("candidates") != 0) {
            const auto& cand = outputs.at("candidates");
            const float mask = *std::min_element(ref_all.begin(), ref_all.end());
            std::vector<float> o, f;
            token_of.clear();
            for (int64_t i = 0; i < cand.numel(); ++i) {
                const int32_t t = cand.data<int32_t>()[i];
                if (ref_all.at(static_cast<size_t>(t)) == mask) continue;
                o.push_back(ours[static_cast<size_t>(i)]);
                f.push_back(ref_all[static_cast<size_t>(t)]);
                token_of.push_back(t);
            }
            CHECK(static_cast<double>(o.size()) >= 0.9 * static_cast<double>(cand.numel()));
            ours = std::move(o);
            ref = std::move(f);
        }
        const Compare r = compare(ours, ref);
        const int32_t ours_token = token_of[static_cast<size_t>(r.argmax_ours)];
        const auto ref_token = static_cast<int32_t>(std::max_element(ref_all.begin(), ref_all.end()) - ref_all.begin());
        if (spec->drafter_keeps_state()) session->truncate(xs::spec::kDraft, 0);

        drafter.add(hidden, hidden.dim(0), 0, ids);
        const std::vector<int32_t> drafts = drafter.propose(first, 3);
        const auto want = c["mtp_drafts"].get<std::vector<int32_t>>();
        MESSAGE(c["name"].get<std::string>(), ": drafter cosine ", r.cosine, ", argmax ", ours_token, " vs ", ref_token,
                "; drafts ", drafts[0], " ", drafts[1], " ", drafts[2], " vs ", want[0], " ", want[1], " ", want[2]);
        CHECK(r.cosine >= (f32 ? 0.9999 : 0.995));
        // f16 may flip a near tie (DESIGN.md 10.4): the reference must rank our token within 1% of its logit range.
        const float range = *std::max_element(ref_all.begin(), ref_all.end()) - *std::min_element(ref_all.begin(), ref_all.end());
        const bool near_tie = !f32 && ref_all[static_cast<size_t>(ref_token)] - ref_all[static_cast<size_t>(ours_token)] < 0.01f * range;
        CHECK((ours_token == ref_token || near_tie));
        if (ours_token == ref_token) CHECK(drafts[0] == want[0]);
        if (f32) CHECK(drafts == want);
    }
}
