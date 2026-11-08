// End-to-end parity of a model's step graph against Hugging Face reference outputs
// (tests/parity/model/hf_reference.py), for whichever model XS_TEST_MODEL_DIR holds. Opt-in: needs XS_TEST_MODEL_DIR
// and the generated fixtures.
//
//   XS_PARITY_BACKEND   cpu-ref (default) | sycl
//   XS_PARITY_DEVICE    backend device, e.g. GPU
//   XS_PARITY_PRECISION f32 (default) | f16
//   XS_PARITY_KV_DTYPE  auto (default) | i8 (the int8 KV cache)
//   XS_PARITY_FIXTURES  default: <source>/tests/fixtures/generated/<model_type of the model>

#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "engine/engine.h"
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

}  // namespace

TEST_CASE("model matches Hugging Face logits and greedy output" * doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const std::string backend_name = env_or("XS_PARITY_BACKEND", "cpu-ref");
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.9999 : 0.995;
    const size_t greedy_must_match = f32 ? 32 : 8;

    auto model = xs::load_model(*xs::test::model_dir());
    xs::SessionOptions options;
    options.device = env_or("XS_PARITY_DEVICE", "");
    options.precision = f32 ? xs::DType::F32 : xs::DType::F16;
    options.max_context = 4096;
    if (env_or("XS_PARITY_KV_DTYPE", "auto") == "i8") options.kv_dtype = xs::DType::I8;
    auto session = xs::make_backend(backend_name)->compile(model->graph(), model->weights(), options);

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
