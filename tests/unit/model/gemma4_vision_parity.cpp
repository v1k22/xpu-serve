// End-to-end parity of Gemma 4 image input against Hugging Face (tests/parity/model/hf_vision_reference.py):
// preprocessing, the vision tower, the expanded prompt, prompt logits and greedy continuations. Opt-in: needs
// XS_TEST_MODEL_DIR and the generated fixtures. Same XS_PARITY_* variables as model_parity.cpp; fixtures default to
// <source>/tests/fixtures/generated/gemma4_vision.

#include <doctest/doctest.h>

#include <map>
#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "chat/chat_template.h"
#include "engine/engine.h"
#include "engine/media.h"
#include "image/image.h"
#include "models/model.h"
#include "parity_util.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"
#include "util/error.h"

namespace {

using xs::test::compare;
using xs::test::Compare;
using xs::test::env_or;
using xs::test::read_f32;
using xs::str_cat;

std::filesystem::path fixtures() {
    return env_or("XS_PARITY_FIXTURES", std::string(XS_SOURCE_DIR) + "/tests/fixtures/generated/gemma4_vision");
}

bool enabled() { return xs::test::model_dir().has_value() && std::filesystem::exists(fixtures() / "manifest.json"); }

// PNGs decode exactly as Pillow does; JPEG decoders differ by a level here and there, which the bars allow for.
bool lossless(const std::string& file) { return file.size() > 4 && file.substr(file.size() - 4) == ".png"; }

}  // namespace

TEST_CASE("gemma4 vision matches Hugging Face preprocessing, tower, logits and greedy output" *
          doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const std::string backend_name = env_or("XS_PARITY_BACKEND", "cpu-ref");
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.9999 : 0.995;
    const size_t greedy_must_match = f32 ? 32 : 8;

    auto model = xs::load_model(*xs::test::model_dir());
    REQUIRE_MESSAGE(model->vision() != nullptr, "the model loaded without its vision encoder");
    const xs::VisionEncoder& vision = *model->vision();
    xs::SessionOptions options;
    options.device = env_or("XS_PARITY_DEVICE", "");
    options.precision = f32 ? xs::DType::F32 : xs::DType::F16;
    options.max_context = 4096;
    const auto backend = xs::make_backend(backend_name);
    auto vision_session = backend->compile(vision.graph(), model->weights(), options);
    const int64_t hidden = model->info().hidden_size;

    // Preprocessing and the tower, image by image.
    std::map<std::string, xs::MediaInput> media;
    std::map<std::string, xs::HostTensor> features;
    std::map<std::string, bool> exact_pixels;
    for (const auto& im : manifest["images"]) {
        const std::string name = im["name"];
        const std::string file = im["file"];
        CAPTURE(name);
        const std::vector<char> raw = xs::test::read_bytes(fixtures() / file);
        const xs::image::Image img =
            xs::image::decode(std::span(reinterpret_cast<const uint8_t*>(raw.data()), raw.size()));
        xs::MediaInput m = vision.prepare(img);
        m.key = xs::image::content_hash(img);
        CHECK(m.num_tokens == im["soft_tokens"].get<int64_t>());

        // The processor's pixel values are in [0, 1]; the tower's first step, 2 * (x - 0.5), is part of our input.
        const xs::HostTensor& pixels = m.encoder_inputs.at("pixels");
        const std::vector<float> ref_pixels = read_f32(fixtures() / im["pixels"].get<std::string>());
        REQUIRE(pixels.numel() == static_cast<int64_t>(ref_pixels.size()));
        double max_diff = 0, sum_diff = 0;
        for (size_t i = 0; i < ref_pixels.size(); ++i) {
            const double d = std::fabs(pixels.data<float>()[i] - 2.0f * (ref_pixels[i] - 0.5f));
            max_diff = std::max(max_diff, d);
            sum_diff += d;
        }
        const double mean_diff = sum_diff / static_cast<double>(ref_pixels.size());
        MESSAGE(name, " (", file, "): ", img.width, "x", img.height, " -> ", m.num_tokens, " soft tokens, pixel max |diff| ",
                max_diff, ", mean ", mean_diff);
        exact_pixels[name] = max_diff == 0;
        if (lossless(file)) {
            CHECK(max_diff == 0);
        } else {
            CHECK(mean_diff < 0.01);  // one 8-bit level is 2/255 = 0.0078 after the tower's 2x - 1
        }
        const auto ref_pos = im["positions"].get<std::vector<std::vector<int32_t>>>();
        const xs::HostTensor& pos = m.encoder_inputs.at("patch_pos");
        bool same_positions = pos.numel() == static_cast<int64_t>(2 * ref_pos.size());
        for (size_t i = 0; same_positions && i < ref_pos.size(); ++i) {
            same_positions = pos.data<int32_t>()[2 * i] == ref_pos[i][0] && pos.data<int32_t>()[2 * i + 1] == ref_pos[i][1];
        }
        CHECK(same_positions);

        xs::TensorMap out;
        vision_session->run(m.encoder_inputs, out);
        const xs::HostTensor& rows = out.at("embeddings");
        REQUIRE(rows.dim(1) == hidden);
        const std::vector<float> ref = read_f32(fixtures() / im["features"].get<std::string>());
        REQUIRE(static_cast<int64_t>(ref.size()) == m.num_tokens * hidden);
        const Compare c = compare(std::span(rows.data<float>(), ref.size()), ref);
        double worst_row = 1.0;
        for (int64_t r = 0; r < m.num_tokens; ++r) {
            const auto off = static_cast<size_t>(r * hidden);
            worst_row = std::min(worst_row, compare(std::span(rows.data<float>() + off, static_cast<size_t>(hidden)),
                                                    std::span(ref.data() + off, static_cast<size_t>(hidden)))
                                                .cosine);
        }
        MESSAGE(name, ": vision features cosine ", c.cosine, ", worst row ", worst_row, ", max |diff| ", c.max_abs);
        CHECK(c.cosine >= (exact_pixels[name] ? min_cosine : 0.99));
        features.emplace(name, rows.clone());
        media.emplace(name, std::move(m));
    }

    // The prompt our chat template, tokenizer and placeholder expansion build is HF's processor output.
    const auto tokenizer = xs::tok::Tokenizer::load(*xs::test::model_dir());
    const auto chat_template = xs::chat::ChatTemplate::load(*xs::test::model_dir());
    auto session = backend->compile(model->graph(), model->weights(), options);
    struct Case {
        std::string name;
        std::vector<int32_t> prompt;
        std::vector<xs::MediaInput> media;
        bool exact = true;  // every image decoded exactly as Pillow decodes it
    };
    std::vector<Case> cases;
    for (const auto& c : manifest["cases"]) {
        Case k;
        k.name = c["name"];
        CAPTURE(k.name);
        for (const auto& n : c["images"]) {
            k.media.push_back(media.at(n.get<std::string>()));
            k.exact = k.exact && exact_pixels.at(n.get<std::string>());
        }
        const std::string text = chat_template->render(c["messages"], nullptr, true);
        k.prompt = xs::expand_media_placeholders(tokenizer->encode(text), k.media, vision);
        CHECK(k.prompt == c["input_ids"].get<std::vector<int32_t>>());

        // Prompt logits: one session run with the soft-token rows spliced in.
        std::vector<const float*> rows(k.prompt.size(), nullptr);
        size_t item = 0;
        int64_t row = 0;
        for (size_t i = 0; i < k.prompt.size(); ++i) {
            if (k.prompt[i] != vision.soft_token()) continue;
            if (row == k.media[item].num_tokens) ++item, row = 0;
            rows[i] = features.at(c["images"][item].get<std::string>()).data<float>() + (row++) * hidden;
        }
        session->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(k.prompt, 0, rows, inputs);
        session->run(inputs, outputs);
        const xs::HostTensor& logits = outputs.at("logits");
        const Compare r = compare(std::span(logits.data<float>(), static_cast<size_t>(logits.numel())),
                                  read_f32(fixtures() / c["last_logits"].get<std::string>()));
        MESSAGE(k.name, ": ", k.prompt.size(), " tokens, cosine ", r.cosine, ", max |diff| ", r.max_abs, ", argmax ",
                r.argmax_ours, " vs ", r.argmax_ref);
        CHECK(r.cosine >= (k.exact ? min_cosine : 0.99));
        if (k.exact) CHECK(r.argmax_ours == r.argmax_ref);
        cases.push_back(std::move(k));
    }

    // Greedy continuations through the engine: once with the whole prompt in one prefill step, once with chunks of
    // 97 tokens so chunk boundaries fall inside images; both must reproduce the reference.
    xs::EngineOptions eo;
    eo.max_context = 4096;
    xs::Engine engine(std::move(model), std::move(session), std::move(vision_session), eo);
    for (const int64_t chunk : {int64_t{4096}, int64_t{97}}) {
        CAPTURE(chunk);
        engine.set_prefill_chunk(chunk);
        for (const Case& k : cases) {
            const auto& c = *std::find_if(manifest["cases"].begin(), manifest["cases"].end(),
                                          [&](const auto& m) { return m["name"] == k.name; });
            const auto ref = c["greedy"].get<std::vector<int32_t>>();
            xs::GenerateRequest req;
            req.prompt = k.prompt;
            req.media = k.media;
            req.sampling.temperature = 0.0f;
            req.max_new_tokens = static_cast<int>(ref.size());
            const xs::GenerateResult r = engine.generate(req, {});
            REQUIRE_MESSAGE(r.finish != xs::FinishReason::Error, r.error);
            size_t same = 0;
            while (same < ref.size() && same < r.tokens.size() && ref[same] == r.tokens[same]) ++same;
            const std::string ours = tokenizer->decode(r.tokens);
            MESSAGE(k.name, " (chunk ", chunk, "): greedy matches ", same, "/", ref.size(), "; encoded ",
                    r.stats.images_encoded, " images in ", r.stats.encode_ms, " ms; \"", ours, "\"",
                    same < ref.size() ? str_cat(" (reference: \"", c["greedy_text"].get<std::string>(), "\")") : "");
            if (k.exact) CHECK(same >= std::min(greedy_must_match, ref.size()));
        }
    }

    // The prefix cache tells images apart by their pixels: the same prompt with a different image key reuses only the
    // text before the image; the same request again reuses everything but the last token.
    const Case& k = cases.front();
    const auto first_image =
        std::find(k.prompt.begin(), k.prompt.end(), engine.model().vision()->soft_token()) - k.prompt.begin();
    xs::GenerateRequest req;
    req.prompt = k.prompt;
    req.media = k.media;
    req.sampling.temperature = 0.0f;
    req.max_new_tokens = 1;
    REQUIRE(engine.generate(req, {}).finish != xs::FinishReason::Error);
    req.media.front().key ^= 1;
    const xs::GenerateResult other = engine.generate(req, {});
    CHECK(other.stats.cached_tokens == first_image);
    const xs::GenerateResult again = engine.generate(req, {});
    CHECK(again.stats.cached_tokens == static_cast<int>(k.prompt.size()) - 1);
    CHECK(again.stats.images_encoded == 0);
}
