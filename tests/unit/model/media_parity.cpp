// End-to-end parity of a model that reads audio or images (Qwen3-ASR, Qwen3-Omni's thinker) against Hugging Face
// reference outputs (tests/parity/model/hf_media_reference.py), for whichever model XS_TEST_MODEL_DIR holds. Per clip:
// the log-mel features and the audio encoder's rows; per image: the processor's patches and the vision tower's rows
// (merged and deepstack); then the prompt (the case's messages through the chat template and tokenizer, a media token
// per encoder row), the last prompt position's logits at the positions the engine gives (Qwen3-Omni's on three axes)
// and the greedy continuation through the engine. Opt-in: needs XS_TEST_MODEL_DIR and the fixtures
// (XS_PARITY_FIXTURES, ...: as model_parity.cpp).

#include <doctest/doctest.h>

#include <array>
#include <fstream>
#include <nlohmann/json.hpp>

#include "audio/audio.h"
#include "audio/mel.h"
#include "audio/resample.h"
#include "backend/backend.h"
#include "chat/chat_template.h"
#include "engine/engine.h"
#include "engine/media.h"
#include "image/image.h"
#include "models/model.h"
#include "parity_util.h"
#include "test_support.h"
#include "tokenizer/tokenizer.h"

namespace {

using xs::test::compare;
using xs::test::Compare;
using xs::test::env_or;
using xs::test::parity_options;
using xs::test::read_f32;

std::filesystem::path fixtures() { return xs::test::model_fixtures(); }

bool enabled() {
    if (!xs::test::model_dir().has_value() || !std::filesystem::exists(fixtures() / "manifest.json")) return false;
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    for (const auto& c : manifest["cases"]) {
        if (c.contains("wav") || c.contains("image")) return true;
    }
    return false;
}

std::span<const uint8_t> bytes_of(const std::vector<char>& raw) {
    return {reinterpret_cast<const uint8_t*>(raw.data()), raw.size()};
}

}  // namespace

TEST_CASE("media input matches Hugging Face: features, encoder rows, prompt, logits, greedy" *
          doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.9999 : 0.995;
    const size_t greedy_must_match = f32 ? 32 : 8;

    const std::filesystem::path dir = *xs::test::model_dir();
    auto model = xs::load_model(dir);
    const auto tokenizer = xs::tok::Tokenizer::load(dir);
    const auto chat = xs::chat::ChatTemplate::load(dir);
    const auto backend = xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"));
    xs::SessionOptions encoder_options = parity_options();
    encoder_options.max_context = int64_t{1} << 20;
    xs::EncoderSessions sessions;
    const std::array<const xs::MediaEncoder*, xs::kModalities> encoders{model->vision(), model->audio()};
    for (size_t k = 0; k < xs::kModalities; ++k) {
        if (encoders[k] != nullptr) {
            sessions[k] = backend->compile(encoders[k]->programs(), model->weights(), encoder_options);
        }
    }
    auto session = backend->compile(model->graph(), model->weights(), parity_options());
    const int64_t width = model->info().media_width > 0 ? model->info().media_width : model->info().hidden_size;

    struct Prepared {
        std::vector<int32_t> prompt;
        std::vector<xs::MediaInput> media;
        std::vector<float> rows;  // the encoder's rows, [media tokens, width]
    };
    std::vector<Prepared> prepared;
    for (const auto& c : manifest["cases"]) {
        const std::string name = c["name"];
        CAPTURE(name);
        Prepared p;
        // The encoder's rows of one item, run by run, as wide as the model's media rows.
        const auto encode = [&](const xs::MediaInput& m, xs::Modality kind) {
            for (const xs::EncoderRun& run : m.runs) {
                xs::TensorMap out;
                sessions[static_cast<size_t>(kind)]->run(run.program, run.inputs, out);
                const xs::HostTensor& e = out.at("embeddings");
                for (int64_t r = 0; r < run.rows; ++r) {
                    std::vector<float> row(static_cast<size_t>(width), 0.0f);
                    std::copy_n(e.data<float>() + r * e.dim(1), e.dim(1), row.begin());
                    p.rows.insert(p.rows.end(), row.begin(), row.end());
                }
            }
        };
        if (c.contains("wav")) {
            const xs::AudioEncoder* tower = model->audio();
            REQUIRE_MESSAGE(tower != nullptr, "the model has no audio encoder");
            const std::vector<char> raw = xs::test::read_bytes(fixtures() / c["wav"].get<std::string>());
            const xs::audio::Audio clip = xs::audio::decode(bytes_of(raw));
            REQUIRE(clip.sample_rate == c["sample_rate"].get<int>());

            // The front end: resampling to 16 kHz and the log-mel features (a tower's), or the raw frames (Gemma 4
            // 12B's, checked after prepare).
            const std::vector<float> pcm = clip.sample_rate == 16000
                                               ? clip.samples
                                               : xs::audio::resample(clip.samples, clip.sample_rate, 16000);
            if (c.contains("mel")) {
                xs::audio::MelParams mel_params;
                mel_params.min_length = manifest.value("mel_min_length", mel_params.min_length);
                const xs::HostTensor mel = xs::audio::log_mel_spectrogram(pcm, mel_params);
                const std::vector<float> ref_mel = read_f32(fixtures() / c["mel"].get<std::string>());
                REQUIRE(mel.dim(1) == c["mel_frames"].get<int64_t>());
                double mel_diff = 0;
                for (size_t i = 0; i < ref_mel.size(); ++i) {
                    mel_diff = std::max(mel_diff, static_cast<double>(std::fabs(mel.data<float>()[i] - ref_mel[i])));
                }
                MESSAGE(name, ": ", clip.samples.size(), " samples at ", clip.sample_rate, " Hz, ", mel.dim(1),
                        " mel frames, max |diff| ", mel_diff);
                CHECK(mel_diff < 2e-3);
            }

            xs::MediaInput m = tower->prepare(clip);
            if (c.contains("frames")) {
                const xs::HostTensor& frames = m.runs.at(0).inputs.at("frames");
                const std::vector<float> ref_frames = read_f32(fixtures() / c["frames"].get<std::string>());
                REQUIRE(static_cast<size_t>(frames.numel()) == ref_frames.size());
                double frame_diff = 0;
                for (size_t i = 0; i < ref_frames.size(); ++i) {
                    frame_diff = std::max(frame_diff,
                                          static_cast<double>(std::fabs(frames.data<float>()[i] - ref_frames[i])));
                }
                MESSAGE(name, ": ", clip.samples.size(), " samples at ", clip.sample_rate, " Hz, ", frames.dim(0),
                        " frames, max |diff| ", frame_diff);
                CHECK(frame_diff < 1e-5);
            }
            m.key = xs::audio::content_hash(clip);
            REQUIRE(m.num_tokens == c["rows"].get<int64_t>());
            encode(m, xs::Modality::Audio);
            std::vector<float> rows;  // the audio rows alone, to compare with the reference's
            for (int64_t r = 0; r < m.num_tokens; ++r) {
                const float* row = p.rows.data() + r * width;
                rows.insert(rows.end(), row, row + model->info().hidden_size);
            }
            const Compare r = compare(rows, read_f32(fixtures() / c["audio_rows"].get<std::string>()));
            MESSAGE(name, ": ", m.runs.size(), " encoder runs, ", m.num_tokens, " rows, cosine ", r.cosine,
                    ", max |diff| ", r.max_abs);
            CHECK(r.cosine >= min_cosine);
            p.media.push_back(std::move(m));
        } else if (c.contains("image")) {
            const xs::VisionEncoder* tower = model->vision();
            REQUIRE_MESSAGE(tower != nullptr, "the model has no vision encoder");
            const std::vector<char> raw = xs::test::read_bytes(fixtures() / c["image"].get<std::string>());
            const xs::image::Image img = xs::image::decode(bytes_of(raw));
            xs::MediaInput m = tower->prepare(img);
            m.key = xs::image::content_hash(img);
            REQUIRE(m.num_tokens == c["rows"].get<int64_t>());
            // The processor: the patches in the reference's order (a token's patches together).
            const xs::HostTensor& pixels = m.runs.at(0).inputs.at("pixels");
            const std::vector<float> ref_pixels = read_f32(fixtures() / c["pixels"].get<std::string>());
            REQUIRE(static_cast<size_t>(pixels.numel()) == ref_pixels.size());
            double pixel_diff = 0;
            for (size_t i = 0; i < ref_pixels.size(); ++i) {
                const double d = std::fabs(pixels.data<float>()[i] - ref_pixels[i]);
                pixel_diff = std::max(pixel_diff, d);
            }
            MESSAGE(name, ": grid ", c["grid"].dump(), ", ", m.num_tokens, " tokens, patches max |diff| ", pixel_diff);
            CHECK(pixel_diff < 1e-5);
            encode(m, xs::Modality::Image);
            const Compare r = compare(p.rows, read_f32(fixtures() / c["image_rows"].get<std::string>()));
            MESSAGE(name, ": vision rows (merged and deepstack) cosine ", r.cosine, ", max |diff| ", r.max_abs);
            CHECK(r.cosine >= min_cosine);
            p.media.push_back(std::move(m));
        }
        const std::string text = chat->render(c["messages"], nullptr, true, nlohmann::json::object());
        p.prompt = xs::expand_media_placeholders(tokenizer->encode(text), p.media, encoders);
        CHECK(p.prompt == c["input_ids"].get<std::vector<int32_t>>());

        // Prompt logits straight from the session: the encoder's rows at the media tokens, at the engine's positions.
        std::vector<const float*> row_of(p.prompt.size(), nullptr);
        std::vector<int32_t> item(p.prompt.size(), -1);
        std::vector<int64_t> row(p.prompt.size(), 0);
        int64_t next = 0;
        for (size_t i = 0; i < p.prompt.size() && !p.media.empty(); ++i) {
            const xs::MediaEncoder* e = encoders[static_cast<size_t>(p.media[0].modality)];
            if (p.prompt[i] != e->soft_token()) continue;
            row_of[i] = p.rows.data() + next * width;
            item[i] = 0;
            row[i] = next++;
        }
        const xs::PromptPositions positions = xs::rotary_positions(item, row, p.media);
        session->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_positioned_inputs(p.prompt, positions.positions,
                                         p.media.empty() ? std::span<const float* const>() : row_of, inputs);
        session->run(inputs, outputs);
        const auto& logits = outputs.at("logits");
        const Compare r = compare(std::span(logits.data<float>(), static_cast<size_t>(logits.numel())),
                                  read_f32(fixtures() / c["last_logits"].get<std::string>()));
        MESSAGE(name, ": ", p.prompt.size(), " prompt tokens, logits cosine ", r.cosine, ", max |diff| ", r.max_abs,
                ", argmax ", r.argmax_ours, " vs ", r.argmax_ref);
        CHECK(r.cosine >= min_cosine);
        CHECK(r.argmax_ours == r.argmax_ref);
        prepared.push_back(std::move(p));
    }

    // Greedy continuations through the engine, which encodes the media itself.
    xs::EngineOptions eo;
    eo.max_context = 4096;
    xs::Engine engine(std::move(model), std::move(session), std::move(sessions), eo);
    size_t k = 0;
    for (const auto& c : manifest["cases"]) {
        const auto ref = c["greedy"].get<std::vector<int32_t>>();
        xs::GenerateRequest req;
        req.prompt = prepared[k].prompt;
        req.media = prepared[k++].media;
        req.sampling.temperature = 0.0f;
        req.max_new_tokens = static_cast<int>(ref.size());
        const xs::GenerateResult r = engine.generate(req, {});
        REQUIRE_MESSAGE(r.finish != xs::FinishReason::Error, r.error);
        size_t same = 0;
        while (same < ref.size() && same < r.tokens.size() && ref[same] == r.tokens[same]) ++same;
        MESSAGE(c["name"].get<std::string>(), ": greedy matches ", same, "/", ref.size(), " tokens (encode ",
                r.stats.encode_ms, " ms, prefill ", r.stats.prefill_ms, " ms, decode ", r.stats.decode_tok_per_s(),
                " tok/s)");
        // f16 may flip a near tie (DESIGN.md 10.4): the reference ranks our token within 1% of its logit range.
        bool near_tie = false;
        if (!f32 && same < ref.size() && same < r.tokens.size() && c.contains("greedy_top")) {
            const auto& step = c["greedy_top"][same];
            const auto ids = step["ids"].get<std::vector<int32_t>>();
            const auto logits = step["logits"].get<std::vector<float>>();
            for (size_t j = 0; j < ids.size(); ++j) {
                if (ids[j] == r.tokens[same]) near_tie = logits[0] - logits[j] < 0.01f * step["range"].get<float>();
            }
            if (near_tie) MESSAGE("  step ", same, " is a near tie");
        }
        CHECK((same >= std::min(greedy_must_match, ref.size()) || near_tie));
    }
}
