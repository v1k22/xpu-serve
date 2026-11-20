// End-to-end parity of a model that hears (Qwen3-ASR, Qwen3-Omni's thinker) against Hugging Face reference outputs
// (tests/parity/model/hf_audio_reference.py), for whichever model XS_TEST_MODEL_DIR holds: per clip the log-mel
// features, the audio encoder's rows, the prompt (the case's messages through the chat template and tokenizer, the
// audio token once per row), the last prompt position's logits and the greedy continuation through the engine. Opt-in: needs XS_TEST_MODEL_DIR and the
// fixtures (XS_PARITY_FIXTURES, ...: as model_parity.cpp).

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
        if (c.contains("wav")) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("audio input matches Hugging Face: mel, encoder rows, prompt, logits, greedy" *
          doctest::skip(!enabled())) {
    std::ifstream mf(fixtures() / "manifest.json");
    const auto manifest = nlohmann::json::parse(mf);
    const bool f32 = env_or("XS_PARITY_PRECISION", "f32") == "f32";
    const double min_cosine = f32 ? 0.9999 : 0.995;
    const size_t greedy_must_match = f32 ? 32 : 8;

    const std::filesystem::path dir = *xs::test::model_dir();
    auto model = xs::load_model(dir);
    const xs::AudioEncoder* tower = model->audio();
    REQUIRE_MESSAGE(tower != nullptr, "the model has no audio encoder");
    const auto tokenizer = xs::tok::Tokenizer::load(dir);
    const auto chat = xs::chat::ChatTemplate::load(dir);
    const auto backend = xs::make_backend(env_or("XS_PARITY_BACKEND", "cpu-ref"));
    xs::SessionOptions encoder_options = parity_options();
    encoder_options.max_context = int64_t{1} << 20;
    auto audio_session = backend->compile(tower->programs(), model->weights(), encoder_options);
    auto session = backend->compile(model->graph(), model->weights(), parity_options());
    const int64_t hidden = model->info().hidden_size;
    const std::array<const xs::MediaEncoder*, xs::kModalities> encoders{nullptr, tower};

    struct Prepared {
        std::vector<int32_t> prompt;
        std::vector<xs::MediaInput> media;
        std::vector<float> rows;  // the encoder's rows, [media tokens, hidden]
    };
    std::vector<Prepared> prepared;
    for (const auto& c : manifest["cases"]) {
        const std::string name = c["name"];
        CAPTURE(name);
        Prepared p;
        if (c.contains("wav")) {
            const std::vector<char> raw = xs::test::read_bytes(fixtures() / c["wav"].get<std::string>());
            const xs::audio::Audio clip =
                xs::audio::decode(std::span(reinterpret_cast<const uint8_t*>(raw.data()), raw.size()));
            REQUIRE(clip.sample_rate == c["sample_rate"].get<int>());

            // The front end: resampling to 16 kHz and the log-mel features.
            const std::vector<float> pcm = clip.sample_rate == 16000
                                               ? clip.samples
                                               : xs::audio::resample(clip.samples, clip.sample_rate, 16000);
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

            // The encoder, run by run.
            xs::MediaInput m = tower->prepare(clip);
            m.key = xs::audio::content_hash(clip);
            REQUIRE(m.num_tokens == c["rows"].get<int64_t>());
            for (const xs::EncoderRun& run : m.runs) {
                xs::TensorMap out;
                audio_session->run(run.program, run.inputs, out);
                const xs::HostTensor& e = out.at("embeddings");
                p.rows.insert(p.rows.end(), e.data<float>(), e.data<float>() + run.rows * hidden);
            }
            const Compare r = compare(p.rows, read_f32(fixtures() / c["audio_rows"].get<std::string>()));
            MESSAGE(name, ": ", m.runs.size(), " encoder runs, ", m.num_tokens, " rows, cosine ", r.cosine,
                    ", max |diff| ", r.max_abs);
            CHECK(r.cosine >= min_cosine);
            p.media.push_back(std::move(m));
        }
        const std::string text = chat->render(c["messages"], nullptr, true, nlohmann::json::object());
        p.prompt = xs::expand_media_placeholders(tokenizer->encode(text), p.media, encoders);
        CHECK(p.prompt == c["input_ids"].get<std::vector<int32_t>>());

        // Prompt logits straight from the session, the encoder's rows at the audio tokens.
        std::vector<const float*> row_of(p.prompt.size(), nullptr);
        int64_t next = 0;
        for (size_t i = 0; i < p.prompt.size() && !p.media.empty(); ++i) {
            if (p.prompt[i] == tower->soft_token()) row_of[i] = p.rows.data() + (next++) * hidden;
        }
        session->reset();
        xs::TensorMap inputs, outputs;
        model->prepare_inputs(p.prompt, 0, p.media.empty() ? std::span<const float* const>() : row_of, inputs);
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

    // Greedy continuations through the engine, which encodes the audio itself.
    xs::EncoderSessions sessions;
    sessions[static_cast<size_t>(xs::Modality::Audio)] = std::move(audio_session);
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
        CHECK(same >= std::min(greedy_must_match, ref.size()));
    }
}
