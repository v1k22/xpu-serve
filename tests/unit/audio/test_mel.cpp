// The FFT, and Qwen3-ASR's log-mel features against transformers (tests/fixtures/audio/mel_cases.json).
#include <doctest/doctest.h>

#include <complex>
#include <numbers>
#include <random>

#include "audio/fft.h"
#include "audio/mel.h"
#include "audio_test_util.h"
#include "util/error.h"

using namespace xs;

namespace {

constexpr double kTolerance = 1e-4;

audio::MelParams fixture_params(const nlohmann::json& fixture) {
    const auto& p = fixture.at("params");
    audio::MelParams params;
    params.sample_rate = p.at("sample_rate");
    params.n_fft = p.at("n_fft");
    params.hop_length = p.at("hop_length");
    params.n_mels = p.at("n_mels");
    params.min_length = p.at("min_length");
    return params;
}

}  // namespace

TEST_SUITE("audio mel") {
    TEST_CASE("audio mel: RealFft equals the DFT for lengths of every kind of factor") {
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (const int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 25, 30, 49, 64, 97, 100, 121, 202, 400, 510, 512}) {
            CAPTURE(n);
            std::vector<double> x(static_cast<size_t>(n));
            for (double& v : x) v = uniform(rng);
            std::vector<std::complex<double>> got(static_cast<size_t>(n / 2 + 1));
            audio::RealFft fft(n);
            fft.forward(x.data(), got.data());
            double worst = 0.0;
            for (int k = 0; k <= n / 2; ++k) {
                std::complex<long double> sum = 0;
                for (int j = 0; j < n; ++j) {
                    const long double angle = -2.0L * std::numbers::pi_v<long double> * ((int64_t{j} * k) % n) / n;
                    sum += static_cast<long double>(x[static_cast<size_t>(j)]) *
                           std::complex<long double>(std::cos(angle), std::sin(angle));
                }
                worst = std::max(worst, static_cast<double>(std::abs(std::complex<long double>(got[k]) - sum)));
            }
            CHECK(worst < 1e-12 * n);
        }
    }

    TEST_CASE("audio mel: log_mel_spectrogram matches Qwen3ASRFeatureExtractor") {
        const nlohmann::json fixture = test::read_audio_json("mel_cases.json");
        const audio::MelParams params = fixture_params(fixture);
        double worst = 0.0;
        for (const auto& c : fixture.at("cases")) {
            CAPTURE(c.at("name").get<std::string>());
            const HostTensor mel = audio::log_mel_spectrogram(test::b64_floats(c.at("input")), params);
            REQUIRE(mel.shape() == Dims{params.n_mels, c.at("frames").get<int64_t>()});
            const double diff = test::max_abs_diff(mel.span<float>(), test::b64_floats(c.at("features")));
            CHECK(diff <= kTolerance);
            worst = std::max(worst, diff);
        }
        MESSAGE("largest difference from transformers: ", worst);
    }

    TEST_CASE("audio mel: log_mel_spectrogram is the extractor's math without its float32 rounding") {
        // The extractor computes in float32: bins 80 dB below a tone's carry its FFT's rounding, up to ~7e-5 here.
        const nlohmann::json fixture = test::read_audio_json("mel_cases.json");
        const audio::MelParams params = fixture_params(fixture);
        double worst = 0.0;
        for (const auto& c : fixture.at("cases")) {
            if (!c.contains("features_f64")) continue;
            CAPTURE(c.at("name").get<std::string>());
            const HostTensor mel = audio::log_mel_spectrogram(test::b64_floats(c.at("input")), params);
            worst = std::max(worst, test::max_abs_diff(mel.span<float>(), test::b64_floats(c.at("features_f64"))));
        }
        CHECK(worst <= 1e-6);
        MESSAGE("largest difference from the same math in float64: ", worst);
    }

    TEST_CASE("audio mel: planted: zero padding in place of reflect, and the symmetric window, are told apart") {
        const nlohmann::json fixture = test::read_audio_json("mel_cases.json");
        const audio::MelParams params = fixture_params(fixture);
        REQUIRE(fixture.at("planted").size() == 2);
        for (const auto& planted : fixture.at("planted")) {
            CAPTURE(planted.at("what").get<std::string>());
            const auto& c = fixture.at("cases").at(planted.at("case").get<size_t>());
            const HostTensor mel = audio::log_mel_spectrogram(test::b64_floats(c.at("input")), params);
            CHECK(test::max_abs_diff(mel.span<float>(), test::b64_floats(planted.at("features"))) > 10 * kTolerance);
        }
    }

    TEST_CASE("audio mel: a frame per hop of the clip after the min_length padding") {
        audio::MelParams params;
        CHECK(audio::log_mel_spectrogram(std::vector<float>(100, 0.1f), params).dim(1) == 50);  // 8000 / 160
        CHECK(audio::log_mel_spectrogram(std::vector<float>(16159, 0.1f), params).dim(1) == 100);
        params.min_length = 0;
        CHECK(audio::log_mel_spectrogram(std::vector<float>(320, 0.1f), params).dim(1) == 2);
        params.n_fft = 401;  // an odd n_fft pads n_fft / 2 = 200 at each end and frames (length - 1) / hop
        CHECK(audio::log_mel_spectrogram(std::vector<float>(320, 0.1f), params).dim(1) == 1);
    }

    TEST_CASE("audio mel: silence is the floor everywhere: log10(1e-10) mapped to -1.5") {
        const HostTensor mel = audio::log_mel_spectrogram(std::vector<float>(1000, 0.0f));
        for (const float v : mel.span<float>()) CHECK(v == -1.5f);
    }

    TEST_CASE("audio mel: a clip the reflect padding cannot cover is refused") {
        audio::MelParams params;
        params.min_length = 0;
        CHECK_THROWS_WITH_AS(audio::log_mel_spectrogram(std::vector<float>(200, 0.1f), params),
                             doctest::Contains("too short for log-mel features"), Error);
        params.hop_length = 0;
        CHECK_THROWS_AS(audio::log_mel_spectrogram(std::vector<float>(8000, 0.1f), params), Error);
    }
}
