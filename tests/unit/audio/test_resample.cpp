// torchaudio-exact resampling, against tests/fixtures/audio/resample_cases.json.
#include <doctest/doctest.h>

#include "audio/audio.h"
#include "audio/resample.h"
#include "audio_test_util.h"
#include "util/error.h"

using namespace xs;

namespace {

constexpr double kTolerance = 2e-6;

}  // namespace

TEST_SUITE("audio resample") {
    TEST_CASE("audio resample: torchaudio's sinc_interp_hann to float32 rounding") {
        const nlohmann::json fixture = test::read_audio_json("resample_cases.json");
        double worst = 0.0;
        for (const auto& c : fixture.at("cases")) {
            const std::vector<float> x = test::b64_floats(c.at("input"));
            const std::vector<float> expected = test::b64_floats(c.at("output"));
            const int from = c.at("from"), to = c.at("to");
            CAPTURE(from);
            CAPTURE(x.size());
            const std::vector<float> y = audio::resample(x, from, to);
            REQUIRE(y.size() == expected.size());
            const double diff = test::max_abs_diff(y, expected);
            CHECK(diff <= kTolerance);
            worst = std::max(worst, diff);
        }
        MESSAGE("largest difference from torchaudio: ", worst);
    }

    TEST_CASE("audio resample: planted: the filter at another rolloff is told apart") {
        const nlohmann::json fixture = test::read_audio_json("resample_cases.json");
        for (const auto& planted : fixture.at("planted")) {
            const auto& c = fixture.at("cases").at(planted.at("case").get<size_t>());
            const std::vector<float> y = audio::resample(test::b64_floats(c.at("input")), c.at("from"), c.at("to"));
            CAPTURE(planted.at("what").get<std::string>());
            CHECK(test::max_abs_diff(y, test::b64_floats(planted.at("output"))) > 100 * kTolerance);
        }
    }

    TEST_CASE("audio resample: the same rate returns the input; empty input gives empty output") {
        const std::vector<float> x = {0.5f, -0.25f, 1.5f};
        CHECK(audio::resample(x, 22050, 22050) == x);
        CHECK(audio::resample({}, 44100, 16000).empty());
        CHECK(audio::resample(x, 8000, 16000).size() == 6);
        CHECK(audio::resample(x, 48000, 16000).size() == 1);  // ceil(3 / 3)
    }

    TEST_CASE("audio resample: rates outside 1 Hz to kMaxSampleRate are refused") {
        const std::vector<float> x(10, 0.0f);
        CHECK_THROWS_WITH_AS(audio::resample(x, 0, 16000), doctest::Contains("resampling at 0 Hz"), Error);
        CHECK_THROWS_WITH_AS(audio::resample(x, 16000, -8000), doctest::Contains("-8000 Hz"), Error);
        CHECK_THROWS_WITH_AS(audio::resample(x, audio::kMaxSampleRate + 1, 16000),
                             doctest::Contains("is not supported"), Error);
    }
}
