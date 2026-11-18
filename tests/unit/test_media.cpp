// Media placeholder expansion (engine/media.h) with fake image and audio encoders.
#include <doctest/doctest.h>

#include <array>

#include "engine/media.h"
#include "util/error.h"

namespace {

// Placeholder 9 becomes [5, 9 x n, 6], like Gemma's <boi> <image>... <eoi>.
class FakeVision final : public xs::VisionEncoder {
public:
    const xs::graph::Graph& graph() const override { return graph_; }
    xs::MediaInput prepare(const xs::image::Image&) const override { return {}; }
    int32_t placeholder_token() const override { return 9; }
    int32_t soft_token() const override { return 9; }
    std::vector<int32_t> token_run(int64_t n) const override {
        std::vector<int32_t> run{5};
        run.insert(run.end(), static_cast<size_t>(n), 9);
        run.push_back(6);
        return run;
    }

private:
    xs::graph::Graph graph_;
};

// Placeholder 8 becomes 8 x n, like Qwen3-ASR's <|audio_pad|> between the template's own start and end tokens.
class FakeAudio final : public xs::AudioEncoder {
public:
    const xs::graph::Graph& graph() const override { return graph_; }
    xs::MediaInput prepare(const xs::audio::Audio&) const override { return {}; }
    int32_t placeholder_token() const override { return 8; }
    int32_t soft_token() const override { return 8; }
    std::vector<int32_t> token_run(int64_t n) const override { return std::vector<int32_t>(static_cast<size_t>(n), 8); }

private:
    xs::graph::Graph graph_;
};

xs::MediaInput item(xs::Modality m, int64_t tokens) {
    xs::MediaInput i;
    i.modality = m;
    i.num_tokens = tokens;
    return i;
}

xs::MediaInput image_of(int64_t tokens) { return item(xs::Modality::Image, tokens); }
xs::MediaInput audio_of(int64_t tokens) { return item(xs::Modality::Audio, tokens); }

}  // namespace

TEST_CASE("each image placeholder becomes its own image's token run, in order") {
    const FakeVision vision;
    const std::array<const xs::MediaEncoder*, xs::kModalities> encoders{&vision, nullptr};
    const std::vector<int32_t> prompt{1, 9, 2, 9, 3};
    const std::vector<xs::MediaInput> media{image_of(2), image_of(3)};
    CHECK(xs::expand_media_placeholders(prompt, media, encoders) ==
          std::vector<int32_t>{1, 5, 9, 9, 6, 2, 5, 9, 9, 9, 6, 3});
    const std::vector<int32_t> text{1, 2, 3};
    CHECK(xs::expand_media_placeholders(text, {}, encoders) == text);
}

TEST_CASE("images and audio in one prompt: each placeholder takes the next item, which must be of its kind") {
    const FakeVision vision;
    const FakeAudio audio;
    const std::array<const xs::MediaEncoder*, xs::kModalities> encoders{&vision, &audio};
    const std::vector<int32_t> prompt{1, 8, 2, 9, 3};
    const std::vector<xs::MediaInput> media{audio_of(3), image_of(1)};
    CHECK(xs::expand_media_placeholders(prompt, media, encoders) == std::vector<int32_t>{1, 8, 8, 8, 2, 5, 9, 6, 3});
    // the items in another order than their placeholders
    const std::vector<xs::MediaInput> swapped{image_of(1), audio_of(3)};
    CHECK_THROWS_WITH_AS(xs::expand_media_placeholders(prompt, swapped, encoders), doctest::Contains("order"),
                         xs::Error);
    // audio for a model without an audio encoder: its placeholder is text, so the item has none
    const std::array<const xs::MediaEncoder*, xs::kModalities> vision_only{&vision, nullptr};
    CHECK_THROWS_WITH_AS(xs::expand_media_placeholders(prompt, media, vision_only),
                         doctest::Contains("0 audio placeholders for 1"), xs::Error);
}

TEST_CASE("placeholders and items that do not pair up are refused") {
    const FakeVision vision;
    const std::array<const xs::MediaEncoder*, xs::kModalities> encoders{&vision, nullptr};
    const std::vector<int32_t> one{1, 9, 2};
    const std::vector<xs::MediaInput> two{image_of(2), image_of(2)};
    CHECK_THROWS_WITH_AS(xs::expand_media_placeholders(one, two, encoders),
                         doctest::Contains("1 image placeholders for 2"), xs::Error);
    CHECK_THROWS_AS(xs::expand_media_placeholders(one, {}, encoders), xs::Error);
}
