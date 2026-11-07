// Image placeholder expansion (engine/media.h) with a fake vision encoder.
#include <doctest/doctest.h>

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

xs::MediaInput image_of(int64_t tokens) {
    xs::MediaInput m;
    m.num_tokens = tokens;
    return m;
}

}  // namespace

TEST_CASE("each image placeholder becomes its own image's token run, in order") {
    const FakeVision vision;
    const std::vector<int32_t> prompt{1, 9, 2, 9, 3};
    const std::vector<xs::MediaInput> media{image_of(2), image_of(3)};
    CHECK(xs::expand_media_placeholders(prompt, media, vision) ==
          std::vector<int32_t>{1, 5, 9, 9, 6, 2, 5, 9, 9, 9, 6, 3});
    const std::vector<int32_t> text{1, 2, 3};
    CHECK(xs::expand_media_placeholders(text, {}, vision) == text);
}

TEST_CASE("placeholders and images that do not pair up are refused") {
    const FakeVision vision;
    const std::vector<int32_t> one{1, 9, 2};
    const std::vector<xs::MediaInput> two{image_of(2), image_of(2)};
    CHECK_THROWS_WITH_AS(xs::expand_media_placeholders(one, two, vision), doctest::Contains("1 image placeholders for 2"),
                         xs::Error);
    CHECK_THROWS_AS(xs::expand_media_placeholders(one, {}, vision), xs::Error);
}
