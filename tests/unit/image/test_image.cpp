// Image decoding, the torchvision-exact bicubic resize and content hashing.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "image/image.h"
#include "util/base64.h"
#include "util/error.h"

using namespace xs;

namespace {

std::vector<uint8_t> b64(std::string_view text) {
    auto bytes = base64_decode(text);
    REQUIRE(bytes.has_value());
    return *bytes;
}

image::Image from_rgb(int64_t width, int64_t height, std::vector<uint8_t> rgb) {
    image::Image img;
    img.width = width;
    img.height = height;
    img.rgb = std::move(rgb);
    return img;
}

// 3x2 RGB, 2x1 RGBA and 2x1 grayscale PNGs, and a uniform 16x8 JPEG, written by Pillow.
constexpr std::string_view kRgbPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAGElEQVR4nGP4z8DA8J+BgeE/I5eIHAQAADEzA++e06gjAAAAAElFTkSuQmCC";
constexpr std::string_view kRgbaPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAEUlEQVR4nGM4kWLEwMjE3AAACsEB5eZCaQEAAAAASUVORK5CYII=";
constexpr std::string_view kGrayPng =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAAAAADRSSBWAAAAC0lEQVR4nGNg/wUAAQsBAsdd+yMAAAAASUVORK5CYII=";
constexpr std::string_view kJpeg =
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsK"
    "DAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAIABAD"
    "ASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKB"
    "kaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
    "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF"
    "BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX"
    "2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDPooor+Kz8fP/Z";

}  // namespace

TEST_SUITE("image") {
    TEST_CASE("decode PNG: RGB as is, RGBA drops alpha, grayscale is replicated") {
        const image::Image rgb = image::decode(b64(kRgbPng));
        CHECK(rgb.width == 3);
        CHECK(rgb.height == 2);
        CHECK(rgb.rgb == std::vector<uint8_t>{255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90});
        CHECK(image::decode(b64(kRgbaPng)).rgb == std::vector<uint8_t>{200, 100, 50, 1, 2, 3});
        CHECK(image::decode(b64(kGrayPng)).rgb == std::vector<uint8_t>{7, 7, 7, 250, 250, 250});
    }

    TEST_CASE("decode JPEG within a level or two of libjpeg") {
        const image::Image img = image::decode(b64(kJpeg));
        REQUIRE(img.width == 16);
        REQUIRE(img.height == 8);
        for (int64_t i = 0; i < img.width * img.height; ++i) {  // Pillow decodes every pixel as (90, 159, 30)
            CHECK(std::abs(img.rgb[static_cast<size_t>(3 * i)] - 90) <= 2);
            CHECK(std::abs(img.rgb[static_cast<size_t>(3 * i + 1)] - 159) <= 2);
            CHECK(std::abs(img.rgb[static_cast<size_t>(3 * i + 2)] - 30) <= 2);
        }
    }

    TEST_CASE("decode refuses empty, truncated and unknown data by name") {
        CHECK_THROWS_WITH_AS(image::decode({}), doctest::Contains("empty"), xs::Error);
        const std::vector<uint8_t> png = b64(kRgbPng);
        CHECK_THROWS_WITH_AS(image::decode(std::span(png).first(30)), doctest::Contains("could not be decoded"),
                             xs::Error);
        const std::string webp = "RIFF\x1a\0\0\0WEBPVP8 ";
        CHECK_THROWS_WITH_AS(image::decode(std::span(reinterpret_cast<const uint8_t*>(webp.data()), webp.size())),
                             doctest::Contains("supported formats: PNG, JPEG"), xs::Error);
    }

    TEST_CASE("resize_bicubic is bit-exact with torchvision's uint8 antialiased bicubic") {
        std::ifstream in(std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/image/resize_cases.json");
        REQUIRE(in.good());
        const nlohmann::json fixture = nlohmann::json::parse(in);
        std::vector<image::Image> sources;
        for (const auto& s : fixture.at("sources")) {
            sources.push_back(from_rgb(s.at("width"), s.at("height"), b64(s.at("rgb").get<std::string>())));
        }
        for (const auto& c : fixture.at("cases")) {
            const image::Image& src = sources.at(c.at("source").get<size_t>());
            const int64_t width = c.at("width"), height = c.at("height");
            CAPTURE(src.width);
            CAPTURE(src.height);
            CAPTURE(width);
            CAPTURE(height);
            const image::Image out = image::resize_bicubic(src, width, height);
            CHECK(out.width == width);
            CHECK(out.height == height);
            const std::vector<uint8_t> expected = b64(c.at("rgb").get<std::string>());
            REQUIRE(out.rgb.size() == expected.size());
            size_t mismatches = 0;
            for (size_t i = 0; i < expected.size(); ++i)
                mismatches += out.rgb[i] != expected[i];
            CHECK(mismatches == 0);
        }
    }

    TEST_CASE("resize to the same size is a copy") {
        const image::Image img = image::decode(b64(kRgbPng));
        CHECK(image::resize_bicubic(img, 3, 2).rgb == img.rgb);
    }

    TEST_CASE("content_hash tells images apart by pixels and by shape") {
        const image::Image a = from_rgb(2, 1, {1, 2, 3, 4, 5, 6});
        image::Image b = a;
        CHECK(image::content_hash(a) == image::content_hash(b));
        b.rgb[5] = 7;
        CHECK(image::content_hash(a) != image::content_hash(b));
        CHECK(image::content_hash(a) != image::content_hash(from_rgb(1, 2, {1, 2, 3, 4, 5, 6})));
    }
}
