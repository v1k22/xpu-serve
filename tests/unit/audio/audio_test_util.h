#pragma once

// Fixtures of tests/fixtures/audio (tests/parity/audio/gen_audio_cases.py) and comparisons of float runs.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "util/base64.h"

namespace xs::test {

inline std::filesystem::path audio_fixture(const std::string& name) {
    return std::filesystem::path(XS_SOURCE_DIR) / "tests/fixtures/audio" / name;
}

inline std::vector<uint8_t> read_audio_fixture(const std::string& name) {
    std::ifstream in(audio_fixture(name), std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "missing fixture ", name);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline nlohmann::json read_audio_json(const std::string& name) {
    std::ifstream in(audio_fixture(name));
    REQUIRE_MESSAGE(in.good(), "missing fixture ", name);
    return nlohmann::json::parse(in);
}

// Little-endian f32 values stored as base64.
inline std::vector<float> b64_floats(const nlohmann::json& text) {
    const auto bytes = base64_decode(text.get<std::string>());
    REQUIRE(bytes.has_value());
    REQUIRE(bytes->size() % 4 == 0);
    std::vector<float> out(bytes->size() / 4);
    std::memcpy(out.data(), bytes->data(), bytes->size());
    return out;
}

// The largest |a[i] - b[i]|; infinity when the lengths differ.
inline double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return INFINITY;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(double{a[i]} - double{b[i]}));
    return worst;
}

}  // namespace xs::test
