#pragma once

// Helpers shared by the model parity tests (comparisons against Hugging Face reference fixtures).

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/backend.h"
#include "test_support.h"

namespace xs::test {

inline std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::string(v) : fallback;
}

inline std::vector<char> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "cannot read ", path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

inline std::vector<float> read_f32(const std::filesystem::path& path) {
    const std::vector<char> bytes = read_bytes(path);
    std::vector<float> v(bytes.size() / sizeof(float));
    std::copy(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(v.size() * sizeof(float)),
              reinterpret_cast<char*>(v.data()));
    return v;
}

// Where the references of the model in XS_TEST_MODEL_DIR are: XS_PARITY_FIXTURES, or the generated directory of its
// model_type.
inline std::filesystem::path model_fixtures() {
    std::string model_type = "unknown";
    if (const auto dir = model_dir()) {
        std::ifstream in(*dir / "config.json");
        if (in.good()) model_type = nlohmann::json::parse(in).value("model_type", model_type);
    }
    return env_or("XS_PARITY_FIXTURES", std::string(XS_SOURCE_DIR) + "/tests/fixtures/generated/" + model_type);
}

// The session a parity test compiles: XS_PARITY_DEVICE, XS_PARITY_PRECISION (f32 by default), XS_PARITY_KV_DTYPE.
inline SessionOptions parity_options() {
    SessionOptions options;
    options.device = env_or("XS_PARITY_DEVICE", "");
    options.precision = env_or("XS_PARITY_PRECISION", "f32") == "f32" ? DType::F32 : DType::F16;
    options.max_context = 4096;
    options.kv = quant::parse_kv_formats(env_or("XS_PARITY_KV_DTYPE", "auto"));
    return options;
}

struct Compare {
    double cosine = 0;
    double max_abs = 0;
    int argmax_ours = -1;
    int argmax_ref = -1;
};

inline Compare compare(std::span<const float> ours, std::span<const float> ref) {
    Compare c;
    REQUIRE(ours.size() == ref.size());
    double dot = 0, a2 = 0, b2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        dot += static_cast<double>(ours[i]) * ref[i];
        a2 += static_cast<double>(ours[i]) * ours[i];
        b2 += static_cast<double>(ref[i]) * ref[i];
        c.max_abs = std::max(c.max_abs, static_cast<double>(std::fabs(ours[i] - ref[i])));
    }
    c.cosine = dot / std::sqrt(a2 * b2);
    c.argmax_ours = static_cast<int>(std::max_element(ours.begin(), ours.end()) - ours.begin());
    c.argmax_ref = static_cast<int>(std::max_element(ref.begin(), ref.end()) - ref.begin());
    return c;
}

}  // namespace xs::test
