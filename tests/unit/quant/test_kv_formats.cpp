// The KV cache formats on the host (quant/kv_format.h, quant/kv_turbo.h): names and sizes; the TurboQuant formats
// against the reference implementation's blocks (tests/fixtures/quant/turbo, from
// tests/parity/quant/gen_turbo_cases.py), their rotation, their error on Gaussian rows and on rows far from Gaussian,
// and planted bugs the checks must catch.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

#include "quant/kv_format.h"
#include "quant/kv_turbo.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

using namespace xs;
using quant::KvFormat;
namespace turbo = xs::quant::turbo;
namespace fs = std::filesystem;

namespace {

const fs::path kFixtures = fs::path(XS_SOURCE_DIR) / "tests/fixtures/quant/turbo";
constexpr int kGroup = turbo::kGroup;

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "missing fixture ", path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<float> fixture_rows() {
    const std::vector<uint8_t> bytes = read_bytes(kFixtures / "rows.f32");
    std::vector<float> rows(bytes.size() / sizeof(float));
    std::memcpy(rows.data(), bytes.data(), rows.size() * sizeof(float));  // little-endian, as the host
    return rows;
}

const float* mags(int bits) {
    return bits == 4 ? turbo::Codebook<4>::mags : bits == 3 ? turbo::Codebook<3>::mags : turbo::Codebook<2>::mags;
}
const float* edges(int bits) {
    return bits == 4 ? turbo::Codebook<4>::edges : bits == 3 ? turbo::Codebook<3>::edges : turbo::Codebook<2>::edges;
}
double centroid(int bits, uint32_t code) {
    const uint32_t half = 1u << (bits - 1);
    return code >= half ? mags(bits)[code - half] : -static_cast<double>(mags(bits)[half - 1 - code]);
}

// A group's rotated coordinates on the unit sphere, in double: what the codes round.
std::vector<double> rotated_unit(const float* x) {
    std::vector<double> w(x, x + kGroup);
    double norm = 0.0;
    for (const double v : w)
        norm += v * v;
    norm = std::sqrt(norm);
    for (int i = 0; i < kGroup; ++i)
        w[static_cast<size_t>(i)] *= ((turbo::kSignsBefore[i / 64] >> (i % 64)) & 1u) != 0 ? -1.0 : 1.0;
    for (int h = 1; h < kGroup; h *= 2) {
        for (int i = 0; i < kGroup; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const double a = w[static_cast<size_t>(j)], b = w[static_cast<size_t>(j + h)];
                w[static_cast<size_t>(j)] = a + b;
                w[static_cast<size_t>(j + h)] = a - b;
            }
        }
    }
    for (int i = 0; i < kGroup; ++i) {
        const double sign = ((turbo::kSignsAfter[i / 64] >> (i % 64)) & 1u) != 0 ? -1.0 : 1.0;
        w[static_cast<size_t>(i)] *= sign / (std::sqrt(static_cast<double>(kGroup)) * (norm > 0.0 ? norm : 1.0));
    }
    return w;
}

// Mean squared error of x's round trip over its mean square.
double relative_error(std::vector<float> x, int64_t rows, int64_t width, int bits) {
    const std::vector<float> exact = x;
    turbo::round_trip(x.data(), rows, width, bits);
    double err = 0.0, total = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        err += (static_cast<double>(x[i]) - exact[i]) * (static_cast<double>(x[i]) - exact[i]);
        total += static_cast<double>(exact[i]) * exact[i];
    }
    return err / total;
}

std::vector<float> gaussian_rows(int64_t rows, int64_t width, uint32_t seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> x(static_cast<size_t>(rows * width));
    for (float& v : x)
        v = dist(gen);
    return x;
}

}  // namespace

TEST_SUITE("kv formats") {
    TEST_CASE("names, the keys:values form, and what a row takes: 66, 50 and 34 bytes per 128 values") {
        CHECK(quant::parse_kv_formats("auto") == quant::KvFormats{});
        CHECK(quant::parse_kv_formats("i8") == quant::KvFormats{KvFormat::I8, KvFormat::I8});
        CHECK(quant::parse_kv_formats("turbo3") == quant::KvFormats{KvFormat::Turbo3, KvFormat::Turbo3});
        CHECK(quant::parse_kv_formats("i8:turbo4") == quant::KvFormats{KvFormat::I8, KvFormat::Turbo4});
        CHECK(quant::kv_formats_name({KvFormat::Turbo2, KvFormat::Turbo2}) == "turbo2");
        CHECK(quant::kv_formats_name({KvFormat::I8, KvFormat::Turbo2}) == "i8:turbo2");
        CHECK_THROWS_WITH_AS(quant::parse_kv_formats("turbo5"), doctest::Contains("'turbo5'"), Error);
        CHECK_THROWS_WITH_AS(quant::parse_kv_formats("i8:q4"), doctest::Contains("'i8:q4'"), Error);

        const auto row = [](KvFormat f, int64_t d) {
            const quant::KvRowBytes b = quant::kv_row_bytes(f, d);
            return b.codes + b.scales;
        };
        CHECK(row(KvFormat::I8, 256) == 256 + 4);
        CHECK(row(KvFormat::Turbo4, 128) == 66);
        CHECK(row(KvFormat::Turbo3, 128) == 50);
        CHECK(row(KvFormat::Turbo2, 128) == 34);
        CHECK(row(KvFormat::Turbo4, 256) == 132);
        CHECK(quant::kv_format_fits(KvFormat::I8, 64));
        CHECK_FALSE(quant::kv_format_fits(KvFormat::Turbo4, 64));
        CHECK_THROWS_WITH_AS(quant::kv_row_bytes(KvFormat::Turbo3, 192), doctest::Contains("multiple of 128"), Error);
        std::vector<float> x(192, 1.0f);
        CHECK_THROWS_WITH_AS(quant::round_trip_kv(KvFormat::Turbo2, x.data(), 1, 192), doctest::Contains("128"), Error);
    }

    TEST_CASE("round_trip_kv: float rows unchanged, int8 as round_trip_kv_i8") {
        const std::vector<float> x = gaussian_rows(3, 256, 5);
        std::vector<float> y = x;
        quant::round_trip_kv(KvFormat::Float, y.data(), 3, 256);
        CHECK(y == x);
        std::vector<float> want = x;
        quant::round_trip_kv_i8(want.data(), 3, 256);
        quant::round_trip_kv(KvFormat::I8, y.data(), 3, 256);
        CHECK(y == want);
    }

    TEST_CASE("turbo formats: the reference implementation's codes and norm of each fixture row") {
        const std::vector<float> rows = fixture_rows();
        const int64_t count = static_cast<int64_t>(rows.size()) / kGroup;
        REQUIRE(count >= 40);
        for (const int bits : {4, 3, 2}) {
            CAPTURE(bits);
            const size_t block = sizeof(uint16_t) + static_cast<size_t>(turbo::group_bytes(bits));
            const std::vector<uint8_t> blocks = read_bytes(kFixtures / ("turbo" + std::to_string(bits) + ".blocks"));
            REQUIRE(blocks.size() == static_cast<size_t>(count) * block);
            int ties = 0;
            for (int64_t r = 0; r < count; ++r) {
                CAPTURE(r);
                const float* x = rows.data() + r * kGroup;
                const uint8_t* ref = blocks.data() + static_cast<size_t>(r) * block;
                uint16_t ref_norm;
                std::memcpy(&ref_norm, ref, sizeof(ref_norm));
                std::vector<uint8_t> codes(static_cast<size_t>(turbo::group_bytes(bits)));
                const uint16_t norm = turbo::quantize_group(x, bits, codes.data());
                CHECK(std::abs(static_cast<int>(norm) - static_cast<int>(ref_norm)) <= 1);  // f16 bits: one step
                if (ref_norm == 0) continue;  // a row of zeros decodes to zeros whatever its codes
                const std::vector<double> w = rotated_unit(x);
                for (int i = 0; i < kGroup; ++i) {
                    const uint32_t got = turbo::code_at(codes.data(), bits, i);
                    const uint32_t want = turbo::code_at(ref + sizeof(uint16_t), bits, i);
                    if (got == want) continue;
                    // The two round a coordinate on an edge in different arithmetic: neighbours, and only there.
                    CAPTURE(i);
                    ++ties;
                    CHECK(std::abs(static_cast<int>(got) - static_cast<int>(want)) == 1);
                    const double a = std::abs(w[static_cast<size_t>(i)]);
                    double nearest = a;  // the edge at zero
                    for (int j = 0; j < (1 << (bits - 1)) - 1; ++j)
                        nearest = std::min(nearest, std::abs(a - edges(bits)[j]));
                    CHECK(nearest < 2e-7);
                }
            }
            MESSAGE("turbo", bits, ": ", ties, " of ", count * kGroup, " codes rounded the other way on an edge");
            CHECK(ties <= 2);
        }
    }

    TEST_CASE("turbo formats: each code is the nearest centroid of the rotated row, the norm restores the row's") {
        const std::vector<float> x = gaussian_rows(64, kGroup, 11, 3.0f);
        for (const int bits : {4, 3, 2}) {
            CAPTURE(bits);
            for (int64_t r = 0; r < 64; ++r) {
                CAPTURE(r);
                const float* row = x.data() + r * kGroup;
                std::vector<uint8_t> codes(static_cast<size_t>(turbo::group_bytes(bits)));
                const uint16_t norm = turbo::quantize_group(row, bits, codes.data());
                const std::vector<double> w = rotated_unit(row);
                double row_sq = 0.0, stored_sq = 0.0;
                std::vector<float> stored(kGroup);
                turbo::decode_group(codes.data(), norm, bits, stored.data());
                for (int i = 0; i < kGroup; ++i) {
                    const uint32_t code = turbo::code_at(codes.data(), bits, i);
                    double best = 1e30;
                    for (uint32_t c = 0; c < (1u << bits); ++c)
                        best = std::min(best, std::abs(w[static_cast<size_t>(i)] - centroid(bits, c)));
                    CHECK(std::abs(w[static_cast<size_t>(i)] - centroid(bits, code)) <= best + 1e-6);
                    CHECK(stored[static_cast<size_t>(i)] ==
                          static_cast<float>(centroid(bits, code)) * f16_to_f32(norm));
                    row_sq += static_cast<double>(row[i]) * row[i];
                    stored_sq += static_cast<double>(stored[static_cast<size_t>(i)]) * stored[static_cast<size_t>(i)];
                }
                CHECK(std::sqrt(stored_sq) == doctest::Approx(std::sqrt(row_sq)).epsilon(1e-3));  // f16 norm
            }
        }
    }

    TEST_CASE("turbo formats: the rotation is orthogonal and its inverse undoes it") {
        const std::vector<float> x = gaussian_rows(2, kGroup, 3);
        std::vector<float> a(x.begin(), x.begin() + kGroup), b(x.begin() + kGroup, x.end());
        double dot = 0.0;
        for (int i = 0; i < kGroup; ++i)
            dot += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
        turbo::rotate_group(a.data(), false);
        turbo::rotate_group(b.data(), false);
        double rotated_dot = 0.0;
        for (int i = 0; i < kGroup; ++i)
            rotated_dot += static_cast<double>(a[static_cast<size_t>(i)]) * b[static_cast<size_t>(i)];
        CHECK(rotated_dot == doctest::Approx(dot).epsilon(1e-5));
        CHECK(std::abs(a[0] - x[0]) > 1e-3);  // it moved the row
        turbo::rotate_group(a.data(), true);
        for (int i = 0; i < kGroup; ++i)
            CHECK(a[static_cast<size_t>(i)] == doctest::Approx(x[static_cast<size_t>(i)]).epsilon(1e-5).scale(1.0));
    }

    TEST_CASE("turbo formats: error near the Gaussian quantizers' on Gaussian rows, and no worse on spiky ones") {
        // Lloyd-Max quantizers of a unit Gaussian leave 0.0095, 0.0345 and 0.1175 of its variance at 4, 3 and 2 bits.
        const struct {
            int bits;
            double low, high;
        } cases[] = {{4, 0.008, 0.012}, {3, 0.030, 0.040}, {2, 0.105, 0.130}};
        const std::vector<float> gaussian = gaussian_rows(200, 256, 21, 0.7f);
        std::vector<float> spiky = gaussian_rows(200, 256, 22, 0.02f);
        std::mt19937 gen(23);
        for (int64_t r = 0; r < 200; ++r) {
            for (int s = 0; s < 3; ++s)
                spiky[static_cast<size_t>(r * 256 + static_cast<int64_t>(gen() % 256))] = gen() % 2 != 0 ? 25.0f : -25.0f;
        }
        std::vector<float> zeros(256, 0.0f);
        for (const auto& c : cases) {
            CAPTURE(c.bits);
            const double e = relative_error(gaussian, 200, 256, c.bits);
            CHECK(e > c.low);
            CHECK(e < c.high);
            CHECK(relative_error(spiky, 200, 256, c.bits) < c.high);
            turbo::round_trip(zeros.data(), 1, 256, c.bits);
            CHECK(std::all_of(zeros.begin(), zeros.end(), [](float v) { return v == 0.0f; }));
        }
    }

    TEST_CASE("planted bugs: codes read at another width, or left rotated, are far from the row") {
        const std::vector<float> x = gaussian_rows(1, kGroup, 31);
        std::vector<uint8_t> codes(static_cast<size_t>(turbo::group_bytes(4)));
        const uint16_t norm = turbo::quantize_group(x.data(), 4, codes.data());
        const auto error_of = [&](int bits, bool rotate_back) {
            std::vector<float> y(kGroup);
            turbo::decode_group(codes.data(), norm, bits, y.data());
            if (rotate_back) turbo::rotate_group(y.data(), true);
            double err = 0.0, total = 0.0;
            for (int i = 0; i < kGroup; ++i) {
                err += std::pow(static_cast<double>(y[static_cast<size_t>(i)]) - x[static_cast<size_t>(i)], 2);
                total += static_cast<double>(x[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)];
            }
            return err / total;
        };
        CHECK(error_of(4, true) < 0.02);
        CHECK(error_of(3, true) > 0.3);
        CHECK(error_of(4, false) > 0.3);
    }
}
