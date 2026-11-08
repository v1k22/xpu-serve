// Round-to-nearest quantization (quant/quant.h) and the safetensors writer: error bounds against the f32 input, the
// exact code layout, edge groups, and a write / read round trip.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

using namespace xs;

namespace {

std::vector<float> random_matrix(int64_t rows, int64_t cols, uint32_t seed, float range = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-range, range);
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (float& v : w)
        v = dist(rng);
    return w;
}

TensorView f32_view(const std::vector<float>& w, int64_t rows, int64_t cols) {
    return TensorView{DType::F32, {rows, cols}, w.data(), {}};
}

std::vector<float> dequantized(const quant::Quantized& q) {
    std::vector<float> out(static_cast<size_t>(q.rows * q.cols));
    quant::dequantize_rows(q.view(), 0, q.rows, out.data());
    return out;
}

// max |w - dequant(w)| / (that group's step), over all elements; at most 0.5 plus f16 rounding of scale and bias.
double worst_step_error(const std::vector<float>& w, const quant::Quantized& q) {
    const std::vector<float> d = dequantized(q);
    double worst = 0;
    for (int64_t r = 0; r < q.rows; ++r) {
        for (int64_t c = 0; c < q.cols; ++c) {
            const size_t i = static_cast<size_t>(r * q.cols + c);
            const float step = f16_to_f32(q.scales[static_cast<size_t>(r * (q.cols / q.group) + c / q.group)]);
            if (step > 0) worst = std::max(worst, std::abs(static_cast<double>(w[i]) - d[i]) / step);
        }
    }
    return worst;
}

}  // namespace

TEST_SUITE("quant") {
    TEST_CASE("q8 and q4 stay within half a step of the input for every group size") {
        for (const DType format : {DType::Q8, DType::Q4}) {
            for (const int64_t group : {32, 64, 128}) {
                CAPTURE(dtype_name(format));
                CAPTURE(group);
                const std::vector<float> w = random_matrix(7, 256, 11);
                const quant::Quantized q = quant::quantize(f32_view(w, 7, 256), format, group);
                CHECK(q.codes.size() == (format == DType::Q4 ? 7 * 128u : 7 * 256u));
                CHECK(q.scales.size() == static_cast<size_t>(7 * 256 / group));
                CHECK(q.biases.size() == (format == DType::Q4 ? q.scales.size() : 0u));
                CHECK(worst_step_error(w, q) <= 0.52);
            }
        }
    }

    TEST_CASE("q4 codes: low nibble is the even column; values on the grid come back exactly") {
        std::vector<float> w(64);
        for (int i = 0; i < 64; ++i)
            w[static_cast<size_t>(i)] = static_cast<float>(i % 16);  // min 0, max 15: step 1, bias 0
        const quant::Quantized q = quant::quantize(f32_view(w, 1, 64), DType::Q4, 32);
        CHECK(q.codes[0] == 0x10);  // columns 0 (low) and 1 (high)
        CHECK(q.codes[7] == 0xFE);  // columns 14 and 15
        CHECK(dequantized(q) == w);
    }

    TEST_CASE("q8 is symmetric: the largest magnitude maps to +-127 and zero to zero") {
        std::vector<float> w(32, 0.0f);
        w[3] = -2.0f;
        w[9] = 1.0f;
        const quant::Quantized q = quant::quantize(f32_view(w, 1, 32), DType::Q8, 32);
        const auto* codes = reinterpret_cast<const int8_t*>(q.codes.data());
        CHECK(codes[3] == -127);
        CHECK(codes[0] == 0);
        CHECK(std::abs(codes[9] - 64) <= 1);
    }

    TEST_CASE("a 3-D expert bank quantizes as its [E * N, K] matrix and keeps its shape") {
        std::mt19937 gen(9);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> w(3 * 8 * 64);
        for (float& v : w)
            v = dist(gen);
        for (const DType format : {DType::Q8, DType::Q4}) {
            CAPTURE(dtype_name(format));
            TensorView bank;
            bank.dtype = DType::F32;
            bank.shape = {3, 8, 64};
            bank.data = w.data();
            const quant::Quantized q3 = quant::quantize(bank, format, 32);
            const quant::Quantized q2 = quant::quantize(f32_view(w, 24, 64), format, 32);
            CHECK(q3.rows == 24);
            CHECK(q3.cols == 64);
            CHECK(q3.view().shape == Dims{3, 8, 64});
            CHECK(q3.codes == q2.codes);
            CHECK(q3.scales == q2.scales);
            CHECK(q3.biases == q2.biases);
            CHECK(quant::component_shape(bank.shape, 2) == Dims{3, 8, 2});
        }
    }

    TEST_CASE("int8 KV rows: within half a step, the largest magnitude is +-127, zero rows stay zero") {
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        const int64_t rows = 6, width = 64;
        std::vector<float> x(static_cast<size_t>(rows * width));
        for (float& v : x)
            v = dist(gen);
        std::fill(x.begin() + 2 * width, x.begin() + 3 * width, 0.0f);  // row 2: all zero
        x[4 * width + 5] = -50.0f;                                         // row 4: one outlier
        for (int64_t r = 0; r < rows; ++r) {
            CAPTURE(r);
            const float* row = x.data() + r * width;
            std::vector<int8_t> codes(static_cast<size_t>(width));
            const float scale = quant::quantize_kv_row_i8(row, width, codes.data());
            float amax = 0.0f;
            for (int64_t i = 0; i < width; ++i)
                amax = std::max(amax, std::abs(row[i]));
            CHECK(scale == amax / 127.0f);
            int largest = 0;
            for (int64_t i = 0; i < width; ++i) {
                largest = std::max(largest, std::abs(static_cast<int>(codes[static_cast<size_t>(i)])));
                CHECK(std::abs(codes[static_cast<size_t>(i)] * scale - row[i]) <= 0.5f * scale * (1 + 1e-5f) + 1e-12f);
            }
            CHECK(largest == (r == 2 ? 0 : 127));
        }
        std::vector<float> y = x;
        quant::round_trip_kv_i8(y.data(), rows, width);
        CHECK(std::all_of(y.begin() + 2 * width, y.begin() + 3 * width, [](float v) { return v == 0.0f; }));
        CHECK(y[4 * width + 5] == -50.0f);
        CHECK(y != x);
    }

    TEST_CASE("all-zero groups are exact; a constant q4 group is exactly its bias") {
        const std::vector<float> zeros(64, 0.0f);
        for (const DType format : {DType::Q8, DType::Q4}) {
            CAPTURE(dtype_name(format));
            CHECK(dequantized(quant::quantize(f32_view(zeros, 1, 64), format, 32)) == zeros);
        }
        const std::vector<float> constant(64, 0.25f);
        CHECK(dequantized(quant::quantize(f32_view(constant, 1, 64), DType::Q4, 32)) == constant);
    }

    TEST_CASE("bf16 input quantizes as its f32 value does") {
        const std::vector<float> w = random_matrix(3, 128, 5);
        std::vector<uint16_t> bf(w.size());
        std::vector<float> rounded(w.size());
        for (size_t i = 0; i < w.size(); ++i) {
            bf[i] = f32_to_bf16(w[i]);
            rounded[i] = bf16_to_f32(bf[i]);
        }
        const quant::Quantized a = quant::quantize(TensorView{DType::BF16, {3, 128}, bf.data(), {}}, DType::Q4, 64);
        const quant::Quantized b = quant::quantize(f32_view(rounded, 3, 128), DType::Q4, 64);
        CHECK(a.codes == b.codes);
        CHECK(a.scales == b.scales);
        CHECK(a.biases == b.biases);
    }

    TEST_CASE("unsupported shapes and settings are refused by name") {
        CHECK(quant::unsupported_reason(DType::Q4, 4, 96, 64).find("multiple of the group size") != std::string::npos);
        CHECK(quant::unsupported_reason(DType::Q4, 4, 128, 48).find("group size 48") != std::string::npos);
        CHECK(quant::unsupported_reason(DType::F16, 4, 128, 64).find("not q8 or q4") != std::string::npos);
        CHECK(quant::unsupported_reason(DType::Q8, 4, 128, 64).empty());
        const std::vector<float> w = random_matrix(2, 96, 1);
        CHECK_THROWS_AS(quant::quantize(f32_view(w, 2, 96), DType::Q4, 64), Error);
    }

    TEST_CASE("awq: never worse than round to nearest on its objective; much better with outlier channels") {
        const std::vector<float> w = random_matrix(16, 256, 21);
        std::vector<double> flat(256, 1.0);
        std::vector<double> outliers(256, 1.0);
        for (int c = 0; c < 256; c += 37)
            outliers[static_cast<size_t>(c)] = 400.0;  // a few channels carry most of the activation energy
        for (const auto* ms : {&flat, &outliers}) {
            quant::AwqReport r;
            const quant::Quantized q = quant::quantize_awq(f32_view(w, 16, 256), 64, *ms, &r);
            CHECK(r.error <= r.error_rtn);
            CHECK(q.input_scales.size() == 256u);
            // The reported error is what the stored weights give: sum ms[c] * (dequant(q) - w)^2.
            const std::vector<float> d = dequantized(q);
            double err = 0;
            for (size_t i = 0; i < d.size(); ++i) {
                const double diff = static_cast<double>(d[i]) - w[i];
                err += (*ms)[i % 256] * diff * diff;
            }
            CHECK(err == doctest::Approx(r.error).epsilon(1e-6));
        }
        quant::AwqReport r;
        quant::quantize_awq(f32_view(w, 16, 256), 64, outliers, &r);
        CHECK(r.alpha > 0);
        CHECK(r.error < 0.8 * r.error_rtn);
    }

    TEST_CASE("planted bug: reading the high nibble as the even column must not reproduce the weights") {
        const std::vector<float> w = random_matrix(2, 64, 9);
        quant::Quantized q = quant::quantize(f32_view(w, 2, 64), DType::Q4, 32);
        for (uint8_t& b : q.codes)
            b = static_cast<uint8_t>((b >> 4) | (b << 4));
        CHECK(worst_step_error(w, q) > 1.0);
    }
}

TEST_SUITE("safetensors writer") {
    TEST_CASE("units round-trip through SafetensorsFile, header padded to 8 bytes") {
        const auto path = std::filesystem::temp_directory_path() / "xs_writer_test.safetensors";
        const std::vector<float> a{1.0f, -2.5f, 3.25f, 0.0f, 7.0f, 8.0f};
        const std::vector<uint8_t> b{1, 2, 3};
        OutUnit u1;
        u1.tensors = {OutTensor{"a", DType::F32, {2, 3}}};
        u1.write = [&](std::ostream& out) { out.write(reinterpret_cast<const char*>(a.data()), 24); };
        OutUnit u2;
        u2.tensors = {OutTensor{"b.codes", DType::U8, {3}}, OutTensor{"b.extra", DType::U8, {0}}};
        u2.write = [&](std::ostream& out) { out.write(reinterpret_cast<const char*>(b.data()), 3); };
        write_safetensors(path, {u1, u2}, {{"format", "pt"}});
        {
            const SafetensorsFile f(path);
            REQUIRE(f.tensors().count("a") == 1);
            CHECK(f.tensors().at("a").shape == Dims{2, 3});
            CHECK(std::memcmp(f.tensors().at("a").data, a.data(), 24) == 0);
            CHECK(std::memcmp(f.tensors().at("b.codes").data, b.data(), 3) == 0);
            CHECK(f.metadata().at("format") == "pt");
        }
        std::ifstream in(path, std::ios::binary);
        uint64_t header_len = 0;
        in.read(reinterpret_cast<char*>(&header_len), 8);
        CHECK(header_len % 8 == 0);
        in.close();
        std::filesystem::remove(path);
    }

    TEST_CASE("a unit that writes the wrong number of bytes is caught") {
        const auto path = std::filesystem::temp_directory_path() / "xs_writer_bad.safetensors";
        OutUnit u;
        u.tensors = {OutTensor{"x", DType::F32, {4}}};
        u.write = [](std::ostream& out) { out.write("abc", 3); };
        CHECK_THROWS_AS(write_safetensors(path, {u}), Error);
        std::filesystem::remove(path);
    }
}
