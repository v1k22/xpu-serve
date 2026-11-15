// FP8 checkpoints (quant/fp8.h): the E4M3 code table, the weight store's F8 views (codes paired with their scales, one
// per block or one per row; tensors joined row-wise), reading and quantizing them, and the refusals.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "quant/fp8.h"
#include "quant/quant.h"
#include "util/error.h"
#include "util/half.h"

using namespace xs;
namespace fs = std::filesystem;

namespace {

struct Dir {
    fs::path path;
    explicit Dir(const std::string& name) : path(fs::temp_directory_path() / name) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~Dir() { fs::remove_all(path); }
};

// One tensor of a test file: its dtype, shape and bytes.
struct Raw {
    std::string name;
    DType dtype;
    Dims shape;
    std::vector<uint8_t> bytes;
};

void write_file(const fs::path& path, const std::vector<Raw>& tensors) {
    std::vector<OutUnit> units;
    for (const Raw& t : tensors) {
        OutUnit u;
        u.tensors = {OutTensor{t.name, t.dtype, t.shape}};
        u.write = [&t](std::ostream& out) {
            out.write(reinterpret_cast<const char*>(t.bytes.data()), static_cast<std::streamsize>(t.bytes.size()));
        };
        units.push_back(std::move(u));
    }
    write_safetensors(path, units);
}

// Finite codes drawn at random (0x7f and 0xff are NaN).
Raw random_codes(const std::string& name, int64_t rows, int64_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    Raw t{name, DType::F8, {rows, cols}, std::vector<uint8_t>(static_cast<size_t>(rows * cols))};
    for (uint8_t& c : t.bytes) {
        c = static_cast<uint8_t>(rng() & 0xFF);
        if ((c & 0x7F) == 0x7F) c ^= 0x01;
    }
    return t;
}

// A scale grid for a [rows, cols] matrix, every block its own value: 0.5, 0.75, 1, ... (exact in bf16).
Raw scale_grid(const std::string& name, int64_t rows, int64_t cols, DType dtype = DType::BF16, float first = 0.5f) {
    const Dims shape{(rows + kFp8Block - 1) / kFp8Block, (cols + kFp8Block - 1) / kFp8Block};
    Raw t{name, dtype, shape, {}};
    for (int64_t i = 0; i < numel(shape); ++i) {
        const float v = first + 0.25f * static_cast<float>(i);
        if (dtype == DType::F32) {
            const auto* p = reinterpret_cast<const uint8_t*>(&v);
            t.bytes.insert(t.bytes.end(), p, p + 4);
        } else {
            const uint16_t h = f32_to_bf16(v);
            const auto* p = reinterpret_cast<const uint8_t*>(&h);
            t.bytes.insert(t.bytes.end(), p, p + 2);
        }
    }
    return t;
}

// A scale per row of a [rows, cols] matrix, as compressed-tensors stores them ([rows, 1]): 0.5, 0.75, 1, ...
Raw row_scales(const std::string& name, int64_t rows, Dims shape = {}) {
    Raw t = scale_grid(name, rows * kFp8Block, 1);  // one value per row
    t.shape = shape.empty() ? Dims{rows, 1} : shape;
    return t;
}

// The magnitude and sign of a code, from its fields through ldexp.
float code_value(uint8_t code) {
    const int e = (code >> 3) & 0xF, m = code & 7;
    const float magnitude =
        e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(static_cast<float>(8 + m), e - 10);
    return (code & 0x80) != 0 ? -magnitude : magnitude;
}

// The values the pair stands for, written without the code under test: the code's fields through ldexp, the block's
// scale by its index.
std::vector<float> expected_values(const Raw& codes, float first_scale = 0.5f) {
    const int64_t rows = codes.shape[0], cols = codes.shape[1];
    const int64_t grid_cols = (cols + 127) / 128;
    std::vector<float> out(static_cast<size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const uint8_t code = codes.bytes[static_cast<size_t>(r * cols + c)];
            const int e = (code >> 3) & 0xF, m = code & 7;
            const float magnitude = e == 0 ? std::ldexp(static_cast<float>(m), -9)
                                           : std::ldexp(static_cast<float>(8 + m), e - 10);
            const float scale = first_scale + 0.25f * static_cast<float>((r / 128) * grid_cols + c / 128);
            out[static_cast<size_t>(r * cols + c)] = ((code & 0x80) != 0 ? -magnitude : magnitude) * scale;
        }
    }
    return out;
}

std::vector<float> rows_of(const TensorView& v, int64_t first, int64_t count) {
    std::vector<float> out(static_cast<size_t>(count * v.shape.back()));
    quant::read_rows(v, first, count, out.data());
    return out;
}

std::vector<float> slice_rows(const std::vector<float>& values, int64_t cols, int64_t first, int64_t count) {
    return {values.begin() + first * cols, values.begin() + (first + count) * cols};
}

}  // namespace

TEST_SUITE("fp8") {
    TEST_CASE("E4M3 codes: anchors, every code against its fields, the two NaNs") {
        CHECK(quant::e4m3_to_f32(0x00) == 0.0f);
        CHECK(quant::e4m3_to_f32(0x01) == 0.001953125f);  // the smallest subnormal, 2^-9
        CHECK(quant::e4m3_to_f32(0x08) == 0.015625f);     // the smallest normal, 2^-6
        CHECK(quant::e4m3_to_f32(0x30) == 0.5f);
        CHECK(quant::e4m3_to_f32(0x38) == 1.0f);
        CHECK(quant::e4m3_to_f32(0x3C) == 1.5f);
        CHECK(quant::e4m3_to_f32(0x40) == 2.0f);
        CHECK(quant::e4m3_to_f32(0x7E) == 448.0f);  // the largest: no infinities
        CHECK(quant::e4m3_to_f32(0xB8) == -1.0f);
        CHECK(quant::e4m3_to_f32(0xFE) == -448.0f);
        CHECK(std::isnan(quant::e4m3_to_f32(0x7F)));
        CHECK(std::isnan(quant::e4m3_to_f32(0xFF)));
        CHECK(std::signbit(quant::e4m3_to_f32(0x80)));
        const auto value = [](int code) { return quant::e4m3_to_f32(static_cast<uint8_t>(code)); };
        for (int code = 1; code < 0x7F; ++code) {  // strictly increasing, and mirrored by the sign bit
            CAPTURE(code);
            CHECK(value(code) > value(code - 1));
            CHECK(value(code | 0x80) == -value(code));
        }
    }

    TEST_CASE("the weight store shows codes and their block scales as one f8 weight; rows read whole and in part") {
        for (const DType scale_dtype : {DType::BF16, DType::F32}) {
            CAPTURE(dtype_name(scale_dtype));
            const Dir dir("xs_fp8_store_test");
            // 200 x 300: two block rows and three block columns, the last of each partial.
            const Raw codes = random_codes("layer.q_proj.weight", 200, 300, 7);
            write_file(dir.path / "model.safetensors",
                       {codes, scale_grid("layer.q_proj.weight_scale_inv", 200, 300, scale_dtype),
                        Raw{"layer.norm.weight", DType::F32, {1}, {0, 0, 128, 63}}});
            const auto store = SafetensorsWeights::open_dir(dir.path);
            CHECK(store->names() == std::vector<std::string>{"layer.norm.weight", "layer.q_proj.weight"});
            const TensorView& w = store->get("layer.q_proj.weight");
            CHECK(w.dtype == DType::F8);
            CHECK(w.shape == Dims{200, 300});
            CHECK(w.nbytes() == 200 * 300);
            const std::vector<float> expected = expected_values(codes);
            CHECK(rows_of(w, 0, 200) == expected);
            CHECK(rows_of(w, 120, 20) == slice_rows(expected, 300, 120, 20));  // across the block boundary at 128
            CHECK(rows_of(w, 199, 1) == slice_rows(expected, 300, 199, 1));
            // Planted bug: the grid read with rows and columns swapped must not reproduce the values.
            std::vector<float> swapped(expected.size());
            for (int64_t r = 0; r < 200; ++r) {
                for (int64_t c = 0; c < 300; ++c) {
                    const float right = 0.5f + 0.25f * static_cast<float>((r / 128) * 3 + c / 128);
                    const float wrong = 0.5f + 0.25f * static_cast<float>((c / 128) * 2 + r / 128);
                    const auto i = static_cast<size_t>(r * 300 + c);
                    swapped[i] = expected[i] / right * wrong;
                }
            }
            CHECK(rows_of(w, 0, 200) != swapped);
        }
    }

    TEST_CASE("codes beside a scale per row are one f8 weight too; rows read whole and in part") {
        for (const bool flat : {false, true}) {  // the scales as [rows, 1], or as [rows]
            CAPTURE(flat);
            const Dir dir("xs_fp8_rows_test");
            const Raw codes = random_codes("layer.q_proj.weight", 200, 300, 9);
            write_file(dir.path / "model.safetensors",
                       {codes, row_scales("layer.q_proj.weight_scale", 200, flat ? Dims{200} : Dims{})});
            const auto store = SafetensorsWeights::open_dir(dir.path);
            CHECK(store->names() == std::vector<std::string>{"layer.q_proj.weight"});
            const TensorView& w = store->get("layer.q_proj.weight");
            CHECK(w.dtype == DType::F8);
            CHECK(w.shape == Dims{200, 300});
            std::vector<float> expected(200 * 300), by_block(200 * 300);
            for (int64_t r = 0; r < 200; ++r) {
                for (int64_t c = 0; c < 300; ++c) {
                    const auto i = static_cast<size_t>(r * 300 + c);
                    expected[i] = code_value(codes.bytes[i]) * (0.5f + 0.25f * static_cast<float>(r));
                    // Planted bug: the scales read as a grid of blocks.
                    const auto block = static_cast<float>((r / 128) * 3 + c / 128);
                    by_block[i] = code_value(codes.bytes[i]) * (0.5f + 0.25f * block);
                }
            }
            CHECK(rows_of(w, 0, 200) == expected);
            CHECK(rows_of(w, 120, 20) == slice_rows(expected, 300, 120, 20));
            CHECK(rows_of(w, 199, 1) == slice_rows(expected, 300, 199, 1));
            CHECK(rows_of(w, 0, 200) != by_block);
        }
    }

    TEST_CASE("join_rows: f8 matrices read as one weight, run after run, without their own names") {
        const Dir dir("xs_fp8_join_test");
        const Raw a = random_codes("experts.0.gate_proj.weight", 64, 320, 1);
        const Raw b = random_codes("experts.0.up_proj.weight", 200, 320, 2);
        const Raw c = random_codes("experts.1.gate_proj.weight", 8, 320, 3);
        write_file(dir.path / "model.safetensors",
                   {a, scale_grid(a.name + "_scale_inv", 64, 320, DType::BF16, 0.5f), b,
                    scale_grid(b.name + "_scale_inv", 200, 320, DType::BF16, 2.0f), c,
                    scale_grid(c.name + "_scale_inv", 8, 320, DType::BF16, 8.0f)});
        const auto store = SafetensorsWeights::open_dir(dir.path);
        store->join_rows("experts.bank", {a.name, b.name, c.name}, {2, 136, 320});
        CHECK(store->names() == std::vector<std::string>{"experts.bank"});
        const TensorView& bank = store->get("experts.bank");
        CHECK(bank.dtype == DType::F8);
        CHECK(bank.shape == Dims{2, 136, 320});
        std::vector<float> expected = expected_values(a, 0.5f);
        for (const auto& more : {expected_values(b, 2.0f), expected_values(c, 8.0f)})
            expected.insert(expected.end(), more.begin(), more.end());
        CHECK(rows_of(bank, 0, 272) == expected);
        CHECK(rows_of(bank, 60, 10) == slice_rows(expected, 320, 60, 10));    // a run's end, the next one's start
        CHECK(rows_of(bank, 190, 82) == slice_rows(expected, 320, 190, 82));  // a block boundary inside the second run
        CHECK(rows_of(bank, 271, 1) == slice_rows(expected, 320, 271, 1));

        // Quantizing it reads the same values.
        const TensorView dense{DType::F32, {2, 136, 320}, expected.data()};
        const quant::Quantized from_fp8 = quant::quantize(bank, DType::Q4, 32);
        const quant::Quantized from_f32 = quant::quantize(dense, DType::Q4, 32);
        CHECK(from_fp8.codes == from_f32.codes);
        CHECK(from_fp8.scales == from_f32.scales);
        CHECK(from_fp8.biases == from_f32.biases);
        CHECK_THROWS_AS(rows_of(bank, 270, 3), Error);
    }

    TEST_CASE("join_rows: runs of either scale layout in one weight; float matrices copied into one") {
        const Dir dir("xs_fp8_join_dense_test");
        const Raw a = random_codes("experts.0.gate_proj.weight", 6, 320, 1);  // a scale per block
        const Raw b = random_codes("experts.0.up_proj.weight", 5, 320, 2);    // a scale per row
        std::vector<Raw> tensors = {a, scale_grid(a.name + "_scale_inv", 6, 320, DType::BF16, 2.0f), b,
                                    row_scales(b.name + "_scale", 5)};
        std::vector<float> dense;  // two float matrices [3, 8] and [2, 8]
        for (int i = 0; i < 40; ++i)
            dense.push_back(0.25f * static_cast<float>(i) - 3.0f);
        const auto f32 = [&](const std::string& name, int64_t first, int64_t rows) {
            const auto* p = reinterpret_cast<const uint8_t*>(dense.data() + first * 8);
            return Raw{name, DType::F32, {rows, 8}, {p, p + rows * 8 * 4}};
        };
        tensors.push_back(f32("mtp.experts.0.down_proj.weight", 0, 3));
        tensors.push_back(f32("mtp.experts.1.down_proj.weight", 3, 2));
        write_file(dir.path / "model.safetensors", tensors);
        const auto store = SafetensorsWeights::open_dir(dir.path);

        store->join_rows("experts.bank", {a.name, b.name}, {11, 320});
        std::vector<float> expected = expected_values(a, 2.0f);
        for (int64_t r = 0; r < 5; ++r)
            for (int64_t c = 0; c < 320; ++c)
                expected.push_back(code_value(b.bytes[static_cast<size_t>(r * 320 + c)]) *
                                   (0.5f + 0.25f * static_cast<float>(r)));
        CHECK(rows_of(store->get("experts.bank"), 0, 11) == expected);
        CHECK(rows_of(store->get("experts.bank"), 4, 5) == slice_rows(expected, 320, 4, 5));

        store->join_rows("mtp.experts.down_proj", {"mtp.experts.0.down_proj.weight", "mtp.experts.1.down_proj.weight"},
                         {5, 8});
        CHECK(store->names() == std::vector<std::string>{"experts.bank", "mtp.experts.down_proj"});
        const TensorView& bank = store->get("mtp.experts.down_proj");
        CHECK(bank.dtype == DType::F32);
        CHECK(bank.shape == Dims{5, 8});
        REQUIRE(bank.data != nullptr);
        CHECK(rows_of(bank, 0, 5) == dense);
        CHECK(rows_of(bank, 2, 2) == slice_rows(dense, 8, 2, 2));  // across the two parts
    }

    TEST_CASE("refusals by name: codes without scales, a wrong grid, other ranks, dense or mismatched parts") {
        // Whether running `fn` is refused with a message that names `part`.
        const auto refuses = [](const auto& fn, const std::string& part) {
            try {
                fn();
            } catch (const Error& e) {
                return std::string(e.what()).find(part) != std::string::npos;
            }
            return false;
        };
        const auto open = [](const Dir& dir) { SafetensorsWeights::open_dir(dir.path); };
        const Raw codes = random_codes("w.weight", 130, 64, 5);
        {
            const Dir dir("xs_fp8_refuse_a");
            write_file(dir.path / "model.safetensors", {codes});
            CHECK(refuses([&] { open(dir); }, "w.weight_scale_inv"));
            CHECK(refuses([&] { open(dir); }, "'w.weight_scale'"));
        }
        {
            const Dir dir("xs_fp8_refuse_e");  // both layouts' scales beside one tensor
            write_file(dir.path / "model.safetensors",
                       {codes, scale_grid("w.weight_scale_inv", 130, 64), row_scales("w.weight_scale", 130)});
            CHECK(refuses([&] { open(dir); }, "only a matrix beside its scales"));
        }
        {
            const Dir dir("xs_fp8_refuse_f");  // a scale per row, for another number of rows
            write_file(dir.path / "model.safetensors", {codes, row_scales("w.weight_scale", 64)});
            CHECK(refuses([&] { open(dir); }, "[64, 1]"));
            CHECK(refuses([&] { open(dir); }, "[130, 1]"));
        }
        {
            const Dir dir("xs_fp8_refuse_b");  // one scale per 64 rows: not this layout
            write_file(dir.path / "model.safetensors", {codes, scale_grid("w.weight_scale_inv", 3 * 128, 64)});
            CHECK(refuses([&] { open(dir); }, "[3, 1]"));
            CHECK(refuses([&] { open(dir); }, "[2, 1]"));
        }
        {
            const Dir dir("xs_fp8_refuse_c");
            Raw bank = random_codes("w.weight", 4, 64, 5);
            bank.shape = {2, 2, 64};
            write_file(dir.path / "model.safetensors", {bank, scale_grid("w.weight_scale_inv", 4, 64)});
            CHECK(refuses([&] { open(dir); }, "only a matrix"));
        }
        {
            const Dir dir("xs_fp8_refuse_d");
            const Raw other = random_codes("v.weight", 16, 32, 6);
            write_file(dir.path / "model.safetensors",
                       {codes, scale_grid("w.weight_scale_inv", 130, 64), other,
                        scale_grid("v.weight_scale_inv", 16, 32),
                        Raw{"dense.weight", DType::F32, {1, 64}, std::vector<uint8_t>(256)},
                        Raw{"half.weight", DType::BF16, {1, 64}, std::vector<uint8_t>(128)},
                        Raw{"ids", DType::I32, {1, 64}, std::vector<uint8_t>(256)}});
            const auto store = SafetensorsWeights::open_dir(dir.path);
            const auto join = [&](const std::string& name, const std::vector<std::string>& parts, const Dims& shape) {
                return [&store, name, parts, shape] { store->join_rows(name, parts, shape); };
            };
            CHECK(refuses(join("bank", {"w.weight", "dense.weight"}, {131, 64}),
                          "'dense.weight', a part of 'bank', is f32"));
            CHECK(refuses(join("bank", {"dense.weight", "half.weight"}, {2, 64}), "is bf16"));
            CHECK(refuses(join("bank", {"ids", "ids"}, {2, 64}), "is i32"));
            CHECK(refuses(join("bank", {"w.weight", "v.weight"}, {146, 64}), "64 columns"));
            CHECK(refuses(join("bank", {"w.weight", "absent"}, {131, 64}), "is missing"));
            CHECK(refuses(join("bank", {"w.weight"}, {2, 64, 64}), "130 rows"));
            CHECK(refuses(join("w.weight", {"w.weight"}, {130, 64}), "exists already"));
            CHECK(store->contains("w.weight"));  // a refused join changes nothing
        }
        // Codes on their own (one file's tensor, not a weight of the store) have no values.
        const TensorView raw{DType::F8, {130, 64}, codes.bytes.data()};
        std::vector<float> out(130 * 64);
        CHECK_THROWS_AS(quant::read_rows(raw, 0, 130, out.data()), Error);
    }
}
