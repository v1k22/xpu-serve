// Weights in GGUF block formats (DType::Block) on the host: read_rows over a Block view, a pack's manifest entries for
// them, and the sign sketch of one. The decoders' values against ggml's, bit for bit: test_block_decode.cpp.
#include <doctest/doctest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "backend/backend_test_util.h"
#include "io/safetensors.h"
#include "io/safetensors_writer.h"
#include "quant/block_decode.h"
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

TensorView block_view(BlockFormat format, Dims shape, const std::vector<uint8_t>& bytes) {
    TensorView v;
    v.dtype = DType::Block;
    v.shape = std::move(shape);
    v.block = format;
    v.data = bytes.data();
    return v;
}

std::vector<float> rows_of(const TensorView& v, int64_t first, int64_t count) {
    std::vector<float> out(static_cast<size_t>(count * v.shape.back()));
    quant::read_rows(v, first, count, out.data());
    return out;
}

void put_f16(std::vector<uint8_t>& block, size_t at, float v) {
    const uint16_t h = f32_to_f16(v);
    block[at] = static_cast<uint8_t>(h & 0xFF);
    block[at + 1] = static_cast<uint8_t>(h >> 8);
}

// Whether running `fn` is refused with a message that contains every one of `parts`.
template <typename Fn>
bool refuses(const Fn& fn, const std::vector<std::string>& parts) {
    try {
        fn();
    } catch (const Error& e) {
        for (const std::string& part : parts) {
            if (std::string(e.what()).find(part) == std::string::npos) return false;
        }
        return true;
    }
    return false;
}

// A pack directory: `tensors` (name, dtype, shape, bytes) in one file, `entries` as the manifest's tensors.
void write_pack(const fs::path& dir,
                const std::vector<std::tuple<std::string, DType, Dims, std::vector<uint8_t>>>& tensors,
                const nlohmann::json& entries) {
    std::vector<OutUnit> units;
    for (const auto& [name, dtype, shape, bytes] : tensors) {
        OutUnit u;
        u.tensors = {OutTensor{name, dtype, shape}};
        u.write = [&bytes](std::ostream& out) {
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        };
        units.push_back(std::move(u));
    }
    write_safetensors(dir / "model.safetensors", units);
    std::ofstream(dir / kPackManifest) << nlohmann::json{{"pack_version", kPackVersion}, {"tensors", entries}}.dump();
}

}  // namespace

TEST_SUITE("block formats") {
    TEST_CASE("read_rows of a block weight: each column from its block's fields (the 32-weight formats)") {
        // One block per format, its scales exact in f16 and its codes all different, and the value of each column
        // written from the format's description: codes q of 4 bits (Q4_*: column c < 16 in the low nibble of byte c,
        // c >= 16 in the high nibble of byte c - 16), plus bit c of a 32-bit word for Q5_*; Q8_0 an int8 per column.
        const float d = 0.25f, m = -1.5f;
        for (const BlockFormat f :
             {BlockFormat::Q4_0, BlockFormat::Q4_1, BlockFormat::Q5_0, BlockFormat::Q5_1, BlockFormat::Q8_0}) {
            CAPTURE(block_format_name(f));
            const bool has_min = f == BlockFormat::Q4_1 || f == BlockFormat::Q5_1;
            const bool five = f == BlockFormat::Q5_0 || f == BlockFormat::Q5_1;
            std::vector<uint8_t> block(static_cast<size_t>(block_layout(f).bytes));
            put_f16(block, 0, d);
            if (has_min) put_f16(block, 2, m);
            const size_t qh_at = has_min ? 4 : 2;
            const uint32_t qh = 0x9E3779B9u;  // high bits, all patterns of neighbours
            const size_t qs_at = five ? qh_at + 4 : qh_at;
            if (five) std::memcpy(block.data() + qh_at, &qh, 4);
            for (size_t j = 0; j < block.size() - qs_at; ++j)
                block[qs_at + j] = static_cast<uint8_t>(j * 37 + 11);
            std::vector<float> expected(32);
            for (int c = 0; c < 32; ++c) {
                if (f == BlockFormat::Q8_0) {
                    expected[static_cast<size_t>(c)] = static_cast<float>(static_cast<int8_t>(block[qs_at + c])) * d;
                    continue;
                }
                const uint8_t byte = block[qs_at + static_cast<size_t>(c % 16)];
                int q = c < 16 ? byte & 0x0F : byte >> 4;
                if (five) q |= static_cast<int>((qh >> c) & 1) << 4;
                const int offset = has_min ? 0 : (five ? 16 : 8);
                expected[static_cast<size_t>(c)] = static_cast<float>(q - offset) * d + (has_min ? m : 0.0f);
            }
            CHECK(rows_of(block_view(f, {1, 32}, block), 0, 1) == expected);
        }
    }

    TEST_CASE("read_rows: any rows of a block weight as the whole decode has them; 3-D shapes are matrices of rows") {
        for (int i = 0; i < kBlockFormats; ++i) {
            const auto f = static_cast<BlockFormat>(i);
            CAPTURE(block_format_name(f));
            const int64_t k = 2 * block_layout(f).weights;  // 2 blocks a row
            const std::vector<uint8_t> bytes = test::random_blocks(f, 2 * 3 * 2, 5);
            std::vector<float> all(static_cast<size_t>(2 * 3 * k));
            quant::decode_blocks(f, bytes.data(), 12, all.data());
            const TensorView bank = block_view(f, {2, 3, k}, bytes);
            CHECK(bank.nbytes() == bytes.size());
            CHECK(rows_of(bank, 0, 6) == all);
            CHECK(rows_of(bank, 2, 3) == std::vector<float>(all.begin() + 2 * k, all.begin() + 5 * k));
            CHECK(rows_of(block_view(f, {6, k}, bytes), 5, 1) == std::vector<float>(all.end() - k, all.end()));
            CHECK(refuses([&] { rows_of(bank, 5, 2); }, {std::string(block_format_name(f)), "rows 5 to 7"}));
            CHECK(refuses([&] { rows_of(block_view(f, {4, k - 16}, bytes), 0, 1); },
                          {std::to_string(k - 16) + " columns", "whole blocks of " + std::to_string(k / 2)}));
        }
    }

    TEST_CASE("sign sketch of a block weight is the sketch of its values") {
        const std::vector<uint8_t> bytes = test::random_blocks(BlockFormat::Q5_1, 5 * 3, 8);
        const TensorView w = block_view(BlockFormat::Q5_1, {5, 96}, bytes);
        const std::vector<float> values = rows_of(w, 0, 5);
        const quant::SignSketch a = quant::sign_sketch(w);
        const quant::SignSketch b = quant::sign_sketch(TensorView{DType::F32, {5, 96}, values.data(), {}});
        CHECK(std::memcmp(a.signs.raw(), b.signs.raw(), a.signs.nbytes()) == 0);
        CHECK(std::memcmp(a.scales.raw(), b.scales.raw(), a.scales.nbytes()) == 0);
    }

    TEST_CASE("pack: a block format entry shows its U8 rows of blocks as one weight of the logical shape") {
        const Dir dir("xs_block_pack_test");
        // An expert bank [2, 3, 64] in Q8_0 (6 rows of 2 blocks) and a matrix [4, 32] in Q4_1, beside a plain tensor.
        const std::vector<uint8_t> bank = test::random_blocks(BlockFormat::Q8_0, 12, 3);
        const std::vector<uint8_t> matrix = test::random_blocks(BlockFormat::Q4_1, 4, 4);
        const std::vector<uint8_t> norm{0, 0, 128, 63, 0, 0, 0, 64};
        write_pack(dir.path,
                   {{"experts.weight", DType::U8, {6, 68}, bank},
                    {"proj.weight", DType::U8, {4, 20}, matrix},
                    {"norm.weight", DType::F32, {2}, norm}},
                   {{"experts.weight", {{"format", "Q8_0"}, {"shape", {2, 3, 64}}}},
                    {"proj.weight", {{"format", "Q4_1"}, {"shape", {4, 32}}}}});
        const auto store = SafetensorsWeights::open_dir(dir.path);
        CHECK(store->is_pack());
        CHECK(store->names() == std::vector<std::string>{"experts.weight", "norm.weight", "proj.weight"});
        const TensorView& e = store->get("experts.weight");
        CHECK(e.dtype == DType::Block);
        CHECK(e.block == BlockFormat::Q8_0);
        CHECK(e.shape == Dims{2, 3, 64});
        CHECK(e.nbytes() == bank.size());
        CHECK(std::memcmp(e.data, bank.data(), bank.size()) == 0);
        CHECK(rows_of(e, 0, 6) == rows_of(block_view(BlockFormat::Q8_0, {6, 64}, bank), 0, 6));
        const TensorView& p = store->get("proj.weight");
        CHECK(p.block == BlockFormat::Q4_1);
        CHECK(rows_of(p, 1, 2) == rows_of(block_view(BlockFormat::Q4_1, {4, 32}, matrix), 1, 2));
        CHECK(store->get("norm.weight").dtype == DType::F32);
    }

    TEST_CASE("pack: block format entries are refused by name when they do not fit") {
        const std::vector<uint8_t> bytes = test::random_blocks(BlockFormat::Q8_0, 4, 1);
        const auto refused = [&](const std::string& tag, const Dims& stored, const nlohmann::json& entry,
                                 const std::vector<std::string>& parts) {
            CAPTURE(tag);
            const Dir dir("xs_block_pack_refuse_" + tag);
            write_pack(dir.path, {{"w", DType::U8, stored, bytes}}, {{"w", entry}});
            return refuses([&] { SafetensorsWeights::open_dir(dir.path); }, parts);
        };
        CHECK(refused("name", {4, 34}, {{"format", "Q8_K"}, {"shape", {4, 32}}}, {"'w'", "unknown format 'Q8_K'"}));
        CHECK(refused("case", {4, 34}, {{"format", "q8_0"}, {"shape", {4, 32}}}, {"'w'", "unknown format 'q8_0'"}));
        CHECK(refused("cols", {2, 68}, {{"format", "Q8_0"}, {"shape", {3, 48}}}, {"'w'", "48 columns", "of 32"}));
        CHECK(refused("rank", {4, 34}, {{"format", "Q8_0"}, {"shape", {128}}}, {"'w'", "at least 2 dims"}));
        CHECK(refused("empty", {4, 34}, {{"format", "Q8_0"}, {"shape", {0, 32}}}, {"'w'", "at least 2 dims"}));
        CHECK(refused("stored", {2, 68}, {{"format", "Q8_0"}, {"shape", {4, 32}}}, {"'w'", "[2, 68]", "[4, 34]"}));
        CHECK(
            refused("group", {4, 34}, {{"format", "Q8_0"}, {"group", 32}, {"shape", {4, 32}}}, {"'w'", "key 'group'"}));
    }
}
