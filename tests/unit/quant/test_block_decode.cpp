// GGUF block formats decoded on the host (quant/block_decode.h) against ggml's own dequantization, bit for bit: every
// format on random and edge blocks, the block layouts and codebook tables ggml was built with, and planted bugs the
// fixtures must catch. Fixtures: tests/fixtures/quant/blocks, from tests/parity/quant/gen_block_cases.py.
#include <doctest/doctest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "quant/block_decode.h"
#include "quant/block_tables.h"
#include "util/half.h"

using namespace xs;
namespace fs = std::filesystem;

namespace {

const fs::path kFixtures = fs::path(XS_SOURCE_DIR) / "tests/fixtures/quant/blocks";

std::vector<uint8_t> read_bytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE_MESSAGE(in.good(), "missing fixture ", path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A format's fixture: its blocks and ggml's values of them.
struct Case {
    std::vector<uint8_t> blocks;
    std::vector<float> values;
    int64_t count = 0;
};

Case load(BlockFormat f) {
    const std::string name(block_format_name(f));
    Case c;
    c.blocks = read_bytes(kFixtures / (name + ".blocks"));
    const std::vector<uint8_t> values = read_bytes(kFixtures / (name + ".values"));
    const BlockLayout l = block_layout(f);
    c.count = static_cast<int64_t>(c.blocks.size()) / l.bytes;
    REQUIRE(c.blocks.size() == static_cast<size_t>(c.count * l.bytes));
    REQUIRE(values.size() == static_cast<size_t>(c.count * l.weights) * sizeof(float));
    c.values.resize(values.size() / sizeof(float));
    std::memcpy(c.values.data(), values.data(), values.size());  // little-endian, as the host
    return c;
}

// "" when `got` has the bits of `want`; else where they first differ: block, weight, both values.
std::string difference(const std::vector<float>& got, const std::vector<float>& want, int64_t weights) {
    if (got.size() != want.size()) return "different sizes";
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            std::ostringstream s;
            s << "block " << static_cast<int64_t>(i) / weights << ", weight " << static_cast<int64_t>(i) % weights
              << ": " << got[i] << " instead of ggml's " << want[i];
            return s.str();
        }
    }
    return "";
}

std::vector<float> decoded(BlockFormat f, const Case& c) {
    std::vector<float> out(static_cast<size_t>(c.count * block_layout(f).weights));
    quant::decode_blocks(f, c.blocks.data(), c.count, out.data());
    return out;
}

float f16_at(const uint8_t* p) {
    return f16_to_f32(static_cast<uint16_t>(p[0] | (p[1] << 8)));
}

// Little-endian integers of `bytes`, read one after another.
struct Reader {
    const std::vector<uint8_t>& bytes;
    size_t at = 0;
    template <typename T>
    bool same_as(const T& table) {
        using V = typename T::value_type;
        for (const V v : table) {
            uint64_t u = 0;
            for (size_t k = 0; k < sizeof(V); ++k)
                u |= uint64_t{bytes.at(at++)} << (8 * k);
            if (static_cast<V>(u) != v) return false;
        }
        return true;
    }
};

}  // namespace

TEST_SUITE("block formats") {
    TEST_CASE("every block format decodes bit for bit as ggml does, on random and edge blocks") {
        for (int i = 0; i < kBlockFormats; ++i) {
            const auto f = static_cast<BlockFormat>(i);
            CAPTURE(block_format_name(f));
            const Case c = load(f);
            CHECK(c.count == 40);  // 32 random blocks, 8 edge blocks
            const int64_t weights = block_layout(f).weights;
            CHECK(difference(decoded(f, c), c.values, weights) == "");
            // As the rows of a weight, a block a row.
            TensorView v;
            v.dtype = DType::Block;
            v.shape = {c.count, weights};
            v.block = f;
            v.data = c.blocks.data();
            std::vector<float> rows(c.values.size());
            quant::decode_block_rows(v, 0, c.count, rows.data());
            CHECK(difference(rows, c.values, weights) == "");
        }
    }

    TEST_CASE("block_layout and the format names agree with ggml's blocks") {
        std::ifstream in(kFixtures / "layout.txt");
        REQUIRE(in.good());
        int formats = 0;
        for (std::string name; in >> name;) {
            int32_t bytes = 0, weights = 0;
            in >> bytes >> weights;
            CAPTURE(name);
            const std::optional<BlockFormat> f = block_format_from_name(name);
            REQUIRE(f.has_value());
            CHECK(block_format_name(*f) == name);
            CHECK(block_layout(*f).bytes == bytes);
            CHECK(block_layout(*f).weights == weights);
            ++formats;
        }
        CHECK(formats == kBlockFormats);
        CHECK_FALSE(block_format_from_name("q4_k").has_value());
    }

    TEST_CASE("the codebook tables are the ones ggml decodes with, value for value") {
        const std::vector<uint8_t> bytes = read_bytes(kFixtures / "tables.bin");
        Reader r{bytes};
        CHECK(r.same_as(quant::kmask_iq2xs));
        CHECK(r.same_as(quant::ksigns_iq2xs));
        CHECK(r.same_as(quant::kvalues_iq4nl));
        CHECK(r.same_as(quant::iq2xxs_grid));
        CHECK(r.same_as(quant::iq2xs_grid));
        CHECK(r.same_as(quant::iq2s_grid));
        CHECK(r.same_as(quant::iq3xxs_grid));
        CHECK(r.same_as(quant::iq3s_grid));
        CHECK(r.same_as(quant::iq1s_grid));
        CHECK(r.at == bytes.size());
    }

    TEST_CASE("planted bugs: decoders with a wrong layout must not reproduce ggml's values") {
        SUBCASE("Q4_0 read with the q4 pack's nibble order (low nibble: the even column)") {
            const Case c = load(BlockFormat::Q4_0);
            std::vector<float> wrong(c.values.size());
            for (int64_t b = 0; b < c.count; ++b) {
                const uint8_t* block = c.blocks.data() + b * 18;
                const float d = f16_at(block);
                for (int j = 0; j < 32; ++j) {
                    const uint8_t byte = block[2 + j / 2];
                    const int q = j % 2 == 0 ? byte & 0x0F : byte >> 4;
                    wrong[static_cast<size_t>(b * 32 + j)] = static_cast<float>(q - 8) * d;
                }
            }
            CHECK(difference(wrong, c.values, 32) != "");
        }
        SUBCASE("Q4_K with the scales and minimums of sub-blocks 4..7 read as those of 0..3 are") {
            const Case c = load(BlockFormat::Q4_K);
            std::vector<float> wrong(c.values.size());
            for (int64_t b = 0; b < c.count; ++b) {
                const uint8_t* block = c.blocks.data() + b * 144;
                const float d = f16_at(block), min = f16_at(block + 2);
                const uint8_t* scales = block + 4;
                for (int p = 0; p < 256; ++p) {
                    const int j = p / 32;                                    // sub-block
                    const int sc = scales[j] & 63, mn = scales[j + 4] & 63;  // right for j < 4 only
                    const uint8_t byte = block[16 + 32 * (j / 2) + p % 32];
                    const int q = j % 2 == 0 ? byte & 0xF : byte >> 4;
                    const float v = d * static_cast<float>(sc) * static_cast<float>(q);
                    wrong[static_cast<size_t>(b * 256 + p)] = v - min * static_cast<float>(mn);
                }
            }
            CHECK(difference(wrong, c.values, 256) != "");
            // The control is ggml's decode but for the planted bug: sub-blocks 0..3 come out right.
            for (int64_t b = 0; b < c.count; ++b) {
                const auto at = static_cast<size_t>(b * 256);
                CHECK(std::memcmp(&wrong[at], &c.values[at], 128 * sizeof(float)) == 0);
            }
        }
    }
}
