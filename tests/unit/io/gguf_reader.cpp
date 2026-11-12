// The GGUF reader on files the tests write (gguf_test_util.h): every metadata value type, arrays (nested, and a string
// array the size of a real vocabulary), tensors at the file's alignment, and the refusals by name.
#include <doctest/doctest.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "io/gguf.h"
#include "io/gguf_test_util.h"
#include "model/tiny_checkpoints.h"
#include "util/error.h"

using namespace xs;
using xs::test::GgufWriter;
using xs::test::pod_bytes;
using xs::test::TempDir;
using T = gguf::ValueType;

namespace {

// The error opening `bytes` as a GGUF file gives; empty when it opens.
std::string refusal(const std::filesystem::path& dir, const std::string& bytes) {
    const auto path = dir / "refused.gguf";
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    try {
        gguf::File file(path);
    } catch (const Error& e) {
        return e.what();
    }
    return "";
}

std::string f32_bytes(std::initializer_list<float> v) {
    return pod_bytes(std::vector<float>(v));
}

}  // namespace

TEST_SUITE("gguf reader") {
    TEST_CASE("metadata of every value type, nested arrays, tensors at the file's alignment") {
        const TempDir dir("xs_gguf_reader_test");
        std::filesystem::create_directories(dir.path);
        std::string q8(34, '\0');  // one Q8_0 block per row, two rows
        q8[0] = 1;
        std::string nested = pod_bytes(static_cast<uint32_t>(T::U16)) + pod_bytes(uint64_t{2}) +
                             pod_bytes(uint16_t{7}) + pod_bytes(uint16_t{8});
        nested += pod_bytes(static_cast<uint32_t>(T::U16)) + pod_bytes(uint64_t{1}) + pod_bytes(uint16_t{9});
        GgufWriter w;
        w.alignment(64)
            .u8("t.u8", 200)
            .i8("t.i8", -100)
            .u16("t.u16", 60000)
            .i16("t.i16", -30000)
            .u32("t.u32", 4000000000u)
            .i32("t.i32", -2000000000)
            .u64("t.u64", 1ull << 40)
            .i64("t.i64", -(1ll << 40))
            .f32("t.f32", 1e-6f)
            .f64("t.f64", 0.125)
            .boolean("t.bool", true)
            .string("t.string", "héllo")
            .strings("t.strings", {"a", "", "Ġthe"})
            .i32s("t.ints", {1, -2, 3})
            .f32s("t.floats", {0.5f, -1.0f})
            .bools("t.bools", {true, false})
            .array("t.nested", T::Array, 2, nested)
            .u64("t.huge", ~0ull)
            .f32_tensor("a", {3, 2}, {1, 2, 3, 4, 5, 6})
            .tensor("b", {2}, xs::test::kGgmlF16, pod_bytes(std::vector<uint16_t>{f32_to_f16(1.5f), f32_to_f16(-2)}))
            .tensor("c", {32, 2}, xs::test::kGgmlQ8_0, q8 + q8)
            .tensor("d", {256}, xs::test::kGgmlQ4_K, std::string(144, '\x5a'))
            .tensor("e", {1}, xs::test::kGgmlBF16, pod_bytes(f32_to_bf16(3.0f)));
        w.write(dir.path / "all.gguf");

        const gguf::File f(dir.path / "all.gguf");
        CHECK(f.version() == 3);
        CHECK(f.alignment() == 64);
        CHECK(f.data_offset() % 64 == 0);
        CHECK(f.metadata().size() == 19);
        CHECK(f.metadata()[1].key() == "t.u8");
        CHECK(f.get_int("t.u8") == 200);
        CHECK(f.get_int("t.i8") == -100);
        CHECK(f.get_int("t.u16") == 60000);
        CHECK(f.get_int("t.i16") == -30000);
        CHECK(f.get_int("t.u32") == 4000000000ll);
        CHECK(f.get_int("t.i32") == -2000000000);
        CHECK(f.get_int("t.u64") == (1ll << 40));
        CHECK(f.get_int("t.i64") == -(1ll << 40));
        CHECK(f.get("t.f32").type() == T::F32);
        CHECK(static_cast<float>(f.get_float("t.f32")) == 1e-6f);
        CHECK(f.get_float("t.f64") == 0.125);
        CHECK(f.get("t.bool").as_bool());
        CHECK(f.get_string("t.string") == "héllo");
        CHECK(f.get("t.strings").strings() == std::vector<std::string_view>{"a", "", "Ġthe"});
        CHECK(f.get("t.strings").size() == 3);
        CHECK(f.get("t.ints").ints() == std::vector<int64_t>{1, -2, 3});
        CHECK(f.get("t.floats").floats() == std::vector<double>{0.5, -1.0});
        CHECK(f.get("t.bools").bools() == std::vector<bool>{true, false});
        const std::vector<gguf::Value> rows = f.get("t.nested").elements();
        REQUIRE(rows.size() == 2);
        CHECK(rows[0].ints() == std::vector<int64_t>{7, 8});
        CHECK(rows[1].ints() == std::vector<int64_t>{9});
        CHECK(f.find("absent") == nullptr);

        REQUIRE(f.tensors().size() == 5);
        const gguf::TensorInfo& a = *f.find_tensor("a");
        CHECK(a.dims == std::vector<int64_t>{3, 2});
        CHECK(a.shape() == Dims{2, 3});
        CHECK(a.rows() == 2);
        CHECK(a.cols() == 3);
        CHECK(a.dtype == DType::F32);
        CHECK(std::memcmp(a.data, f32_bytes({1, 2, 3, 4, 5, 6}).data(), 24) == 0);
        CHECK(f.find_tensor("b")->dtype == DType::F16);
        CHECK(f.find_tensor("e")->dtype == DType::BF16);
        const gguf::TensorInfo& c = *f.find_tensor("c");
        CHECK(c.dtype == DType::Block);
        CHECK(c.block == BlockFormat::Q8_0);
        CHECK(c.nbytes == 68);
        CHECK(c.type_name() == "Q8_0");
        const gguf::TensorInfo& d = *f.find_tensor("d");
        CHECK(d.block == BlockFormat::Q4_K);
        CHECK(d.nbytes == 144);
        CHECK(static_cast<unsigned char>(d.data[143]) == 0x5a);
        for (const gguf::TensorInfo& t : f.tensors()) {
            CAPTURE(t.name);
            CHECK(t.offset % 64 == 0);
            CHECK(t.data - a.data == static_cast<std::ptrdiff_t>(t.offset - a.offset));
        }
        const TensorView v = c.view({2, 32});
        CHECK(v.dtype == DType::Block);
        CHECK(v.block == BlockFormat::Q8_0);
        CHECK(v.nbytes() == 68);
        CHECK_THROWS_AS(c.view({64}), Error);  // a block view keeps the rows' columns

        // Accessors check the type and name the key.
        CHECK_THROWS_WITH_AS(f.get_string("t.u8"), doctest::Contains("'t.u8' is u8, not a string"), Error);
        CHECK_THROWS_WITH_AS(f.get("t.ints").strings(), doctest::Contains("'t.ints' is array of i32"), Error);
        CHECK_THROWS_WITH_AS(f.get_int("t.huge"), doctest::Contains("beyond the int64 range"), Error);
        CHECK_THROWS_WITH_AS(f.get_float("t.i32"), doctest::Contains("not a float"), Error);
        CHECK_THROWS_WITH_AS(f.get("absent"), doctest::Contains("no metadata 'absent'"), Error);
        CHECK(gguf::has_gguf_magic(dir.path / "all.gguf"));
    }

    TEST_CASE("version 2 reads like version 3") {
        const TempDir dir("xs_gguf_reader_v2_test");
        std::filesystem::create_directories(dir.path);
        GgufWriter w;
        w.string("general.architecture", "x").f32_tensor("t", {4}, {1, 2, 3, 4});
        w.write(dir.path / "v2.gguf", 2);
        const gguf::File f(dir.path / "v2.gguf");
        CHECK(f.version() == 2);
        CHECK(f.alignment() == 32);
        CHECK(f.get_string("general.architecture") == "x");
        CHECK(f.find_tensor("t")->nbytes == 16);
    }

    TEST_CASE("a vocabulary-sized string array reads as views, without copying the strings") {
        const TempDir dir("xs_gguf_reader_vocab_test");
        std::filesystem::create_directories(dir.path);
        std::vector<std::string> vocab;
        for (int i = 0; i < 250000; ++i)
            vocab.push_back("tok" + std::to_string(i));
        GgufWriter w;
        w.strings("tokenizer.ggml.tokens", vocab);
        w.write(dir.path / "vocab.gguf");
        const gguf::File f(dir.path / "vocab.gguf");
        const auto start = std::chrono::steady_clock::now();
        const std::vector<std::string_view> tokens = f.get("tokenizer.ggml.tokens").strings();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        MESSAGE("250k strings as views in " << ms << " ms");
        REQUIRE(tokens.size() == vocab.size());
        CHECK(tokens[123456] == "tok123456");
        CHECK(tokens.back() == "tok249999");
    }

    TEST_CASE("refusals name what is wrong") {
        const TempDir dir("xs_gguf_reader_refusal_test");
        std::filesystem::create_directories(dir.path);
        const auto good = [] {
            GgufWriter w;
            w.string("general.architecture", "x").f32_tensor("w", {32, 2}, std::vector<float>(64, 1.0f));
            return w;
        };
        CHECK(refusal(dir.path, good().bytes()).empty());

        CHECK(refusal(dir.path, "GGML" + good().bytes().substr(4)).find("is not a GGUF file") != std::string::npos);
        CHECK(refusal(dir.path, good().bytes(1)).find("version 1; versions 2 and 3 are read") != std::string::npos);
        CHECK(refusal(dir.path, good().bytes(4)).find("version 4") != std::string::npos);
        CHECK(refusal(dir.path, good().bytes(0x03000000)).find("is big-endian") != std::string::npos);

        GgufWriter split = good();
        split.u16("split.count", 3);
        CHECK(refusal(dir.path, split.bytes()).find("split into 3 files (split.count)") != std::string::npos);
        GgufWriter one_part = good();
        one_part.u16("split.count", 1);
        CHECK(refusal(dir.path, one_part.bytes()).empty());

        GgufWriter ternary = good();
        ternary.tensor("blk.0.ffn_up.weight", {256}, xs::test::kGgmlTQ1_0, std::string(54, '\0'));
        const std::string tq = refusal(dir.path, ternary.bytes());
        CHECK(tq.find("tensor 'blk.0.ffn_up.weight' has type TQ1_0, which xpu-serve does not read") !=
              std::string::npos);
        CHECK(tq.find("IQ4_XS") != std::string::npos);  // the list of what it reads
        GgufWriter unknown = good();
        unknown.tensor("u", {4}, 99, std::string(16, '\0'));
        CHECK(refusal(dir.path, unknown.bytes()).find("has type type 99") != std::string::npos);
        GgufWriter ints = good();
        ints.tensor("i", {4}, 26, std::string(16, '\0'));
        CHECK(refusal(dir.path, ints.bytes()).find("has type I32") != std::string::npos);

        GgufWriter partial_block = good();
        partial_block.tensor("p", {40}, xs::test::kGgmlQ8_0, std::string(34, '\0'));
        CHECK(refusal(dir.path, partial_block.bytes()).find("rows of 40 columns, not a whole number of blocks of 32") !=
              std::string::npos);
        GgufWriter five = good();
        five.tensor("five", {1, 1, 1, 1, 1}, xs::test::kGgmlF32, std::string(4, '\0'));
        CHECK(refusal(dir.path, five.bytes()).find("'five' has 5 dimensions") != std::string::npos);

        GgufWriter twice = good();
        twice.string("general.architecture", "y");
        CHECK(refusal(dir.path, twice.bytes()).find("'general.architecture' appears twice") != std::string::npos);
        GgufWriter twin = good();
        twin.f32_tensor("w", {1}, {0});
        CHECK(refusal(dir.path, twin.bytes()).find("tensor 'w' appears twice") != std::string::npos);

        GgufWriter odd = good();
        odd.u32("general.alignment", 48);
        CHECK(refusal(dir.path, odd.bytes()).find("general.alignment 48 is not a power of two") != std::string::npos);
        GgufWriter signed_alignment = good();
        signed_alignment.i32("general.alignment", 64);
        CHECK(refusal(dir.path, signed_alignment.bytes()).find("'general.alignment' is i32, not a u32") !=
              std::string::npos);
        GgufWriter misaligned;  // tensors placed at 32 bytes, the file says 64
        misaligned.u32("general.alignment", 64)
            .f32_tensor("first", {8}, std::vector<float>(8, 0.0f))
            .f32_tensor("second", {8}, std::vector<float>(8, 0.0f));
        CHECK(refusal(dir.path, misaligned.bytes()).find("'second' starts at offset 32, not a multiple") !=
              std::string::npos);

        // Truncated: in the header, in a string, in an array's declared length, in the tensor data.
        const std::string whole = good().bytes();
        CHECK(refusal(dir.path, whole.substr(0, 10)).find("is truncated") != std::string::npos);
        CHECK(refusal(dir.path, whole.substr(0, 30)).find("is truncated") != std::string::npos);
        CHECK(refusal(dir.path, whole.substr(0, whole.size() - 1)).find("is truncated: tensor 'w' needs bytes") !=
              std::string::npos);
        GgufWriter lying;
        lying.array("tokenizer.ggml.tokens", T::String, 1ull << 60, "");
        CHECK(refusal(dir.path, lying.bytes()).find("declares 1152921504606846976 entries") != std::string::npos);
        GgufWriter bad_type;
        bad_type.raw("k", static_cast<T>(13), "");
        CHECK(refusal(dir.path, bad_type.bytes()).find("'k' has unknown value type 13") != std::string::npos);
        CHECK_FALSE(gguf::has_gguf_magic(dir.path / "absent.gguf"));
    }
}
