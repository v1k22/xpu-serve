// A model given by its path (hub/model_source.h): a GGUF file imported into xpu-serve's cache once and again when one
// of its files changes, the companions its import reads found beside it by -hf's rules or named by the user, a split
// file by its first part, a directory as it is, and the refusals.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "convert/tiny_inkling2_gguf.h"
#include "fake_hub.h"
#include "hub/model_source.h"
#include "io/gguf.h"
#include "util/error.h"

using namespace xs;
using namespace xs::hub;
namespace fs = std::filesystem;
using xs::test::HubTempDir;
using xs::test::import_into;
using xs::test::pack_files;

namespace {

SourceOptions cache_in(const fs::path& dir) {
    SourceOptions o;
    o.cache = dir / "xs";
    return o;
}

// The file's modification time a second later: what a rewrite does, whatever the file system's clock resolution.
void touch(const fs::path& file) { fs::last_write_time(file, fs::last_write_time(file) + std::chrono::seconds(1)); }

}  // namespace

TEST_SUITE("model source") {
    TEST_CASE("a GGUF file by its path: imported into the cache once, again when it changes") {
        HubTempDir dir("xs_source_gguf");
        const SourceOptions o = cache_in(dir.path);
        fs::create_directories(dir.path / "models");
        const fs::path file = dir.path / "models" / "Tiny-Q8_0.gguf";
        xs::test::build_tiny_gguf(dir.path / "a", 3).write(file);
        import_into(file, dir.path / "a-pack");

        const ModelSource m = resolve_model({.path = file.string()}, FileRole::Model, o);
        CHECK(m.name == "Tiny-Q8_0");
        CHECK(m.dir.filename() == "Tiny-Q8_0");
        CHECK(m.dir.parent_path().parent_path() == o.cache / "packs" / "local");
        CHECK(pack_files(m.dir) == pack_files(dir.path / "a-pack"));
        CHECK(fs::exists(m.dir / ".source"));
        // Again, and by another spelling of the path: the same pack, not imported again.
        const auto written = fs::last_write_time(m.dir / "model.safetensors");
        fs::create_directory_symlink(dir.path / "models", dir.path / "link");
        CHECK(resolve_model({.path = (dir.path / "link" / "Tiny-Q8_0.gguf").string()}, FileRole::Model, o).dir ==
              m.dir);
        CHECK(fs::last_write_time(m.dir / "model.safetensors") == written);
        CHECK_FALSE(fs::exists(m.dir.string() + ".partial"));

        // Another model written over the file: imported again, in the same place.
        xs::test::build_tiny_gguf(dir.path / "b", 8).write(file);
        touch(file);
        import_into(file, dir.path / "b-pack");
        REQUIRE(pack_files(dir.path / "b-pack") != pack_files(dir.path / "a-pack"));
        CHECK(resolve_model({.path = file.string()}, FileRole::Model, o).dir == m.dir);
        CHECK(pack_files(m.dir) == pack_files(dir.path / "b-pack"));
    }

    TEST_CASE("a GGUF file's companions: found beside it as -hf takes them from a repository, or named") {
        HubTempDir dir("xs_source_companions");
        const SourceOptions o = cache_in(dir.path);
        const xs::test::TinyMtpFiles files = xs::test::split_tiny_gguf(xs::test::build_tiny_gguf(dir.path / "src", 5));
        const fs::path folder = dir.path / "Tiny-GGUF";
        fs::create_directories(folder / "MTP");
        files.model.write(folder / "Tiny-Q8_0.gguf");
        files.mtp.write(folder / "MTP" / "mtp-Tiny-Q8_0.gguf");
        std::ofstream(folder / "mmproj-F16.gguf") << "a projector the model's import does not read";
        import_into(folder / "Tiny-Q8_0.gguf", dir.path / "with-head", folder / "MTP" / "mtp-Tiny-Q8_0.gguf");

        const SourceFiles s = source_files({.path = (folder / "Tiny-Q8_0.gguf").string()}, FileRole::Model, o);
        CHECK(s.mtp == folder / "MTP" / "mtp-Tiny-Q8_0.gguf");
        CHECK(s.mmproj.empty());
        const ModelSource m = model_dir(s, o);
        CHECK(m.dir.filename() == "Tiny-Q8_0+mtp-Tiny-Q8_0");
        CHECK(pack_files(m.dir) == pack_files(dir.path / "with-head"));
        // A drafter takes no companions; a head the user names wins.
        CHECK(source_files({.path = (folder / "Tiny-Q8_0.gguf").string()}, FileRole::Drafter, o).mtp.empty());
        fs::copy_file(folder / "MTP" / "mtp-Tiny-Q8_0.gguf", dir.path / "head.gguf");
        const ModelSpec named{.path = (folder / "Tiny-Q8_0.gguf").string(), .mtp = dir.path / "head.gguf"};
        CHECK(resolve_model(named, FileRole::Model, o).dir.filename() == "Tiny-Q8_0+head");
        // A companion that changes is imported again with the model.
        const auto written = fs::last_write_time(m.dir / "model.safetensors");
        touch(folder / "MTP" / "mtp-Tiny-Q8_0.gguf");
        CHECK(resolve_model({.path = (folder / "Tiny-Q8_0.gguf").string()}, FileRole::Model, o).dir == m.dir);
        CHECK(fs::last_write_time(m.dir / "model.safetensors") != written);

        // Inkling 2's import reads its drafter and its vision tower: both found beside it.
        fs::create_directories(dir.path / "inkling");
        xs::test::tiny_inkling2_gguf(true).write(dir.path / "inkling" / "Tiny-Inkling-Q8_0.gguf");
        xs::test::tiny_inkling2_drafter_gguf(true).write(dir.path / "inkling" / "dspark-Tiny-Inkling-Q8_0.gguf");
        for (const char* p : {"mmproj-F32.gguf", "mmproj-BF16.gguf"})
            std::ofstream(dir.path / "inkling" / p) << "a projector";
        const SourceFiles ink =
            source_files({.path = (dir.path / "inkling" / "Tiny-Inkling-Q8_0.gguf").string()}, FileRole::Model, o);
        CHECK(ink.mtp == dir.path / "inkling" / "dspark-Tiny-Inkling-Q8_0.gguf");
        CHECK(ink.mmproj == dir.path / "inkling" / "mmproj-BF16.gguf");
    }

    TEST_CASE("a split GGUF file by its first part, imported as the whole file; a later part refused by name") {
        HubTempDir dir("xs_source_split");
        const SourceOptions o = cache_in(dir.path);
        const xs::test::GgufWriter whole = xs::test::build_tiny_gguf(dir.path / "src", 6);
        whole.write(dir.path / "Tiny-Q8_0.gguf");
        import_into(dir.path / "Tiny-Q8_0.gguf", dir.path / "whole");
        fs::create_directories(dir.path / "parts");
        const auto parts = whole.split(12);
        const int count = static_cast<int>(parts.size());
        const fs::path first = xs::test::GgufWriter::write_parts(parts, (dir.path / "parts" / "Tiny-Q8_0").string());
        const ModelSource m = resolve_model({.path = first.string()}, FileRole::Model, o);
        CHECK(m.name == "Tiny-Q8_0");
        CHECK(m.dir.filename() == "Tiny-Q8_0");
        CHECK(pack_files(m.dir) == pack_files(dir.path / "whole"));
        const fs::path second = gguf::split_path((dir.path / "parts" / "Tiny-Q8_0").string(), 2, count);
        CHECK_THROWS_WITH_AS(resolve_model({.path = second.string()}, FileRole::Model, o),
                             doctest::Contains("give its first part"), Error);
        // Any part that changes is a change of the model.
        const auto written = fs::last_write_time(m.dir / "model.safetensors");
        touch(gguf::split_path((dir.path / "parts" / "Tiny-Q8_0").string(), count, count));
        resolve_model({.path = first.string()}, FileRole::Model, o);
        CHECK(fs::last_write_time(m.dir / "model.safetensors") != written);
    }

    TEST_CASE("a directory is the model as it is; what is neither a directory nor a GGUF file is refused") {
        HubTempDir dir("xs_source_dir");
        const SourceOptions o = cache_in(dir.path);
        const ModelSource m = resolve_model({.path = dir.path.string()}, FileRole::Model, o);
        CHECK(m.dir == dir.path);
        CHECK(m.name.empty());
        CHECK_FALSE(fs::exists(o.cache));
        CHECK_THROWS_WITH_AS(resolve_model({.path = (dir.path / "absent").string()}, FileRole::Model, o),
                             doctest::Contains("does not exist"), Error);
        std::ofstream(dir.path / "notes.txt") << "not a model";
        CHECK_THROWS_WITH_AS(resolve_model({.path = (dir.path / "notes.txt").string()}, FileRole::Model, o),
                             doctest::Contains("is neither a model directory nor a GGUF file"), Error);
        xs::test::GgufWriter().string("general.architecture", "x").write(dir.path / "x.gguf");
        const ModelSpec with_tower{.path = dir.path.string(), .mmproj = dir.path / "x.gguf"};
        CHECK_THROWS_WITH_AS(resolve_model(with_tower, FileRole::Model, o),
                             doctest::Contains("--mmproj names a GGUF file's vision tower"), Error);
        CHECK_THROWS_WITH_AS(resolve_model({.path = (dir.path / "x.gguf").string(), .mtp = dir.path / "notes.txt"},
                                           FileRole::Model, o),
                             doctest::Contains("is not a GGUF file"), Error);
        CHECK_THROWS_WITH_AS(resolve_model({}, FileRole::Model, o), doctest::Contains("give one"), Error);
    }
}
