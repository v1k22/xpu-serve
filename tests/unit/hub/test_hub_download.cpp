// The hub client against a Hub in the test process (fake_hub.h): what lands in the cache and how (the Hugging Face
// layout: blobs by hash, a snapshot of relative links, refs), the second time nothing is downloaded, files fetched in
// ranges over several connections and resumed after a failure, a wrong byte caught, tokens sent to the Hub and never to
// the CDN, gated and missing repositories, offline use; then `--hf` as the commands use it: a GGUF file imported into
// a pack with the MTP head its repository publishes apart, and a checkpoint given the file it needs from another
// repository.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "convert/gguf/gguf_import.h"
#include "convert/tiny_gguf.h"
#include "fake_hub.h"
#include "hub/download.h"
#include "hub/hf_hub.h"
#include "hub/model_source.h"
#include "util/error.h"

using namespace xs;
using namespace xs::hub;
namespace fs = std::filesystem;
using xs::test::FakeHub;
using xs::test::HubTempDir;
using xs::test::pattern_bytes;
using xs::test::read_file;

namespace {

// A checkpoint-shaped repository: two small git files, one in a folder, and a weight file of several ranges.
void add_checkpoint(FakeHub& hub, const std::string& repo) {
    hub.add(repo, "config.json", R"({"model_type": "tiny"})", false);
    hub.add(repo, "1_Pooling/config.json", R"({"pooling_mode_mean_tokens": true})", false);
    hub.add(repo, "model.safetensors", pattern_bytes(4096 * 5 + 123, 1));
    hub.add(repo, "pytorch_model.bin", pattern_bytes(1000, 2));
}

fs::path fetch_all(FakeHub&, const std::string& repo, const HubOptions& o) {
    const RepoListing l = list_repo(repo, "main", o);
    const Selection s = select_files(l, parse_hub_spec(repo), FileRole::Model);
    return fetch_files(l, s.paths, o);
}

}  // namespace

TEST_SUITE("hub download") {
    TEST_CASE("a snapshot in the Hugging Face cache layout: blobs by hash, relative links, refs; the token to the Hub only") {
        FakeHub hub;
        add_checkpoint(hub, "org/tiny");
        HubTempDir cache("xs_hub_layout");
        const fs::path snap = fetch_all(hub, "org/tiny", hub.options(cache.path, FakeHub::kToken));
        const fs::path folder = cache.path / "models--org--tiny";
        CHECK(snap == folder / "snapshots" / FakeHub::kCommit);
        CHECK(read_file(folder / "refs" / "main") == FakeHub::kCommit);
        const std::string weights = pattern_bytes(4096 * 5 + 123, 1);
        CHECK(read_file(snap / "model.safetensors") == weights);
        CHECK(read_file(snap / "1_Pooling" / "config.json") == R"({"pooling_mode_mean_tokens": true})");
        CHECK_FALSE(fs::exists(snap / "pytorch_model.bin"));
        // As huggingface_hub lays it out: an LFS file's blob is named by its SHA-256, the link is relative.
        CHECK(fs::is_symlink(snap / "model.safetensors"));
        CHECK(fs::read_symlink(snap / "model.safetensors") ==
              fs::path("../../blobs") / xs::test::sha256_hex(weights));
        CHECK(fs::read_symlink(snap / "1_Pooling" / "config.json").string().rfind("../../../blobs/", 0) == 0);
        CHECK(hub.cdn_auth == 0);
        CHECK(hub.hub_auth > 0);
        CHECK(hub.cdn_requests >= 6);  // the weights in ranges of 4 KiB
        for (const auto& e : fs::recursive_directory_iterator(folder)) {
            CAPTURE(e.path().string());
            CHECK(e.path().string().find(".xs-") == std::string::npos);  // nothing of a download left behind
        }
    }

    TEST_CASE("what the cache holds is not downloaded again; offline, the cache alone serves") {
        FakeHub hub;
        add_checkpoint(hub, "org/tiny");
        HubTempDir cache("xs_hub_again");
        fetch_all(hub, "org/tiny", hub.options(cache.path));
        const int files = hub.file_requests;
        const fs::path again = fetch_all(hub, "org/tiny", hub.options(cache.path));
        CHECK(hub.file_requests == files);
        HubOptions offline = hub.options(cache.path);
        offline.offline = true;
        const int listings = hub.listings;
        const RepoListing l = list_repo("org/tiny", "main", offline);
        CHECK(hub.listings == listings);
        CHECK(l.from_cache);
        CHECK(l.commit == FakeHub::kCommit);
        CHECK(fetch_files(l, select_files(l, parse_hub_spec("org/tiny"), FileRole::Model).paths, offline) == again);
        CHECK_THROWS_WITH_AS(list_repo("org/other", "main", offline), doctest::Contains("no snapshot of org/other"),
                             Error);
        // A Hub that cannot be reached falls back to the cache.
        HubOptions unreachable = hub.options(cache.path);
        unreachable.endpoint = "http://127.0.0.1:1";
        CHECK(list_repo("org/tiny", "main", unreachable).from_cache);
        CHECK_THROWS_WITH_AS(list_repo("org/other", "main", unreachable), doctest::Contains("cannot reach"), Error);
    }

    TEST_CASE("an interrupted download resumes with the ranges it lacks; a wrong byte is caught and not kept") {
        FakeHub hub;
        const std::string weights = pattern_bytes(4096 * 12, 7);
        hub.add("org/big", "config.json", "{}", false);
        hub.add("org/big", "model.safetensors", weights);
        HubTempDir cache("xs_hub_resume");
        HubOptions o = hub.options(cache.path);
        o.connections = 1;
        hub.cdn_limit = 5;
        CHECK_THROWS_WITH_AS(fetch_all(hub, "org/big", o), doctest::Contains("run again to resume"), Error);
        const fs::path blob = cache.path / "models--org--big" / "blobs" / xs::test::sha256_hex(weights);
        CHECK(fs::exists(blob.string() + ".xs-partial"));
        CHECK_FALSE(fs::exists(blob));
        hub.cdn_limit = -1;
        const int before = hub.cdn_requests;
        const fs::path snap = fetch_all(hub, "org/big", o);
        CHECK(read_file(snap / "model.safetensors") == weights);
        CHECK(hub.cdn_requests - before < 12);  // the ranges written before were not asked again
        CHECK_FALSE(fs::exists(blob.string() + ".xs-partial"));

        // A list of ranges made with another range size is not trusted: the file starts over.
        const std::string other = pattern_bytes(4096 * 9, 8);
        hub.add("org/other", "model.safetensors", other);
        hub.cdn_limit = hub.cdn_served + 3;
        CHECK_THROWS_AS(fetch_all(hub, "org/other", o), Error);
        hub.cdn_limit = -1;
        HubOptions wider = o;
        wider.chunk_bytes = 8192;
        CHECK(read_file(fetch_all(hub, "org/other", wider) / "model.safetensors") == other);

        hub.add("org/bad", "model.safetensors", pattern_bytes(9000, 9));
        hub.corrupt = true;
        CHECK_THROWS_WITH_AS(fetch_all(hub, "org/bad", o), doctest::Contains("org/bad/model.safetensors: SHA-256"),
                             Error);
        hub.corrupt = false;
        const fs::path bad = cache.path / "models--org--bad" / "blobs" / xs::test::sha256_hex(pattern_bytes(9000, 9));
        CHECK_FALSE(fs::exists(bad));
        CHECK_FALSE(fs::exists(bad.string() + ".xs-partial"));
        CHECK(read_file(fetch_all(hub, "org/bad", o) / "model.safetensors") == pattern_bytes(9000, 9));
    }

    TEST_CASE("errors name the repository and what to do: not found or private, a revision, a gated repository") {
        FakeHub hub;
        add_checkpoint(hub, "org/tiny");
        hub.add("org/gated", "model.safetensors", pattern_bytes(100, 3));
        hub.set_gated("org/gated");
        HubTempDir cache("xs_hub_errors");
        CHECK_THROWS_WITH_AS(list_repo("org/missing", "main", hub.options(cache.path)),
                             doctest::Contains("or it is private: check the name, or set HF_TOKEN"), Error);
        CHECK_THROWS_WITH_AS(list_repo("org/tiny", "v9", hub.options(cache.path)),
                             doctest::Contains("no revision 'v9'"), Error);
        CHECK_THROWS_WITH_AS(fetch_all(hub, "org/gated", hub.options(cache.path)),
                             doctest::Contains("org/gated is gated: accept its terms"), Error);
        CHECK_THROWS_WITH_AS(fetch_all(hub, "org/gated", hub.options(cache.path, "hf_wrong")),
                             doctest::Contains("may be gated"), Error);
        CHECK(read_file(fetch_all(hub, "org/gated", hub.options(cache.path, FakeHub::kToken)) / "model.safetensors") ==
              pattern_bytes(100, 3));
        CHECK_THROWS_WITH_AS(list_repo("not-a-repo", "main", hub.options(cache.path)),
                             doctest::Contains("not a Hugging Face repository"), Error);
    }

    TEST_CASE("--hf of a GGUF file: imported into a pack once, with the MTP head its repository publishes apart") {
        for (const std::string arch : {"qwen35", "qwen35moe"}) {
            CAPTURE(arch);
            FakeHub hub;
            HubTempDir dir("xs_hub_gguf_" + arch);
            const test::TinyMtpFiles files =
                arch == "qwen35" ? test::split_tiny_gguf(test::build_tiny_gguf(dir.path / "tiny", 3))
                                 : test::split_tiny_gguf(test::build_tiny_moe_gguf(dir.path / "tiny", 3), arch);
            files.model.write(dir.path / "model.gguf");
            files.mtp.write(dir.path / "mtp.gguf");
            hub.add("org/Tiny-GGUF", "Tiny-Q8_0.gguf", read_file(dir.path / "model.gguf"));
            hub.add("org/Tiny-GGUF", "MTP/mtp-Tiny-Q8_0.gguf", read_file(dir.path / "mtp.gguf"));
            hub.add("org/Tiny-GGUF", "mmproj-F16.gguf", "not a model");
            hub.add("org/Tiny-GGUF", "README.md", "# Tiny", false);
            SourceOptions o;
            o.hub = hub.options(dir.path / "hf");
            o.cache = dir.path / "xs";
            const ModelSource m = resolve_model("org/Tiny-GGUF:Q8_0", FileRole::Model, o);
            CHECK(m.name == "org/Tiny-GGUF:Q8_0");
            REQUIRE(fs::exists(m.dir / "xpu_pack.json"));
            CHECK(m.dir.parent_path().parent_path().filename() == "models--org--Tiny-GGUF");
            CHECK(m.dir.filename() == "Tiny-Q8_0+mtp-Tiny-Q8_0");
            // The same pack convert writes from the two files.
            gguf_import::ImportOptions io;
            io.source = dir.path / "model.gguf";
            io.mtp = dir.path / "mtp.gguf";
            io.output = dir.path / "local";
            gguf_import::import_gguf(io);
            CHECK(read_file(m.dir / "model.safetensors") == read_file(dir.path / "local" / "model.safetensors"));
            CHECK(read_file(m.dir / "config.json") == read_file(dir.path / "local" / "config.json"));
            // Neither the projector nor the README was downloaded; the second time nothing is imported again.
            CHECK_FALSE(fs::exists(fs::path(o.hub.cache) / "models--org--Tiny-GGUF" / "snapshots" / FakeHub::kCommit /
                                   "mmproj-F16.gguf"));
            const auto written = fs::last_write_time(m.dir / "model.safetensors");
            const int requests = hub.file_requests;
            CHECK(resolve_model("org/Tiny-GGUF:Q8_0", FileRole::Model, o).dir == m.dir);
            CHECK(fs::last_write_time(m.dir / "model.safetensors") == written);
            CHECK(hub.file_requests == requests);
            CHECK_FALSE(fs::exists(m.dir.string() + ".partial"));
            // fetch_source hands convert the files themselves.
            const SourceFiles s = fetch_source("org/Tiny-GGUF:Q8_0", FileRole::Model, o);
            CHECK(s.gguf.filename() == "Tiny-Q8_0.gguf");
            CHECK(s.mtp.filename() == "mtp-Tiny-Q8_0.gguf");
        }
    }

    TEST_CASE("a checkpoint that reads a file its repository does not publish gets it beside its files") {
        FakeHub hub;
        hub.add("lerobot/tiny-pi05", "config.json", R"({"type": "pi05"})", false);
        hub.add("lerobot/tiny-pi05", "model.safetensors", pattern_bytes(5000, 4));
        hub.add("google/paligemma-3b-pt-224", "tokenizer.json", R"({"model": {}})", false);
        hub.add("google/paligemma-3b-pt-224", "model-00001-of-00003.safetensors", pattern_bytes(8000, 5));
        hub.set_gated("google/paligemma-3b-pt-224");
        HubTempDir dir("xs_hub_companion");
        SourceOptions o;
        o.hub = hub.options(dir.path / "hf");
        o.cache = dir.path / "xs";
        CHECK_THROWS_WITH_AS(resolve_model("lerobot/tiny-pi05", FileRole::Model, o),
                             doctest::Contains("google/paligemma-3b-pt-224 is gated"), Error);
        CHECK(hub.cdn_requests == 0);  // refused before the checkpoint's weights were downloaded
        o.hub.token = FakeHub::kToken;
        const ModelSource m = resolve_model("lerobot/tiny-pi05", FileRole::Model, o);
        CHECK(m.dir.parent_path().parent_path() == o.cache / "models");
        CHECK(read_file(m.dir / "config.json") == R"({"type": "pi05"})");
        CHECK(read_file(m.dir / "model.safetensors") == pattern_bytes(5000, 4));
        CHECK(read_file(m.dir / "tokenizer.json") == R"({"model": {}})");
        // Only the one file of the other repository.
        CHECK_FALSE(fs::exists(fs::path(o.hub.cache) / "models--google--paligemma-3b-pt-224" / "snapshots" /
                               FakeHub::kCommit / "model-00001-of-00003.safetensors"));
        // A checkpoint that needs nothing more is its snapshot.
        add_checkpoint(hub, "org/tiny");
        CHECK(resolve_model("org/tiny", FileRole::Model, o).dir ==
              fs::path(o.hub.cache) / "models--org--tiny" / "snapshots" / FakeHub::kCommit);
    }
}
