// GGUF files split into parts as llama.cpp's gguf-split writes them (gguf_test_util.h): the tiny qwen35 and qwen35moe
// GGUFs (tiny_gguf.h) in parts, with a first part of metadata alone or with tensors, and a model without its MTP head
// beside the head's file, both split, import into the pack the whole files give, every file byte for byte.
#include <doctest/doctest.h>

#include <filesystem>
#include <string>

#include "convert/import_test_util.h"
#include "convert/tiny_gguf.h"
#include "io/gguf.h"

using namespace xs;
namespace fs = std::filesystem;
using xs::test::GgufWriter;
using xs::test::import_into;
using xs::test::pack_files;

namespace {

// `w` in parts of `per_part` tensors under `dir`/`name`-0000k-of-0000N.gguf: the first part's path.
fs::path write_split(const GgufWriter& w, const fs::path& dir, const std::string& name, size_t per_part,
                     bool empty_first) {
    fs::create_directories(dir);
    return GgufWriter::write_parts(w.split(per_part, empty_first), (dir / name).string());
}

}  // namespace

TEST_SUITE("gguf split") {
    TEST_CASE("a split GGUF imports into the pack the whole file gives, byte for byte") {
        for (const std::string arch : {"qwen35", "qwen35moe"}) {
            CAPTURE(arch);
            const xs::test::TempDir work("xs_gguf_split_" + arch);
            const GgufWriter whole = arch == "qwen35" ? xs::test::build_tiny_gguf(work.path / "src", 5)
                                                      : xs::test::build_tiny_moe_gguf(work.path / "src", 5);
            whole.write(work.path / "Tiny-Q8_0.gguf");
            import_into(work.path / "Tiny-Q8_0.gguf", work.path / "whole");
            const auto reference = pack_files(work.path / "whole");
            CHECK(reference.size() >= 6);
            for (const auto& [per_part, empty_first] : {std::pair{size_t{7}, false}, std::pair{size_t{25}, true}}) {
                CAPTURE(per_part);
                CAPTURE(empty_first);
                const std::string at = "parts-" + std::to_string(per_part);
                const fs::path first = write_split(whole, work.path / at, "Tiny-Q8_0", per_part, empty_first);
                CHECK(gguf::File(first).parts().size() > 2);
                import_into(first, work.path / (at + "-pack"));
                CHECK(pack_files(work.path / (at + "-pack")) == reference);
            }
        }
    }

    TEST_CASE("a split model without its MTP head and the head's split file import as the whole files") {
        const xs::test::TempDir work("xs_gguf_split_mtp");
        const xs::test::TinyMtpFiles files = xs::test::split_tiny_gguf(xs::test::build_tiny_gguf(work.path / "src", 9));
        fs::create_directories(work.path);
        files.model.write(work.path / "Tiny-Q8_0.gguf");
        files.mtp.write(work.path / "mtp-Tiny-Q8_0.gguf");
        import_into(work.path / "Tiny-Q8_0.gguf", work.path / "whole", work.path / "mtp-Tiny-Q8_0.gguf");
        const fs::path model = write_split(files.model, work.path / "parts", "Tiny-Q8_0", 9, false);
        const fs::path mtp = write_split(files.mtp, work.path / "parts", "mtp-Tiny-Q8_0", 4, false);
        CHECK(gguf::File(mtp).parts().size() > 1);
        import_into(model, work.path / "split", mtp);
        CHECK(pack_files(work.path / "split") == pack_files(work.path / "whole"));
    }
}
