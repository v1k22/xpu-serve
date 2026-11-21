// Which files of a Hugging Face repository the hub client downloads (hub/select.h), on listings shaped as real
// repositories are: a sentence-transformers checkpoint with other runtimes' weights beside it, a decision model with
// variants in subfolders, unsloth's GGUF repositories with dynamic quantizations, split files, vision projectors and
// MTP drafter files.
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "hub/select.h"
#include "util/error.h"

using namespace xs;
using namespace xs::hub;

namespace {

RepoListing listing(const std::string& repo, const std::vector<std::string>& paths) {
    RepoListing l;
    l.repo = repo;
    l.commit = std::string(40, 'a');
    for (const std::string& p : paths)
        l.files.push_back(RepoFile{p, 100, "b", ""});
    return l;
}

bool has(const Selection& s, const std::string& path) {
    return std::find(s.paths.begin(), s.paths.end(), path) != s.paths.end();
}

const std::vector<std::string> kQwen38Gguf = {
    ".gitattributes", "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q4_0.gguf", "Qwen3.8-27B-Q8_0.gguf", "Qwen3.8-27B-UD-IQ2_XXS.gguf",
    "Qwen3.8-27B-UD-Q4_K_M.gguf", "Qwen3.8-27B-UD-Q4_K_XL.gguf", "README.md", "config.json", "imatrix_unsloth.gguf",
    "mmproj-BF16.gguf", "mmproj-F16.gguf"};

const std::vector<std::string> kGemmaGguf = {
    "MTP/README.md", "MTP/mtp-gemma-4-E2B-it-BF16.gguf", "MTP/mtp-gemma-4-E2B-it-Q8_0.gguf", "README.md",
    "gemma-4-E2B-it-BF16.gguf", "gemma-4-E2B-it-Q4_K_M.gguf", "gemma-4-E2B-it-UD-Q4_K_XL.gguf",
    "gemma-4-E2B-it-UD-Q4_K_M.gguf", "mmproj-F16.gguf", "mtp-gemma-4-E2B-it.gguf"};

}  // namespace

TEST_SUITE("hub select") {
    TEST_CASE("specs: org/name, :QUANT, @revision; anything else refused by name") {
        HubSpec s = parse_hub_spec("Qwen/Qwen3.5-4B");
        CHECK(s.repo == "Qwen/Qwen3.5-4B");
        CHECK(s.revision == "main");
        CHECK(s.quant.empty());
        CHECK(s.name() == "Qwen/Qwen3.5-4B");
        s = parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL");
        CHECK(s.quant == "UD-Q4_K_XL");
        CHECK(s.name() == "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL");
        s = parse_hub_spec("org/name@v1.2:Q8_0");
        CHECK(s.repo == "org/name");
        CHECK(s.revision == "v1.2");
        CHECK(s.quant == "Q8_0");
        CHECK(s.name() == "org/name:Q8_0");
        for (const char* bad : {"Qwen3.5-4B", "/name", "org/", "a/b/c", "org/name:", "org/name@"}) {
            CAPTURE(bad);
            CHECK_THROWS_AS(parse_hub_spec(bad), Error);
        }
    }

    TEST_CASE("a checkpoint: safetensors and what goes with them; other runtimes' weights, pictures, git files left") {
        const Selection s = select_files(
            listing("sentence-transformers/all-MiniLM-L6-v2",
                    {".gitattributes", "1_Pooling/config.json", "README.md", "config.json", "model.safetensors",
                     "modules.json", "onnx/model.onnx", "onnx/model_O1.onnx", "openvino/openvino_model.bin",
                     "openvino/openvino_model.xml", "pytorch_model.bin", "rust_model.ot", "tf_model.h5",
                     "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                     "train_script.py"}),
            parse_hub_spec("sentence-transformers/all-MiniLM-L6-v2"), FileRole::Model);
        CHECK(s.gguf.empty());
        const std::vector<std::string> want = {"1_Pooling/config.json", "README.md", "config.json", "model.safetensors",
                                               "modules.json", "sentence_bert_config.json", "tokenizer.json",
                                               "tokenizer_config.json", "vocab.txt", "train_script.py"};
        CHECK(s.paths.size() == want.size());
        for (const std::string& p : want) {
            CAPTURE(p);
            CHECK(has(s, p));
        }
    }

    TEST_CASE("subfolders with weights of their own are other variants: left out; folders of configs kept") {
        const Selection s = select_files(
            listing("convaiinnovations/laya",
                    {"assets/logo.png", "config.json", "encoder/config.json", "eval/results.json", "model.safetensors",
                     "multilingual/encoder/config.json", "multilingual/model.safetensors",
                     "multilingual/tokenizer/tokenizer.json", "rl_agent_config.json", "rl_agent_api.py",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                     "typed-decisions/encoder/config.json", "typed-decisions/model.safetensors"}),
            parse_hub_spec("convaiinnovations/laya"), FileRole::Model);
        for (const char* p : {"config.json", "encoder/config.json", "model.safetensors", "rl_agent_config.json",
                              "rl_agent_api.py", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}) {
            CAPTURE(p);
            CHECK(has(s, p));
        }
        for (const char* p : {"assets/logo.png", "eval/results.json", "multilingual/model.safetensors",
                              "multilingual/tokenizer/tokenizer.json", "typed-decisions/model.safetensors"}) {
            CAPTURE(p);
            CHECK_FALSE(has(s, p));
        }
        // Weights only in a subfolder are the model's.
        const Selection sub = select_files(listing("o/m", {"config.json", "weights/model.safetensors"}),
                                           parse_hub_spec("o/m"), FileRole::Model);
        CHECK(has(sub, "weights/model.safetensors"));
    }

    TEST_CASE("a repository with neither safetensors nor GGUF files, or a quantization of a checkpoint, refused") {
        CHECK_THROWS_WITH_AS(select_files(listing("o/m", {"config.json", "pytorch_model.bin"}), parse_hub_spec("o/m"),
                                          FileRole::Model),
                             doctest::Contains("no safetensors weights and no GGUF file"), Error);
        CHECK_THROWS_WITH_AS(select_files(listing("o/m", {"config.json", "model.safetensors"}),
                                          parse_hub_spec("o/m:Q4_K_M"), FileRole::Model),
                             doctest::Contains("has no GGUF file"), Error);
    }

    TEST_CASE("GGUF: the file of the quantization asked for, the plain one before unsloth's dynamic, Q4_K_M by default") {
        const RepoListing qwen = listing("unsloth/Qwen3.8-27B-GGUF", kQwen38Gguf);
        CHECK(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:Q4_0"), FileRole::Model).paths ==
              std::vector<std::string>{"Qwen3.8-27B-Q4_0.gguf"});
        CHECK(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:q8_0"), FileRole::Model).gguf ==
              "Qwen3.8-27B-Q8_0.gguf");
        CHECK(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"), FileRole::Model).gguf ==
              "Qwen3.8-27B-UD-Q4_K_XL.gguf");
        // No plain Q4_K_M here: the dynamic one is the file of that name.
        CHECK(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF"), FileRole::Model).gguf ==
              "Qwen3.8-27B-UD-Q4_K_M.gguf");
        const RepoListing gemma = listing("unsloth/gemma-4-E2B-it-GGUF", kGemmaGguf);
        CHECK(select_files(gemma, parse_hub_spec("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"), FileRole::Model).gguf ==
              "gemma-4-E2B-it-Q4_K_M.gguf");
        CHECK(select_files(gemma, parse_hub_spec("unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_M"), FileRole::Model).gguf ==
              "gemma-4-E2B-it-UD-Q4_K_M.gguf");
        CHECK(select_files(gemma, parse_hub_spec("unsloth/gemma-4-E2B-it-GGUF"), FileRole::Model).gguf ==
              "gemma-4-E2B-it-Q4_K_M.gguf");
        // One GGUF file: no quantization needed. A repository of GGUF files only needs no ':'.
        CHECK(select_files(listing("ggml-org/Laya-GGUF", {"Laya-BF16.gguf", "README.md"}),
                           parse_hub_spec("ggml-org/Laya-GGUF"), FileRole::Model)
                  .gguf == "Laya-BF16.gguf");
    }

    TEST_CASE("GGUF refusals name what the repository has: no such file, no default, a split file") {
        const RepoListing qwen = listing("unsloth/Qwen3.8-27B-GGUF", kQwen38Gguf);
        CHECK_THROWS_WITH_AS(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:Q5_K_M"), FileRole::Model),
                             doctest::Contains("Qwen3.8-27B-Q4_0.gguf"), Error);
        CHECK_THROWS_WITH_AS(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:BF16"), FileRole::Model),
                             doctest::Contains("split into 2 parts"), Error);
        CHECK_THROWS_WITH_AS(select_files(listing("ggml-org/Laya-GGUF", {"Laya-BF16.gguf", "Laya-Q8_0.gguf"}),
                                          parse_hub_spec("ggml-org/Laya-GGUF"), FileRole::Model),
                             doctest::Contains("name one by its quantization"), Error);
        // Projectors and importance matrices are never the model.
        CHECK_THROWS_WITH_AS(select_files(qwen, parse_hub_spec("unsloth/Qwen3.8-27B-GGUF:F16"), FileRole::Model),
                             doctest::Contains("no GGUF file of quantization F16"), Error);
    }

    TEST_CASE("a drafter from a GGUF repository: its mtp-*.gguf, Q8_0 by default") {
        const RepoListing gemma = listing("unsloth/gemma-4-E2B-it-GGUF", kGemmaGguf);
        CHECK(select_files(gemma, parse_hub_spec("unsloth/gemma-4-E2B-it-GGUF"), FileRole::Drafter).gguf ==
              "MTP/mtp-gemma-4-E2B-it-Q8_0.gguf");
        CHECK(select_files(gemma, parse_hub_spec("unsloth/gemma-4-E2B-it-GGUF:BF16"), FileRole::Drafter).gguf ==
              "MTP/mtp-gemma-4-E2B-it-BF16.gguf");
        CHECK_THROWS_WITH_AS(
            select_files(listing("o/m-GGUF", {"m-Q4_0.gguf"}), parse_hub_spec("o/m-GGUF"), FileRole::Drafter),
            doctest::Contains("no drafter"), Error);
    }

    // unsloth/Inkling-2-Mini-GGUF: the model's files, their copies in another layout under a name that is not
    // .gguf, the DSpark drafter's two files and the vision projectors.
    TEST_CASE("a DSpark drafter's files (dspark-*.gguf) are drafters: never the model, fetched for a model file") {
        const RepoListing inkling = listing(
            "unsloth/Inkling-2-Mini-GGUF",
            {".gitattributes", "Inkling-2-Mini-UD-IQ1_S.gguf", "Inkling-2-Mini-UD-Q4_K_M.gguf",
             "Inkling-2-Mini-UD-Q4_K_XL.gguf", "Inkling-2-Mini-UD-Q8_K_XL.gguf", "Mainline_Support/HOW_TO_RUN.md",
             "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file", "Mainline_Support/README.md", "README.md",
             "assets/size_vs_kld.png", "dspark-Inkling-2-Mini-Q8_0.gguf", "dspark/README.md",
             "dspark/dspark-Inkling-2-Mini-BF16.gguf", "mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"});
        const auto pick = [&](const char* spec, FileRole role) {
            return select_files(inkling, parse_hub_spec(spec), role).gguf;
        };
        CHECK(pick("unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL", FileRole::Model) == "Inkling-2-Mini-UD-Q4_K_XL.gguf");
        CHECK(pick("unsloth/Inkling-2-Mini-GGUF", FileRole::Model) == "Inkling-2-Mini-UD-Q4_K_M.gguf");
        // the only Q8_0 and BF16 files are the drafter's
        CHECK_THROWS_AS(pick("unsloth/Inkling-2-Mini-GGUF:Q8_0", FileRole::Model), Error);
        CHECK_THROWS_AS(pick("unsloth/Inkling-2-Mini-GGUF:BF16", FileRole::Model), Error);
        CHECK(pick("unsloth/Inkling-2-Mini-GGUF", FileRole::Drafter) == "dspark-Inkling-2-Mini-Q8_0.gguf");
        CHECK(pick("unsloth/Inkling-2-Mini-GGUF:BF16", FileRole::Drafter) == "dspark/dspark-Inkling-2-Mini-BF16.gguf");
        CHECK(select_mtp_head(inkling, "Inkling-2-Mini-UD-Q4_K_XL.gguf") == "dspark-Inkling-2-Mini-Q8_0.gguf");
        CHECK(select_mtp_head(inkling, "Inkling-2-Mini-UD-IQ1_S.gguf") == "dspark-Inkling-2-Mini-Q8_0.gguf");
    }

    TEST_CASE("the MTP head of a model file published apart: named after it, else the only one, else none") {
        const RepoListing qwen = listing("unsloth/Qwen3.8-27B-GGUF", kQwen38Gguf);
        CHECK(select_mtp_head(qwen, "Qwen3.8-27B-Q4_0.gguf") == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf");
        CHECK(select_mtp_head(qwen, "Qwen3.8-27B-UD-IQ2_XXS.gguf") == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf");
        CHECK(select_mtp_head(listing("o/m", {"m-Q4_0.gguf"}), "m-Q4_0.gguf").empty());
        const RepoListing two = listing("o/m", {"m-Q4_0.gguf", "MTP/mtp-m-Q4_0.gguf", "MTP/mtp-m-Q8_0.gguf"});
        CHECK(select_mtp_head(two, "m-Q4_0.gguf") == "MTP/mtp-m-Q4_0.gguf");
        CHECK(select_mtp_head(two, "m-IQ2_XS.gguf") == "MTP/mtp-m-Q8_0.gguf");
    }
}
