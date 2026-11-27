#pragma once

// A command's model, whatever names it, as the directory the command reads. A model directory is itself. A GGUF file
// (a split model by its first part) is imported into a pack in xpu-serve's cache, once, and again when one of its
// files changes, with the files of its own its import reads: its MTP head or drafter, its vision tower, as the user
// names them or else as they are found beside it. A repository of the Hugging Face Hub (`-hf org/name[:QUANT]`) is
// downloaded into the Hugging Face cache: a checkpoint is its snapshot (with a file its repository does not publish,
// missing_companion_files, beside it), a GGUF file of it is imported as above with the companions the repository
// publishes. docs/decisions/0021, 0030, 0031.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "hub/hf_hub.h"
#include "hub/select.h"

namespace xs::hub {

struct SourceOptions {
    HubOptions hub;
    std::filesystem::path cache;  // xpu-serve's: imported packs, checkpoints with companion files
    std::string converter = "xpu-serve";  // recorded in an imported pack's manifest
    // HubOptions::from_env, and XS_CACHE_DIR, else $XDG_CACHE_HOME/xpu-serve, else ~/.cache/xpu-serve.
    static SourceOptions from_env();
};

// What a command is given as its model.
struct ModelSpec {
    std::string path;  // a model directory, or a GGUF file
    std::string hf;    // in place of `path`: org/name[@revision][:QUANT]
    // A GGUF model's MTP head (or drafter) and vision tower from files the user names (--mtp, --mmproj), in place of
    // the ones found beside it or published with it.
    std::filesystem::path mtp;
    std::filesystem::path mmproj;
};

// What a spec names, on disk (for -hf, downloaded): a checkpoint's directory and the files it needs from other
// repositories, or a GGUF file and its companions. A checkpoint's weights come last, so one whose companion cannot be
// had (a gated repository without a token) is refused before they are downloaded.
struct SourceFiles {
    std::filesystem::path snapshot;  // a checkpoint's directory: the model directory, or the repository's snapshot
    std::vector<std::pair<std::string, std::filesystem::path>> companions;  // path beside the checkpoint, file
    std::filesystem::path gguf;      // empty for a checkpoint
    std::filesystem::path mtp;       // a GGUF file's MTP head or drafter, when it has none in it
    std::filesystem::path mmproj;    // a GGUF file's vision tower, for a model whose importer reads one
    std::string name;  // what /v1/models calls it: org/name[:QUANT], or a GGUF file's model name; empty for a directory
    // Every file a repository's at `commit` (-hf, none named by the user): what an import of them is filed under.
    std::string folder;  // the repository's cache folder name, `models--org--name`
    std::string commit;
};
SourceFiles source_files(const ModelSpec& spec, FileRole role, const SourceOptions& options);

struct ModelSource {
    std::filesystem::path dir;  // what serve, run, bench and convert read
    std::string name;           // SourceFiles::name
};
// The directory a command reads for `files`: a checkpoint's own (or a folder of links to it and its companions), or a
// GGUF file's pack.
ModelSource model_dir(const SourceFiles& files, const SourceOptions& options);
// model_dir(source_files(...)).
ModelSource resolve_model(const ModelSpec& spec, FileRole role, const SourceOptions& options);

}  // namespace xs::hub
