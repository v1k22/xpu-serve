#pragma once

// `--hf <org/name[:QUANT]>`: a model on the Hugging Face Hub as a directory the commands read. A checkpoint is its
// snapshot in the Hugging Face cache; a GGUF file is imported once into a pack in xpu-serve's own cache (with its MTP
// head, when the repository publishes the head in a file of its own); a checkpoint that needs a file its repository
// does not publish (missing_companion_files) gets it beside its files.

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

// What a spec names, downloaded: a checkpoint's snapshot and the files it needs from other repositories, or a GGUF
// file (and the file of its MTP head). A checkpoint's weights come last, so one whose companion cannot be had (a gated
// repository without a token) is refused before they are downloaded.
struct SourceFiles {
    std::filesystem::path snapshot;
    std::vector<std::pair<std::string, std::filesystem::path>> companions;  // path beside the checkpoint, file
    std::filesystem::path gguf;  // empty for a checkpoint
    std::filesystem::path mtp;   // a GGUF file's MTP head, when it has none and the repository publishes one
    std::filesystem::path mmproj;  // a GGUF file's vision tower, for a model whose importer reads one
    std::string name;            // HubSpec::name
    std::string folder;          // the repository's cache folder name, `models--org--name`
    std::string commit;
};
SourceFiles fetch_source(const std::string& spec, FileRole role, const SourceOptions& options);

struct ModelSource {
    std::filesystem::path dir;  // what serve, run, bench and convert read
    std::string name;           // what /v1/models calls it: org/name, or org/name:QUANT
};
ModelSource resolve_model(const std::string& spec, FileRole role, const SourceOptions& options);

}  // namespace xs::hub
