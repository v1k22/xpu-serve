#pragma once

// Which files of a Hugging Face repository xpu-serve needs: a checkpoint's safetensors and the files beside them, or
// one GGUF file of a repository of them.

#include <string>
#include <string_view>
#include <vector>

#include "hub/hf_hub.h"

namespace xs::hub {

// "org/name", "org/name:QUANT" (a GGUF file of the repository, by its quantization: Q4_K_M, UD-Q4_K_XL, BF16, ...),
// either with "@revision" after the name (a branch, tag or commit; main by default).
struct HubSpec {
    std::string repo;
    std::string revision = "main";
    std::string quant;
    std::string name() const;  // as given, without the revision: what /v1/models calls the model
};
HubSpec parse_hub_spec(std::string_view text);

// A model, or a drafter for one (a GGUF repository's drafter files are its `mtp-*.gguf` and `dspark-*.gguf`).
enum class FileRole { Model, Drafter };

struct Selection {
    std::vector<std::string> paths;  // to download
    std::string gguf;                // the GGUF file among them; empty for a checkpoint
};

// A checkpoint: every file but other formats' weights (PyTorch, ONNX, OpenVINO, TensorFlow, GGUF), pictures and
// media, and subfolders that hold weights of their own (other variants of the model). A GGUF repository (one with no
// safetensors, or a spec with a quantization): the one file of the quantization asked for, Q4_K_M by default (a
// drafter: Q8_0), never a vision projector (`mmproj`) or an importance matrix. Refused by name: no safetensors, no
// file of the quantization (the error lists the ones there are), several, a file split into parts.
Selection select_files(const RepoListing& listing, const HubSpec& spec, FileRole role);

// For a GGUF file of a model whose drafter is published apart (llama.cpp's draft model: an MTP head, `mtp-*.gguf`,
// or a DSpark drafter, `dspark-*.gguf`): the drafter's file named after the model's file, else the only one, else
// its Q8_0; empty when the repository has none.
std::string select_mtp_head(const RepoListing& listing, const std::string& model_file);

// A GGUF repository's vision tower (llama.cpp's projector, `mmproj-*.gguf`): its bf16 file, else its f16, else its
// f32, else the first; empty when the repository has none.
std::string select_projector(const RepoListing& listing);

}  // namespace xs::hub
