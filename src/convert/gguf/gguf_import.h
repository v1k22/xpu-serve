#pragma once

// `xpu-serve convert <file.gguf> <out-dir>`: a GGUF file (llama.cpp's format) in, an xpu-serve pack out, which the
// engine serves like any model directory.
//
// The pack: config.json (from the file's metadata, as the model code's config parser reads it), tokenizer.json and
// tokenizer_config.json (the file's vocabulary and merges; the normalizer and pre-tokenizer from a table keyed by
// tokenizer.ggml.pre), chat_template.jinja, generation_config.json when the file holds sampling defaults, one
// model.safetensors, and xpu_pack.json. A tensor in a block format (Q4_K, IQ2_XS, ...) is copied byte for byte as a U8
// tensor [rows, row bytes] and named in the manifest with its format and logical shape; a float tensor is written as
// stored, or as F32 when the import undoes a transform llama.cpp's converter applied to it. Tensors are streamed from
// the memory map: a file of any size never sits in memory.
//
// The architecture (general.architecture) decides names, config and transforms (convert/gguf/architecture.h); one
// it does not know is refused by name. A drafter's pack (a Gemma 4 assistant, for --drafter) is config.json, the
// weights and the manifest: the target's pack has the tokenizer and chat template.
//
// A model file without its MTP head can be imported with a second file that holds the head (`--mtp`, llama.cpp's
// draft model of it): the pack is the one a file that held the head gives. Only the head is read from the second
// file; tokenizer, chat template and sampling defaults are the model file's.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace xs::gguf_import {

struct ImportOptions {
    std::filesystem::path source;         // the .gguf file
    std::filesystem::path output;         // created; must not exist or be empty
    std::filesystem::path mtp;            // a .gguf file of the source's MTP head, or empty
    std::string converter = "xpu-serve";  // recorded in the manifest
};

// The tensors of one storage format the import reads from the files (every tensor but an MTP file's copies of the
// model's).
struct FormatShare {
    std::string format;  // "Q4_K", "F32", ...
    int64_t tensors = 0;
    int64_t weights = 0;  // values
    uint64_t bytes = 0;   // as stored
};

struct ImportReport {
    std::string architecture;          // general.architecture
    std::string model_name;            // general.name, when the file has one
    std::vector<FormatShare> formats;  // most weights first
    int64_t tensors = 0;               // of the files, as `formats` counts them
    int64_t weights = 0;
    uint64_t output_bytes = 0;  // every file of the pack
};

ImportReport import_gguf(const ImportOptions& options);

}  // namespace xs::gguf_import
