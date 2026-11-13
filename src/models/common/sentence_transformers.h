#pragma once

// What a sentence-transformers checkpoint lists around its transformer (modules.json): the Pooling module's mode and
// dimension, whether a Normalize module follows, and the longest input its Transformer module reads
// (sentence_bert_config.json). The pipeline the embedding models here run is Transformer, Pooling (mean or CLS),
// then an optional Normalize, as sentence-transformers 6 reads those files (its legacy pooling_mode_* flags included);
// any other module, mode or setting that changes an embedding is refused by name.

#include <cstdint>
#include <filesystem>
#include <optional>

#include "models/embedding.h"

namespace xs::st {

struct Pipeline {
    Pooling pooling = Pooling::Mean;
    int64_t dimension = 0;  // the Pooling module's embedding dimension
    bool normalize = false;
    std::optional<int64_t> max_seq_length;  // sentence_bert_config.json's, when it gives one
};

// True when `dir` holds a modules.json that lists a Pooling module.
bool has_pooling(const std::filesystem::path& dir);

// Reads and checks the pipeline. Throws xs::Error naming what is not supported.
Pipeline read_pipeline(const std::filesystem::path& dir);

}  // namespace xs::st
