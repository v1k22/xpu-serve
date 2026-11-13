#pragma once

// The Qwen3.5 decoder as llama.cpp's converter writes it (Qwen3_5TextModel), shared by the GGUF architectures whose
// files hold one: qwen35, and clef, which writes the same keys under clef.* and adds a decision head.

#include <set>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/gguf/layout.h"

namespace xs::gguf_import {

struct Qwen35Decoder {
    nlohmann::ordered_json config;                 // config.json (Qwen3_5ForCausalLM, model_type qwen3_5_text)
    Layout layout;                                 // the decoder's tensors, and an MTP head's
    std::vector<const gguf::TensorInfo*> tensors;  // the file's, and the head's block of an MTP file
    std::vector<const gguf::TensorInfo*> copies;   // an MTP file's copies of the model's tensors (ImportPlan::copies)
};

// The decoder of `f`, its metadata under "<arch>.", with its MTP head from `mtp` (a file of its own) when one is given.
// Refuses by name, as "<arch>: ...", a key of the namespace that neither the decoder nor `keys` (the architecture's
// own, without the namespace) names, and what the decoder does not implement. An MTP head needs
// "nextn_predict_layers" among `keys`.
Qwen35Decoder read_qwen35_decoder(const gguf::File& f, std::string_view arch, const std::set<std::string_view>& keys,
                                  const gguf::File* mtp = nullptr);

}  // namespace xs::gguf_import
