#pragma once

// The Qwen3.5 decoder as llama.cpp's converter writes it (Qwen3_5TextModel, Qwen3_5MoeTextModel), shared by the GGUF
// architectures whose files hold one: qwen35 and qwen35moe, and clef, which writes qwen35's keys under clef.* and adds
// a decision head.

#include <cstdint>
#include <set>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "convert/gguf/layout.h"

namespace xs::gguf_import {

// The MLP of every decoder layer, and of an MTP layer: one gated MLP (feed_forward_length; ffn_gate, ffn_up, ffn_down),
// or a mixture of experts beside a gated shared expert (expert_count, expert_used_count, expert_feed_forward_length,
// expert_shared_feed_forward_length; the router ffn_gate_inp, the banks ffn_gate_exps, ffn_up_exps and ffn_down_exps,
// the shared expert's ffn_*_shexp). The architecture names which; a file never holds both.
enum class Qwen35Mlp : uint8_t { Dense, Experts };

struct Qwen35Decoder {
    // config.json: Qwen3_5ForCausalLM (model_type qwen3_5_text), or with experts Qwen3_5MoeForCausalLM
    // (qwen3_5_moe_text).
    nlohmann::ordered_json config;
    Layout layout;                                 // the decoder's tensors, and an MTP head's
    std::vector<const gguf::TensorInfo*> tensors;  // the file's, and the head's block of an MTP file
    std::vector<const gguf::TensorInfo*> copies;   // an MTP file's copies of the model's tensors (ImportPlan::copies)
};

// The decoder of `f`, its metadata under "<arch>." and its layers' MLP of kind `mlp`, with its MTP head from `mtp` (a
// file of its own) when one is given. Refuses by name, as "<arch>: ...", a key of the namespace that neither the
// decoder nor `keys` (the architecture's own, without the namespace) names, and what the decoder does not implement.
// An MTP head needs "nextn_predict_layers" among `keys`.
Qwen35Decoder read_qwen35_decoder(const gguf::File& f, std::string_view arch, Qwen35Mlp mlp,
                                  const std::set<std::string_view>& keys, const gguf::File* mtp = nullptr);

}  // namespace xs::gguf_import
