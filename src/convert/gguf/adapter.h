#pragma once

// A LoRA adapter as llama.cpp writes it (general.type "adapter", adapter.type "lora"): for each weight it changes,
// <tensor>.lora_a and <tensor>.lora_b under the file names of the model's architecture, and adapter.lora_alpha. Read
// as the changes graph/low_rank.h adds, under the names the model code reads.

#include <filesystem>
#include <memory>

#include "graph/low_rank.h"
#include "io/host_weights.h"

namespace xs::gguf_import {

struct Adapter {
    std::string architecture;              // general.architecture
    std::unique_ptr<HostWeights> tensors;  // each change's A and B as f32: "<weight>.lora_a", "<weight>.lora_b"
    graph::LowRanks changes;               // by the weight's name; A and B as named in `tensors`
};

// The changes of `file`, each scaled by `scale` times alpha / rank (llama.cpp's scale; alpha 0: `scale` alone).
// Refuses by name another kind of file, a tensor without its pair, and one the architecture has no whole weight for.
Adapter read_lora(const std::filesystem::path& file, float scale = 1.0f);

}  // namespace xs::gguf_import
