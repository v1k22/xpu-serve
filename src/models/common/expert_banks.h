#pragma once

// A mixture of experts stored expert by expert (experts.<e>.gate_proj.weight, .up_proj.weight, .down_proj.weight: the
// original Qwen3-MoE checkpoints, Qwen3-Omni's, the FP8 releases), shown as the banks the model code reads.

#include <cstdint>
#include <span>
#include <string>

#include "io/safetensors.h"

namespace xs {

// For each prefix in `experts` ("<layer>.mlp.experts."): experts.gate_up_proj [count, 2 inter, hidden] (an expert's
// gate rows, then its up rows) and experts.down_proj [count, hidden, inter]. A layer whose banks are already fused (a
// pack, a transformers 5 save), or that has no experts.0.gate_proj.weight, is left as it is.
void join_expert_banks(SafetensorsWeights& weights, std::span<const std::string> experts, int64_t count, int64_t inter,
                       int64_t hidden);

}  // namespace xs
