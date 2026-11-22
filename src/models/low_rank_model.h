#pragma once

// A model whose programs read some of its weights with a low-rank change added (a LoRA adapter, graph/low_rank.h), as
// if merged into them: the weights stay as stored, and everything else of the model is the same.

#include <memory>

#include "graph/low_rank.h"
#include "models/model.h"

namespace xs {

// `model` with `changes`, whose A and B are in `tensors`. Refuses by name a change to a weight none of the model's
// programs reads, and one to a weight it reads on the host (Model::host_tables).
std::unique_ptr<Model> with_low_rank(std::unique_ptr<Model> model, std::unique_ptr<WeightStore> tensors,
                                     const graph::LowRanks& changes);

}  // namespace xs
