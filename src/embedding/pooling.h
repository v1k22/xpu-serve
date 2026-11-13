#pragma once

// A sentence embedding from an encoder's final hidden states, as sentence-transformers' Pooling and Normalize modules
// make it from one unpadded sequence (an attention mask of ones).

#include <vector>

#include "models/embedding.h"
#include "tensor/tensor.h"

namespace xs {

// `hidden` [T, D] f32 -> [D]: the mean of every row ([CLS] and [SEP] included) or row 0, summed in double; with
// `normalize`, then divided by its L2 norm (at least 1e-12, as torch's F.normalize).
std::vector<float> pool(const HostTensor& hidden, Pooling pooling, bool normalize);

}  // namespace xs
