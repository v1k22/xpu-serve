#pragma once

// Low-rank changes to a model's weights (a LoRA adapter), read as if added to them: where a program reads an adapted
// weight, the change's product is added beside it. The weights themselves stay as stored (quantized, say).

#include <map>
#include <set>
#include <string>

#include "graph/graph.h"

namespace xs::graph {

// The change to one weight, its A and B weights by name, times `scale`:
//   a matmul's weight W [out, in]:          W + scale B A,          A [r, in],     B [out, r]
//   an expert bank W [E, out, in]:          W[e] + scale B[e] A[e], A [E, r, in],  B [E, out, r]
//   an embedding table W [vocab, width]:    W + scale A B^T,        A [vocab, r],  B [width, r]  (gathered from)
struct LowRank {
    std::string a, b;
    float scale = 1.0f;
};
using LowRanks = std::map<std::string, LowRank>;  // by the adapted weight's name

// `graph` with every MatMul, ExpertMatMul (the bank) and Gather (the table) of an adapted weight followed by its
// change, A and B read from `store`. Refuses by name another op reading an adapted weight and shapes that do not fit.
// `used` gets the names of the adapted weights the graph reads.
Graph with_low_rank(const Graph& graph, const WeightStore& store, const LowRanks& changes, std::set<std::string>* used);

}  // namespace xs::graph
