#pragma once

// Elementwise groups: trees of elementwise ops over one output shape that a backend can run as one pass, the ops'
// values in registers. A value read only by the next op of its tree is never written out; a slice read only by the
// tree is read in place; a value broadcast to the output's shape (repeating, or held across blocks) is read through
// its map. A group has at least two ops, or one op and a slice it reads in place.

#include <cstdint>
#include <functional>
#include <vector>

#include "graph/graph.h"

namespace xs::graph {

// How a group reads a leaf for each element of its output.
enum class LeafMap : uint8_t {
    Same,    // a value with the output's elements in its order
    Period,  // a value of the output's trailing dims, repeating
    Block,   // a value of the output's shape with trailing dims of 1, each element held across a block
    Slice,   // a slice along one axis of `value` (the Slice node, `slice`, runs inside the group)
};

struct GroupLeaf {
    Value value;         // the memory read: a reshape's input in its place
    LeafMap map = LeafMap::Same;
    Value operand;       // what the op reads (Period, Block: its shape gives the map's numbers)
    int32_t slice = -1;  // LeafMap::Slice
};

// An op of the group, node `node`'s: its operands are leaves (-1 - leaf) or earlier ops (their index); b is unused
// for a unary op.
struct GroupOp {
    int32_t node = -1;
    int16_t a = 0;
    int16_t b = 0;
};

struct ElementwiseGroup {
    int32_t root = -1;              // the node that runs the group and holds its output
    std::vector<GroupLeaf> leaves;  // distinct
    std::vector<GroupOp> ops;       // in order; the root's last
    std::vector<int32_t> folded;    // the other nodes the group runs (ops and slices)
};

struct GroupRules {
    int max_leaves = 8;
    int max_ops = 24;
    std::function<bool(size_t)> free;      // node i may run inside a group (no other fusion claimed it)
    std::function<bool(Value)> readable;   // value v can be a leaf (the backend reads its dtype)
};

// The groups of `graph`, roots last to first. `readers[i]`: the readers of node i's outputs (a graph output counts
// as more than one).
std::vector<ElementwiseGroup> elementwise_groups(const Graph& graph, const std::vector<int>& readers,
                                                 const GroupRules& rules);

// Whether an op of this kind can be an op of a group: Add, Mul, Div and the unary ops of graph.h.
bool elementwise_op(OpKind kind);

}  // namespace xs::graph
