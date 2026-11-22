#include "graph/elementwise_groups.h"

#include <algorithm>
#include <map>
#include <optional>

namespace xs::graph {

bool elementwise_op(OpKind kind) {
    switch (kind) {
        case OpKind::Add:
        case OpKind::Mul:
        case OpKind::Div:
        case OpKind::Scale:
        case OpKind::Tanh:
        case OpKind::GeluTanh:
        case OpKind::Gelu:
        case OpKind::Softcap:
        case OpKind::Sigmoid:
        case OpKind::Silu:
        case OpKind::Softplus:
        case OpKind::Clamp:
            return true;
        default:
            return false;
    }
}

namespace {

bool is_static(int64_t d) {
    return d != kT && d != kL;
}

bool binary(OpKind kind) {
    return kind == OpKind::Add || kind == OpKind::Mul || kind == OpKind::Div;
}

// Leading dims of 1 dropped.
Dims trimmed(const Dims& d) {
    size_t lead = 0;
    while (lead + 1 < d.size() && d[lead] == 1) ++lead;
    return Dims(d.begin() + static_cast<std::ptrdiff_t>(lead), d.end());
}

class Planner {
public:
    Planner(const Graph& g, const std::vector<int>& readers, const GroupRules& rules)
        : g_(g), readers_(readers), rules_(rules), claimed_(g.nodes().size(), 0) {}

    std::vector<ElementwiseGroup> run() {
        std::vector<ElementwiseGroup> groups;
        for (size_t i = g_.nodes().size(); i-- > 0;) {
            if (claimed_[i] != 0 || !op(i)) continue;
            std::optional<ElementwiseGroup> group = grow(static_cast<int32_t>(i));
            if (!group || group->folded.empty()) continue;
            for (const int32_t f : group->folded) claimed_[static_cast<size_t>(f)] = 1;
            claimed_[i] = 1;
            groups.push_back(std::move(*group));
        }
        return groups;
    }

private:
    bool op(size_t i) const {
        const Node& n = g_.nodes()[i];
        return elementwise_op(n.kind) && n.outputs.size() == 1 && n.inputs.size() == (binary(n.kind) ? 2u : 1u) &&
               rules_.free(i);
    }

    // The group rooted at `root`, or nothing when it cannot be one (a leaf it cannot read, past the limits).
    std::optional<ElementwiseGroup> grow(int32_t root) {
        group_ = ElementwiseGroup{};
        group_.root = root;
        out_ = g_.nodes()[static_cast<size_t>(root)].outputs[0].shape;
        reads_.clear();
        if (!emit(root)) return std::nullopt;
        keep_shared_slices();
        return std::move(group_);
    }

    // A slice (and the reshapes it is read through) that a node outside the group reads too runs on its own: the
    // leaf reads its output.
    void keep_shared_slices() {
        for (GroupLeaf& leaf : group_.leaves) {
            if (leaf.map != LeafMap::Slice) continue;
            bool inside = reads_[leaf.operand.node] == readers_[static_cast<size_t>(leaf.operand.node)];
            for (Value v = leaf.operand; g_.node(v).kind == OpKind::Reshape;) {
                v = g_.node(v).inputs[0];
                inside = inside && readers_[static_cast<size_t>(v.node)] == 1;
            }
            if (inside) continue;
            group_.folded.erase(std::find(group_.folded.begin(), group_.folded.end(), leaf.slice));
            leaf.value = Value{leaf.slice, 0};
            leaf.map = LeafMap::Same;
            leaf.slice = -1;
        }
    }

    // Node i's op after its operands; false when the group cannot hold them.
    bool emit(int32_t i) {
        const Node& n = g_.nodes()[static_cast<size_t>(i)];
        int16_t regs[2] = {0, 0};
        for (size_t k = 0; k < n.inputs.size(); ++k) {
            const std::optional<int16_t> r = operand(n.inputs[k]);
            if (!r) return false;
            regs[k] = *r;
        }
        if (group_.ops.size() >= static_cast<size_t>(rules_.max_ops)) return false;
        group_.ops.push_back({i, regs[0], regs[1]});
        return true;
    }

    // Operand v as an op of the group (its index), or a leaf (-1 - its index).
    std::optional<int16_t> operand(Value v) {
        const auto u = static_cast<size_t>(v.node);
        if (v.index == 0 && claimed_[u] == 0 && readers_[u] == 1 && op(u) && g_.type(v).shape == out_) {
            if (!emit(v.node)) return std::nullopt;
            group_.folded.push_back(v.node);
            return static_cast<int16_t>(group_.ops.size() - 1);
        }
        const std::optional<GroupLeaf> leaf = leaf_of(v);
        if (!leaf || !rules_.readable(leaf->value)) return std::nullopt;
        ++reads_[v.node];
        for (size_t j = 0; j < group_.leaves.size(); ++j) {
            const GroupLeaf& l = group_.leaves[j];
            if (l.value == leaf->value && l.map == leaf->map && l.slice == leaf->slice &&
                g_.type(l.operand).shape == g_.type(leaf->operand).shape) {
                return static_cast<int16_t>(-1 - static_cast<int>(j));
            }
        }
        if (group_.leaves.size() >= static_cast<size_t>(rules_.max_leaves)) return std::nullopt;
        if (leaf->slice >= 0) group_.folded.push_back(leaf->slice);
        group_.leaves.push_back(*leaf);
        return static_cast<int16_t>(-static_cast<int>(group_.leaves.size()));
    }

    // A reshape is a view of its input's memory.
    Value memory(Value v) const {
        while (g_.node(v).kind == OpKind::Reshape) v = g_.node(v).inputs[0];
        return v;
    }

    std::optional<GroupLeaf> leaf_of(Value v) const {
        const Dims& s = g_.type(v).shape;
        GroupLeaf leaf;
        leaf.operand = v;
        leaf.value = memory(v);
        if (s == out_) {
            const Node& src = g_.node(leaf.value);
            const auto u = static_cast<size_t>(leaf.value.node);
            if (src.kind == OpKind::Slice && claimed_[u] == 0 && readers_[u] == 1 && rules_.free(u)) {
                leaf.map = LeafMap::Slice;
                leaf.slice = leaf.value.node;
                leaf.value = memory(src.inputs[0]);
            }
            return leaf;
        }
        // Broadcasts: the operand's dims are static where they differ from the output's.
        const Dims t = trimmed(s);
        if (t.size() <= out_.size() && std::all_of(t.begin(), t.end(), is_static) &&
            std::equal(t.begin(), t.end(), out_.end() - static_cast<std::ptrdiff_t>(t.size()))) {
            leaf.map = LeafMap::Period;
            return leaf;
        }
        if (s.size() == out_.size()) {
            size_t j = s.size();
            while (j > 0 && s[j - 1] == 1 && is_static(out_[j - 1])) --j;
            if (j < s.size() && std::equal(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(j), out_.begin())) {
                leaf.map = LeafMap::Block;
                return leaf;
            }
        }
        return std::nullopt;
    }

    const Graph& g_;
    const std::vector<int>& readers_;
    const GroupRules& rules_;
    std::vector<uint8_t> claimed_;
    std::map<int32_t, int> reads_;  // how often the group reads each node it reads as a leaf
    ElementwiseGroup group_;
    Dims out_;
};

}  // namespace

std::vector<ElementwiseGroup> elementwise_groups(const Graph& graph, const std::vector<int>& readers,
                                                 const GroupRules& rules) {
    return Planner(graph, readers, rules).run();
}

}  // namespace xs::graph
