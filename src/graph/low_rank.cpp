#include "graph/low_rank.h"

#include "util/error.h"

namespace xs::graph {
namespace {

// The adapted weight `v` names, if any.
const LowRank* change_of(const Graph& g, Value v, const LowRanks& changes, std::string* name) {
    const Node& n = g.node(v);
    if (n.kind != OpKind::Weight) return nullptr;
    *name = std::get<WeightAttrs>(n.attrs).name;
    const auto it = changes.find(*name);
    return it == changes.end() ? nullptr : &it->second;
}

void check_shapes(const Graph& g, const std::string& name, Value w, Value a, Value b, bool table) {
    const Dims& ws = g.type(w).shape;
    const Dims& as = g.type(a).shape;
    const Dims& bs = g.type(b).shape;
    const auto fits = [&] {
        if (table) return ws.size() == 2 && as.size() == 2 && bs.size() == 2 && as[0] == ws[0] && bs[0] == ws[1] &&
                          as[1] == bs[1];
        const size_t rank = ws.size();
        if ((rank != 2 && rank != 3) || as.size() != rank || bs.size() != rank) return false;
        if (rank == 3 && (as[0] != ws[0] || bs[0] != ws[0])) return false;
        return as.back() == ws.back() && bs[rank - 2] == ws[rank - 2] && bs.back() == as[rank - 2];
    };
    XS_CHECK(fits(), "adapter: the change to ", name, " ", dims_to_string(ws), " has A ", dims_to_string(as),
             " and B ", dims_to_string(bs), ", which do not fit it");
}

}  // namespace

Graph with_low_rank(const Graph& graph, const WeightStore& store, const LowRanks& changes, std::set<std::string>* used) {
    return graph.rewritten([&](Graph& g, const Node& n, Value value) {
        std::string name;
        for (size_t k = 0; k < n.inputs.size(); ++k) {
            if (change_of(graph, n.inputs[k], changes, &name) == nullptr) continue;
            const bool read = (n.kind == OpKind::MatMul && k == 1) || (n.kind == OpKind::ExpertMatMul && k == 1) ||
                              (n.kind == OpKind::Gather && k == 0);
            XS_CHECK(read, "adapter: ", name, " is read by ", op_name(n.kind), " (", n.label,
                     "), which a low-rank change does not reach (only a matmul's weight, an expert bank or an embedding "
                     "table gathered from)");
        }
        if (n.kind != OpKind::MatMul && n.kind != OpKind::ExpertMatMul && n.kind != OpKind::Gather) return value;
        const bool table = n.kind == OpKind::Gather;
        const LowRank* c = change_of(graph, n.inputs[table ? 0 : 1], changes, &name);
        if (c == nullptr) return value;
        if (used != nullptr) used->insert(name);
        const std::vector<Value> in = g.node(value).inputs;  // the copy's (g's nodes move as ops are added)
        const Value a = g.weight(store, c->a);
        const Value b = g.weight(store, c->b);
        check_shapes(g, name, in[table ? 0 : 1], a, b, table);
        Value delta;
        if (n.kind == OpKind::MatMul) {
            delta = g.matmul(g.matmul(in[0], a), b);
        } else if (n.kind == OpKind::ExpertMatMul) {
            delta = g.expert_matmul(g.expert_matmul(in[0], a, in[2]), b, in[2]);
        } else {
            delta = g.matmul(g.gather(a, in[1]), b);
        }
        return g.add(value, c->scale == 1.0f ? delta : g.scale(delta, c->scale));
    });
}

}  // namespace xs::graph
