#include "grammar/grammar.h"

#include <algorithm>
#include <functional>

namespace xs::grammar {

struct Expr::Node {
    enum Kind { kBytes, kLiteral, kRule, kSeq, kAlt, kRepeat, kToken };
    Kind kind = kSeq;
    ByteSet set;
    std::string text;
    RuleId rule = 0;
    int32_t token = 0;
    std::vector<Expr> parts;  // a seq's, an alt's, or the one a repeat repeats
    int min = 0, max = 0;
};

namespace {

Expr make(Expr::Node node) { return Expr(std::make_shared<const Expr::Node>(std::move(node))); }

}  // namespace

ByteSet ByteSet::of(std::string_view bytes) {
    ByteSet s;
    for (const char c : bytes) s.add(static_cast<uint8_t>(c), static_cast<uint8_t>(c));
    return s;
}

ByteSet ByteSet::range(uint8_t first, uint8_t last) { return ByteSet().add(first, last); }

ByteSet& ByteSet::add(uint8_t first, uint8_t last) {
    for (unsigned b = first; b <= last; ++b) bits_[b >> 6] |= uint64_t{1} << (b & 63);
    return *this;
}

ByteSet& ByteSet::add(const ByteSet& other) {
    for (size_t i = 0; i < bits_.size(); ++i) bits_[i] |= other.bits_[i];
    return *this;
}

Expr bytes(const ByteSet& set) { return make({.kind = Expr::Node::kBytes, .set = set}); }
Expr literal(std::string_view text) { return make({.kind = Expr::Node::kLiteral, .text = std::string(text)}); }
Expr rule(RuleId id) { return make({.kind = Expr::Node::kRule, .rule = id}); }

Expr token(int32_t id) {
    XS_CHECK(id >= 0, "grammar: token ", id);
    return make({.kind = Expr::Node::kToken, .token = id});
}
Expr seq(std::vector<Expr> parts) { return make({.kind = Expr::Node::kSeq, .parts = std::move(parts)}); }

Expr alt(std::vector<Expr> options) {
    XS_CHECK(!options.empty(), "grammar: an alternative of no options");
    return make({.kind = Expr::Node::kAlt, .parts = std::move(options)});
}

Expr repeat(Expr e, int min, int max) {
    XS_CHECK(min >= 0 && (max == kUnbounded || max >= min), "grammar: a repeat of ", min, " to ", max);
    return make({.kind = Expr::Node::kRepeat, .parts = {std::move(e)}, .min = min, .max = max});
}

Expr text_without(Grammar& g, std::string_view stop, const std::string& name) {
    XS_CHECK(!stop.empty(), "grammar: text without an empty string");
    // The Knuth-Morris-Pratt automaton of `stop`: state j, the longest prefix of it the text ends with. A rule a state,
    // each byte that does not complete `stop` going to its next state's rule (in a tail position: no stack grows).
    const size_t m = stop.size();
    std::vector<size_t> fail(m, 0);
    for (size_t i = 1, k = 0; i < m; ++i) {
        while (k > 0 && stop[i] != stop[k]) k = fail[k - 1];
        if (stop[i] == stop[k]) ++k;
        fail[i] = k;
    }
    const auto next = [&](size_t j, char b) {
        while (j > 0 && stop[j] != b) j = fail[j - 1];
        return stop[j] == b ? j + 1 : 0;
    };
    std::vector<RuleId> state(m);
    for (size_t j = 0; j < m; ++j) state[j] = g.add_rule(name);
    for (size_t j = 0; j < m; ++j) {
        std::vector<ByteSet> to(m);  // the bytes that lead to each state
        for (unsigned b = 0; b < 256; ++b) {
            const size_t k = next(j, static_cast<char>(b));
            if (k < m) to[k].add(static_cast<uint8_t>(b), static_cast<uint8_t>(b));
        }
        std::vector<Expr> options{seq({})};
        for (size_t k = 0; k < m; ++k) {
            if (!to[k].empty()) options.push_back(seq({bytes(to[k]), rule(state[k])}));
        }
        g.define(state[j], alt(std::move(options)));
    }
    return rule(state[0]);
}

RuleId Grammar::add_rule(std::string name) {
    XS_CHECK(!finished_, "grammar: a rule added after finish()");
    rules_.push_back({std::move(name), {}, false});
    return static_cast<RuleId>(rules_.size() - 1);
}

RuleId Grammar::embed(const Grammar& other) {
    XS_CHECK(!finished_ && other.finished(), "grammar: embedding needs a finished grammar into an open one");
    const auto offset = static_cast<RuleId>(rules_.size());
    for (const Rule& r : other.rules()) rules_.push_back({r.name, {}, true});
    for (RuleId r = 0; r < other.rules().size(); ++r) {
        for (uint32_t at : other.rules()[r].alternatives) {
            std::vector<Item> items;
            for (; other.code()[at].kind != Item::kEnd; ++at) {
                Item it = other.code()[at];
                if (it.kind == Item::kBytes) it.index = set_index(other.sets()[it.index]);
                if (it.kind == Item::kRule) it.index += offset;
                items.push_back(it);
            }
            add_alternative(offset + r, std::move(items));
        }
    }
    return offset + other.root();
}

void Grammar::define(RuleId r, const Expr& e) {
    XS_CHECK(!finished_ && r < rules_.size() && !rules_[r].defined, "grammar: rule ", r, " defined twice");
    rules_[r].defined = true;
    helpers_.clear();
    const std::string name = rules_[r].name;
    if (e.node().kind == Expr::Node::kAlt) {
        for (const Expr& option : e.node().parts) add_alternative(r, lower(option.node(), name));
    } else {
        add_alternative(r, lower(e.node(), name));
    }
    helpers_.clear();  // the nodes may not outlive this call
}

void Grammar::add_alternative(RuleId r, std::vector<Item> items) {
    rules_[r].alternatives.push_back(static_cast<uint32_t>(code_.size()));
    code_.insert(code_.end(), items.begin(), items.end());
    code_.push_back({Item::kEnd, 0});
}

uint32_t Grammar::set_index(const ByteSet& set) {
    const auto [it, added] = set_ids_.emplace(set, static_cast<uint32_t>(sets_.size()));
    if (added) sets_.push_back(set);
    return it->second;
}

std::vector<Grammar::Item> Grammar::lower(const Expr::Node& n, const std::string& name) {
    std::vector<Item> out;
    const auto append = [&out](const std::vector<Item>& items) { out.insert(out.end(), items.begin(), items.end()); };
    switch (n.kind) {
        case Expr::Node::kBytes:
            out.push_back({Item::kBytes, set_index(n.set)});
            break;
        case Expr::Node::kLiteral:
            for (const char c : n.text) out.push_back({Item::kBytes, set_index(ByteSet::of(std::string_view(&c, 1)))});
            break;
        case Expr::Node::kRule:
            out.push_back({Item::kRule, n.rule});
            break;
        case Expr::Node::kToken:
            out.push_back({Item::kToken, static_cast<uint32_t>(n.token)});
            break;
        case Expr::Node::kSeq:
            for (const Expr& part : n.parts) append(lower(part.node(), name));
            break;
        case Expr::Node::kAlt:
            if (n.parts.size() == 1) return lower(n.parts[0].node(), name);
            out.push_back({Item::kRule, helper(n, name)});
            break;
        case Expr::Node::kRepeat:
            for (int i = 0; i < n.min; ++i) append(lower(n.parts[0].node(), name));
            if (n.max == kUnbounded || n.max > n.min) out.push_back({Item::kRule, helper(n, name)});
            break;
    }
    return out;
}

// The rule of an alt's options, or of a repeat's optional tail: up to max - min more of it as a chain of rules
// (r_k ::= "" | x r_{k-1}), or any number as one rule that ends in itself (a tail position: no stack grows).
RuleId Grammar::helper(const Expr::Node& n, const std::string& name) {
    if (const auto it = helpers_.find(&n); it != helpers_.end()) return it->second;
    const RuleId h = add_rule(name);
    helpers_.emplace(&n, h);
    rules_[h].defined = true;
    if (n.kind == Expr::Node::kAlt) {
        for (const Expr& option : n.parts) add_alternative(h, lower(option.node(), name));
        return h;
    }
    const Expr::Node& x = n.parts[0].node();
    if (n.max == kUnbounded) {
        add_alternative(h, {});
        std::vector<Item> items = lower(x, name);
        items.push_back({Item::kRule, h});
        add_alternative(h, std::move(items));
        return h;
    }
    const int extra = n.max - n.min;
    RuleId inner = 0;
    for (int k = 1; k <= extra; ++k) {
        const RuleId r = k == extra ? h : add_rule(name);
        rules_[r].defined = true;
        add_alternative(r, {});
        std::vector<Item> items = lower(x, name);
        if (k > 1) items.push_back({Item::kRule, inner});
        add_alternative(r, std::move(items));
        inner = r;
    }
    return h;
}

void Grammar::finish(RuleId root) {
    XS_CHECK(!finished_ && root < rules_.size(), "grammar: finish() of rule ", root, " of ", rules_.size());
    for (const Rule& r : rules_) XS_CHECK(r.defined, "grammar: rule '", r.name, "' is never defined");
    const size_t n = rules_.size();

    // Which rules derive a finite string: some alternative's items all do (a byte of a non-empty set does).
    std::vector<bool> ends(n, false);
    const auto alternative_ends = [&](uint32_t at) {
        for (; code_[at].kind != Item::kEnd; ++at) {
            const Item& it = code_[at];
            if (it.kind == Item::kToken) continue;
            if (it.kind == Item::kBytes ? sets_[it.index].empty() : !ends[it.index]) return false;
        }
        return true;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (RuleId r = 0; r < n; ++r) {
            if (ends[r]) continue;
            ends[r] = std::any_of(rules_[r].alternatives.begin(), rules_[r].alternatives.end(), alternative_ends);
            changed = changed || ends[r];
        }
    }
    if (!ends[root]) {
        // Name the deepest such rule the root reaches: the one a schema path names most precisely.
        std::vector<bool> seen(n, false);
        std::vector<RuleId> todo{root};
        seen[root] = true;
        RuleId worst = root;
        while (!todo.empty()) {
            const RuleId r = todo.back();
            todo.pop_back();
            if (!ends[r] && rules_[r].name.size() > rules_[worst].name.size()) worst = r;
            for (const uint32_t start : rules_[r].alternatives) {
                for (uint32_t at = start; code_[at].kind != Item::kEnd; ++at) {
                    if (code_[at].kind == Item::kRule && !seen[code_[at].index]) {
                        seen[code_[at].index] = true;
                        todo.push_back(code_[at].index);
                    }
                }
            }
        }
        throw GrammarError(rules_[worst].name, "has no finite value: every way through it comes back to it");
    }
    for (Rule& r : rules_) std::erase_if(r.alternatives, [&](uint32_t at) { return !alternative_ends(at); });

    // Left recursion: a rule that reaches itself through the first items of its alternatives, past those that may
    // match nothing.
    std::vector<bool> nullable(n, false);
    const auto alternative_nullable = [&](uint32_t at) {
        for (; code_[at].kind != Item::kEnd; ++at) {
            if (code_[at].kind != Item::kRule || !nullable[code_[at].index]) return false;
        }
        return true;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (RuleId r = 0; r < n; ++r) {
            if (nullable[r]) continue;
            const std::vector<uint32_t>& alternatives = rules_[r].alternatives;
            nullable[r] = std::any_of(alternatives.begin(), alternatives.end(), alternative_nullable);
            changed = changed || nullable[r];
        }
    }
    enum class Mark : uint8_t { none, open, done };
    std::vector<Mark> mark(n, Mark::none);
    const std::function<void(RuleId)> visit = [&](RuleId r) {
        mark[r] = Mark::open;
        for (const uint32_t start : rules_[r].alternatives) {
            for (uint32_t at = start; code_[at].kind == Item::kRule; ++at) {
                const RuleId s = code_[at].index;
                if (mark[s] == Mark::open) {
                    throw GrammarError(rules_[s].name, "comes back to itself before any byte (left recursion)");
                }
                if (mark[s] == Mark::none) visit(s);
                if (!nullable[s]) break;
            }
        }
        mark[r] = Mark::done;
    };
    for (RuleId r = 0; r < n; ++r) {
        if (mark[r] == Mark::none) visit(r);
    }
    root_ = root;
    finished_ = true;
}

}  // namespace xs::grammar
