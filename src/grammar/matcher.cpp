#include "grammar/matcher.h"

#include <algorithm>

#include "util/error.h"

namespace xs::grammar {
namespace {

// A walk of the trie this long costs more than keeping its tokens (a few tens of KiB) for the state to come again.
constexpr size_t kKeepVisits = 4096;
constexpr size_t kKeepStates = 64;

}  // namespace

Automaton::Automaton(std::shared_ptr<const Grammar> grammar, std::shared_ptr<const Vocabulary> vocabulary)
    : grammar_(std::move(grammar)), vocabulary_(std::move(vocabulary)) {
    XS_CHECK(grammar_ && grammar_->finished() && vocabulary_, "grammar: an automaton needs a finished grammar");
    states_.push_back({0, 0, false});
    next_.assign(256, kDead);  // the dead state stays dead
    Closure c;
    for (const uint32_t a : grammar_->rules()[grammar_->root()].alternatives) expand(a, -1, c);
    start_ = intern(c);
    scratch_ = TokenSet(vocabulary_->size());
    walk_.resize(static_cast<size_t>(vocabulary_->trie().max_depth) + 1);
}

int32_t Automaton::frame(uint32_t pos, int32_t parent) {
    const uint64_t key = (uint64_t{pos} << 32) | static_cast<uint32_t>(parent + 1);
    const auto [it, added] = frame_ids_.emplace(key, static_cast<int32_t>(frames_.size()));
    if (added) frames_.push_back({pos, parent});
    return it->second;
}

void Automaton::expand(uint32_t pos, int32_t parent, Closure& out) {
    const uint64_t key = (uint64_t{pos} << 32) | static_cast<uint32_t>(parent + 1);
    if (std::find(out.seen.begin(), out.seen.end(), key) != out.seen.end()) return;
    out.seen.push_back(key);
    const std::vector<Grammar::Item>& code = grammar_->code();
    const Grammar::Item item = code[pos];
    switch (item.kind) {
        case Grammar::Item::kEnd:
            if (parent < 0) {
                out.accepting = true;
            } else {
                const Frame f = frames_[static_cast<size_t>(parent)];
                expand(f.pos, f.parent, out);
            }
            return;
        case Grammar::Item::kBytes:
            out.tops.push_back(frame(pos, parent));
            return;
        case Grammar::Item::kRule: {
            const int32_t cont = code[pos + 1].kind == Grammar::Item::kEnd ? parent : frame(pos + 1, parent);
            for (const uint32_t a : grammar_->rules()[item.index].alternatives) expand(a, cont, out);
            return;
        }
    }
}

Automaton::State Automaton::intern(Closure& c) {
    std::sort(c.tops.begin(), c.tops.end());
    c.tops.erase(std::unique(c.tops.begin(), c.tops.end()), c.tops.end());
    if (c.tops.empty() && !c.accepting) return kDead;
    std::string key(1, c.accepting ? '\1' : '\0');
    key.append(reinterpret_cast<const char*>(c.tops.data()), c.tops.size() * sizeof(int32_t));
    const auto [it, added] = state_ids_.emplace(std::move(key), static_cast<State>(states_.size()));
    if (!added) return it->second;
    const auto first = static_cast<uint32_t>(tops_.size());
    tops_.insert(tops_.end(), c.tops.begin(), c.tops.end());
    states_.push_back({first, static_cast<uint32_t>(tops_.size()), c.accepting});
    next_.resize(next_.size() + 256, -1);
    return it->second;
}

Automaton::State Automaton::compute_step(State s, uint8_t byte) {
    Closure c;
    const StateInfo info = states_[static_cast<size_t>(s)];
    for (uint32_t i = info.first; i < info.last; ++i) {
        const Frame f = frames_[static_cast<size_t>(tops_[i])];
        if (grammar_->sets()[grammar_->code()[f.pos].index].has(byte)) expand(f.pos + 1, f.parent, c);
    }
    const State t = intern(c);
    next_[static_cast<size_t>(s) * 256 + byte] = t;
    return t;
}

Automaton::State Automaton::step(State s, std::string_view bytes) {
    for (const char b : bytes) {
        if (s == kDead) break;
        s = step(s, static_cast<uint8_t>(b));
    }
    return s;
}

Automaton::State Automaton::step_token(State s, int32_t token) {
    if (token < 0 || token >= vocabulary_->size() || !vocabulary_->writable(token)) return kDead;
    return step(s, vocabulary_->bytes(token));
}

const TokenSet& Automaton::allowed(State s) {
    if (const auto it = kept_.find(s); it != kept_.end()) return it->second;
    scratch_.clear();
    if (s == kDead) return scratch_;
    const Vocabulary::Trie& t = vocabulary_->trie();
    const uint8_t* byte = t.byte.data();
    const uint16_t* depth = t.depth.data();
    const uint32_t* end = t.end.data();
    const int32_t* tokens = t.token.data();
    const size_t nodes = t.nodes();
    State* walk = walk_.data();
    uint64_t* allowed = scratch_.data();
    const int32_t* next = next_.data();  // read again after a step had to be computed: the table may have grown
    uint64_t none = 0;                   // where a node that ends no token writes: no branch on it
    walk[0] = s;
    size_t visited = 0;
    for (size_t i = 0; i < nodes; ++visited) {
        const State from = walk[depth[i] - 1u];
        State to = next[static_cast<size_t>(from) * 256 + byte[i]];
        if (to < 0) {
            to = compute_step(from, byte[i]);
            next = next_.data();
        }
        if (to == kDead) {
            i = end[i];
            continue;
        }
        walk[depth[i]] = to;
        const int32_t token = tokens[i];
        uint64_t* word = token >= 0 ? allowed + (static_cast<uint32_t>(token) >> 6) : &none;
        *word |= uint64_t{1} << (token & 63);
        ++i;
    }
    for (const auto& [token, same] : t.same_bytes) {
        if (scratch_.has(token)) scratch_.set(same);
    }
    if (visited < kKeepVisits) return scratch_;
    if (kept_order_.size() == kKeepStates) {
        kept_.erase(kept_order_.front());
        kept_order_.pop_front();
    }
    kept_order_.push_back(s);
    return kept_.emplace(s, scratch_).first->second;
}

}  // namespace xs::grammar
