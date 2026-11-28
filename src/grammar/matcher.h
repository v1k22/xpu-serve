#pragma once

// Matching a grammar a byte at a time, and the tokens of a vocabulary it allows next (docs/decisions/0033).
//
// A matcher's configuration is the set of stacks a pushdown recognizer of the grammar may be in: each stack the item
// of an alternative it expects next, over the places to go on from when that alternative ends. Stacks are interned
// (equal ones are one frame), so a configuration is a sorted set of frames, interned as a state, and each state's
// step by a byte is computed once (a DFA built lazily over the configurations reached). A rule's last item takes no
// frame of its own (a tail position), so a repetition loops through one state instead of growing a stack.
//
// The tokens a state allows come from one walk of the vocabulary's trie: a subtree is skipped at the first byte the
// state does not take, and each step of the walk is a lookup in the transition table; marker tokens come from the
// state's stacks that expect one. The tokens of a state whose
// walk was long (the inside of a string: nearly every token) are kept, so a state that repeats costs nothing again.

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "grammar/grammar.h"
#include "grammar/vocabulary.h"

namespace xs::grammar {

// The states of one grammar over one vocabulary, found as matchers reach them. Not thread-safe: the matchers of one
// generation share it.
class Automaton {
public:
    using State = int32_t;
    static constexpr State kDead = 0;  // no stack left: no byte is allowed, and the grammar cannot end here

    Automaton(std::shared_ptr<const Grammar> grammar, std::shared_ptr<const Vocabulary> vocabulary);

    State start() const { return start_; }
    State step(State s, uint8_t byte) {
        const int32_t next = next_[static_cast<size_t>(s) * 256 + byte];
        return next >= 0 ? next : compute_step(s, byte);
    }
    State step(State s, std::string_view bytes);
    // The token's bytes from `s`, or for a marker token the places `s` expects it (Grammar::Item::kToken); dead for a
    // token the grammar does not take there.
    State step_token(State s, int32_t token);
    // The root has ended at `s`.
    bool accepting(State s) const { return states_[s].accepting; }
    // The tokens whose bytes `s` takes (each a prefix of something the grammar matches). The reference holds until the
    // next call.
    const TokenSet& allowed(State s);

    const Vocabulary& vocabulary() const { return *vocabulary_; }
    size_t states() const { return states_.size(); }

private:
    struct Frame {
        uint32_t pos;    // an item in the grammar's code: what this stack matches next
        int32_t parent;  // the frame to go on from once this alternative ends; -1: the root's end
    };
    struct StateInfo {
        uint32_t first, last;  // its top frames: frame ids in tops_[first, last)
        bool accepting;
    };
    struct Closure {
        std::vector<int32_t> tops;
        bool accepting = false;
        std::vector<uint64_t> seen;
    };

    int32_t frame(uint32_t pos, int32_t parent);
    // Everything matching from `pos` on with `parent` below can expect next: the frames of byte items reached through
    // rules (pushed) and alternatives' ends (popped).
    void expand(uint32_t pos, int32_t parent, Closure& out);
    State intern(Closure& c);
    State compute_step(State s, uint8_t byte);
    State step_marker(State s, int32_t token);

    std::shared_ptr<const Grammar> grammar_;
    std::shared_ptr<const Vocabulary> vocabulary_;
    std::vector<Frame> frames_;
    std::unordered_map<uint64_t, int32_t> frame_ids_;
    std::vector<int32_t> tops_;
    std::vector<StateInfo> states_;
    std::unordered_map<std::string, State> state_ids_;
    std::vector<int32_t> next_;  // by state and byte; -1: not computed yet
    std::unordered_map<uint64_t, State> marker_next_;  // by state and marker token
    State start_ = kDead;

    std::unordered_map<State, TokenSet> kept_;  // the tokens of states whose walk was long
    std::deque<State> kept_order_;
    TokenSet scratch_;
    std::vector<State> walk_;  // the state at each depth of the walk
};

// Where the bytes so far left a grammar's matcher. Cheap to copy: a copy steps on without changing the original.
class Matcher {
public:
    explicit Matcher(std::shared_ptr<Automaton> automaton)
        : automaton_(std::move(automaton)), state_(automaton_->start()) {}

    // False (and dead from then on) when the grammar does not take the bytes.
    bool accept_token(int32_t token) {
        state_ = automaton_->step_token(state_, token);
        return !dead();
    }
    bool accept_bytes(std::string_view bytes) {
        state_ = automaton_->step(state_, bytes);
        return !dead();
    }
    bool dead() const { return state_ == Automaton::kDead; }
    // The grammar's root has ended: the output may stop here.
    bool complete() const { return automaton_->accepting(state_); }
    // The tokens that may come next. The reference holds until the automaton's next allowed().
    const TokenSet& allowed() const { return automaton_->allowed(state_); }
    bool allows(int32_t token) const { return automaton_->step_token(state_, token) != Automaton::kDead; }

private:
    std::shared_ptr<Automaton> automaton_;
    Automaton::State state_;
};

}  // namespace xs::grammar
