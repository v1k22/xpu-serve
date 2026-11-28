#pragma once

// response_format's json_object and json_schema as the engine's output constraint (engine/constraint.h): the answer
// held to a grammar over the model's tokens, from where the dialect's answer starts (chat::answer_start).

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/constraint.h"
#include "grammar/grammar.h"
#include "grammar/matcher.h"
#include "grammar/vocabulary.h"

namespace xs::tok {
class Tokenizer;
}

namespace xs::server {

// The model's tokens as a grammar reads them: each token's bytes, its added tokens markers. Built once per model.
std::shared_ptr<const grammar::Vocabulary> grammar_vocabulary(const tok::Tokenizer& tokenizer);

// The automata of the grammars requests are held to, by a name of each grammar (OutputFormat::key, a tool grammar's
// description): requests held to the same grammar share its states and the token masks those keep, which the first
// such request pays for (decision 0033 had each request walk the vocabulary's trie again). Bounded: the least recently
// used is let go. Constraints start on the engine thread, which is the one that reads them.
class AutomatonCache {
public:
    explicit AutomatonCache(size_t most = 32) : most_(most) {}
    // The automaton of `grammar` over `vocabulary`, kept under `key` (empty: a fresh one, not kept).
    std::shared_ptr<grammar::Automaton> get(const std::string& key, const std::shared_ptr<const grammar::Grammar>& grammar,
                                            const std::shared_ptr<const grammar::Vocabulary>& vocabulary);

private:
    std::mutex mutex_;
    std::list<std::pair<std::string, std::shared_ptr<grammar::Automaton>>> kept_;  // most recently used first
    size_t most_;
};

// A generation's output: free until the model has written `reasoning_end` (empty: from the start; the engine keeps the
// stop tokens out meanwhile, since the answer is still to come), then the tokens of `opening`, then whatever the
// grammar allows, token by token, until it is complete.
class GrammarConstraint final : public OutputConstraint {
public:
    // `automata` and `key`: where the grammar's automaton is kept, and its name there (none: one a generation).
    GrammarConstraint(std::shared_ptr<const grammar::Grammar> grammar,
                      std::shared_ptr<const grammar::Vocabulary> vocabulary, std::string reasoning_end,
                      std::vector<int32_t> opening, std::shared_ptr<AutomatonCache> automata = nullptr,
                      std::string key = {});
    std::unique_ptr<TokenConstraint> start() const override;

    // What each generation's constraint reads, shared by them.
    struct Spec {
        std::shared_ptr<const grammar::Grammar> grammar;
        std::shared_ptr<const grammar::Vocabulary> vocabulary;
        std::string reasoning_end;
        std::vector<int32_t> opening;
        std::shared_ptr<AutomatonCache> automata;
        std::string key;
    };

private:
    std::shared_ptr<const Spec> spec_;
};

// A generation free but for its calls: once it writes the marker `opener`, what follows is held to `grammar` until the
// grammar is complete (a call to its closer), then it is free again (tool_choice "auto" with strict tools).
class OpenerConstraint final : public OutputConstraint {
public:
    OpenerConstraint(int32_t opener, std::shared_ptr<const grammar::Grammar> grammar,
                     std::shared_ptr<const grammar::Vocabulary> vocabulary,
                     std::shared_ptr<AutomatonCache> automata = nullptr, std::string key = {});
    std::unique_ptr<TokenConstraint> start() const override;

private:
    int32_t opener_;
    std::shared_ptr<const grammar::Grammar> grammar_;
    std::shared_ptr<const grammar::Vocabulary> vocabulary_;
    std::shared_ptr<AutomatonCache> automata_;
    std::string key_;
};

}  // namespace xs::server
