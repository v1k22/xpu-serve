#pragma once

// response_format's json_object and json_schema as the engine's output constraint (engine/constraint.h): the answer
// held to a grammar over the model's tokens, from where the dialect's answer starts (chat::answer_start).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/constraint.h"
#include "grammar/grammar.h"
#include "grammar/vocabulary.h"

namespace xs::tok {
class Tokenizer;
}

namespace xs::server {

// The model's tokens as a grammar reads them: each token's bytes, its added tokens markers. Built once per model.
std::shared_ptr<const grammar::Vocabulary> grammar_vocabulary(const tok::Tokenizer& tokenizer);

// A generation's output: free until the model has written `reasoning_end` (empty: from the start; the engine keeps the
// stop tokens out meanwhile, since the answer is still to come), then the tokens of `opening`, then whatever the
// grammar allows, token by token, until it is complete.
class GrammarConstraint final : public OutputConstraint {
public:
    GrammarConstraint(std::shared_ptr<const grammar::Grammar> grammar,
                      std::shared_ptr<const grammar::Vocabulary> vocabulary, std::string reasoning_end,
                      std::vector<int32_t> opening);
    std::unique_ptr<TokenConstraint> start() const override;

    // What each generation's constraint reads, shared by them.
    struct Spec {
        std::shared_ptr<const grammar::Grammar> grammar;
        std::shared_ptr<const grammar::Vocabulary> vocabulary;
        std::string reasoning_end;
        std::vector<int32_t> opening;
    };

private:
    std::shared_ptr<const Spec> spec_;
};

// A generation free but for its calls: once it writes the marker `opener`, what follows is held to `grammar` until the
// grammar is complete (a call to its closer), then it is free again (tool_choice "auto" with strict tools).
class OpenerConstraint final : public OutputConstraint {
public:
    OpenerConstraint(int32_t opener, std::shared_ptr<const grammar::Grammar> grammar,
                     std::shared_ptr<const grammar::Vocabulary> vocabulary);
    std::unique_ptr<TokenConstraint> start() const override;

private:
    int32_t opener_;
    std::shared_ptr<const grammar::Grammar> grammar_;
    std::shared_ptr<const grammar::Vocabulary> vocabulary_;
};

}  // namespace xs::server
