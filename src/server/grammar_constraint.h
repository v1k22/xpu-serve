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

}  // namespace xs::server
