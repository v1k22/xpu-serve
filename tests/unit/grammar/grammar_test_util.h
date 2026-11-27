#pragma once

// What the grammar tests share: a vocabulary of the 256 bytes, and whether a grammar matches a text.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "grammar/grammar.h"
#include "grammar/matcher.h"
#include "grammar/vocabulary.h"

namespace xs::test {

// Token b is the byte b.
inline std::shared_ptr<const grammar::Vocabulary> byte_vocabulary() {
    static const auto v = [] {
        std::vector<grammar::Vocabulary::Token> tokens;
        for (int b = 0; b < 256; ++b) tokens.push_back({std::string(1, static_cast<char>(b)), false});
        return std::make_shared<const grammar::Vocabulary>(std::move(tokens));
    }();
    return v;
}

inline grammar::Matcher byte_matcher(std::shared_ptr<const grammar::Grammar> g) {
    return grammar::Matcher(std::make_shared<grammar::Automaton>(std::move(g), byte_vocabulary()));
}

// The whole text is one of the grammar's strings.
inline bool matches(std::shared_ptr<const grammar::Grammar> g, std::string_view text) {
    grammar::Matcher m = byte_matcher(std::move(g));
    return m.accept_bytes(text) && m.complete();
}

// The text begins one of the grammar's strings.
inline bool begins(std::shared_ptr<const grammar::Grammar> g, std::string_view text) {
    grammar::Matcher m = byte_matcher(std::move(g));
    return m.accept_bytes(text);
}

}  // namespace xs::test
