#pragma once

// Grammars written as text, as vLLM's structured outputs take them (docs/decisions/0036): a regular expression the
// whole output matches, an EBNF grammar (xgrammar's dialect, which takes GBNF's), and a choice among strings. Each is
// compiled into a byte grammar (grammar.h) whose matcher holds the output to it token by token.

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "grammar/grammar.h"

namespace xs::grammar {

// Code points as UTF-8: the byte sequences of every code point in the ranges (surrogates left out), each range split
// where its encoding changes length or lead bytes.
Expr code_points(const std::vector<std::pair<char32_t, char32_t>>& ranges);

// The whole text matches `pattern` (an ECMAScript-like regular expression, as xgrammar reads one): alternation, groups
// (capturing, `(?:`, named), the quantifiers * + ? {n} {n,} {n,m} (a lazy `?` after them changes nothing), classes with
// ranges and negation, `.` (any code point, a newline too), the escapes \d \D \w \W \s \S \n \t \r \f \v \0 \xHH \uHHHH
// \u{H...} and escaped punctuation; ^ and $ are ignored (the whole text matches either way). Lookaround,
// backreferences, \b \B, \p{...}, inline flags and an empty class are refused: GrammarError with the pattern's offset.
std::shared_ptr<const Grammar> regex_grammar(std::string_view pattern);

// An EBNF grammar in xgrammar's dialect (GBNF's): rules `name ::= body`, the root rule `root`, `#` comments, literals
// in double quotes with C and JSON escapes (\n \t \" \\ \xHH \uHHHH \UHHHHHHHH ...), classes [...] and [^...] of code
// points ([^] any), groups, `|`, and the quantifiers ? * + {n} {n,} {n,m}. A rule may refer to rules defined after it
// but not, before any character, to itself (left recursion is refused, where xgrammar takes it). xgrammar's own
// extensions (rule options, lookahead, macros such as Regex() or TagDispatch()) are refused by name. GrammarError
// names the line and column, or the rule.
std::shared_ptr<const Grammar> ebnf_grammar(std::string_view text);

// The text is exactly one of `options`.
std::shared_ptr<const Grammar> choice_grammar(const std::vector<std::string>& options);

}  // namespace xs::grammar
