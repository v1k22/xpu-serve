// Grammars written as text (grammar/text_grammars.h), as vLLM's structured outputs take them: regular expressions
// matched whole, xgrammar's EBNF (GBNF's), a choice among strings, and code point classes as UTF-8; what each refuses
// is named by its place.
#include <doctest/doctest.h>

#include <functional>
#include <string>

#include "grammar/text_grammars.h"
#include "grammar_test_util.h"

namespace xs {

using namespace grammar;
using test::matches;

namespace {

std::string refused(const std::function<void()>& build) {
    try {
        build();
    } catch (const GrammarError& e) {
        return e.path() + ": " + e.what();
    }
    return "(accepted)";
}

std::shared_ptr<const Grammar> rx(const char* p) { return regex_grammar(p); }

}  // namespace

TEST_SUITE("grammar: text") {
    TEST_CASE("code points as UTF-8: every length, surrogates left out, ranges across lead bytes") {
        auto g = std::make_shared<Grammar>();
        const RuleId r = g->add_rule("root");
        g->define(r, code_points({{0x7E, 0x801}, {0xD7FF, 0xE000}, {0xFFFF, 0x10001}, {0x10FFFF, 0x10FFFF}}));
        g->finish(r);
        std::shared_ptr<const Grammar> cg = g;
        CHECK(matches(cg, "~"));
        CHECK(matches(cg, "\x7F"));
        CHECK(matches(cg, "\xC2\x80"));      // U+0080
        CHECK(matches(cg, "\xDF\xBF"));      // U+07FF
        CHECK(matches(cg, "\xE0\xA0\x81"));  // U+0801
        CHECK_FALSE(matches(cg, "\xE0\xA0\x82"));
        CHECK(matches(cg, "\xED\x9F\xBF"));  // U+D7FF
        CHECK_FALSE(matches(cg, "\xED\xA0\x80"));  // a surrogate
        CHECK(matches(cg, "\xEE\x80\x80"));  // U+E000
        CHECK(matches(cg, "\xF0\x90\x80\x81"));  // U+10001
        CHECK(matches(cg, "\xF4\x8F\xBF\xBF"));  // U+10FFFF
        CHECK_FALSE(matches(cg, "}"));
    }

    TEST_CASE("regular expressions: the whole text, classes, escapes, groups and quantifiers") {
        CHECK(matches(rx("[A-Z][a-z]+ \\d{3}-\\d{4}"), "Ann 555-0199"));
        CHECK_FALSE(matches(rx("[A-Z][a-z]+ \\d{3}-\\d{4}"), "Ann 555-0199 "));  // the whole text
        CHECK(matches(rx("^(yes|no)$"), "no"));
        CHECK(matches(rx("(?:ab){2,3}c?"), "ababab"));
        CHECK_FALSE(matches(rx("(?:ab){2,3}c?"), "ab"));
        CHECK(matches(rx("(?<year>\\d{4})-(\\d{2})"), "2026-11"));
        CHECK(matches(rx("a.c"), "a\nc"));      // . takes a newline, as xgrammar's
        CHECK(matches(rx("a.c"), "a\xC3\xA9" "c"));  // and any code point
        CHECK_FALSE(matches(rx("a.c"), "a\xC3" "c"));
        CHECK(matches(rx("[^a-z]+"), "ÉÜ1 "));
        CHECK_FALSE(matches(rx("[^a-z]+"), "x"));
        CHECK(matches(rx("[α-ω]{2}"), "λμ"));
        CHECK(matches(rx("\\w+\\s\\W"), "a_1 !"));
        CHECK(matches(rx("\\x41\\u0042\\u{1F600}[\\-.]"), "AB😀-"));
        CHECK(matches(rx("a\\.b\\*c\\[\\]"), "a.b*c[]"));
        CHECK(matches(rx("x{2,}"), "xxxxx"));
        CHECK(matches(rx("x{2}?y*?"), "xxyy"));  // lazy quantifiers: the same texts
        CHECK(matches(rx("a{,3}"), "a{,3}"));    // a brace that bounds nothing is itself
        CHECK(matches(rx(""), ""));
    }

    TEST_CASE("regular expressions refused, each at its offset") {
        CHECK(refused([] { rx("a(?=b)"); }) == "offset 1: has a lookahead, which is not supported");
        CHECK(refused([] { rx("(?<!a)b"); }) == "offset 0: has a lookbehind, which is not supported");
        CHECK(refused([] { rx("(a)\\1"); }) == "offset 3: has a backreference, which is not supported");
        CHECK(refused([] { rx("\\bword"); }) == "offset 0: has a word boundary (\\b), which is not supported");
        CHECK(refused([] { rx("\\p{L}"); }) == "offset 0: has a Unicode property class, which is not supported");
        CHECK(refused([] { rx("(?i)a"); }) == "offset 0: has a group modifier (an inline flag), which is not supported");
        CHECK(refused([] { rx("[]"); }) == "offset 0: has an empty class");
        CHECK(refused([] { rx("a**"); }) == "offset 2: has a quantifier after another (one at a time; group the first)");
        CHECK(refused([] { rx("(ab"); }) == "offset 0: has a group that never closes");
        CHECK(refused([] { rx("ab)"); }) == "offset 2: has an unmatched ')'");
        CHECK(refused([] { rx("*a"); }) == "offset 0: has '*' with nothing to repeat");
        CHECK(refused([] { rx("[z-a]"); }) == "offset 3: has a class range out of order");
        CHECK(refused([] { rx("a{3,2}"); }) == "offset 1: has a repeat whose maximum is below its minimum");
        CHECK(refused([] { rx("a{20000}"); }).starts_with("offset 1: repeats more than"));
    }

    TEST_CASE("EBNF: rules, the root, literals with escapes, classes, groups, repeats, comments") {
        const char* json_like = R"(# a key and a value
root   ::= "{" ws pair ("," ws pair)* ws "}"
pair   ::= string ws ":" ws value
value  ::= string | number | "true" | "false"
         | "null"
string ::= "\"" [^"\\\n]* "\""
number ::= "-"? [0-9]+ ("." [0-9]{1,6})?
ws     ::= [ \t\n]{0,4}
)";
        const auto g = ebnf_grammar(json_like);
        CHECK(matches(g, R"({"a": 1.25, "b":"x y", "c": null})"));
        CHECK(matches(g, "{\n  \"k\" : -7\n}"));
        CHECK_FALSE(matches(g, R"({"a": 1.1234567})"));
        CHECK_FALSE(matches(g, R"({"a": maybe})"));
        CHECK(matches(ebnf_grammar("root ::= \"\\u00e9\\x41\\t\" [^] \"\""), "\xC3\xA9" "A\t\n"));
        CHECK(matches(ebnf_grammar("root ::= item item\nitem ::= [a-c]"), "ab"));  // a rule defined after its use
        CHECK(matches(ebnf_grammar("root ::= [0-9]{2,} \"!\" [a]{3}"), "123!aaa"));
        CHECK(matches(ebnf_grammar("root ::= (\"x\" | \"y\")+ ( )"), "xyyx"));
        CHECK(matches(ebnf_grammar("root ::= a\na ::= \"(\" a \")\" | \"\""), "((()))"));  // recursive, not on the left
        CHECK(matches(ebnf_grammar("root-rule ::= \"ok\"\nroot ::= root-rule"), "ok"));
    }

    TEST_CASE("EBNF refused: by line and column, or by rule") {
        CHECK(refused([] { ebnf_grammar("start ::= \"a\""); }) == "root: is required: the rule the output matches");
        CHECK(refused([] { ebnf_grammar("root ::= missing"); }) == "missing: is used but never defined");
        CHECK(refused([] { ebnf_grammar("root ::= 'a'"); }) ==
              "line 1, column 10: has a literal in single quotes: write it in double quotes");
        CHECK(refused([] { ebnf_grammar("root ::= \"a\"\nroot ::= \"b\""); }) ==
              "line 2, column 1: defines the rule 'root' twice");
        CHECK(refused([] { ebnf_grammar("root ::= [\\d]"); }) ==
              "line 1, column 11: has a class escape (\\d, \\w, \\s), which EBNF does not take: write the class");
        CHECK(refused([] { ebnf_grammar("root ::= Regex(\"a+\")"); }) ==
              "line 1, column 10: calls 'Regex(...)', an xgrammar macro, which is not supported");
        CHECK(refused([] { ebnf_grammar("root ::= \"a\" (= \"b\")"); }).ends_with("has a lookahead ((= ...)), which is not supported"));
        CHECK(refused([] { ebnf_grammar("root[max_tokens=3] ::= \"a\""); }).ends_with(
            "has rule options (xgrammar's [...] after a name), which are not supported"));
        CHECK(refused([] { ebnf_grammar("root ::= root \"a\" | \"b\""); }) ==
              "root: comes back to itself before any byte (left recursion)");
        CHECK(refused([] { ebnf_grammar("root ::= \"a"); }) == "line 1, column 10: has a literal that never closes");
        CHECK(refused([] { ebnf_grammar("root ::= \"a\"++"); }).ends_with("has a quantifier after another (one at a time; group the first)"));
    }

    TEST_CASE("a choice: exactly one of the strings") {
        const auto g = choice_grammar({"positive", "negative", "neutral 😐"});
        CHECK(matches(g, "negative"));
        CHECK(matches(g, "neutral 😐"));
        CHECK_FALSE(matches(g, "neg"));
        CHECK_FALSE(matches(g, "positive "));
        CHECK(refused([] { choice_grammar({}); }) == ": must be a non-empty list of strings");
    }
}

}  // namespace xs
