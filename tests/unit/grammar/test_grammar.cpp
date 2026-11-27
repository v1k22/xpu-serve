// The byte grammar and its matcher: expressions lowered into rules, repeats with and without bounds, recursion, the
// states a repetition loops through, and what finish() refuses or drops.
#include <doctest/doctest.h>

#include <functional>
#include <memory>
#include <string>

#include "grammar_test_util.h"

namespace xs {

using namespace grammar;
using test::begins;
using test::matches;

namespace {

// One rule `root` defined by `e`, finished.
std::shared_ptr<const Grammar> one_rule(const Expr& e) {
    auto g = std::make_shared<Grammar>();
    const RuleId r = g->add_rule("root");
    g->define(r, e);
    g->finish(r);
    return g;
}

std::string error_path(const std::function<void()>& build) {
    try {
        build();
    } catch (const GrammarError& e) {
        return e.path();
    }
    return "(no error)";
}

}  // namespace

TEST_SUITE("grammar") {
    TEST_CASE("literals, byte sets, sequences and alternatives") {
        const auto g = one_rule(alt({literal("cat"), seq({literal("d"), bytes(ByteSet::range('a', 'c')), literal("g")}),
                                     seq({})}));
        CHECK(matches(g, "cat"));
        CHECK(matches(g, "dag"));
        CHECK(matches(g, "dcg"));
        CHECK(matches(g, ""));
        CHECK_FALSE(matches(g, "ca"));
        CHECK(begins(g, "ca"));
        CHECK_FALSE(begins(g, "dd"));
        CHECK_FALSE(matches(g, "cats"));
        CHECK_FALSE(matches(g, "dog"));
    }

    TEST_CASE("repeats: bounded, at least some, any number") {
        const auto twice_to_four = one_rule(repeat(literal("ab"), 2, 4));
        CHECK_FALSE(matches(twice_to_four, "ab"));
        CHECK(matches(twice_to_four, "abab"));
        CHECK(matches(twice_to_four, "ababababa") == false);
        CHECK(matches(twice_to_four, "abababab"));
        CHECK_FALSE(begins(twice_to_four, "ababababab"));
        const auto some = one_rule(seq({repeat(bytes(ByteSet::of("xy")), 1, kUnbounded), literal(";")}));
        CHECK_FALSE(matches(some, ";"));
        CHECK(matches(some, "xyyx;"));
        const auto exactly = one_rule(repeat(literal("z"), 3, 3));
        CHECK(matches(exactly, "zzz"));
        CHECK_FALSE(matches(exactly, "zz"));
        CHECK_FALSE(begins(exactly, "zzzz"));
    }

    TEST_CASE("a repetition loops through the same states however long it runs") {
        const auto g = one_rule(seq({literal("\""), star(bytes(ByteSet::range('a', 'z'))), literal("\"")}));
        auto automaton = std::make_shared<Automaton>(g, test::byte_vocabulary());
        Matcher m(automaton);
        REQUIRE(m.accept_bytes("\"ab"));
        const size_t states = automaton->states();
        REQUIRE(m.accept_bytes(std::string(10000, 'q')));
        CHECK(automaton->states() == states);
        CHECK(m.accept_bytes("\""));
        CHECK(m.complete());
    }

    TEST_CASE("rules that refer to each other and to themselves, nested to any depth") {
        auto g = std::make_shared<Grammar>();
        const RuleId list = g->add_rule("list");
        const RuleId item = g->add_rule("item");
        g->define(list, seq({literal("("), optional(seq({rule(item), star(seq({literal(" "), rule(item)}))})),
                             literal(")")}));
        g->define(item, alt({literal("a"), rule(list)}));
        g->finish(list);
        CHECK(matches(g, "()"));
        CHECK(matches(g, "(a (a a) ((a)) ())"));
        CHECK(matches(g, std::string(500, '(') + std::string(500, ')')));
        CHECK_FALSE(matches(g, "(a (a a)"));
        CHECK_FALSE(begins(g, "(a))"));
        CHECK_FALSE(begins(g, "(a  a)"));
    }

    TEST_CASE("ambiguous alternatives are all followed") {
        const auto g = one_rule(alt({seq({literal("ab"), literal("c")}), seq({literal("a"), literal("bd")}),
                                     seq({star(literal("a")), literal("b")})}));
        CHECK(matches(g, "abc"));
        CHECK(matches(g, "abd"));
        CHECK(matches(g, "aaab"));
        CHECK(matches(g, "ab"));
        CHECK_FALSE(matches(g, "abe"));
    }

    TEST_CASE("an alternative that can never end is dropped; a root that can never end is refused by name") {
        auto g = std::make_shared<Grammar>();
        const RuleId root = g->add_rule("root");
        const RuleId forever = g->add_rule("root.forever");
        g->define(root, alt({literal("a"), rule(forever)}));
        g->define(forever, seq({literal("b"), rule(forever)}));
        g->finish(root);
        CHECK(matches(g, "a"));
        CHECK_FALSE(begins(g, "b"));  // no way to end once it starts

        CHECK(error_path([] {
                  Grammar h;
                  const RuleId r = h.add_rule("doc");
                  const RuleId node = h.add_rule("doc.node");
                  const RuleId next = h.add_rule("doc.node.next");
                  h.define(r, rule(node));
                  h.define(node, seq({literal("{"), rule(next), literal("}")}));
                  h.define(next, rule(node));
                  h.finish(r);
              }) == "doc.node.next");
        CHECK(error_path([] {
                  Grammar h;
                  const RuleId r = h.add_rule("r");
                  h.define(r, bytes(ByteSet()));
                  h.finish(r);
              }) == "r");
    }

    TEST_CASE("left recursion is refused by name, also past items that may match nothing") {
        CHECK(error_path([] {
                  Grammar h;
                  const RuleId r = h.add_rule("expr");
                  h.define(r, alt({seq({rule(r), literal("+1")}), literal("1")}));
                  h.finish(r);
              }) == "expr");
        CHECK(error_path([] {
                  Grammar h;
                  const RuleId r = h.add_rule("a");
                  const RuleId maybe = h.add_rule("maybe");
                  const RuleId b = h.add_rule("b");
                  h.define(r, seq({rule(maybe), rule(b)}));
                  h.define(maybe, alt({seq({}), literal("m")}));
                  h.define(b, alt({rule(r), literal("x")}));
                  h.finish(r);
              }) != "(no error)");
        // Right recursion is fine.
        const auto g = one_rule(alt({literal("1"), seq({literal("1+"), rule(0)})}));
        CHECK(matches(g, "1+1+1"));
    }

    TEST_CASE("the matcher: complete where the root may end, dead past what it takes") {
        const auto g = one_rule(seq({literal("1"), star(literal("0"))}));
        grammar::Matcher m = test::byte_matcher(g);
        CHECK_FALSE(m.complete());
        CHECK(m.allows('1'));
        CHECK_FALSE(m.allows('0'));
        REQUIRE(m.accept_token('1'));
        CHECK(m.complete());
        CHECK(m.allows('0'));  // complete, and it may go on
        grammar::Matcher copy = m;
        CHECK_FALSE(copy.accept_token('2'));
        CHECK(copy.dead());
        CHECK(copy.allowed().count() == 0);
        CHECK_FALSE(m.dead());  // a copy steps on alone
        CHECK(m.allowed().count() == 1);
    }
}

}  // namespace xs
