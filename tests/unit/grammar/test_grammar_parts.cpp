// Grammar parts a model's own syntax needs (a tool call): marker tokens at their places, text up to a closer spelled in
// bytes, and JSON values (any, or a schema's) inside a grammar that holds more than them.
#include <doctest/doctest.h>

#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "grammar/json_schema.h"
#include "grammar_test_util.h"

namespace xs {

using namespace grammar;
using test::matches;

namespace {

// Tokens 0 to 255 are the bytes; 256 is the marker <T> (an added token), 257 the marker <U>.
std::shared_ptr<const Vocabulary> marker_vocabulary() {
    std::vector<Vocabulary::Token> tokens;
    for (int b = 0; b < 256; ++b) tokens.push_back({std::string(1, static_cast<char>(b)), false});
    tokens.push_back({"<T>", true});
    tokens.push_back({"<U>", true});
    return std::make_shared<const Vocabulary>(std::move(tokens));
}

constexpr int32_t kT = 256, kU = 257;

std::shared_ptr<const Grammar> finished(const std::function<Expr(Grammar&)>& build) {
    auto g = std::make_shared<Grammar>();
    const RuleId r = g->add_rule("root");
    g->define(r, build(*g));
    g->finish(r);
    return g;
}

}  // namespace

TEST_SUITE("grammar: parts") {
    TEST_CASE("marker tokens: allowed where the grammar names them, never their bytes spelled out") {
        const auto g = finished([](Grammar&) { return seq({literal("a"), token(kT), alt({literal("b"), token(kU)})}); });
        auto a = std::make_shared<Automaton>(g, marker_vocabulary());
        Matcher m(a);
        CHECK(m.allowed().count() == 1);
        CHECK(m.allows('a'));
        CHECK_FALSE(m.allows(kT));
        REQUIRE(m.accept_token('a'));
        CHECK(m.allowed().count() == 1);
        CHECK(m.allowed().has(kT));
        CHECK_FALSE(m.allows('<'));  // the marker's bytes spelled are not the marker
        Matcher spelled = m;
        CHECK_FALSE(spelled.accept_bytes("<T>"));
        REQUIRE(m.accept_token(kT));
        CHECK(m.allowed().has('b'));
        CHECK(m.allowed().has(kU));
        CHECK_FALSE(m.allowed().has(kT));
        Matcher other = m;
        REQUIRE(m.accept_token(kU));
        CHECK(m.complete());
        REQUIRE(other.accept_token('b'));
        CHECK(other.complete());
        // A rule of markers and bytes that repeats: the same states each time.
        const auto r = finished([](Grammar&) { return repeat(seq({token(kT), literal("x")}), 1, kUnbounded); });
        auto ra = std::make_shared<Automaton>(r, marker_vocabulary());
        Matcher rm(ra);
        for (int i = 0; i < 5; ++i) {
            REQUIRE(rm.accept_token(kT));
            REQUIRE(rm.accept_token('x'));
            CHECK(rm.complete());
        }
        const size_t states = ra->states();
        REQUIRE(rm.accept_token(kT));
        REQUIRE(rm.accept_token('x'));
        CHECK(ra->states() == states);
    }

    TEST_CASE("text without a stop: any bytes in which it never occurs, then the closer") {
        const auto g = finished([](Grammar& g) {
            return seq({literal("<p>"), text_without(g, "\n</p>", "value"), literal("\n</p>")});
        });
        CHECK(matches(g, "<p>\n</p>"));
        CHECK(matches(g, "<p>hello\n</p>"));
        CHECK(matches(g, "<p>a\n<b>\n</q\n</p>"));     // prefixes of the stop inside
        CHECK(matches(g, "<p>\n\n</p>"));             // a newline then the stop
        CHECK(matches(g, "<p>\n</p\n</p>"));
        CHECK_FALSE(matches(g, "<p>a\n</p>b\n</p>"));  // the stop inside the text
        CHECK_FALSE(matches(g, "<p>a\n</p"));
        // A stop that overlaps itself.
        const auto aab = finished([](Grammar& g) { return seq({text_without(g, "aab", "t"), literal("aab")}); });
        CHECK(matches(aab, "aab"));
        CHECK(matches(aab, "abaab"));
        CHECK(matches(aab, "aaab"));  // "a" then the stop
        CHECK_FALSE(matches(aab, "aabaab"));
    }

    TEST_CASE("JSON values inside a larger grammar: any object, a schema's values, errors under the path given") {
        const nlohmann::ordered_json root = nlohmann::ordered_json::parse(R"({"type": "object",
            "properties": {"city": {"type": "string"}, "days": {"$ref": "#/$defs/n"}}, "required": ["city"],
            "additionalProperties": false, "$defs": {"n": {"type": "integer"}}})");
        const auto g = finished([&](Grammar& g) {
            JsonValues json(g);
            return alt({seq({literal("call("), json.schema(root, root, "tools[0]"), literal(")")}),
                        seq({literal("any("), json.any_object(), literal(")")})});
        });
        CHECK(matches(g, R"(call({"city": "Paris", "days": 2}))"));
        CHECK(matches(g, R"(call({"city":"Paris"}))"));
        CHECK_FALSE(matches(g, R"(call({"days": 2}))"));
        CHECK_FALSE(matches(g, R"(call({"city": 3}))"));
        CHECK(matches(g, R"(any({"x": [1, {"y": null}]}))"));
        // A property's schema on its own, its $ref read in the whole schema.
        const auto days = finished([&](Grammar& g) {
            JsonValues json(g);
            return json.schema(root["properties"]["days"], root, "p");
        });
        CHECK(matches(days, "12"));
        CHECK_FALSE(matches(days, "1.5"));
        // What a schema does not take is named under the path it was given.
        const nlohmann::ordered_json bad = nlohmann::ordered_json::parse(R"({"type": "integer", "minimum": 1})");
        try {
            finished([&](Grammar& g) {
                JsonValues json(g);
                return json.schema(bad, bad, "tools[2].function.parameters");
            });
            FAIL("accepted");
        } catch (const GrammarError& e) {
            CHECK(e.path() == "tools[2].function.parameters.minimum");
        }
    }
}

}  // namespace xs
