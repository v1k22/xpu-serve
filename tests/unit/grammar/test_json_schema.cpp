// JSON grammars: any JSON object, and what a JSON Schema describes. Each schema with documents it takes (which also
// parse as JSON) and documents it does not; each keyword the compiler does not take refused by its path.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "grammar/json_schema.h"
#include "grammar_test_util.h"

namespace xs {

using grammar::GrammarError;
using grammar::json_object_grammar;
using grammar::json_schema_grammar;
using nlohmann::ordered_json;
using test::begins;
using test::matches;

namespace {

std::shared_ptr<const grammar::Grammar> schema(const char* text) {
    return json_schema_grammar(ordered_json::parse(text));
}

// Each of `good` matches (and is JSON), none of `bad` does.
void check(const std::shared_ptr<const grammar::Grammar>& g, const std::vector<std::string>& good,
           const std::vector<std::string>& bad) {
    for (const std::string& text : good) {
        CAPTURE(text);
        CHECK(matches(g, text));
        CHECK_FALSE(nlohmann::json::parse(text, nullptr, false).is_discarded());
    }
    for (const std::string& text : bad) {
        CAPTURE(text);
        CHECK_FALSE(matches(g, text));
    }
}

// The path a schema is refused at.
std::string refused_at(const char* text) {
    try {
        json_schema_grammar(ordered_json::parse(text));
    } catch (const GrammarError& e) {
        return e.path();
    }
    return "(taken)";
}

const char* kPerson = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string", "description": "as written"},
    "age": {"type": "integer"},
    "height_m": {"type": "number"},
    "member": {"type": "boolean"},
    "nickname": {"type": ["string", "null"]},
    "role": {"enum": ["admin", "user", 3, null]},
    "kind": {"const": "person"},
    "address": {"$ref": "#/$defs/address"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "manager": {"anyOf": [{"$ref": "#/$defs/report"}, {"type": "null"}]}
  },
  "required": ["name", "age", "height_m", "member", "nickname", "role", "kind", "address", "tags", "manager"],
  "additionalProperties": false,
  "$defs": {
    "address": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                "required": ["city", "zip"], "additionalProperties": false},
    "report": {"type": "object", "title": "Report",
               "properties": {"name": {"type": "string"},
                              "reports": {"type": "array", "items": {"$ref": "#/$defs/report"}}},
               "required": ["name", "reports"], "additionalProperties": false}
  }
})";

const char* kAda = R"({"name": "Ada", "age": 36, "height_m": 1.65, "member": true, "nickname": null, "role": 3, )"
                   R"("kind": "person", "address": {"city": "Zürich", "zip": "8001"}, "tags": ["a", "b\"c", "é"], )"
                   R"("manager": {"name": "Bo", "reports": [{"name": "Cy", "reports": []}]}})";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    return text.replace(at, from.size(), to);
}

}  // namespace

TEST_SUITE("grammar: JSON") {
    TEST_CASE("any JSON object: whitespace bounded, strings valid UTF-8, numbers as JSON writes them") {
        const auto g = json_object_grammar();
        check(g,
              {"{}", "{ }", R"({"a": 1})", R"({"a":[1,2,{"b":null}],"c":{"d":"e"}})", "\n\n{\n    \"k\": -0.5e+3\n}",
               R"({"x": "\n\t\"\\\/é é 😀 中"})", R"({"n": [0, -0, 12, 3.25, 1E5, 2e-3, true, false]})",
               "{\"deep\": " + std::string(200, '[') + std::string(200, ']') + "}"},
              {"[]", "1", "\"x\"", R"({"a" 1})", R"({"a": 1,})", R"({"a": 01})", R"({"a": 1.})", R"({"a": .5})",
               R"({"a": +1})", R"({"a": 1e})", R"({"a": NaN})", R"({'a': 1})", "{\"a\": \"\x01\"}",
               "{\"a\": \"\t\"}", "{\"a\": \"\xC3(\"}", "{\"a\": \"\xED\xA0\x80\"}", "{\"a\": \"\xC0\xAF\"}",
               "{\"a\": \"\xF5\x80\x80\x80\"}", R"({"a": "\x"})", R"({"a": "\u12g4"})", "{}\n", "{} ",
               "\n\n\n{}", "{\n" + std::string(21, ' ') + "}", "{" + std::string(3, '\n') + "}"});
        CHECK(begins(g, "{\"a\": \"\xE2\x82"));  // a token may end inside a character
        CHECK(matches(g, "{\n" + std::string(20, ' ') + "}"));
    }

    TEST_CASE("a schema's object: its properties in order, each of its type, none other") {
        const auto g = schema(kPerson);
        const std::string ada = kAda;
        const std::string pretty = ordered_json::parse(ada).dump(4);
        check(g, {ada, pretty, "\n\n" + ada, replaced(ada, R"("manager": {"name": "Bo", "reports": [{"name": "Cy", )"
                                                            R"("reports": []}]})", R"("manager": null)")},
              {replaced(ada, R"("kind": "person", )", ""),                           // a required property missing
               replaced(ada, R"("member": true, )", R"("member": true, "x": 1, )"),  // one more
               replaced(ada, R"("name": "Ada", "age": 36, )", R"("age": 36, "name": "Ada", )"),  // out of order
               replaced(ada, "36", "36.5"), replaced(ada, "36", "\"36\""), replaced(ada, "\"role\": 3", "\"role\": 4"),
               replaced(ada, "\"role\": 3", "\"role\": \"guest\""), replaced(ada, "\"person\"", "\"Person\""),
               replaced(ada, "\"nickname\": null", "\"nickname\": 5"), replaced(ada, R"(, "zip": "8001")", ""),
               replaced(ada, R"(["a", )", R"([1, )"), replaced(ada, "\"reports\": []", "\"reports\": [null]"),
               ada + "\n", replaced(ada, "1.65", "1.65,")});
        CHECK(begins(g, R"({"name": "Ad)"));
        CHECK_FALSE(begins(g, R"({"nam ")"));
        CHECK_FALSE(begins(g, R"({"age")"));
    }

    TEST_CASE("optional properties: any of them left out, the rest in order, commas only between") {
        const auto g = schema(R"({"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"},
                                  "c": {"type": "integer"}}, "required": ["b"]})");
        check(g, {R"({"b": 1})", R"({"a": 1, "b": 2})", R"({"b": 2, "c": 3})", R"({"a":1,"b":2,"c":3})"},
              {"{}", R"({"a": 1})", R"({"a": 1,, "b": 2})", R"({, "b": 1})", R"({"b": 1,})", R"({"c": 1, "b": 2})"});
        check(schema(R"({"properties": {"a": {}, "b": {"type": "null"}}})"),
              {"{}", R"({"a": [1, {"x": "y"}]})", R"({"b": null})", R"({"a": "s", "b": null})"},
              {R"({"b": 1})", R"({"c": 1})", "[]"});
        check(schema(R"({"type": "object", "additionalProperties": false})"), {"{}", "{ }"}, {R"({"a": 1})"});
        check(schema(R"({"type": "object"})"), {"{}", R"({"any": {"thing": [1]}})"}, {"[]", "null"});
    }

    TEST_CASE("types, lists of them, nullable, enum, const, anyOf, any value") {
        check(schema(R"({"type": "integer"})"), {"0", "-12", "7"}, {"1.5", "1e3", "\"1\"", "", "-"});
        check(schema(R"({"type": "number"})"), {"0", "-1.5e-7", "10"}, {"1.", "Infinity"});
        check(schema(R"({"type": ["boolean", "null"]})"), {"true", "false", "null"}, {"0", "\"true\""});
        check(schema(R"({"type": "string", "nullable": true})"), {"\"x\"", "null"}, {"1"});
        check(schema(R"({"enum": ["a", {"b": [1, 2]}, 1.5, false]})"), {"\"a\"", "{\"b\":[1,2]}", "1.5", "false"},
              {"\"b\"", "{\"b\": [1, 2]}", "1.50", "true"});
        check(schema(R"({"const": {"k": "v"}, "description": "fixed"})"), {"{\"k\":\"v\"}"}, {"{\"k\":\"w\"}"});
        check(schema(R"({"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "integer"}}]})"),
              {"\"s\"", "[]", "[1, 2]"}, {"[\"s\"]", "1"});
        check(schema("{}"), {"1", "\"s\"", "[null, {}]", "{\"a\": true}"}, {"", "nope"});
        check(schema("true"), {"[]"}, {"]"});
        check(schema(R"({"items": {"type": "boolean"}})"), {"[true, false]"}, {"[1]", "{}"});
    }

    TEST_CASE("references: to the root, recursive, under definitions, with escaped names") {
        check(schema(R"({"type": "object", "properties": {"child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}},
                         "required": ["child"]})"),
              {R"({"child": null})", R"({"child": {"child": {"child": null}}})"}, {R"({"child": {}})"});
        check(schema(R"({"definitions": {"a/b": {"type": "integer"}, "t~": {"type": "string"}}, "type": "array",
                         "items": {"anyOf": [{"$ref": "#/definitions/a~1b"}, {"$ref": "#/definitions/t~0"}]}})"),
              {"[1, \"x\", 2]"}, {"[true]"});
    }

    TEST_CASE("every keyword it does not take is refused, by its path") {
        CHECK(refused_at(R"({"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}})") ==
              "properties.age.minimum");
        CHECK(refused_at(R"({"type": "string", "pattern": "^a"})") == "pattern");
        CHECK(refused_at(R"({"type": "string", "format": "date-time"})") == "format");
        CHECK(refused_at(R"({"type": "array", "items": {"type": "string"}, "maxItems": 3})") == "maxItems");
        CHECK(refused_at(R"({"oneOf": [{"type": "string"}]})") == "oneOf");
        CHECK(refused_at(R"({"allOf": [{"type": "string"}]})") == "allOf");
        CHECK(refused_at(R"({"anyOf": [{"type": "string"}, {"type": "string", "maxLength": 2}]})") ==
              "anyOf[1].maxLength");
        CHECK(refused_at(R"({"$defs": {"x": {"type": "integer", "multipleOf": 2}}, "$ref": "#/$defs/x"})") ==
              "$defs.x.multipleOf");
        CHECK(refused_at(R"({"type": "object", "additionalProperties": true})") == "additionalProperties");
        CHECK(refused_at(R"({"type": "object", "additionalProperties": {"type": "string"}})") ==
              "additionalProperties");
        CHECK(refused_at(R"({"type": "object", "patternProperties": {}})") == "patternProperties");
        CHECK(refused_at(R"({"type": "date"})") == "type");
        CHECK(refused_at(R"({"type": []})") == "type");
        CHECK(refused_at(R"({"type": "object", "properties": {"a": {}}, "required": ["b"]})") == "required[0]");
        CHECK(refused_at(R"({"type": "string", "properties": {}})") == "properties");
        CHECK(refused_at(R"({"type": "object", "items": {}})") == "items");
        CHECK(refused_at(R"({"anyOf": [{"type": "string"}], "type": "string"})") == "type");
        CHECK(refused_at(R"({"$ref": "#/$defs/a", "type": "object", "$defs": {"a": {}}})") == "type");
        CHECK(refused_at(R"({"$ref": "https://example.com/schema.json"})") == "$ref");
        CHECK(refused_at(R"({"$ref": "#/$defs/missing"})") == "$ref");
        CHECK(refused_at(R"({"enum": ["a", 1], "type": "string"})") == "enum[1]");
        CHECK(refused_at(R"({"enum": []})") == "enum");
        CHECK(refused_at(R"({"type": "object", "properties": {"never": false}})") == "properties.never");
        CHECK(refused_at(R"({"type": "object", "properties": {"n": {"type": "integer", "nullable": "yes"}}})") ==
              "properties.n.nullable");
        CHECK(refused_at("[1]") == "");
        // Recursion without an end, and a reference to itself before any byte.
        CHECK(refused_at(R"({"$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}},
                                                "required": ["next"]}}, "$ref": "#/$defs/node"})") ==
              "$defs.node.properties.next");
        CHECK(refused_at(R"({"$defs": {"a": {"anyOf": [{"$ref": "#/$defs/a"}, {"type": "null"}]}},
                             "$ref": "#/$defs/a"})") == "$defs.a");
        // Annotations constrain nothing and are taken.
        CHECK(refused_at(R"({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "T",
                             "description": "d", "$comment": "c", "examples": [1], "default": 1, "deprecated": false,
                             "readOnly": false, "writeOnly": false, "type": "integer"})") == "(taken)");
    }
}

}  // namespace xs
