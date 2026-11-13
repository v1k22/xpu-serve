// The text Python writes for a JSON value (models/common/py_text.h), against what Python itself wrote: json.dumps with
// and without sorted keys and compact separators, str() of the loaded value, repr() of floats.

#include <doctest/doctest.h>

#include <string>

#include <nlohmann/json.hpp>

#include "models/common/py_text.h"

namespace {

using Json = nlohmann::ordered_json;

// A string from its JSON encoding (the expectations hold controls, quotes and non-ASCII text).
std::string text(const char* json) { return Json::parse(json).get<std::string>(); }

// json.dumps(value): the value every case reads.
const char* kValue = R"J("{\"zeta\": 1e-05, \"alpha\": 1e+16, \"mid\": -0.5, \"count\": 12345678901234, \"ok\": false, \"none\": null, \"text\": \"quote \\\" backslash \\\\ newline \\n tab \\t done \\u0001 \\u007f\", \"nested\": {\"b\": [1, 2.5, \"x\"], \"a\": {\"y\": 1, \"x\": 0.1}}, \"unicode\": \"na\\u00efve caf\\u00e9 \\u2014 \\u65e5\\u672c\\u8a9e \\u2713\", \"it's\": \"single ' only\", \"both\": \"a ' and a \\\" here\", \"nbsp\": \"a\\u00a0b\\u200bc\\u2028d\\ud83d\\ude00e\", \"empty\": {}, \"list\": []}")J";

}  // namespace

TEST_SUITE("python text") {
    TEST_CASE("floats print as repr() prints them") {
        const std::pair<double, const char*> cases[] = {
        {1250.0, "1250.0"},
        {1e-05, "1e-05"},
        {1e+16, "1e+16"},
        {1000000000000000.0, "1000000000000000.0"},
        {0.1, "0.1"},
        {123456789.125, "123456789.125"},
        {-0.5, "-0.5"},
        {2.5e-07, "2.5e-07"},
        {1e+22, "1e+22"},
        {5e-324, "5e-324"},
        {1.7976931348623157e+308, "1.7976931348623157e+308"},
        {0.0001, "0.0001"},
        {0.00012345, "0.00012345"},
        {100.0, "100.0"},
        {3.0, "3.0"},
        {9999999999999998.0, "9999999999999998.0"},
        {1.2345678901234567e+19, "1.2345678901234567e+19"},
        {-0.0, "-0.0"},
        {0.30000000000000004, "0.30000000000000004"}};
        for (const auto& [value, expected] : cases) CHECK(xs::py::float_repr(value) == expected);
    }

    TEST_CASE("json.dumps keeps the key order, or sorts it, with Python's separators") {
        const Json value = Json::parse(text(kValue));
        CHECK(xs::py::json_dumps(value) == text(R"J("{\"zeta\": 1e-05, \"alpha\": 1e+16, \"mid\": -0.5, \"count\": 12345678901234, \"ok\": false, \"none\": null, \"text\": \"quote \\\" backslash \\\\ newline \\n tab \\t done \\u0001 \u007f\", \"nested\": {\"b\": [1, 2.5, \"x\"], \"a\": {\"y\": 1, \"x\": 0.1}}, \"unicode\": \"na\u00efve caf\u00e9 \u2014 \u65e5\u672c\u8a9e \u2713\", \"it's\": \"single ' only\", \"both\": \"a ' and a \\\" here\", \"nbsp\": \"a\u00a0b\u200bc\u2028d\ud83d\ude00e\", \"empty\": {}, \"list\": []}")J"));
        CHECK(xs::py::json_dumps(value, {.sort_keys = true, .compact = true}) == text(R"J("{\"alpha\":1e+16,\"both\":\"a ' and a \\\" here\",\"count\":12345678901234,\"empty\":{},\"it's\":\"single ' only\",\"list\":[],\"mid\":-0.5,\"nbsp\":\"a\u00a0b\u200bc\u2028d\ud83d\ude00e\",\"nested\":{\"a\":{\"x\":0.1,\"y\":1},\"b\":[1,2.5,\"x\"]},\"none\":null,\"ok\":false,\"text\":\"quote \\\" backslash \\\\ newline \\n tab \\t done \\u0001 \u007f\",\"unicode\":\"na\u00efve caf\u00e9 \u2014 \u65e5\u672c\u8a9e \u2713\",\"zeta\":1e-05}")J"));
    }

    TEST_CASE("json.dumps with an indent: a member a line, \",\" between them, empty containers closed at once") {
        const Json value = Json::parse(text(kValue));
        CHECK(xs::py::json_dumps(value, {.indent = 2}) == text(R"J("{\n  \"zeta\": 1e-05,\n  \"alpha\": 1e+16,\n  \"mid\": -0.5,\n  \"count\": 12345678901234,\n  \"ok\": false,\n  \"none\": null,\n  \"text\": \"quote \\\" backslash \\\\ newline \\n tab \\t done \\u0001 \u007f\",\n  \"nested\": {\n    \"b\": [\n      1,\n      2.5,\n      \"x\"\n    ],\n    \"a\": {\n      \"y\": 1,\n      \"x\": 0.1\n    }\n  },\n  \"unicode\": \"naïve café — 日本語 ✓\",\n  \"it's\": \"single ' only\",\n  \"both\": \"a ' and a \\\" here\",\n  \"nbsp\": \"a b​c d😀e\",\n  \"empty\": {},\n  \"list\": []\n}")J"));
        CHECK(xs::py::json_dumps(Json::parse(R"([[], {}, [1, [2, {"a": []}]], "s"])"), {.indent = 2}) ==
              text(R"J("[\n  [],\n  {},\n  [\n    1,\n    [\n      2,\n      {\n        \"a\": []\n      }\n    ]\n  ],\n  \"s\"\n]")J"));
        CHECK(xs::py::json_dumps(Json("top"), {.indent = 2}) == "\"top\"");
        CHECK(xs::py::json_dumps(Json(3.0), {.indent = 2}) == "3.0");
    }

    TEST_CASE("str() of a loaded value is Python's repr; a string is itself") {
        const Json value = Json::parse(text(kValue));
        CHECK(xs::py::str(value) == text(R"J("{'zeta': 1e-05, 'alpha': 1e+16, 'mid': -0.5, 'count': 12345678901234, 'ok': False, 'none': None, 'text': 'quote \" backslash \\\\ newline \\n tab \\t done \\x01 \\x7f', 'nested': {'b': [1, 2.5, 'x'], 'a': {'y': 1, 'x': 0.1}}, 'unicode': 'na\u00efve caf\u00e9 \u2014 \u65e5\u672c\u8a9e \u2713', \"it's\": \"single ' only\", 'both': 'a \\' and a \" here', 'nbsp': 'a\\xa0b\\u200bc\\u2028d\ud83d\ude00e', 'empty': {}, 'list': []}")J"));
        CHECK(xs::py::str(Json("it's \"plain\"")) == "it's \"plain\"");
        CHECK(xs::py::str(Json::parse("[true, null, 1.0, \"a\"]")) == "[True, None, 1.0, 'a']");
    }
}
