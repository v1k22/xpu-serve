#pragma once

// A JSON value as the Python reference code of a model writes it into a prompt. Decision models put states,
// instructions and option descriptions into their prompts with json.dumps or str(); the tokens the model was trained
// on follow that text exactly, so it is reproduced here rather than approximated by another JSON writer.

#include <string>

#include <nlohmann/json.hpp>

namespace xs::py {

struct JsonStyle {
    bool sort_keys = false;  // json.dumps(sort_keys=True): keys in code point order, at every depth
    bool compact = false;    // separators=(",", ":"); otherwise Python's default ", " and ": "
    // json.dumps' default: DEL and every code point past ASCII as \uxxxx (a surrogate pair past U+FFFF); otherwise
    // as it is (ensure_ascii=False).
    bool ensure_ascii = false;
};

// json.dumps(value, ...): floats as Python's repr writes them (1250.0, 1e-05, 1e+16).
std::string json_dumps(const nlohmann::ordered_json& value, JsonStyle style = {});

// str(value) of what json.loads gives for it: {'a': 1, 'b': [True, None]}. A string at the top level is itself.
std::string str(const nlohmann::ordered_json& value);

// repr(float): the shortest digits that read back as `value`.
std::string float_repr(double value);

}  // namespace xs::py
