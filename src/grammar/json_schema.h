#pragma once

// Grammars of JSON documents (docs/decisions/0033): any JSON object (OpenAI's json_object), or the values a JSON Schema
// describes (json_schema), in the subset OpenAI's strict structured outputs take.
//
// Whitespace is bounded wherever it may go: at most two newlines and then twenty spaces or tabs, so a model cannot
// loop on it. A document may start with such whitespace and ends with its value. Strings are valid UTF-8.
//
// Schemas: type (one, or a list), object (properties in their order, required, additionalProperties false), array
// (items), string, number, integer, boolean, null; nullable; enum and const (the values as compact JSON); anyOf;
// $ref to the schema's own parts ("#", "#/$defs/..."), recursive ones included; and the annotations that constrain
// nothing (title, description, $comment, examples, default, deprecated, readOnly, writeOnly, $schema). Every other
// keyword is refused by name. An object with properties takes no others, whether or not additionalProperties says
// so; one with neither is any object.

#include <memory>

#include <nlohmann/json.hpp>

#include "grammar/grammar.h"

namespace xs::grammar {

std::shared_ptr<const Grammar> json_object_grammar();

// Throws GrammarError naming the path of what it does not take, e.g. "properties.age.minimum".
std::shared_ptr<const Grammar> json_schema_grammar(const nlohmann::ordered_json& schema);

}  // namespace xs::grammar
