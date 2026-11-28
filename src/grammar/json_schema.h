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

class JsonRules;

// JSON values in a grammar that holds more than them (a tool call in a model's own syntax): JSON's rules made once a
// grammar, and the values of schemas added to it.
class JsonValues {
public:
    explicit JsonValues(Grammar& g);
    ~JsonValues();
    JsonValues(const JsonValues&) = delete;
    JsonValues& operator=(const JsonValues&) = delete;

    Expr whitespace();  // bounded, as between a document's tokens
    Expr any_value();
    Expr any_object();
    // The values `part` describes, a part of the schema `root` (where its $refs point); a GrammarError names `path`
    // and the keyword below it. `root` must outlive the grammar's definition.
    Expr schema(const nlohmann::ordered_json& part, const nlohmann::ordered_json& root, const std::string& path);

private:
    Grammar& g_;
    std::unique_ptr<JsonRules> rules_;
};

}  // namespace xs::grammar
