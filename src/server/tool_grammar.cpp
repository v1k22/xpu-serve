#include "server/tool_grammar.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "grammar/json_schema.h"
#include "server/request_fields.h"
#include "util/error.h"

namespace xs::server {
namespace {

using namespace grammar;
using ojson = nlohmann::ordered_json;

const ojson kNoParameters = ojson::parse(R"({"type": "object", "properties": {}})");

std::string tool_path(size_t i) { return str_cat("tools[", i, "].function.parameters"); }

}  // namespace

bool writes_tool_calls(std::string_view dialect) {
    return dialect == "qwen3_5" || dialect == "qwen3" || dialect == "gemma4" || dialect == "inkling2";
}

namespace {

// The syntax of one dialect's calls, over the grammar being built.
class CallBuilder {
public:
    CallBuilder(Grammar& g, std::string_view dialect, const Vocabulary& vocabulary)
        : g_(g), json_(g), dialect_(dialect), vocabulary_(vocabulary) {
        if (!writes_tool_calls(dialect_)) {
            throw InvalidRequest(str_cat("tool_choice: the model's output dialect '", dialect_, "' writes no tool calls"),
                                 "tool_choice");
        }
    }

    // One call of the tool `function` (tools[index]), its arguments held to its schema when it is strict: the
    // dialect's opener, then rest().
    Expr call(const RequestJson& function, size_t index) {
        if (dialect_ == "inkling2") {  // its opener, the message's header, holds the tool's name
            return seq({marker("<|message_model|>"), literal(function.at("name").get<std::string>()),
                        marker(opener()), rest(function, index)});
        }
        return seq({marker(opener()), rest(function, index)});
    }

    // The marker a call starts with, after which (Inkling 2: after its header) a call of any tool may follow.
    std::string_view opener() const {
        if (dialect_ == "gemma4") return "<|tool_call>";
        if (dialect_ == "inkling2") return "<|content_invoke_tool_json|>";
        return "<tool_call>";
    }
    std::optional<int32_t> opener_token() const { return vocabulary_.marker(opener()); }

    // A call of the tool `function` past the opener, to its closer.
    Expr rest(const RequestJson& function, size_t index) {
        const std::string name = function.at("name").get<std::string>();
        const bool strict = function.contains("strict") && function.at("strict").is_boolean() &&
                            function.at("strict").get<bool>();
        const RequestJson* params = function.contains("parameters") && !function.at("parameters").is_null()
                                        ? &function.at("parameters")
                                        : nullptr;
        const ojson& schema = params != nullptr ? *params : kNoParameters;
        const std::string path = tool_path(index);
        if (dialect_ == "qwen3_5") {
            return seq({literal("\n<function=" + name + ">\n"),
                        strict ? xml_parameters(schema, path) : xml_any_parameters(), literal("</function>\n"),
                        marker("</tool_call>")});
        }
        if (dialect_ == "qwen3") {
            return seq({literal("\n{\"name\": " + ojson(name).dump() + ", \"arguments\": "),
                        strict ? json_.schema(schema, schema, path) : json_.any_object(), literal("}\n"),
                        marker("</tool_call>")});
        }
        if (dialect_ == "gemma4") {
            return seq({literal("call:" + name), strict ? gemma_parameters(schema, path) : gemma_object(),
                        marker("<tool_call|>")});
        }
        // inkling2: the body {"name": ..., "args": ...} of a message of its own.
        return seq({literal("{\"name\":" + ojson(name).dump() + ",\"args\":"),
                    strict ? json_.schema(schema, schema, path) : json_.any_object(), literal("}"),
                    marker("<|end_message|>")});
    }

    // What separates two calls, and what may come before the first (after a thought, a newline or two).
    Expr between() { return dialect_ == "qwen3_5" || dialect_ == "qwen3" ? repeat(bytes(ByteSet::of("\n")), 0, 2) : seq({}); }
    Expr lead() { return repeat(bytes(ByteSet::of(" \n")), 0, 3); }

    // `text` with the marker tokens it spells as their tokens (an answer's opening: Inkling 2's message header).
    Expr text_with_markers(std::string_view text) {
        std::vector<Expr> parts;
        std::string plain;
        for (size_t at = 0; at < text.size();) {
            size_t taken = 0;
            for (size_t end = text.size(); end > at && taken == 0; --end) {
                if (vocabulary_.marker(text.substr(at, end - at))) taken = end - at;
            }
            if (taken == 0) {
                plain += text[at++];
                continue;
            }
            if (!plain.empty()) parts.push_back(literal(plain));
            plain.clear();
            parts.push_back(marker(text.substr(at, taken)));
            at += taken;
        }
        if (!plain.empty()) parts.push_back(literal(plain));
        return seq(std::move(parts));
    }

private:
    // A marker of the syntax: its token when the vocabulary has one, else its bytes.
    Expr marker(std::string_view text) {
        const std::optional<int32_t> id = vocabulary_.marker(text);
        return id ? token(*id) : literal(text);
    }

    // A tool's parameters schema read as an object's properties in order, each required one present: the members a
    // dialect writes one after another (`member` gives one), an optional one possibly left out.
    template <typename Member>
    Expr properties(const ojson& schema, const std::string& path, Member member, const Expr& separator) {
        const auto refuse = [&](const std::string& key, const std::string& why) {
            throw InvalidRequest(str_cat("'", path, ".", key, "' ", why), str_cat(path, ".", key));
        };
        if (!schema.is_object()) throw InvalidRequest(str_cat("'", path, "' must be an object schema"), path);
        const ojson* props = schema.contains("properties") ? &schema.at("properties") : nullptr;
        if (props != nullptr && !props->is_object()) refuse("properties", "must be an object of schemas");
        std::set<std::string> required;
        if (schema.contains("required")) {
            const ojson& names = schema.at("required");
            if (!names.is_array()) refuse("required", "must be an array of property names");
            for (const ojson& n : names) {
                if (!n.is_string() || props == nullptr || !props->contains(n.get<std::string>())) {
                    refuse("required", "names no property of 'properties'");
                }
                required.insert(n.get<std::string>());
            }
        }
        std::vector<std::pair<std::string, const ojson*>> members;
        if (props != nullptr) {
            for (const auto& [name, s] : props->items()) members.emplace_back(name, &s);
        }
        // first[i]: the members from property i on, none written yet; later[i]: the same after one was.
        const size_t n = members.size();
        std::vector<RuleId> first(n + 1), later(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            first[i] = g_.add_rule(path);
            later[i] = g_.add_rule(path);
        }
        g_.define(first[n], seq({}));
        g_.define(later[n], seq({}));
        for (size_t i = n; i-- > 0;) {
            const Expr one = member(members[i].first, *members[i].second, str_cat(path, ".properties.", members[i].first));
            const Expr here = seq({one, rule(later[i + 1])});
            const Expr after = seq({separator, one, rule(later[i + 1])});
            if (required.contains(members[i].first)) {
                g_.define(first[i], here);
                g_.define(later[i], after);
            } else {
                g_.define(first[i], alt({here, rule(first[i + 1])}));
                g_.define(later[i], alt({after, rule(later[i + 1])}));
            }
        }
        return rule(first[0]);
    }

    // A JSON schema of strings alone: its value then is the text itself, as the XML and Gemma syntaxes write strings.
    static bool string_only(const ojson& s) {
        return s.is_object() && s.contains("type") && s.at("type") == "string" && !s.contains("enum") &&
               !s.contains("const");
    }

    // Qwen 3.5: <parameter=NAME>\nVALUE\n</parameter>\n per argument, a string's value as text, others as JSON.
    Expr xml_parameters(const ojson& schema, const std::string& path) {
        const auto member = [&](const std::string& name, const ojson& s, const std::string& at) {
            const Expr value = string_only(s) ? text_without(g_, "\n</parameter>", at) : json_.schema(s, schema, at);
            return seq({literal("<parameter=" + name + ">\n"), value, literal("\n</parameter>\n")});
        };
        return properties(schema, path, member, seq({}));
    }

    Expr xml_any_parameters() {
        const Expr name = seq({bytes(ByteSet::range('a', 'z').add('A', 'Z').add('_', '_')),
                               star(bytes(ByteSet::range('a', 'z').add('A', 'Z').add('0', '9').add('_', '_')))});
        return star(seq({literal("<parameter="), name, literal(">\n"), text_without(g_, "\n</parameter>", "(parameter)"),
                         literal("\n</parameter>\n")}));
    }

    // Gemma 4: {key:value,...}, strings between <|"|> tokens, numbers and literals as JSON, objects and lists nested.
    Expr gemma_string() {
        return seq({marker("<|\"|>"), text_without(g_, "<|\"|>", "(string)"), marker("<|\"|>")});
    }

    Expr gemma_value() {
        if (!gemma_value_) {
            gemma_value_ = g_.add_rule("(gemma value)");
            const ojson number = ojson::parse(R"({"type": "number"})");
            g_.define(*gemma_value_, alt({gemma_string(), json_.schema(number, number, "(number)"), literal("true"),
                                          literal("false"), literal("null"), gemma_object(), gemma_list()}));
        }
        return rule(*gemma_value_);
    }

    Expr gemma_key() {
        return seq({bytes(ByteSet::range('a', 'z').add('A', 'Z').add('_', '_')),
                    star(bytes(ByteSet::range('a', 'z').add('A', 'Z').add('0', '9').add('_', '_')))});
    }

    Expr gemma_object() {
        if (!gemma_object_) {
            gemma_object_ = g_.add_rule("(gemma object)");
            const Expr member = seq({gemma_key(), literal(":"), gemma_value()});
            g_.define(*gemma_object_,
                      seq({literal("{"), optional(seq({member, star(seq({literal(","), member}))})), literal("}")}));
        }
        return rule(*gemma_object_);
    }

    Expr gemma_list() {
        return seq({literal("["), optional(seq({gemma_value(), star(seq({literal(","), gemma_value()}))})),
                    literal("]")});
    }

    // A strict tool's arguments: its properties in order, a string's value between <|"|>, an enum of strings as its
    // values, numbers, booleans and null as the schema's JSON, anything else any Gemma value.
    Expr gemma_parameters(const ojson& schema, const std::string& path) {
        const auto member = [&](const std::string& name, const ojson& s, const std::string& at) {
            Expr value = gemma_value();
            if (string_only(s)) {
                value = gemma_string();
            } else if (s.is_object() && s.contains("enum") && s.at("enum").is_array() &&
                       std::all_of(s.at("enum").begin(), s.at("enum").end(), [](const ojson& v) { return v.is_string(); })) {
                std::vector<Expr> options;
                for (const ojson& v : s.at("enum")) {
                    options.push_back(seq({marker("<|\"|>"), literal(v.get<std::string>()), marker("<|\"|>")}));
                }
                value = alt(std::move(options));
            } else if (s.is_object() && s.contains("type") && s.at("type").is_string() &&
                       std::set<std::string>{"integer", "number", "boolean", "null"}.contains(s.at("type"))) {
                value = json_.schema(s, schema, at);
            }
            return seq({literal(name + ":"), value});
        };
        return seq({literal("{"), properties(schema, path, member, literal(",")), literal("}")});
    }

    Grammar& g_;
    JsonValues json_;
    std::string dialect_;
    const Vocabulary& vocabulary_;
    std::optional<RuleId> gemma_value_, gemma_object_;
};

}  // namespace

std::optional<CallsAfterOpener> strict_calls(std::string_view dialect, const RequestJson& tools,
                                             const Vocabulary& vocabulary) {
    const bool any_strict = std::any_of(tools.begin(), tools.end(), [](const RequestJson& t) {
        const RequestJson& f = t.at("function");
        return f.contains("strict") && f.at("strict").is_boolean() && f.at("strict").get<bool>();
    });
    if (!any_strict || !writes_tool_calls(dialect)) return std::nullopt;
    auto g = std::make_shared<Grammar>();
    CallBuilder b(*g, dialect, vocabulary);
    const std::optional<int32_t> opener = b.opener_token();
    if (!opener) return std::nullopt;  // spelled in plain tokens: no point to start holding it from
    std::vector<Expr> options;
    for (size_t i = 0; i < tools.size(); ++i) {
        try {
            options.push_back(b.rest(tools[i].at("function"), i));
        } catch (const GrammarError& e) {
            const std::string param = e.path().empty() ? tool_path(i) : e.path();
            throw InvalidRequest(str_cat("'", param, "' ", e.what()), param);
        }
    }
    const RuleId root = g->add_rule("");
    g->define(root, alt(std::move(options)));
    g->finish(root);
    return CallsAfterOpener{*opener, g};
}

std::shared_ptr<const Grammar> tool_call_grammar(std::string_view dialect, const ToolCalls& calls,
                                                 const Vocabulary& vocabulary) {
    XS_CHECK(calls.tools != nullptr && calls.tools->is_array(), "tool_call_grammar: no tools");
    auto g = std::make_shared<Grammar>();
    CallBuilder b(*g, dialect, vocabulary);
    std::vector<Expr> options;
    for (size_t i = 0; i < calls.tools->size(); ++i) {
        const RequestJson& function = (*calls.tools)[i].at("function");
        const bool named = calls.choice.mode == ToolChoice::Mode::named;
        if (named && function.at("name").get<std::string>() != calls.choice.name) continue;
        try {
            options.push_back(b.call(function, i));
        } catch (const GrammarError& e) {
            const std::string param = e.path().empty() ? tool_path(i) : e.path();
            throw InvalidRequest(str_cat("'", param, "' ", e.what()), param);
        }
    }
    XS_CHECK(!options.empty(), "tool_call_grammar: no tool named '", calls.choice.name, "'");
    const RuleId one = g->add_rule("(call)");
    g->define(one, alt(std::move(options)));
    const bool several = calls.parallel && calls.choice.mode != ToolChoice::Mode::named;
    const Expr list = several ? seq({rule(one), star(seq({b.between(), rule(one)}))}) : rule(one);
    Expr reply = seq({b.lead(), list});
    if (calls.answer) {
        const RuleId answer = g->embed(*calls.answer);
        reply = alt({reply, seq({b.text_with_markers(calls.answer_opening), rule(answer)})});
    }
    const RuleId root = g->add_rule("");
    g->define(root, reply);
    try {
        g->finish(root);
    } catch (const GrammarError& e) {
        throw InvalidRequest(str_cat("'", e.path(), "' ", e.what()), e.path());
    }
    return g;
}

}  // namespace xs::server
