#include "grammar/json_schema.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace xs::grammar {
namespace {

using ojson = nlohmann::ordered_json;

std::string join(const std::string& path, std::string_view key) {
    return path.empty() ? std::string(key) : str_cat(path, ".", key);
}

std::string at(const std::string& path, std::string_view key, size_t i) {
    return str_cat(join(path, key), "[", i, "]");
}

ByteSet byte_range(unsigned first, unsigned last) {
    return ByteSet::range(static_cast<uint8_t>(first), static_cast<uint8_t>(last));
}

// JSON's own rules in a grammar, each made the first time it is asked for.
class JsonRules {
public:
    explicit JsonRules(Grammar& g) : g_(g) {}

    Expr ws() {
        return made(ws_, "(whitespace)", [] {
            return alt({seq({}), literal(" "),
                        seq({repeat(literal("\n"), 1, 2), repeat(bytes(ByteSet::of(" \t")), 0, 20)})});
        });
    }

    Expr string() {
        return made(string_, "(string)", [] {
            const Expr cont = bytes(byte_range(0x80, 0xBF));
            const Expr hex = bytes(byte_range('0', '9').add('a', 'f').add('A', 'F'));
            const Expr character = alt({
                bytes(byte_range(0x20, 0x21).add(0x23, 0x5B).add(0x5D, 0x7F)),  // ASCII but " \ and controls
                seq({bytes(byte_range(0xC2, 0xDF)), cont}),
                seq({literal("\xE0"), bytes(byte_range(0xA0, 0xBF)), cont}),
                seq({bytes(byte_range(0xE1, 0xEC).add(0xEE, 0xEF)), cont, cont}),
                seq({literal("\xED"), bytes(byte_range(0x80, 0x9F)), cont}),  // not a surrogate
                seq({literal("\xF0"), bytes(byte_range(0x90, 0xBF)), cont, cont}),
                seq({bytes(byte_range(0xF1, 0xF3)), cont, cont, cont}),
                seq({literal("\xF4"), bytes(byte_range(0x80, 0x8F)), cont, cont}),
                seq({literal("\\"), alt({bytes(ByteSet::of("\"\\/bfnrt")), seq({literal("u"), hex, hex, hex, hex})})}),
            });
            return seq({literal("\""), star(character), literal("\"")});
        });
    }

    Expr integer() {
        return made(integer_, "(integer)", [] {
            const Expr natural = seq({bytes(byte_range('1', '9')), star(digit())});
            return seq({optional(literal("-")), alt({literal("0"), natural})});
        });
    }

    Expr number() {
        return made(number_, "(number)", [this] {
            return seq({integer(), optional(seq({literal("."), repeat(digit(), 1, kUnbounded)})),
                        optional(seq({bytes(ByteSet::of("eE")), optional(bytes(ByteSet::of("+-"))),
                                      repeat(digit(), 1, kUnbounded)}))});
        });
    }

    Expr value() {
        return made(value_, "(value)", [this] {
            return alt({object(), array(), string(), number(), literal("true"), literal("false"), literal("null")});
        });
    }

    Expr object() {
        return made(object_, "(object)", [this] {
            const Expr member = seq({string(), ws(), literal(":"), ws(), value(), ws()});
            return seq({literal("{"), ws(), optional(seq({member, star(seq({literal(","), ws(), member}))})),
                        literal("}")});
        });
    }

    // `[ item, item, ... ]`.
    Expr array_of(const Expr& item) {
        return seq({literal("["), ws(),
                    optional(seq({item, ws(), star(seq({literal(","), ws(), item, ws()}))})), literal("]")});
    }

private:
    static Expr digit() { return bytes(byte_range('0', '9')); }

    Expr array() {
        return made(array_, "(array)", [this] { return array_of(value()); });
    }

    // The rule in `slot`, made and defined by `body` the first time: the slot is set before the body runs, so rules
    // can refer to each other.
    template <typename Body>
    Expr made(std::optional<RuleId>& slot, const char* name, Body body) {
        if (!slot) {
            slot = g_.add_rule(name);
            g_.define(*slot, body());
        }
        return rule(*slot);
    }

    Grammar& g_;
    std::optional<RuleId> ws_, string_, integer_, number_, value_, object_, array_;
};

const std::set<std::string, std::less<>> kAnnotations = {
    "title", "description", "$comment", "examples", "default", "deprecated", "readOnly", "writeOnly", "$schema"};
const std::set<std::string, std::less<>> kKeywords = {
    "type",  "nullable", "properties", "required", "additionalProperties", "items", "enum", "const",
    "anyOf", "$ref",     "$defs",      "definitions"};
const std::set<std::string, std::less<>> kTypes = {"object", "array",   "string", "number",
                                                   "integer", "boolean", "null"};

bool is_type(const ojson& v, std::string_view type) {
    if (type == "null") return v.is_null();
    if (type == "boolean") return v.is_boolean();
    if (type == "string") return v.is_string();
    if (type == "number") return v.is_number();
    if (type == "integer") {
        return v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>());
    }
    if (type == "object") return v.is_object();
    return v.is_array();
}

class SchemaCompiler {
public:
    SchemaCompiler(Grammar& g, const ojson& root) : g_(g), json_(g), root_(root) {}

    RuleId document() {
        const RuleId doc = g_.add_rule("");
        const RuleId root = g_.add_rule("");
        refs_.emplace("", root);
        g_.define(root, value(root_, ""));
        g_.define(doc, seq({json_.ws(), rule(root)}));
        return doc;
    }

private:
    Expr value(const ojson& s, const std::string& path) {
        if (s.is_boolean()) {
            if (s.get<bool>()) return json_.value();
            throw GrammarError(path, "is false, a schema no value matches");
        }
        if (!s.is_object()) throw GrammarError(path, "must be a schema: an object or a boolean");
        for (const auto& [key, v] : s.items()) {
            if (!kKeywords.contains(key) && !kAnnotations.contains(key)) {
                throw GrammarError(join(path, key), "is not supported");
            }
        }
        for (const char* defs : {"$defs", "definitions"}) {
            if (s.contains(defs) && !s.at(defs).is_object()) throw GrammarError(join(path, defs), "must be an object");
        }
        if (s.contains("$ref")) {
            alone(s, path, "$ref", {});
            return ref(s.at("$ref"), join(path, "$ref"));
        }
        if (s.contains("anyOf")) {
            alone(s, path, "anyOf", {});
            const ojson& options = s.at("anyOf");
            if (!options.is_array() || options.empty()) {
                throw GrammarError(join(path, "anyOf"), "must be a non-empty array of schemas");
            }
            std::vector<Expr> out;
            for (size_t i = 0; i < options.size(); ++i) out.push_back(value(options[i], at(path, "anyOf", i)));
            return alt(std::move(out));
        }
        const std::vector<std::string> types = types_of(s, path);
        if (s.contains("const") || s.contains("enum")) return values(s, path, types);
        if (types.empty()) return json_.value();
        const auto has = [&](const char* type) { return std::find(types.begin(), types.end(), type) != types.end(); };
        for (const char* key : {"properties", "required", "additionalProperties"}) {
            if (s.contains(key) && !has("object")) throw GrammarError(join(path, key), "applies to type object only");
        }
        if (s.contains("items") && !has("array")) throw GrammarError(join(path, "items"), "applies to type array only");
        std::vector<Expr> out;
        for (const std::string& type : types) {
            if (type == "object") {
                out.push_back(object(s, path));
            } else if (type == "array") {
                out.push_back(s.contains("items") ? json_.array_of(named(join(path, "items"), s.at("items")))
                                                  : json_.array_of(json_.value()));
            } else if (type == "string") {
                out.push_back(json_.string());
            } else if (type == "number") {
                out.push_back(json_.number());
            } else if (type == "integer") {
                out.push_back(json_.integer());
            } else if (type == "boolean") {
                out.push_back(alt({literal("true"), literal("false")}));
            } else {
                out.push_back(literal("null"));
            }
        }
        return alt(std::move(out));
    }

    // Every keyword of `s` but `key`, `also`, annotations and definitions is refused: what would apply beside it.
    static void alone(const ojson& s, const std::string& path, const char* key, std::set<std::string> also) {
        for (const auto& [k, v] : s.items()) {
            const bool beside = k == key || also.contains(k) || kAnnotations.contains(k);
            if (beside || k == "$defs" || k == "definitions") continue;
            throw GrammarError(join(path, k), str_cat("is not supported beside '", key, "'"));
        }
    }

    // The types `s` allows, from "type" and "nullable", or from its keywords when it names none (empty: any value).
    static std::vector<std::string> types_of(const ojson& s, const std::string& path) {
        std::vector<std::string> types;
        const std::string where = join(path, "type");
        if (s.contains("type")) {
            const ojson& t = s.at("type");
            const std::vector<ojson> listed = t.is_array() ? t.get<std::vector<ojson>>() : std::vector<ojson>{t};
            if (listed.empty()) throw GrammarError(where, "must name a type or a list of them");
            for (const ojson& name : listed) {
                if (!name.is_string() || !kTypes.contains(name.get<std::string>())) {
                    throw GrammarError(where, str_cat("names ", name.dump(), ", not a JSON Schema type (object, array, "
                                                      "string, number, integer, boolean, null)"));
                }
                if (std::find(types.begin(), types.end(), name.get<std::string>()) == types.end()) {
                    types.push_back(name.get<std::string>());
                }
            }
        } else if (!s.contains("const") && !s.contains("enum")) {
            if (s.contains("properties") || s.contains("required") || s.contains("additionalProperties")) {
                types.push_back("object");
            }
            if (s.contains("items")) types.push_back("array");
        }
        if (s.contains("nullable")) {
            if (!s.at("nullable").is_boolean()) throw GrammarError(join(path, "nullable"), "must be a boolean");
            if (s.at("nullable").get<bool>() && !types.empty() &&
                std::find(types.begin(), types.end(), "null") == types.end()) {
                types.push_back("null");
            }
        }
        return types;
    }

    // const or enum: the values as compact JSON, each of a type the schema allows.
    static Expr values(const ojson& s, const std::string& path, const std::vector<std::string>& types) {
        const bool one = s.contains("const");
        const char* key = one ? "const" : "enum";
        alone(s, path, key, {"type", "nullable"});
        const ojson& v = s.at(key);
        if (!one && (!v.is_array() || v.empty())) throw GrammarError(join(path, key), "must be a non-empty array");
        std::vector<Expr> out;
        for (size_t i = 0; i < (one ? 1 : v.size()); ++i) {
            const ojson& value = one ? v : v[i];
            const bool typed = types.empty() || std::any_of(types.begin(), types.end(), [&](const std::string& t) {
                                   return is_type(value, t);
                               });
            if (!typed) throw GrammarError(one ? join(path, key) : at(path, key, i), "is not of the schema's type");
            out.push_back(literal(value.dump()));
        }
        if (s.contains("nullable") && s.at("nullable").get<bool>()) out.push_back(literal("null"));
        return alt(std::move(out));
    }

    // A rule of its own for the value `s` at `path`: named by its path, for the grammar's errors.
    Expr named(const std::string& path, const ojson& s) {
        Expr e = value(s, path);
        const RuleId r = g_.add_rule(path);
        g_.define(r, e);
        return rule(r);
    }

    Expr object(const ojson& s, const std::string& path) {
        const ojson* properties = s.contains("properties") ? &s.at("properties") : nullptr;
        if (properties != nullptr && !properties->is_object()) {
            throw GrammarError(join(path, "properties"), "must be an object of schemas");
        }
        if (s.contains("additionalProperties")) {
            const ojson& extra = s.at("additionalProperties");
            if (!extra.is_boolean() || extra.get<bool>()) {
                throw GrammarError(join(path, "additionalProperties"), "is supported as false only");
            }
        }
        std::set<std::string> required;
        if (s.contains("required")) {
            const ojson& names = s.at("required");
            if (!names.is_array()) throw GrammarError(join(path, "required"), "must be an array of property names");
            for (size_t i = 0; i < names.size(); ++i) {
                const bool known = names[i].is_string() && properties != nullptr &&
                                   properties->contains(names[i].get<std::string>());
                if (!known) throw GrammarError(at(path, "required", i), "names no property of 'properties'");
                required.insert(names[i].get<std::string>());
            }
        }
        if (properties == nullptr && !s.contains("additionalProperties")) return json_.object();

        // first[i]: the members from property i on, none written yet; later[i]: the same after one was (each then
        // after a comma). An optional property may be left out.
        std::vector<std::pair<std::string, const ojson*>> members;
        if (properties != nullptr) {
            for (const auto& [name, schema] : properties->items()) members.emplace_back(name, &schema);
        }
        const size_t n = members.size();
        std::vector<RuleId> first(n + 1), later(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            first[i] = g_.add_rule(path);
            later[i] = g_.add_rule(path);
        }
        g_.define(first[n], seq({}));
        g_.define(later[n], seq({}));
        for (size_t i = n; i-- > 0;) {
            const auto& [name, schema] = members[i];
            const Expr member = seq({literal(ojson(name).dump()), json_.ws(), literal(":"), json_.ws(),
                                     named(join(join(path, "properties"), name), *schema), json_.ws()});
            const Expr here = seq({member, rule(later[i + 1])});
            const Expr after = seq({literal(","), json_.ws(), member, rule(later[i + 1])});
            if (required.contains(name)) {
                g_.define(first[i], here);
                g_.define(later[i], after);
            } else {
                g_.define(first[i], alt({here, rule(first[i + 1])}));
                g_.define(later[i], alt({after, rule(later[i + 1])}));
            }
        }
        return seq({literal("{"), json_.ws(), rule(first[0]), literal("}")});
    }

    // A $ref to a part of this schema: its rule, made once per JSON pointer, so a part can refer to itself.
    Expr ref(const ojson& v, const std::string& path) {
        if (!v.is_string()) throw GrammarError(path, "must be a string");
        const std::string& target = v.get_ref<const std::string&>();
        if (!target.starts_with("#")) {
            throw GrammarError(path, str_cat("refers to '", target, "', outside this schema: only references into it "
                                            "(#/...) are supported"));
        }
        const std::string pointer = target.substr(1);
        if (const auto it = refs_.find(pointer); it != refs_.end()) return rule(it->second);
        const ojson* part = nullptr;
        try {
            const ojson::json_pointer p(pointer);
            if (root_.contains(p)) part = &root_.at(p);
        } catch (const nlohmann::json::exception&) {
        }
        if (part == nullptr) throw GrammarError(path, str_cat("refers to '", target, "', which is not in the schema"));
        std::string name;  // the pointer as a path: "#/$defs/node" -> "$defs.node"
        for (size_t at = 1; at <= pointer.size();) {
            const size_t end = std::min(pointer.find('/', at), pointer.size());
            std::string segment = pointer.substr(at, end - at);
            for (size_t k = 0; (k = segment.find("~1", k)) != std::string::npos;) segment.replace(k, 2, "/");
            for (size_t k = 0; (k = segment.find("~0", k)) != std::string::npos;) segment.replace(k, 2, "~");
            name = join(name, segment);
            at = end + 1;
        }
        const RuleId r = g_.add_rule(name);
        refs_.emplace(pointer, r);
        g_.define(r, value(*part, name));
        return rule(r);
    }

    Grammar& g_;
    JsonRules json_;
    const ojson& root_;
    std::map<std::string, RuleId> refs_;  // by JSON pointer
};

}  // namespace

std::shared_ptr<const Grammar> json_object_grammar() {
    auto g = std::make_shared<Grammar>();
    JsonRules json(*g);
    const RuleId doc = g->add_rule("");
    g->define(doc, seq({json.ws(), json.object()}));
    g->finish(doc);
    return g;
}

std::shared_ptr<const Grammar> json_schema_grammar(const nlohmann::ordered_json& schema) {
    auto g = std::make_shared<Grammar>();
    SchemaCompiler compiler(*g, schema);
    g->finish(compiler.document());
    return g;
}

}  // namespace xs::grammar
