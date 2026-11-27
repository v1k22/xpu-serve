#pragma once

// A context-free grammar over bytes (docs/decisions/0033): rules of alternatives, each a sequence of items, an item a
// byte out of a set or a whole rule. Over bytes, a byte-level token or a part of a UTF-8 character needs no special
// case: a token is the bytes it decodes to. Expressions (Expr) are lowered into rules by Grammar::define; a rule may
// refer to itself, but not before a byte (finish() refuses left recursion).

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "util/error.h"

namespace xs::grammar {

// A grammar the matcher cannot run, or a schema the compiler does not take; `path` names where: a rule, or a keyword of
// the schema (e.g. "properties.age.minimum"; empty for the root). The message says what is wrong with it, to follow its
// name ("is not supported").
class GrammarError : public Error {
public:
    GrammarError(std::string path, const std::string& message) : Error(message), path_(std::move(path)) {}
    const std::string& path() const { return path_; }

private:
    std::string path_;
};

class ByteSet {
public:
    ByteSet() = default;
    static ByteSet of(std::string_view bytes);  // each byte of `bytes`
    static ByteSet range(uint8_t first, uint8_t last);
    ByteSet& add(uint8_t first, uint8_t last);
    ByteSet& add(const ByteSet& other);
    bool has(uint8_t b) const { return (bits_[b >> 6] >> (b & 63)) & 1; }
    bool empty() const { return bits_ == std::array<uint64_t, 4>{}; }
    bool operator==(const ByteSet&) const = default;
    auto operator<=>(const ByteSet&) const = default;

private:
    std::array<uint64_t, 4> bits_{};
};

using RuleId = uint32_t;

// What a rule's alternatives are written as: built with the functions below, lowered by Grammar::define.
class Expr {
public:
    struct Node;
    explicit Expr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
    const Node& node() const { return *node_; }

private:
    std::shared_ptr<const Node> node_;
};

inline constexpr int kUnbounded = -1;

Expr bytes(const ByteSet& set);
Expr literal(std::string_view text);
Expr rule(RuleId id);
Expr seq(std::vector<Expr> parts);
Expr alt(std::vector<Expr> options);
Expr repeat(Expr e, int min, int max);  // `e` min to max times (max kUnbounded: no limit)
inline Expr optional(Expr e) { return repeat(std::move(e), 0, 1); }
inline Expr star(Expr e) { return repeat(std::move(e), 0, kUnbounded); }

class Grammar {
public:
    // One item of an alternative in code(): a byte of sets()[index], a whole rules()[index], or the alternative's end.
    struct Item {
        enum Kind : uint8_t { kBytes, kRule, kEnd };
        Kind kind = kEnd;
        uint32_t index = 0;
    };
    struct Rule {
        std::string name;                   // where it comes from, for errors (a schema path)
        std::vector<uint32_t> alternatives;  // where each starts in code()
        bool defined = false;
    };

    // A rule to define later, so that rules can refer to each other.
    RuleId add_rule(std::string name);
    // The rule's alternatives: the options of `e` when it is an alt(), else `e` alone. Helper rules the expression
    // needs (nested alternatives, repeats) take the rule's name.
    void define(RuleId r, const Expr& e);
    // Checks the grammar and readies it for matching, from `root`: every rule defined; alternatives that can never end
    // (through a rule that derives no finite string) dropped, and the grammar refused when the root is such a rule;
    // a rule that can reach itself before any byte (left recursion) refused. Throws GrammarError naming the rule.
    void finish(RuleId root);

    bool finished() const { return finished_; }
    RuleId root() const { return root_; }
    const std::vector<Item>& code() const { return code_; }
    const std::vector<Rule>& rules() const { return rules_; }
    const std::vector<ByteSet>& sets() const { return sets_; }

private:
    std::vector<Item> lower(const Expr::Node& n, const std::string& name);
    RuleId helper(const Expr::Node& n, const std::string& name);
    void add_alternative(RuleId r, std::vector<Item> items);
    uint32_t set_index(const ByteSet& set);

    std::vector<Item> code_;
    std::vector<Rule> rules_;
    std::vector<ByteSet> sets_;
    std::map<ByteSet, uint32_t> set_ids_;
    std::map<const Expr::Node*, RuleId> helpers_;  // within one define(): the rule lowered for an alt or a repeat
    RuleId root_ = 0;
    bool finished_ = false;
};

}  // namespace xs::grammar
