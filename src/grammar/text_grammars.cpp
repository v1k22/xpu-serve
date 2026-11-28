#include "grammar/text_grammars.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>

#include "util/error.h"

namespace xs::grammar {
namespace {

constexpr char32_t kMaxCodePoint = 0x10FFFF;
// The most a bounded repeat may ask for: each copy is lowered into the grammar.
constexpr int kMostRepeats = 10000;

using Ranges = std::vector<std::pair<char32_t, char32_t>>;

void append_utf8(std::string& out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}

std::string utf8(char32_t c) {
    std::string s;
    append_utf8(s, c);
    return s;
}

// The code point at text[at] (advancing `at`), or nothing when the bytes there are not valid UTF-8.
std::optional<char32_t> decode(std::string_view text, size_t& at) {
    const auto b = static_cast<unsigned char>(text[at]);
    const size_t n = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
    if (n == 0 || at + n > text.size()) return std::nullopt;
    char32_t c = n == 1 ? b : n == 2 ? (b & 0x1F) : n == 3 ? (b & 0x0F) : (b & 0x07);
    for (size_t i = 1; i < n; ++i) {
        const auto cont = static_cast<unsigned char>(text[at + i]);
        if ((cont & 0xC0) != 0x80) return std::nullopt;
        c = (c << 6) | (cont & 0x3F);
    }
    at += n;
    return c;
}

// Sorted, merged, without surrogates.
Ranges normalized(Ranges r) {
    Ranges cut;
    for (const auto& [lo, hi] : r) {
        if (lo > hi) continue;
        if (lo < 0xD800 && hi > 0xDFFF) {
            cut.emplace_back(lo, 0xD7FF);
            cut.emplace_back(0xE000, hi);
        } else if (lo >= 0xD800 && hi <= 0xDFFF) {
            continue;
        } else if (lo >= 0xD800 && lo <= 0xDFFF) {
            cut.emplace_back(0xE000, hi);
        } else if (hi >= 0xD800 && hi <= 0xDFFF) {
            cut.emplace_back(lo, 0xD7FF);
        } else {
            cut.emplace_back(lo, hi);
        }
    }
    std::sort(cut.begin(), cut.end());
    Ranges out;
    for (const auto& [lo, hi] : cut) {
        if (!out.empty() && lo <= out.back().second + 1) {
            out.back().second = std::max(out.back().second, hi);
        } else {
            out.emplace_back(lo, hi);
        }
    }
    return out;
}

Ranges complement(const Ranges& r) {
    Ranges out;
    char32_t next = 0;
    for (const auto& [lo, hi] : normalized(r)) {
        if (lo > next) out.emplace_back(next, lo - 1);
        next = hi + 1;
    }
    if (next <= kMaxCodePoint) out.emplace_back(next, kMaxCodePoint);
    return out;
}

// One range of code points whose UTF-8 encodings are byte sequences of ranges, position by position (Go's and Rust's
// regex compilers split them the same way).
void split_utf8(char32_t lo, char32_t hi, std::vector<std::vector<ByteSet>>& out) {
    if (lo > hi) return;
    for (const char32_t edge : {char32_t{0x7F}, char32_t{0x7FF}, char32_t{0xFFFF}}) {
        if (lo <= edge && hi > edge) {
            split_utf8(lo, edge, out);
            split_utf8(edge + 1, hi, out);
            return;
        }
    }
    if (hi < 0x80) {
        out.push_back({ByteSet::range(static_cast<uint8_t>(lo), static_cast<uint8_t>(hi))});
        return;
    }
    const int n = hi < 0x800 ? 2 : hi < 0x10000 ? 3 : 4;
    for (int i = 1; i < n; ++i) {
        const char32_t m = (char32_t{1} << (6 * i)) - 1;
        if ((lo & ~m) != (hi & ~m)) {
            if ((lo & m) != 0) {
                split_utf8(lo, lo | m, out);
                split_utf8((lo | m) + 1, hi, out);
                return;
            }
            if ((hi & m) != m) {
                split_utf8(lo, (hi & ~m) - 1, out);
                split_utf8(hi & ~m, hi, out);
                return;
            }
        }
    }
    const std::string a = utf8(lo), b = utf8(hi);
    std::vector<ByteSet> seq_sets;
    for (size_t i = 0; i < a.size(); ++i) {
        seq_sets.push_back(ByteSet::range(static_cast<uint8_t>(a[i]), static_cast<uint8_t>(b[i])));
    }
    out.push_back(std::move(seq_sets));
}

[[noreturn]] void refuse(size_t at, const std::string& what) {
    throw GrammarError(str_cat("offset ", at), what);
}

// ECMAScript's \s: its whitespace and line terminators.
const Ranges kSpace = {{'\t', '\r'}, {' ', ' '},       {0xA0, 0xA0},     {0x1680, 0x1680}, {0x2000, 0x200A},
                       {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000}, {0xFEFF, 0xFEFF}};
const Ranges kDigit = {{'0', '9'}};
const Ranges kWord = {{'0', '9'}, {'A', 'Z'}, {'_', '_'}, {'a', 'z'}};

// A regular expression read into expressions; `g` holds the rules alternations and repeats lower into.
class RegexParser {
public:
    explicit RegexParser(std::string_view p) : p_(p) {}

    Expr parse() {
        Expr e = alternation();
        if (at_ < p_.size()) refuse(at_, str_cat("has an unmatched '", p_[at_], "'"));
        return e;
    }

private:
    bool more() const { return at_ < p_.size(); }
    char peek() const { return p_[at_]; }

    Expr alternation() {
        std::vector<Expr> options{concatenation()};
        while (more() && peek() == '|') {
            ++at_;
            options.push_back(concatenation());
        }
        return options.size() == 1 ? options.front() : alt(std::move(options));
    }

    Expr concatenation() {
        std::vector<Expr> parts;
        while (more() && peek() != '|' && peek() != ')') {
            if (peek() == '^' || peek() == '$') {  // the whole text matches: anchors change nothing
                ++at_;
                continue;
            }
            Expr atom_expr = atom();
            parts.push_back(quantified(std::move(atom_expr)));
        }
        return seq(std::move(parts));
    }

    Expr quantified(Expr e) {
        bool quantified_once = false;
        while (more()) {
            const size_t start = at_;
            int min = 0, max = 0;
            const char c = peek();
            if (c == '*') {
                min = 0, max = kUnbounded;
                ++at_;
            } else if (c == '+') {
                min = 1, max = kUnbounded;
                ++at_;
            } else if (c == '?') {
                min = 0, max = 1;
                ++at_;
            } else if (c == '{' && bounds(min, max)) {
            } else {
                break;
            }
            if (quantified_once) refuse(start, "has a quantifier after another (one at a time; group the first)");
            quantified_once = true;
            if (more() && peek() == '?') ++at_;  // lazy: the same text either way
            e = repeat(std::move(e), min, max);
        }
        return e;
    }

    // {n}, {n,}, {n,m} at at_: true and past it when it is one; a `{` that starts none is a literal.
    bool bounds(int& min, int& max) {
        size_t i = at_ + 1;
        const auto number = [&](int& v) {
            const size_t begin = i;
            long long n = 0;
            while (i < p_.size() && p_[i] >= '0' && p_[i] <= '9') n = std::min<long long>(n * 10 + (p_[i++] - '0'), 1LL << 40);
            v = static_cast<int>(std::min<long long>(n, kMostRepeats + 1));
            return i > begin;
        };
        if (!number(min)) return false;
        max = min;
        if (i < p_.size() && p_[i] == ',') {
            ++i;
            max = kUnbounded;
            if (i < p_.size() && p_[i] != '}' && !number(max)) return false;
        }
        if (i >= p_.size() || p_[i] != '}') return false;
        if (min > kMostRepeats || max > kMostRepeats) refuse(at_, str_cat("repeats more than ", kMostRepeats, " times"));
        if (max != kUnbounded && max < min) refuse(at_, "has a repeat whose maximum is below its minimum");
        at_ = i + 1;
        return true;
    }

    Expr atom() {
        const size_t start = at_;
        const char c = peek();
        if (c == '(') return group();
        if (c == '[') return code_points(char_class());
        if (c == '.') {
            ++at_;
            return code_points({{0, kMaxCodePoint}});
        }
        if (c == '\\') {
            Ranges set;
            if (const std::optional<char32_t> one = escape(set, /*in_class=*/false)) return literal(utf8(*one));
            return code_points(set);
        }
        if (c == '*' || c == '+' || c == '?') refuse(start, str_cat("has '", c, "' with nothing to repeat"));
        if (c == '{') {
            int min = 0, max = 0;
            const size_t save = at_;
            if (bounds(min, max)) refuse(save, "has a repeat with nothing to repeat");
        }
        const std::optional<char32_t> cp = decode(p_, at_);
        if (!cp) refuse(start, "is not valid UTF-8");
        return literal(utf8(*cp));
    }

    Expr group() {
        const size_t start = at_++;
        if (more() && peek() == '?') {
            const std::string_view rest = p_.substr(at_);
            if (rest.starts_with("?:")) {
                at_ += 2;
            } else if (rest.starts_with("?=") || rest.starts_with("?!")) {
                refuse(start, "has a lookahead, which is not supported");
            } else if (rest.starts_with("?<=") || rest.starts_with("?<!")) {
                refuse(start, "has a lookbehind, which is not supported");
            } else if (rest.starts_with("?<") || rest.starts_with("?P<")) {
                const size_t close = p_.find('>', at_);
                if (close == std::string_view::npos) refuse(start, "has a group name that never closes");
                at_ = close + 1;
            } else {
                refuse(start, "has a group modifier (an inline flag), which is not supported");
            }
        }
        Expr inner = alternation();
        if (!more() || peek() != ')') refuse(start, "has a group that never closes");
        ++at_;
        return inner;
    }

    Ranges char_class() {
        const size_t start = at_++;
        const bool negated = more() && peek() == '^';
        if (negated) ++at_;
        Ranges set;
        bool first = true;
        while (true) {
            if (!more()) refuse(start, "has a class that never closes");
            if (peek() == ']' && !first) break;
            if (peek() == ']' && first) refuse(start, "has an empty class");
            first = false;
            std::optional<char32_t> lo = member(set);
            if (lo && more() && peek() == '-' && at_ + 1 < p_.size() && p_[at_ + 1] != ']') {
                ++at_;
                const size_t hi_at = at_;
                std::optional<char32_t> hi = member(set);
                if (!hi) refuse(hi_at, "has a class range whose end is a class");
                if (*hi < *lo) refuse(hi_at, "has a class range out of order");
                set.emplace_back(*lo, *hi);
            } else if (lo) {
                set.emplace_back(*lo, *lo);
            }
        }
        ++at_;
        return negated ? complement(set) : set;
    }

    // One member of a class: a code point (returned), or a class escape added to `set` (nothing returned).
    std::optional<char32_t> member(Ranges& set) {
        if (peek() == '\\') return escape(set, /*in_class=*/true);
        const size_t start = at_;
        const std::optional<char32_t> cp = decode(p_, at_);
        if (!cp) refuse(start, "is not valid UTF-8");
        return cp;
    }

    // An escape at at_: one code point, or a class (added to `set`, nothing returned).
    std::optional<char32_t> escape(Ranges& set, bool in_class) {
        const size_t start = at_++;
        if (!more()) refuse(start, "ends with a lone backslash");
        const char c = p_[at_++];
        const auto add = [&](const Ranges& r) {
            set.insert(set.end(), r.begin(), r.end());
            return std::optional<char32_t>{};
        };
        switch (c) {
            case 'd': return add(kDigit);
            case 'D': return add(complement(kDigit));
            case 'w': return add(kWord);
            case 'W': return add(complement(kWord));
            case 's': return add(kSpace);
            case 'S': return add(complement(kSpace));
            case 'n': return U'\n';
            case 't': return U'\t';
            case 'r': return U'\r';
            case 'f': return U'\f';
            case 'v': return U'\v';
            case '0': return U'\0';
            case 'b':
                if (in_class) return U'\b';
                refuse(start, "has a word boundary (\\b), which is not supported");
            case 'B': refuse(start, "has a word boundary (\\B), which is not supported");
            case 'p':
            case 'P': refuse(start, "has a Unicode property class, which is not supported");
            case 'k': refuse(start, "has a backreference, which is not supported");
            case 'x': return hex(start, 2);
            case 'u':
                if (more() && peek() == '{') {
                    const size_t close = p_.find('}', at_);
                    if (close == std::string_view::npos) refuse(start, "has a \\u{ that never closes");
                    const std::string digits(p_.substr(at_ + 1, close - at_ - 1));
                    at_ = close + 1;
                    return checked(start, digits);
                }
                return hex(start, 4);
            case 'c':
                if (more() && std::isalpha(static_cast<unsigned char>(peek()))) {
                    return static_cast<char32_t>(p_[at_++] % 32);
                }
                refuse(start, "has a \\c without a letter");
            default:
                if (c >= '1' && c <= '9') refuse(start, "has a backreference, which is not supported");
                --at_;  // any other escaped character is itself (punctuation, or an unknown escape)
                const std::optional<char32_t> cp = decode(p_, at_);
                if (!cp) refuse(start, "is not valid UTF-8");
                return cp;
        }
    }

    char32_t hex(size_t start, size_t digits) {
        if (at_ + digits > p_.size()) refuse(start, "has a short hexadecimal escape");
        const std::string text(p_.substr(at_, digits));
        at_ += digits;
        return checked(start, text);
    }

    static char32_t checked(size_t start, const std::string& digits) {
        if (digits.empty() || digits.size() > 6 ||
            !std::all_of(digits.begin(), digits.end(), [](char d) { return std::isxdigit(static_cast<unsigned char>(d)); })) {
            refuse(start, "has an invalid hexadecimal escape");
        }
        const auto v = static_cast<char32_t>(std::stoul(digits, nullptr, 16));
        if (v > kMaxCodePoint) refuse(start, "has a code point past U+10FFFF");
        return v;
    }

    std::string_view p_;
    size_t at_ = 0;
};

// xgrammar's EBNF: a lexer over the text and a parser of its rules.
class EbnfParser {
public:
    explicit EbnfParser(std::string_view text) : t_(text) {}

    std::shared_ptr<const Grammar> parse() {
        auto g = std::make_shared<Grammar>();
        g_ = g.get();
        skip();
        while (at_ < t_.size()) {
            const size_t start = at_;
            const std::string name = identifier();
            if (name.empty()) refuse_here("expects a rule's name");
            skip_inline();
            if (at_ < t_.size() && t_[at_] == '[') refuse_here("has rule options (xgrammar's [...] after a name), which are not supported");
            if (!t_.substr(at_).starts_with("::=")) refuse_here(str_cat("expects '::=' after the rule name '", name, "'"));
            at_ += 3;
            if (bodies_.contains(name)) refuse_at(start, str_cat("defines the rule '", name, "' twice"));
            const RuleId id = rule_id(name);
            defined_.emplace(name, id);
            const Expr body = alternation();
            bodies_.emplace(name, true);
            g_->define(id, body);
            skip();
        }
        for (const auto& [name, id] : ids_) {
            if (!bodies_.contains(name)) throw GrammarError(name, str_cat("is used but never defined"));
        }
        const auto root = defined_.find("root");
        if (root == defined_.end()) throw GrammarError("root", "is required: the rule the output matches");
        g->finish(root->second);
        return g;
    }

private:
    [[noreturn]] void refuse_at(size_t at, const std::string& what) const {
        size_t line = 1, column = 1;
        for (size_t i = 0; i < at && i < t_.size(); ++i) {
            if (t_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw GrammarError(str_cat("line ", line, ", column ", column), what);
    }
    [[noreturn]] void refuse_here(const std::string& what) const { refuse_at(at_, what); }

    RuleId rule_id(const std::string& name) {
        if (const auto it = ids_.find(name); it != ids_.end()) return it->second;
        const RuleId id = g_->add_rule(name);
        ids_.emplace(name, id);
        return id;
    }

    static bool name_char(char c, bool first) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_' ||
               (!first && (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '.'));
    }
    std::string identifier() {
        const size_t start = at_;
        while (at_ < t_.size() && name_char(t_[at_], at_ == start)) ++at_;
        return std::string(t_.substr(start, at_ - start));
    }

    // Spaces, newlines and comments.
    void skip() {
        while (at_ < t_.size()) {
            if (std::isspace(static_cast<unsigned char>(t_[at_]))) {
                ++at_;
            } else if (t_[at_] == '#') {
                while (at_ < t_.size() && t_[at_] != '\n') ++at_;
            } else {
                break;
            }
        }
    }
    void skip_inline() {
        while (at_ < t_.size() && (t_[at_] == ' ' || t_[at_] == '\t')) ++at_;
    }

    // At the start of the next rule: a name first on its line, then ::= (rules end where the next begins).
    bool at_rule_start() const {
        size_t back = at_;
        while (back > 0 && (t_[back - 1] == ' ' || t_[back - 1] == '\t')) --back;
        if (back > 0 && t_[back - 1] != '\n') return false;
        size_t i = at_;
        while (i < t_.size() && name_char(t_[i], i == at_)) ++i;
        if (i == at_) return false;
        while (i < t_.size() && (t_[i] == ' ' || t_[i] == '\t')) ++i;
        return t_.substr(i).starts_with("::=") || (i < t_.size() && t_[i] == '[');
    }

    Expr alternation() {
        std::vector<Expr> options{sequence()};
        while (at_ < t_.size() && t_[at_] == '|') {
            ++at_;
            options.push_back(sequence());
        }
        return options.size() == 1 ? options.front() : alt(std::move(options));
    }

    Expr sequence() {
        std::vector<Expr> parts;
        for (;;) {
            skip();
            if (at_ >= t_.size() || t_[at_] == '|' || t_[at_] == ')' || at_rule_start()) break;
            parts.push_back(quantified(primary()));
        }
        return seq(std::move(parts));
    }

    Expr quantified(Expr e) {
        skip_inline();
        if (at_ >= t_.size()) return e;
        const char c = t_[at_];
        if (c == '*' || c == '+' || c == '?') {
            ++at_;
            e = repeat(std::move(e), c == '+' ? 1 : 0, c == '?' ? 1 : kUnbounded);
        } else if (c == '{') {
            const size_t start = at_++;
            const auto number = [&]() -> int {
                skip_inline();
                const size_t begin = at_;
                long long n = 0;
                while (at_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[at_]))) {
                    n = std::min<long long>(n * 10 + (t_[at_++] - '0'), 1LL << 40);
                }
                if (at_ == begin) refuse_here("expects a number in a repeat");
                if (n > kMostRepeats) refuse_at(start, str_cat("repeats more than ", kMostRepeats, " times"));
                skip_inline();
                return static_cast<int>(n);
            };
            const int min = number();
            int max = min;
            if (at_ < t_.size() && t_[at_] == ',') {
                ++at_;
                skip_inline();
                max = at_ < t_.size() && t_[at_] == '}' ? kUnbounded : number();
            }
            if (at_ >= t_.size() || t_[at_] != '}') refuse_here("expects '}' to close a repeat");
            ++at_;
            if (max != kUnbounded && max < min) refuse_at(start, "has a repeat whose maximum is below its minimum");
            e = repeat(std::move(e), min, max);
        } else {
            return e;
        }
        skip_inline();
        if (at_ < t_.size() && (t_[at_] == '*' || t_[at_] == '+' || t_[at_] == '?' || t_[at_] == '{')) {
            refuse_here("has a quantifier after another (one at a time; group the first)");
        }
        return e;
    }

    Expr primary() {
        const char c = t_[at_];
        if (c == '"') return string_literal();
        if (c == '\'') refuse_here("has a literal in single quotes: write it in double quotes");
        if (c == '[') return code_points(char_class());
        if (c == '(') {
            const size_t start = at_++;
            skip();
            if (at_ < t_.size() && t_[at_] == '=') refuse_at(start, "has a lookahead ((= ...)), which is not supported");
            Expr inner = alternation();
            skip();
            if (at_ >= t_.size() || t_[at_] != ')') refuse_at(start, "has a group that never closes");
            ++at_;
            return inner;
        }
        if (name_char(c, true)) {
            const size_t start = at_;
            const std::string name = identifier();
            if (at_ < t_.size() && t_[at_] == '(') {
                refuse_at(start, str_cat("calls '", name, "(...)', an xgrammar macro, which is not supported"));
            }
            return rule(rule_id(name));
        }
        refuse_here(str_cat("has an unexpected '", c, "'"));
    }

    Expr string_literal() {
        const size_t start = at_++;
        std::string bytes;
        for (;;) {
            if (at_ >= t_.size() || t_[at_] == '\n') refuse_at(start, "has a literal that never closes");
            if (t_[at_] == '"') break;
            if (t_[at_] == '\\') {
                append_utf8(bytes, escape());
            } else {
                const size_t c_at = at_;
                const std::optional<char32_t> cp = decode(t_, at_);
                if (!cp) refuse_at(c_at, "is not valid UTF-8");
                append_utf8(bytes, *cp);
            }
        }
        ++at_;
        return literal(bytes);
    }

    // A backslash escape at at_ (in a literal or a class): the code point it stands for.
    char32_t escape() {
        const size_t start = at_++;
        if (at_ >= t_.size()) refuse_at(start, "ends with a lone backslash");
        const char c = t_[at_++];
        switch (c) {
            case 'n': return U'\n';
            case 'r': return U'\r';
            case 't': return U'\t';
            case 'b': return U'\b';
            case 'f': return U'\f';
            case 'a': return U'\a';
            case 'v': return U'\v';
            case '0': return U'\0';
            case 'e': return 0x1B;
            case 'x': {
                const size_t begin = at_;
                while (at_ < t_.size() && std::isxdigit(static_cast<unsigned char>(t_[at_]))) ++at_;
                return hex(start, t_.substr(begin, at_ - begin));
            }
            case 'u':
            case 'U': {
                const size_t n = c == 'u' ? 4 : 8;
                if (at_ + n > t_.size()) refuse_at(start, "has a short \\u escape");
                const std::string_view digits = t_.substr(at_, n);
                at_ += n;
                return hex(start, digits);
            }
            case 'd':
            case 'w':
            case 's':
            case 'D':
            case 'W':
            case 'S': refuse_at(start, "has a class escape (\\d, \\w, \\s), which EBNF does not take: write the class");
            default:
                if (std::string_view("\"\\/'?^$.*+()[]{}|-").find(c) != std::string_view::npos) return static_cast<char32_t>(c);
                refuse_at(start, str_cat("has an unknown escape '\\", c, "'"));
        }
    }

    char32_t hex(size_t start, std::string_view digits) const {
        if (digits.empty() || digits.size() > 8 ||
            !std::all_of(digits.begin(), digits.end(), [](char d) { return std::isxdigit(static_cast<unsigned char>(d)); })) {
            refuse_at(start, "has an invalid hexadecimal escape");
        }
        const auto v = static_cast<char32_t>(std::stoull(std::string(digits), nullptr, 16));
        if (v > kMaxCodePoint) refuse_at(start, "has a code point past U+10FFFF");
        return v;
    }

    Ranges char_class() {
        const size_t start = at_++;
        const bool negated = at_ < t_.size() && t_[at_] == '^';
        if (negated) ++at_;
        Ranges set;
        while (true) {
            if (at_ >= t_.size()) refuse_at(start, "has a class that never closes");
            if (t_[at_] == ']') break;
            const char32_t lo = member();
            if (at_ + 1 < t_.size() && t_[at_] == '-' && t_[at_ + 1] != ']') {
                ++at_;
                const size_t hi_at = at_;
                const char32_t hi = member();
                if (hi < lo) refuse_at(hi_at, "has a class range out of order");
                set.emplace_back(lo, hi);
            } else {
                set.emplace_back(lo, lo);
            }
        }
        ++at_;
        if (set.empty() && !negated) refuse_at(start, "has an empty class");
        return negated ? complement(set) : set;
    }

    char32_t member() {
        if (t_[at_] == '\\') return escape();
        const size_t start = at_;
        const std::optional<char32_t> cp = decode(t_, at_);
        if (!cp) refuse_at(start, "is not valid UTF-8");
        return *cp;
    }

    std::string_view t_;
    size_t at_ = 0;
    Grammar* g_ = nullptr;
    std::map<std::string, RuleId> ids_;      // every rule named so far
    std::map<std::string, RuleId> defined_;  // the ones with a body
    std::map<std::string, bool> bodies_;
};

}  // namespace

Expr code_points(const std::vector<std::pair<char32_t, char32_t>>& ranges) {
    std::vector<std::vector<ByteSet>> sequences;
    for (const auto& [lo, hi] : normalized(ranges)) split_utf8(lo, std::min(hi, kMaxCodePoint), sequences);
    XS_CHECK(!sequences.empty(), "grammar: a class of no code points");
    std::vector<Expr> options;
    for (const std::vector<ByteSet>& s : sequences) {
        std::vector<Expr> parts;
        for (const ByteSet& b : s) parts.push_back(bytes(b));
        options.push_back(seq(std::move(parts)));
    }
    return options.size() == 1 ? options.front() : alt(std::move(options));
}

std::shared_ptr<const Grammar> regex_grammar(std::string_view pattern) {
    if (pattern.find('\0') != std::string_view::npos) throw GrammarError("", "must not hold a NUL character");
    RegexParser parser(pattern);
    const Expr e = parser.parse();
    auto g = std::make_shared<Grammar>();
    const RuleId root = g->add_rule("");
    g->define(root, e);
    g->finish(root);
    return g;
}

std::shared_ptr<const Grammar> ebnf_grammar(std::string_view text) {
    EbnfParser parser(text);
    return parser.parse();
}

std::shared_ptr<const Grammar> choice_grammar(const std::vector<std::string>& options) {
    if (options.empty()) throw GrammarError("", "must be a non-empty list of strings");
    std::vector<Expr> literals;
    for (const std::string& o : options) literals.push_back(literal(o));
    auto g = std::make_shared<Grammar>();
    const RuleId root = g->add_rule("");
    g->define(root, alt(std::move(literals)));
    g->finish(root);
    return g;
}

}  // namespace xs::grammar
