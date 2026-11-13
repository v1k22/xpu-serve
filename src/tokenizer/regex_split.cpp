#include "tokenizer/regex_split.h"

#include <cstdint>
#include <string>

#include "tokenizer/unicode.h"
#include "tokenizer/utf8.h"

namespace xs::tok {
namespace {

enum CharFlag : uint8_t {
    kLetter = 1,
    kMark = 2,
    kNumber = 4,
    kSpace = 8,     // \s: Unicode White_Space
    kNewline = 16,  // \r or \n
};

struct CharInfo {
    size_t offset;  // byte offset in the text
    char32_t c;
    uint8_t flags;
};

uint8_t flags_of(char32_t c) {
    if (c == '\r' || c == '\n') return kSpace | kNewline;
    if (utf8::is_whitespace(c)) return kSpace;
    switch (unicode::category(c)) {
        case unicode::Category::kLetter:
            return kLetter;
        case unicode::Category::kMark:
            return kMark;
        case unicode::Category::kNumber:
            return kNumber;
        case unicode::Category::kOther:
            return 0;
    }
    return 0;
}

// The building blocks of the supported patterns.
enum class AltKind {
    kContractions,        // (?i:'s|'t|'re|'ve|'m|'ll|'d), or without (?i)
    kLetters,             // [^\r\n\p{L}\p{N}]?<letters>+
    kSpaceThenRun,        //  ?<class>+, the class the flags in `letters` (or, with `negated`, its complement)
    kNumbers,             // \p{N}{1,max}
    kPunctuation,         //  ?[^\s\p{L}\p{N}<\p{M} with marks>]+[\r\n]*
    kSpacesThenNewlines,  // \s*[\r\n]+
    kSpacesNotBeforeNonSpace,  // \s+(?!\S)
    kSpaces,                   // \s+
};

struct Alternative {
    AltKind kind;
    uint8_t letters = kLetter;  // kLetters: the body class; kPunctuation: also excluded; kSpaceThenRun: the class
    size_t max_count = SIZE_MAX;  // kNumbers
    bool negated = false;         // kSpaceThenRun
    bool ignore_case = true;      // kContractions
};

// Simple case folding restricted to the contraction letters: ASCII, and U+017F LONG S which folds to 's'.
char32_t fold(char32_t c) {
    if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
    if (c == 0x17F) return 's';
    return c;
}

class AlternativesSplitter final : public RegexSplitter {
public:
    explicit AlternativesSplitter(std::vector<Alternative> alternatives) : alternatives_(std::move(alternatives)) {}

    std::vector<std::pair<size_t, size_t>> find_matches(std::string_view text) const override {
        std::vector<CharInfo> chars;
        chars.reserve(text.size() + 1);
        for (size_t pos = 0; pos < text.size();) {
            const auto b = static_cast<uint8_t>(text[pos]);
            const char32_t c = b < 0x80 ? b : utf8::decode_at(text, pos);
            chars.push_back({pos, c, flags_of(c)});
            pos += b < 0x80 ? 1 : utf8::char_length(text, pos);
        }
        const size_t n = chars.size();
        chars.push_back({text.size(), 0, 0});  // sentinel: offset of the end

        std::vector<std::pair<size_t, size_t>> matches;
        for (size_t i = 0; i < n;) {
            size_t end = 0;
            for (const Alternative& alt : alternatives_) {
                end = match_at(alt, chars, n, i);
                if (end > i) break;
            }
            if (end > i) {
                matches.emplace_back(chars[i].offset, chars[end].offset);
                i = end;
            } else {
                ++i;
            }
        }
        return matches;
    }

private:
    // End (char index) of the match of `alt` starting at char i, or i when it does not match.
    static size_t match_at(const Alternative& alt, const std::vector<CharInfo>& s, size_t n, size_t i) {
        auto run = [&](size_t j, auto pred) {
            while (j < n && pred(s[j].flags)) ++j;
            return j;
        };
        switch (alt.kind) {
            case AltKind::kContractions: {
                if (s[i].c != '\'' || i + 1 >= n) return i;
                const char32_t a = alt.ignore_case ? fold(s[i + 1].c) : s[i + 1].c;
                if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
                if (i + 2 >= n) return i;
                const char32_t b = alt.ignore_case ? fold(s[i + 2].c) : s[i + 2].c;
                if ((a == 'r' || a == 'v') && b == 'e') return i + 3;
                if (a == 'l' && b == 'l') return i + 3;
                return i;
            }
            case AltKind::kLetters: {
                auto body = [&](uint8_t f) { return (f & alt.letters) != 0; };
                auto prefix = [](uint8_t f) { return (f & (kNewline | kLetter | kNumber)) == 0; };
                if (i + 1 < n && prefix(s[i].flags) && body(s[i + 1].flags)) return run(i + 1, body);
                return body(s[i].flags) ? run(i, body) : i;
            }
            case AltKind::kSpaceThenRun: {
                auto in_class = [&](uint8_t f) { return ((f & alt.letters) != 0) != alt.negated; };
                if (s[i].c == ' ' && i + 1 < n && in_class(s[i + 1].flags)) return run(i + 1, in_class);
                return in_class(s[i].flags) ? run(i, in_class) : i;
            }
            case AltKind::kNumbers: {
                size_t j = i;
                while (j < n && j - i < alt.max_count && (s[j].flags & kNumber) != 0) ++j;
                return j;
            }
            case AltKind::kPunctuation: {
                const uint8_t excluded = kSpace | kNumber | alt.letters;
                auto punct = [&](uint8_t f) { return (f & excluded) == 0; };
                size_t j = i;
                if (s[i].c == ' ' && i + 1 < n && punct(s[i + 1].flags)) ++j;
                if (j >= n || !punct(s[j].flags)) return i;
                j = run(j, punct);
                return run(j, [](uint8_t f) { return (f & kNewline) != 0; });
            }
            case AltKind::kSpacesThenNewlines: {
                // \s* gives back characters until [\r\n]+ can match: the match ends after the run's last newline.
                const size_t k = run(i, [](uint8_t f) { return (f & kSpace) != 0; });
                for (size_t j = k; j > i; --j) {
                    if ((s[j - 1].flags & kNewline) != 0) return j;
                }
                return i;
            }
            case AltKind::kSpacesNotBeforeNonSpace: {
                const size_t k = run(i, [](uint8_t f) { return (f & kSpace) != 0; });
                if (k == n) return k;
                return k - i >= 2 ? k - 1 : i;  // give back one space so that a space follows
            }
            case AltKind::kSpaces:
                return run(i, [](uint8_t f) { return (f & kSpace) != 0; });
        }
        return i;
    }

    std::vector<Alternative> alternatives_;
};

// The Qwen2 / Llama 3 family: they differ in the letter class and how many digits a number piece takes.
std::vector<Alternative> qwen_family(uint8_t letters, size_t max_digits) {
    return {
        {AltKind::kContractions},
        {AltKind::kLetters, letters},
        {AltKind::kNumbers, kLetter, max_digits},
        {AltKind::kPunctuation, letters},
        {AltKind::kSpacesThenNewlines},
        {AltKind::kSpacesNotBeforeNonSpace},
        {AltKind::kSpaces},
    };
}
}  // namespace

std::unique_ptr<RegexSplitter> RegexSplitter::for_pattern(std::string_view pattern) {
    static const std::string kContractions = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)";
    static const std::string kQwen35 = kContractions + R"([^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|)" +
                                       R"( ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const std::string kQwen2 =
        kContractions + R"([^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    static const std::string kLlama3 =
        kContractions + R"([^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    if (pattern == kQwen35) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter | kMark, 1));
    if (pattern == kQwen2) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter, 1));
    if (pattern == kLlama3) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter, 3));
    if (pattern == R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)") return gpt2();
    return nullptr;
}

std::unique_ptr<RegexSplitter> RegexSplitter::gpt2() {
    return std::make_unique<AlternativesSplitter>(std::vector<Alternative>{
        {.kind = AltKind::kContractions, .ignore_case = false},
        {.kind = AltKind::kSpaceThenRun, .letters = kLetter},
        {.kind = AltKind::kSpaceThenRun, .letters = kNumber},
        {.kind = AltKind::kSpaceThenRun, .letters = kSpace | kLetter | kNumber, .negated = true},
        {.kind = AltKind::kSpacesNotBeforeNonSpace},
        {.kind = AltKind::kSpaces},
    });
}

}  // namespace xs::tok
