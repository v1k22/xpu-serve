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
    kUpper = 32,    // a letter of \p{Lu} or \p{Lt}
    kLower = 64,    // a letter of \p{Ll}
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
            switch (unicode::letter_case(c)) {
                case unicode::LetterCase::kUpper: return kLetter | kUpper;
                case unicode::LetterCase::kLower: return kLetter | kLower;
                case unicode::LetterCase::kNone: return kLetter;
            }
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
    // [^\r\n\p{L}\p{N}]?U*L+C? (`lower_first`) or [^\r\n\p{L}\p{N}]?U+L*C?, with U = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}],
    // L = [\p{Ll}\p{Lm}\p{Lo}\p{M}] and C the contractions, any case
    kCasedWord,
    kSpaceThenRun,        //  ?<class>+, the class the flags in `letters` (or, with `negated`, its complement)
    kNumbers,             // \p{N}{1,max}
    kPunctuation,         //  ?[^\s\p{L}\p{N}<\p{M} with marks>]+[\r\n]* (`slashes`: [\r\n/]*)
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
    bool lower_first = false;     // kCasedWord: the lowercase run is the one that cannot be empty
    bool slashes = false;         // kPunctuation
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
    // End of 's, 't, 're, 've, 'm, 'll or 'd at char i, or i.
    static size_t contraction(const std::vector<CharInfo>& s, size_t n, size_t i, bool ignore_case) {
        if (i + 1 >= n || s[i].c != '\'') return i;
        const char32_t a = ignore_case ? fold(s[i + 1].c) : s[i + 1].c;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 >= n) return i;
        const char32_t b = ignore_case ? fold(s[i + 2].c) : s[i + 2].c;
        if ((a == 'r' || a == 'v') && b == 'e') return i + 3;
        if (a == 'l' && b == 'l') return i + 3;
        return i;
    }

    // End (char index) of the match of `alt` starting at char i, or i when it does not match.
    static size_t match_at(const Alternative& alt, const std::vector<CharInfo>& s, size_t n, size_t i) {
        auto run = [&](size_t j, auto pred) {
            while (j < n && pred(s[j].flags)) ++j;
            return j;
        };
        switch (alt.kind) {
            case AltKind::kContractions:
                return contraction(s, n, i, alt.ignore_case);
            case AltKind::kCasedWord: {
                auto upper = [](uint8_t f) { return (f & kMark) != 0 || ((f & kLetter) != 0 && (f & kLower) == 0); };
                auto lower = [](uint8_t f) { return (f & kMark) != 0 || ((f & kLetter) != 0 && (f & kUpper) == 0); };
                // The word from j: its end, or j when there is none.
                auto word = [&](size_t j) {
                    const size_t ups = run(j, upper);
                    if (!alt.lower_first) return ups == j ? j : run(ups, lower);
                    // U* gives characters back until L+ can start: at the last of them that L takes too
                    for (size_t k = ups + 1; k-- > j;) {
                        if (k < n && lower(s[k].flags)) return run(k, lower);
                    }
                    return j;
                };
                auto prefix = [](uint8_t f) { return (f & (kNewline | kLetter | kNumber)) == 0; };
                size_t end = i;
                if (i + 1 < n && prefix(s[i].flags)) end = word(i + 1) > i + 1 ? word(i + 1) : i;
                if (end == i) end = word(i);
                return end == i ? i : contraction(s, n, end, true);
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
                while (j < n && ((s[j].flags & kNewline) != 0 || (alt.slashes && s[j].c == '/'))) ++j;
                return j;
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

// o200k_base's: no contraction stands alone, and a slash ends a run of punctuation.
std::vector<Alternative> o200k() {
    return {
        {.kind = AltKind::kCasedWord, .lower_first = true},
        {.kind = AltKind::kCasedWord},
        {AltKind::kNumbers, kLetter, 3},
        {.kind = AltKind::kPunctuation, .letters = kLetter, .slashes = true},
        {AltKind::kSpacesThenNewlines},
        {AltKind::kSpacesNotBeforeNonSpace},
        {AltKind::kSpaces},
    };
}
}  // namespace

std::unique_ptr<RegexSplitter> RegexSplitter::for_pattern(std::string_view pattern) {
    if (pattern == split_regex::kO200k) return std::make_unique<AlternativesSplitter>(o200k());
    if (pattern == split_regex::kQwen35) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter | kMark, 1));
    if (pattern == split_regex::kQwen2) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter, 1));
    if (pattern == split_regex::kLlama3) return std::make_unique<AlternativesSplitter>(qwen_family(kLetter, 3));
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
