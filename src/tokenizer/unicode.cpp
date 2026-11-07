#include "tokenizer/unicode.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "tokenizer/unicode_tables.h"
#include "tokenizer/utf8.h"

namespace xs::tok::unicode {
namespace {
namespace t = tables;

constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

// The range containing c, or nullptr.
template <typename Range>
const Range* find_range(const Range* ranges, size_t count, char32_t c) {
    const Range* end = ranges + count;
    const Range* it = std::upper_bound(ranges, end, c, [](char32_t v, const Range& r) { return v < r.first; });
    if (it == ranges) return nullptr;
    --it;
    return c <= it->last ? it : nullptr;
}

const t::Decomposition* find_decomposition(char32_t c) {
    const t::Decomposition* end = t::kDecompositions + t::kDecompositionCount;
    const t::Decomposition* it = std::lower_bound(
        t::kDecompositions, end, c, [](const t::Decomposition& d, char32_t v) { return d.code_point < v; });
    return it != end && it->code_point == c ? it : nullptr;
}

uint64_t pair_key(char32_t a, char32_t b) { return (static_cast<uint64_t>(a) << 21) | b; }

const std::unordered_map<uint64_t, char32_t>& compositions() {
    static const std::unordered_map<uint64_t, char32_t> table = [] {
        std::unordered_map<uint64_t, char32_t> m;
        for (size_t i = 0; i < t::kDecompositionCount; ++i) {
            const t::Decomposition& d = t::kDecompositions[i];
            if (d.composes) m.emplace(pair_key(d.first, d.second), d.code_point);
        }
        return m;
    }();
    return table;
}

bool nfc_unsafe(char32_t c) {
    return c >= 0x300 && find_range(t::kNfcUnsafeRanges, t::kNfcUnsafeRangeCount, c) != nullptr;
}

void decompose(char32_t c, std::vector<char32_t>& out) {
    if (c >= kSBase && c < kSBase + kSCount) {
        const char32_t s = c - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount != 0) out.push_back(kTBase + s % kTCount);
        return;
    }
    const t::Decomposition* d = find_decomposition(c);
    if (d == nullptr) {
        out.push_back(c);
        return;
    }
    decompose(d->first, out);
    if (d->second != 0) decompose(d->second, out);
}

char32_t compose_pair(char32_t a, char32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    }
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) {
        return a + (b - kTBase);
    }
    const auto& table = compositions();
    const auto it = table.find(pair_key(a, b));
    return it == table.end() ? 0 : it->second;
}

// NFC of a sequence of code points (UAX #15: decompose, reorder by combining class, compose).
void nfc_chars(std::vector<char32_t>& chars) {
    std::vector<char32_t> d;
    d.reserve(chars.size() + 4);
    for (const char32_t c : chars) decompose(c, d);
    for (size_t i = 1; i < d.size(); ++i) {
        const uint8_t cc = combining_class(d[i]);
        if (cc == 0) continue;
        for (size_t j = i; j > 0 && combining_class(d[j - 1]) > cc; --j) std::swap(d[j - 1], d[j]);
    }
    chars.clear();
    if (d.empty()) return;
    size_t starter = 0;
    chars.push_back(d[0]);
    int last_class = combining_class(d[0]) == 0 ? 0 : 256;  // 256: no starter to compose with yet
    for (size_t i = 1; i < d.size(); ++i) {
        const int cc = combining_class(d[i]);
        if (last_class < cc || last_class == 0) {
            if (const char32_t composite = compose_pair(chars[starter], d[i])) {
                chars[starter] = composite;
                continue;
            }
        }
        if (cc == 0) starter = chars.size();
        last_class = cc;
        chars.push_back(d[i]);
    }
}

// The code point at text[pos] and its length; an invalid byte is reported as length 1 and treated as NFC-safe.
struct Char {
    char32_t c;
    size_t len;
    bool valid;
};

Char char_at(std::string_view text, size_t pos) {
    const size_t len = utf8::valid_char_length(text, pos);
    if (len == 0) return {0, 1, false};
    return {utf8::decode_at(text, pos), len, true};
}

bool unsafe_at(std::string_view text, size_t pos, size_t& len) {
    const Char ch = char_at(text, pos);
    len = ch.len;
    return ch.valid && nfc_unsafe(ch.c);
}
}  // namespace

Category category(char32_t c) {
    if (c < 0x80) {
        if (c >= '0' && c <= '9') return Category::kNumber;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Category::kLetter;
        return Category::kOther;
    }
    const t::CategoryRange* r = find_range(t::kCategoryRanges, t::kCategoryRangeCount, c);
    return r == nullptr ? Category::kOther : static_cast<Category>(r->value);
}

uint8_t combining_class(char32_t c) {
    if (c < 0x300) return 0;
    const t::CombiningRange* r = find_range(t::kCombiningRanges, t::kCombiningRangeCount, c);
    return r == nullptr ? 0 : r->ccc;
}

// Text is copied through up to the first unsafe character; from the safe character before it (which may compose with
// what follows) to the next safe character, the segment is normalized. A safe character (quick check Yes, combining
// class 0) never interacts with what precedes it, so segments normalize independently.
std::string nfc(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    size_t copied = 0;     // text[0, copied) is in out
    size_t last_safe = 0;  // start of the last safe character seen at or after `copied`
    bool have_safe = false;
    std::vector<char32_t> chars;
    for (size_t pos = 0; pos < text.size();) {
        if (static_cast<uint8_t>(text[pos]) < 0xCC) {  // lead bytes below 0xCC encode code points below U+0300
            last_safe = pos;
            have_safe = true;
            pos += static_cast<uint8_t>(text[pos]) < 0x80 ? 1 : utf8::char_length(text, pos);
            continue;
        }
        size_t len = 0;
        if (!unsafe_at(text, pos, len)) {
            last_safe = pos;
            have_safe = true;
            pos += len;
            continue;
        }
        const size_t begin = have_safe ? last_safe : pos;
        size_t end = pos + len;
        while (end < text.size()) {
            size_t next_len = 0;
            if (!unsafe_at(text, end, next_len)) break;
            end += next_len;
        }
        out.append(text.substr(copied, begin - copied));
        chars.clear();
        for (size_t p = begin; p < end;) {
            const Char ch = char_at(text, p);
            chars.push_back(ch.c);
            p += ch.len;
        }
        nfc_chars(chars);
        for (const char32_t c : chars) utf8::append(out, c);
        copied = end;
        have_safe = false;
        pos = end;
    }
    out.append(text.substr(copied));
    return out;
}

}  // namespace xs::tok::unicode
