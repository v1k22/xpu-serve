#include "models/common/py_text.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

#include "util/error.h"

namespace xs::py {
namespace {

using Json = nlohmann::ordered_json;

void append_hex(std::string& out, const char* prefix, uint32_t value, int digits) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%s%0*x", prefix, digits, value);
    out += buf;
}

// json.dumps with ensure_ascii=False escapes the quote, the backslash and the C0 controls, nothing else.
void json_string(std::string& out, const std::string& s) {
    out += '"';
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (c < 0x20) {
                    append_hex(out, "\\u", c, 4);
                } else {
                    out += ch;
                }
        }
    }
    out += '"';
}

// One code point of valid UTF-8 starting at s[i]; advances i. Invalid bytes come back one at a time.
uint32_t next_code_point(const std::string& s, size_t& i) {
    const auto b = [&](size_t k) { return static_cast<uint32_t>(static_cast<unsigned char>(s[k])); };
    const uint32_t c = b(i);
    const int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    if (extra == 0 || i + static_cast<size_t>(extra) >= s.size()) {
        ++i;
        return c;
    }
    uint32_t cp = c & (0x3Fu >> extra);
    for (int k = 1; k <= extra; ++k)
        cp = (cp << 6) | (b(i + static_cast<size_t>(k)) & 0x3Fu);
    i += static_cast<size_t>(extra) + 1;
    return cp;
}

// Code points str.isprintable() rejects, among those that turn up in text: controls, format characters, the
// separators and the spaces other than U+0020. (Unassigned code points, which Python also escapes, are kept.)
bool printable(uint32_t cp) {
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0xA0) || cp == 0xAD) return false;
    if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200F) || (cp >= 0x2028 && cp <= 0x202F)) return false;
    if ((cp >= 0x205F && cp <= 0x2064) || (cp >= 0x2066 && cp <= 0x206F) || cp == 0x3000 || cp == 0xFEFF) return false;
    if ((cp >= 0xD800 && cp <= 0xDFFF) || (cp >= 0xE000 && cp <= 0xF8FF) || (cp >= 0xFFF9 && cp <= 0xFFFB))
        return false;
    return true;
}

// repr(str): single quotes unless the text holds one and no double quote.
void repr_string(std::string& out, const std::string& s) {
    const bool has_single = s.find('\'') != std::string::npos, has_double = s.find('"') != std::string::npos;
    const char quote = has_single && !has_double ? '"' : '\'';
    out += quote;
    for (size_t i = 0; i < s.size();) {
        const size_t start = i;
        const uint32_t cp = next_code_point(s, i);
        if (cp == static_cast<uint32_t>(quote) || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\n') {
            out += "\\n";
        } else if (cp == '\r') {
            out += "\\r";
        } else if (cp == '\t') {
            out += "\\t";
        } else if (printable(cp)) {
            out.append(s, start, i - start);
        } else if (cp < 0x100) {
            append_hex(out, "\\x", cp, 2);
        } else if (cp < 0x10000) {
            append_hex(out, "\\u", cp, 4);
        } else {
            append_hex(out, "\\U", cp, 8);
        }
    }
    out += quote;
}

void number(std::string& out, const Json& v) {
    if (v.is_number_float()) {
        out += float_repr(v.get<double>());
    } else if (v.is_number_unsigned()) {
        out += std::to_string(v.get<uint64_t>());
    } else {
        out += std::to_string(v.get<int64_t>());
    }
}

// The members of an object in the order the style asks for.
std::vector<std::pair<const std::string*, const Json*>> members(const Json& object, bool sorted) {
    std::vector<std::pair<const std::string*, const Json*>> out;
    for (auto it = object.begin(); it != object.end(); ++it)
        out.emplace_back(&it.key(), &it.value());
    if (sorted) {
        std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return *a.first < *b.first; });
    }
    return out;
}

void dump_json(std::string& out, const Json& v, const JsonStyle& style) {
    const char* item_sep = style.compact ? "," : ", ";
    const char* key_sep = style.compact ? ":" : ": ";
    if (v.is_null()) {
        out += "null";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "true" : "false";
    } else if (v.is_number()) {
        number(out, v);
    } else if (v.is_string()) {
        json_string(out, v.get_ref<const std::string&>());
    } else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const Json& item : v) {
            if (!first) out += item_sep;
            first = false;
            dump_json(out, item, style);
        }
        out += ']';
    } else if (v.is_object()) {
        out += '{';
        bool first = true;
        for (const auto& [key, value] : members(v, style.sort_keys)) {
            if (!first) out += item_sep;
            first = false;
            json_string(out, *key);
            out += key_sep;
            dump_json(out, *value, style);
        }
        out += '}';
    } else {
        fail("a JSON value of type ", v.type_name(), " has no Python text");
    }
}

void dump_repr(std::string& out, const Json& v) {
    if (v.is_null()) {
        out += "None";
    } else if (v.is_boolean()) {
        out += v.get<bool>() ? "True" : "False";
    } else if (v.is_number()) {
        number(out, v);
    } else if (v.is_string()) {
        repr_string(out, v.get_ref<const std::string&>());
    } else if (v.is_array()) {
        out += '[';
        bool first = true;
        for (const Json& item : v) {
            if (!first) out += ", ";
            first = false;
            dump_repr(out, item);
        }
        out += ']';
    } else if (v.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            repr_string(out, it.key());
            out += ": ";
            dump_repr(out, it.value());
        }
        out += '}';
    } else {
        fail("a JSON value of type ", v.type_name(), " has no Python text");
    }
}

}  // namespace

std::string float_repr(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
    // The shortest digits that round-trip, as d.ddde[+-]xx; then Python's choice of notation: exponential when the
    // decimal point would sit more than 16 digits right of the first digit or more than 4 left of it.
    char buf[64];
    const auto [end, ec] = std::to_chars(buf, buf + sizeof buf, value, std::chars_format::scientific);
    XS_CHECK(ec == std::errc(), "cannot format a float");
    const std::string sci(buf, end);
    const size_t e = sci.find('e');
    std::string digits;
    for (size_t i = 0; i < e; ++i) {
        if (sci[i] >= '0' && sci[i] <= '9') digits += sci[i];
    }
    const bool negative = sci[0] == '-';
    const int exponent = std::stoi(sci.substr(e + 1));
    const int point = exponent + 1;  // digits before the decimal point
    const auto n = static_cast<int>(digits.size());
    std::string out = negative ? "-" : "";
    if (point > 16 || point <= -4) {
        out += digits[0];
        if (n > 1) out += "." + digits.substr(1);
        char exp[16];
        std::snprintf(exp, sizeof exp, "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
        return out + exp;
    }
    if (point <= 0) return out + "0." + std::string(static_cast<size_t>(-point), '0') + digits;
    if (point >= n) return out + digits + std::string(static_cast<size_t>(point - n), '0') + ".0";
    return out + digits.substr(0, static_cast<size_t>(point)) + "." + digits.substr(static_cast<size_t>(point));
}

std::string json_dumps(const Json& value, JsonStyle style) {
    std::string out;
    dump_json(out, value, style);
    return out;
}

std::string str(const Json& value) {
    if (value.is_string()) return value.get<std::string>();
    std::string out;
    dump_repr(out, value);
    return out;
}

}  // namespace xs::py
