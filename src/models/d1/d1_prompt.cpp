#include <algorithm>
#include <array>

#include "models/common/py_text.h"
#include "models/d1/d1.h"

namespace xs::d1 {
namespace {

constexpr const char* kImStart = "<|im_start|>";
constexpr const char* kImEnd = "<|im_end|>";
// What the LFM3 templates write between the assistant header and an answer without thinking (lfm3.py: LEADS).
constexpr const char* kLead = "<think></think>";
// The end of the state's block, and what opens each further question packed after it (prompt.py: question_header).
constexpr const char* kHeader = "QUESTION:\n";
constexpr std::string_view kBreak = "\xE2\x80\x8B";  // U+200B

size_t utf8_length(unsigned char lead) {
    return lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
}

// The code points of `s` (valid UTF-8, as JSON text is).
std::vector<char32_t> code_points(std::string_view s) {
    std::vector<char32_t> out;
    for (size_t i = 0; i < s.size();) {
        const auto lead = static_cast<unsigned char>(s[i]);
        const size_t n = std::min(utf8_length(lead), s.size() - i);
        char32_t c = n == 1 ? lead : lead & (0x7Fu >> n);
        for (size_t k = 1; k < n; ++k)
            c = (c << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
        out.push_back(c);
        i += n;
    }
    return out;
}

// str.isspace(): what str.strip() removes.
bool python_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::vector<char32_t> stripped(std::string_view s) {
    std::vector<char32_t> c = code_points(s);
    const auto first = std::find_if_not(c.begin(), c.end(), python_space);
    const auto last = std::find_if_not(c.rbegin(), std::make_reverse_iterator(first), python_space).base();
    return {first, last};
}

bool ascii_letter(char32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Python's truth of a JSON value: what `value or default` keeps.
bool truthy(const Json& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number()) return v.get<double>() != 0;
    if (v.is_string()) return !v.get_ref<const std::string&>().empty();
    return !v.empty();
}

// The codes prompt.py falls back on, in its order, when a code is no single token of its own or is taken.
const std::vector<std::string>& fallback_pool() {
    static const std::vector<std::string> pool = [] {
        std::vector<std::string> p;
        for (char c = 'A'; c <= 'Z'; ++c)
            p.emplace_back(1, c);
        for (int i = 0; i < 100; ++i)
            p.push_back(str_cat(i < 10 ? "0" : "", i));
        for (char c = 'a'; c <= 'z'; ++c)
            p.emplace_back(1, c);
        for (int i = 0; i < 200; ++i)
            p.push_back(str_cat("#", i));
        for (char a = 'A'; a <= 'Z'; ++a) {
            for (char b = 'A'; b <= 'Z'; ++b)
                p.push_back(std::string{a, b});
        }
        return p;
    }();
    return pool;
}

// The forms of `texts` that are one token each, without repeats (prompt.py: _ids).
std::vector<int32_t> single_tokens(const PromptTools& tools, std::initializer_list<std::string> texts) {
    std::vector<int32_t> out;
    for (const std::string& t : texts) {
        const std::vector<int32_t> ids = tools.encode(t);
        if (ids.size() == 1 && std::find(out.begin(), out.end(), ids[0]) == out.end()) out.push_back(ids[0]);
    }
    return out;
}

[[noreturn]] void refuse(const DecisionQuestion& q, const std::string& why) {
    throw DecisionRefused(str_cat("question '", q.id, "': ", why), "unsupported_question");
}

// What a question asks, as the prompt shows it, and the tokens each of its options is read at.
struct Layout {
    std::string suffix;  // the question and the assistant header, up to the answer slot
    std::vector<std::vector<int32_t>> groups;
};

class Writer {
public:
    Writer(const PromptTools& tools, const Config& config)
        : tools_(tools), config_(config), special_(tools.special_tokens()) {}

    std::string plain_text(std::string_view text) const { return plain(text, special_); }

    // Everything before the first question: the user turn's head and the state (prompt.py: prefix_text).
    std::string prefix(const Json& state) const {
        const std::string block = (state.is_string() ? state.get<std::string>() : py::json_dumps(state, {.indent = 2}));
        return str_cat(config_.bos, kImStart, "user\n", plain_text(block + "\n\n"), "\n", kHeader);
    }

    Layout layout(const DecisionQuestion& q) const {
        const Json& said = q.instructions;
        if (said.is_null() || (said.is_string() && stripped(said.get_ref<const std::string&>()).empty())) {
            refuse(q, "instructions are required: the question as text");
        }
        if (said.is_object() && said.contains("type") &&
            (said["type"] == "image" || said["type"] == "audio" || said["type"] == "video")) {
            refuse(q, "instructions that are media (an image, audio or video) are not read; this server reads text");
        }
        Layout out;
        std::string body = py::str(said);
        if (q.type == QuestionType::Noul) {
            // prompt.py writes the outcomes when the request describes either, None for one it does not.
            const auto described = [](const DecisionOption& o) {
                return o.description && !o.description->is_null() ? py::str(*o.description) : std::string("None");
            };
            if (q.options[0].description || q.options[1].description) {
                body += str_cat("\nYes: ", described(q.options[0]), "\nNo: ", described(q.options[1]));
            }
            body += "\n\nReply with yes or no only.";
            out.groups = {single_tokens(tools_, {"yes", "Yes", "YES"}), single_tokens(tools_, {"no", "No", "NO"})};
            XS_CHECK(!out.groups[0].empty() && !out.groups[1].empty(),
                     "d1: the tokenizer has no single-token yes / no");
        } else if (q.type == QuestionType::Score) {
            if (q.options.size() < 2 || q.options.size() > 10) refuse(q, "a score has 2 to 10 levels, lowest first");
            body += "\n\n";
            for (size_t i = 0; i < q.options.size(); ++i) {
                const std::optional<Json>& level = q.options[i].description;
                if (!level || !level->is_string()) refuse(q, "a score's levels are text");
                body += str_cat(i > 0 ? "\n" : "", i, " ", level->get_ref<const std::string&>());
                out.groups.push_back(single_tokens(tools_, {std::to_string(i)}));
                XS_CHECK(!out.groups.back().empty(), "d1: the digit ", i, " is not one token");
            }
            body += str_cat("\n\nReply with a single digit 0-", q.options.size() - 1, " only.");
        } else {
            std::vector<std::string> names;
            for (const DecisionOption& o : q.options)
                names.push_back(o.id);
            body += "\n\nOptions:\n";
            const auto codes = aliases(q, names);
            for (size_t i = 0; i < q.options.size(); ++i) {
                const DecisionOption& o = q.options[i];
                std::string text = o.id;
                if (o.description && truthy(*o.description)) {
                    text = py::str(*o.description);
                } else {
                    std::replace(text.begin(), text.end(), '_', ' ');
                }
                body += str_cat(i > 0 ? "\n" : "", codes[i].first, " ", text);
                std::vector<int32_t> group{codes[i].second};
                for (const int32_t id : single_tokens(tools_, {" " + codes[i].first})) {
                    if (id != codes[i].second) group.push_back(id);
                }
                out.groups.push_back(std::move(group));
            }
            body += "\n\nReply with the option code only.";
        }
        out.suffix = str_cat(plain_text(body), kImEnd, "\n", kImStart, "assistant\n", kLead);
        return out;
    }

private:
    // Each option's code and its token, every code a single token no other option has (prompt.py: aliases).
    std::vector<std::pair<std::string, int32_t>> aliases(const DecisionQuestion& q,
                                                         const std::vector<std::string>& names) const {
        // prompt.py tests a one-character name with str.isalpha, which knows every script's letters.
        bool single = true, foreign = false;
        for (const std::string& name : names) {
            const std::vector<char32_t> c = stripped(name);
            single = single && c.size() == 1 && (c[0] >= 0x80 || ascii_letter(c[0]));
            foreign = foreign || (c.size() == 1 && c[0] >= 0x80);
        }
        if (single && foreign) refuse(q, "option names of one character outside ASCII are not supported");
        std::vector<std::pair<std::string, int32_t>> out;
        const auto take = [&](const std::string& code) {
            const std::vector<int32_t> ids = tools_.encode(code);
            const auto taken = [&](const auto& c) { return c.second == ids[0]; };
            if (ids.size() != 1 || std::any_of(out.begin(), out.end(), taken)) return false;
            out.emplace_back(code, ids[0]);
            return true;
        };
        for (const std::string& code : option_codes(names, config_.code_scheme)) {
            if (take(code)) continue;
            const auto& pool = fallback_pool();
            if (!std::any_of(pool.begin(), pool.end(), take)) {
                refuse(q, str_cat("no single-token code is left for ", names.size(), " options"));
            }
        }
        XS_CHECK(out.size() == names.size(), "d1: ", out.size(), " codes for ", names.size(), " options");
        return out;
    }

    const PromptTools& tools_;
    const Config& config_;
    std::vector<std::string> special_;
};

}  // namespace

std::string plain(std::string_view text, const std::vector<std::string>& special_tokens) {
    std::vector<std::string> names;
    std::array<bool, 256> starts{};
    for (const std::string& t : special_tokens) {
        if (code_points(t).size() > 1 && t.find(kBreak) == std::string::npos) {
            names.push_back(t);
            starts[static_cast<unsigned char>(t[0])] = true;
        }
    }
    // A longer name first where two start alike, as the reference's alternation is ordered.
    std::stable_sort(names.begin(), names.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
    std::string current(text);
    // Again for a match that overlapped the one broken before it.
    for (bool changed = true; changed;) {
        changed = false;
        std::string out;
        out.reserve(current.size());
        for (size_t i = 0; i < current.size();) {
            const std::string* hit = nullptr;
            if (starts[static_cast<unsigned char>(current[i])]) {
                for (const std::string& name : names) {
                    if (current.compare(i, name.size(), name) == 0) {
                        hit = &name;
                        break;
                    }
                }
            }
            if (hit == nullptr) {
                out += current[i++];
                continue;
            }
            const size_t first = utf8_length(static_cast<unsigned char>((*hit)[0]));
            out.append(*hit, 0, first).append(kBreak).append(*hit, first);
            i += hit->size();
            changed = true;
        }
        current = std::move(out);
    }
    return current;
}

std::vector<std::string> option_codes(const std::vector<std::string>& names, int scheme) {
    const size_t n = names.size();
    std::vector<std::string> out;
    const bool letters = n > 0 && std::all_of(names.begin(), names.end(), [](const std::string& name) {
                             const std::vector<char32_t> c = stripped(name);
                             return c.size() == 1 && ascii_letter(c[0]);
                         });
    for (size_t i = 0; i < n; ++i) {
        if (letters) {
            out.emplace_back(1, static_cast<char>(stripped(names[i])[0]));
        } else if (n <= 26 || (scheme == 2 && i < 26)) {
            out.emplace_back(1, static_cast<char>('A' + i));
        } else if (scheme == 2 && i < 26 + 26 * 26) {
            out.push_back(std::string{static_cast<char>('A' + (i - 26) / 26), static_cast<char>('A' + (i - 26) % 26)});
        } else if (scheme != 2) {
            out.push_back(str_cat(i < 10 ? "0" : "", i));
        }
    }
    return out;
}

Plan plan(const DecisionRequest& request, const PromptTools& tools, const Config& config, int64_t set_size,
          int64_t max_tokens) {
    XS_CHECK(set_size >= 1, "d1: a sequence holds at least one question");
    const Writer writer(tools, config);
    const std::string prefix = writer.prefix(request.state);
    Plan out;
    std::vector<std::string> suffixes;
    for (const DecisionQuestion& q : request.questions) {
        Layout layout = writer.layout(q);
        suffixes.push_back(std::move(layout.suffix));
        out.groups.push_back(std::move(layout.groups));
    }
    const size_t n = request.questions.size(), size = static_cast<size_t>(set_size);
    for (size_t first = 0; first < n; first += size) {
        // The first question is tokenized with the state, as its lone prompt is; each further one on its own after
        // the header (attention.py: pack_set).
        Sequence s;
        s.ids = tools.encode(prefix + suffixes[first]);
        s.state = static_cast<int64_t>(s.ids.size() - tools.encode(suffixes[first]).size());
        s.questions = {first};
        s.answers = {static_cast<int64_t>(s.ids.size()) - 1};
        for (size_t q = first + 1; q < std::min(n, first + size); ++q) {
            const std::vector<int32_t> own = tools.encode(kHeader + suffixes[q]);
            s.ids.insert(s.ids.end(), own.begin(), own.end());
            s.questions.push_back(q);
            s.answers.push_back(static_cast<int64_t>(s.ids.size()) - 1);
        }
        if (static_cast<int64_t>(s.ids.size()) > max_tokens) {
            throw DecisionRefused(str_cat("question '", request.questions[first].id, "' needs ", s.ids.size(),
                                          " tokens with the state; the limit is ", max_tokens),
                                  "max_tokens_exceeded");
        }
        out.sequences.push_back(std::move(s));
    }
    return out;
}

}  // namespace xs::d1
