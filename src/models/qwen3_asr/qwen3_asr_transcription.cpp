#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>
#include <vector>

#include "models/qwen3_asr/qwen3_asr.h"
#include "util/error.h"

namespace xs::qwen3_asr {
namespace {

constexpr std::string_view kTag = "<asr_text>";
constexpr std::string_view kLanguage = "language ";
constexpr size_t kLongestHeader = 50;  // past this, text without the tag is a transcript, not a header being written

// ISO 639-1 codes (and 639-3 where 639-1 has none) of the languages Qwen3-ASR names.
constexpr std::pair<std::string_view, std::string_view> kCodes[] = {
    {"zh", "Chinese"},    {"en", "English"},  {"yue", "Cantonese"}, {"ar", "Arabic"},     {"de", "German"},
    {"fr", "French"},     {"es", "Spanish"},  {"pt", "Portuguese"}, {"id", "Indonesian"}, {"it", "Italian"},
    {"ko", "Korean"},     {"ru", "Russian"},  {"th", "Thai"},       {"vi", "Vietnamese"}, {"ja", "Japanese"},
    {"tr", "Turkish"},    {"hi", "Hindi"},    {"ms", "Malay"},      {"nl", "Dutch"},      {"sv", "Swedish"},
    {"da", "Danish"},     {"fi", "Finnish"},  {"pl", "Polish"},     {"cs", "Czech"},      {"fil", "Filipino"},
    {"tl", "Filipino"},   {"fa", "Persian"},  {"el", "Greek"},      {"ro", "Romanian"},   {"hu", "Hungarian"},
    {"mk", "Macedonian"},
};

std::string trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return "";
    return std::string(s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1));
}

// "cHINese" -> "Chinese", as the reference package writes a language's name.
std::string canonical(std::string_view name) {
    std::string s = trim(name);
    for (size_t i = 0; i < s.size(); ++i) {
        s[i] = static_cast<char>(i == 0 ? std::toupper(static_cast<unsigned char>(s[i]))
                                        : std::tolower(static_cast<unsigned char>(s[i])));
    }
    return s;
}

// The characters of UTF-8 text (code points, as Python counts them; decoded text is valid UTF-8).
std::vector<std::string_view> characters(std::string_view s) {
    std::vector<std::string_view> out;
    for (size_t i = 0; i < s.size();) {
        const auto lead = static_cast<unsigned char>(s[i]);
        const size_t n = lead < 0x80 ? 1 : lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        out.push_back(s.substr(i, std::min(n, s.size() - i)));
        i += n;
    }
    return out;
}

using Chars = std::vector<std::string_view>;

bool same_run(const Chars& s, size_t a, size_t b, size_t k) {
    for (size_t j = 0; j < k; ++j) {
        if (s[a + j] != s[b + j]) return false;
    }
    return true;
}

// detect_and_fix_repetitions' fix_pattern_repeats, over s[from, n): at the first place a pattern of up to 20
// characters repeats 20 times or more, it is kept once and the rest is fixed the same way.
void fix_pattern_repeats(const Chars& s, size_t from, std::string& out) {
    constexpr size_t kThreshold = 20, kMaxLen = 20;
    const size_t n = s.size() - from;
    if (n < 2 * kThreshold) {
        for (size_t i = from; i < s.size(); ++i) out += s[i];
        return;
    }
    size_t i = from;
    while (i <= s.size() - 2 * kThreshold) {
        for (size_t k = 1; k <= kMaxLen && i + k * kThreshold <= s.size(); ++k) {
            bool valid = true;
            for (size_t rep = 1; rep < kThreshold && valid; ++rep) valid = same_run(s, i, i + rep * k, k);
            if (!valid) continue;
            size_t end = i + kThreshold * k;
            while (end + k <= s.size() && same_run(s, i, end, k)) end += k;
            for (size_t j = 0; j < k; ++j) out += s[i + j];
            fix_pattern_repeats(s, end, out);
            return;
        }
        out += s[i++];
    }
    for (; i < s.size(); ++i) out += s[i];
}

// The Qwen3-ASR package's detect_and_fix_repetitions (threshold 20): a character repeated more than 20 times in a
// row is kept once, then a pattern of up to 20 characters repeated 20 times or more is kept once.
std::string fix_repetitions(std::string_view text) {
    const Chars s = characters(text);
    Chars runs;
    for (size_t i = 0; i < s.size();) {
        size_t count = 1;
        while (i + count < s.size() && s[i + count] == s[i]) ++count;
        if (count > 20) {
            runs.push_back(s[i]);
        } else {
            runs.insert(runs.end(), s.begin() + static_cast<std::ptrdiff_t>(i),
                        s.begin() + static_cast<std::ptrdiff_t>(i + count));
        }
        i += count;
    }
    std::string out;
    fix_pattern_repeats(runs, 0, out);
    return out;
}

class AsrTranscription final : public Transcription {
public:
    AsrTranscription(std::vector<std::string> languages, double audio_tokens_per_second)
        : languages_(std::move(languages)), audio_tokens_per_second_(audio_tokens_per_second) {
        if (languages_.empty()) {
            for (const auto& [code, name] : kCodes) {
                if (std::find(languages_.begin(), languages_.end(), name) == languages_.end()) {
                    languages_.emplace_back(name);
                }
            }
        }
    }

    double max_piece_seconds() const override { return 1200.0; }
    int max_reply_tokens() const override { return 4096; }
    double audio_tokens_per_second() const override { return audio_tokens_per_second_; }

    std::string language_prefix(std::string_view language) const override {
        if (trim(language).empty()) return "";
        std::string name = canonical(language);
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        for (const auto& [code, full] : kCodes) {
            if (lower == code) name = full;
        }
        if (std::find(languages_.begin(), languages_.end(), name) == languages_.end()) {
            std::string list;
            for (const std::string& l : languages_) list += (list.empty() ? "" : ", ") + l;
            fail("language '", std::string(language), "' is not one this model transcribes (", list,
                 ", or their ISO 639-1 codes)");
        }
        return std::string(kLanguage) + name + std::string(kTag);
    }

    Transcript parse(std::string_view prefix, std::string_view reply, bool finished) const override {
        Transcript t;
        if (!prefix.empty()) {  // a forced language: the reply is the transcript
            t.language = std::string(prefix.substr(kLanguage.size(), prefix.size() - kLanguage.size() - kTag.size()));
            t.text = finished ? fix_repetitions(trim(reply)) : std::string(reply);
            return t;
        }
        const size_t tag = reply.find(kTag);
        if (tag == std::string_view::npos) {
            const std::string head = trim(reply);
            const bool may_be_header = head.size() < kLongestHeader && head.find('\n') == std::string::npos &&
                                       (kLanguage.substr(0, std::min(head.size(), kLanguage.size())) ==
                                        std::string_view(head).substr(0, std::min(head.size(), kLanguage.size())));
            if (!finished && may_be_header) return t;  // perhaps "language English" before its tag
            t.text = finished ? fix_repetitions(head) : std::string(reply);
            return t;
        }
        const std::string_view meta = reply.substr(0, tag);
        std::string_view text = reply.substr(tag + kTag.size());
        // "language <Name>" on the first line that has it; "language None": silence
        size_t at = 0;
        while (at < meta.size()) {
            const size_t end = std::min(meta.find('\n', at), meta.size());
            const std::string line = trim(meta.substr(at, end - at));
            if (line.rfind(kLanguage, 0) == 0) {
                const std::string name = canonical(std::string_view(line).substr(kLanguage.size()));
                if (name != "None") t.language = name;
                break;
            }
            at = end + 1;
        }
        t.text = finished ? fix_repetitions(trim(text)) : std::string(text);
        return t;
    }

private:
    std::vector<std::string> languages_;
    double audio_tokens_per_second_;
};

}  // namespace

std::unique_ptr<Transcription> make_transcription(std::vector<std::string> languages,
                                                  double audio_tokens_per_second) {
    return std::make_unique<AsrTranscription>(std::move(languages), audio_tokens_per_second);
}

}  // namespace xs::qwen3_asr
