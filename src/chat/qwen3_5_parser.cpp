// Qwen 3.5 output dialect. The chat template renders an assistant turn as
//   <think>\n REASONING \n</think>\n\n CONTENT [\n\n]
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>[\n<tool_call>...]
// and the generation prompt already holds the "<think>\n" opener (thinking on) or a whole empty block "<think>\n\n
// </think>\n\n" (thinking off). This parser inverts that, one feed() at a time. End-of-turn tokens (<|im_end|>) are
// stripped by the engine before text reaches here.
//
// Whitespace: vLLM's qwen3 reasoning parser splits at the markers and keeps every byte (reasoning ends in "\n",
// content starts with "\n\n"). Here the template's own inverse is applied instead, so that a parsed reply rendered back
// by the template is the reply the model wrote: newlines after <think> and </think> and whitespace before </think> are
// dropped (the template's lstrip('\n') / rstrip('\n') / trim), whitespace before a tool call and between or after tool
// calls is dropped (the template writes it). Other whitespace is kept; trailing whitespace streams once more text
// follows.

#include <optional>
#include <random>
#include <span>
#include <string>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

namespace xs::chat {
namespace {

using ordered_json = nlohmann::ordered_json;

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";
constexpr std::string_view kCallOpen = "<tool_call>";
constexpr std::string_view kCallClose = "</tool_call>";
constexpr std::string_view kFunctionOpen = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParameterOpen = "<parameter=";
constexpr std::string_view kParameterClose = "</parameter>";

// tool_call: inside <tool_call>...</tool_call>. bare_function: a <function=...> block the model wrote without the
// <tool_call> wrapper, which vLLM's Qwen3-Coder parser also accepts.
enum class Mode { content, reasoning, tool_call, bare_function };
enum class Action { open_reasoning, close_reasoning, open_call, open_bare_function, close_call, drop };

struct Marker {
    std::string_view text;
    Action action;
};

// Per mode, the markers that change it; stray closers in content are dropped.
constexpr Marker kContentMarkers[] = {
    {kThinkOpen, Action::open_reasoning}, {kCallOpen, Action::open_call}, {kFunctionOpen, Action::open_bare_function},
    {kThinkClose, Action::drop},          {kCallClose, Action::drop},
};
constexpr Marker kReasoningMarkers[] = {
    {kThinkClose, Action::close_reasoning},
    {kCallOpen, Action::open_call},
};
constexpr Marker kToolCallMarkers[] = {{kCallClose, Action::close_call}};
constexpr Marker kBareFunctionMarkers[] = {{kFunctionClose, Action::close_call}};

std::span<const Marker> markers_for(Mode mode) {
    switch (mode) {
        case Mode::content:
            return kContentMarkers;
        case Mode::reasoning:
            return kReasoningMarkers;
        case Mode::tool_call:
            return kToolCallMarkers;
        case Mode::bare_function:
            return kBareFunctionMarkers;
    }
    return {};
}

// Where the next marker starts in a buffer. `marker` is null when the buffer ends inside a possible marker (hold
// from `pos`); pos == npos when there is none.
struct MarkerHit {
    size_t pos = std::string::npos;
    const Marker* marker = nullptr;
};

MarkerHit find_marker(std::string_view buffer, std::span<const Marker> markers, bool at_end) {
    for (size_t pos = buffer.find('<'); pos < buffer.size(); pos = buffer.find('<', pos + 1)) {
        const std::string_view rest = buffer.substr(pos);
        for (const Marker& m : markers) {
            if (rest.starts_with(m.text)) return {pos, &m};
            if (!at_end && m.text.starts_with(rest)) return {pos, nullptr};
        }
    }
    return {};
}

constexpr std::string_view kWhitespace = " \t\r\n";

std::string_view trim(std::string_view s) {
    const size_t begin = s.find_first_not_of(kWhitespace);
    if (begin == std::string_view::npos) return {};
    return s.substr(begin, s.find_last_not_of(kWhitespace) - begin + 1);
}

// A parameter value is text. Without the tool's schema it becomes the JSON value it spells (number, true/false/null,
// object, array, "string"), else a string, as vLLM's Qwen3 XML parser does. The one newline the template writes on
// each side is not part of the value.
ordered_json parameter_value(std::string_view text) {
    if (text.starts_with('\n')) text.remove_prefix(1);
    if (text.ends_with('\n')) text.remove_suffix(1);
    ordered_json parsed = ordered_json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (!parsed.is_discarded()) return parsed;
    return std::string(text);
}

// "<function=NAME>" then "<parameter=KEY>VALUE</parameter>"* then "</function>", with whitespace between the parts.
// Lenient where vLLM is: a missing </parameter> ends at the next <parameter= or the end; a missing </function> is
// accepted (the caller only gets here once the call was closed). Returns the name and the arguments as JSON text.
std::optional<std::pair<std::string, std::string>> parse_function(std::string_view body) {
    body = trim(body);
    if (!body.starts_with(kFunctionOpen)) return std::nullopt;
    body.remove_prefix(kFunctionOpen.size());
    const size_t name_end = body.find('>');
    if (name_end == std::string_view::npos) return std::nullopt;
    const std::string_view name = trim(body.substr(0, name_end));
    if (name.empty() || name.find_first_of(" \t\r\n<") != std::string_view::npos) return std::nullopt;
    body.remove_prefix(name_end + 1);
    if (const size_t end = body.find(kFunctionClose); end != std::string_view::npos) {
        if (!trim(body.substr(end + kFunctionClose.size())).empty()) return std::nullopt;
        body = body.substr(0, end);
    }

    ordered_json arguments = ordered_json::object();
    for (body = trim(body); !body.empty(); body = trim(body)) {
        if (!body.starts_with(kParameterOpen)) return std::nullopt;
        body.remove_prefix(kParameterOpen.size());
        const size_t key_end = body.find('>');
        if (key_end == std::string_view::npos) return std::nullopt;
        const std::string_view key = trim(body.substr(0, key_end));
        if (key.empty()) return std::nullopt;
        body.remove_prefix(key_end + 1);
        size_t value_end = body.find(kParameterClose);
        size_t next = value_end + kParameterClose.size();
        if (value_end == std::string_view::npos) {
            value_end = next = std::min(body.find(kParameterOpen), body.size());
        }
        arguments[std::string(key)] = parameter_value(body.substr(0, value_end));
        body.remove_prefix(next);
    }
    return std::pair{std::string(name), arguments.dump(-1, ' ', false, ordered_json::error_handler_t::replace)};
}

std::string random_hex(size_t digits) {
    std::random_device rd;
    std::uniform_int_distribution<int> nibble(0, 15);
    std::string out;
    for (size_t i = 0; i < digits; ++i)
        out += "0123456789abcdef"[nibble(rd)];
    return out;
}

// Which leading characters of the next text to drop.
enum class Lead { keep, newlines, whitespace };

class Qwen35Parser final : public OutputParser {
public:
    explicit Qwen35Parser(std::string_view prompt_tail) : id_prefix_("call_" + random_hex(8)) {
        start_after(prompt_tail);
    }

    ParsedDelta feed(std::string_view text) override {
        pending_ += text;
        ParsedDelta delta;
        drain(delta, /*at_end=*/false);
        return delta;
    }

    ParsedDelta finish() override {
        ParsedDelta delta;
        drain(delta, /*at_end=*/true);
        // A call cut short is kept only when its function is complete and just </tool_call> is missing: a call whose
        // last argument may be truncated must not run. Otherwise its text is content.
        if (mode_ == Mode::tool_call && call_body_.find(kFunctionClose) != std::string::npos) {
            complete_call(delta);
        } else if (mode_ == Mode::tool_call || mode_ == Mode::bare_function) {
            delta.content += call_opener() + call_body_;
        } else {
            text_out(delta) += held_space_;
        }
        call_body_.clear();
        held_space_.clear();
        mode_ = Mode::content;
        lead_ = Lead::keep;
        return delta;
    }

private:
    // The generation prompt ends inside "<think>\n" (thinking on) or after "</think>\n\n" (thinking off, or a
    // prefilled reply); whichever of the two markers comes last in the open assistant turn decides the start.
    void start_after(std::string_view prompt_tail) {
        constexpr std::string_view turn = "<|im_start|>";
        if (const size_t last_turn = prompt_tail.rfind(turn); last_turn != std::string_view::npos) {
            prompt_tail.remove_prefix(last_turn);
            if (!prompt_tail.starts_with("<|im_start|>assistant")) return;
        }
        const size_t open = prompt_tail.rfind(kThinkOpen);
        const size_t close = prompt_tail.rfind(kThinkClose);
        if (open == std::string_view::npos && close == std::string_view::npos) return;
        if (close == std::string_view::npos || (open != std::string_view::npos && open > close)) {
            mode_ = Mode::reasoning;
        }
        lead_ = Lead::newlines;
    }

    void drain(ParsedDelta& delta, bool at_end) {
        for (;;) {
            const MarkerHit hit = find_marker(pending_, markers_for(mode_), at_end);
            const size_t text_end = hit.pos == std::string::npos ? pending_.size() : hit.pos;
            emit(delta, std::string_view(pending_).substr(0, text_end));
            if (hit.marker == nullptr) {
                pending_.erase(0, text_end);
                return;
            }
            pending_.erase(0, text_end + hit.marker->text.size());
            apply(hit.marker->action, delta);
        }
    }

    std::string& text_out(ParsedDelta& delta) { return mode_ == Mode::reasoning ? delta.reasoning : delta.content; }

    void emit(ParsedDelta& delta, std::string_view text) {
        if (mode_ == Mode::tool_call || mode_ == Mode::bare_function) {
            call_body_ += text;
            return;
        }
        if (lead_ != Lead::keep) {
            const size_t keep = text.find_first_not_of(lead_ == Lead::newlines ? "\n" : kWhitespace);
            if (keep == std::string_view::npos) return;
            text.remove_prefix(keep);
            lead_ = Lead::keep;
        }
        const size_t last = text.find_last_not_of(kWhitespace);
        if (last == std::string_view::npos) {
            held_space_ += text;
            return;
        }
        std::string& out = text_out(delta);
        out += held_space_;
        out += text.substr(0, last + 1);
        held_space_.assign(text.substr(last + 1));
    }

    void apply(Action action, ParsedDelta& delta) {
        switch (action) {
            case Action::open_reasoning:
                text_out(delta) += held_space_;
                held_space_.clear();
                mode_ = Mode::reasoning;
                lead_ = Lead::newlines;
                break;
            case Action::close_reasoning:
                held_space_.clear();
                mode_ = Mode::content;
                lead_ = Lead::newlines;
                break;
            case Action::open_call:
                held_space_.clear();
                mode_ = Mode::tool_call;
                break;
            case Action::open_bare_function:
                held_space_.clear();
                mode_ = Mode::bare_function;
                break;
            case Action::close_call:
                complete_call(delta);
                mode_ = Mode::content;
                break;
            case Action::drop:
                break;
        }
    }

    std::string call_opener() const { return std::string(mode_ == Mode::bare_function ? kFunctionOpen : kCallOpen); }

    // A body that is not one well-formed function is surfaced as content, markers included, rather than lost.
    void complete_call(ParsedDelta& delta) {
        const std::string body = mode_ == Mode::bare_function ? call_opener() + call_body_ : call_body_;
        if (auto function = parse_function(body)) {
            delta.tool_calls.push_back(ToolCall{id_prefix_ + std::to_string(calls_made_++), std::move(function->first),
                                                std::move(function->second)});
            lead_ = Lead::whitespace;
        } else {
            delta.content +=
                call_opener() + call_body_ + std::string(mode_ == Mode::bare_function ? kFunctionClose : kCallClose);
            lead_ = Lead::keep;
        }
        call_body_.clear();
    }

    const std::string id_prefix_;  // random per response; the call counter makes ids unique within it
    size_t calls_made_ = 0;
    Mode mode_ = Mode::content;
    Lead lead_ = Lead::keep;
    std::string pending_;     // text not yet classified: at most a possible marker prefix
    std::string held_space_;  // trailing whitespace of content or reasoning, released once more text follows
    std::string call_body_;   // the open tool call's text so far
};

}  // namespace

std::unique_ptr<OutputParser> make_qwen3_5_parser(std::string_view prompt_tail) {
    return std::make_unique<Qwen35Parser>(prompt_tail);
}

}  // namespace xs::chat
