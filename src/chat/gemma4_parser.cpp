// Gemma 4 output dialect. The chat template renders an assistant turn as
//   [<|channel>thought\n REASONING \n<channel|>] CONTENT [<|tool_call>call:NAME{ARGS}<tool_call|>]...
// and this parser inverts that, one feed() at a time. End-of-turn tokens (<turn|>, <|tool_response>, <eos>) are
// stripped by the engine before text reaches here.

#include <optional>
#include <random>
#include <span>
#include <string>

#include "chat/gemma4_args.h"
#include "chat/output_parser.h"

namespace xs::chat {
namespace {

enum class Mode { content, reasoning, tool_call };
enum class Action { open_reasoning, close_reasoning, open_call, close_call, drop };

struct Marker {
    std::string_view text;
    Action action;
};

// Per mode, the markers that change it. Longer spellings come with their shorter ones so the template's "\n" around
// the reasoning is not part of the reasoning; stray closers in content are dropped.
constexpr Marker kContentMarkers[] = {
    {"<|channel>thought\n", Action::open_reasoning},
    {"<|channel>", Action::open_reasoning},
    {"<|tool_call>", Action::open_call},
    {"<channel|>", Action::drop},
    {"<tool_call|>", Action::drop},
};
constexpr Marker kReasoningMarkers[] = {
    {"\n<channel|>", Action::close_reasoning},
    {"<channel|>", Action::close_reasoning},
    {"<|tool_call>", Action::open_call},
};
constexpr Marker kToolCallMarkers[] = {{"<tool_call|>", Action::close_call}};

std::span<const Marker> markers_for(Mode mode) {
    switch (mode) {
        case Mode::content:
            return kContentMarkers;
        case Mode::reasoning:
            return kReasoningMarkers;
        case Mode::tool_call:
            return kToolCallMarkers;
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
    for (size_t pos = 0; pos < buffer.size(); ++pos) {
        const std::string_view rest = buffer.substr(pos);
        const Marker* longest = nullptr;
        bool could_grow = false;
        for (const Marker& m : markers) {
            if (rest.starts_with(m.text)) {
                if (longest == nullptr || m.text.size() > longest->text.size()) longest = &m;
            } else if (!at_end && m.text.starts_with(rest)) {
                could_grow = true;
            }
        }
        if (could_grow) return {pos, nullptr};
        if (longest != nullptr) return {pos, longest};
    }
    return {};
}

std::string random_hex(size_t digits) {
    std::random_device rd;
    std::uniform_int_distribution<int> nibble(0, 15);
    std::string out;
    for (size_t i = 0; i < digits; ++i)
        out += "0123456789abcdef"[nibble(rd)];
    return out;
}

class Gemma4Parser final : public OutputParser {
public:
    explicit Gemma4Parser(std::string_view prompt_tail) : id_prefix_("call_" + random_hex(8)) {
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
        // An unclosed call (generation cut short) is kept if its body is already a complete call, else it is content.
        if (mode_ == Mode::tool_call) complete_call(delta);
        mode_ = Mode::content;
        return delta;
    }

private:
    // The template ends some prompts inside a thought channel ("<|channel>thought\n" after a tool response with
    // thinking on). Replaying the prompt from its last channel opener puts the parser where the prompt left off.
    void start_after(std::string_view prompt_tail) {
        const size_t opener = prompt_tail.rfind("<|channel>");
        if (opener == std::string_view::npos) return;
        const std::string_view from_opener = prompt_tail.substr(opener);
        pending_ = from_opener;
        ParsedDelta prompt_text;
        drain(prompt_text, /*at_end=*/false);
        // Anything still held is prompt text, not model output, unless the opener itself is still incomplete.
        if (pending_.size() != from_opener.size()) pending_.clear();
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

    void emit(ParsedDelta& delta, std::string_view text) {
        switch (mode_) {
            case Mode::content:
                delta.content += text;
                break;
            case Mode::reasoning:
                delta.reasoning += text;
                break;
            case Mode::tool_call:
                call_body_ += text;
                break;
        }
    }

    void apply(Action action, ParsedDelta& delta) {
        switch (action) {
            case Action::open_reasoning:
                mode_ = Mode::reasoning;
                break;
            case Action::close_reasoning:
                mode_ = Mode::content;
                break;
            case Action::open_call:
                mode_ = Mode::tool_call;
                break;
            case Action::close_call:
                complete_call(delta);
                mode_ = Mode::content;
                break;
            case Action::drop:
                break;
        }
    }

    // A body that is not "call:NAME{ARGS}" is surfaced as content (without its markers) rather than lost.
    void complete_call(ParsedDelta& delta) {
        if (auto call = parse_call(call_body_)) {
            delta.tool_calls.push_back(std::move(*call));
        } else {
            delta.content += call_body_;
        }
        call_body_.clear();
    }

    std::optional<ToolCall> parse_call(std::string_view body) {
        constexpr std::string_view prefix = "call:";
        if (!body.starts_with(prefix)) return std::nullopt;
        body.remove_prefix(prefix.size());
        const size_t brace = body.find('{');
        if (brace == 0 || brace == std::string_view::npos) return std::nullopt;
        const std::string_view name = body.substr(0, brace);
        if (name.find_first_of(" \t\r\n") != std::string_view::npos) return std::nullopt;
        auto arguments = gemma4::args_to_json(body.substr(brace));
        if (!arguments) return std::nullopt;
        return ToolCall{id_prefix_ + std::to_string(calls_made_++), std::string(name), std::move(*arguments)};
    }

    const std::string id_prefix_;  // random per response; the call counter makes ids unique within it
    size_t calls_made_ = 0;
    Mode mode_ = Mode::content;
    std::string pending_;    // text not yet classified: at most a possible marker prefix
    std::string call_body_;  // the open tool call's text so far
};

}  // namespace

std::unique_ptr<OutputParser> make_gemma4_parser(std::string_view prompt_tail) {
    return std::make_unique<Gemma4Parser>(prompt_tail);
}

}  // namespace xs::chat
