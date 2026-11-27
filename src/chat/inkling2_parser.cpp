// Inkling 2 output dialect. The chat template renders an assistant turn as messages, each
//   <|message_model|>[NAME]<|content_KIND|> BODY <|end_message|>
// of kind thinking (reasoning), text (content) or invoke_tool_json (a tool call NAME, its body
// {"name":NAME,"args":{...}}), and closes the turn with <|content_model_end_sampling|>. The model writes the same,
// its <|message_model|> included, and this parser inverts it one feed() at a time.

#include <cctype>
#include <random>
#include <string>

#include <nlohmann/json.hpp>

#include "chat/output_parser.h"

namespace xs::chat {
namespace {

constexpr std::string_view kMessage = "<|message_model|>";
constexpr std::string_view kEnd = "<|end_message|>";
constexpr std::string_view kEndTurn = "<|content_model_end_sampling|>";
constexpr std::string_view kKindOpen = "<|content_";
constexpr std::string_view kClose = "|>";

std::string random_hex(size_t digits) {
    std::random_device rd;
    std::uniform_int_distribution<int> nibble(0, 15);
    std::string out;
    for (size_t i = 0; i < digits; ++i)
        out += "0123456789abcdef"[nibble(rd)];
    return out;
}

class Inkling2Parser final : public OutputParser {
public:
    explicit Inkling2Parser(std::string_view prompt_tail) : id_prefix_("call_" + random_hex(8)) {
        // A prompt that ends inside a message the model continues: its header, or a body of some kind.
        const size_t message = prompt_tail.rfind(kMessage);
        if (message == std::string_view::npos || prompt_tail.find(kEnd, message) != std::string_view::npos) return;
        const std::string_view opened = prompt_tail.substr(message + kMessage.size());
        const size_t kind = opened.find(kKindOpen);
        const size_t close = kind == std::string_view::npos ? kind : opened.find(kClose, kind);
        if (close == std::string_view::npos) {
            where_ = Where::header;
            name_.assign(opened);
        } else {
            name_.assign(opened.substr(0, kind));
            open(opened.substr(kind + kKindOpen.size(), close - kind - kKindOpen.size()));
        }
    }

    ParsedDelta feed(std::string_view text) override {
        pending_ += text;
        ParsedDelta delta;
        drain(delta, false);
        return delta;
    }
    ParsedDelta finish() override {
        ParsedDelta delta;
        drain(delta, true);
        if (where_ == Where::call) close_call(delta);  // a call the model never closed, if its body is whole
        return delta;
    }
    std::string_view reasoning_end() const override { return where_ == Where::thinking ? kEnd : ""; }

private:
    enum class Where { between, header, thinking, text, call };

    void open(std::string_view kind) {
        where_ = kind == "thinking" ? Where::thinking : kind == "invoke_tool_json" ? Where::call : Where::text;
        if (where_ == Where::text && wrote_text_) separate_ = true;  // a second text message: on a line of its own
    }

    void emit(ParsedDelta& delta, std::string_view text) {
        if (text.empty()) return;
        switch (where_) {
            case Where::thinking:
                delta.reasoning += text;
                break;
            case Where::call:
                body_ += text;
                break;
            case Where::header:
                name_ += text;
                break;
            case Where::text:
            case Where::between:  // text outside any message: the model's all the same
                if (separate_) delta.content += "\n";
                separate_ = false;
                wrote_text_ = true;
                delta.content += text;
                break;
        }
    }

    // A tool call's body as a call; one that is not the JSON the template writes goes to the content as it is.
    void close_call(ParsedDelta& delta) {
        const nlohmann::json parsed = nlohmann::json::parse(body_, nullptr, false);
        const bool object = parsed.is_object();
        const bool named = object && parsed.contains("name") && parsed.at("name").is_string();
        std::string name = named ? parsed.at("name").get<std::string>() : name_;
        if (!object || name.empty()) {
            where_ = Where::text;
            emit(delta, body_);
        } else {
            const nlohmann::json args = parsed.contains("args") && parsed.at("args").is_object()
                                            ? parsed.at("args")
                                            : nlohmann::json::object();
            delta.tool_calls.push_back({id_prefix_ + "_" + std::to_string(calls_++), std::move(name), args.dump()});
        }
        body_.clear();
    }

    void drain(ParsedDelta& delta, bool at_end) {
        for (;;) {
            const size_t at = pending_.find("<|");
            if (at == std::string::npos) {
                // A trailing '<' may open a marker.
                const size_t keep = !at_end && !pending_.empty() && pending_.back() == '<' ? 1 : 0;
                emit(delta, std::string_view(pending_).substr(0, pending_.size() - keep));
                pending_.erase(0, pending_.size() - keep);
                return;
            }
            emit(delta, std::string_view(pending_).substr(0, at));
            pending_.erase(0, at);
            // A marker is <| a name of letters, digits and underscores |>; anything else after "<|" is text.
            size_t close = 2;
            while (close < pending_.size() && (std::isalnum(static_cast<unsigned char>(pending_[close])) != 0 ||
                                               pending_[close] == '_')) {
                ++close;
            }
            const bool whole = pending_.compare(close, kClose.size(), kClose) == 0 && close > 2;
            if (!whole) {
                // Still growing into one (a name so far, or its closing '|'), or not one.
                const bool growing = close == pending_.size() || (close + 1 == pending_.size() && pending_[close] == '|');
                if (growing && !at_end) return;
                emit(delta, std::string_view(pending_).substr(0, growing ? pending_.size() : 2));
                pending_.erase(0, growing ? pending_.size() : 2);
                if (pending_.empty()) return;
                continue;
            }
            const std::string marker = pending_.substr(0, close + kClose.size());
            pending_.erase(0, marker.size());
            if (marker == kMessage) {
                if (where_ == Where::call) close_call(delta);
                where_ = Where::header;
                name_.clear();
            } else if (marker == kEnd) {
                if (where_ == Where::call) close_call(delta);
                where_ = Where::between;
            } else if (marker == kEndTurn) {
                if (where_ == Where::call) close_call(delta);
                where_ = Where::between;
            } else if (where_ == Where::header && marker.starts_with(kKindOpen)) {
                const size_t kind = marker.size() - kKindOpen.size() - kClose.size();
                open(std::string_view(marker).substr(kKindOpen.size(), kind));
            } else {
                emit(delta, marker);  // not a marker of the dialect: text
            }
        }
    }

    std::string pending_;  // text not yet classified: it may end inside a marker
    Where where_ = Where::between;
    std::string name_;     // the open message's header (a tool call's name)
    std::string body_;     // a tool call's body so far
    bool wrote_text_ = false;
    bool separate_ = false;
    const std::string id_prefix_;
    int calls_ = 0;
};

}  // namespace

std::unique_ptr<OutputParser> make_inkling2_parser(std::string_view prompt_tail) {
    return std::make_unique<Inkling2Parser>(prompt_tail);
}

}  // namespace xs::chat
