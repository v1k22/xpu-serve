#include "server/token_output.h"

#include <iterator>
#include <utility>

namespace xs::server {

void append_delta(chat::ParsedDelta& into, chat::ParsedDelta from) {
    into.content += from.content;
    into.reasoning += from.reasoning;
    into.tool_calls.insert(into.tool_calls.end(), std::make_move_iterator(from.tool_calls.begin()),
                           std::make_move_iterator(from.tool_calls.end()));
}

TokenOutput::TokenOutput(const tok::Tokenizer& tokenizer, std::unique_ptr<chat::OutputParser> parser,
                         std::vector<std::string> stop_strings)
    : decoder_(tokenizer, /*skip_special=*/false), parser_(std::move(parser)), stops_(std::move(stop_strings)) {}

chat::ParsedDelta TokenOutput::push(int32_t token) {
    if (stop_string_hit()) return {};
    return apply_stop_strings(parser_->feed(decoder_.push(token)));
}

chat::ParsedDelta TokenOutput::finish() {
    if (stop_string_hit()) return {};
    chat::ParsedDelta delta = parser_->feed(decoder_.flush());
    append_delta(delta, parser_->finish());
    delta = apply_stop_strings(std::move(delta));
    delta.content += stops_.flush();
    return delta;
}

chat::ParsedDelta TokenOutput::apply_stop_strings(chat::ParsedDelta delta) {
    delta.content = stops_.push(delta.content);
    // Reasoning comes before the answer, but a tool call in the same delta may follow the match: drop it.
    if (stop_string_hit()) delta.tool_calls.clear();
    return delta;
}

}  // namespace xs::server
