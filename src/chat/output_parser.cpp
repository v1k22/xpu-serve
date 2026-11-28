#include "chat/output_parser.h"

#include <random>

#include "util/error.h"

namespace xs::chat {
namespace {

class PlainParser final : public OutputParser {
public:
    ParsedDelta feed(std::string_view text) override {
        ParsedDelta d;
        d.content.assign(text);
        return d;
    }
    ParsedDelta finish() override { return {}; }
};

}  // namespace

// Dialect parsers register here; each lives in its own file in this directory.
std::unique_ptr<OutputParser> make_gemma4_parser(std::string_view prompt_tail);
std::unique_ptr<OutputParser> make_qwen3_parser(std::string_view prompt_tail);
std::unique_ptr<OutputParser> make_qwen3_5_parser(std::string_view prompt_tail);
std::unique_ptr<OutputParser> make_inkling2_parser(std::string_view prompt_tail);
std::unique_ptr<OutputParser> make_llama3_parser(bool tools);

std::unique_ptr<OutputParser> make_output_parser(std::string_view dialect, std::string_view prompt_tail, bool tools) {
    if (dialect == "plain") return std::make_unique<PlainParser>();
    if (dialect == "gemma4") return make_gemma4_parser(prompt_tail);
    if (dialect == "qwen3") return make_qwen3_parser(prompt_tail);
    if (dialect == "qwen3_5") return make_qwen3_5_parser(prompt_tail);
    if (dialect == "inkling2") return make_inkling2_parser(prompt_tail);
    if (dialect == "llama3") return make_llama3_parser(tools);
    fail("unknown output dialect '", dialect, "' (known: gemma4, inkling2, llama3, plain, qwen3, qwen3_5)");
}

DirectAnswer direct_answer(std::string_view dialect) {
    make_output_parser(dialect);  // a known dialect
    if (dialect == "inkling2") {
        return {{{"enable_thinking", false}, {"reasoning_effort", "none"}}, "<|message_model|><|content_text|>"};
    }
    return {{{"enable_thinking", false}}, ""};
}

std::string call_id_prefix() {
    std::random_device rd;
    std::uniform_int_distribution<int> nibble(0, 15);
    std::string out = "call_";
    for (int i = 0; i < 8; ++i)
        out += "0123456789abcdef"[nibble(rd)];
    return out;
}

AnswerStart answer_start(std::string_view dialect, std::string_view prompt_tail) {
    const auto parser = make_output_parser(dialect, prompt_tail);
    const OutputParser::Thought own = parser->own_thought();
    return {std::string(parser->reasoning_end()), std::string(own.open), std::string(own.close),
            direct_answer(dialect).opening};
}

}  // namespace xs::chat
