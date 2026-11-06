#include "chat/output_parser.h"

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

std::unique_ptr<OutputParser> make_output_parser(std::string_view dialect, std::string_view prompt_tail) {
    if (dialect == "plain") return std::make_unique<PlainParser>();
    if (dialect == "gemma4") return make_gemma4_parser(prompt_tail);
    fail("unknown output dialect '", dialect, "' (known: gemma4, plain)");
}

}  // namespace xs::chat
