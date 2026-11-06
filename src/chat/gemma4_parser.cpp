#include "chat/output_parser.h"
#include "util/error.h"

namespace xs::chat {

// Placeholder until the Gemma 4 dialect lands.
std::unique_ptr<OutputParser> make_gemma4_parser() { fail("the gemma4 output dialect is not implemented yet"); }

}  // namespace xs::chat
