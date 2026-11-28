#pragma once

#include <functional>
#include <span>
#include <vector>

#include "chat/output_parser.h"
#include "engine/service.h"
#include "server/logprobs.h"
#include "server/token_output.h"

namespace xs::server {

// What a generation's output adds at a time: a delta of its text and, when the request asked for log-probabilities,
// the entries of the tokens behind it (a piece a token then, its delta empty where the parser holds the token's text
// back: no token's entry is left out).
struct OutputPiece {
    chat::ParsedDelta delta;
    std::vector<LogprobEntry> logprobs;
    bool empty() const { return delta.empty() && logprobs.empty(); }
};

// Appends `from` to `into`: the deltas (append_delta) and the entries.
void append_piece(OutputPiece& into, OutputPiece from);

struct GenerationOutcome {
    GenerateResult result;  // a throwing service is reported as finish == Error
    bool stop_string_hit = false;
    std::string stop_string;  // the one that matched
    bool client_gone = false;  // cancelled because on_delta failed or client_alive() turned false
};

// Runs one generation and hands its output to the calling (HTTP) thread. Tokens are turned into pieces on the engine
// thread, so a stop string cancels exactly at its match, while the engine never waits on a socket.
// `on_piece` runs on the calling thread and returns false when the client is gone; `client_alive` is polled every
// 100 ms (a non-stream response writes nothing until the end). Either one cancels the generation.
GenerationOutcome run_generation(InferenceService& service, const GenerateRequest& request, TokenOutput& output,
                                 const std::function<bool(const OutputPiece&)>& on_piece,
                                 const std::function<bool()>& client_alive);

// The choices of one response (n): `requests[i]` generates choice i into `outputs[i]`. Choice 0 starts first and the
// others once its prompt is in (its first token, or its end), so that they find the prompt cached and copy it rather
// than each prefilling it (Engine::take_prefix); with sequences enough for all of them they then run together.
// `on_piece(i, piece)` and `on_done(i, outcome)` run on the calling thread, in the order the engine produced them; a
// false from `on_piece` or `client_alive` cancels every choice. Returns each choice's outcome.
std::vector<GenerationOutcome> run_choices(InferenceService& service, std::span<const GenerateRequest> requests,
                                           std::span<TokenOutput* const> outputs,
                                           const std::function<bool(int, const OutputPiece&)>& on_piece,
                                           const std::function<void(int, const GenerationOutcome&)>& on_done,
                                           const std::function<bool()>& client_alive);

}  // namespace xs::server
