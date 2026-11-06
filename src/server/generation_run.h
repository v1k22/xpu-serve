#pragma once

#include <functional>

#include "chat/output_parser.h"
#include "engine/service.h"
#include "server/token_output.h"

namespace xs::server {

struct GenerationOutcome {
    GenerateResult result;  // a throwing service is reported as finish == Error
    bool stop_string_hit = false;
    bool client_gone = false;  // cancelled because on_delta failed or client_alive() turned false
};

// Runs one generation and hands its output to the calling (HTTP) thread. Tokens are turned into deltas on the engine
// thread, so a stop string cancels exactly at its match, while the engine never waits on a socket.
// `on_delta` runs on the calling thread and returns false when the client is gone; `client_alive` is polled every
// 100 ms (a non-stream response writes nothing until the end). Either one cancels the generation.
GenerationOutcome run_generation(InferenceService& service, const GenerateRequest& request, TokenOutput& output,
                                 const std::function<bool(const chat::ParsedDelta&)>& on_delta,
                                 const std::function<bool()>& client_alive);

}  // namespace xs::server
