#include "server/generation_run.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <iterator>
#include <thread>
#include <utility>

#include "server/channel.h"

namespace xs::server {
namespace {

constexpr auto kAlivePollInterval = std::chrono::milliseconds(100);

GenerateResult generate_or_error(InferenceService& service, const GenerateRequest& request,
                                 const TokenCallback& on_token) {
    try {
        return service.generate(request, on_token);
    } catch (const std::exception& e) {
        GenerateResult failed;
        failed.finish = FinishReason::Error;
        failed.error = e.what();
        return failed;
    }
}

}  // namespace

void append_piece(OutputPiece& into, OutputPiece from) {
    append_delta(into.delta, std::move(from.delta));
    into.logprobs.insert(into.logprobs.end(), std::make_move_iterator(from.logprobs.begin()),
                         std::make_move_iterator(from.logprobs.end()));
}

GenerationOutcome run_generation(InferenceService& service, const GenerateRequest& request, TokenOutput& output,
                                 const std::function<bool(const OutputPiece&)>& on_piece,
                                 const std::function<bool()>& client_alive) {
    std::atomic<bool> client_gone = false;
    Channel<OutputPiece> channel;
    GenerateResult result;
    LogprobTexts texts(service.tokenizer());

    const TokenCallback on_token = [&](int32_t token, const TokenLogprobs* logprobs) {
        if (client_gone.load()) return false;
        OutputPiece piece{output.push(token), {}};
        if (logprobs != nullptr) piece.logprobs.push_back(texts.entry(token, *logprobs));
        if (!piece.empty()) channel.push(std::move(piece));
        return !output.stop_string_hit();
    };
    std::jthread worker([&] {
        result = generate_or_error(service, request, on_token);
        if (OutputPiece rest{output.finish(), {}}; !rest.empty()) channel.push(std::move(rest));
        channel.close();
    });

    std::deque<OutputPiece> batch;
    auto next_alive_check = std::chrono::steady_clock::now() + kAlivePollInterval;
    while (channel.wait_take(batch, kAlivePollInterval)) {
        for (const OutputPiece& piece : batch) {
            if (!client_gone && !on_piece(piece)) client_gone = true;
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_alive_check) {
            if (!client_gone && !client_alive()) client_gone = true;
            next_alive_check = now + kAlivePollInterval;
        }
    }
    worker.join();
    return {std::move(result), output.stop_string_hit(), output.stop_string(), client_gone.load()};
}

}  // namespace xs::server
