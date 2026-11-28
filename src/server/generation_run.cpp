#include "server/generation_run.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
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

std::vector<GenerationOutcome> run_choices(InferenceService& service, std::span<const GenerateRequest> requests,
                                           std::span<TokenOutput* const> outputs,
                                           const std::function<bool(int, const OutputPiece&)>& on_piece,
                                           const std::function<void(int, const GenerationOutcome&)>& on_done,
                                           const std::function<bool()>& client_alive) {
    const size_t n = requests.size();
    std::atomic<bool> client_gone = false;
    // What the generating threads hand over: a piece of choice `index`, or (done) the end of it.
    struct Event {
        int index = 0;
        OutputPiece piece;
        bool done = false;
    };
    Channel<Event> channel;
    std::vector<GenerationOutcome> outcomes(n);
    std::mutex started_mutex;
    std::condition_variable started_cv;
    bool first_in = false;  // choice 0's prompt is in: the others may start

    const auto let_others_start = [&] {
        {
            std::lock_guard lock(started_mutex);
            first_in = true;
        }
        started_cv.notify_all();
    };
    std::vector<std::jthread> workers;
    std::atomic<size_t> open = n;
    for (size_t i = 0; i < n; ++i) {
        workers.emplace_back([&, i] {
            if (i > 0) {
                std::unique_lock lock(started_mutex);
                started_cv.wait(lock, [&] { return first_in; });
            }
            TokenOutput& output = *outputs[i];
            LogprobTexts texts(service.tokenizer());
            const TokenCallback on_token = [&](int32_t token, const TokenLogprobs* logprobs) {
                if (i == 0) let_others_start();
                if (client_gone.load()) return false;
                OutputPiece piece{output.push(token), {}};
                if (logprobs != nullptr) piece.logprobs.push_back(texts.entry(token, *logprobs));
                if (!piece.empty()) channel.push({static_cast<int>(i), std::move(piece), false});
                return !output.stop_string_hit();
            };
            GenerationOutcome& o = outcomes[i];
            o.result = generate_or_error(service, requests[i], on_token);
            if (i == 0) let_others_start();
            if (OutputPiece rest{output.finish(), {}}; !rest.empty()) channel.push({static_cast<int>(i), std::move(rest), false});
            o.stop_string_hit = output.stop_string_hit();
            o.stop_string = output.stop_string();
            channel.push({static_cast<int>(i), {}, true});
            if (--open == 0) channel.close();
        });
    }

    std::deque<Event> batch;
    auto next_alive_check = std::chrono::steady_clock::now() + kAlivePollInterval;
    while (channel.wait_take(batch, kAlivePollInterval)) {
        for (Event& e : batch) {
            if (e.done) {
                outcomes[static_cast<size_t>(e.index)].client_gone = client_gone.load();
                if (!client_gone) on_done(e.index, outcomes[static_cast<size_t>(e.index)]);
            } else if (!client_gone && !on_piece(e.index, e.piece)) {
                client_gone = true;
            }
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_alive_check) {
            if (!client_gone && !client_alive()) client_gone = true;
            next_alive_check = now + kAlivePollInterval;
        }
    }
    workers.clear();
    for (GenerationOutcome& o : outcomes) o.client_gone = client_gone.load();
    return outcomes;
}

GenerationOutcome run_generation(InferenceService& service, const GenerateRequest& request, TokenOutput& output,
                                 const std::function<bool(const OutputPiece&)>& on_piece,
                                 const std::function<bool()>& client_alive) {
    TokenOutput* const outputs[] = {&output};
    return run_choices(
               service, std::span<const GenerateRequest>(&request, 1), outputs,
               [&](int, const OutputPiece& piece) { return on_piece(piece); }, [](int, const GenerationOutcome&) {},
               client_alive)
        .front();
}

}  // namespace xs::server
