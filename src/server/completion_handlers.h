#pragma once

#include <httplib/httplib.h>

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "engine/service.h"
#include "server/grammar_constraint.h"
#include "server/openai_request.h"

namespace xs::grammar {
class Vocabulary;
}

namespace xs::server {
class AutomatonCache;
}

namespace xs::server {

// POST /v1/chat/completions and /v1/completions (OpenAI), /v1/messages and /v1/messages/count_tokens (Anthropic).
// Handlers throw InvalidRequest for client errors (400) and xs::Error for engine failures (500); once a stream has
// started, a failure is sent as the format's error event instead.
class CompletionHandlers {
public:
    explicit CompletionHandlers(InferenceService& service) : service_(service) {}

    void chat(const httplib::Request& req, httplib::Response& res) const;
    void completion(const httplib::Request& req, httplib::Response& res) const;
    void messages(const httplib::Request& req, httplib::Response& res) const;
    void count_tokens(const httplib::Request& req, httplib::Response& res) const;

private:
    struct Job;
    // A chat request's prompt: its media prepared, the template rendered, tokenized, the media's runs expanded.
    struct Prompt {
        std::vector<int32_t> tokens;
        std::string text;
        std::vector<MediaInput> media;
        // Where the messages end in `tokens`, before the generation prompt (GenerateRequest::snapshot_at); 0 when the
        // template's rendering without that prompt is not where the prompt's tokens split.
        int64_t messages_end = 0;
    };
    Prompt chat_prompt(const ChatRequest& chat) const;
    // The constraint that holds the answer to `options.format` (null without one), the answer starting where the
    // dialect's does after a prompt ending in `prompt_tail`.
    std::shared_ptr<const OutputConstraint> output_constraint(const GenerationOptions& options,
                                                              std::string_view dialect,
                                                              std::string_view prompt_tail) const;
    // A chat reply's constraint: its tool calls in the dialect's syntax when tool_choice forces one (required, a
    // named function) or "auto" sits beside a response_format (the calls or the formatted answer), else
    // output_constraint's.
    std::shared_ptr<const OutputConstraint> chat_constraint(const ChatRequest& chat, std::string_view dialect,
                                                            std::string_view prompt_tail) const;
    const grammar::Vocabulary& vocabulary() const;
    void respond(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    void write_full(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    // Returns false when the client went away before the stream was complete.
    bool write_stream(const Job& job, const httplib::Request& req, httplib::DataSink& sink) const;

    InferenceService& service_;
    // The model's tokens as grammars read them, built for the first request with a format.
    mutable std::once_flag vocabulary_once_;
    mutable std::shared_ptr<const grammar::Vocabulary> vocabulary_;
    // The automata of the grammars requests hold to, shared by requests holding to the same one.
    std::shared_ptr<AutomatonCache> automata_ = std::make_shared<AutomatonCache>();
};

}  // namespace xs::server
