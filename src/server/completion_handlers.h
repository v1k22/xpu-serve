#pragma once

#include <httplib/httplib.h>

#include <string>
#include <vector>

#include "engine/service.h"
#include "server/openai_request.h"

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
    };
    Prompt chat_prompt(const ChatRequest& chat) const;
    void respond(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    void write_full(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    // Returns false when the client went away before the stream was complete.
    bool write_stream(const Job& job, const httplib::Request& req, httplib::DataSink& sink) const;

    InferenceService& service_;
};

}  // namespace xs::server
