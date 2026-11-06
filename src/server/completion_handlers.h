#pragma once

#include <httplib/httplib.h>

#include "engine/service.h"

namespace xs::server {

// POST /v1/chat/completions and /v1/completions. Handlers throw InvalidRequest for client errors (400) and xs::Error
// for engine failures (500); once a stream has started, a failure is sent as an SSE error event instead.
class CompletionHandlers {
public:
    explicit CompletionHandlers(InferenceService& service) : service_(service) {}

    void chat(const httplib::Request& req, httplib::Response& res) const;
    void completion(const httplib::Request& req, httplib::Response& res) const;

private:
    struct Job;
    void respond(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    void write_full(const Job& job, const httplib::Request& req, httplib::Response& res) const;
    // Returns false when the client went away before the stream was complete.
    bool write_stream(const Job& job, const httplib::Request& req, httplib::DataSink& sink) const;

    InferenceService& service_;
};

}  // namespace xs::server
