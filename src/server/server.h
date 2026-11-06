#pragma once

#include <memory>
#include <string>

#include "engine/service.h"

namespace xs::server {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;  // empty = no auth
    int http_threads = 8;
};

// OpenAI-compatible HTTP server over an InferenceService.
// Endpoints (v0.1): POST /v1/chat/completions (stream and non-stream), POST /v1/completions, GET /v1/models,
// GET /health.
class Server {
public:
    Server(InferenceService& service, ServerOptions options);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binds and serves until stop(). Throws xs::Error if the address cannot be bound.
    void run();
    void stop();
    // Port actually bound (useful with port 0 in tests); valid once run() has started listening.
    int bound_port() const;
    bool wait_until_listening(int timeout_ms) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace xs::server
