#pragma once

#include <memory>
#include <string>

#include "engine/service.h"
#include "policy/service.h"

namespace xs::server {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;  // empty = no auth
    int http_threads = 8;
};

// HTTP server over one loaded model. A language model gets the OpenAI-compatible endpoints: POST
// /v1/chat/completions (stream and non-stream), POST /v1/completions, GET /v1/models; a robot policy the policy API
// (server/policy_handlers.h): GET /v1/policy/info, POST /v1/policy/act, GET /ready. Both: GET /health.
class Server {
public:
    Server(InferenceService& service, ServerOptions options);
    Server(PolicyService& policy, ServerOptions options);
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
