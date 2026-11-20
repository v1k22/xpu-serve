#include "server/server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include <httplib/httplib.h>

#include "server/completion_handlers.h"
#include "server/decision_handlers.h"
#include "server/embedding_handlers.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "server/policy_handlers.h"
#include "server/realtime_session.h"
#include "server/transcription_handlers.h"
#include "util/log.h"

namespace xs::server {
namespace {

using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;

// The subprotocols a WebSocket upgrade offers (Sec-WebSocket-Protocol, comma-separated).
std::set<std::string> websocket_protocols(const httplib::Request& req) {
    std::set<std::string> out;
    std::string list = req.get_header_value("Sec-WebSocket-Protocol");
    for (size_t at = 0; at <= list.size();) {
        const size_t comma = std::min(list.find(',', at), list.size());
        std::string p = list.substr(at, comma - at);
        p.erase(0, p.find_first_not_of(' '));
        p.erase(p.find_last_not_of(' ') + 1);
        if (!p.empty()) out.insert(p);
        at = comma + 1;
    }
    return out;
}

void set_error(httplib::Response& res, int status, std::string_view message, std::string_view type,
               std::string_view param = {}, std::string_view code = {}) {
    res.status = status;
    res.set_content(error_body(message, type, param, code).dump(), "application/json");
}

// Maps handler exceptions to OpenAI error responses.
Handler guarded(Handler handler) {
    return [handler = std::move(handler)](const httplib::Request& req, httplib::Response& res) {
        try {
            handler(req, res);
        } catch (const InvalidRequest& e) {
            log::info(req.method, " ", req.path, " -> 400: ", e.what());
            set_error(res, 400, e.what(), "invalid_request_error", e.param());
        } catch (const std::exception& e) {
            log::error(req.method, " ", req.path, " -> 500: ", e.what());
            set_error(res, 500, e.what(), "server_error");
        }
    };
}

}  // namespace

struct Server::Impl {
    Impl(InferenceService& svc, ServerOptions opts) : service(&svc), options(std::move(opts)) {
        handlers.emplace(svc);
        if (svc.transcribes()) transcription_handlers.emplace(svc);
        install_routes();
    }
    Impl(PolicyService& p, ServerOptions opts) : policy(&p), options(std::move(opts)) {
        policy_handlers.emplace(p);
        install_routes();
    }
    Impl(DecisionService& d, ServerOptions opts) : decisions(&d), options(std::move(opts)) {
        decision_handlers.emplace(d);
        install_routes();
    }
    Impl(EmbeddingService& e, ServerOptions opts) : embeddings(&e), options(std::move(opts)) {
        embedding_handlers.emplace(e);
        install_routes();
    }

    void install_routes();
    void serve_realtime(const httplib::Request& req, httplib::ws::WebSocket& ws);
    bool authorized(const httplib::Request& req) const;
    std::string model_id() const {
        if (service != nullptr) return service->model().id;
        if (policy != nullptr) return policy->spec().id;
        return decisions != nullptr ? decisions->spec().id : embeddings->spec().id;
    }

    InferenceService* service = nullptr;  // one of the four
    PolicyService* policy = nullptr;
    DecisionService* decisions = nullptr;
    EmbeddingService* embeddings = nullptr;
    const ServerOptions options;
    std::optional<CompletionHandlers> handlers;
    std::optional<TranscriptionHandlers> transcription_handlers;
    std::optional<PolicyHandlers> policy_handlers;
    std::optional<DecisionHandlers> decision_handlers;
    std::optional<EmbeddingHandlers> embedding_handlers;
    const int64_t started_at = unix_seconds_now();
    httplib::Server http;
    std::mutex bind_mutex;  // orders bind against stop(), so a stop() before run() is not lost
    bool stop_requested = false;
    std::atomic<int> port = 0;
};

bool Server::Impl::authorized(const httplib::Request& req) const {
    if (options.api_key.empty() || req.path == "/health" || req.path == "/ready" ||
        req.get_header_value("Authorization") == "Bearer " + options.api_key) {
        return true;
    }
    // A browser cannot set headers on a WebSocket: OpenAI's clients send the key as a subprotocol.
    return websocket_protocols(req).count("openai-insecure-api-key." + options.api_key) != 0;
}

// One realtime transcription session per connection, until the client closes it.
void Server::Impl::serve_realtime(const httplib::Request& req, httplib::ws::WebSocket& ws) {
    const std::string intent = req.get_param_value("intent");
    if (!intent.empty() && intent != "transcription") {
        ws.send(nlohmann::ordered_json{{"type", "error"},
                                       {"error",
                                        {{"type", "invalid_request_error"},
                                         {"code", "invalid_value"},
                                         {"message", "this server's realtime sessions transcribe audio"},
                                         {"param", "intent"}}}}
                    .dump());
        ws.close();
        return;
    }
    log::info("realtime: a session opened");
    {
        RealtimeSession session(*service, [&ws](const nlohmann::ordered_json& event) { ws.send(event.dump()); });
        std::string message;
        for (;;) {
            const httplib::ws::ReadResult r = ws.read(message);
            if (r == httplib::ws::Fail) break;
            // A binary frame is read as text too: what is not a JSON event gets an error event.
            if (r == httplib::ws::Text || r == httplib::ws::Binary) session.handle(message);
        }
    }
    log::info("realtime: a session closed");
}

void Server::Impl::install_routes() {
    // httplib's API takes ownership of a raw pointer here.
    const size_t threads = size_t(std::max(1, options.http_threads));
    http.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };

    http.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (authorized(req)) return httplib::Server::HandlerResponse::Unhandled;
        set_error(res, 401, "missing or invalid API key (send 'Authorization: Bearer <key>')", "invalid_request_error",
                  {}, "invalid_api_key");
        return httplib::Server::HandlerResponse::Handled;
    });
    // Errors raised by httplib itself (unknown route, malformed HTTP) get an OpenAI-shaped body too.
    http.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        if (res.status == 404) {
            set_error(res, 404, str_cat("unknown route: ", req.method, " ", req.path), "invalid_request_error");
        } else {
            set_error(res, res.status, httplib::status_message(res.status), "invalid_request_error");
        }
        return httplib::Server::HandlerResponse::Handled;
    });

    http.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    if (policy_handlers) {
        http.Get("/ready", [this](const httplib::Request& req, httplib::Response& res) { policy_handlers->ready(req, res); });
        http.Get("/v1/policy/info", guarded([this](const httplib::Request& req, httplib::Response& res) {
                     policy_handlers->info(req, res);
                 }));
        http.Post("/v1/policy/act", guarded([this](const httplib::Request& req, httplib::Response& res) {
                      policy_handlers->act(req, res);
                  }));
        return;
    }
    if (decision_handlers) {  // its errors have the System One API's bodies, written by the handler
        http.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
            decision_handlers->models(req, res, started_at);
        });
        http.Post("/v1/systemone",
                  [this](const httplib::Request& req, httplib::Response& res) { decision_handlers->decide(req, res); });
        return;
    }
    if (embedding_handlers) {
        http.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
            embedding_handlers->models(req, res, started_at);
        });
        http.Post("/v1/embeddings", guarded([this](const httplib::Request& req, httplib::Response& res) {
                      embedding_handlers->embed(req, res);
                  }));
        return;
    }
    http.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(model_list_json(service->model(), started_at).dump(), "application/json");
    });
    http.Post("/v1/chat/completions",
              guarded([this](const httplib::Request& req, httplib::Response& res) { handlers->chat(req, res); }));
    http.Post("/v1/completions",
              guarded([this](const httplib::Request& req, httplib::Response& res) { handlers->completion(req, res); }));
    if (transcription_handlers) {
        http.Post("/v1/audio/transcriptions", guarded([this](const httplib::Request& req, httplib::Response& res) {
                      transcription_handlers->transcribe(req, res);
                  }));
        http.WebSocket(
            "/v1/realtime",
            [this](const httplib::Request& req, httplib::ws::WebSocket& ws) { serve_realtime(req, ws); },
            [](const std::vector<std::string>& offered) {
                return std::find(offered.begin(), offered.end(), "realtime") != offered.end() ? "realtime" : "";
            });
        http.Post("/v1/audio/translations", guarded([this](const httplib::Request& req, httplib::Response& res) {
                      transcription_handlers->translate(req, res);
                  }));
    }
}

Server::Server(InferenceService& service, ServerOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {}

Server::Server(PolicyService& policy, ServerOptions options)
    : impl_(std::make_unique<Impl>(policy, std::move(options))) {}

Server::Server(DecisionService& decisions, ServerOptions options)
    : impl_(std::make_unique<Impl>(decisions, std::move(options))) {}

Server::Server(EmbeddingService& embeddings, ServerOptions options)
    : impl_(std::make_unique<Impl>(embeddings, std::move(options))) {}

Server::~Server() = default;

void Server::run() {
    const ServerOptions& o = impl_->options;
    {
        std::lock_guard lock(impl_->bind_mutex);
        if (impl_->stop_requested) return;
        const int port = o.port == 0 ? impl_->http.bind_to_any_port(o.host)
                                     : (impl_->http.bind_to_port(o.host, o.port) ? o.port : -1);
        if (port < 0) fail("cannot listen on ", o.host, ":", o.port, " (address in use or not available)");
        impl_->port = port;
    }
    log::info("listening on http://", o.host, ":", impl_->port.load(), " (model ", impl_->model_id(), ")");
    impl_->http.listen_after_bind();
}

void Server::stop() {
    std::lock_guard lock(impl_->bind_mutex);
    impl_->stop_requested = true;
    impl_->http.stop();
}

int Server::bound_port() const { return impl_->port; }

bool Server::wait_until_listening(int timeout_ms) const {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!impl_->http.is_running()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

}  // namespace xs::server
