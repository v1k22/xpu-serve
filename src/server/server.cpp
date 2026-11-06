#include "server/server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <httplib/httplib.h>

#include "server/completion_handlers.h"
#include "server/openai_format.h"
#include "server/openai_request.h"
#include "util/log.h"

namespace xs::server {
namespace {

using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;

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
    Impl(InferenceService& svc, ServerOptions opts) : service(svc), options(std::move(opts)), handlers(svc) {
        install_routes();
    }

    void install_routes();
    bool authorized(const httplib::Request& req) const;

    InferenceService& service;
    const ServerOptions options;
    const CompletionHandlers handlers;
    const int64_t started_at = unix_seconds_now();
    httplib::Server http;
    std::mutex bind_mutex;  // orders bind against stop(), so a stop() before run() is not lost
    bool stop_requested = false;
    std::atomic<int> port = 0;
};

bool Server::Impl::authorized(const httplib::Request& req) const {
    return options.api_key.empty() || req.path == "/health" ||
           req.get_header_value("Authorization") == "Bearer " + options.api_key;
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
    http.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(model_list_json(service.model(), started_at).dump(), "application/json");
    });
    http.Post("/v1/chat/completions",
              guarded([this](const httplib::Request& req, httplib::Response& res) { handlers.chat(req, res); }));
    http.Post("/v1/completions",
              guarded([this](const httplib::Request& req, httplib::Response& res) { handlers.completion(req, res); }));
}

Server::Server(InferenceService& service, ServerOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {}

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
    log::info("listening on http://", o.host, ":", impl_->port.load(), " (model ", impl_->service.model().id, ")");
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
