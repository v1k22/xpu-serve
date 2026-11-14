#pragma once

// HTTP(S) GETs for downloads: proxies from the environment, redirects followed by hand (credentials only to the host
// they were meant for), a body streamed to a sink.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace xs::hub {

// Where requests go through: http_proxy / https_proxy (either case) per scheme, and the hosts of no_proxy that
// bypass them (as curl reads them).
struct Proxy {
    std::string http;   // http://[user:password@]host:port, or empty
    std::string https;
    std::vector<std::string> no_proxy;
    static Proxy from_env();
};

// A URL split into its origin ("https://host:port") and the rest ("/path?query").
struct Url {
    std::string scheme, host;
    int port = 0;
    std::string target;
    std::string origin() const;
};
Url parse_url(const std::string& url);

using Headers = std::map<std::string, std::string>;

struct Response {
    int status = 0;  // 0: no response (the error says why)
    Headers headers;  // names in lower case
    std::string body;  // when no sink took it
    std::string error;
    std::string url;  // the URL that answered, after redirects
    const std::string* header(const std::string& lower_name) const;
};

// Connections are kept per origin; one client per thread.
class HttpClient {
public:
    explicit HttpClient(Proxy proxy, int timeout_seconds = 60);
    ~HttpClient();
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // GET `url`. `auth` (an Authorization header value) goes only to `auth_origin` (Url::origin: scheme, host and
    // port), never to where a redirect leads elsewhere. Redirects are followed when `follow` (up to 8). A 2xx body goes
    // to `sink` when one is given (returning false aborts), else to Response::body. Other statuses keep their body (an
    // error message) in Response::body.
    Response get(const std::string& url, const Headers& headers, const std::string& auth, const std::string& auth_origin,
                 bool follow, const std::function<bool(const char*, size_t)>& sink = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace xs::hub
