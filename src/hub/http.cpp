#include "hub/http.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <httplib/httplib.h>

#include "util/error.h"

namespace xs::hub {

namespace {

std::string env(const char* lower, const char* upper) {
    for (const char* name : {lower, upper}) {
        if (const char* v = std::getenv(name); v != nullptr && *v != '\0') return v;
    }
    return {};
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

// The redirect target of `location` seen from `from`: absolute, or a path on the same origin.
std::string resolve_location(const Url& from, const std::string& location) {
    if (location.find("://") != std::string::npos) return location;
    if (location.rfind("//", 0) == 0) return from.scheme + ":" + location;
    if (!location.empty() && location[0] == '/') return from.origin() + location;
    const size_t slash = from.target.rfind('/');
    return from.origin() + from.target.substr(0, slash + 1) + location;
}

}  // namespace

Proxy Proxy::from_env() {
    Proxy p;
    p.http = env("http_proxy", "HTTP_PROXY");
    p.https = env("https_proxy", "HTTPS_PROXY");
    const std::string list = env("no_proxy", "NO_PROXY");
    for (size_t pos = 0; pos <= list.size();) {
        const size_t comma = std::min(list.find(',', pos), list.size());
        std::string entry = list.substr(pos, comma - pos);
        entry.erase(0, entry.find_first_not_of(" \t"));
        entry.erase(entry.find_last_not_of(" \t") + 1);
        if (!entry.empty()) p.no_proxy.push_back(entry);
        pos = comma + 1;
    }
    return p;
}

std::string Url::origin() const {
    const bool default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
    return scheme + "://" + host + (default_port ? "" : ":" + std::to_string(port));
}

Url parse_url(const std::string& url) {
    Url u;
    const size_t sep = url.find("://");
    if (sep == std::string::npos) fail("not a URL: '", url, "'");
    u.scheme = lower(url.substr(0, sep));
    if (u.scheme != "http" && u.scheme != "https") fail("not an http or https URL: '", url, "'");
    const size_t path = url.find('/', sep + 3);
    std::string authority = url.substr(sep + 3, path == std::string::npos ? std::string::npos : path - sep - 3);
    if (const size_t at = authority.rfind('@'); at != std::string::npos) authority.erase(0, at + 1);
    u.port = u.scheme == "https" ? 443 : 80;
    const size_t colon = authority.rfind(':');
    if (colon != std::string::npos && authority.find(']', colon) == std::string::npos) {
        u.port = std::atoi(authority.c_str() + colon + 1);
        authority.resize(colon);
    }
    u.host = authority;
    if (u.host.empty()) fail("no host in URL '", url, "'");
    u.target = path == std::string::npos ? "/" : url.substr(path);
    return u;
}

const std::string* Response::header(const std::string& lower_name) const {
    const auto it = headers.find(lower_name);
    return it == headers.end() ? nullptr : &it->second;
}

struct HttpClient::Impl {
    Proxy proxy;
    int timeout = 60;
    std::map<std::string, std::unique_ptr<httplib::Client>> clients;  // by origin

    httplib::Client& client(const Url& u) {
        auto& c = clients[u.origin()];
        if (c) return *c;
        c = std::make_unique<httplib::Client>(u.origin());
        c->set_connection_timeout(timeout, 0);
        c->set_read_timeout(timeout, 0);
        c->set_write_timeout(timeout, 0);
        c->set_keep_alive(true);
        c->set_follow_location(false);
        const std::string& via = u.scheme == "https" ? proxy.https : proxy.http;
        if (!via.empty()) {
            const Url p = parse_url(via);
            c->set_proxy(p.host, p.port);
            const size_t sep = via.find("://") + 3;
            const size_t at = via.find('@', sep);
            if (at != std::string::npos && at < via.find('/', sep)) {
                const std::string user_pass = via.substr(sep, at - sep);
                const size_t colon = user_pass.find(':');
                c->set_proxy_basic_auth(httplib::decode_uri_component(user_pass.substr(0, colon)),
                                        colon == std::string::npos
                                            ? std::string()
                                            : httplib::decode_uri_component(user_pass.substr(colon + 1)));
            }
            c->set_no_proxy(proxy.no_proxy);
        }
        return *c;
    }
};

HttpClient::HttpClient(Proxy proxy, int timeout_seconds) : impl_(std::make_unique<Impl>()) {
    impl_->proxy = std::move(proxy);
    impl_->timeout = timeout_seconds;
}

HttpClient::~HttpClient() = default;

Response HttpClient::get(const std::string& url, const Headers& headers, const std::string& auth,
                         const std::string& auth_origin, bool follow,
                         const std::function<bool(const char*, size_t)>& sink) {
    Response r;
    std::string current = url;
    for (int hop = 0;; ++hop) {
        const Url u = parse_url(current);
        httplib::Headers h(headers.begin(), headers.end());
        if (!auth.empty() && lower(u.origin()) == lower(auth_origin)) h.emplace("Authorization", auth);
        int status = 0;
        httplib::Headers got;
        const auto result = impl_->client(u).Get(
            u.target, h,
            [&](const httplib::Response& response) {
                status = response.status;
                got = response.headers;
                return true;
            },
            [&](const char* data, size_t n) {
                if (sink && status >= 200 && status < 300) return sink(data, n);
                r.body.append(data, n);
                return true;
            });
        r.url = current;
        if (!result) {
            if (status != 0 && result.error() == httplib::Error::Canceled) {
                r.status = status;
                r.error = "the download was stopped";
                return r;
            }
            r.status = 0;
            r.error = httplib::to_string(result.error());
            return r;
        }
        r.status = status;
        r.headers.clear();
        for (const auto& [k, v] : got)
            r.headers[lower(k)] = v;
        const std::string* location = r.header("location");
        if (!follow || status < 300 || status >= 400 || location == nullptr) return r;
        if (hop == 8) fail("too many redirects from ", url);
        current = resolve_location(u, *location);
        r.body.clear();
    }
}

}  // namespace xs::hub
