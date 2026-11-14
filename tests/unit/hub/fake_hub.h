#pragma once

// A Hugging Face Hub in the test process, enough for the hub client: the listing API, `resolve` URLs (a file kept in
// git served from the Hub, one kept in LFS redirected to a second server standing for the CDN), byte ranges,
// tokens, gated repositories, and faults on demand (a CDN that stops answering, a corrupted byte).

#include <doctest/doctest.h>

#include <atomic>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <httplib/httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include "hub/hf_hub.h"

namespace xs::test {

inline std::string sha256_hex(const std::string& data) {
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    EVP_Digest(data.data(), data.size(), md, &n, EVP_sha256(), nullptr);
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (unsigned int i = 0; i < n; ++i) {
        out += hex[md[i] >> 4];
        out += hex[md[i] & 15];
    }
    return out;
}

// Bytes that differ from position to position, so a range written at the wrong place shows.
inline std::string pattern_bytes(size_t n, unsigned seed) {
    std::string s(n, '\0');
    uint32_t x = seed * 2654435761u + 1;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        s[i] = static_cast<char>(x >> 24);
    }
    return s;
}

class FakeHub {
public:
    static constexpr const char* kCommit = "0123456789abcdef0123456789abcdef01234567";
    static constexpr const char* kToken = "hf_test_token";

    struct File {
        std::string content;
        bool lfs = true;
    };
    struct Repo {
        std::map<std::string, File> files;
        bool gated = false;   // its files need kToken
        bool private_ = false;  // its listing too
    };

    FakeHub() {
        hub_.Get(R"(/api/models/([^/]+)/([^/]+)/revision/([^/?]+))", [this](const auto& req, auto& res) { list(req, res); });
        hub_.Get(R"(/([^/]+)/([^/]+)/resolve/([^/]+)/(.+))", [this](const auto& req, auto& res) { resolve(req, res); });
        cdn_.Get(R"(/blob/([0-9a-f]+))", [this](const auto& req, auto& res) { blob(req, res); });
        hub_port_ = hub_.bind_to_any_port("127.0.0.1");
        cdn_port_ = cdn_.bind_to_any_port("127.0.0.1");
        hub_thread_ = std::thread([this] { hub_.listen_after_bind(); });
        cdn_thread_ = std::thread([this] { cdn_.listen_after_bind(); });
        hub_.wait_until_ready();
        cdn_.wait_until_ready();
    }
    ~FakeHub() {
        hub_.stop();
        cdn_.stop();
        hub_thread_.join();
        cdn_thread_.join();
    }
    FakeHub(const FakeHub&) = delete;
    FakeHub& operator=(const FakeHub&) = delete;

    void add(const std::string& repo, const std::string& path, std::string content, bool lfs = true) {
        std::lock_guard<std::mutex> lock(m_);
        repos_[repo].files[path] = File{std::move(content), lfs};
    }
    void set_gated(const std::string& repo) {
        std::lock_guard<std::mutex> lock(m_);
        repos_[repo].gated = true;
    }
    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(hub_port_); }

    // Options for a client of this hub: no proxy, the cache in `cache`, small ranges over a few connections.
    hub::HubOptions options(const std::filesystem::path& cache, std::string token = {}) const {
        hub::HubOptions o;
        o.endpoint = endpoint();
        o.cache = cache;
        o.token = std::move(token);
        o.connections = 3;
        o.chunk_bytes = 4096;
        o.attempts = 1;
        return o;
    }

    std::atomic<int> listings{0};
    std::atomic<int> file_requests{0};  // resolve and CDN requests together
    std::atomic<int> cdn_requests{0};
    std::atomic<int> cdn_auth{0};       // CDN requests that carried an Authorization header: never wanted
    std::atomic<int> hub_auth{0};       // Hub requests that carried the token
    std::atomic<int> cdn_limit{-1};         // >= 0: the CDN answers that many requests, then only 503s
    std::atomic<int> cdn_served{0};
    std::atomic<bool> corrupt{false};       // the CDN flips one byte of every answer
    std::atomic<bool> down{false};          // the Hub answers nothing usable (503) for listings

private:
    const Repo* find(const std::string& repo) const {
        const auto it = repos_.find(repo);
        return it == repos_.end() ? nullptr : &it->second;
    }

    bool authorized(const httplib::Request& req) {
        const bool ok = req.get_header_value("Authorization") == std::string("Bearer ") + kToken;
        if (ok) ++hub_auth;
        return ok;
    }

    void list(const httplib::Request& req, httplib::Response& res) {
        ++listings;
        if (down) {
            res.status = 503;
            return;
        }
        std::lock_guard<std::mutex> lock(m_);
        const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
        const Repo* r = find(repo);
        const bool authed = authorized(req);
        if (r == nullptr || (r->private_ && !authed)) {
            res.status = 401;
            res.set_content(R"({"error":"Repository Not Found"})", "application/json");
            return;
        }
        if (req.matches[3].str() != "main" && req.matches[3].str() != kCommit) {
            res.status = 404;
            res.set_content(R"({"error":"Revision Not Found"})", "application/json");
            return;
        }
        nlohmann::json siblings = nlohmann::json::array();
        for (const auto& [path, f] : r->files) {
            nlohmann::json s = {{"rfilename", path}, {"size", f.content.size()},
                                {"blobId", sha256_hex("git " + f.content).substr(0, 40)}};
            if (f.lfs) s["lfs"] = {{"sha256", sha256_hex(f.content)}, {"size", f.content.size()}};
            siblings.push_back(s);
        }
        nlohmann::json j = {{"id", repo}, {"sha", kCommit}, {"siblings", siblings}};
        j["gated"] = r->gated ? nlohmann::json("manual") : nlohmann::json(false);
        res.set_content(j.dump(), "application/json");
    }

    void resolve(const httplib::Request& req, httplib::Response& res) {
        ++file_requests;
        std::lock_guard<std::mutex> lock(m_);
        const std::string repo = req.matches[1].str() + "/" + req.matches[2].str();
        const Repo* r = find(repo);
        const auto it = r == nullptr ? decltype(r->files.end()){} : r->files.find(httplib::decode_uri_component(req.matches[4].str()));
        if (r == nullptr || req.matches[3].str() != kCommit || it == r->files.end()) {
            res.status = 404;
            return;
        }
        if (r->gated && !authorized(req)) {
            res.status = 403;
            res.set_content("Access to this repository is restricted", "text/plain");
            return;
        }
        if (it->second.lfs) {
            res.status = 302;
            res.set_header("Location", "http://127.0.0.1:" + std::to_string(cdn_port_) + "/blob/" +
                                           sha256_hex(it->second.content));
            return;
        }
        res.set_content(it->second.content, "application/octet-stream");  // httplib answers a Range with a 206
    }

    void blob(const httplib::Request& req, httplib::Response& res) {
        ++file_requests;
        ++cdn_requests;
        if (req.has_header("Authorization")) ++cdn_auth;
        if (cdn_limit >= 0 && cdn_served >= cdn_limit) {
            res.status = 503;
            return;
        }
        ++cdn_served;
        std::lock_guard<std::mutex> lock(m_);
        for (const auto& [name, r] : repos_) {
            for (const auto& [path, f] : r.files) {
                if (f.lfs && sha256_hex(f.content) == req.matches[1].str()) {
                    std::string body = f.content;
                    if (corrupt && !body.empty()) body[body.size() / 2] ^= 0x5a;
                    res.set_content(body, "application/octet-stream");
                    return;
                }
            }
        }
        res.status = 404;
    }

    std::mutex m_;
    std::map<std::string, Repo> repos_;
    httplib::Server hub_, cdn_;
    int hub_port_ = 0, cdn_port_ = 0;
    std::thread hub_thread_, cdn_thread_;
};

// A directory of this process's own, emptied on both ends.
struct HubTempDir {
    std::filesystem::path path;
    explicit HubTempDir(const std::string& name)
        : path(std::filesystem::temp_directory_path() / (name + "." + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~HubTempDir() { std::filesystem::remove_all(path); }
};

inline std::string read_file(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace xs::test
