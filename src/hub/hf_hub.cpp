#include "hub/hf_hub.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>

#include <nlohmann/json.hpp>

#include "hub/download.h"
#include "hub/file_lock.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::hub {

namespace fs = std::filesystem;

namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
}

std::string read_first_line(const fs::path& p) {
    std::ifstream in(p);
    std::string line;
    std::getline(in, line);
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    return line;
}

bool is_commit(const std::string& s) {
    return s.size() == 40 && std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); });
}

// A repository path in a URL: every byte but the unreserved ones and '/' percent-encoded.
std::string encode_path(const std::string& path) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char c : path) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' || c == '/') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

void check_repo_name(const std::string& repo) {
    const size_t slash = repo.find('/');
    const bool ok = slash != std::string::npos && slash > 0 && slash + 1 < repo.size() &&
                    repo.find('/', slash + 1) == std::string::npos &&
                    std::all_of(repo.begin(), repo.end(), [](unsigned char c) {
                        return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '/';
                    }) &&
                    repo.find("..") == std::string::npos;
    if (!ok) fail("'", repo, "' is not a Hugging Face repository name (org/name)");
}

std::string auth_of(const HubOptions& o) { return o.token.empty() ? std::string() : "Bearer " + o.token; }

// The snapshot the cache holds of repo@revision, its files listed from disk.
std::optional<RepoListing> cached_listing(const std::string& repo, const std::string& revision, const HubOptions& o) {
    const fs::path folder = repo_folder(o.cache, repo);
    std::string commit = is_commit(revision) ? revision : read_first_line(folder / "refs" / revision);
    if (commit.empty() || !fs::is_directory(folder / "snapshots" / commit)) return std::nullopt;
    RepoListing l;
    l.repo = repo;
    l.revision = revision;
    l.commit = commit;
    l.from_cache = true;
    const fs::path root = folder / "snapshots" / commit;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        if (!fs::is_regular_file(e.path())) continue;  // follows the link to the blob; a dangling link is left out
        RepoFile f;
        f.path = e.path().lexically_relative(root).generic_string();  // not fs::relative: it follows the link
        f.size = static_cast<int64_t>(fs::file_size(e.path()));
        l.files.push_back(std::move(f));
    }
    std::sort(l.files.begin(), l.files.end(), [](const RepoFile& a, const RepoFile& b) { return a.path < b.path; });
    return l;
}

}  // namespace

HubOptions HubOptions::from_env() {
    HubOptions o;
    if (const std::string e = env("HF_ENDPOINT"); !e.empty()) o.endpoint = e;
    while (!o.endpoint.empty() && o.endpoint.back() == '/')
        o.endpoint.pop_back();
    const std::string home = env("HOME");
    const fs::path hf_home = !env("HF_HOME").empty() ? fs::path(env("HF_HOME"))
                             : !env("XDG_CACHE_HOME").empty() ? fs::path(env("XDG_CACHE_HOME")) / "huggingface"
                                                              : fs::path(home) / ".cache" / "huggingface";
    o.cache = !env("HF_HUB_CACHE").empty() ? fs::path(env("HF_HUB_CACHE")) : hf_home / "hub";
    o.token = env("HF_TOKEN");
    if (o.token.empty()) {
        o.token = read_first_line(!env("HF_TOKEN_PATH").empty() ? fs::path(env("HF_TOKEN_PATH")) : hf_home / "token");
    }
    const std::string offline = env("HF_HUB_OFFLINE");
    o.offline = offline == "1" || offline == "true" || offline == "TRUE" || offline == "yes";
    o.proxy = Proxy::from_env();
    return o;
}

fs::path repo_folder(const fs::path& cache, const std::string& repo) {
    std::string name = "models--" + repo;
    for (size_t at; (at = name.find('/')) != std::string::npos;)
        name.replace(at, 1, "--");
    return cache / name;
}

RepoListing list_repo(const std::string& repo, const std::string& revision, const HubOptions& o) {
    check_repo_name(repo);
    if (o.offline) {
        if (auto cached = cached_listing(repo, revision, o)) return *cached;
        fail("offline (HF_HUB_OFFLINE) and the Hugging Face cache (", o.cache.string(), ") has no snapshot of ", repo,
             "@", revision);
    }
    HttpClient http(o.proxy, 30);
    const std::string url = o.endpoint + "/api/models/" + repo + "/revision/" + encode_path(revision) + "?blobs=true";
    const Response r = http.get(url, {}, auth_of(o), parse_url(o.endpoint).origin(), true);
    if (r.status == 0) {
        if (auto cached = cached_listing(repo, revision, o)) {
            log::warn("cannot reach ", o.endpoint, " (", r.error, "): using the cached snapshot of ", repo, " (",
                      cached->commit.substr(0, 12), ")");
            return *cached;
        }
        fail("cannot reach ", o.endpoint, " to list ", repo, ": ", r.error,
             " (check the proxy settings: https_proxy, no_proxy)");
    }
    if (r.status == 401 || r.status == 404) {
        if (r.body.find("Revision Not Found") != std::string::npos || r.body.find("Invalid rev") != std::string::npos) {
            fail(repo, " has no revision '", revision, "'");
        }
        fail(repo, " was not found on ", o.endpoint,
             o.token.empty() ? ", or it is private: check the name, or set HF_TOKEN to a token that can read it"
                             : ", or the token in HF_TOKEN cannot read it");
    }
    if (r.status != 200) fail("listing ", repo, " on ", o.endpoint, ": HTTP ", r.status, " ", r.body.substr(0, 200));
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(r.body);
    } catch (const std::exception& e) {
        fail("listing ", repo, ": the Hub's answer is not JSON (", e.what(), ")");
    }
    RepoListing l;
    l.repo = repo;
    l.revision = revision;
    l.commit = j.value("sha", "");
    if (!is_commit(l.commit)) fail("listing ", repo, ": no commit in the Hub's answer");
    for (const auto& s : j.value("siblings", nlohmann::json::array())) {
        RepoFile f;
        f.path = s.value("rfilename", "");
        f.size = s.value("size", int64_t{0});
        f.blob_id = s.value("blobId", "");
        if (s.contains("lfs") && s["lfs"].is_object()) {
            f.sha256 = s["lfs"].value("sha256", "");
            f.size = s["lfs"].value("size", f.size);
        }
        if (f.path.empty() || f.path.find("..") != std::string::npos || f.path[0] == '/') continue;
        l.files.push_back(std::move(f));
    }
    if (const auto gated = j.find("gated"); gated != j.end() && gated->is_string()) l.gated = gated->get<std::string>();
    return l;
}

fs::path fetch_files(const RepoListing& l, const std::vector<std::string>& paths, const HubOptions& o) {
    const fs::path folder = repo_folder(o.cache, l.repo);
    const fs::path snapshot = folder / "snapshots" / l.commit;
    std::vector<const RepoFile*> wanted;
    for (const std::string& p : paths) {
        const auto it = std::find_if(l.files.begin(), l.files.end(), [&](const RepoFile& f) { return f.path == p; });
        XS_CHECK(it != l.files.end(), "fetch_files: ", p, " is not a file of ", l.repo);
        wanted.push_back(&*it);
    }
    if (l.from_cache) return snapshot;  // listed from the snapshot itself: every file is there

    std::vector<FileJob> jobs;
    std::set<std::string> blobs;
    for (const RepoFile* f : wanted) {
        const std::string blob = f->sha256.empty() ? f->blob_id : f->sha256;
        if (blob.empty()) fail("listing ", l.repo, ": no hash for ", f->path);
        if (!blobs.insert(blob).second) continue;  // the same content under two names
        FileJob job;
        job.url = o.endpoint + "/" + l.repo + "/resolve/" + l.commit + "/" + encode_path(f->path);
        job.dest = folder / "blobs" / blob;
        job.size = f->size;
        job.sha256 = f->sha256;
        job.name = l.repo + "/" + f->path;
        if (!fs::exists(job.dest)) jobs.push_back(std::move(job));
    }
    if (!jobs.empty()) {
        if (!l.gated.empty() && o.token.empty()) {  // listing a gated repository works without a token, its files not
            fail(l.repo, " is gated: accept its terms at ", o.endpoint, "/", l.repo,
                 " and set HF_TOKEN to an access token of that account");
        }
        std::vector<std::unique_ptr<FileLock>> locks;
        std::sort(jobs.begin(), jobs.end(), [](const FileJob& a, const FileJob& b) { return a.dest < b.dest; });
        for (const FileJob& job : jobs)
            locks.push_back(std::make_unique<FileLock>(o.cache / ".locks" / folder.filename() /
                                                       (job.dest.filename().string() + ".lock")));
        DownloadOptions d;
        d.proxy = o.proxy;
        d.auth = auth_of(o);
        d.auth_origin = parse_url(o.endpoint).origin();
        d.connections = o.connections;
        d.chunk_bytes = o.chunk_bytes;
        d.attempts = o.attempts;
        d.label = l.repo;
        int64_t bytes = 0;
        for (const FileJob& job : jobs)
            bytes += job.size;
        char size[32];
        std::snprintf(size, sizeof size, "%.2f GB", static_cast<double>(bytes) / 1e9);
        log::info("downloading ", jobs.size(), " file(s) of ", l.repo, " (", l.commit.substr(0, 12), "), ", size,
                  ", into ", o.cache.string());
        try {
            download_files(jobs, d);
        } catch (const Error& e) {
            const std::string what = e.what();
            if (what.find("HTTP 401") != std::string::npos || what.find("HTTP 403") != std::string::npos) {
                fail(what, ". ", l.repo, " may be gated: accept its terms at ", o.endpoint, "/", l.repo,
                     o.token.empty() ? " and set HF_TOKEN to an access token" : " with the account of HF_TOKEN");
            }
            throw;
        }
    }
    for (const RepoFile* f : wanted) {
        const fs::path link = snapshot / f->path;
        const fs::path blob = folder / "blobs" / (f->sha256.empty() ? f->blob_id : f->sha256);
        if (fs::exists(link)) continue;
        fs::create_directories(link.parent_path());
        std::error_code ec;
        fs::remove(link, ec);  // a link left dangling
        fs::create_symlink(fs::relative(blob, link.parent_path()), link);
    }
    if (l.revision != l.commit) {
        fs::create_directories(folder / "refs");
        if (read_first_line(folder / "refs" / l.revision) != l.commit) {
            std::ofstream(folder / "refs" / l.revision) << l.commit;
        }
    }
    return snapshot;
}

}  // namespace xs::hub
