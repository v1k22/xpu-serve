#include "hub/download.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>

#include <openssl/evp.h>

#include "util/error.h"
#include "util/log.h"

namespace xs::hub {

namespace fs = std::filesystem;

namespace {

fs::path partial_of(const fs::path& dest) { return dest.string() + ".xs-partial"; }
fs::path ranges_of(const fs::path& dest) { return dest.string() + ".xs-ranges"; }

// One file being downloaded: its partial file, the ranges still to do, the URL it redirects to.
struct Active {
    const FileJob* job = nullptr;
    int64_t range = 0;  // bytes a range
    int fd = -1;
    std::vector<int64_t> todo;  // first bytes of the ranges not yet written
    int64_t ranges = 0;
    int64_t done = 0;
    std::string direct;  // the URL a redirect led to, reused for every range
    std::FILE* ledger = nullptr;
    std::mutex m;
};

std::string human_bytes(double b) {
    char s[32];
    if (b >= 1e9) {
        std::snprintf(s, sizeof s, "%.2f GB", b / 1e9);
    } else {
        std::snprintf(s, sizeof s, "%.1f MB", b / 1e6);
    }
    return s;
}

}  // namespace

std::string sha256_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot read ", path.string());
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    XS_CHECK(ctx != nullptr && EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1, "sha256: OpenSSL failed");
    std::vector<char> buf(size_t{4} << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        if (in.gcount() > 0) EVP_DigestUpdate(ctx, buf.data(), static_cast<size_t>(in.gcount()));
    }
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int n = 0;
    EVP_DigestFinal_ex(ctx, md, &n);
    EVP_MD_CTX_free(ctx);
    std::string hex;
    char b[3];
    for (unsigned int i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, "%02x", md[i]);
        hex += b;
    }
    return hex;
}

void download_files(const std::vector<FileJob>& jobs, const DownloadOptions& options) {
    XS_CHECK(options.connections >= 1 && options.chunk_bytes >= 1, "download: bad connections or chunk size");
    std::vector<std::unique_ptr<Active>> files;
    int64_t total = 0, already = 0;
    for (const FileJob& job : jobs) {
        if (fs::exists(job.dest)) continue;
        fs::create_directories(job.dest.parent_path());
        auto a = std::make_unique<Active>();
        a->job = &job;
        // Enough ranges for every connection on a file of a few hundred megabytes, none longer than chunk_bytes.
        const int64_t floor = std::min<int64_t>(options.chunk_bytes, int64_t{8} << 20);
        a->range = std::max(floor, std::min(options.chunk_bytes, (job.size + options.connections - 1) / options.connections));
        a->ranges = job.size == 0 ? 0 : (job.size + a->range - 1) / a->range;
        // Ranges listed as written count only while the partial file is there, of the file's size, and the list was
        // made with the same range size (its first line).
        std::set<int64_t> written;
        bool resume = fs::exists(partial_of(job.dest)) &&
                      static_cast<int64_t>(fs::file_size(partial_of(job.dest))) == job.size;
        if (resume) {
            std::ifstream in(ranges_of(job.dest));
            std::string word;
            long long range = 0;
            resume = (in >> word >> range) && word == "range" && range == a->range;
            for (long long first; resume && in >> first;)
                written.insert(first);
        }
        a->fd = ::open(partial_of(job.dest).c_str(), O_RDWR | O_CREAT | (resume ? 0 : O_TRUNC), 0644);
        if (a->fd < 0) fail("cannot write ", partial_of(job.dest).string());
        if (!resume && ::ftruncate(a->fd, job.size) != 0) fail("cannot size ", partial_of(job.dest).string());
        a->ledger = std::fopen(ranges_of(job.dest).c_str(), resume ? "a" : "w");
        if (a->ledger == nullptr) fail("cannot write ", ranges_of(job.dest).string());
        if (!resume) {
            std::fprintf(a->ledger, "range %lld\n", static_cast<long long>(a->range));
            std::fflush(a->ledger);
        }
        for (int64_t r = 0; r < a->ranges; ++r) {
            const int64_t first = r * a->range;
            if (written.count(first) != 0) {
                ++a->done;
                already += std::min(a->range, job.size - first);
            } else {
                a->todo.push_back(first);
            }
        }
        total += job.size;
        files.push_back(std::move(a));
    }
    if (files.empty()) return;

    // The work: (file, range) pairs, handed out in order.
    std::vector<std::pair<Active*, int64_t>> work;
    for (auto& f : files) {
        for (const int64_t first : f->todo)
            work.emplace_back(f.get(), first);
    }
    std::atomic<size_t> next{0};
    std::atomic<int64_t> got{already};
    std::mutex error_m;
    std::string error;
    std::atomic<bool> stop{false};

    const auto worker = [&] {
        HttpClient http(options.proxy);
        for (size_t i; !stop && (i = next++) < work.size();) {
            Active& f = *work[i].first;
            const int64_t first = work[i].second;
            const int64_t last = std::min(first + f.range, f.job->size) - 1;
            bool ok = false;
            std::string why;
            for (int attempt = 0; attempt < options.attempts && !ok && !stop; ++attempt) {
                if (attempt > 0) std::this_thread::sleep_for(std::chrono::seconds(1 << std::min(attempt, 4)));
                std::string url;
                {
                    std::lock_guard<std::mutex> lock(f.m);
                    url = f.direct.empty() ? f.job->url : f.direct;
                }
                int64_t at = first;
                const Response r = http.get(
                    url, {{"Range", "bytes=" + std::to_string(first) + "-" + std::to_string(last)}}, options.auth,
                    options.auth_origin, true, [&](const char* data, size_t n) {
                        if (at + static_cast<int64_t>(n) > last + 1) return false;  // more than the range
                        if (::pwrite(f.fd, data, n, at) != static_cast<ssize_t>(n)) return false;
                        at += static_cast<int64_t>(n);
                        got += static_cast<int64_t>(n);
                        return true;
                    });
                const bool whole = r.status == 200 && first == 0 && last + 1 == f.job->size;
                if ((r.status == 206 || whole) && at == last + 1) {
                    ok = true;
                    if (r.url != f.job->url) {
                        std::lock_guard<std::mutex> lock(f.m);
                        f.direct = r.url;
                    }
                    break;
                }
                got -= at - first;
                if (r.status == 401 || r.status == 403 || r.status == 404 || r.status == 410) {
                    std::lock_guard<std::mutex> lock(f.m);
                    if (f.direct.empty() || url == f.job->url) {  // the source refuses: no retry helps
                        why = "HTTP " + std::to_string(r.status);
                        break;
                    }
                    f.direct.clear();  // a signed redirect URL that expired: ask again
                }
                why = r.status == 0 ? r.error : "HTTP " + std::to_string(r.status);
            }
            if (!ok) {
                std::lock_guard<std::mutex> lock(error_m);
                if (error.empty()) error = str_cat("downloading ", f.job->name, " failed: ", why);
                stop = true;
                break;
            }
            std::lock_guard<std::mutex> lock(f.m);
            std::fprintf(f.ledger, "%lld\n", static_cast<long long>(first));
            std::fflush(f.ledger);
            ++f.done;
        }
    };

    const auto start = std::chrono::steady_clock::now();
    std::mutex progress_m;
    std::condition_variable progress_cv;
    bool finished = false;
    std::thread progress([&] {
        std::unique_lock<std::mutex> lock(progress_m);
        while (!progress_cv.wait_for(lock, std::chrono::seconds(5), [&] { return finished; })) {
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double rate = static_cast<double>(got - already) / std::max(secs, 1e-3);
            const double left = rate > 0 ? static_cast<double>(total - got) / rate : 0.0;
            log::info("downloading ", options.label, ": ", human_bytes(static_cast<double>(got)), " of ",
                      human_bytes(static_cast<double>(total)), ", ", human_bytes(rate), "/s, ",
                      static_cast<int>(left / 60), " min ", static_cast<int>(left) % 60, " s left");
        }
    });
    std::vector<std::thread> threads;
    const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(options.connections), std::max<size_t>(work.size(), 1)));
    for (int t = 0; t < n; ++t)
        threads.emplace_back(worker);
    for (auto& t : threads)
        t.join();
    {
        std::lock_guard<std::mutex> lock(progress_m);
        finished = true;
    }
    progress_cv.notify_all();
    progress.join();

    for (auto& f : files) {
        ::close(f->fd);
        std::fclose(f->ledger);
    }
    if (!error.empty()) fail(error, " (", options.label, "; run again to resume)");
    for (auto& f : files) {
        const FileJob& job = *f->job;
        const fs::path partial = partial_of(job.dest);
        if (static_cast<int64_t>(fs::file_size(partial)) != job.size) {
            fs::remove(partial);
            fs::remove(ranges_of(job.dest));
            fail("downloading ", job.name, ": got a file of the wrong size");
        }
        if (!job.sha256.empty()) {
            const std::string got_sha = sha256_file(partial);
            if (got_sha != job.sha256) {
                fs::remove(partial);
                fs::remove(ranges_of(job.dest));
                fail("downloading ", job.name, ": SHA-256 ", got_sha, " is not the repository's ", job.sha256,
                     "; the file was deleted, run again to download it anew");
            }
        }
        fs::rename(partial, job.dest);
        fs::remove(ranges_of(job.dest));
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    log::info("downloaded ", options.label, ": ", human_bytes(static_cast<double>(total - already)), " in ",
              static_cast<int>(secs), " s");
}

}  // namespace xs::hub
