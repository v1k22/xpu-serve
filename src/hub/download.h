#pragma once

// Files downloaded in ranges over several connections, resumable, checked against their size and SHA-256.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "hub/http.h"

namespace xs::hub {

struct FileJob {
    std::string url;               // where the file is (redirects allowed)
    std::filesystem::path dest;    // written there once whole and checked
    int64_t size = 0;
    std::string sha256;            // lower-case hex; empty: the size is the only check
    std::string name;              // for messages
};

struct DownloadOptions {
    Proxy proxy;
    std::string auth;         // Authorization header value, sent to `auth_origin` only (HttpClient::get)
    std::string auth_origin;
    int connections = 8;
    int64_t chunk_bytes = int64_t{64} << 20;  // the longest range; a file has at least `connections` of 8 MiB or more
    int attempts = 5;         // per range, before the download fails; waits of 2, 4, 8, 16 s between them
    std::string label;      // progress lines name it
};

// Downloads every job whose dest does not exist. A file in progress is `<dest>.xs-partial`, the ranges done so far
// listed in `<dest>.xs-ranges`, so an interrupted download resumes where it stopped. A file is renamed to its dest
// only when its size and SHA-256 are right; on a wrong one it is deleted and the error names it.
void download_files(const std::vector<FileJob>& jobs, const DownloadOptions& options);

// SHA-256 of a file, lower-case hex.
std::string sha256_file(const std::filesystem::path& path);

}  // namespace xs::hub
