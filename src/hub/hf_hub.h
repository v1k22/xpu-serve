#pragma once

// A Hugging Face repository's files, downloaded into the Hugging Face cache in its own layout (what huggingface_hub,
// `hf download` and other engines read and write: `models--<org>--<name>/{blobs,snapshots,refs}`), so a model
// downloaded by one is found by the others.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "hub/http.h"

namespace xs::hub {

struct HubOptions {
    std::string endpoint = "https://huggingface.co";
    std::filesystem::path cache;  // the hub cache: the directory of the `models--*` folders
    std::string token;            // a Hugging Face access token, for gated and private repositories
    bool offline = false;         // only what the cache holds
    Proxy proxy;
    int connections = 8;
    int64_t chunk_bytes = int64_t{64} << 20;
    int attempts = 5;  // per range of a file
    // As huggingface_hub reads them: HF_ENDPOINT; HF_HUB_CACHE, else HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub,
    // else ~/.cache/huggingface/hub; HF_TOKEN, else the file HF_TOKEN_PATH or HF_HOME/token; HF_HUB_OFFLINE=1; the
    // proxies of Proxy::from_env.
    static HubOptions from_env();
};

struct RepoFile {
    std::string path;     // in the repository, '/'-separated
    int64_t size = 0;
    std::string blob_id;  // the git object id
    std::string sha256;   // of a file stored in LFS / Xet; empty for one stored in git
};

struct RepoListing {
    std::string repo;      // org/name
    std::string revision;  // what was asked for: a branch, tag or commit
    std::string commit;    // what it points to
    std::vector<RepoFile> files;
    bool from_cache = false;  // the cached snapshot's files (offline): sizes only, no hashes
    std::string gated;        // "auto" or "manual": its files need a token of an account that accepted its terms
};

// The files of repo@revision, from the Hub, or from the cache when offline or when the Hub cannot be reached (with a
// warning) and the cache has a snapshot of it. Errors name the repository and what to do (a token, a license to
// accept on the Hub).
RepoListing list_repo(const std::string& repo, const std::string& revision, const HubOptions& options);

// The snapshot directory of `listing` holding `paths` (each a RepoFile::path of it): what the cache lacks is downloaded
// first, every file checked against its size and SHA-256.
std::filesystem::path fetch_files(const RepoListing& listing, const std::vector<std::string>& paths,
                                  const HubOptions& options);

// The cache's folder of a repository (`models--org--name`).
std::filesystem::path repo_folder(const std::filesystem::path& cache, const std::string& repo);

}  // namespace xs::hub
