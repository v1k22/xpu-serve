#include "hub/model_source.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>

#include "convert/gguf/gguf_import.h"
#include "hub/file_lock.h"
#include "io/gguf.h"
#include "models/policy.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::hub {

namespace fs = std::filesystem;

namespace {

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
}

// A model file whose drafter is not in it: one of the Qwen3.5 family (qwen35, qwen35moe) without its MTP head
// (nextn_predict_layers 0: unsloth's smallest Qwen3.8-27B files), or Inkling 2's, whose drafter is always a file of
// its own.
bool lacks_mtp_head(const fs::path& file) {
    const gguf::File f(file);
    const gguf::Value* arch = f.find("general.architecture");
    if (arch != nullptr && (arch->as_string() == "inkling2" || arch->as_string() == "inkling2_loop")) return true;
    if (arch == nullptr || (arch->as_string() != "qwen35" && arch->as_string() != "qwen35moe")) return false;
    const gguf::Value* n = f.find(std::string(arch->as_string()) + ".nextn_predict_layers");
    return n == nullptr || n->as_int() == 0;
}

// A model file whose vision tower the importer reads from the repository's projector file: Inkling 2's, in either
// layout.
bool reads_projector(const fs::path& file) {
    const gguf::File f(file);
    const gguf::Value* arch = f.find("general.architecture");
    return arch != nullptr && (arch->as_string() == "inkling2" || arch->as_string() == "inkling2_loop");
}

std::string stem(const fs::path& p) { return p.stem().string(); }

// `dir` once and for all: made by `make` in a partial directory renamed into place when whole, under a lock, so an
// interrupted run or a second process never leaves or reads half of it.
void build_once(const fs::path& dir, const fs::path& marker, const std::function<void(const fs::path&)>& make) {
    if (fs::exists(marker)) return;
    FileLock lock(dir.string() + ".lock");
    if (fs::exists(marker)) return;
    const fs::path partial = dir.string() + ".partial";
    fs::remove_all(partial);
    fs::remove_all(dir);
    make(partial);
    fs::rename(partial, dir);
}

}  // namespace

SourceOptions SourceOptions::from_env() {
    SourceOptions o;
    o.hub = HubOptions::from_env();
    o.cache = !env("XS_CACHE_DIR").empty()     ? fs::path(env("XS_CACHE_DIR"))
              : !env("XDG_CACHE_HOME").empty() ? fs::path(env("XDG_CACHE_HOME")) / "xpu-serve"
                                               : fs::path(env("HOME")) / ".cache" / "xpu-serve";
    return o;
}

SourceFiles fetch_source(const std::string& text, FileRole role, const SourceOptions& o) {
    const HubSpec spec = parse_hub_spec(text);
    const RepoListing listing = list_repo(spec.repo, spec.revision, o.hub);
    const Selection sel = select_files(listing, spec, role);
    SourceFiles s;
    s.name = spec.name();
    s.folder = repo_folder(o.hub.cache, spec.repo).filename().string();
    s.commit = listing.commit;
    if (sel.gguf.empty()) {
        constexpr int64_t kSmall = int64_t{64} << 20;
        std::vector<std::string> small, large;
        for (const std::string& p : sel.paths) {
            const auto f = std::find_if(listing.files.begin(), listing.files.end(),
                                        [&](const RepoFile& r) { return r.path == p; });
            const bool weights = p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
            (weights || f->size >= kSmall ? large : small).push_back(p);
        }
        s.snapshot = fetch_files(listing, small, o.hub);
        for (const CompanionFile& c : missing_companion_files(s.snapshot)) {
            const RepoListing l = list_repo(c.repo, "main", o.hub);
            if (std::none_of(l.files.begin(), l.files.end(), [&](const RepoFile& f) { return f.path == c.path; })) {
                fail(c.repo, " has no ", c.path, ", which ", s.name, " reads");
            }
            s.companions.emplace_back(c.path, fetch_files(l, {c.path}, o.hub) / c.path);
        }
        fetch_files(listing, large, o.hub);
        return s;
    }
    s.snapshot = fetch_files(listing, sel.paths, o.hub);
    s.gguf = s.snapshot / sel.gguf;
    if (role == FileRole::Model && lacks_mtp_head(s.gguf)) {
        const std::string head = select_mtp_head(listing, sel.gguf);
        if (head.empty()) {
            log::info(sel.gguf, " has no drafter and ", spec.repo, " publishes none: --spec is not available");
        } else {
            fetch_files(listing, {head}, o.hub);
            s.mtp = s.snapshot / head;
            log::info(sel.gguf, " has no drafter: importing the one of ", head);
        }
    }
    if (role == FileRole::Model && reads_projector(s.gguf)) {
        if (const std::string tower = select_projector(listing); !tower.empty()) {
            fetch_files(listing, {tower}, o.hub);
            s.mmproj = s.snapshot / tower;
            log::info(sel.gguf, " reads images through the vision tower of ", tower);
        } else {
            log::warn(spec.repo, " publishes no vision tower (mmproj*.gguf): ", sel.gguf,
                      " is imported without images");
        }
    }
    return s;
}

ModelSource resolve_model(const std::string& text, FileRole role, const SourceOptions& o) {
    const SourceFiles s = fetch_source(text, role, o);
    ModelSource m;
    m.name = s.name;
    if (!s.gguf.empty()) {
        const std::string name = stem(s.gguf) + (s.mtp.empty() ? "" : "+" + stem(s.mtp)) +
                                 (s.mmproj.empty() ? "" : "+" + stem(s.mmproj));
        m.dir = o.cache / "packs" / s.folder / s.commit.substr(0, 12) / name;
        build_once(m.dir, m.dir / "xpu_pack.json", [&](const fs::path& out) {
            log::info("importing ", s.gguf.filename().string(), " into ", m.dir.string());
            gguf_import::ImportOptions io;
            io.source = s.gguf;
            io.mtp = s.mtp;
            io.mmproj = s.mmproj;
            io.output = out;
            io.converter = o.converter;
            gguf_import::import_gguf(io);
        });
        return m;
    }
    if (s.companions.empty()) {
        m.dir = s.snapshot;
        return m;
    }
    // The snapshot's files and the companions, side by side: links in a folder of xpu-serve's cache.
    m.dir = o.cache / "models" / s.folder / s.commit;
    build_once(m.dir, m.dir / ".complete", [&](const fs::path& out) {
        fs::create_directories(out);
        for (const auto& e : fs::directory_iterator(s.snapshot))
            fs::create_symlink(fs::absolute(e.path()), out / e.path().filename());
        for (const auto& [path, file] : s.companions) {
            fs::create_directories((out / path).parent_path());
            fs::create_symlink(fs::absolute(file), out / path);
        }
        std::ofstream(out / ".complete");
    });
    return m;
}

}  // namespace xs::hub
