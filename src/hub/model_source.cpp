#include "hub/model_source.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <system_error>

#include "convert/gguf/architecture.h"
#include "convert/gguf/gguf_import.h"
#include "hub/download.h"
#include "hub/file_lock.h"
#include "io/gguf.h"
#include "io/safetensors.h"
#include "models/policy.h"
#include "util/error.h"
#include "util/log.h"

namespace xs::hub {

namespace fs = std::filesystem;

namespace {

// The identity an import of files the user named is filed with, in the pack.
constexpr const char* kSourceStamp = ".source";

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The importer of a model file, or null when none reads its architecture (the import then refuses it by name).
const gguf_import::Architecture* importer(const gguf::File& f) {
    const gguf::Value* arch = f.find("general.architecture");
    return arch != nullptr ? gguf_import::architecture_named(arch->as_string()) : nullptr;
}

// A model file whose drafter comes in a file of its own (Architecture::drafter_apart).
bool lacks_mtp_head(const gguf::File& f) {
    const gguf_import::Architecture* a = importer(f);
    return a != nullptr && a->drafter_apart(f);
}

// A model file whose vision tower its importer reads from a file of its own (Architecture::reads_projector).
bool reads_projector(const gguf::File& f) {
    const gguf_import::Architecture* a = importer(f);
    return a != nullptr && a->reads_projector();
}

// The companions of `model` (`model_path` in `listing`) that its import reads, the user did not name, and `listing`
// holds: paths in the listing, empty where there is none. `where` says whose listing it is ("org/name publishes").
struct Picks {
    std::string mtp, mmproj;
};
Picks pick_companions(const gguf::File& model, const RepoListing& listing, const std::string& model_path,
                      const ModelSpec& spec, const std::string& where) {
    Picks p;
    const std::string name = fs::path(model_path).filename().string();
    if (spec.mtp.empty() && lacks_mtp_head(model)) {
        p.mtp = select_mtp_head(listing, model_path);
        if (p.mtp.empty()) {
            log::info(name, " has no drafter and ", where, " none: --spec is not available");
        } else {
            log::info(name, " has no drafter: importing the one of ", p.mtp);
        }
    }
    if (spec.mmproj.empty() && reads_projector(model)) {
        p.mmproj = select_projector(listing);
        if (p.mmproj.empty()) {
            log::warn(where, " no vision tower (mmproj*.gguf): ", name, " is imported without images");
        } else {
            log::info(name, " reads images through the vision tower of ", p.mmproj);
        }
    }
    return p;
}

// The GGUF files of `dir` and of the folders in it, as a repository's listing (paths relative to `dir`): a model file's
// folder read as the repository it came from (`MTP/mtp-*.gguf` beside `<model>.gguf`).
RepoListing folder_listing(const fs::path& dir) {
    RepoListing l;
    l.repo = dir.string();
    const auto add = [&](const fs::path& file) {
        std::string ext = file.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        std::error_code ec;
        if (ext == ".gguf" && fs::is_regular_file(file, ec)) {
            l.files.push_back(RepoFile{file.lexically_relative(dir).generic_string(), 0, "", ""});
        }
    };
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
        std::error_code sub;
        if (!e.is_directory(sub)) {
            add(e.path());
            continue;
        }
        for (const auto& f : fs::directory_iterator(e.path(), fs::directory_options::skip_permission_denied, sub))
            add(f.path());
    }
    std::sort(l.files.begin(), l.files.end(), [](const RepoFile& a, const RepoFile& b) { return a.path < b.path; });
    return l;
}

// The companions the user named belong to a GGUF file; `what` is a model directory or a checkpoint.
void refuse_companions(const ModelSpec& spec, const std::string& what) {
    if (!spec.mtp.empty()) fail("--mtp names a GGUF file's MTP head, and ", what, " is not a GGUF file");
    if (!spec.mmproj.empty()) fail("--mmproj names a GGUF file's vision tower, and ", what, " is not a GGUF file");
}

SourceFiles hub_files(const ModelSpec& spec, FileRole role, const SourceOptions& o) {
    const HubSpec repo = parse_hub_spec(spec.hf);
    const RepoListing listing = list_repo(repo.repo, repo.revision, o.hub);
    const Selection sel = select_files(listing, repo, role);
    SourceFiles s;
    s.name = repo.name();
    s.folder = repo_folder(o.hub.cache, repo.repo).filename().string();
    s.commit = listing.commit;
    if (sel.gguf.empty()) {
        refuse_companions(spec, s.name);
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
    s.mtp = spec.mtp;
    s.mmproj = spec.mmproj;
    if (!spec.mtp.empty() || !spec.mmproj.empty()) {  // not the repository's files alone
        s.folder.clear();
        s.commit.clear();
    }
    if (role == FileRole::Model) {
        const Picks p = pick_companions(gguf::File(s.gguf), listing, sel.gguf, spec, repo.repo + " publishes");
        for (const auto& [pick, file] : {std::pair{&p.mtp, &s.mtp}, std::pair{&p.mmproj, &s.mmproj}}) {
            if (pick->empty()) continue;
            fetch_files(listing, {*pick}, o.hub);
            *file = s.snapshot / *pick;
        }
    }
    return s;
}

SourceFiles local_files(const ModelSpec& spec, FileRole role) {
    const fs::path path = spec.path;
    SourceFiles s;
    if (fs::is_directory(path)) {
        refuse_companions(spec, str_cat("'", path.string(), "'"));
        s.snapshot = path;
        return s;
    }
    if (!fs::exists(path)) fail("'", path.string(), "' does not exist");
    if (!fs::is_regular_file(path) || !gguf::has_gguf_magic(path)) {
        fail("'", path.string(), "' is neither a model directory nor a GGUF file");
    }
    s.gguf = path;
    s.name = gguf::model_name(path);
    s.mtp = spec.mtp;
    s.mmproj = spec.mmproj;
    if (role == FileRole::Model) {
        const fs::path folder = path.has_parent_path() ? path.parent_path() : fs::path(".");
        const Picks p = pick_companions(gguf::File(path), folder_listing(folder), path.filename().string(), spec,
                                        str_cat("its folder ", folder.string(), " holds"));
        if (!p.mtp.empty()) s.mtp = folder / p.mtp;
        if (!p.mmproj.empty()) s.mmproj = folder / p.mmproj;
    }
    return s;
}

// The identity an import of files the user named is current for: per part of each file, its canonical path, size and
// modification time. A stat a part: a rewrite moves the time and a cut changes the size, where hashing the contents
// would read every byte of a file of many gigabytes at each start.
std::string identity(const std::vector<fs::path>& files) {
    std::string s;
    for (const fs::path& f : files) {
        if (f.empty()) continue;
        const gguf::File file(f);
        for (const gguf::File::Part& p : file.parts()) {
            const fs::path c = fs::canonical(p.path);
            s += str_cat(c.string(), " ", fs::file_size(c), " ", fs::last_write_time(c).time_since_epoch().count(),
                         "\n");
        }
    }
    return s;
}

// `dir` made by `make` in a partial directory renamed into place when whole, under a lock, unless `current`: an
// interrupted run or a second process never leaves or reads half of it.
void build_once(const fs::path& dir, const std::function<bool()>& current,
                const std::function<void(const fs::path&)>& make) {
    if (current()) return;
    FileLock lock(dir.string() + ".lock");
    if (current()) return;
    const fs::path partial = dir.string() + ".partial";
    fs::remove_all(partial);
    fs::remove_all(dir);
    make(partial);
    fs::rename(partial, dir);
}

// A GGUF file's pack in xpu-serve's cache. A repository's files at a commit are filed under it and imported once;
// files the user named, under the model file's canonical path, imported again when one of them changes.
fs::path gguf_pack(const SourceFiles& s, const SourceOptions& o) {
    std::string leaf = gguf::model_name(s.gguf);
    for (const fs::path& f : {s.mtp, s.mmproj}) {
        if (!f.empty()) leaf += "+" + gguf::model_name(f);
    }
    std::string stamp;
    fs::path dir;
    if (!s.commit.empty()) {
        dir = o.cache / "packs" / s.folder / s.commit.substr(0, 12) / leaf;
    } else {
        stamp = identity({s.gguf, s.mtp, s.mmproj});
        dir = o.cache / "packs" / "local" / sha256_hex(fs::canonical(s.gguf).string()).substr(0, 12) / leaf;
    }
    const auto current = [&] {
        return stamp.empty() ? fs::exists(dir / kPackManifest) : read_text(dir / kSourceStamp) == stamp;
    };
    if (!stamp.empty() && fs::exists(dir / kPackManifest) && !current()) {
        log::info(s.gguf.filename().string(), " or a file imported with it changed: importing them again");
    }
    build_once(dir, current, [&](const fs::path& out) {
        log::info("importing ", s.gguf.filename().string(), " into ", dir.string());
        gguf_import::ImportOptions io;
        io.source = s.gguf;
        io.mtp = s.mtp;
        io.mmproj = s.mmproj;
        io.output = out;
        io.converter = o.converter;
        gguf_import::import_gguf(io);
        if (stamp.empty()) return;
        std::ofstream stamped(out / kSourceStamp, std::ios::binary);
        stamped << stamp;
        if (!stamped) fail("writing ", (out / kSourceStamp).string(), " failed");
    });
    return dir;
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

SourceFiles source_files(const ModelSpec& spec, FileRole role, const SourceOptions& o) {
    if (spec.path.empty() == spec.hf.empty()) fail("a model is a directory, a GGUF file or -hf <org/name>: give one");
    for (const fs::path& f : {spec.mtp, spec.mmproj}) {
        if (f.empty()) continue;
        if (!fs::is_regular_file(f)) fail("'", f.string(), "' is not a file");
        if (!gguf::has_gguf_magic(f)) fail("'", f.string(), "' is not a GGUF file");
    }
    return spec.hf.empty() ? local_files(spec, role) : hub_files(spec, role, o);
}

ModelSource model_dir(const SourceFiles& s, const SourceOptions& o) {
    ModelSource m;
    m.name = s.name;
    if (!s.gguf.empty()) {
        m.dir = gguf_pack(s, o);
        return m;
    }
    if (s.companions.empty()) {
        m.dir = s.snapshot;
        return m;
    }
    // The snapshot's files and the companions, side by side: links in a folder of xpu-serve's cache.
    m.dir = o.cache / "models" / s.folder / s.commit;
    build_once(
        m.dir, [&] { return fs::exists(m.dir / ".complete"); },
        [&](const fs::path& out) {
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

ModelSource resolve_model(const ModelSpec& spec, FileRole role, const SourceOptions& o) {
    return model_dir(source_files(spec, role, o), o);
}

}  // namespace xs::hub
