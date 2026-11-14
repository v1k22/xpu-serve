#include "hub/select.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <set>

#include "util/error.h"

namespace xs::hub {

namespace {

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::toupper(c); });
    return s;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string basename(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string extension(const std::string& path) {
    const std::string base = basename(path);
    const size_t dot = base.rfind('.');
    return dot == std::string::npos ? std::string() : lower(base.substr(dot));
}

std::string top_folder(const std::string& path) {
    const size_t slash = path.find('/');
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

std::string folder_of(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

// A GGUF file's name without its extension and without a split part's "-00001-of-00003".
std::string gguf_stem(const std::string& path, bool* split) {
    static const std::regex part("-[0-9]{5}-of-[0-9]{5}$");
    std::string stem = basename(path);
    stem.resize(stem.size() - 5);  // ".gguf"
    std::smatch m;
    *split = std::regex_search(stem, m, part);
    if (*split) stem.resize(static_cast<size_t>(m.position(0)));
    return stem;
}

// Whether a GGUF file's name ends in the quantization tag `quant` (case aside): "...-Q4_K_M", "....Q4_K_M".
bool has_quant(const std::string& stem, const std::string& quant) {
    const std::string s = upper(stem), q = upper(quant);
    if (s == q) return true;
    if (s.size() <= q.size() || s.compare(s.size() - q.size(), q.size(), q) != 0) return false;
    const char before = s[s.size() - q.size() - 1];
    return before == '-' || before == '.' || before == '_';
}

const std::set<std::string> kOtherWeights = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".h5", ".msgpack", ".ot",
                                             ".onnx", ".onnx_data", ".tflite", ".mlmodel", ".gguf", ".npz", ".pb"};
const std::set<std::string> kMedia = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".mp4",
                                      ".mov", ".mp3", ".wav", ".pdf"};
const std::set<std::string> kSkippedFolders = {"onnx", "openvino", "coreml", "original", "assets", "images",
                                               "media", "figures", ".eval_results", "eval", "evals", "metrics"};

struct Candidate {
    std::string path, stem;
    bool split = false;
};

std::string list_of(const std::vector<Candidate>& cs) {
    std::string s;
    for (const Candidate& c : cs)
        s += (s.empty() ? "" : ", ") + basename(c.path);
    return s;
}

Selection select_gguf(const RepoListing& l, const HubSpec& spec, FileRole role) {
    std::vector<Candidate> cs;
    for (const RepoFile& f : l.files) {
        if (extension(f.path) != ".gguf") continue;
        const std::string base = lower(basename(f.path));
        const bool drafter = base.rfind("mtp-", 0) == 0;
        if (base.rfind("mmproj", 0) == 0 || base.rfind("imatrix", 0) == 0) continue;
        if (drafter != (role == FileRole::Drafter)) continue;
        Candidate c;
        c.path = f.path;
        c.stem = gguf_stem(f.path, &c.split);
        cs.push_back(std::move(c));
    }
    const char* what = role == FileRole::Drafter ? "drafter (mtp-*.gguf) " : "";
    if (cs.empty()) fail(l.repo, " has no ", what, "GGUF file");
    std::string quant = spec.quant;
    std::vector<Candidate> match;
    const auto pick = [&](const std::string& q) {
        match.clear();
        for (const Candidate& c : cs) {
            if (has_quant(c.stem, q)) match.push_back(c);
        }
    };
    if (quant.empty()) {
        pick(role == FileRole::Drafter ? "Q8_0" : "Q4_K_M");
        if (match.empty() && cs.size() == 1) match = cs;
        if (match.empty()) {
            fail(l.repo, " has several ", what, "GGUF files; name one by its quantization (", spec.repo,
                 ":<QUANT>): ", list_of(cs));
        }
    } else {
        pick(quant);
        if (match.empty()) fail(l.repo, " has no ", what, "GGUF file of quantization ", quant, "; it has ", list_of(cs));
    }
    // "Q4_K_M" names "<model>-Q4_K_M" before an Unsloth dynamic "<model>-UD-Q4_K_M".
    if (match.size() > 1 && upper(quant).rfind("UD-", 0) != 0) {
        std::vector<Candidate> plain;
        for (const Candidate& c : match) {
            if (upper(c.stem).find("UD-") == std::string::npos) plain.push_back(c);
        }
        if (!plain.empty()) match = plain;
    }
    std::set<std::string> stems;
    for (const Candidate& c : match)
        stems.insert(folder_of(c.path) + "/" + c.stem);
    if (stems.size() > 1) fail(l.repo, ": quantization ", quant, " names several files: ", list_of(match));
    if (match.front().split) {
        fail(l.repo, ": the ", quant.empty() ? std::string("default") : quant, " file is split into ", match.size(),
             " parts (", list_of(match), "); a GGUF file is imported whole: pick another quantization");
    }
    return {{match.front().path}, match.front().path};
}

}  // namespace

std::string HubSpec::name() const { return quant.empty() ? repo : repo + ":" + quant; }

HubSpec parse_hub_spec(std::string_view text) {
    HubSpec s;
    std::string rest(text);
    if (const size_t colon = rest.find(':'); colon != std::string::npos) {
        s.quant = rest.substr(colon + 1);
        rest.resize(colon);
        if (s.quant.empty()) fail("'", text, "': a quantization after ':' is empty");
    }
    if (const size_t at = rest.find('@'); at != std::string::npos) {
        s.revision = rest.substr(at + 1);
        rest.resize(at);
        if (s.revision.empty()) fail("'", text, "': a revision after '@' is empty");
    }
    s.repo = rest;
    const size_t slash = s.repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 == s.repo.size() ||
        s.repo.find('/', slash + 1) != std::string::npos) {
        fail("'", text, "' is not a Hugging Face repository: org/name, or org/name:QUANT for a GGUF file of it");
    }
    return s;
}

Selection select_files(const RepoListing& l, const HubSpec& spec, FileRole role) {
    const bool has_safetensors = std::any_of(l.files.begin(), l.files.end(), [](const RepoFile& f) {
        return extension(f.path) == ".safetensors";
    });
    const bool has_gguf = std::any_of(l.files.begin(), l.files.end(), [](const RepoFile& f) {
        return extension(f.path) == ".gguf";
    });
    if (!spec.quant.empty() || (!has_safetensors && has_gguf)) return select_gguf(l, spec, role);
    if (!has_safetensors) {
        fail(l.repo, " has no safetensors weights and no GGUF file (xpu-serve reads either)");
    }
    // Subfolders with weights of their own when the top has weights: other variants of the model.
    const bool top_weights = std::any_of(l.files.begin(), l.files.end(), [](const RepoFile& f) {
        return extension(f.path) == ".safetensors" && f.path.find('/') == std::string::npos;
    });
    std::set<std::string> variants;
    for (const RepoFile& f : l.files) {
        if (top_weights && extension(f.path) == ".safetensors" && !top_folder(f.path).empty()) {
            variants.insert(top_folder(f.path));
        }
    }
    Selection s;
    for (const RepoFile& f : l.files) {
        const std::string base = basename(f.path);
        const std::string top = top_folder(f.path);
        if (base == ".gitattributes" || base == ".gitignore") continue;
        if (kOtherWeights.count(extension(f.path)) != 0 || kMedia.count(extension(f.path)) != 0) continue;
        if (kSkippedFolders.count(top) != 0 || variants.count(top) != 0) continue;
        s.paths.push_back(f.path);
    }
    return s;
}

std::string select_mtp_head(const RepoListing& l, const std::string& model_file) {
    bool split = false;
    const std::string model_stem = upper(gguf_stem(model_file, &split));
    std::vector<Candidate> heads;
    for (const RepoFile& f : l.files) {
        if (extension(f.path) != ".gguf" || lower(basename(f.path)).rfind("mtp-", 0) != 0) continue;
        Candidate c;
        c.path = f.path;
        c.stem = gguf_stem(f.path, &c.split);
        if (!c.split) heads.push_back(std::move(c));
    }
    for (const Candidate& h : heads) {
        if (upper(h.stem.substr(4)) == model_stem) return h.path;  // "mtp-<the model file's name>"
    }
    if (heads.size() == 1) return heads.front().path;
    for (const Candidate& h : heads) {
        if (has_quant(h.stem, "Q8_0")) return h.path;
    }
    return {};
}

}  // namespace xs::hub
