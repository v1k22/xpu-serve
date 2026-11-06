#include "cli/args.h"

#include <cstdlib>

#include "util/error.h"

namespace xs::cli {

Args& Args::value(const std::string& name, std::string help, std::string fallback) {
    specs_[name] = Spec{std::move(help), std::move(fallback), false, false};
    order_.push_back(name);
    return *this;
}

Args& Args::flag(const std::string& name, std::string help) {
    specs_[name] = Spec{std::move(help), {}, true, false};
    order_.push_back(name);
    return *this;
}

void Args::parse(const std::vector<std::string>& argv) {
    for (size_t i = 0; i < argv.size(); ++i) {
        const std::string& a = argv[i];
        if (a.size() < 3 || a.compare(0, 2, "--") != 0) {
            positionals_.push_back(a);
            continue;
        }
        std::string name = a.substr(2);
        std::optional<std::string> inline_value;
        if (const auto eq = name.find('='); eq != std::string::npos) {
            inline_value = name.substr(eq + 1);
            name.resize(eq);
        }
        const auto it = specs_.find(name);
        if (it == specs_.end()) fail("unknown option --", name);
        Spec& spec = it->second;
        spec.given = true;
        if (spec.is_flag) {
            if (inline_value) fail("option --", name, " takes no value");
            continue;
        }
        if (inline_value) {
            spec.value = *inline_value;
        } else {
            if (i + 1 >= argv.size()) fail("option --", name, " needs a value");
            spec.value = argv[++i];
        }
    }
}

const std::string& Args::get(const std::string& name) const {
    const auto it = specs_.find(name);
    XS_CHECK(it != specs_.end(), "undeclared option ", name);
    return it->second.value;
}

int Args::get_int(const std::string& name) const {
    const std::string& v = get(name);
    char* end = nullptr;
    const long n = std::strtol(v.c_str(), &end, 10);
    if (v.empty() || *end != '\0') fail("option --", name, " expects an integer, got '", v, "'");
    return static_cast<int>(n);
}

double Args::get_double(const std::string& name) const {
    const std::string& v = get(name);
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (v.empty() || *end != '\0') fail("option --", name, " expects a number, got '", v, "'");
    return d;
}

bool Args::has(const std::string& name) const {
    const auto it = specs_.find(name);
    return it != specs_.end() && it->second.given;
}

std::string Args::usage() const {
    std::string out;
    for (const auto& name : order_) {
        const Spec& s = specs_.at(name);
        std::string left = "  --" + name + (s.is_flag ? "" : " <value>");
        if (left.size() < 28) left.resize(28, ' ');
        out += left + " " + s.help;
        if (!s.is_flag && !s.value.empty()) out += " (default: " + s.value + ")";
        out += '\n';
    }
    return out;
}

}  // namespace xs::cli
