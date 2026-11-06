#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace xs::cli {

// Minimal flag parser: "--name value", "--name=value" and boolean "--flag". Positionals are kept in order.
// Every flag must be declared; unknown flags are refused by name.
class Args {
public:
    Args& value(const std::string& name, std::string help, std::string fallback = {});
    Args& flag(const std::string& name, std::string help);

    // Throws xs::Error on an unknown flag or a missing value.
    void parse(const std::vector<std::string>& argv);

    const std::string& get(const std::string& name) const;
    int get_int(const std::string& name) const;
    double get_double(const std::string& name) const;
    bool has(const std::string& name) const;  // flag given, or value given explicitly
    const std::vector<std::string>& positionals() const { return positionals_; }
    std::string usage() const;

private:
    struct Spec {
        std::string help;
        std::string value;
        bool is_flag = false;
        bool given = false;
    };
    std::map<std::string, Spec> specs_;
    std::vector<std::string> order_;
    std::vector<std::string> positionals_;
};

}  // namespace xs::cli
