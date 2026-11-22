#pragma once

// Readers of a request body's fields that refuse a wrong one by name (InvalidRequest, a 400): what the OpenAI and the
// Anthropic request parsers share. A field set to null counts as absent.

#include <cstdint>
#include <optional>
#include <string>

#include "server/openai_request.h"
#include "util/error.h"

namespace xs::server::fields {

[[noreturn]] inline void reject(const std::string& param, const std::string& message) {
    throw InvalidRequest(message, param);
}

// The field's value, or nullptr when it is absent or null.
inline const RequestJson* find(const RequestJson& object, const char* name) {
    const auto it = object.find(name);
    return it == object.end() || it->is_null() ? nullptr : &*it;
}

inline float number(const RequestJson& body, const char* name, float fallback, double min, double max) {
    const RequestJson* v = find(body, name);
    if (v == nullptr) return fallback;
    if (!v->is_number() || v->get<double>() < min || v->get<double>() > max) {
        reject(name, str_cat("'", name, "' must be a number between ", min, " and ", max));
    }
    return v->get<float>();
}

inline std::optional<int64_t> integer(const RequestJson& body, const char* name, int64_t min, int64_t max) {
    const RequestJson* v = find(body, name);
    if (v == nullptr) return std::nullopt;
    // Parsed JSON stores non-negative integers as unsigned, so both representations are range-checked.
    bool in_range = false;
    if (v->is_number_unsigned()) {
        const uint64_t u = v->get<uint64_t>();
        in_range = u <= uint64_t(max) && (min <= 0 || u >= uint64_t(min));
    } else if (v->is_number_integer()) {
        in_range = v->get<int64_t>() >= min && v->get<int64_t>() <= max;
    }
    if (!in_range) reject(name, str_cat("'", name, "' must be an integer between ", min, " and ", max));
    return v->get<int64_t>();
}

inline bool boolean(const RequestJson& object, const char* name, const std::string& param) {
    const RequestJson* v = find(object, name);
    if (v == nullptr) return false;
    if (!v->is_boolean()) reject(param, str_cat("'", param, "' must be a boolean"));
    return v->get<bool>();
}

inline const RequestJson* object(const RequestJson& body, const char* name) {
    const RequestJson* v = find(body, name);
    if (v != nullptr && !v->is_object()) reject(name, str_cat("'", name, "' must be an object"));
    return v;
}

}  // namespace xs::server::fields
