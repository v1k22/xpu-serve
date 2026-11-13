#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xs {

// Standard base64 (RFC 4648 alphabet). Padding is optional; ASCII whitespace is skipped (line-wrapped payloads).
// Returns nullopt on any other character or a truncated final group.
std::optional<std::vector<uint8_t>> base64_decode(std::string_view text);
// Standard base64 with padding.
std::string base64_encode(std::span<const uint8_t> bytes);

}  // namespace xs
