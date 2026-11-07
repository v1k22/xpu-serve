#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine/types.h"
#include "models/model.h"

namespace xs {

// Replaces the k-th image placeholder in `prompt` (as the chat template rendered it) with the token run of
// media[k]. Throws xs::Error when placeholders and images do not pair up one to one, e.g. a template that did not
// render an image, or message text that spells the placeholder token itself.
std::vector<int32_t> expand_media_placeholders(std::span<const int32_t> prompt, std::span<const MediaInput> media,
                                               const VisionEncoder& vision);

}  // namespace xs
