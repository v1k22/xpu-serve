#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine/types.h"
#include "models/model.h"

namespace xs {

// Replaces the k-th media placeholder in `prompt` (as the chat template rendered it) with the token run of media[k],
// read by the encoder of its modality among `encoders` (null where the model takes no such media). Throws xs::Error
// when placeholders and items do not pair up one to one, in kind and in order: e.g. a template that did not render
// an item, or message text that spells a placeholder token itself.
std::vector<int32_t> expand_media_placeholders(std::span<const int32_t> prompt, std::span<const MediaInput> media,
                                               std::span<const MediaEncoder* const> encoders);

}  // namespace xs
