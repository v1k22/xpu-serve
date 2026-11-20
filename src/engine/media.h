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

// The rotary positions of a prompt's tokens (transformers' get_rope_index): text one past what came before, a media
// item's tokens at their offsets from the item's first position (MediaInput::positions; none: one a token), the token
// after an item one past its largest. `item[i]` is token i's media item (-1: text) and `row[i]` its row in it. `next`
// is the position after the prompt, where generated tokens go on one a token.
struct PromptPositions {
    std::vector<RotaryPosition> positions;
    int64_t next = 0;
};
PromptPositions rotary_positions(std::span<const int32_t> item, std::span<const int64_t> row,
                                 std::span<const MediaInput> media);

}  // namespace xs
