#include "engine/media.h"

#include <algorithm>

#include "util/error.h"

namespace xs {

std::vector<int32_t> expand_media_placeholders(std::span<const int32_t> prompt, std::span<const MediaInput> media,
                                               const VisionEncoder& vision) {
    const int32_t placeholder = vision.placeholder_token();
    const auto placeholders = static_cast<size_t>(std::count(prompt.begin(), prompt.end(), placeholder));
    if (placeholders != media.size()) {
        fail("the prompt has ", placeholders, " image placeholders for ", media.size(),
             " images: the chat template must render each image part once, and message text must not contain the "
             "image token itself");
    }
    std::vector<int32_t> out;
    out.reserve(prompt.size());
    size_t k = 0;
    for (const int32_t token : prompt) {
        if (token != placeholder) {
            out.push_back(token);
            continue;
        }
        const std::vector<int32_t> run = vision.token_run(media[k++].num_tokens);
        out.insert(out.end(), run.begin(), run.end());
    }
    return out;
}

}  // namespace xs
