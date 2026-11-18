#include "engine/media.h"

#include <array>

#include "util/error.h"

namespace xs {

std::vector<int32_t> expand_media_placeholders(std::span<const int32_t> prompt, std::span<const MediaInput> media,
                                               std::span<const MediaEncoder* const> encoders) {
    // The modality whose placeholder `token` is, if any.
    const auto placeholder_of = [&](int32_t token) -> const MediaEncoder* {
        for (const MediaEncoder* e : encoders) {
            if (e != nullptr && e->placeholder_token() == token) return e;
        }
        return nullptr;
    };
    std::array<size_t, kModalities> placeholders{}, items{};
    for (const int32_t token : prompt) {
        if (const MediaEncoder* e = placeholder_of(token)) ++placeholders[static_cast<size_t>(e->modality())];
    }
    for (const MediaInput& m : media) ++items[static_cast<size_t>(m.modality)];
    for (size_t k = 0; k < kModalities; ++k) {
        if (placeholders[k] == items[k]) continue;
        const char* name = modality_name(static_cast<Modality>(k));
        fail("the prompt has ", placeholders[k], " ", name, " placeholders for ", items[k], " ", name, " inputs: the ",
             "chat template must render each ", name, " part once, and message text must not contain the ", name,
             " token itself");
    }
    std::vector<int32_t> out;
    out.reserve(prompt.size());
    size_t k = 0;
    for (const int32_t token : prompt) {
        const MediaEncoder* e = placeholder_of(token);
        if (e == nullptr) {
            out.push_back(token);
            continue;
        }
        const MediaInput& item = media[k++];
        if (item.modality != e->modality()) {
            fail("media item ", k, " is ", modality_name(item.modality), " but the prompt's placeholder ", k, " is ",
                 modality_name(e->modality()), ": the items must come in the order the messages hold them");
        }
        const std::vector<int32_t> run = e->token_run(item.num_tokens);
        out.insert(out.end(), run.begin(), run.end());
    }
    return out;
}

}  // namespace xs
