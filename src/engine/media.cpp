#include "engine/media.h"

#include <algorithm>
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

PromptPositions rotary_positions(std::span<const int32_t> item, std::span<const int64_t> row,
                                 std::span<const MediaInput> media) {
    XS_CHECK(item.size() == row.size(), item.size(), " media items for ", row.size(), " rows");
    PromptPositions out;
    out.positions.resize(item.size());
    int64_t base = 0;
    for (size_t i = 0; i < item.size(); ++i) {
        if (item[i] < 0) {
            const auto p = static_cast<int32_t>(out.next++);
            out.positions[i] = {p, p, p};
            continue;
        }
        const MediaInput& m = media[static_cast<size_t>(item[i])];
        XS_CHECK(m.positions.empty() || static_cast<int64_t>(m.positions.size()) == m.num_tokens, "a ",
                 modality_name(m.modality), " item with ", m.positions.size(), " positions for ", m.num_tokens,
                 " tokens");
        const int64_t r = row[i];
        if (r == 0) base = out.next;
        const auto r32 = static_cast<int32_t>(r);
        const RotaryPosition offset =
            m.positions.empty() ? RotaryPosition{r32, r32, r32} : m.positions[static_cast<size_t>(r)];
        for (size_t a = 0; a < 3; ++a) out.positions[i][a] = static_cast<int32_t>(base + offset[a]);
        if (r + 1 == m.num_tokens) {
            int64_t largest = m.num_tokens - 1;
            if (!m.positions.empty()) {
                largest = 0;
                for (const RotaryPosition& p : m.positions) {
                    largest = std::max<int64_t>({largest, p[0], p[1], p[2]});
                }
            }
            out.next = base + largest + 1;
        }
    }
    return out;
}

}  // namespace xs
