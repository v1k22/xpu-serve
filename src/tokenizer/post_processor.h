#pragma once

// tokenizer.json post-processors, for one sequence: TemplateProcessing (its `single` template) and ByteLevel (which
// only adjusts offsets, so adds nothing here). Others are refused by name.

#include <cstdint>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

namespace xs::tok {

class PostProcessor {
public:
    virtual ~PostProcessor() = default;
    // The ids of one sequence with the special tokens the post-processor puts around it.
    virtual std::vector<int32_t> single(std::vector<int32_t> ids) const = 0;
};

// Returns nullptr for a null post-processor or one that adds no tokens; refuses unsupported types by name.
std::unique_ptr<PostProcessor> make_post_processor(const nlohmann::json& config);

}  // namespace xs::tok
