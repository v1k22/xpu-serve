#include "tokenizer/tokenizer_json.h"

#include <string>
#include <utility>

namespace xs::tok {
namespace {

nlohmann::json byte_level(bool add_prefix_space) {
    return {
        {"type", "ByteLevel"}, {"add_prefix_space", add_prefix_space}, {"trim_offsets", false}, {"use_regex", false}};
}

}  // namespace

TokenizerPipeline qwen_byte_level_pipeline(std::string_view split_regex, bool add_prefix_space) {
    nlohmann::json split = {{"type", "Split"},
                            {"pattern", {{"Regex", std::string(split_regex)}}},
                            {"behavior", "Isolated"},
                            {"invert", false}};
    TokenizerPipeline p;
    p.normalizer = {{"type", "NFC"}};
    p.pre_tokenizer = {{"type", "Sequence"},
                       {"pretokenizers", nlohmann::json::array({std::move(split), byte_level(add_prefix_space)})}};
    p.post_processor = byte_level(false);
    p.decoder = byte_level(false);
    p.model = {{"type", "BPE"},
               {"dropout", nullptr},
               {"unk_token", nullptr},
               {"continuing_subword_prefix", ""},
               {"end_of_word_suffix", ""},
               {"fuse_unk", false},
               {"byte_fallback", false},
               {"ignore_merges", false}};
    return p;
}

nlohmann::json make_tokenizer_json(TokenizerPipeline pipeline, nlohmann::json vocab, nlohmann::json merges,
                                   nlohmann::json added_tokens) {
    nlohmann::json out = {{"version", "1.0"}, {"truncation", nullptr}, {"padding", nullptr}};
    out["added_tokens"] = std::move(added_tokens);
    out["normalizer"] = std::move(pipeline.normalizer);
    out["pre_tokenizer"] = std::move(pipeline.pre_tokenizer);
    out["post_processor"] = std::move(pipeline.post_processor);
    out["decoder"] = std::move(pipeline.decoder);
    nlohmann::json& model = out["model"] = std::move(pipeline.model);
    model["vocab"] = std::move(vocab);
    model["merges"] = std::move(merges);
    return out;
}

}  // namespace xs::tok
