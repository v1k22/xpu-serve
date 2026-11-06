#include <algorithm>

#include "tokenizer/tokenizer.h"

namespace xs::tok {
namespace {
constexpr size_t kContextTokens = 4;  // enough look-back for space/metaspace handling in decoders
constexpr size_t kMaxPending = 8;     // a UTF-8 character never spans more byte-fallback tokens than this

bool ends_with_replacement(const std::string& s) {
    static const std::string kReplacement = "\xEF\xBF\xBD";
    return s.size() >= kReplacement.size() && s.compare(s.size() - kReplacement.size(), kReplacement.size(), kReplacement) == 0;
}
}  // namespace

StreamDecoder::StreamDecoder(const Tokenizer& tokenizer, bool skip_special)
    : tokenizer_(tokenizer), skip_special_(skip_special) {}

std::string StreamDecoder::push(int32_t id) {
    if (skip_special_ && tokenizer_.is_special(id)) return {};
    pending_.push_back(id);
    std::vector<int32_t> all = context_;
    all.insert(all.end(), pending_.begin(), pending_.end());
    const std::string full = tokenizer_.decode(all, skip_special_);
    if (ends_with_replacement(full) && pending_.size() < kMaxPending) return {};  // a character is still incomplete
    const std::string prefix = tokenizer_.decode(context_, skip_special_);
    std::string delta = full.compare(0, prefix.size(), prefix) == 0 ? full.substr(prefix.size()) : full;
    context_ = std::move(all);
    if (context_.size() > kContextTokens) context_.erase(context_.begin(), context_.end() - kContextTokens);
    pending_.clear();
    return delta;
}

std::string StreamDecoder::flush() {
    if (pending_.empty()) return {};
    std::vector<int32_t> all = context_;
    all.insert(all.end(), pending_.begin(), pending_.end());
    const std::string full = tokenizer_.decode(all, skip_special_);
    const std::string prefix = tokenizer_.decode(context_, skip_special_);
    pending_.clear();
    context_.clear();
    return full.compare(0, prefix.size(), prefix) == 0 ? full.substr(prefix.size()) : full;
}

}  // namespace xs::tok
