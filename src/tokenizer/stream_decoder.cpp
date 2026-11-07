#include <algorithm>

#include "tokenizer/tokenizer.h"

namespace xs::tok {
namespace {
// After this many pending tokens, text ending in U+FFFD is taken as final (it is a real U+FFFD or invalid bytes).
constexpr size_t kMaxPending = 32;
constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

bool ends_with_replacement(std::string_view s) { return s.ends_with(kReplacement); }
}  // namespace

StreamDecoder::StreamDecoder(const Tokenizer& tokenizer, bool skip_special)
    : tokenizer_(tokenizer), skip_special_(skip_special) {}

std::string StreamDecoder::push(int32_t id) {
    if (skip_special_ && tokenizer_.is_special(id)) return {};
    pending_.push_back(id);
    std::vector<int32_t> all = context_;
    all.insert(all.end(), pending_.begin(), pending_.end());
    const std::string full = tokenizer_.decode(all, skip_special_);
    const std::string prefix = tokenizer_.decode(context_, skip_special_);
    const bool extends = full.starts_with(prefix);
    if (ends_with_replacement(full) && pending_.size() < kMaxPending) {
        // A character is still incomplete. The text before the trailing U+FFFD run is final already (a token can
        // end with complete characters followed by the first bytes of the next one); return it, keep the rest.
        if (!extends) return {};
        std::string_view stable = std::string_view(full).substr(prefix.size());
        while (ends_with_replacement(stable)) stable.remove_suffix(kReplacement.size());
        if (stable.size() <= emitted_) return {};
        std::string delta(stable.substr(emitted_));
        emitted_ = stable.size();
        return delta;
    }
    std::string delta = extends ? full.substr(prefix.size()) : full;
    delta.erase(0, std::min(emitted_, delta.size()));
    // The next context is the chunk just emitted: it starts at a character boundary, unlike an arbitrary tail of
    // tokens, which can cut a byte run and decode to U+FFFD.
    context_ = std::move(pending_);
    pending_.clear();
    emitted_ = 0;
    return delta;
}

std::string StreamDecoder::flush() {
    if (pending_.empty()) return {};
    std::vector<int32_t> all = context_;
    all.insert(all.end(), pending_.begin(), pending_.end());
    const std::string full = tokenizer_.decode(all, skip_special_);
    const std::string prefix = tokenizer_.decode(context_, skip_special_);
    std::string delta = full.starts_with(prefix) ? full.substr(prefix.size()) : full;
    delta.erase(0, std::min(emitted_, delta.size()));
    pending_.clear();
    context_.clear();
    emitted_ = 0;
    return delta;
}

}  // namespace xs::tok
