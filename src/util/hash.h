#pragma once

#include <cstdint>
#include <cstring>
#include <span>

namespace xs {

// A 64-bit hash of content (media caches key on it): FNV-1a over 64-bit words, then the murmur3 finalizer so nearby
// inputs spread over all bits. Not stable across versions; never store one.
class ContentHash {
public:
    void add(uint64_t word) {
        h_ ^= word;
        h_ *= 0x100000001b3ULL;
    }

    // Bytes as 64-bit words in memory order; the last word is the remaining bytes zero-padded, added even when empty.
    void add_bytes(std::span<const uint8_t> bytes) {
        const size_t words = bytes.size() / 8;
        for (size_t i = 0; i < words; ++i) {
            uint64_t w;
            std::memcpy(&w, bytes.data() + i * 8, 8);
            add(w);
        }
        uint64_t tail = 0;
        if (const size_t rest = bytes.size() - words * 8; rest > 0) std::memcpy(&tail, bytes.data() + words * 8, rest);
        add(tail);
    }

    uint64_t finish() const {
        uint64_t h = h_;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }

private:
    uint64_t h_ = 0xcbf29ce484222325ULL;
};

}  // namespace xs
