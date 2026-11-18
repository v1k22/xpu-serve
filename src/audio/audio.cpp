#include "audio/audio.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

#include "audio/codecs.h"
#include "util/error.h"
#include "util/hash.h"

namespace xs::audio {

namespace {

bool has(std::span<const uint8_t> b, size_t offset, std::string_view text) {
    return b.size() >= offset + text.size() && std::memcmp(b.data() + offset, text.data(), text.size()) == 0;
}

bool contains(std::span<const uint8_t> b, std::string_view text) {
    return std::search(b.begin(), b.end(), text.begin(), text.end()) != b.end();
}

// An MPEG audio frame header: 11 sync bits, then a version and a layer that are not the reserved values.
bool mpeg_audio_sync(std::span<const uint8_t> b, size_t offset) {
    if (b.size() < offset + 2 || b[offset] != 0xFF) return false;
    const uint8_t h = b[offset + 1];
    return (h & 0xE0) == 0xE0 && (h & 0x18) != 0x08 && (h & 0x06) != 0;
}

// The bytes an ID3v2 tag at the start takes (its header, a syncsafe size, and a footer when flagged), or 0.
size_t id3v2_size(std::span<const uint8_t> b) {
    if (!has(b, 0, "ID3") || b.size() < 10) return 0;
    const size_t body = (size_t{b[6]} & 0x7F) << 21 | (size_t{b[7]} & 0x7F) << 14 | (size_t{b[8]} & 0x7F) << 7 |
                        (size_t{b[9]} & 0x7F);
    return 10 + body + ((b[5] & 0x10) ? 10 : 0);
}

struct Container {
    const char* name;
    const char* extension;  // for the conversion command
};

// Containers people send that the decoder does not read, by their signature, so the refusal can name them.
std::optional<Container> unsupported_container(std::span<const uint8_t> b) {
    if (has(b, 0, "OggS")) {
        const auto first_page = b.first(std::min<size_t>(b.size(), 128));
        if (contains(first_page, "OpusHead")) return Container{"Ogg Opus", "ogg"};
        if (contains(first_page, "\x01vorbis")) return Container{"Ogg Vorbis", "ogg"};
        if (contains(first_page, "\x7f" "FLAC")) return Container{"Ogg FLAC", "ogg"};
        return Container{"Ogg", "ogg"};
    }
    if (has(b, 4, "ftyp")) {
        if (has(b, 8, "M4A ") || has(b, 8, "M4B ") || has(b, 8, "M4P ")) return Container{"M4A", "m4a"};
        if (has(b, 8, "qt  ")) return Container{"QuickTime (MOV)", "mov"};
        if (has(b, 8, "3g")) return Container{"3GP", "3gp"};
        return Container{"MP4", "mp4"};
    }
    if (has(b, 0, "\x1A\x45\xDF\xA3")) {
        if (contains(b.first(std::min<size_t>(b.size(), 64)), "webm")) return Container{"WebM", "webm"};
        return Container{"Matroska", "mkv"};
    }
    if (has(b, 0, "FORM") && (has(b, 8, "AIFF") || has(b, 8, "AIFC"))) return Container{"AIFF", "aiff"};
    if (has(b, 0, "RIFF") && has(b, 8, "AVI ")) return Container{"AVI", "avi"};
    if (has(b, 0, "caff")) return Container{"CAF", "caf"};
    if (has(b, 0, ".snd")) return Container{"Sun AU", "au"};
    if (has(b, 0, "#!AMR")) return Container{"AMR", "amr"};
    if (has(b, 0, "\x30\x26\xB2\x75\x8E\x66\xCF\x11")) return Container{"ASF (WMA / WMV)", "wma"};
    if (has(b, 0, "wvpk")) return Container{"WavPack", "wv"};
    if (has(b, 0, "MAC ")) return Container{"Monkey's Audio", "ape"};
    if (has(b, 0, "FLV")) return Container{"FLV", "flv"};
    if (has(b, 0, "ADIF")) return Container{"AAC (ADIF)", "aac"};
    // ADTS shares MPEG audio's sync word, with layer bits 00.
    if (b.size() >= 2 && b[0] == 0xFF && (b[1] & 0xF6) == 0xF0) return Container{"AAC (ADTS)", "aac"};
    if (b.size() > 188 && b[0] == 0x47 && b[188] == 0x47) return Container{"MPEG transport stream", "ts"};
    if (has(b, 0, std::string_view("\x00\x00\x01\xBA", 4))) return Container{"MPEG program stream", "mpg"};
    return std::nullopt;
}

std::string leading_bytes(std::span<const uint8_t> b) {
    std::string out;
    for (size_t i = 0; i < std::min<size_t>(b.size(), 8); ++i) {
        char hex[4];
        std::snprintf(hex, sizeof(hex), i ? " %02x" : "%02x", b[i]);
        out += hex;
    }
    return out;
}

Audio decode_container(std::span<const uint8_t> bytes) {
    if ((has(bytes, 0, "RIFF") || has(bytes, 0, "RF64") || has(bytes, 0, "BW64")) && has(bytes, 8, "WAVE")) {
        return codecs::decode_wav(bytes);
    }
    if (has(bytes, 0, "fLaC")) return codecs::decode_flac(bytes);
    if (const size_t tag = id3v2_size(bytes); tag > 0) {
        if (has(bytes, tag, "fLaC")) return codecs::decode_flac(bytes);  // dr_flac skips the tag
        return codecs::decode_mp3(bytes);  // dr_mp3 skips the tag and finds the first frame
    }
    if (mpeg_audio_sync(bytes, 0)) return codecs::decode_mp3(bytes);
    if (const auto refused = unsupported_container(bytes)) {
        fail("audio: ", refused->name, " files are not supported (WAV, MP3 and FLAC are); convert with: ffmpeg -i in.",
             refused->extension, " -ar 16000 -ac 1 out.wav");
    }
    if (has(bytes, 0, "RIFF") && bytes.size() >= 12) {
        fail("audio: RIFF files of form '", std::string_view(reinterpret_cast<const char*>(bytes.data()) + 8, 4),
             "' are not supported (WAV, MP3 and FLAC are); convert with: ffmpeg -i <file> -ar 16000 -ac 1 out.wav");
    }
    fail("audio: the data is not a WAV, MP3 or FLAC file (it starts with bytes ", leading_bytes(bytes),
         "); convert with: ffmpeg -i <file> -ar 16000 -ac 1 out.wav");
}

}  // namespace

namespace codecs {

std::vector<float> average_channels(std::vector<float> interleaved, int channels) {
    if (channels == 1) return interleaved;
    const size_t n = static_cast<size_t>(channels);
    const float count = static_cast<float>(channels);
    std::vector<float> mono(interleaved.size() / n);
    for (size_t i = 0; i < mono.size(); ++i) {
        const float* frame = interleaved.data() + i * n;
        float sum = frame[0];
        for (size_t c = 1; c < n; ++c) sum += frame[c];
        mono[i] = sum / count;
    }
    return mono;
}

}  // namespace codecs

Audio decode(std::span<const uint8_t> bytes) {
    if (bytes.empty()) fail("audio: the file is empty");
    Audio audio = decode_container(bytes);
    if (audio.sample_rate < 1 || audio.sample_rate > kMaxSampleRate) {
        fail("audio: a sample rate of ", audio.sample_rate, " Hz is not supported (1 to ", kMaxSampleRate, " Hz are)");
    }
    return audio;
}

uint64_t content_hash(const Audio& audio) {
    ContentHash h;
    h.add(static_cast<uint64_t>(audio.sample_rate));
    h.add(audio.samples.size());
    h.add_bytes({reinterpret_cast<const uint8_t*>(audio.samples.data()), audio.samples.size() * sizeof(float)});
    return h.finish();
}

}  // namespace xs::audio
