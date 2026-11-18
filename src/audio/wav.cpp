// WAV files, read and written by hand: RIFF chunks, the fmt chunk's sample formats, RF64's 64-bit sizes.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string_view>

#include "audio/codecs.h"
#include "util/error.h"

namespace xs::audio {

static_assert(std::endian::native == std::endian::little, "WAV samples are read and written in host byte order");

namespace {

constexpr uint16_t kFormatPcm = 0x0001;
constexpr uint16_t kFormatFloat = 0x0003;
constexpr uint16_t kFormatExtensible = 0xFFFE;
constexpr uint32_t kSizeUnknown = 0xFFFFFFFF;  // RF64's data size, or a stream written before its length was known
constexpr std::string_view kConvert = "; convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav";

uint16_t le16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | p[1] << 8); }
uint32_t le32(const uint8_t* p) { return uint32_t{le16(p)} | uint32_t{le16(p + 2)} << 16; }
uint64_t le64(const uint8_t* p) { return uint64_t{le32(p)} | uint64_t{le32(p + 4)} << 32; }

bool is(const uint8_t* p, std::string_view id) { return std::memcmp(p, id.data(), 4) == 0; }

const char* format_tag_name(uint16_t tag) {
    switch (tag) {
        case 0x0002: return "Microsoft ADPCM";
        case 0x0006: return "A-law";
        case 0x0007: return "mu-law";
        case 0x0011: return "IMA ADPCM";
        case 0x0031: return "GSM 6.10";
        case 0x0050: return "MPEG audio";
        case 0x0055: return "MP3";
        case 0x00FF: return "AAC";
        case 0x0161: return "Windows Media Audio";
        case 0x2000: return "AC-3";
        default: return "this format";
    }
}

struct Format {
    bool is_float = false;
    int channels = 0;
    int sample_rate = 0;
    int bytes_per_sample = 0;  // the container: integer samples of up to 8 bits take one byte, of 9 to 16 two, ...
};

Format parse_fmt(const uint8_t* p, uint64_t size) {
    if (size < 16) fail("audio: the WAV file's fmt chunk is ", size, " bytes, shorter than 16");
    uint16_t tag = le16(p);
    if (tag == kFormatExtensible) {
        // The sub-format is a GUID whose first two bytes are a format tag and the rest KSDATAFORMAT_SUBTYPE's.
        static constexpr uint8_t kGuidTail[14] = {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80,
                                                  0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71};
        if (size < 40) fail("audio: the WAV file's WAVE_FORMAT_EXTENSIBLE fmt chunk is ", size, " bytes, not 40");
        if (std::memcmp(p + 26, kGuidTail, sizeof(kGuidTail)) != 0) {
            fail("audio: WAV files whose WAVE_FORMAT_EXTENSIBLE sub-format is not a format tag GUID are not supported",
                 kConvert);
        }
        tag = le16(p + 24);
    }
    if (tag != kFormatPcm && tag != kFormatFloat) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "0x%04x", tag);
        fail("audio: WAV files holding ", format_tag_name(tag), " (format tag ", hex,
             ") are not supported (integer PCM and IEEE float WAV, MP3 and FLAC are)", kConvert);
    }
    Format f;
    f.is_float = tag == kFormatFloat;
    f.channels = le16(p + 2);
    const uint32_t rate = le32(p + 4);
    const int block_align = le16(p + 12);
    const int bits = le16(p + 14);
    if (f.channels < 1) fail("audio: the WAV file has no channels");
    if (rate < 1 || rate > static_cast<uint32_t>(kMaxSampleRate)) {
        fail("audio: a sample rate of ", rate, " Hz is not supported (1 to ", kMaxSampleRate, " Hz are)");
    }
    f.sample_rate = static_cast<int>(rate);
    if (f.is_float && bits != 32 && bits != 64) {
        fail("audio: IEEE float WAV samples of ", bits, " bits are not supported (32 and 64 are)", kConvert);
    }
    if (!f.is_float && (bits < 1 || bits > 32)) {
        fail("audio: integer PCM WAV samples of ", bits, " bits are not supported (1 to 32 are)", kConvert);
    }
    f.bytes_per_sample = (bits + 7) / 8;
    if (block_align != f.channels * f.bytes_per_sample) {
        fail("audio: the WAV file's block align is ", block_align, " bytes, where ", f.channels, " channels of ", bits,
             "-bit samples take ", f.channels * f.bytes_per_sample);
    }
    return f;
}

// A sample's value as libsndfile reads it into float: integers divided by 2^(8 * bytes - 1), float as stored.
std::vector<float> samples_as_float(const Format& f, std::span<const uint8_t> data) {
    const size_t n = data.size() / static_cast<size_t>(f.bytes_per_sample);
    std::vector<float> out(n);
    const uint8_t* p = data.data();
    if (f.is_float && f.bytes_per_sample == 4) {
        std::memcpy(out.data(), p, n * 4);
    } else if (f.is_float) {
        for (size_t i = 0; i < n; ++i) {
            double v;
            std::memcpy(&v, p + 8 * i, 8);
            out[i] = static_cast<float>(v);
        }
    } else if (f.bytes_per_sample == 1) {
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(int{p[i]} - 128) / 128.0f;
    } else if (f.bytes_per_sample == 2) {
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(static_cast<int16_t>(le16(p + 2 * i))) / 32768.0f;
    } else if (f.bytes_per_sample == 3) {
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* s = p + 3 * i;
            const uint32_t high = uint32_t{s[0]} << 8 | uint32_t{s[1]} << 16 | uint32_t{s[2]} << 24;
            out[i] = static_cast<float>(static_cast<int32_t>(high) >> 8) / 8388608.0f;  // sign-extended
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>(static_cast<int32_t>(le32(p + 4 * i))) / 2147483648.0f;
        }
    }
    return out;
}

void put16(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
}
void put32(std::vector<uint8_t>& out, uint32_t v) {
    put16(out, v & 0xFFFF);
    put16(out, v >> 16);
}
void put_id(std::vector<uint8_t>& out, std::string_view id) { out.insert(out.end(), id.begin(), id.end()); }

}  // namespace

namespace codecs {

Audio decode_wav(std::span<const uint8_t> bytes) {
    if (bytes.size() < 12) fail("audio: the WAV file is truncated (", bytes.size(), " bytes)");
    const bool rf64 = !is(bytes.data(), "RIFF");
    std::optional<Format> format;
    std::optional<std::span<const uint8_t>> data;
    bool data_size_unknown = false;
    std::optional<uint64_t> ds64_data_size;
    size_t pos = 12;
    while (bytes.size() - pos >= 8) {
        const uint8_t* id = bytes.data() + pos;
        uint64_t size = le32(id + 4);
        pos += 8;
        const size_t left = bytes.size() - pos;
        if (rf64 && is(id, "ds64")) {
            if (size < 28 || size > left) fail("audio: the RF64 file's ds64 chunk is malformed (", size, " bytes)");
            ds64_data_size = le64(bytes.data() + pos + 8);
        }
        if (is(id, "data") && size == kSizeUnknown) {
            if (rf64 && !ds64_data_size) fail("audio: the RF64 file has no ds64 chunk before its data");
            data_size_unknown = !rf64;
            size = rf64 ? *ds64_data_size : left;
        }
        if (size > left) {
            if (is(id, "data")) {
                fail("audio: the WAV file is truncated (its data chunk declares ", size, " bytes, ", left, " follow)");
            }
            if (format && data) break;  // a cut chunk after the fmt and data chunks: nothing in it is read
            fail("audio: the WAV file is truncated (its '", std::string_view(reinterpret_cast<const char*>(id), 4),
                 "' chunk declares ", size, " bytes, ", left, " follow)");
        }
        if (is(id, "fmt ")) format = parse_fmt(bytes.data() + pos, size);
        if (is(id, "data")) data = bytes.subspan(pos, size);
        pos += size + (size & 1);
        if (pos > bytes.size()) break;  // the pad byte of an odd-sized last chunk is often left out
    }
    if (!format) fail("audio: the WAV file has no fmt chunk");
    if (!data) fail("audio: the WAV file has no data chunk");
    const size_t frame_bytes = static_cast<size_t>(format->channels) * static_cast<size_t>(format->bytes_per_sample);
    std::span<const uint8_t> samples = *data;
    if (samples.size() % frame_bytes != 0) {
        if (!data_size_unknown) {
            fail("audio: the WAV file's data chunk holds ", samples.size(), " bytes, not a whole number of ",
                 frame_bytes, "-byte frames");
        }
        samples = samples.first(samples.size() - samples.size() % frame_bytes);  // a stream cut mid-frame
    }
    Audio audio;
    audio.sample_rate = format->sample_rate;
    audio.samples = average_channels(samples_as_float(*format, samples), format->channels);
    return audio;
}

}  // namespace codecs

std::vector<uint8_t> encode_wav(const Audio& audio, WavSample sample) {
    if (audio.sample_rate < 1 || audio.sample_rate > kMaxSampleRate) {
        fail("audio: a WAV file at ", audio.sample_rate, " Hz cannot be written (1 to ", kMaxSampleRate, " Hz can)");
    }
    const bool pcm = sample == WavSample::Pcm16;
    const uint64_t bytes_per_sample = pcm ? 2 : 4;
    const uint64_t data_bytes = audio.samples.size() * bytes_per_sample;
    const uint64_t fmt_bytes = pcm ? 16 : 18;            // IEEE float carries cbSize = 0
    const uint64_t fact_bytes = pcm ? 0 : 8 + 4;         // and a fact chunk with the frame count
    const uint64_t riff_bytes = 4 + (8 + fmt_bytes) + fact_bytes + (8 + data_bytes);
    if (riff_bytes > 0xFFFFFFFFull) fail("audio: ", audio.samples.size(), " samples are too many for a WAV file");
    for (size_t i = 0; i < audio.samples.size(); ++i) {
        if (!std::isfinite(audio.samples[i])) fail("audio: sample ", i, " is not a finite number");
    }

    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(riff_bytes) + 8);
    put_id(out, "RIFF");
    put32(out, static_cast<uint32_t>(riff_bytes));
    put_id(out, "WAVE");
    put_id(out, "fmt ");
    put32(out, static_cast<uint32_t>(fmt_bytes));
    put16(out, pcm ? kFormatPcm : kFormatFloat);
    put16(out, 1);
    put32(out, static_cast<uint32_t>(audio.sample_rate));
    put32(out, static_cast<uint32_t>(audio.sample_rate * bytes_per_sample));
    put16(out, static_cast<uint32_t>(bytes_per_sample));
    put16(out, static_cast<uint32_t>(bytes_per_sample * 8));
    if (!pcm) {
        put16(out, 0);
        put_id(out, "fact");
        put32(out, 4);
        put32(out, static_cast<uint32_t>(audio.samples.size()));
    }
    put_id(out, "data");
    put32(out, static_cast<uint32_t>(data_bytes));
    const size_t start = out.size();
    out.resize(start + static_cast<size_t>(data_bytes));
    uint8_t* p = out.data() + start;
    if (pcm) {
        for (const float x : audio.samples) {
            const auto v = static_cast<int16_t>(std::lrint(std::clamp(x * 32768.0f, -32768.0f, 32767.0f)));
            p[0] = static_cast<uint8_t>(v);
            p[1] = static_cast<uint8_t>(static_cast<uint16_t>(v) >> 8);
            p += 2;
        }
    } else {
        std::memcpy(p, audio.samples.data(), static_cast<size_t>(data_bytes));
    }
    return out;
}

}  // namespace xs::audio
