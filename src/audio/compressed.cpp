// MP3 and FLAC through the vendored dr_mp3 and dr_flac (third_party/dr_libs, compiled in dr_libs.cpp).

#include <memory>
#include <utility>

#include <dr_flac.h>
#include <dr_mp3.h>

#include "audio/codecs.h"
#include "util/error.h"

namespace xs::audio::codecs {

namespace {

// Reads a decoder's frames in blocks until it stops: memory grows with what the file holds, not what it claims.
template <typename Read>
std::vector<float> read_all(int channels, Read read) {
    constexpr uint64_t kBlock = 4096;
    std::vector<float> interleaved;
    for (;;) {
        const size_t used = interleaved.size();
        interleaved.resize(used + kBlock * static_cast<size_t>(channels));
        const uint64_t got = read(kBlock, interleaved.data() + used);
        interleaved.resize(used + static_cast<size_t>(got) * static_cast<size_t>(channels));
        if (got < kBlock) return interleaved;
    }
}

}  // namespace

Audio decode_mp3(std::span<const uint8_t> bytes) {
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, bytes.data(), bytes.size(), nullptr)) {
        fail("audio: the MP3 file could not be decoded (no MPEG audio frame was found)");
    }
    const std::unique_ptr<drmp3, decltype(&drmp3_uninit)> close(&mp3, drmp3_uninit);
    const int channels = static_cast<int>(mp3.channels);
    std::vector<float> interleaved =
        read_all(channels, [&](uint64_t n, float* out) { return drmp3_read_pcm_frames_f32(&mp3, n, out); });
    const uint64_t frames = interleaved.size() / static_cast<size_t>(channels);
    // A Xing / LAME tag announces the length (delay and padding included), so a cut file shows.
    if (mp3.totalPCMFrameCount != DRMP3_UINT64_MAX) {
        const uint64_t trim = uint64_t{mp3.delayInPCMFrames} + mp3.paddingInPCMFrames;
        const uint64_t announced = mp3.totalPCMFrameCount > trim ? mp3.totalPCMFrameCount - trim : 0;
        if (frames < announced) {
            fail("audio: the MP3 file is truncated (its header announces ", announced, " samples per channel, ",
                 frames, " were decoded)");
        }
    }
    if (frames == 0) fail("audio: the MP3 file holds no audio");
    Audio audio;
    audio.sample_rate = static_cast<int>(mp3.sampleRate);
    audio.samples = average_channels(std::move(interleaved), channels);
    return audio;
}

Audio decode_flac(std::span<const uint8_t> bytes) {
    const std::unique_ptr<drflac, decltype(&drflac_close)> flac(drflac_open_memory(bytes.data(), bytes.size(), nullptr),
                                                                 drflac_close);
    if (!flac) fail("audio: the FLAC file could not be decoded (its stream header is malformed)");
    const int channels = flac->channels;
    std::vector<float> interleaved =
        read_all(channels, [&](uint64_t n, float* out) { return drflac_read_pcm_frames_f32(flac.get(), n, out); });
    const uint64_t frames = interleaved.size() / static_cast<size_t>(channels);
    // STREAMINFO's total is 0 when the encoder did not know it.
    if (frames < flac->totalPCMFrameCount) {
        fail("audio: the FLAC file is truncated (its header announces ", flac->totalPCMFrameCount,
             " samples per channel, ", frames, " were decoded)");
    }
    Audio audio;
    audio.sample_rate = static_cast<int>(flac->sampleRate);
    audio.samples = average_channels(std::move(interleaved), channels);
    return audio;
}

}  // namespace xs::audio::codecs
