// Audio decoding (WAV by hand, MP3 and FLAC through dr_libs), the WAV writer and content hashing.
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <tuple>

#include "audio/audio.h"
#include "audio_test_util.h"
#include "util/error.h"

using namespace xs;
using namespace std::literals;

namespace {

void put(std::vector<uint8_t>& out, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void put_id(std::vector<uint8_t>& out, std::string_view id) { out.insert(out.end(), id.begin(), id.end()); }

std::vector<uint8_t> chunk(std::string_view id, std::span<const uint8_t> body, std::optional<uint32_t> size = {}) {
    std::vector<uint8_t> out;
    put_id(out, id);
    put(out, size.value_or(static_cast<uint32_t>(body.size())), 4);
    out.insert(out.end(), body.begin(), body.end());
    if (body.size() % 2) out.push_back(0);
    return out;
}

struct Fmt {
    uint16_t tag = 1;  // the fmt chunk's format tag, or the sub-format of a WAVE_FORMAT_EXTENSIBLE one
    int channels = 1;
    uint32_t rate = 16000;
    int bits = 16;
    bool extensible = false;
    std::optional<int> block_align;  // default: what the channels and bits take
};

std::vector<uint8_t> fmt_chunk(const Fmt& f) {
    const int align = f.block_align.value_or(f.channels * ((f.bits + 7) / 8));
    std::vector<uint8_t> body;
    put(body, f.extensible ? 0xFFFE : f.tag, 2);
    put(body, static_cast<uint64_t>(f.channels), 2);
    put(body, f.rate, 4);
    put(body, f.rate * static_cast<uint32_t>(align), 4);
    put(body, static_cast<uint64_t>(align), 2);
    put(body, static_cast<uint64_t>(f.bits), 2);
    if (f.extensible) {
        put(body, 22, 2);                         // cbSize
        put(body, static_cast<uint64_t>(f.bits), 2);  // valid bits
        put(body, 0, 4);                          // channel mask
        put(body, f.tag, 2);                      // the sub-format GUID: the tag, then KSDATAFORMAT_SUBTYPE's tail
        for (const uint8_t b : {0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}) {
            body.push_back(b);
        }
    }
    return chunk("fmt ", body);
}

std::vector<uint8_t> riff(const std::vector<std::vector<uint8_t>>& chunks, std::string_view form = "WAVE") {
    std::vector<uint8_t> body;
    put_id(body, form);
    for (const auto& c : chunks) body.insert(body.end(), c.begin(), c.end());
    std::vector<uint8_t> out;
    put_id(out, "RIFF");
    put(out, body.size(), 4);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

std::vector<uint8_t> wav(const Fmt& f, std::span<const uint8_t> data) {
    return riff({fmt_chunk(f), chunk("data", data)});
}

std::span<const uint8_t> bytes_of(std::string_view s) {
    return {reinterpret_cast<const uint8_t*>(s.data()), s.size()};
}

// `count` samples of a WAV sample format as a data chunk holds them, and the float each one reads as: integers over
// the container's full range, extremes first, narrower samples left-justified in their container; floats as stored,
// values past [-1, 1] included.
struct Samples {
    std::vector<uint8_t> data;
    std::vector<float> values;
};

Samples make_samples(uint16_t tag, int bits, size_t count, std::mt19937& rng) {
    Samples s;
    const int bytes = (bits + 7) / 8;
    for (size_t i = 0; i < count; ++i) {
        if (tag == 3 && bits == 32) {
            const float v = i == 0 ? 1.5f : i == 1 ? -2.0f : std::uniform_real_distribution<float>(-1, 1)(rng);
            put(s.data, std::bit_cast<uint32_t>(v), 4);
            s.values.push_back(v);
        } else if (tag == 3) {
            const double v = i == 0 ? -1.25 : std::uniform_real_distribution<double>(-1, 1)(rng);
            put(s.data, std::bit_cast<uint64_t>(v), 8);
            s.values.push_back(static_cast<float>(v));
        } else {
            const int64_t top = int64_t{1} << (8 * bytes - 1);
            int64_t v = i == 0 ? -top : i == 1 ? top - 1 : std::uniform_int_distribution<int64_t>(-top, top - 1)(rng);
            v &= ~((int64_t{1} << (8 * bytes - bits)) - 1);
            put(s.data, static_cast<uint64_t>(bytes == 1 ? v + 128 : v), bytes);  // 8-bit WAV is unsigned
            s.values.push_back(static_cast<float>(v) / static_cast<float>(top));
        }
    }
    return s;
}

// Channels averaged as librosa.to_mono does it.
std::vector<float> mono(const std::vector<float>& interleaved, int channels) {
    const auto n = static_cast<size_t>(channels);
    std::vector<float> out(interleaved.size() / n);
    for (size_t i = 0; i < out.size(); ++i) {
        float sum = interleaved[i * n];
        for (size_t c = 1; c < n; ++c) sum += interleaved[i * n + c];
        out[i] = sum / static_cast<float>(channels);
    }
    return out;
}

void check_refused(std::span<const uint8_t> bytes, const std::string& message) {
    CAPTURE(message);
    CHECK_THROWS_WITH_AS(audio::decode(bytes), doctest::Contains(message.c_str()), Error);
}

}  // namespace

TEST_SUITE("audio decode") {
    TEST_CASE("audio decode: WAV of every sample format, channel count and fmt layout reads as libsndfile reads it") {
        std::mt19937 rng(11);
        for (const auto& [tag, bits] : std::vector<std::pair<uint16_t, int>>{
                 {1, 8}, {1, 16}, {1, 24}, {1, 32}, {1, 12}, {1, 20}, {3, 32}, {3, 64}}) {
            for (const int channels : {1, 2, 3, 6}) {
                for (const bool extensible : {false, true}) {
                    CAPTURE(tag);
                    CAPTURE(bits);
                    CAPTURE(channels);
                    CAPTURE(extensible);
                    const Samples s = make_samples(tag, bits, 37 * static_cast<size_t>(channels), rng);
                    const audio::Audio audio = audio::decode(wav({tag, channels, 22050, bits, extensible}, s.data));
                    CHECK(audio.sample_rate == 22050);
                    CHECK(audio.samples == mono(s.values, channels));
                }
            }
        }
    }

    TEST_CASE("audio decode: WAV chunks: unknown ones skipped, fmt after data, a last pad byte missing") {
        const std::vector<uint8_t> data = {0x00, 0x40, 0x00, 0xC0, 0xFF, 0x7F};  // 0.5, -0.5, 32767 / 32768
        const std::vector<float> expected = {0.5f, -0.5f, 32767.0f / 32768.0f};
        const std::vector<uint8_t> odd = {1, 2, 3};
        CHECK(audio::decode(riff({chunk("LIST", odd), fmt_chunk({}), chunk("junk", odd), chunk("data", data),
                                  chunk("id3 ", odd)}))
                  .samples == expected);
        CHECK(audio::decode(riff({chunk("data", data), chunk("bext", odd), fmt_chunk({})})).samples == expected);
        std::vector<uint8_t> unpadded = riff({fmt_chunk({}), chunk("data", data), chunk("LIST", odd)});
        unpadded.pop_back();
        CHECK(audio::decode(unpadded).samples == expected);
        CHECK(audio::decode(wav({}, {})).samples.empty());
    }

    TEST_CASE("audio decode: RF64 and BW64 take the data size from ds64; a stream's data runs to the end of the file") {
        const std::vector<uint8_t> data = {0x00, 0x40, 0x00, 0xC0, 0xFF, 0x7F};
        const std::vector<float> expected = {0.5f, -0.5f, 32767.0f / 32768.0f};
        for (const std::string_view form : {"RF64", "BW64"}) {
            std::vector<uint8_t> ds64;
            put(ds64, 0, 8);            // RIFF size (unused)
            put(ds64, data.size(), 8);  // data size
            put(ds64, 3, 8);            // sample count
            put(ds64, 0, 4);            // table entries
            std::vector<uint8_t> file = riff({chunk("ds64", ds64), fmt_chunk({}), chunk("data", data, 0xFFFFFFFF),
                                              chunk("LIST", std::vector<uint8_t>(4, 0))});
            std::copy(form.begin(), form.end(), file.begin());
            CHECK(audio::decode(file).samples == expected);
        }
        std::vector<uint8_t> stream = riff({fmt_chunk({}), chunk("data", data, 0xFFFFFFFF)});
        stream.push_back(0x12);  // a stream cut inside a frame
        CHECK(audio::decode(stream).samples == expected);
        std::vector<uint8_t> no_ds64 = riff({fmt_chunk({}), chunk("data", data, 0xFFFFFFFF)});
        std::copy_n("RF64", 4, no_ds64.begin());
        check_refused(no_ds64, "the RF64 file has no ds64 chunk");
    }

    TEST_CASE("audio decode: files from libsndfile and ffmpeg read as libsndfile reads them; FLAC twins alike") {
        const nlohmann::json fixture = test::read_audio_json("decode_cases.json");
        for (const auto& c : fixture.at("pcm")) {
            const std::string file = c.at("file");
            CAPTURE(file);
            const audio::Audio audio = audio::decode(test::read_audio_fixture(file));
            CHECK(audio.sample_rate == c.at("sample_rate").get<int>());
            CHECK(audio.samples == test::b64_floats(c.at("samples")));
        }
        const audio::Audio stereo = audio::decode(test::read_audio_fixture("tone_s16_stereo_44100.wav"));
        const audio::Audio stereo_flac = audio::decode(test::read_audio_fixture("tone_s16_stereo_44100.flac"));
        CHECK(stereo_flac.sample_rate == 44100);
        CHECK(stereo_flac.samples == stereo.samples);
        const audio::Audio mono24 = audio::decode(test::read_audio_fixture("tone_s24_mono_16000.wav"));
        CHECK(audio::decode(test::read_audio_fixture("tone_s24_mono_16000.flac")).samples == mono24.samples);
        // ffmpeg writes 24-bit as WAVE_FORMAT_EXTENSIBLE, as RF64 on request, and to a pipe with unknown sizes.
        CHECK(audio::decode(test::read_audio_fixture("tone_s24_mono_16000_rf64.wav")).samples == mono24.samples);
        CHECK(audio::decode(test::read_audio_fixture("tone_s24_mono_16000_stream.wav")).samples == mono24.samples);
    }

    TEST_CASE("audio decode: MP3 as libsndfile (mpg123) decodes it, encoder delay and padding left out") {
        const nlohmann::json fixture = test::read_audio_json("decode_cases.json");
        for (const auto& c : fixture.at("mp3")) {
            const std::string file = c.at("file");
            CAPTURE(file);
            const audio::Audio audio = audio::decode(test::read_audio_fixture(file));
            CHECK(audio.sample_rate == c.at("sample_rate").get<int>());
            const std::vector<float> expected = test::b64_floats(c.at("samples"));
            REQUIRE(audio.samples.size() == expected.size());
            const double diff = test::max_abs_diff(audio.samples, expected);
            CHECK(diff < 1e-5);  // two float decoders; a sample off in time would differ by about 0.1
            MESSAGE(file, ": largest difference from libsndfile: ", diff);
        }
        // Against the source it was encoded from: aligned in time, so the difference is the codec's noise.
        const audio::Audio source = audio::decode(test::read_audio_fixture("tone_s16_stereo_44100.wav"));
        const audio::Audio mp3 = audio::decode(test::read_audio_fixture("tone_s16_stereo_44100.mp3"));
        REQUIRE(mp3.samples.size() == source.samples.size());
        double error = 0.0, energy = 0.0;
        for (size_t i = 0; i < source.samples.size(); ++i) {
            error += std::pow(mp3.samples[i] - source.samples[i], 2);
            energy += std::pow(source.samples[i], 2);
        }
        const double snr_db = 10 * std::log10(energy / error);
        CHECK(snr_db > 15.0);
        MESSAGE("MP3 against its source: ", snr_db, " dB");
    }

    TEST_CASE("audio decode: truncated and malformed files are refused by name") {
        const std::vector<uint8_t> wav_file = test::read_audio_fixture("tone_s16_stereo_44100.wav");
        check_refused(std::span(wav_file).first(wav_file.size() / 2), "the WAV file is truncated (its data chunk");
        check_refused(std::span(wav_file).first(30), "the WAV file is truncated");
        check_refused(std::span(wav_file).first(8), "not a WAV, MP3 or FLAC file");
        const std::vector<uint8_t> mp3 = test::read_audio_fixture("tone_s16_stereo_44100.mp3");
        check_refused(std::span(mp3).first(mp3.size() / 2), "the MP3 file is truncated");
        const std::vector<uint8_t> flac = test::read_audio_fixture("tone_s16_stereo_44100.flac");
        check_refused(std::span(flac).first(flac.size() / 2), "the FLAC file is truncated");
        check_refused(bytes_of("fLaC\0\0\0\x22garbage"sv), "the FLAC file could not be decoded");
        check_refused(bytes_of("ID3\x04\0\0\0\0\0\x02xx"sv), "the MP3 file could not be decoded");
        check_refused({}, "the file is empty");
        check_refused(bytes_of("hello, world"sv), "it starts with bytes 68 65 6c 6c 6f 2c 20 77");
    }

    TEST_CASE("audio decode: WAV refusals name what is wrong") {
        const std::vector<uint8_t> two = {0, 0};
        check_refused(wav({.tag = 6, .bits = 8}, two), "WAV files holding A-law (format tag 0x0006) are not supported");
        check_refused(wav({.tag = 6, .bits = 8}, two), "convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav");
        check_refused(wav({.tag = 2, .bits = 4, .extensible = true}, two), "holding Microsoft ADPCM");
        check_refused(wav({.tag = 0x1234}, two), "holding this format (format tag 0x1234)");
        check_refused(wav({.tag = 3, .bits = 16}, two), "IEEE float WAV samples of 16 bits");
        check_refused(wav({.bits = 40}, two), "integer PCM WAV samples of 40 bits");
        check_refused(wav({.channels = 0}, two), "no channels");
        check_refused(wav({.rate = 0}, two), "a sample rate of 0 Hz");
        check_refused(wav({.rate = 1000000}, two), "a sample rate of 1000000 Hz");
        check_refused(wav({.channels = 2, .block_align = 2}, two), "block align is 2 bytes");
        check_refused(wav({.channels = 2}, two), "holds 2 bytes, not a whole number of 4-byte frames");
        check_refused(riff({chunk("data", two)}), "no fmt chunk");
        check_refused(riff({fmt_chunk({})}), "no data chunk");
        check_refused(riff({chunk("fmt ", two), chunk("data", two)}), "fmt chunk is 2 bytes");
        check_refused(riff({chunk("LIST", two, 1000), fmt_chunk({}), chunk("data", two)}),
                      "its 'LIST' chunk declares 1000 bytes");
    }

    TEST_CASE("audio decode: other containers are refused by name with the command that converts them") {
        for (const auto& [file, name, extension] : std::vector<std::tuple<std::string, std::string, std::string>>{
                 {"refuse_opus.ogg", "Ogg Opus", "ogg"},
                 {"refuse_vorbis.ogg", "Ogg Vorbis", "ogg"},
                 {"refuse_aac.m4a", "M4A", "m4a"},
                 {"refuse_opus.webm", "WebM", "webm"},
                 {"refuse_adts.aac", "AAC (ADTS)", "aac"},
                 {"refuse_pcm.aiff", "AIFF", "aiff"}}) {
            check_refused(test::read_audio_fixture(file), "audio: " + name +
                                                              " files are not supported (WAV, MP3 and FLAC are); "
                                                              "convert with: ffmpeg -i in." +
                                                              extension + " -ar 16000 -ac 1 out.wav");
        }
        check_refused(bytes_of("\0\0\0\x18" "ftypisom\0\0\0\0"sv), "MP4 files are not supported");
        check_refused(bytes_of("\0\0\0\x14" "ftypqt  \0\0\0\0"sv), "QuickTime (MOV) files");
        check_refused(bytes_of("\x1A\x45\xDF\xA3\x9F\x42\x86\x81\x01\x42\x82\x88matroska"sv), "Matroska files");
        check_refused(bytes_of("RIFF\x10\0\0\0AVI LIST"sv), "AVI files");
        check_refused(bytes_of("RIFF\x10\0\0\0WEBPVP8 "sv), "RIFF files of form 'WEBP'");
        check_refused(bytes_of("caff\0\x01\0\0"sv), "CAF files");
        check_refused(bytes_of("OggS\0\x02\0\0\0\0\0\0\0\0"sv), "Ogg files");
    }

    TEST_CASE("audio decode: encode_wav's 16-bit PCM rounds, clips and round-trips; its float keeps every value") {
        audio::Audio clip{8000, {0.0f, 0.5f, -1.0f, 1.0f, 1.5f, -2.0f, 1.0f / 65536.0f, 3.0f / 65536.0f}};
        const std::vector<uint8_t> pcm = audio::encode_wav(clip);
        const std::vector<uint8_t> header = {'R', 'I', 'F', 'F', 52, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                                             16, 0, 0, 0, 1, 0, 1, 0, 0x40, 0x1F, 0, 0, 0x80, 0x3E, 0, 0, 2, 0, 16, 0,
                                             'd', 'a', 't', 'a', 16, 0, 0, 0};
        REQUIRE(pcm.size() == header.size() + 16);
        CHECK(std::vector<uint8_t>(pcm.begin(), pcm.begin() + 44) == header);
        const audio::Audio back = audio::decode(pcm);
        CHECK(back.sample_rate == 8000);
        // 1 and 1.5 clip to 32767; half a step rounds to even (0), a step and a half to 2.
        CHECK(back.samples == std::vector<float>{0.0f, 0.5f, -1.0f, 32767 / 32768.0f, 32767 / 32768.0f, -1.0f, 0.0f,
                                                 2 / 32768.0f});
        CHECK(audio::encode_wav(back) == pcm);  // 16-bit samples are written back bit for bit

        const audio::Audio floats = audio::decode(audio::encode_wav(clip, audio::WavSample::Float32));
        CHECK(floats.samples == clip.samples);
        CHECK_THROWS_WITH_AS(audio::encode_wav({16000, {0.0f, NAN}}), doctest::Contains("sample 1 is not a finite"),
                             Error);
    }

    TEST_CASE("audio decode: content_hash tells clips apart by samples, length and rate") {
        const audio::Audio a{16000, {0.1f, 0.2f, 0.3f}};
        audio::Audio b = a;
        CHECK(audio::content_hash(a) == audio::content_hash(b));
        b.samples[2] = std::nextafter(b.samples[2], 1.0f);
        CHECK(audio::content_hash(a) != audio::content_hash(b));
        CHECK(audio::content_hash(a) != audio::content_hash({8000, a.samples}));
        CHECK(audio::content_hash(a) != audio::content_hash({16000, {0.1f, 0.2f, 0.3f, 0.0f}}));
    }
}
