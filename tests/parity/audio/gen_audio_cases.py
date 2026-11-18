"""Reference outputs for src/audio: torchaudio's resample (the resampler vLLM runs), the log-mel features of
transformers' Qwen3ASRFeatureExtractor as Qwen3ASRProcessor calls it, and small compressed files with what librosa's
decoder (soundfile, libsndfile) reads from them. Writes tests/fixtures/audio/ for tests/unit/audio/.

    python -I tests/parity/audio/gen_audio_cases.py

Needs torch, torchaudio, transformers with models/qwen3_asr (5.20 or later), numpy and soundfile, and an ffmpeg
binary with libmp3lame, libopus, libvorbis and the native AAC encoder on PATH. Every signal is synthetic (tones,
chirps, seeded noise). Floats are stored as base64 of little-endian f32.

Files written:
  resample_cases.json   torchaudio.functional.resample(x, rate, 16000) with its defaults (sinc_interp_hann,
                        lowpass_filter_width 6, rolloff 0.99) for 44.1, 48, 8 and 22.05 kHz, and edge lengths;
                        planted: the same with rolloff 0.98
  mel_cases.json        Qwen3ASRFeatureExtractor()(x, sampling_rate=16000, padding=True, truncation=False,
                        n_window=50): the frames its attention mask marks valid, [128, frames]; for two clips the
                        same math in float64 too; planted: the same STFT with zero padding in place of reflect, and
                        with the symmetric Hann window
  tone_s16_stereo_44100.wav     soundfile, PCM_16;  .flac: soundfile;  .mp3: ffmpeg -c:a libmp3lame -b:a 128k
  tone_s24_mono_16000.wav       soundfile, PCM_24;  .flac: soundfile (24-bit);  _rf64.wav: ffmpeg -rf64 always;
                                _stream.wav: ffmpeg writing to a pipe (RIFF and data sizes 0xFFFFFFFF)
  sweep_mono_16000.mp3          ffmpeg -c:a libmp3lame -b:a 32k (MPEG-2 layer III), from a float32 sweep
  decode_cases.json     soundfile's float32 decode of the two PCM WAVs and of each MP3, channels averaged as
                        librosa.load(mono=True) does
  refuse_*              tiny files of formats the decoder refuses: Ogg Opus, Ogg Vorbis, M4A (AAC), WebM (Opus),
                        ADTS AAC, AIFF (ffmpeg)
"""

import base64
import json
import math
import pathlib
import re
import subprocess
import tempfile

import numpy as np
import soundfile
import torch
import torchaudio
import transformers
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor

OUT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "audio"
TARGET_RATE = 16000


def b64(x):
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode()


def tones(n, rate, seed, freqs, noise=0.05):
    """A sum of tones at random phases plus seeded Gaussian noise, as float32."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    x = np.zeros(n)
    for amplitude, freq in freqs:
        x += amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    x += noise * rng.standard_normal(n)
    return x.astype(np.float32)


def chirp(n, rate, f0, f1, amplitude):
    t = np.arange(n) / rate
    duration = n / rate
    return (amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * duration)))).astype(np.float32)


def resample(x, rate, **kwargs):
    return torchaudio.functional.resample(torch.from_numpy(x), rate, TARGET_RATE, **kwargs).numpy()


def resample_cases():
    cases, planted = [], []
    for rate in (44100, 48000, 8000, 22050):
        orig = rate // math.gcd(rate, TARGET_RATE)
        nyquist = min(rate, TARGET_RATE) / 2
        # Content below the new Nyquist, close to it, and (when downsampling) above it, which the filter removes.
        main = tones(rate // 10, rate, rate, [(0.3, 440.0), (0.2, 0.93 * nyquist), (0.2, 0.45 * rate)])
        cases.append({"from": rate, "to": TARGET_RATE, "input": b64(main), "output": b64(resample(main, rate))})
        if rate == 44100:
            planted.append({"case": len(cases) - 1, "what": "rolloff 0.98",
                            "output": b64(resample(main, rate, rolloff=0.98))})
        rng = np.random.default_rng(rate + 1)
        for n in sorted({1, 2, 5, orig - 1, orig, orig + 1, 2 * orig + 3} - {0}):
            x = rng.uniform(-1, 1, n).astype(np.float32)
            cases.append({"from": rate, "to": TARGET_RATE, "input": b64(x), "output": b64(resample(x, rate))})
    return {"cases": cases, "planted": planted}


def extractor_features(fe, x):
    out = fe(x, sampling_rate=TARGET_RATE, padding=True, truncation=False, return_attention_mask=True,
             n_window=50, return_tensors="np")
    features, mask = out["input_features"][0], out["attention_mask"][0]
    valid = int(mask.sum())
    assert mask[:valid].all() and not mask[valid:].any()
    assert not features[:, valid:].any(), "the extractor pads the time axis with zeros"
    return features[:, :valid], features.shape[1]


def stft_log_mel(fe, x, pad_mode="reflect", periodic=True, dtype=torch.float32):
    """The extractor's STFT and log-mel with one setting changed: the planted controls, and the same math in float64
    (the bank still rounded to float32, as the extractor rounds it), which shows the extractor's own rounding."""
    if x.shape[0] < fe.min_length:
        x = np.pad(x, (0, fe.min_length - x.shape[0]))
    waveform = torch.from_numpy(x).to(dtype)
    window = torch.hann_window(fe.n_fft, periodic=periodic, dtype=dtype)
    stft = torch.stft(waveform, fe.n_fft, fe.hop_length, window=window, pad_mode=pad_mode, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel = torch.from_numpy(fe.mel_filters).to(torch.float32).to(dtype).T @ magnitudes
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).numpy()


def mel_cases():
    fe = Qwen3ASRFeatureExtractor()
    rate = TARGET_RATE
    rng = np.random.default_rng(16000)
    sweep = np.concatenate([chirp(12000, rate, 100.0, 7900.0, 0.5), np.zeros(4000, np.float32)])
    clips = {
        "chirp then silence": sweep,
        "noise": (0.1 * rng.standard_normal(9600)).astype(np.float32),
        "silence": np.zeros(8000, np.float32),
        "below min_length": tones(3000, rate, 3, [(0.4, 1000.0)], noise=0.0),
        "not a multiple of the hop": tones(12345, rate, 4, [(0.3, 300.0), (0.2, 2500.0), (0.1, 7000.0)]),
        "quiet noise": (3e-6 * rng.standard_normal(8800)).astype(np.float32),
        "full scale": np.clip(4 * tones(8160, rate, 5, [(1.0, 220.0)], noise=0.0), -1, 1).astype(np.float32),
    }
    cases, planted = [], []
    for name, x in clips.items():
        features, padded = extractor_features(fe, x)
        reference = stft_log_mel(fe, x)
        assert np.abs(reference - features).max() < 1e-6, "stft_log_mel is the extractor's math"
        cases.append({"name": name, "input": b64(x), "frames": features.shape[1], "padded_frames": padded,
                      "features": b64(features)})
        if name in ("chirp then silence", "full scale"):
            cases[-1]["features_f64"] = b64(stft_log_mel(fe, x, dtype=torch.float64))
        if name == "not a multiple of the hop":
            for what, kwargs in (("zero padding in place of reflect", {"pad_mode": "constant"}),
                                 ("symmetric Hann window", {"periodic": False})):
                planted.append({"case": len(cases) - 1, "what": what, "features": b64(stft_log_mel(fe, x, **kwargs))})
    params = {"sample_rate": fe.sampling_rate, "n_fft": fe.n_fft, "hop_length": fe.hop_length,
              "n_mels": fe.feature_size, "min_length": fe.min_length}
    return {"params": params, "cases": cases, "planted": planted}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def ffmpeg_version():
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    return re.match(r"ffmpeg version (\S+)", first).group(1)


def pcm(x, bits):
    """x as integer samples soundfile writes unchanged: int16 for 16 bits, int32 with the low bits zero for 24 (its
    float conversion rounds differently for WAV and FLAC)."""
    scale = 2 ** (bits - 1)
    q = np.clip(np.round(x.astype(np.float64) * scale), -scale, scale - 1).astype(np.int32)
    return q.astype(np.int16) if bits == 16 else q << (32 - bits)


def mono(path):
    x, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return np.mean(x.T, axis=0), rate  # librosa.to_mono


def compressed_files():
    clean = ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact"]
    rate = 44100
    left = tones(rate // 4, rate, 7, [(0.3, 440.0), (0.2, 3000.0)], noise=0.02)
    right = chirp(rate // 4, rate, 200.0, 12000.0, 0.4)
    stereo = OUT / "tone_s16_stereo_44100.wav"
    soundfile.write(stereo, pcm(np.stack([left, right], axis=1), 16), rate, subtype="PCM_16")
    soundfile.write(OUT / "tone_s16_stereo_44100.flac", pcm(np.stack([left, right], axis=1), 16), rate,
                    subtype="PCM_16")
    ffmpeg("-i", stereo, "-c:a", "libmp3lame", "-b:a", "128k", *clean, OUT / "tone_s16_stereo_44100.mp3")

    x24 = tones(TARGET_RATE // 4, TARGET_RATE, 8, [(0.5, 1000.0)], noise=0.01)
    mono24 = OUT / "tone_s24_mono_16000.wav"
    soundfile.write(mono24, pcm(x24, 24), TARGET_RATE, subtype="PCM_24")
    soundfile.write(OUT / "tone_s24_mono_16000.flac", pcm(x24, 24), TARGET_RATE, subtype="PCM_24")
    ffmpeg("-i", mono24, "-c:a", "pcm_s24le", "-rf64", "always", *clean, OUT / "tone_s24_mono_16000_rf64.wav")
    streamed = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", mono24, "-c:a", "pcm_s24le", *clean,
                               "-f", "wav", "-"], capture_output=True, check=True).stdout
    (OUT / "tone_s24_mono_16000_stream.wav").write_bytes(streamed)

    with tempfile.TemporaryDirectory() as tmp:
        sweep_wav = pathlib.Path(tmp) / "sweep.wav"
        soundfile.write(sweep_wav, chirp(TARGET_RATE // 2, TARGET_RATE, 100.0, 7000.0, 0.5), TARGET_RATE,
                        subtype="FLOAT")
        ffmpeg("-i", sweep_wav, "-c:a", "libmp3lame", "-b:a", "32k", *clean, OUT / "sweep_mono_16000.mp3")
        short = pathlib.Path(tmp) / "short.wav"
        soundfile.write(short, tones(1600, TARGET_RATE, 9, [(0.3, 500.0)], noise=0.0), TARGET_RATE, subtype="PCM_16")
        for name, codec in (("refuse_opus.ogg", ["-c:a", "libopus"]), ("refuse_vorbis.ogg", ["-c:a", "libvorbis"]),
                            ("refuse_aac.m4a", ["-c:a", "aac"]), ("refuse_opus.webm", ["-c:a", "libopus"]),
                            ("refuse_adts.aac", ["-c:a", "aac"]), ("refuse_pcm.aiff", ["-c:a", "pcm_s16be"])):
            ffmpeg("-i", short, *codec, *clean, OUT / name)

    def decodes(*names):
        out = []
        for name in names:
            x, file_rate = mono(OUT / name)
            out.append({"file": name, "sample_rate": file_rate, "samples": b64(x)})
        return out

    return {"ffmpeg": ffmpeg_version(), "libsndfile": soundfile.__libsndfile_version__,
            "pcm": decodes("tone_s16_stereo_44100.wav", "tone_s24_mono_16000.wav"),
            "mp3": decodes("tone_s16_stereo_44100.mp3", "sweep_mono_16000.mp3")}


def write(name, payload):
    path = OUT / name
    path.write_text(json.dumps({
        "generator": "tests/parity/audio/gen_audio_cases.py",
        "torch": torch.__version__, "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
        **payload,
    }, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("resample_cases.json", resample_cases())
    write("mel_cases.json", mel_cases())
    write("decode_cases.json", compressed_files())
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:32} {path.stat().st_size:8} bytes")


if __name__ == "__main__":
    main()
