# 0023: Speech recognition (Qwen3-ASR) and audio input

Status: **accepted** · 2026-11-18

## Context

The first audio model is Qwen/Qwen3-ASR-1.7B (and -0.6B, `model_type` `qwen3_asr`): a Whisper-style audio encoder and
a Qwen3 decoder that answers a chat turn holding the audio with `language <Name><asr_text><transcript>`. It opens a
line of work (Qwen3-Omni's audio in and speech out, image generation), so the way audio enters the engine had to hold
for those too.

The ecosystem serves it with a stack of Python packages: vLLM decodes audio through torchcodec, libsndfile or FFmpeg
and resamples with torchaudio; Qwen's own `qwen-asr` package pulls in librosa, soundfile, transformers 4.57 and, for
its forced aligner, Japanese and Korean word segmenters (nagisa, soynlp). Read closely, almost all of that is glue
(prompt layout, cutting long audio, parsing the reply), a reference copy of the model, or signal math that is small
and deterministic. Only the decoding of container formats really needs outside code.

## Decision

- **Three layers per media item: container decoding, signal front end, network.** The front end (resampling, log-mel)
  is our own code in `src/audio/`, held against the Python libraries by committed fixtures (those libraries are
  test-time only). Decoding is WAV by hand, MP3 and FLAC through the public-domain single headers of dr_libs (as
  stb_image serves images); Ogg, Opus, AAC, M4A and video containers are refused by name with an ffmpeg command line
  to convert them. Resampling reproduces torchaudio's (vLLM's default); the reference package's soxr differs slightly,
  and 16 kHz input, the common case, is not resampled at all.
- **Media by modality.** A `MediaInput` names its modality and holds encoder runs, each with the rows of its output that
  belong to the item; `VisionEncoder` and `AudioEncoder` share a `MediaEncoder` interface; the engine keeps an encoder
  session per modality and maps each soft token to the next item of its kind.
- **Qwen3-ASR's encoder runs one window at a time.** Its convolutions read chunks of 100 mel frames, its positions
  restart in each chunk and its attention never crosses a window of 8 chunks (8 s, 104 rows), so a window is a static,
  stateless graph: a clip's last chunk is zero-padded (as the reference pads it) and the padded rows' keys are masked.
  The result is exact, not an approximation, and needs no new attention mode.
- **One graph op: Conv2d** (stride, padding per side, dilation, groups; im2col and a GEMM on SYCL), general enough for
  the codecs and VAEs of the next models.
- **A `Transcription` interface on Model** says what is model-specific: how long a piece of audio may be, the text that
  forces a language (`language English<asr_text>`), and how a reply becomes a transcript (held back while streaming
  until the header is past). The service does the rest for any such model: decode, cut long audio at quiet points
  (Qwen's rule: the quietest 100 ms within 5 s of 1,200 s), render the turn through the chat template with an empty
  system turn as the reference package does, generate, and join the pieces.
- **A piece's reply stops at 4,096 tokens, and repeated text is cleaned**, both as the reference package does
  (`max_new_tokens`, `detect_and_fix_repetitions`: a character repeated more than 20 times, or a pattern of up to 20
  characters repeated 20 times, kept once). Greedy decoding can fall into a loop; without the limit a reply ran to the
  end of the context.
- **`POST /v1/audio/transcriptions`**, OpenAI's form: `file`, `model`, `language` (ISO 639-1 or a name), `prompt` (the
  system turn), `temperature`, `response_format` json or text, `stream` (server-sent `transcript.text.delta` and
  `transcript.text.done` events). Timestamps (`verbose_json`, `srt`, `vtt`, `timestamp_granularities[]`) need the forced
  aligner and are refused by name, as are translations (the model only transcribes). Chat takes `input_audio` parts
  too, its reply in the model's own format.
- **Both checkpoint layouts**: the released one (weights under `thinker.*`, a `thinker_config`) and transformers'
  converted one (`model.*`, flat configs). The tokenizer ships as `vocab.json` and `merges.txt`, read as
  transformers' Qwen2 converter builds it; the template as `chat_template.json`.
- **The Qwen3 decoder** (`models/qwen3/`) is a builder used by Qwen3-ASR; plain Qwen3 language models are not served
  yet, because their tool calls are JSON, which no output parser here reads.

## What was measured

Against transformers' Qwen3-ASR in fp32 on the CPU (`tests/parity/model/hf_audio_reference.py`), per clip the mel
features, the encoder's rows, the prompt, the last logits and the greedy text (`tests/unit/model/audio_parity.cpp`):

| Checkpoint | Backend | Mel | Encoder rows | Logits | Greedy tokens equal |
|---|---|---|---|---|---|
| tiny, both layouts | cpu-ref f32 | 1.4e-5 | 3.7e-8 | cosine 1, 1.2e-6 | 32 of 32, five clips |
| tiny, both layouts | Arc Pro B60 f32 / f16 | 1.4e-5 | cosine 1 (6.9e-5 at f16) | cosine 1 | 32 of 32 |
| Qwen3-ASR-1.7B | Arc Pro B60 f32 | (< 2e-3) | cosine 1, 1.7e-6 | cosine 1, 1.8e-4 | all: 26, 90, 22 and 106 tokens |
| Qwen3-ASR-1.7B | Arc Pro B60 f16 | (< 2e-3) | cosine 1, 3.7e-4 | cosine 0.999998 | all |

The real clips are LibriSpeech's (one of 5.9 s, two of about 6 to 25 s, one of 33 s across five encoder windows); the
synthetic ones include a clip below the 0.5 s minimum, one of two windows and one at 44.1 kHz.

On the Arc Pro B60 through the API (model_recipes.md has the table): word error rate on LibriSpeech's 73 clips of
`hf-internal-testing/librispeech_asr_dummy` 3.34 % at f16, 3.51 % at q8, 4.11 % at q4; a 600 s clip in 28.4 s (f16)
and 19.7 s (q8), its first streamed text after about 2.8 s. Decoding is most of it (17.1 s of q8's 19.7 s); before
the first token, the encoder's 75 windows (0.87 s, one run each), prefill (1.27 s) and the CPU's log-mel and decoding
(about 0.5 s). q4 transcribed the first 1,418 words of that clip as q8 did, then missed the end of the speech and
began the clip again until the 4,096-token limit: q8 is the format for long audio.
