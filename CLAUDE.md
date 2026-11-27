Always build code cleanly, modularly, and with proper architecture and separation of concerns. Never simply bolt on a feature because I asked for it. If you can’t find an easy way to build it naturally, and you'd recommend a refactor to make the whole thing cleanly, stop your work ahead of time and say so.

# CLAUDE.md: how this repo is laid out

The orientation guide for anyone (person or AI) about to change this code. Read this first, then
[DESIGN.md](DESIGN.md) for the reasoning and [TODO.md](TODO.md) for what is planned. Keep this file current: a change
that moves, adds or renames a file updates the map below in the same commit.

## 1. What xpu-serve is

A C++20 LLM inference engine and OpenAI-compatible server for Intel GPUs. One repo, one binary (`xpu-serve`):
model code, engine, tokenizer, chat templates and HTTP server all live here, and the math runs through a backend
(SYCL with our own kernels on Intel GPUs, plus a CPU reference used as the test oracle).

The rule that shapes everything: **one feature = one PR in this repo.** Supporting a model or a feature must never
need changes in an exporter, a separate pipeline library and a separate server.

How that works:

- **Model code is C++ in this repo** (`src/models/<arch>/`). It reads the Hugging Face `config.json`, and builds a
  backend-neutral **step graph** out of ~20 ops (matmul, rms_norm, rope, attention, ...).
- **Weights are read from Hugging Face safetensors at load time** (memory-mapped): the original checkpoint, an FP8
  release of it, or a q8 / q4 pack written by `convert` (DESIGN.md section 6), or a pack `convert` imported from a
  GGUF file (its tensors in GGUF's block formats, byte for byte). There is no exported IR.
- **A backend compiles the step graph** for a device and runs it. Backends know nothing about models.

Status (see TODO.md): Gemma 4 E2B/E4B (text and images), Gemma 4 26B-A4B (text), the Qwen3.5 family (Qwen3.5,
Qwen3.6, Qwen3.8; text and images through Qwen3-VL's vision tower, docs/decisions/0032; a hybrid of Gated DeltaNet
linear attention and full attention, dense or mixture of experts;
and SenseNova Loop, a Qwen3.5 decoder that runs its middle layers twice inside hyper-connections: docs/decisions/0022)
and LFM3 (text; a hybrid of gated short convolutions and full or sliding-window attention), Qwen3-Omni's thinker (text,
audio and images in, text out: Qwen's audio encoder, Qwen3-VL's vision tower and a Qwen3 mixture-of-experts decoder
with positions on three axes; docs/decisions/0025), Inkling 2 Mini (text and images; a decoder without rotary
positions whose middle layers run three times: attention with a bias by distance and a sink, short convolutions,
block residuals; its own block drafter, DSpark, with `--spec dspark`; docs/decisions/0026),
and decision models behind the System One API (`POST /v1/systemone`: Clef-Flash, a Qwen3.5 decoder with a joint
schema head; pplx-decider-flash, one whose full-attention layers see the whole prompt, with a read-out per question;
OpenJev, a language model read at the option letters with `--readout`; docs/decisions/0015; d1, LFM3-VL's text
decoder read at the tokens of each option's code, up to 16 questions packed after one state: docs/decisions/0019;
Laya and Julia-1, ModernBERT encoders read at a mask token per option, a sequence per question:
docs/decisions/0018),
served end to end on the SYCL backend (Arc Pro B60), matching Hugging Face fp32 logits and greedy output; q8 / q4 packs
from `convert`; speculative decoding with the model's drafter (`--spec mtp`: Qwen's MTP head, a Gemma 4 assistant
with `--drafter`; `--spec dflash|dspine`: a block drafter checkpoint for Qwen3.8-27B or Ornith-1.5-35B-A3B with
`--drafter`; `--spec dspark`: Inkling 2's drafter, in its checkpoint; docs/design-spec-decode.md). Several requests in flight on sequences of one session (`--sequences`; docs/decisions/0027),
their decode steps run as one (docs/decisions/0028). Embedding models behind `POST /v1/embeddings` (all-MiniLM-L6-v2
and bge-small-en-v1.5: BERT; nomic-embed-text-v1.5: NomicBERT; docs/decisions/0017), checked against
sentence-transformers on cpu-ref. Speech recognition behind `POST /v1/audio/transcriptions` (Qwen3-ASR: an audio
encoder run by windows of 8 s and a Qwen3 decoder; audio in chat as `input_audio`; docs/decisions/0023) and OpenAI's
realtime API (`GET /v1/realtime`, a WebSocket: audio streamed in, turns found by a voice activity detector or committed
by the client, each turn's text streamed back; docs/decisions/0024).

## 2. Build, run, test

```bash
source /opt/intel/oneapi/setvars.sh                       # oneAPI 2026.2: icpx + oneMKL (SYCL backend)
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
build/tests/xs_tests                                      # every unit test (doctest); model tests skip without a model
cmake --install build --prefix ~/.local                   # optional: the binary as ~/.local/bin/xpu-serve
build/src/xpu-serve info                                  # backends and devices
build/src/xpu-serve run   <hf-model-dir> --device GPU --prompt "Hi"
build/src/xpu-serve run   <hf-model-dir> --device GPU --image photo.jpg --prompt "Describe this image."
build/src/xpu-serve serve <hf-model-dir> --device GPU     # http://127.0.0.1:8080/v1
build/src/xpu-serve serve -hf Qwen/Qwen3.5-4B --device GPU   # the same, downloaded into the Hugging Face cache first
build/src/xpu-serve serve <file.gguf> --device GPU        # a GGUF file, imported once into a pack in xpu-serve's cache
build/src/xpu-serve bench <hf-model-dir> --device GPU --prompt-tokens 1024 --gen-tokens 128
build/src/xpu-serve run   <decision-model-dir> --device GPU --request request.json   # a System One request's answers
build/src/xpu-serve run   <embedding-model-dir> --device GPU --prompt "text"        # its embedding, a JSON array
```

- OpenSSL 3 (`libssl-dev`) is needed: TLS and SHA-256 of the Hub downloads (`-hf`). icpx turns the SYCL backend on
  (`XS_WITH_SYCL`); `-DXS_SYCL_TARGETS=intel_gpu_bmg_g21` compiles its kernels ahead
  of time for the Arc Pro B60. Any other C++20 compiler builds only the CPU reference backend (the test oracle).
- Useful flags: `--backend sycl|cpu-ref` (default: sycl when built), `--device GPU|GPU.0|GPU.1`, `--precision f16|f32`,
  `--kv-dtype auto|i8|turbo4|turbo3|turbo2` (or `<keys>:<values>`, e.g. `i8:turbo4`), `--ctx N`, `--prefill-chunk N`,
  `--sequences N` (requests in flight at once, docs/decisions/0027),
  `--backend-opt attention=auto|rows|gemm|flash`, `--backend-opt gemv=auto|vector|xmx`, `--backend-opt few_rows=N`,
  `--backend-opt tiled_rows=N`,
  `--backend-opt block=auto|f16`, `--backend-opt kv_chunk=N`,
  `--spec mtp|dflash|dspine|dspark --spec-k N`,
  `--readout letters|openjev` (a language model served as a decision model), `--set-size N` (d1: questions packed
  after one state, 1 for each alone),
  `--drafter <dir | file.gguf>`, `--mtp <file.gguf>` and `--mmproj <file.gguf>` (a GGUF model's MTP head or drafter
  and vision tower from files the user names, else the ones beside it), `--lora <adapter.gguf>` (a LoRA adapter in
  llama.cpp's form; `--lora-scale X`),
  `--draft-shortlist N`, `--no-vision`, `--image-max-pixels N` (Qwen3-VL's processor),
  `--log-level debug`. `xpu-serve <command> --help` lists all of them;
  `src/cli/main.cpp` is the source of truth.
- Run one test case: `build/tests/xs_tests -tc='model matches*'`. Environment for the tests that need files:
  `XS_TEST_MODEL_DIR` (a Hugging Face model directory), `XS_PARITY_BACKEND` (`cpu-ref` or `sycl`),
  `XS_PARITY_DEVICE`, `XS_PARITY_PRECISION`, `XS_PARITY_FIXTURES`, `XS_PARITY_KV_DTYPE`, `XS_TEST_DRAFTER_DIR` (a
  drafter checkpoint beside the model); `XS_KV_QUALITY_TEXT` (a text file: what each KV cache format changes in the
  model's predictions over it, `kv_format_quality.cpp`); `XS_QWEN35_DIR` (a Qwen3.5 checkpoint, for tokenizer parity and what a grammar's mask costs a step over its vocabulary; with `XS_GGUF_FILE`, a qwen35 GGUF of it, for the GGUF import's real-file checks); `XS_QWEN35MOE_DIR` (with `XS_GGUF_FILE`, a qwen35moe GGUF: the small files of the checkpoint it was converted from, its config.json, and tokenizer.json, tokenizer_config.json and chat_template.jinja when there); `XS_GEMMA4_DIR` (with `XS_GGUF_FILE`, a gemma4 GGUF: the small files of the checkpoint it was converted from); `XS_GGUF_DRAFTER_FILE` (a gemma4-assistant GGUF) with `XS_GEMMA4_ASSISTANT_DIR` (the assistant checkpoint's config.json); `XS_CLEF_DIR` (with `XS_GGUF_FILE`, a clef GGUF: the Clef checkpoint it was converted from, its config.json and joint_head_config.json, and joint_head.safetensors and tokenizer.json when there); `XS_LFM3_DIR`
  (an LFM3 checkpoint, for its tokenizer and chat template, which are not kept in this repository);
  `XS_INKLING2_DIR` (a thinkingmachines/Inkling-2-Mini checkpoint, or only its tokenizer.json, for tokenizer parity;
  with `XS_GGUF_FILE`, an inkling2 GGUF, its config.json too, and `XS_GGUF_DRAFTER_FILE` for the DSpark file);
  `XS_QWEN3_ASR_DIR` (a Qwen/Qwen3-ASR-1.7B checkpoint, or only its vocab.json, merges.txt and tokenizer_config.json:
  a tokenizer read from slow files, for tokenizer parity; the vocabulary and merges are not kept in this repository);
  `XS_WORDPIECE_DIRS` (WordPiece checkpoints for tokenizer parity, separated by `:`, each directory named as its
  repository: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5`); `XS_LAYA_DIR` and `XS_JULIA_DIR`
  (the convaiinnovations/laya and SupersonicLabs/Julia-1 checkpoints, for the tokenizers in their `tokenizer/`
  folders: Laya's three, Julia-1's one; with `XS_GGUF_FILE`, a modern-bert GGUF of one of them, for the GGUF import's
  real-file checks);
  `XS_TEST_DECISION_DIR` (a decision model) with `XS_DECISION_FIXTURES` (its reference answers; for d1, at the set
  size they record, which the test reads the model at) and `XS_DECISION_CHUNK` (tokens a step feeds);
  `XS_TEST_EMBEDDING_DIR` (an embedding model) with `XS_EMBEDDING_FIXTURES` (its reference embeddings) and
  `XS_EMBEDDING_TEXTS=1` (also tokenize the texts).
  `tests/parity/model/make_tiny.py` writes a tiny random checkpoint of an architecture (real layer pattern and tensor
  names; with the checkpoint's own modeling code for one transformers does not ship, like LFM3 and SenseNova Loop,
  whose code pins transformers 5.18; Inkling 2's with its experts stored in NVFP4 and MXFP8 as the release has
  them), or of a block
  drafter for one (`--drafter`), or an encoder in a sentence-transformers checkpoint (`--arch bert|nomic_bert`), to run
  the reference scripts and the parity tests on in seconds.
- `XS_SYCL_PROFILE=1` logs what each kind of op costs the SYCL backend per step. A step of up to 8 tokens (decode, a
  speculative verify) is profiled once it has run: every launch of a kind is recorded as one command graph and run
  again several times back to back, with no host work between two kernels, so a kind's time is the device's and the
  kinds add up to the step's (the plan of such a step gives every value memory of its own, so the ops can run again).
  In a longer step each op is timed on the host with a wait after it. `XS_SYCL_SYNC=1` waits after every op and names
  the op a device error comes from (`XS_SYCL_TRACE=1` also prints each op before it runs).
- The GPU suites ("device ops", "device attention", "device session", "device vs cpu-ref", ...) run when
  `XS_DEVICE_TEST_DEVICE=GPU.0` is set (`XS_DEVICE_TEST_PRECISION=f32` for f32; `XS_DEVICE_BENCH=1` adds the decode
  timing case).

## 3. Repository map

```text
xpu-serve/
├── CMakeLists.txt          options XS_WITH_SYCL, XS_SYCL_TARGETS, XS_BUILD_TESTS, XS_WERROR; C++20; warnings target;
│                           `cmake --install` puts the one binary in <prefix>/bin (src/CMakeLists.txt)
├── README.md               user-facing intro: what is supported, the models, build, quick start, the APIs
├── model_recipes.md        per model: the reference command, the optimized one, and every measured number
├── CLAUDE.md               this file
├── DESIGN.md               the design (why), roadmap, open questions
├── TODO.md                 phase checklists; [x] = done and tested
├── docs/decisions/         one short record per decision (NNNN-title.md)
├── docs/gguf_models.md     what was measured with GGUF files: every variant of seven models that fits a 24 GB card
├── third_party/            vendored header-only deps (see third_party/README.md)
├── src/                    everything that ships (section 4)
└── tests/                  unit tests, parity scripts, fixtures (section 7)
```

## 4. Source layout (`src/`)

Each directory is a static library whose `.cpp` files are globbed (`src/CMakeLists.txt`), so adding a file never
needs a CMake edit. Headers are included as `"area/file.h"`. Everything is in namespace `xs` (sub-namespaces per
area).

| Directory | Library | What lives there |
|---|---|---|
| `util/` | xs_core | `error.h` (`xs::Error`, `fail(...)`, `XS_CHECK`, `str_cat`), `log`, `timer.h`, `mmap_file`, `half` (f16/bf16 conversions), `base64`, `hash.h` (`ContentHash`: the content hashes media caches key on) |
| `tensor/` | xs_core | `DType` (incl. `Q8` / `Q4` pack weights, `F8`, an FP8 checkpoint's weight, `F4`, an NVFP4 checkpoint's, and `Block`, a weight in one of GGUF's block formats kept as the file stores it), `block_format` (those formats: names, bytes and weights per block; ggml's own, and NVFP4, MXFP4 and Q2_0 of Unsloth's llama.cpp builds), `TensorView` (non-owning, e.g. into an mmap; `quant` points at a pack weight's scales and biases, `fp8` at an FP8 weight's runs of rows, `fp4` at an NVFP4 weight's, `block` names a Block weight's format), `HostTensor` (owning), `TensorMap` (name -> HostTensor) |
| `io/` | xs_core | `safetensors`: `WeightStore` interface, `SafetensorsWeights::open_dir` (single file or sharded index, mmapped; a pack's quantized weights appear as one Q8 / Q4 view each, or one Block view over their U8 rows of blocks, an FP8 checkpoint's `F8_E4M3` matrices as one F8 view each over codes and their scales, one per block or one per row; `join_rows` shows tensors stored in pieces as one weight: F8 ones without a copy, float ones and ones of one block format copied once; `open_file` for a head kept in a file of its own; a model's own layouts of float codes paired when its loader asks: `pair_mxfp8`, codes beside a power of two per 32 columns, and `pair_nvfp4`, E2M1 codes beside E4M3 scales and a factor per expert); `safetensors_writer` (streamed writing); `joined_weights` (a drafter checkpoint's weights under a prefix beside the target's); `host_weights` (tensors held in host memory by name: an adapter's); `gguf` (a GGUF file read in place: metadata of every type, the tensor table, each tensor's bytes; a model split into parts by llama.cpp's gguf-split read as one file from its first part, the others found by name, each tensor knowing its part: `split_name`, `split_path`, `model_name`; refusals by name) |
| `quant/` | xs_core | round-to-nearest quantization into the pack formats (DESIGN.md 6.3) and dequantization; a tensor `[..., K]` is a matrix of `numel / K` rows (expert banks `[E, N, K]` included); `read_rows`, a float weight's rows whatever it is stored as (what backends and `convert` read through); `fp8` (the E4M3 codes of FP8 checkpoints and their scales: one per block of 128 rows and columns, or one per row; NVFP4's E2M1 codes with an E4M3 scale per 16 columns); `block_decode` (GGUF block formats decoded on the host, bit for bit as ggml decodes them) and `block_tables.h` (the codebooks of their i-quant formats, ggml's own: plain constexpr data that device kernels include too); `sign_sketch`, the one-bit copy of a weight a drafter's shortlist scores with; `kv_format` (the formats a KV cache is stored in, their names and sizes, and `round_trip_kv`: the value a format stores for a row, which every backend's cache must hold) and `kv_turbo` (the TurboQuant formats turbo4 / turbo3 / turbo2: groups of 128 values rotated, coded against fixed centroids, an f16 norm per group; constants device kernels include too, and the arithmetic host and device share so they store the same bits) |
| `convert/` | xs_core | `convert_model`: checkpoint (language model, decision model, policy, or a drafter beside its model with `--target`) -> pack; the model's own graphs decide which weights to quantize (`plan_graph_formats`: bandwidth-bound matmuls, or all with `--scope all`); `gguf/`: `import_gguf`, a GGUF file -> pack (`gguf_import.cpp`: block format tensors copied byte for byte, float tensors as stored or through a transform, streamed; config.json, tokenizer, chat template, manifest), `architecture.h` (what an architecture's importer provides: config.json from the metadata and any other JSON file of the pack, the tensor name map, float transforms, the named chat templates it has no use for, and a model's MTP head from a file of its own, `--mtp`: `plan_with_mtp`, refused by default), `layout` (the tensor layout an architecture describes and the checks every file goes through: each tensor of the file and the pack tensors it holds, fused ones cut into runs of rows, float ones the converter transposed or flipped put back (`FloatReorder`), ones the converter computed from the config checked and left out, a shape's leading dims of 1 not counted), one file per architecture (`qwen35.cpp` for `qwen35` and `qwen35moe`, `gemma4.cpp` for `gemma4` and `gemma4-assistant`, `clef.cpp`, `modern_bert.cpp` for the decision models on `modern-bert`: a pack in the original checkpoint's layout, `ImportPlan::Kind::Decision`; `inkling2.cpp` for `inkling2` and its drafter's `dflash` file), `qwen35.h` (the Qwen3.5 decoder's reading under any metadata namespace and with either MLP in its layers, dense or a mixture of experts: `qwen35`'s, `qwen35moe`'s, and `clef`'s, which adds a decision head), `tokenizer` (tokenizer.json and tokenizer_config.json from the file's vocabulary and merges, the normalizer, pre-tokenizer and BPE settings from a table keyed by `tokenizer.ggml.model` and, for gpt2, `tokenizer.ggml.pre` (qwen35's: `tokenizer/tokenizer_json`'s Qwen byte-level BPE; inkling2's: the same over o200k_base's regex, merges ignored for a word in the vocabulary), with the type exceptions and end-of-turn tokens of each, and a BERT-style post-processor over the file's [CLS] and [SEP] where the original has one); `gguf/adapter` (a LoRA adapter file: its A and B under the checkpoint's names, an architecture's `adapter_target`) |
| `image/` | xs_core | `Image` (8-bit RGB), `decode` (stb_image), `resize_bicubic` (bit-exact with torchvision), `content_hash` |
| `audio/` | xs_core | `Audio` (mono f32), `decode` (sniffs the bytes: WAV read by hand, RIFF / RF64 / BW64, integer PCM and float, any channels averaged as librosa does; MP3 and FLAC through dr_libs, `compressed.cpp`; other containers refused by name with an ffmpeg command; `codecs.h` is the decoders behind it), `encode_wav`, `content_hash`; `resample` (torchaudio's sinc_interp_hann, vLLM's resampler); `log_mel_spectrogram` (`MelParams`: transformers' Qwen3ASRFeatureExtractor, Whisper's math; frames in parts on threads of their own) over `fft` (`RealFft`: any length, mixed radix); `split_at_quiet` (long audio cut at its quietest 100 ms near a limit, the Qwen3-ASR package's rule); `vad` (`EnergyVad`: where turns of speech start and stop in a stream, by frame energy against a tracked noise floor: a realtime session's server VAD) |
| `graph/` | xs_core | the step graph IR and builder (`graph.h` holds the op spec); `elementwise_groups` (trees of elementwise ops a backend runs as one pass); `low_rank` (a LoRA adapter's changes added where a program reads the weights they change) |
| `models/` | xs_core | `model.h` (the `Model` interface: language models, with a batch graph for decoding several requests at once (`batch_graph`, `LoadOptions::batch`); their media encoders by modality, `MediaEncoder` with `VisionEncoder` and `AudioEncoder`; `Transcription`, what a speech recognition model adds), `policy.h` (the `Policy` interface: robot policies, one static program per call shape: camera subset, step count, padded prompt length), `decision.h` (the `DecisionModel` interface: decision models, which lay out a request's prompt and name the steps to run), `embedding.h` (the `EmbeddingModel` interface: embedding models, one stateless encoder program and a spec: dimension, longest input, pooling, normalize), `low_rank_model` (a model whose programs read a LoRA adapter's changes beside its weights), `registry.cpp` (model_type / LeRobot type / a head beside the decoder -> code, `model_kind`), `config_util` (checked JSON readers), `common/blocks` (the shared building blocks: gated MLPs, mixture of experts (a bias that chooses the experts without weighing them, gate and up rows fused in halves or alternating, a hidden state the model computes), RMSNorm variants, RoPE frequencies, attention, a drafter's head over a shortlist), `common/py_text` (a JSON value as Python's `json.dumps` and `str` write it: what reference code puts in a prompt), `common/sentence_transformers` (the pipeline a sentence-transformers checkpoint lists: pooling mode, Normalize, the longest input; the rest refused by name), `common/encoder` (BERT's post-norm layer and the `EmbeddingModel` around an encoder graph), `common/torch_nn` (torch.nn's Linear and LayerNorm, with or without a bias, over a checkpoint's state-dict names), `common/expert_banks` (a mixture of experts stored expert by expert shown as the banks the model code reads), one folder per architecture (`pi05/`: π0.5; `clef/` and `pplx_decider/`: decision models on the Qwen3.5 decoder; `d1/`: a decision model on the LFM3 decoder; `laya/`: Laya and Julia-1, decision models on ModernBERT; `bert/` and `nomic_bert/`: encoders of embedding models; `modern_bert/`: ModernBERT, pre-norm layers with GeGLU and a local window in two of every three, the encoder of Laya and Julia-1), `letter_readout/` (any language model as a decision model: OpenJev), `drafters/dflash` (block drafters that serve any target they were trained for: DFlash, DFlash 2, DSpine; each layer's attention its own kind, a block cut to the drafts a round asks for), `drafters/dspark` (Inkling 2's block drafter: each layer's own mix of the target's states as its context, a block that sees the window before its start, a recurrent head that picks the drafts one after another; a pack of mainline llama.cpp's drafter file holds its mixes a row per reader, its mask a copy per draft and its head folded through the output layer) |
| `engine/` | xs_core | `Engine` (requests as generations advanced a step at a time, each on a sequence of the session (`SessionOptions::sequences`, the free one sharing the longest cached prefix), the decode steps of all that decode as one step of the model's batch graph (`advance_decoding`, `EngineOptions::batch`; speculating, their rounds' verifies as one step of the verify graph, sharing `EngineOptions::verify_rows`), prefix reuse (for a model with conv or recurrent state, from a snapshot of the sequence's state, `GenerateRequest::snapshot_at`; another sequence's prefix taken as a copy, `take_prefix`), chunked prefill, media encoding (an encoder session per modality; an item may take several runs, e.g. a clip's windows), each token's rotary position on three axes (a media item's own offsets, text one past what came before: Qwen's multimodal RoPE), decode loop, speculative rounds: draft, verify, accept, roll back), `Sampler` (and rejection sampling of a draft), request/result types (`MediaInput`, `TranscriptionRequest` / `TranscriptionResult`), `media` (placeholder expansion; `rotary_positions`, a prompt's positions on three axes), `service.h` (the `InferenceService` interface the server uses), `constraint.h` (`TokenConstraint`: what a request's output follows token by token, which the engine drives without knowing what it is: a constrained request reads its rows' logits, each masked as the tokens before it leave the constraint, the stop tokens kept only where it is complete, a round's drafts cut at the first it refuses; `OutputConstraint`, what a request carries to start one, `GenerateRequest::constraint`), `spec/` (the drafter side of speculative decoding: `drafter.h` is what the engine drives and the session's program order; `chain_drafter` feeds a token-a-run drafter the target's hidden states and chains its drafts; `block_drafter` runs a block drafter's program, hands it the rows of the target's last token it asks for, and follows its candidates) |
| `backend/` | xs_backends | `backend.h` (Backend/Session interfaces), `session.cpp` (a step of several sequences' segments as each segment run alone: the definition, cpu-ref's oracle; in xs_core), `registry.cpp`, `cpu_ref/`, `sycl/` |
| `tokenizer/` | xs_tokenizer | Hugging Face `tokenizer.json`, natively: `model` (the model interface and its vocabulary: `bpe_model`, `wordpiece_model`), normalizers (Replace, NFC, BertNormalizer, from generated Unicode tables), pre-tokenizers (Split with hand-written GPT-style regexes, `regex_split.h` naming the patterns: Qwen's, Llama 3's and o200k_base's, which tells letters apart by case, ByteLevel with or without the GPT-2 regex, Metaspace, BertPreTokenizer; pieces may be rewritten, so they are owned strings), `post_processor` (TemplateProcessing's special tokens around one sequence: `Tokenizer::with_special_tokens`), byte fallback, byte-level or WordPiece decoding, added tokens, streaming decoder, a token's own bytes (`token_bytes`: what it adds in the middle of a text, as they are, for a grammar to read; `is_added`: a marker, not a piece); tokenizer.json's truncation and padding are not applied. `tokenizer_json` (a tokenizer.json built in memory: Qwen's byte-level BPE over a Split regex, as transformers' Qwen2Converter builds it; the GGUF importer's qwen35 tokenizer too), `slow_files` (a checkpoint without tokenizer.json: vocab.json, merges.txt and tokenizer_config.json read as the tokenizer.json transformers builds from them, its added tokens and special tokens included; Qwen2Tokenizer only, any other tokenizer_class refused by name; `Tokenizer::load` takes either form) |
| `chat/` | xs_chat | `ChatTemplate` (renders the model's Jinja template with vendored jinja.cpp, in the request's key order; the template from chat_template.json, chat_template.jinja or tokenizer_config.json, the first there in transformers' order), output parsers per dialect: `gemma4`, `qwen3_5` (thinking, tool calls as XML; LFM3 writes the same format) and `qwen3` (the same, tool calls as JSON: Qwen3, Qwen3-Omni), one parser in `qwen3_parser.cpp`, `inkling2` (messages of a kind each: thinking, text, a tool call as JSON), `plain`; `direct_answer`, how a prompt asks each dialect for an answer at once; `answer_start`, where an answer held to a format (`response_format`) starts: after the reasoning a prompt leaves open (`OutputParser::reasoning_end`, the marker that closes it), then after the dialect's opening |
| `grammar/` | xs_grammar | Constrained decoding's grammars (docs/decisions/0033), knowing nothing of the engine or the server: `grammar` (a context-free grammar over bytes: rules of alternatives of byte sets and rules, written as expressions (`Expr`: literals, sets, sequences, alternatives, repeats) and lowered into rules; `finish` drops the alternatives that can never end and refuses left recursion, `GrammarError` naming the rule), `json_schema` (any JSON object, and the values a JSON Schema describes in OpenAI's strict subset, every other keyword refused by its path; whitespace bounded, strings valid UTF-8), `vocabulary` (a model's tokens as bytes, markers apart, and the trie of their bytes, built once per model; `TokenSet`), `matcher` (`Automaton`: a pushdown matcher's configurations interned as states, each state's step by a byte computed once; the tokens a state allows from one walk of the trie, a subtree skipped at its first dead byte, a long walk's tokens kept for the state to come again; `Matcher`, a position in it that copies cheaply) |
| `server/` | xs_server | HTTP (cpp-httplib), OpenAI request parsing and response formatting, Anthropic's Messages API (`anthropic_request`: its blocks, tools and thinking read as the chat request a template renders; `anthropic_format`: its message and stream of events); `response_format.h` (what each endpoint's format writes of a generation: SSE events or one body), `request_fields.h` (the field readers both parsers share), `grammar_constraint` (OpenAI's `response_format` json_object and json_schema as the engine's output constraint: the model's tokens as a grammar reads them, built at the first request with a format; a generation free while the prompt's reasoning goes on, then the dialect's opening, then held to the grammar; docs/decisions/0033), stop strings, streaming (`channel.h`: output handed from the generating thread to the HTTP thread); the transcription API (`transcription_handlers`: multipart uploads, json / text / server-sent events); the realtime API (`realtime_session`: one transcription session's state, transport aside: OpenAI's GA events in and out, the audio buffer, the server VAD, committed turns transcribed in order on a thread of its own; the WebSocket route and the API key as a subprotocol in `server.cpp`); the policy API; the System One API (`decision_handlers`); the embeddings API (`embedding_handlers`) |
| `service/` | xs_service | `EngineService`: loads tokenizer + template + model, compiles on a backend, owns the engine thread |
| `policy/` | xs_policy | `PolicyRunner`: a policy, its tokenizer and one compiled program per call shape; warm-up; per-call timings (`xpu-serve bench <policy-dir>`) |
| `decision/` | xs_decision | `DecisionRunner`: a decision model, its tokenizer and chat template (`PromptTools`; from the folder the model names, the template only when it asks for one), its programs in one session, a request's steps run as the model names them; `answers` (probabilities -> choice, score, confidence); `DecisionService` (one device thread, requests in arrival order) |
| `embedding/` | xs_embedding | `EmbeddingRunner`: an embedding model and its one stateless session, a run per token sequence, timings; `pooling` (mean or CLS, then the L2 norm, on the host); `EmbeddingService` (the tokenizer for text, one device thread) |
| `hub/` | xs_hub | `-hf`: models of the Hugging Face Hub. `http` (GETs through the environment's proxies, redirects followed by hand, a token only to the origin it is for), `download` (files in ranges over several connections, resumable from a list of the ranges written, checked against size and SHA-256), `hf_hub` (the Hub's listing API and the Hugging Face cache in huggingface_hub's own layout: blobs, snapshot links, refs, its lock files; offline use), `select` (which files of a repository a model needs; a GGUF file by its quantization tag, every part of a split one; a drafter's `mtp-*.gguf` or `dspark-*.gguf`; the drafter published apart from a model file), `model_source` (one path from every command's model spec, a directory, a GGUF file or `-hf`, to the directory it reads: `source_files`, the files a spec names (a GGUF file's companions its import reads, the ones the user names or else the ones `select` picks from its repository, or from its folder read as one), then `model_dir`: a directory as it is, a snapshot, a checkpoint beside the files it reads from another repository (`missing_companion_files`), or a GGUF file's pack in xpu-serve's cache, a repository's under its commit and imported once, a file's on disk under its canonical path and imported again when the stamp of its parts (path, size, modification time) changes), `file_lock.h`. docs/decisions/0021, 0030, 0031 |
| `cli/` | (exe) | `main.cpp` (commands serve, run, bench, convert, info; the model (`<model-dir>`, `<file.gguf>` or `-hf`, with `--mtp` and `--mmproj`) and the drafter (`--drafter`, `-hfd`) resolved through `hub::resolve_model` before anything loads), `args` (flag parser, with aliases like `-hf`) |

### 4.1 Files that matter most

| File | Role |
|---|---|
| `src/graph/graph.h` | **The op contract.** Every op's semantics are written next to `OpKind`; every backend must implement exactly that |
| `src/models/model.h` | What a model provides: `info()`, `graph()`, `weights()`, `prepare_inputs()`, and optionally `vision()` / `audio()` (a `VisionEncoder` / `AudioEncoder`: preprocessing into encoder runs, the encoder's programs (each run names one: e.g. one window of audio or eight), a warm-up item that runs each, an item's prompt layout) and `transcription()` (a speech recognizer's turn and reply format, its longest piece and reply, its prompt tokens per second of audio) |
| `src/models/qwen3/` | `qwen3::Decoder`: the Qwen3 text decoder (q/k norms, whole-head RoPE, SwiGLU, or a mixture of SwiGLU experts in the layers `Config::sparse` names, Qwen3-MoE's rule) as builders over one checkpoint's weights under a prefix; read by Qwen3-ASR and Qwen3-Omni's thinker (not served alone yet) |
| `src/models/qwen3_audio/` | Qwen's audio encoder, Qwen3-ASR's and Qwen3-Omni's: `qwen3_audio_config.cpp` (its audio_config, rows per mel frame, the log-mel settings of preprocessor_config.json), `qwen3_audio_encoder.cpp` (the front end: resampling, log-mel, windows of `n_window_infer` frames, eight a run (program 1) while that many remain and then one a run (program 0), the clip's last chunk zero-padded and the padded keys masked; the encoder graph: three stride-2 Conv2d, conv_out, positions per chunk, pre-LayerNorm layers attending within a window (blocks), ln_post, the projector; its weights where the model says) |
| `src/models/qwen3_asr/` | Qwen3-ASR: `qwen3_asr_config.cpp` (released `thinker_config` or transformers' flat layout; the forced aligner refused by name), `qwen3_asr_model.cpp` (the Qwen3 decoder with the audio encoder's rows at `<|audio_pad|>`), `qwen3_asr_transcription.cpp` (`language <Name><asr_text>`: forcing, parsing, ISO 639-1 codes, held back while streaming) |
| `src/models/qwen3_vision/` | Qwen3-VL's vision tower, Qwen3-Omni's and the Qwen3.5 family's (no deepstack rows; mergers under either naming, the patch embedding whole or a kernel a frame, a GGUF import's): `qwen3_vision_config.cpp` (its vision_config; Qwen2VLImageProcessor's settings and `smart_resize`), `qwen3_vision_encoder.cpp` (the processor: resize, normalize, patches in merge-block order, the position table's bilinear taps, rotary (row, column), the decoder's (0, row, column); the program over T = merged tokens with each token's 4 patches as rows: patch projection, the interpolated position table, blocks with 2-D RoPE and attention over all the image's patches, deepstack mergers, the last merger; output row = merged, then deepstack rows) |
| `src/models/qwen3_omni/` | Qwen3-Omni's thinker (`qwen3_omni_moe`): `qwen3_omni_config.cpp` (the thinker_config's text, audio and vision configs), `qwen3_omni_model.cpp` (the audio encoder at its larger size, the vision tower, and the Qwen3 MoE decoder reading positions on three axes, the encoders' rows at `<|audio_pad|>` / `<|image_pad|>`, an image's deepstack rows added after the decoder's first layers; experts joined from one tensor each; text, audio and images in, text out in the `qwen3` dialect; the talker not read) |
| `src/models/inkling2/` | Inkling 2 Mini (`inkling2_loop`): `inkling2_config.cpp` (the released config.json's own keys; refusals by name, its quantization_config among them), `inkling2_model.cpp` (the decoder: block residuals mixed by learned queries, attention with a relative bias and a sink over clipped queries and keys and normed values, short convolutions on keys, values and each sublayer's output (taps per token in the full-attention layers), softcapped experts behind a sigmoid router; the looped layers once per pass with KV and convolution slots of their own (`layers.<i>.pass<n>`) and messages from the pass before through gates; the released experts paired from NVFP4 and MXFP8, a GGUF import's banks with their rows in halves and a factor per expert, and its messages' queries per layer when mainline llama.cpp's layout keeps them so; with the checkpoint's drafter (`mtp.safetensors`, `mtp_config`) the prefill and verify programs, which append the drafter's context from the stream entering the layer visits it names, and the block program; an image's rows in place of its tokens' normed embeddings), `inkling2_vision.cpp` (the vision tower of a checkpoint or a pack of one: Inkling2ImageProcessor's padding to whole image tokens of patch x merge pixels, no resizing; patch embeddings with a position table interpolated over half-pixel centres, a class token, pre-norm layers with 2-D rotary attention and layer scale, a two-layer projection of each token's patches; its program runs over the image's tokens and one more block that holds the class token beside padding a key bias hides) |
| `src/models/drafters/dspark.cpp` | Inkling 2's block drafter (DSpark): its settings from the target's `mtp_config` (refusals by name), the context each layer reads (its row of `fuse_weight`, one tensor or a tensor per reader, through a softmax mixes the target's states; own key and value projections), the head's context row a target program outputs (`draft_context`), and the block program: the pending token's normed embedding and mask rows (one vector, or mainline's copy per draft), layers over the context and the block (local layers: the window before the block's start), a gated recurrent step per draft whose latent corrects the row the target's output layer reads (`hidden_out`), or with the correction folded through that layer (mainline's `markov_head.fold`) the block's rows through the output layer once and the fold's logits added per draft |
| `src/convert/gguf/inkling2.cpp` | GGUF import of Inkling 2 in either layout of its tensors, one reader told the layout (`Flavor`): `inkling2`, Unsloth's llama.cpp builds', and `inkling2_loop`, mainline llama.cpp's (its keys, layers counted once, the loop's gates under other names, each looped layer's four message queries as `layers.<i>.message_queries`): per-visit metadata arrays read back into the loop (refused unless the passes agree), names mapped to the checkpoint's, the transposes and flips the converter applied to float tensors undone (`FloatReorder`), the expert banks kept in the file's formats (NVFP4, MXFP4, K- and i-quants) with gate rows then up rows (`gate_up_rows: halves`) and each bank's factor per expert kept beside it; the drafter's file (`dflash`, `--mtp`) in either layout (Unsloth's, `dflash.dspark.markov_rank`; mainline's, `dflash.has_markov_context`: its mixes' logs as rows of `fuse_weight` per reader, its folded head kept as `markov_head.fold.*`, its window one shorter, its copy of the target's embedding norm left out) checked against the model's and mapped to `mtp.*` with config.json's `mtp_config`; the vision tower's file (`clip` with projector type `inkling2v`, `--mmproj`) mapped to `model.visual.*` with config.json's `vision_config`, its patch embedding put back from a convolution's layout |
| `src/server/transcription_handlers.cpp` | `POST /v1/audio/transcriptions`: the multipart form (each refusal names its field; timestamps refused), json / text / streamed events; `/v1/audio/translations` refused |
| `src/server/realtime_session.cpp` | `GET /v1/realtime`'s session: `session.update` (every refusal an `error` event naming its field), the audio buffer at 24 kHz, the server VAD's turns, commits, and the worker that transcribes committed turns in order (deltas, completed / failed) |
| `src/backend/sycl/sycl_conv.cpp` | Conv2d: the input unfolded into rows of patches (any stride, padding, dilation; group by group), one GEMM, the result laid out NCHW with the bias |
| `src/models/registry.cpp` | Dispatch on `config.json` `model_type`; the list of supported types |
| `src/models/gemma4/gemma4_model.cpp` | Gemma 4 text model: graph builder + per-step host inputs; experts beside the MLP and K = V global layers (26B-A4B); an assistant checkpoint as the drafter (query-only layers over the target's KV through `kv_read`, the centroid head) |
| `src/models/gemma4/gemma4_config.cpp` | Gemma 4 `config.json` parsing; refuses unsupported variants by name |
| `src/models/gemma4/gemma4_vision.cpp` | Gemma 4 vision: image processor (resize, patches, pooling map) and vision tower graph |
| `src/models/qwen3_5/qwen3_5_model.cpp` | Qwen3.5 (and SenseNova Loop, `sensenova_loop`: block drafters refused, they read layers by index): with the vision tower (`qwen3_vision`, under `model.visual.`), an image's rows spliced in place of its tokens' embeddings and positions on three axes for the full-attention layers (not beside a block drafter: docs/decisions/0032); Gated DeltaNet layers (causal conv + gated delta rule) and gated full attention; dense MLP or `blocks::moe` (`qwen3_5_moe`: fused gate_up expert banks, an FP8 release's per-expert tensors joined at load, or a GGUF import's gate and up banks apart; gated shared expert); the MTP head (`mtp.*`) as the drafter, built from the same layer builders, or a block drafter checkpoint in its place |
| `src/models/qwen3_5/qwen3_5_decoder.cpp` | `qwen3_5::Decoder`: the decoder's layer builders over one checkpoint's weights, shared by the language model and the decision models built on it; full-attention layers causal through a KV slot, or over the whole step; linear attention's value heads in the checkpoint's order (`linear_value_head_order`: grouped as Hugging Face stores them, or tiled as GGUF files do, moved to the recurrence's order and back around GatedDelta); a mixture of experts' gate and up banks fused (`experts.gate_up_proj`) or apart (`experts.gate_proj` and `experts.up_proj`, a GGUF import's), any other layout refused by name; the layers in the order `Config::plan` runs them (SenseNova Loop's `loop_settings`: each run with state slots of its own, `layers.<i>.pass<n>`), and the hyper-connections of its `mhc` layers (`mhc_layer`: streams `[T, S, hidden]`, a router per branch through MhcRoute, the branch's input and output through StreamMix) |
| `src/convert/gguf/qwen35.cpp` | GGUF import of the Qwen3.5 family, dense (`qwen35`) and mixture of experts (`qwen35moe`: one class and one decoder reader; the vision tower from llama.cpp's mmproj file (`--mmproj`, projector type `qwen3vl_merger`: under `model.visual.*`, the patch embedding kept a kernel a frame, config.json in the release's layout and its preprocessor_config.json), told the layers' MLP by `Qwen35Mlp`): metadata -> config.json, `blk.<i>.*` -> Hugging Face names, the converter's transforms undone on float tensors (norms stored as 1 + w, `ssm_a` = -exp(A_log)), value heads left tiled (`linear_value_head_order`), a dense model's merged input projections cut by rows, a mixture of experts' router, shared expert and banks (gate and up kept apart as `experts.gate_proj` and `experts.up_proj`, each in its file's format; the shared expert's gate back to [1, hidden]), the MTP block -> `mtp.*`, also from the head's own file (`--mtp`: llama.cpp's draft model, its metadata checked against the model file's, its own embedding, output head and final norm checked and left out) |
| `src/convert/gguf/clef.cpp` | GGUF import of Clef (Clef-Flash, Clef): the decoder read as a `qwen35` file's under `clef.*` (`read_qwen35_decoder`, `qwen35.h`), the head's `dec.blk.<i>` (evidence layers, then decoder layers) and `decision.*` tensors mapped back to the checkpoint's names under `joint_head.`, each attention's q / k / v kept apart as a pack of the checkpoint holds them, `decision.scales` (the values the head multiplies by) kept as `joint_head.scales`, joint_head_config.json from the metadata and the head's tensor shapes; llama.cpp's `systemone` prompt template left out (the model code writes the prompt) |
| `src/convert/gguf/modern_bert.cpp` | GGUF import of Laya and Julia-1 (`modern-bert` with `decision.type` laya), told apart by the prompt template llama.cpp's converter writes for each: the original checkpoint's layout (Laya's config.json and rl_agent_config.json, Julia's config.json, julia_config.json and inference-policy.json; encoder/config.json; the tokenizer in tokenizer/), the head's blocks after the encoder's mapped back to torch's names, Laya's temperatures from the metadata; what the file does not hold (Laya's state budget, Julia's inference policy) from the published checkpoint it identifies |
| `src/convert/gguf/gemma4.cpp` | GGUF import of Gemma 4 E2B / E4B as a text-only checkpoint, and of its assistant (`gemma4-assistant`) as a drafter pack (config.json and weights; no centroid head in the file, so `use_ordered_embeddings: false`, the whole-vocabulary head llama.cpp computes): metadata (per-layer head counts and MLP widths, the sliding pattern, KV sharing) -> config.json, names mapped back with no value changed, a fused `attn_qkv` cut by rows, `rope_freqs` read as the proportional RoPE's `partial_rotary_factor` and left out; refuses a layout whose KV sharing llama.cpp reads otherwise than Gemma 4 |
| `src/models/decision.h` | What a decision model provides: `spec()`, `programs()`, `weights()`, and `start(request, tools, limits)` -> a `DecisionRun` that hands the runtime one step at a time and ends with each question's probabilities |
| `src/models/clef/` | Clef: `clef_prompt.cpp` (the reference's record encoding: token ids and the spans of each instruction and option), `clef_model.cpp` (the encode program, which keeps what the head's attentions read of the sequence in KV slots, the route, fields and score programs, and the host work between them; the head's three scalars as the checkpoint's parameters, or as a GGUF import's `joint_head.scales`) |
| `src/models/pplx_decider/` | pplx-decider (protocol version 3): `pplx_prompt.cpp` (the chat-template prompt with answer codes and a marker per question, special tokens in user text neutralized, a long request split by question), `pplx_model.cpp` (one program: the decoder with whole-step full attention and the read-out on every row; temperature and softmax on the host) |
| `src/models/d1/` | d1 (LiquidAI/d1-2B, an `lfm3_vl` checkpoint whose config.json names `D1Model`): `d1_prompt.cpp` (the reference's prompt.py: the state as text or indented JSON, a question's ChatML turn ending in the empty thought, special tokens in user text broken by U+200B, option codes with their fallback pool, the tokens each option is read at; a request as sequences of a state and up to the set size of questions after it, as attention.py packs them, each token's group and each question's answer position), `d1_model.cpp` (one program: `lfm3::Decoder` with full attention in groups, the final norm at the answer rows; each option's best logit from the output layer's rows on the host, over the calibrated temperature of its question's bucket, through a softmax; config and calibration.json, the set size (`d1_set_size`, or `DecisionLoadOptions::set_size`), refusals by name) |
| `src/models/laya/` | Laya and Julia-1, one model whose differences are data (`Policy`): `laya_config.cpp` (Laya's rl_agent_config.json; Julia's config.json, julia_config.json and inference-policy.json; the encoder's config and the tokenizer's marker tokens), `laya_prompt.cpp` (a sequence per question: [CLS] question [SEP], the mask token and up to 48 tokens per option, [SEP] state [SEP]; Laya's wording and cuts, Julia's descriptions and refusals), `laya_model.cpp` (one program: the encoder, the question type's row, two torch TransformerEncoderLayers, the scorer on every row; the logits at the markers over Laya's temperature through a softmax on the host) |
| `src/models/letter_readout/letter_readout.cpp` | A `Model` behind the `DecisionModel` interface (OpenJev's helper): a prompt per question through the chat template, the logits of the option letters at the first output position, temperature and yes / no calibration (`Settings`, presets by name), more options than letters in parts and a pass over their winners; the state's part of a prompt read once and each question's part undone for the next |
| `src/decision/runner.cpp` | Wiring from a decision model directory to answers: compiles the programs, runs a request's steps, times them; its prompts end where the model's answer starts (`chat::direct_answer` of the spec's dialect: Inkling 2's own message header) |
| `src/server/decision_handlers.cpp` | `POST /v1/systemone`: the request schema (422 bodies naming the field), the answers, `Server-Timing` |
| `src/models/embedding.h` | What an embedding model provides: `spec()`, `weights()`, `graph()` (the encoder, its final hidden states as "hidden") and `prepare_inputs(ids)` |
| `src/models/common/sentence_transformers.cpp` | A sentence-transformers checkpoint's pipeline (`modules.json`, the Pooling config in both spellings, `sentence_bert_config.json`), as sentence-transformers 6 reads it; every other module, pooling mode or setting refused by name |
| `src/server/embedding_handlers.cpp` | `POST /v1/embeddings`: text, token ids or lists of them; float or base64; 400s naming the field (`input[2]`), over-long inputs refused |
| `src/models/lfm3/lfm3_decoder.cpp` | `lfm3::Decoder`: LFM3's layer builders over one checkpoint's weights (under `model.`, or `model.language_model.` in an `lfm3_vl` checkpoint, whose `text_config` `lfm3::text_config` reads), shared by the language model and d1: gated short convolutions (`causal_conv` between two gates of one projection), attention with q/k norms and a sigmoid gate per head (full layers rotate half of each head, sliding layers all of it within their window; full layers causal through a KV slot, or over the step's tokens in groups), every operator and MLP output normed again before the residual add |
| `src/models/lfm3/lfm3_model.cpp` | LFM3 as a language model: the decoder, its last row through the final norm and the head tied to the embedding |
| `src/models/drafters/dflash.cpp` | Block drafters (DFlash, DFlash 2, DSpine): their config (refusals by name), the context a target's prefill and verify programs append for them, and the block program (DFlash 2's convolutions and candidate selector, DSpine's code bridge and refined last write) |
| `src/backend/backend.h` | `Backend::compile(graphs, weights, options) -> Session` (programs sharing weights by name and state slots by label); `Session::run(program, inputs, outputs, only)` (every output, or just the ones asked for), `reset`, `truncate` (inside a step of up to `rewind_tokens` tokens too) |
| `src/backend/cpu_ref/ops.cpp` | f32/f64 reference kernels, one per op: the accuracy oracle |
| `src/backend/sycl/sycl_backend.cpp` | SYCL session: device dtypes, fusions, per-T memory plan (one arena), op dispatch, KV and state buffers; with `SessionOptions::max_step_tokens`, the KV cache of the whole context and the largest step's arena reserved at compile time, a context that does not fit the device refused with its bill; a step of several sequences (`run_segments`): every op over all the rows, the stateful ones through a table of the segments uploaded once a step (`segment_tables`: one launch each), or without such a kernel a segment at a time over its rows (`Rows`) and its sequence's slots |
| `src/backend/sycl/sycl_elementwise.cpp` | An elementwise program: a tree of elementwise ops (`graph/elementwise_groups.h`) run as one kernel, each leaf read through its map (same shape, repeating, held across blocks, a slice in place), the steps from device memory over eight registers the host allocates; `sycl_unary.h` is the unary ops it and the other kernels share |
| `src/graph/elementwise_groups.cpp` | Which trees of elementwise ops a backend can run as one pass: an op read only by the next op of its tree folds in, a slice read only by the tree is read in place, broadcasts by their map; within a backend's limits of leaves and ops |
| `src/backend/sycl/sycl_kernels.cpp` | SYCL kernels: elementwise, norms, RoPE, gather, copies, causal conv, gated delta, decode attention (one kernel over a reader of the keys and a reader of the values, `sycl_kv_rows.h`: floats, int8 codes or TurboQuant codes decoded in registers; a work-group serves up to eight (token, query head) rows that share a KV head, the heads of several tokens of a verify among them, so each key and value is read once for them), GEMV, top-k (rows as wide as a vocabulary by blocks), SignMatMul (bit planes and popcounts) |
| `src/backend/sycl/sycl_kv_cache.cpp` | A KV cache's quantized formats on the device: rows written in a format as they are appended (the host's bits), decoded for the attention paths that read floats, and the TurboQuant rotation of queries and outputs; `sycl_kv_turbo.h` is what these kernels and the readers share (a group of 128 values in a sub-group's registers: the transform, a code's centroid by selects, a lane's codes) |
| `src/backend/sycl/sycl_matmul.cpp` | MatMul (GEMV up to 8 rows, oneMKL GEMM beyond; quantized: `quant_path` picks the GEMV or its matrix-engine version, quant_gemm (a block format: gemm_block_xmx), quant_gemm_tiled or dequantize + oneMKL per device and shape), ExpertMatMul over a quantized bank along the path its pairs take (`expert_matmul_bank`) and over many pairs through oneMKL (grouped by expert into one grouped GEMM; a quantized bank's experts decoded into scratch, a float bank in the GEMM's type read in place) and Attention (per-row kernel with split-K, the fused kernel, or GEMMs) |
| `src/backend/sycl/sycl_flash_attention.cpp` | Fused attention on the matrix engines (joint_matrix; head_dim up to 128; blocks of tokens in one launch; a relative bias and a sink in its online softmax) |
| `src/backend/sycl/sycl_quant_gemm.cpp` | Quantized matmuls on the matrix engines (DPAS and 2D block load builtins): gemv_quant_xmx (q4 and q8, up to 8 rows at about the cost of one: a speculative verify, a decode step of several requests; x in f16 and its group sums from one small launch before), quant_gemm (9 to 64 rows, the weight read once, 64 bytes of codes a lane and 32 rows of x a load; weights that read the same x share a launch) and quant_gemm_tiled (many rows: each weight tile dequantized once per work-group of 512 rows into local memory, no f16 copy of the weight) |
| `src/backend/sycl/sycl_replay.h` | A run of launches recorded as a command graph and run again (sycl_ext_oneapi_graph): what the profile of a step of a few tokens times its kinds of op with; what a recording cannot hold refuses by throwing |
| `src/backend/sycl/sycl_xmx.h` | What the kernels written against the matrix engines directly share: the DPAS (over 1, 2, 4 or 8 rows) and block load builtins, their operand types, and `xmx_prep` (x in f16 and its sums per group of columns: the A operand of the GEMVs on the matrix engines) |
| `src/backend/sycl/sycl_block_layout.cpp` | GGUF block-format weights for the device: a block's fields regrouped into aligned streams with codes in column order (the same bits, nothing re-encoded); Q4_0, Q4_1 and Q8_0 rewritten as a pack's q4 / q8 with a group of 32 (the same values in the file's bytes; Q4_0 without biases, a symmetric q4 the pack kernels read around 8: `QuantCode`), which the pack kernels then read |
| `src/backend/sycl/sycl_block_kernels.cpp` | Kernels over those streams, one decoder per format (`Fmt<F>`: K-quants read their codes without an int-to-float conversion, codebook formats look their magnitudes up in a table on the device): the GEMV (up to 8 rows, each weight read once) and its matrix-engine version (`gemv_block_xmx`: each lane writes its own output feature's codes or levels, exact in f16, into DPAS B operands, so the cost hardly grows with the rows: a speculative verify, a decode step of several requests; `gemm_block_xmx`: the same over tiles of 8 rows, 9 to 64 rows with the weight read once per 32), the weight written out as f16 / f32 for a GEMM, rows gathered by id. an expert bank's GEMV per expert over its pairs on the matrix engines, per (token, expert) pair without them, or grouped by expert past 64 pairs (`expert_matmul_block`); past 64 pairs on the matrix engines, the pairs in passes of up to 64 rows of one expert, the bank read once a pass and never written out (`expert_gemm_block_xmx`). `QuantWeight::block` routes a weight to them through the pack kernels' entry points |
| `src/engine/engine.cpp` | Generations advanced a step at a time (`start`, `advance`, `finish`), each on a sequence of the session; encodes images and splices their rows into prefill chunks; speculative rounds (`spec_round`) |
| `src/models/model.h` (`Speculation`) | What a model with a drafter adds: prefill, verify and draft programs, and its way of drafting (`ChainDrafting`: a token a run from the target's `hidden`; `BlockDrafting`: a block a run from state the target's programs write) |
| `src/service/engine_service.cpp` | Wiring from a model directory to a ready service |
| `src/server/completion_handlers.cpp` | `/v1/chat/completions` and `/v1/completions` |
| `src/cli/main.cpp` | Commands, flags and their defaults |

## 5. How a request flows

```text
HTTP POST /v1/chat/completions                                     server/completion_handlers.cpp
  -> parse + validate the OpenAI body (400 names the field);       server/openai_request.cpp
     image_url data: URLs and input_audio parts come out as bytes; response_format's schema -> a grammar   grammar/
  -> prepare_media(modality, bytes): decode, preprocess -> MediaInput   image/, audio/, models/<arch>/ (encoders)
  -> ChatTemplate::render(messages, tools)                         chat/chat_template.cpp (+ third_party/jinja_cpp)
  -> Tokenizer::encode(prompt)                                     tokenizer/
  -> expand_media: each image placeholder -> its token run         engine/media.cpp
  -> with a format: the request's constraint, from where the dialect's answer starts   server/grammar_constraint.cpp
  -> InferenceService::generate(request, on_token)                 service/engine_service.cpp (queues to engine thread)
       Engine::generate                                            engine/engine.cpp
         reuse_prefix: keep the KV of the longest common prefix with the previous request (images by content hash)
         prefill in chunks (--prefill-chunk); a media item's rows come from its encoder runs (cached by hash)
         -> decode one token per step:
           Model::prepare_inputs(tokens, start_pos, rows) -> TensorMap   models/<arch>/
           Session::run(inputs) -> logits                          backend/<name>/
           TokenConstraint::mask(logits), a constrained request's  engine/constraint.h (grammar/matcher.cpp)
           Sampler::sample(logits)                                 engine/sampler.cpp
         on_token(token) per generated token
  -> StreamDecoder (text deltas) -> OutputParser (content / reasoning / tool calls)   tokenizer/, chat/
  -> stop strings, SSE chunks or one JSON body                     server/
```

Threads: HTTP threads parse, render and tokenize; one engine thread owns the device and advances the generations in
flight (one with `--sequences 1`): a prefill chunk of one request, or one decode step of all that decode (the batch
graph over a segment a request; speculating, the verify graph over each request's pending token and drafts), the
two sides sharing its time.

A transcription:

```text
HTTP POST /v1/audio/transcriptions (multipart)                     server/transcription_handlers.cpp
  -> parse the form (400 names the field), decode the file -> TranscriptionRequest
  -> InferenceService::transcribe (queues to the engine thread)    service/engine_service.cpp
       cut the audio at quiet points into pieces the model and the context hold (Transcription::max_piece_seconds,
       max_reply_tokens, audio_tokens_per_second)                  audio/
       per piece: the turn through the chat template + the forced language, the audio's MediaInput, generate
       Transcription::parse: the language and the text (held back while the header is being written)   models/<arch>/
  -> {"text", "usage"}, plain text, or transcript.text.delta / .done events
```

A realtime session (one per WebSocket connection):

```text
GET /v1/realtime (WebSocket; the key as a header or as the openai-insecure-api-key subprotocol)   server/server.cpp
  -> RealtimeSession: session.created; each client event to handle()                         server/realtime_session.cpp
       session.update: format, transcription settings, server_vad or manual turns (error events name what is refused)
       input_audio_buffer.append: 24 kHz PCM into the buffer; server_vad: EnergyVad -> speech_started / speech_stopped
                                                                                                audio/vad.cpp
       a turn's end (speech_stopped or input_audio_buffer.commit): committed, conversation.item.added, the turn queued
  -> the session's worker: InferenceService::transcribe(the turn) as above -> ...transcription.delta events,
     .completed (or .failed), conversation.item.done
```

A decision model's request is shorter:

```text
HTTP POST /v1/systemone                                            server/decision_handlers.cpp
  -> parse + validate the body (422 names the field) -> DecisionRequest (the client's key order kept)
  -> DecisionService::decide (queues to the device thread)         decision/service.cpp
       DecisionRunner::decide                                      decision/runner.cpp
         DecisionModel::start(request, tools, limits)              models/<arch>/: prompt layout and tokens
         until the run has no step left:
           DecisionRun::next(outputs of the step before) -> program, inputs      models/<arch>/
           Session::run(program, inputs) -> outputs                backend/<name>/
         DecisionRun::probabilities() -> make_answer per question  decision/answers.cpp
  -> answers as JSON, timings in Server-Timing                     server/decision_handlers.cpp
```

An embedding model's:

```text
HTTP POST /v1/embeddings                                           server/embedding_handlers.cpp
  -> parse + validate (400 names the field); a text -> encode + the special tokens around it   tokenizer/
  -> EmbeddingService::embed (queues to the device thread)         embedding/service.cpp
       EmbeddingRunner::embed, one run per input                   embedding/runner.cpp
         EmbeddingModel::prepare_inputs(ids) -> Session::run -> hidden [T, D]   models/<arch>/, backend/<name>/
         pool: mean or CLS, then the L2 norm when the checkpoint lists Normalize   embedding/pooling.cpp
  -> {"data": [embedding per input], "usage"}, floats or base64; timings in Server-Timing
```

## 6. The step graph and the backends

**The step graph** (`src/graph/graph.h`) describes one engine step (a prefill chunk or one decode token) for one
sequence, or, when it allows segments, for several. Model code builds it once at load time; a backend compiles it once.

- Two symbolic dims: `kT` (tokens fed this step) and `kL` (tokens in the KV cache after this step). All other dims
  are static. No batch dim: a graph that allows segments (`Graph::allow_segments`, docs/decisions/0028) keeps its rows
  independent (`validate()` checks it), and a run of it (`Session::run_segments`) takes segments of rows, each the next
  tokens of one sequence, whose stateful ops act on each segment with that sequence's state.
- Values are logically f32; weights keep their stored dtype and backends choose the compute precision.
- MatMul weights use the Hugging Face layout `[out, in]`: `matmul(x, w) = x @ w^T`.
- Ops: Input, Weight, Constant, Gather, MatMul, Add, Mul, Div, Scale, Tanh, GeluTanh, Gelu, Softcap, Sigmoid, Silu, Softplus,
  Clamp, RmsNorm, L2Clip (a vector longer than a threshold scaled down to it), Softmax, TopK, ReduceSum, Reshape, Slice, Transpose, PrevRow (each row of a step replaced by the row
  before it), Rope (1-D, axial, or multimodal: each frequency reads one of a token's position axes, Qwen's time /
  height / width; optionally partial; over rows of several a token too), KvCache, KvRead (another program's KV slot,
  read only), Attention (causal with optional window, or non-causal: every key, the keys of the query's own block of
  tokens, a window as wide after each query as before it (a drafter's block over a cache, an encoder's local
  attention), or the step's keys and a window before the step, the same for every query (`anchored`: DSpark's block); or the step's tokens in groups, group 0 seeing and seen by all, any other group itself and group 0:
  questions packed after their state; or rows of several a token as one sequence, a vision tower's patches, with
  a bias a key when given one; causal
  with a bias by how far back a key sits and a sink, one more logit without a value: Inkling 2),
  CausalConv and GatedDelta (stateful, for linear attention), DynamicConv (a causal convolution whose taps are per
  token, over a conv slot),
  ExpertMatMul (each token's chosen experts from a stacked `[E, N, K]` bank, for mixture of experts), SignMatMul (a
  matmul by a one-bit weight: the cheap score of every row that a shortlist starts from), MhcRoute and StreamMix
  (manifold-constrained hyper-connections: a router's read and write weights and its Sinkhorn stream mix; residual
  streams mixed and read), Conv2d (strided, padded per side, dilated, grouped; 1-D as height 1: audio front ends,
  codecs). The exact semantics are the comments in `graph.h`.
- KV: `add_kv_slot(...)` declares per-sequence state; `kv_cache(slot, k, v)` appends this step's K/V and returns the
  whole cache. A slot can be read by later layers (Gemma 4's KV-shared layers reuse an earlier layer's outputs).
- Graph inputs are filled per step by `Model::prepare_inputs` (token ids, positions, and model-specific host inputs
  such as Gemma 4's per-layer embedding rows, and the image rows that replace soft-token embeddings).
- Conv and recurrent slots (`add_conv_slot`, `add_recurrent_slot`) hold fixed-size per-sequence state for
  `causal_conv`, `dynamic_conv` and `gated_delta`. They summarize the whole sequence, so a session cannot cut them back to a prefix,
  with two exceptions: the last step can be undone whole (the state before it is the buffer the step did not write:
  how a decision model reads one question after another on one state), and inside a last step of up to
  `SessionOptions::rewind_tokens` tokens any count can be kept (a speculative verify; the backend keeps that step's
  inputs and replays the kept ones). The engine reuses a cached prefix for such models only when the new prompt
  extends it, else it starts over.
- A session can hold several programs (the target's step, prefill and verify graphs and a drafter): they share
  weights by name and state slots by label; a program that only reads a KV slot (`kv_read`) attends to another's.
- A graph without slots is stateless (every run stands alone): that is how vision encoders run, one image per run,
  compiled as a second session next to the text one.

**Backends** (`src/backend/`):

| Backend | Status | Notes |
|---|---|---|
| `cpu-ref` | test oracle | f32 values, f64 accumulation, threaded matmul. Slow by design; correctness only |
| `sycl` | main GPU backend, the default when built (icpx turns it on) | Runs the graph op by op on an in-order Level Zero queue with its own kernels; oneMKL for prefill GEMMs. Weights and KV f16 (f32 at f32 precision), activations f32; `--kv-dtype` stores the KV slots only attention reads in a quantized format: `i8` (int8 with an f32 scale per token and KV head) or the TurboQuant formats `turbo4` / `turbo3` / `turbo2` (head dims that are multiples of 128; rows stored rotated), keys and values alike or `i8` keys with TurboQuant values. Decode attention reads the codes; prefill decodes the keys and values in view to f16 for its GEMMs, with a TurboQuant format after rotating the queries, and rotates the output back. Memory: one arena per step, planned per T from value lifetimes; KV slots grow by doubling; conv/recurrent state double-buffered so a failed step changes nothing. Fuses act(x)*gate and conv+SiLU. Decode attention reads only the sliding window, splits long caches across work-groups, and reads each key and value once for the query heads that share its KV head. A KV slot with a sliding window that attention alone reads keeps only that window's keys when the session has conv or recurrent state (it can then never be cut back before its last step, so no later step reads older keys): its memory stops growing with the context. Options `attention=auto|rows|gemm`, `gemv=auto|vector|xmx` (quantized matmuls of up to 8 rows: the GEMV, or its matrix-engine version from 2 rows, from 3 for a pack's q8), `few_rows=N` (quantized matmuls of 9 to N rows on quant_gemm, 64 by default; 0: dequantize + GEMM), `tiled_rows=N` (past those, up to N rows on quant_gemm_tiled: 512 by default on an integrated GPU, 0 on a discrete one). GGUF block formats keep the file's bits, regrouped for aligned loads (`block=f16` decodes them to f16 instead: the reference the kernels are tested against). See docs/decisions/0007 |

## 7. Tests (`tests/`)

One test binary, `xs_tests` (doctest). Every `tests/unit/**/*.cpp` is compiled in automatically.

| Path | What it checks | Against |
|---|---|---|
| `tests/unit/test_graph.cpp`, `test_half.cpp`, `test_base64.cpp` | IR shape inference and validation; f16/bf16 conversions; base64 | hand-written expectations, RFC 4648 vectors |
| `tests/unit/image/` | decoding, resize, hashing | Pillow-written images; torchvision resize outputs in `tests/fixtures/image/` (bit-exact) |
| `tests/unit/audio/` | WAV decoding of every sample format, channel count and fmt layout (files the test writes), RF64 and streamed WAVs, chunks skipped; MP3 and FLAC; refusals by name (truncated and malformed files, other containers); the WAV writer; hashing; resampling (44.1, 48, 8 and 22.05 kHz to 16 kHz, edge lengths); the FFT; log-mel features (a chirp, noise, silence, a clip below min_length, a length off the hop, full scale; frames computed in parts on threads, a clip of period 1 s giving equal frames 1 s apart); the VAD (`test_vad.cpp`: where turns start and stop with their prefix and closing silence, clicks, short pauses, the threshold, a loud room, the same events however the stream is cut); planted: another rolloff, zero padding in place of reflect, the symmetric window | libsndfile's decodes (soundfile) of files soundfile and ffmpeg wrote; torchaudio's resample; transformers' Qwen3ASRFeatureExtractor, and its math in float64; in `tests/fixtures/audio/` (`tests/parity/audio/gen_audio_cases.py`); the DFT in long double |
| `tests/unit/backend/cpu_*.cpp` | each op on cpu-ref (`cpu_encoder_ops.cpp`: the encoder ops and stateless sessions, attention in groups, multimodal RoPE (`mrope_cases.h`), attention and RoPE over rows of several a token (`rows_cases.h`)), KV sessions (a quantized cache holds the values its formats store), attention masks | fp64 references in `reference_ops.h`, with planted-bug controls |
| `tests/unit/backend/device_*.cpp` | each op on the GPU backend (`device_encoder_ops.cpp` likewise), sessions, agreement with cpu-ref (also with an int8 or a TurboQuant KV cache on both; a query that sees one key reads the value the host stores for it), decode timing and quantized matmul timing by rows, format and path (`XS_DEVICE_BENCH=1`, `device_quant_timing.cpp`) | the same references; skip unless `XS_DEVICE_TEST_DEVICE` is set |
| `tests/unit/backend/*_conv_ops.cpp` | Conv2d: Qwen3-ASR's 3 x 3 stride-2 layers over one channel and many, a causal dilated 1-D convolution, depthwise and grouped ones, padding on two sides only, a symbolic batch; on cpu-ref and the GPU | fp64 references (`conv_ops_cases.h`); planted: the kernel flipped |
| `tests/unit/test_media.cpp` | media placeholder expansion: each item's token run in order, images and audio in one prompt, items out of order or of a modality the model does not take refused | fake encoders |
| `tests/unit/model/qwen3_asr_transcription.cpp` | Qwen3-ASR's turn and reply: forcing a language by code or name (refusals), the language and text of a reply (silence, no header, a forced language), a header held back while streaming, repeated text cleaned | the reference package's own `detect_and_fix_repetitions` outputs |
| `tests/unit/model/media_parity.cpp` | a model that reads audio or images (`XS_TEST_MODEL_DIR`: Qwen3-ASR, Qwen3-Omni's thinker, the Qwen3.5 family with its tower; at f16 a near tie may flip, as model_parity allows): per clip the log-mel features and the audio encoder's rows run by run, per image the processor's patches and the vision tower's rows (merged and deepstack); the prompt (the case's messages through the template and tokenizer, a media token per row), last-token logits at the engine's positions (`rotary_positions`), greedy text through the engine; Qwen3-ASR's two checkpoint layouts | transformers' Qwen3-ASR, Qwen3OmniMoeThinkerForConditionalGeneration and the Qwen3.5 family's ForConditionalGeneration classes in fp32, from `tests/parity/model/hf_media_reference.py` (`make_tiny.py --arch qwen3_asr`, `--arch qwen3_omni`, `--arch qwen3_5_vl` or `--arch qwen3_5_moe_vl` writes a tiny one in the released layout; `--converted-out` writes transformers' layout of Qwen3-ASR's; synthetic images of sizes the processor resizes each of its ways) |
| `tests/unit/backend/*_hc_ops.cpp` | MhcRoute (2 and 4 streams, a bias or none, 1, 3 and 20 Sinkhorn iterations, logits large enough to need the peak subtracted) and StreamMix (a read, a mix, a mix with the branch's update; one token and several) on cpu-ref and the GPU | fp64 references (`hc_ops_cases.h`); planted: too few iterations, the bias dropped, the mix transposed |
| `tests/unit/backend/*_moe_ops.cpp` | Div, Softmax, TopK (ties to the lower index), ReduceSum, ExpertMatMul (a float bank over few pairs and past a grouped GEMM's threshold, quantized banks likewise) and the `moe` block (softmax and sigmoid routers, shared expert; a selection bias with alternating gate and up rows and a softcapped hidden state; banks stored without a factor of each expert's, gate rows then up rows), on cpu-ref and the GPU | fp64 references and an fp64 composition of the block (`moe_ops_cases.h`) |
| `tests/unit/backend/*_relative_ops.cpp` | L2Clip (rows on both sides of the threshold, a row of zeros), DynamicConv over several steps (state carried, taps per token and group) and causal attention with a relative bias and a sink over a cache fed in steps (on the per-row, GEMM and fused paths; an extent shorter and longer than the cache, a window shorter and longer than it, keys split between work-groups), attention over a window anchored at the step (blocks after contexts shorter and longer than the window), on cpu-ref and the GPU | fp64 references (`relative_ops_cases.h`); planted: a norm in place of the clip, taps in time order, the sink left out, the bias read toward the future, a window around each row |
| `tests/unit/backend/*_shortlist_ops.cpp` | SignMatMul (words loaded whole and one by one, rows that fill no sub-group) and the shortlist head: the best rows by sketch, their exact logits, the head's pick, on cpu-ref and the GPU | fp64 references (`shortlist_cases.h`) |
| `tests/unit/backend/*_programs.cpp` | sessions of several programs: KV slots shared by label, KvRead, truncate inside a short step (conv, gated delta, KV) equals never running the dropped tokens, the last step undone whole equals never running it; a KV slot with a sliding window beside conv state over many steps, some undone or cut, far past the window (a backend may keep the window alone); a run asked for some outputs returns those and advances the state all the same; a snapshot of a sequence's state restored after steps past its sliding window, and refused when its tokens are gone; a prefix copied from another sequence with its snapshot | the same session without them (`session_cases.h`) |
| `tests/unit/engine/` | rotary positions (`rotary_positions.cpp`: an image's tokens at their offsets, the text after it one past its largest, generated tokens after that, prefix reuse, a one-position model refusing media with positions); speculative decoding emits exactly the greedy tokens (a stateful tiny model, a partly wrong drafter that drafts a token a run or a block a run, prefix reuse, stops); a block drafter's runner handing the draft program the target's row of the last token kept (a prefill's rows or its last alone, a verify cut inside, the sequence cut back); the same tokens when greedy decoding reads each row's `argmax` output instead of the logits; prefix reuse from a snapshot of a sequence's state (`prefix_snapshot.cpp`: a turn departing after the snapshot point, before it, a request naming none, the sequence a request can use the most of, a busy sequence's prefix taken as a copy past kCopyGain tokens); rejection sampling draws the target's distribution; constrained decoding (`constrained_engine.cpp`, the tiny model's tokens read as pieces of JSON): the output parses and is of the schema, greedy or sampled, with a chain or a block drafter (the constrained greedy tokens, drafts the grammar refuses never verified), where the free output already conforms the same tokens, a stop token only where the output may end, batched beside a free request, refusals | plain greedy decoding; exact distributions; the output parsed as JSON |
| `tests/unit/backend/*_recurrent_ops.cpp` | sigmoid/silu/softplus, partial rope, CausalConv and GatedDelta over several steps (state carried), on cpu-ref and the GPU | one fp64 pass over the whole sequence (`recurrent_ops_cases.h`) |
| `tests/unit/tokenizer/` | encode/decode, streaming, the post-processor's special tokens; NFC and regex Split hermetically (`test_tokenizer_qwen35.cpp`); WordPiece, BertNormalizer, BertPreTokenizer and the WordPiece decoder on a hand-written vocabulary, and the BERT normalizer and pre-tokenizer over every code point (`test_tokenizer_wordpiece.cpp`); the GPT-2 regex, ByteLevel and Metaspace, the regex over every code point (`test_tokenizer_laya_julia.cpp`); a tokenizer read from slow files, a hand-written Qwen2Tokenizer's (merges.txt as tokenizers reads it, added tokens and their flags, the special tokens transformers names, the class's eos; refusals by name) and Qwen3-ASR's (`test_tokenizer_slow_files.cpp`); the o200k regex over hand-picked and random text, and the letter cases it reads (`test_tokenizer_inkling2.cpp`); whole tokenizers of Gemma 4, Qwen3.5, LFM3, Inkling 2, Qwen3-ASR, the WordPiece embedding models and the Laya and Julia-1 decision models (ModernBERT's and mmBERT's) (`parity_cases.h`; each needs its checkpoint's `tokenizer.json`, Qwen3-ASR its vocab.json, merges.txt and tokenizer_config.json; `scratch_model_dir.h` writes hand-written files to load) | Hugging Face `tokenizers` outputs in `tests/fixtures/tokenizer/` (`tests/parity/tokenizer/gen_cases.py`, `gen_qwen35_cases.py`, `gen_lfm3_cases.py`, `gen_inkling2_cases.py`, `gen_wordpiece_cases.py`, `gen_laya_julia_cases.py`; `gen_qwen3_asr_cases.py`: AutoTokenizer over the slow files, checked against the Qwen2Converter's tokenizer; tables from `gen_unicode_tables.py`) |
| `tests/unit/chat/` | template rendering (LFM3's template comes from `XS_LFM3_DIR`; Qwen3-ASR's is a chat_template.json, with audio parts), which template file wins, Gemma 4, Qwen3.5 and Qwen3 output parsing (`test_qwen3_parser.cpp`: XML and JSON tool calls), the Qwen3.5 dialect on LFM3's prompts too | `apply_chat_template` outputs in `tests/fixtures/chat/` (`tests/parity/chat/gen_cases.py`) |
| `tests/unit/grammar/` | the byte grammar (`test_grammar.cpp`: repeats, recursion, a repetition's states not growing, what `finish` drops and refuses); JSON grammars (`test_json_schema.cpp`: documents a schema takes and does not: properties in order, optional ones, types, enum, const, anyOf, recursive `$ref`s, whitespace bounds, UTF-8, numbers; every keyword it does not take refused by its path); the tokens a state allows (`test_token_mask.cpp`) on a hand-written vocabulary and on a random one whose long walks are kept, and with `XS_QWEN35_DIR` what that costs a step on its vocabulary | every taken document parsed as JSON too; a brute-force check of each token's bytes from the same state |
| `tests/unit/server/` | request parsing (image and input_audio parts; `response_format`: text, json_object, json_schema, each refusal down to the schema's keyword, beside tools), response format, stop strings, response_format's constraint (`test_grammar_constraint.cpp`: a tokenizer's tokens as a grammar reads them, masked rows, drafts checked without committing, the reasoning before the answer and the dialect's opening), HTTP end to end (a format's constraint reaching the engine, also after a prompt that leaves the model thinking; `test_transcriptions_http.cpp`: the transcription form's fields, json / text / streamed events, every refusal); a realtime session (`test_realtime_session.cpp`: OpenAI's GA events in order, turns committed by the client and by the VAD, the audio and settings each turn hands the service, failures, every refusal as an error event naming its field) and over a WebSocket (`test_realtime_http.cpp`: the realtime subprotocol, the API key as a header or a subprotocol) | a fake `InferenceService` (`fake_service.h`) |
| `tests/unit/hub/test_hub_select.cpp` | which files of a repository `-hf` fetches, on listings shaped as real ones (a sentence-transformers checkpoint beside other runtimes' weights, Laya's variants in subfolders, unsloth's GGUF repositories with dynamic quantizations, split files, projectors, MTP files and a DSpark drafter's files, which are never taken for the model); specs and their refusals | hand-written expectations |
| `tests/unit/hub/test_hub_download.cpp` | the hub client against a Hub in the test process (`fake_hub.h`: listing API, resolve URLs redirecting to a second server standing for the CDN, ranges, tokens, gated repositories, faults): the cache layout huggingface_hub writes, nothing downloaded twice, offline and unreachable, a download resumed with the ranges it lacked, a corrupted byte caught and not kept, the token never sent to the CDN, refusals by name; a GGUF file (`qwen35`, `qwen35moe`) imported once with the MTP head published apart (or the one the user names, filed by the files), a file split into parts downloaded whole, a checkpoint given its companion file, refused before its weights without a token, a companion named for a checkpoint refused | the files served; convert's import of the same files |
| `tests/unit/hub/test_model_source.cpp` | a model by its path: a GGUF file imported into the cache once (by any spelling of its path), again in place when it or a companion changes; its companions found beside it as `-hf` takes them (a Qwen3.5 file's `MTP/mtp-*.gguf`, Inkling 2's drafter and bf16 projector), none for a drafter, the user's in their place; a split file by its first part, a later part refused, any part's change a change; a directory as it is; refusals by name | convert's import of the same files |
| `tests/unit/io/gguf_reader.cpp` | the GGUF reader: every metadata value type, nested arrays, a vocabulary-sized string array, tensors at the file's alignment, versions 2 and 3, values compared exactly (`same_as`); a model split into parts read as one file from its first part (gguf-split's parts, `GgufWriter::split`, with and without tensors in the first), part names; refusals (big-endian, a split model's later part given, a missing part, parts that disagree, tensor types it does not read, truncation, misalignment) | files written by `gguf_test_util.h` (which also encodes Q8_0 and Q4_0 blocks as ggml does, and splits a file into parts as gguf-split does) |
| `tests/unit/convert/gguf_split.cpp` | split GGUF files imported: the tiny qwen35 and qwen35moe GGUFs in parts (a first part of metadata alone or with tensors), and a model without its MTP head beside the head's file, both split | the pack of the whole files, every file byte for byte |
| `tests/unit/quant/test_block_decode.cpp` | every GGUF block format decoded on the host, on random and edge blocks; the block layouts and codebook tables; planted wrong layouts | ggml's own `dequantize_row_*`, bit for bit, in `tests/fixtures/quant/blocks/` (`tests/parity/quant/gen_block_cases.py`) |
| `tests/unit/quant/test_kv_formats.cpp` | the KV cache formats on the host: names and sizes; the TurboQuant formats' codes and norm of rows of every kind, their rotation, their error on Gaussian and on spiky rows; planted: codes read at another width, values left rotated | the reference implementation's blocks in `tests/fixtures/quant/turbo/` (`tests/parity/quant/gen_turbo_cases.py`), equal but where a coordinate sits on an edge; nearest centroids in fp64 |
| `tests/unit/quant/test_block_weights.cpp` | weights in GGUF block formats: rows read whole, in part and of 3-D banks, a pack's block format entries and their refusals, the sign sketch of one | values written from the formats' fields; the same weight decoded whole |
| `tests/unit/backend/cpu_block_ops.cpp` | MatMul, Gather and ExpertMatMul reading block format weights on cpu-ref | the same graph over the decoded f32 weights, bit for bit |
| `tests/unit/backend/sycl_block_layout.cpp` | the SYCL backend's rewrite of Q4_0, Q4_1 and Q8_0 for a pack's kernels, on the host (compiled with the backend): the streams take the file's bytes and hold its values | ggml's decode of the blocks, bit for bit |
| `tests/unit/backend/device_block_ops.cpp` | MatMul (every row count the GEMV's variants take, and past them through a GEMM or the many-row kernel, with a residual folded in; weights of Q4_0, Q4_1 and Q8_0 that read one x) Gather and ExpertMatMul (a GEMV per pair, pairs grouped by expert, passes of an expert's pairs at every size, grouped GEMMs; ids outside the bank) reading block format weights on the GPU, and the same with `block=f16`; timing by format with `XS_DEVICE_BENCH=1` (`device_block_timing.cpp`) | fp64 over the values the host decodes; blocks of random bytes |
| `tests/unit/convert/gguf_import.cpp` | GGUF import end to end: a tiny qwen35 GGUF written as llama.cpp's converter writes one (`tiny_gguf.h`: Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, tiled value heads, merged input projections, an MTP head, a byte-level BPE vocabulary): its weights, logits over prefill and decode steps, greedy tokens and drafts on cpu-ref, its tokenizer, chat template, stop token and sampling defaults; the same model as a file without the head and the head's own file in unsloth's layout (`--mtp`, `split_tiny_gguf`): the same pack byte for byte and the same chained drafts; refusals by name (both ways in) | the twin checkpoint its blocks decode to, in the Hugging Face layout; the original tokenizer; the import of the file that holds the head; planted: the pack read in the grouped head order, the head of another model |
| `tests/unit/convert/gguf_import_qwen35_vision.cpp` | GGUF import of a Qwen3.5-family vision tower: a tiny projector file (`tiny_qwen35_projector.h`, as llama.cpp's converter writes one) beside the tiny qwen35 GGUF: config.json in the release's layout, the processor's file, every tower tensor under the checkpoint's name with the file's bytes; the tower over the kernels a frame gives the rows of the whole kernel (planted: one frame's kernel left out); the projector found beside a file given by its path; refusals by name (another projector, deepstack layers, another model's tower, another normalization, an unknown key) | the file's bytes; the tower over the joined kernel |
| `tests/unit/convert/gguf_import_moe.cpp` | GGUF import of a Qwen3.5 mixture of experts end to end: a tiny qwen35moe GGUF written as llama.cpp's converter writes one (`tiny_gguf.h`: the router, 3-D gate, up and down banks in Q8_0 / Q4_0 / F16 / BF16 with gate and up in different formats in some blocks, the shared expert and its gate stored 1-D, tiled value heads, an MTP block with experts of its own): the pack's config, the banks apart with the file's bytes, its weights, logits over prefill and decode steps, greedy tokens, draft logits and chained drafts on cpu-ref; a file without the head and the head's own file (`--mtp`): the same pack byte for byte; refusals by name (unknown and missing keys, a bank of another shape, a dense model's tensors) | the twin checkpoint its blocks decode to, with fused banks; planted: gate and up banks under each other's name (the decoder's, the head's), the pack read in the grouped head order |
| `tests/unit/convert/gguf_real_file.cpp` | a real qwen35 GGUF (`XS_GGUF_FILE`) against its checkpoint (`XS_QWEN35_DIR`), or a real qwen35moe GGUF against its checkpoint's small files (`XS_QWEN35MOE_DIR`; config.json, the tokenizer and chat template when there): tokenizer parity (`parity_cases.h`) and token-by-token equality, the config fields the model code reads, the chat template renders, the import loads with its drafter and holds every block format tensor of the file byte for byte, a mixture of experts' banks apart | Hugging Face's tokenizer and config.json; the file's bytes |
| `tests/unit/convert/gguf_import_gemma4.cpp` | GGUF import of Gemma 4 end to end: a tiny gemma4 GGUF written as llama.cpp's converter writes one (`tiny_gemma4_gguf.h`, from the tiny Gemma 4 of `tiny_checkpoints.h`: per-layer embeddings, KV-shared layers with a double-wide MLP, per-layer KV heads, a fused `attn_qkv`, `rope_freqs`, Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, a Gemma-style vocabulary with the converter's token types): its weights, logits over prefill and decode steps (to the bit) and greedy tokens on cpu-ref, its tokenizer, chat template, stop tokens and sampling defaults; a tiny gemma4-assistant GGUF beside it: the drafter's weights, draft logits and chained drafts; refusals by name | the twin checkpoints their blocks decode to; the original tokenizer; planted: the full-attention RoPE read wrong (target and drafter) |
| `tests/unit/convert/gguf_import_modern_bert.cpp` | GGUF import of Laya and Julia-1 end to end: a tiny modern-bert GGUF of each written as llama.cpp's converter writes one (`tiny_modern_bert_gguf.h`: the head's blocks after the encoder's, the scorer's last row 1-D as llama-quantize leaves it, Q8_0 / F16 / BF16 / F32 tensors, Laya's temperatures, the prompt template, ModernBERT's and mmBERT's vocabularies in miniature): its configuration, tokenizer (token by token, the mask's lstrip, the post-processor), and through the decision runtime the same sequences and answers as its twin on cpu-ref; refusals by name | the twin checkpoint its blocks decode to; the original tokenizer |
| `tests/unit/convert/gguf_modern_bert_real_file.cpp` | a real modern-bert GGUF (`XS_GGUF_FILE`) against its checkpoint (`XS_LAYA_DIR` or `XS_JULIA_DIR`): tokenizer parity cases and equality over the whole vocabulary, the configuration the model reads, and the pack against the checkpoint on the decision parity requests: the same sequences and refusals, answers within the file's rounding | the checkpoint's tokenizer, configuration and answers |
| `tests/unit/convert/gguf_gemma4_real_file.cpp` | a real gemma4 GGUF (`XS_GGUF_FILE`) against its checkpoint's small files (`XS_GEMMA4_DIR`): tokenizer parity and equality over the whole vocabulary (text, special flag, encoding, decoding), the config fields and stop tokens the model code reads, the chat template (thinking, tools, a tool round trip, the checkpoint's renderings), the pack loads and answers on cpu-ref; a real gemma4-assistant GGUF (`XS_GGUF_DRAFTER_FILE`): its config against the assistant's (`XS_GEMMA4_ASSISTANT_DIR`), and beside the target it loads and drafts the target's next token | Hugging Face's tokenizer, config.json, generation_config.json and `tests/fixtures/chat/gemma4_cases.json`; the target's greedy tokens |
| `tests/unit/convert/gguf_import_clef.cpp` | GGUF import of Clef end to end: a tiny clef GGUF written as llama.cpp's converter writes one (`tiny_clef_gguf.h`: `tiny_gguf.h`'s decoder under `clef.*`, a joint schema head as `dec.blk.<i>` and `decision.*`, each attention's in_proj as three tensors, the scales computed from the parameters, the head's matrices Q8_0 / Q4_0 / BF16 / F16, `scorer_out` stored 1-D as llama-quantize writes it or [1, width] as the converter does): the pack's files, names, weights and joint_head_config.json, its answers on cpu-ref (and the GPU against cpu-ref); refusals by name | the tiny Clef checkpoint its blocks decode to, in the original layout; planted: query and key projections exchanged, the parameters taken for the scales |
| `tests/unit/convert/gguf_clef_real_file.cpp` | a real clef GGUF (`XS_GGUF_FILE`) against its checkpoint (`XS_CLEF_DIR`): the config fields the model code reads and joint_head_config.json, the head's tensors by name, shape and value (float ones exactly, block formats within their rounding, the scales from the parameters), the tokenizer token by token, the pack loads as a decision model | the checkpoint's config.json, joint_head_config.json, joint_head.safetensors and tokenizer.json |
| `tests/unit/test_fp8.cpp` | FP8 checkpoints: the E4M3 code table, codes paired with their scales (one per block, one per row), tensors joined row-wise (F8 runs of either layout; float tensors copied), rows read whole and in part, refusals; the E2M1 table, an MXFP8 weight (an exponent per 32 columns) and an NVFP4 one (two codes a byte, a scale per 16 columns, a factor per expert) paired by a loader | values written from the codes' fields and the blocks' or rows' indices; planted: swapped grid, row scales read as blocks, an exponent read as a scale, nibbles in the other order |
| `tests/unit/model/fp8_checkpoint.cpp` | a tiny Qwen3.5 (dense; mixture of experts stored expert by expert) written the way an FP8 release stores it (transformers' 128 x 128 blocks, or compressed-tensors' scale per channel): weights, logits (cpu-ref and GPU) and `convert`'s packs; banks stored expert by expert as bf16; `quantization_config` refusals of both schemes | the same checkpoint with the decoded values in the usual layout (`tiny_checkpoints.h`) |
| `tests/unit/model/gemma4_config.cpp`, `qwen3_5_config.cpp`, `qwen3_config.cpp`, `qwen3_omni_config.cpp`, `lfm3_config.cpp`, `inkling2_config.cpp`, `modern_bert_config.cpp` | config parsing and refusals (LFM3: also an `lfm3_vl` checkpoint's `text_config`; SenseNova Loop's run order and hyper-connections; Inkling 2's loop, its quantization_config, the order of a bank's rows and its drafter's `mtp_config`) | |
| `tests/unit/model/sensenova_loop.cpp` | SenseNova Loop on a tiny checkpoint the test writes (`tiny_checkpoints.h`, `TinyQwen::loop_count`): its runs and state slots, a prompt fed in pieces equal to the whole, the GPU against cpu-ref, a pack that keeps the hyper-connection routers, a block drafter refused | the same model fed differently; its unquantized checkpoint; planted: the same weights run once |
| `tests/unit/model/qwen3_5_value_heads.cpp` | linear attention's value heads in the tiled order: the same logits over a prefill and decode steps (`step_logits.h`), no ops added for a grouped checkpoint or when the orders coincide | the tiny checkpoint in the grouped order; planted: the tiled weights read as grouped |
| `tests/unit/model/qwen3_5_expert_banks.cpp` | a mixture of experts' gate and up banks stored apart (`experts.gate_proj`, `experts.up_proj`): the same logits over a prefill and decode steps on cpu-ref (and the GPU); a checkpoint with neither layout, both, one half or a bank of another shape refused by name | the tiny MoE checkpoint with fused banks; planted: the two halves under each other's name |
| `tests/unit/model/model_parity.cpp` | whole model: last-token logits and greedy continuations; its drafter (MTP head, or an assistant with `XS_TEST_DRAFTER_DIR`): logits and chained drafts after each prompt | Hugging Face fp32, from `tests/parity/model/hf_reference.py` (`--assistant` for a Gemma 4 assistant; a checkpoint that names its own modeling code, like LFM3, runs with it) |
| `tests/unit/model/kv_format_quality.cpp` | what a quantized KV cache changes in a model's predictions over a text (`XS_KV_QUALITY_TEXT`): perplexity, KL divergence, the most likely token, per format; a bar on the rise in perplexity | the same model over the float cache |
| `tests/unit/model/block_drafter_parity.cpp` | a block drafter (`XS_TEST_DRAFTER_DIR`) over rounds whose context grows by several tokens, at the block the references were written with (`--block-size`): drafts, candidates and scores | the drafters' own PyTorch code, from `tests/parity/model/block_drafter_reference.py` (clones of z-lab/dflash and Tencent/DSpine) |
| `tests/unit/model/dspark_parity.cpp` | Inkling 2's drafter (the checkpoint in `XS_TEST_MODEL_DIR`) through the engine's runner, round after round along the target's greedy continuation: the drafts, at f16 the ones that lead the runner-up clearly | the drafter's math in float64 over transformers' Inkling 2, from `tests/parity/model/dspark_reference.py` (no PyTorch module of it is public; `make_tiny.py --arch inkling2_loop` writes a tiny one beside the model) |
| `tests/unit/convert/gguf_inkling2_real_file.cpp` | a real inkling2 GGUF (`XS_GGUF_FILE`) against its checkpoint's small files (`XS_INKLING2_DIR`): tokenizer parity and equality over the whole vocabulary, the config the model code reads; with the drafter's file (`XS_GGUF_DRAFTER_FILE`) the pack's `mtp_config` and drafter tensors, and a model file refused as a drafter; with a copy of the file in mainline llama.cpp's layout (`XS_GGUF_MAINLINE_FILE`), the same pack tensors and values from both and each looped layer's message queries equal to the checkpoint's rows. The weights: `model_parity.cpp` on the imported pack | Hugging Face's tokenizer and config.json; the lossless file's pack against Hugging Face fp32 |
| `tests/unit/model/block_drafters.cpp` | block drafters on tiny checkpoints the test writes (`tiny_checkpoints.h`): a context fed in chunks or cut back after a verify, a block cut to the drafts asked for, layers of both kinds in one drafter, a drafter's pack, config refusals | the same drafter fed differently; its unquantized checkpoint |
| `tests/unit/model/policy_parity.cpp` | a robot policy end to end: prompt ids and the action chunk, all cameras and one camera (`XS_TEST_POLICY_DIR`, `XS_POLICY_FIXTURES`) | LeRobot fp32 with fixed noise, from `tests/parity/policy/lerobot_reference.py` |
| `tests/unit/backend/*_policy_ops.cpp` | LayerNorm, Concat (also of computed values that share a part), attention with a key bias over static queries (head dims 72, 256) and within blocks of tokens | fp64 references (`policy_ops_cases.h`) |
| `tests/unit/model/vision_parity.cpp` | image input (Gemma 4, Inkling 2): pixels, vision features, expanded prompt, logits, greedy (whole and chunked prefill), prefix reuse by image where a session can be cut back to a prefix | Hugging Face fp32, from `tests/parity/model/hf_vision_reference.py` (`make_tiny.py --arch inkling2_loop` writes a tiny Inkling 2 with its tower) |
| `tests/unit/model/decision_parity.cpp` | a decision model (`XS_TEST_DECISION_DIR`): each request's prompt tokens and every option's probability; a request the reference refuses is refused | the checkpoint's own Python code in fp32, from `tests/parity/decision/reference.py` over `requests.json` (and `requests_laya_julia.json`: the budgets, cuts, wordings and refusals of Laya and Julia-1) (`make_tiny.py` writes a tiny checkpoint in a released one's layout; d1's with its own classes, so the reference runs on it; d1's fixtures record the set size they were read at) |
| `tests/unit/model/decision_models.cpp` | decision models on tiny checkpoints the test writes (`decision_test_util.h`). Clef: a prompt fed in steps, a choice's options reordered, questions decided jointly, a request over the limit refused, the head's scales stored as a GGUF file stores them. pplx-decider: answers that depend on later questions, a noul's code order, a long request answered in parts, special tokens neutralized, refusals. Letter read-out (a tiny language model): each answer equal to its question asked alone, the state read once per request and not again by the next request about it, more options than letters, the calibration settings, refusals. d1 (a tiny LFM3-VL in d1-2B's layout, read with a byte tokenizer that has its chat markers, yes / no forms and some two-letter codes as words: `d1_tools`): the packed prompt as text and its groups, full-attention layers in groups and the sliding one causal, at a set size of 1 each answer its question's alone, packed ones hearing each other, a set size of 2 over three questions, a lone question the same at any set size, a noul's outcomes, codes of letter names and past 26 options down the fallback pool, special tokens in user text broken, a bucket's temperature, refusals (a pack over the limit named whole; a set size below 1, or for another model); its prompt pieces and config refusals on their own. Laya and Julia-1: each answer equal to its question asked alone, a noul's order, Laya's cuts (state, options, question; past the sequence refused), a request of single options refused and wording (json.dumps, str(), names), its temperatures by type and option count, Julia's strict refusals by name, the tokenizer read from its folder by the runtime, config refusals, no pack. All: the GPU against cpu-ref. Clef, pplx-decider and d1: a q8 pack from `convert` loads (without the head's file) and answers nearly alike | the same model run differently; its unquantized checkpoint |
| `tests/unit/model/py_text.cpp` | Python's text for a JSON value: `json.dumps` (key order kept or sorted, both separators, an indent, ASCII only or not), `str()`, float `repr` | what Python wrote |
| `tests/unit/decision/` | answers from probabilities: the API's confidence formulas, each question type | the API's documented example and formulas |
| `tests/unit/server/test_systemone_http.cpp` | the System One API: the schema refused field by field (422 bodies), answers of each type, 400 / 500, concurrent clients | a fake decision function |
| `tests/unit/model/embedding_config.cpp` | BERT's and NomicBERT's config.json, the sentence-transformers pipeline (both pooling spellings, Normalize, the longest input, `do_lower_case`), the model kind; every refusal by name | hand-written expectations |
| `tests/unit/model/embedding_models.cpp` | encoders on tiny checkpoints the test writes (`embedding_test_util.h`): runs that keep nothing, every token seeing every other, BERT's absolute and NomicBERT's relative positions; the GPU against cpu-ref | the same model run differently |
| `tests/unit/model/embedding_parity.cpp` | an embedding model (`XS_TEST_EMBEDDING_DIR`): hidden states, pooled vector and embedding of each case's ids, up to the model's longest input; with `XS_EMBEDDING_TEXTS=1` each text's ids too | sentence-transformers fp32, from `tests/parity/embedding/reference.py` (`make_tiny.py --arch bert|nomic_bert` writes tiny checkpoints) |
| `tests/unit/embedding/` | pooling; the runner (a run per sequence, the limits it refuses, `--ctx` below the model's); the service (text through a tokenizer, clients on several threads) | values computed by hand; the runner on its own |
| `tests/unit/server/test_embeddings_http.cpp` | the embeddings API: every input form, each 400 naming its field, float and base64 (decoded), models / health / no chat routes, concurrent clients, a tiny model and its tokenizer end to end | a fake embedding function; the runner |
| `tests/e2e/` | a running chat server: 23 tool-calling cases with simulated tools, one turn and several (`tool_calls.py`, graded against `tool_calls_rubric.md`; an original checkpoint and its GGUF import compared call by call); a fact planted at several depths of texts of several lengths and asked for (`needle.py`: what a KV cache format or a context limit changes at length). A running decision server: 25 decisions with one accepted answer each (`decisions25.json`), the API's shape, latency (`systemone_eval.py`, also against reference answers or another run's saved ones: what a pack or a setting changes; a request the model refuses must be one the reference refuses); `make_mmlu_pro.py` writes a larger, harder set from MMLU-Pro (not kept here) | accepted answers; the model's reference code; another run |

- `tests/parity/**/*.py` generate fixtures from Hugging Face (run with a Python venv that has `transformers`,
  `torch`). Small fixtures are committed; large ones go to `tests/fixtures/generated/` (gitignored).
  `tests/parity/quant/gen_block_cases.py` builds ggml's own dequantization (`ggml_dequantize.c` with the
  `ggml-quants.c` of a llama.cpp checkout) to write the block format fixtures.
  `tests/parity/quant/gen_turbo_cases.py` builds the TurboQuant reference quantizer (`turbo_quantize.c` with the
  `ggml-turbo-quant.c` of a TheTom/llama-cpp-turboquant checkout) to write the KV format fixtures.
  `tests/parity/model/to_fp8.py` rewrites a Qwen3.5 checkpoint the way an FP8 release stores it and writes the model
  those bytes decode to: reference outputs of the second are the fixtures for the parity tests on the first.
- Parity bar for the whole model: cosine >= 0.9999 (f32) / 0.995 (f16) on last-token logits, same argmax, greedy
  continuations equal for >= 32 (f32) / >= 8 (f16) tokens.

## 8. Recipes

### 8.1 Add a model architecture

1. Read the Hugging Face `modeling_<arch>.py` and the checkpoint's `config.json` and safetensors names.
2. Create `src/models/<arch>/` with:
   - `<arch>.h`: a `Config` struct, `parse_config(json)`, `load(dir, json)`.
   - `<arch>_config.cpp`: parse with `models/config_util.h` readers; **refuse every variant you do not implement by
     name** (`fail("<arch>: <feature> is not supported yet")`), never serve it silently wrong.
   - `<arch>_model.cpp`: a class implementing `Model`. Build the graph once in the constructor with `graph::Graph`
     (inputs, weights by safetensors name, ops, `output("logits", ...)`), wrap layers in `Graph::Scope` for readable
     node names, and fill every graph input in `prepare_inputs`. The last-token logits output must be named `logits`;
     beside it, output `argmax` (`blocks::argmax_rows(g, logits)`): greedy decoding then reads a token per row instead
     of the logits (the engine asks a run for the outputs it reads, and the backend skips the rest).
     Build layers from `models/common/blocks.h` (`attention`, `gated_mlp`, `moe`, `shortlist_head`, `rms_norm_one_plus`,
     `gated_rms_norm`, `rope_inv_freq`): the model maps its weight names onto a block's projections and norms; the math lives once.
     `models/lfm3/` is the shortest example of a whole model.
3. Register the `model_type` in `src/models/registry.cpp` (`load_model` and `supported_model_types`).
4. If the model needs an op the IR does not have, add the op first (8.2).
5. Tests: a config test (`tests/unit/model/<arch>_config.cpp`) and a parity test against Hugging Face fp32 modeled
   on `model_parity.cpp` + `tests/parity/model/hf_reference.py` (both are model-agnostic).
6. Images (optional): implement `VisionEncoder` (see `gemma4_vision.cpp`): `prepare` reproduces the Hugging Face image
   processor on an `image::Image`, `graph()` is the tower ending in an `embeddings` output, and `token_run` lays out
   one image in the prompt. Return it from `Model::vision()` and add the `media` / `media_keep` inputs to the text
   graph. Parity test modeled on `vision_parity.cpp` + `hf_vision_reference.py`.
7. Chat: if the model's output format needs parsing (thinking markers, tool-call syntax), add a dialect in `src/chat/`
   and set `ModelInfo::dialect`.
8. Quantized packs need nothing more: `convert` reads the new model's graph to find its matmul weights and gather
   tables, and the loaded pack hands the same graph Q8 / Q4 weights. Only weights the model reads on the host (like
   Qwen3.5's norm weights or Gemma 4's per-layer embedding table) must stay out of matmuls and gathers, which they
   already are. Check quality with the parity test on a pack (`XS_TEST_MODEL_DIR=<pack>`).
9. Update TODO.md, this file's map, and for a change users see README (what is supported) and model_recipes.md (the
   model's commands and what was measured).

### 8.2 Add a graph op

Touch, in this order:

1. `src/graph/graph.h`: `OpKind` entry with its exact semantics in the comment; attrs struct if any; builder method.
2. `src/graph/graph.cpp`: `op_name`, builder with operand checks and shape inference; extend `tests/unit/test_graph.cpp`.
3. `tests/unit/backend/reference_ops.h`: an fp64 reference.
4. `src/backend/cpu_ref/ops.{h,cpp}` + the switch in `cpu_backend.cpp`; test in `tests/unit/backend/cpu_ops.cpp`.
5. `src/backend/sycl/`: a kernel in `sycl_kernels.cpp` and a case in `SyclSession::launch` (operand dtypes in
   `check_ops`); test in `tests/unit/backend/device_ops.cpp`, run with `XS_DEVICE_TEST_DEVICE=GPU.0` at f16 and f32.

### 8.3 Add a backend option

`SessionOptions::extra` carries `--backend-opt key=value`. Each backend parses its own keys and refuses unknown ones by
name (the option loop in `SyclBackend::compile`, `sycl_backend.cpp`).

### 8.4 Add an API field

Parse and validate it in `src/server/openai_request.cpp` (a bad value is a 400 naming the field), carry it in
`GenerateRequest` or the handler, format it in `openai_format.cpp`, and test it in `tests/unit/server/`.

### 8.5 Give a model a drafter

The model implements `Speculation` (`models/model.h`): prefill and verify programs from its own layer builders, a
draft program, and one way of drafting. The engine needs nothing new.

- A drafter of the model's own kind (an MTP head, an assistant): `ChainDrafting`, with the target's post-norm state
  as the `hidden` output of prefill and verify (see `qwen3_5_model.cpp`, `gemma4_model.cpp`).
- A block drafter of the DFlash family: load its config and weights (`dflash::parse_config`, `open_weights`,
  `JoinedWeights` under a prefix), call `Drafter::append_context` in the prefill and verify builders with the states
  after the layers it names, and `build_block` with the model's embedding table and head (`qwen3_5_model.cpp`).
- A block drafter that also reads rows of the target's last token on its own (DSpark's head): the prefill and verify
  programs output them, `BlockDrafting::target_rows` names them, and the engine's runner hands the last kept row to
  the draft program as an input of the same name (`inkling2_model.cpp`, `drafters/dspark.cpp`).
- Tests: the engine's equivalence tests cover every drafter; add the drafter's parity against its reference
  (`model_parity.cpp` or `block_drafter_parity.cpp`, fixtures from `tests/parity/model/`).

### 8.6 Add a decision model

1. Read the checkpoint's reference code: how a request becomes a prompt (to the token), what is read off the decoder
   and where, how scores become probabilities.
2. `src/models/<arch>/`: a class implementing `DecisionModel` (`models/decision.h`). Build its programs once, from
   the decoder's builders (`qwen3_5::Decoder` for a Qwen3.5 decoder, `lfm3::Decoder` for LFM3's) plus the read-out.
   `start()` lays out the prompt
   with `PromptTools` (the tokenizer and chat template) and returns a `DecisionRun` whose `next()` names each step:
   the program, its inputs, the outputs to bring back, and how much of the sequence's state to keep. Rows whose
   count varies per request (options, questions) are a program's `kT`; what has a per-request shape beyond that is
   host work between steps. Refuse what the model cannot take with `DecisionRefused` (the client sees the message).
   `models/clef/` (a head of several programs), `models/pplx_decider/` (one program), `models/d1/` (one program,
   the output layer's rows of the answer tokens read on the host), `models/laya/` (one stateless program, a sequence
   per question, a tokenizer in a folder and no chat template: `DecisionSpec`) and `models/letter_readout/` (a
   language model's own step graph, a prompt per question) are the examples.
3. Write prompts exactly as the reference does (`models/common/py_text.h` for JSON the Python way), and give
   probabilities in `DecisionQuestion::options` order whatever order the prompt uses.
4. Register it in `registry.cpp` (`model_kind`, `load_decision_model`, `supported_decision_models`).
5. Tests: `tests/parity/decision/reference.py` learns to run the checkpoint's code, `make_tiny.py` to write a tiny
   one; `decision_parity.cpp` then checks tokens and probabilities, on cpu-ref and the GPU; invariants on a tiny
   checkpoint the test writes go in `decision_models.cpp`. End to end: `tests/e2e/systemone_eval.py --reference`.
6. Update TODO.md, this file's map, the README's model list and the model's recipe in model_recipes.md.

### 8.7 Add an embedding model

1. An encoder in `src/models/<arch>/`: its config (refusals by name) and a graph from `models/common/encoder.h` (BERT's
   post-norm layer, attention over the whole sequence) and `blocks`, made into a model with `encoder::embedding_spec`
   and `encoder::make_model`; register the `model_type` in `registry.cpp` (`supported_embedding_models`,
   `load_embedding_model`).
2. A pipeline beyond Transformer, Pooling (mean, CLS) and Normalize (another pooling mode, a Dense module, a prompt)
   is refused today: teach `models/common/sentence_transformers` and `embedding/pooling` first.
3. Tests: `make_tiny.py` learns to write a tiny one and `tests/parity/embedding/reference.py` runs it;
   `embedding_parity.cpp` then compares hidden states and embeddings; refusals in `embedding_config.cpp`.
4. Update TODO.md, this file's map, the README's model list and the model's recipe in model_recipes.md.

### 8.8 Add a GGUF architecture

`xpu-serve convert <file.gguf> <out-dir>` imports the file into a pack for a model the engine already serves from
Hugging Face safetensors (8.1 first, if it does not).

1. Read llama.cpp's converter for the architecture (`convert_hf_to_gguf.py` and the `conversion/` class registered for
   the Hugging Face architecture) and its graph (`src/models/<arch>.cpp`): the metadata keys it writes, the tensor
   names, and every transform it applies to a tensor (a reshape, a value transform, a reordering of rows or columns).
2. Add `src/convert/gguf/<arch>.cpp`: a class implementing `gguf_import::Architecture` (`architecture.h`) and a
   function returning it, listed in `find_architecture` (`gguf_import.cpp`). `plan()` refuses `<arch>.*` metadata keys
   it does not know (`refuse_unknown_keys`), writes the config.json the model's config parser reads (the Hugging Face
   fields, numbers stored as f32 through `config_number`), and describes every tensor of the file as a `layout.h`
   entry: the shape the metadata implies and the pack tensors it holds; `map_tensors` then refuses an unknown tensor
   name (never dropped), a wrong shape, a missing or doubled pack tensor. A tensor the converter computed from the
   config (Gemma 4's `rope_freqs`) is an entry without pieces, checked against the config the import writes.
   `gemma4.cpp` is the short example, `qwen35.cpp` the one with transforms. If the architecture's MTP heads are
   also published in files of their own (llama.cpp's draft models), override `plan_with_mtp` as `qwen35.cpp` does:
   check the second file's metadata against the model file's, add its head's tensors to the model's before
   `map_tensors` (the same layout as a file that holds the head), and put its copies of the model's tensors in
   `ImportPlan::copies`; the default refuses `--mtp` by name. An architecture that puts a head on a decoder the
   importer already reads (`clef` on Qwen3.5's) reads the decoder under its own namespace (`read_qwen35_decoder`,
   naming its own keys), adds its head's entries to the layout, and writes the head's config as one more JSON file of
   the pack (`ImportPlan::files`), as `clef.cpp` does. Architectures whose files differ in one part of a model
   (`qwen35` and `qwen35moe`: the layers' MLP) are one class and one reader told which part it reads (`Qwen35Mlp`),
   each with the keys and tensors of its own part and no other's.
3. Transforms: a float tensor can go through a `FloatMap` (value by value, written as F32). A tensor in a block format
   is never decoded, re-encoded or reordered (a K-quant block spans 256 columns): it can only be cut into runs of whole
   rows. When the converter reordered a quantized weight's rows or columns, the model code computes in the file's
   order instead, behind a config.json key the importer writes (as `linear_value_head_order` for Qwen3.5); add that
   key to the model's config parser and graph first, with a test that a reordered checkpoint gives the same logits.
   When the converter stored a value computed from parameters (Clef's `decision.scales`), the model reads the stored
   values under a name of their own rather than the import computing the parameters back, which rounds again.
   When the converter stored one tensor of the checkpoint as several that are not runs of its rows, or that can
   differ in format (a mixture of experts' `gate_up_proj` as a gate bank and an up bank), the pack keeps them as the
   file has them, under names of their own, and the model code reads either layout (`qwen3_5::Decoder::moe`), with a
   test that the two give the same logits.
4. The tokenizer: a `tokenizer.ggml.model` (or, for gpt2, a `tokenizer.ggml.pre`) the table in
   `convert/gguf/tokenizer.cpp` does not have needs an entry with the original tokenizer.json's normalizer,
   pre-tokenizer, post-processor, decoder and BPE settings, where the converter's token types depart from the
   original's special flags (read the converter's `set_vocab`), and the end-of-turn tokens llama.cpp stops at by text
   (`llama-vocab.cpp`) that the file's eos_token_id does not name.
5. Tests: a tiny GGUF of the architecture written as the converter writes one, imported and compared with the
   checkpoint its blocks decode to (modeled on `tests/unit/convert/gguf_import.cpp`), with a planted control for every
   reordering and for every pair of tensors a name map could exchange; real-file checks behind an environment
   variable (`gguf_real_file.cpp`).
6. Update this file (map, tests, section 10), the README's GGUF line and the GGUF recipes in model_recipes.md.

## 9. Conventions

- **Correctness first, measured against a reference.** Every op has an fp64 reference and every model a Hugging Face
  fp32 parity test. A speed change must keep parity.
- **Refuse by name, never degrade silently**: unknown config keys that change the math, unsupported request fields,
  unknown backend options all fail with a message naming the thing.
- Errors: throw `xs::Error` via `fail(...)` / `XS_CHECK(cond, ...)` with a message that says what and where.
- Style: `.clang-format` (4-space indent, 120 columns). Comments say what the code cannot: a contract, a unit, a
  non-obvious why. No history or measurements in comments; those go in commit messages and `docs/decisions/`.
- Dependencies are vendored in `third_party/` with their license and provenance (`third_party/README.md`).
- Decisions that DESIGN.md leaves open get a short record in `docs/decisions/`.
- The repository is public: no credentials, internal hostnames or IP addresses, customer information or
  confidential numbers in code, docs or commit messages.
- Commits: one topic each, imperative subject line, body says why.

## 10. Known limitations

- Inkling 2 Mini (`inkling2_loop`) reads images from its checkpoint, a pack of it, or a GGUF import with the
  repository's projector file (`convert --mmproj`, fetched by `-hf`; without it the import has no vision tower and
  refuses an image). An image is not resized (one token per 40 x
  40 pixels, so a large one is many tokens), and a transparent one is read without its alpha, where the reference
  lays it on a checkerboard. Its released
  checkpoint (experts in NVFP4 and MXFP8) is the float model its codes decode to, which a backend holds as floats:
  far more than the download, so it is served from a pack `convert` makes of it, or from a GGUF file's import, which
  keeps the experts in NVFP4. The variable-pass research checkpoint (`is_varloop`) is refused by name. A decode step
  is bound by the device, not by its launches: two thirds of it reads the 2.8 GB of weights a token goes through, the
  rest is some 3,400 small kernels of 1 to 3 us each (48 layer applications; docs/decisions/0026). Its drafter
  (DSpark) gains little: every row of a verify reads the experts of its own token, so each draft adds about a fifth
  of a decode step to the verify, and two drafts a round are the default. Its GGUF files import in Unsloth's layout
  (`inkling2`, with the drafter's file) and in mainline llama.cpp's (`inkling2_loop`: AtomicChat's files, Unsloth's
  `Mainline_Support` copies); a LoRA adapter file (`general.type` adapter) is refused by name.
- Three model families: Gemma 4 E2B/E4B (text and images) and 26B-A4B (text), Qwen3.5 dense and MoE (text and
  images: the vision tower is not loaded beside a block drafter, which was trained on text, and its MTP head drafts
  at one position a token, the cache's, where the target reads an image's positions on three axes; SenseNova Loop's
  tower is not loaded, and a block drafter beside SenseNova Loop is refused; a MoE
  checkpoint's expert banks come fused, as the released ones store them, expert by expert as an FP8 release does, or
  with gate and up apart as a GGUF import keeps them), and LFM3 (text:
  `lfm3`; the vision and audio variants are other model types and are refused, but for `lfm3_vl`'s text decoder
  under d1). Audio for Qwen3-ASR (speech recognition: WAV, MP3, FLAC; `/v1/audio/transcriptions` gives json or
  text, no timestamps yet: the forced aligner is refused; `/v1/realtime` takes 16-bit PCM at 24 kHz only, its VAD is
  frame energy, not a model, and a turn's text comes once the turn ends) and Qwen3-Omni's thinker (audio and images in
  chat; images resized within the processor's pixel bounds, which `--image-max-pixels` lowers; video and its speech
  out, the talker, not yet). No video.
  Images only as base64 data: URLs.
- Speculative decoding drafts greedily (rejection sampling keeps sampled output exact). Drafters: the MTP heads, the
  Gemma 4 assistants, block drafters for Qwen3.5-family targets (checkpoints exist for Qwen3.8-27B and
  Ornith-1.5-35B-A3B), and Inkling 2's own (DSpark, whose head reads the whole output layer for each draft). An MTP
  head picks each draft among 64 tokens (`--draft-shortlist`; a one-bit sketch of the output layer chooses them, the
  exact layer ranks them): a pick the shortlist misses costs a draft. A verify of more than 8
  tokens leaves the fast path (a block drafter drafts 7 by default).
- Models with conv or recurrent state (Qwen3.5's linear attention, LFM3's convolutions) keep a cached prefix whole,
  or from the sequence's snapshot of its state where the last chat's messages ended (decision 0029: one snapshot a
  sequence, in host memory; Qwen3.8-27B's is 149 MiB); a prompt departing from the cache before that point (an edited
  older message) prefills from the start. A request takes another sequence's prefix as a copy when its own offers
  64 tokens less; with conv or recurrent state only up to that sequence's snapshot, so two users sharing only a system
  prompt prefill it each.
- An FP8 checkpoint (`quantization_config`: fp8, e4m3, one scale per 128 x 128 block; or compressed-tensors'
  float-quantized format, 8-bit float weights with a scale per channel and dynamic activations) is the float model its
  codes decode to: weights go to the device as f16, like the original's, and activations are not rounded to FP8 as the
  reference runtime does. It is half the download; `convert` makes the pack that is smaller and faster to serve.
  Other quantized checkpoints (bitsandbytes, AWQ, GPTQ) are refused by name.
- GGUF files are imported, not served as they are: `serve <file.gguf>` (and `run`, `bench`) imports the file into a
  pack in xpu-serve's cache, again when its size or modification time changes, and `convert <file.gguf> <out-dir>`
  into a directory; the pack's tensors keep the file's block formats byte for byte, so a file takes its size twice on
  disk. The files beside it are read as its repository: a companion its import reads that the user does not name
  (`--mtp`, `--mmproj`) comes from there, so a folder that mixes models' drafters can hand a file another model's,
  which its import refuses by name (docs/decisions/0031). Architectures `qwen35` and `qwen35moe` (the Qwen3.5
  family's text models, dense and mixture of experts) and `gemma4`
  (Gemma 4 E2B / E4B, the text model: llama.cpp keeps the vision and audio towers in a separate mmproj file, which is
  not read, so images need the original checkpoint), and `gemma4-assistant`, its drafter (`--drafter <pack> --spec
  mtp`): the file has no centroid head, so the imported assistant scores the whole vocabulary, as llama.cpp does,
  where the original picks among its top centroids' tokens (drafts can differ; the output is the target's either
  way). A `qwen35` file without the MTP head (unsloth's smallest Qwen3.8-27B files), or a `qwen35moe` one, imports
  with the head's own file,
  llama.cpp's draft model (`convert <file.gguf> <out-dir> --mtp <mtp.gguf>`), into the pack a file that held the head
  gives: the draft model's own token_embd, output and output_norm (low-bit copies of the model's) are checked against
  the model's shapes and left out, since the head reads the model's embedding and lm_head; its metadata must equal
  the model file's but for the head's block. A `qwen35moe` pack keeps each layer's gate and up expert banks apart, as
  the file stores them and each in its own format, where the checkpoint holds one fused bank (the model code reads
  either); a file whose banks the converter fused (`--fuse-gate-up-exps`), or one with linear attention's input
  projections merged as a `qwen35` file can have them, is refused by name. `clef` (Clef-Flash and Clef, decision
  models) imports into the pack
  `convert` makes of the checkpoint, but for the head's three scalars, kept as the file stores them (the values they
  give, `joint_head.scales`); llama.cpp's `systemone` prompt template is not imported (the model code writes Clef's
  prompt) and its mmproj file (the vision tower) is not read. `modern-bert` files of the decision models Laya and
  Julia-1 (ggml-org's; `decision.type` laya) import into the original checkpoint's layout; any other `modern-bert` file
  (an encoder alone, another decision type or prompt template) is refused, as are Laya's files with a question budget
  no published Laya has (the file holds no state budget, which is taken from the published checkpoint with that
  question budget). A Julia-1 file's max_head_tokens is the converter's default (256); the pack takes Julia's own
  inference-policy.json (512, strict encoding), which accepts more and encodes the same. `inkling2` and `inkling2_loop`
  (Inkling 2 Mini, Unsloth's layout and mainline llama.cpp's) import with their experts in the file's formats and,
  with `--mtp`, the drafter's file (`dflash` with `dflash.dspark.markov_rank`; any other `dflash` file is refused).
  A Gemma 4 with experts (26B-A4B) and every other architecture are refused by name. Tokenizers of the importer's table: byte-level BPE
  (`tokenizer.ggml.model` gpt2) with `tokenizer.ggml.pre` `qwen35`, `inkling2`, `modern-bert` and `mmbert`, and
  `gemma4`. The pack's chat template and stop tokens are the file's: a publisher's changed template is what it
  renders, and a stop token only the original checkpoint's config names (Qwen3.5's `<|endoftext|>`) is not one; for
  `gemma4` the import adds the end-of-turn tokens llama.cpp stops at (`<eos>`, `<turn|>`, `<|tool_response>`, the
  original's), and the tokenizer_config.json eos_token is the file's (unsloth's `<turn|>`, Google's `<eos>`). A model
  split into parts is read from its first part, the others beside it (docs/decisions/0030); a later part given in its
  place, a missing part, parts that disagree, and a file that holds tensor types beyond F32, F16, BF16 and the block
  formats are refused by name.
- A pack can hold weights in GGUF's block formats (`DType::Block`, kept as the file stores them): the host decodes all
  22 as ggml does (its own 19, and NVFP4, MXFP4 and Q2_0 of Unsloth's llama.cpp builds), so cpu-ref runs them. The
  SYCL backend has kernels for all of them, expert banks included (up to 64 (token, expert) pairs a GEMV per expert
  over its pairs on the matrix engines, or a GEMV per pair; more grouped by expert: on the matrix engines in passes
  of up to 64 of an expert's pairs that read its weights in place, without them a GEMV and from 256 pairs each
  expert dequantized for a grouped GEMM); `--backend-opt block=f16` decodes block formats to f16 as it uploads them. Up to 8 rows a block
  format's matmul is a GEMV, from 2 rows its matrix-engine version (x read in f16, a cost that hardly grows with the
  rows: a speculative verify), from 9 to 64 rows that kernel over tiles of 8 rows (the weight read once per 32 rows,
  never written out), past them the weight is dequantized for a GEMM. Q4_0,
  Q4_1 and Q8_0 are read by a pack's q4 / q8 kernels, each in its file's bytes.
- Quantization is round-to-nearest only (no AWQ / GPTQ yet). Gemma 4's per-layer embedding table (read on the host)
  and vision towers stay bf16 in a pack. Prefill dequantizes each quantized weight to f16 for oneMKL's GEMM, except
  on integrated GPUs up to 512 rows (quant_gemm / quant_gemm_tiled dequantize on the way; sycl_be::quant_path).
- Decision models: text states only (`assets`, `images` and box questions are refused by name); a decision's
  latency is the prefill of its prompt, so a pack of one is smaller but slower on a discrete GPU, and q4 changes
  answers the model is unsure of (docs/decisions/0015); `convert` rounds to nearest for them (no AWQ).
  pplx-decider reads a prompt in one step (its attention looks ahead), which costs more per token past a few
  thousand tokens. A language model read with `--readout` serves decisions only (no chat on the same server).
  d1 packs up to 16 questions after one state as its own code does (`--set-size` changes it); a pack longer than
  `--ctx` is refused with its token count, where the reference would read it (docs/decisions/0019).
  d1's weights, tokenizer and chat template are under the LFM Open License v1.0 (commercial use only below $10M
  annual revenue); none of them is in this repository.
  Laya and Julia-1 (docs/decisions/0018) read a sequence per question, the state again for each; Laya cuts what does
  not fit as its reference does (a state past 512 tokens, logged at info), Julia-1 refuses what its strict encoding
  cannot read whole; neither has a q8 / q4 pack (`convert` refuses one by name; their GGUF files import), and
  neither was measured on the GPU yet.
- Embedding models: BERT and NomicBERT encoders of sentence-transformers checkpoints, mean or CLS pooling; a request's
  inputs run one at a time (no batching); an input over the limit is refused, not cut; `dimensions` only as the
  model's own; no rerank or score API; no packs (`convert` refuses them).
- A quantized KV cache (`--kv-dtype`): the TurboQuant formats need a head_dim that is a multiple of 128 (a model
  with another is refused by name); keys and values take one format, or `i8` keys with TurboQuant values, other
  pairs are refused. Prefill decodes the keys and values in view of one layer to f16 for the matrix paths: on the
  fused kernel past 65,536 of them a chunk of that many at a time (`--backend-opt kv_chunk=N`, 0 for whole), so
  the scratch stays a chunk's; on the GEMM path (a head_dim past 128) the layer's whole cache. Lossy and opt-in
  (docs/decisions/0010, 0020).
- `-hf` reads the Hugging Face Hub's API and resolve URLs: a GGUF file is imported into a pack, so it takes its size
  twice on disk; a file split into parts is downloaded part by part, every part (one the repository lacks is refused
  by name); no Xet chunk deduplication. The MTP
  head a repository publishes apart is fetched for a `qwen35` or `qwen35moe` file without one, and Inkling 2's
  drafter (`dspark-*.gguf`) and vision tower (`mmproj-*.gguf`, the bf16 one) for an `inkling2` or `inkling2_loop`
  file (the drafter when the repository has one).
- Requests in flight (`--sequences N`, each sequence its own KV cache, billed at load) decode a step together;
  speculating, their verifies run as one step, sharing 8 rows (fewer drafts as more decode), while each drafts on its
  own; prefill chunks run a request at a time (docs/decisions/0028).
- Constrained decoding (`response_format`, docs/decisions/0033) takes JSON in OpenAI's strict subset of JSON Schema;
  an object's properties come in the schema's order, and an object with properties takes no others even where
  `additionalProperties` is absent. Not beside `tools`, not on Anthropic's API, no other grammars at the API. The
  reasoning comes first only where the prompt leaves the model inside it (Qwen3.5 to 3.8 with thinking on); Gemma 4
  and Inkling 2, which open their thoughts themselves, answer at once. A constrained request reads its rows' logits
  (the whole vocabulary a row) instead of their argmax, and its masks are made on the host between steps: the first
  token of a string walks the vocabulary's trie (milliseconds), later ones read a kept mask. The drafter is not
  constrained: its drafts from the first the grammar refuses are not verified. The dialect's parser still reads its
  markers inside the JSON's strings (a `<think>` spelled in plain tokens), and `max_tokens` can cut the JSON short.
- Sliding-window layers keep every key in memory in a session that can be cut back to any prefix (Gemma 4, a DFlash
  drafter's local layers): the kernels read only the window, memory still grows with the context. With conv or
  recurrent state beside them (LFM3, Inkling 2) a windowed slot keeps its window alone, on the SYCL backend.
