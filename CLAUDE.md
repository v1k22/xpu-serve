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
Qwen3.6, Qwen3.8; text; a hybrid of Gated DeltaNet linear attention and full attention, dense or mixture of experts)
and LFM3 (text; a hybrid of gated short convolutions and full or sliding-window attention),
and decision models behind the System One API (`POST /v1/systemone`: Clef-Flash, a Qwen3.5 decoder with a joint
schema head; pplx-decider-flash, one whose full-attention layers see the whole prompt, with a read-out per question;
OpenJev, a language model read at the option letters with `--readout`; docs/decisions/0015),
served end to end on the SYCL backend (Arc Pro B60), matching Hugging Face fp32 logits and greedy output; q8 / q4 packs
from `convert`; speculative decoding with the model's drafter (`--spec mtp`: Qwen's MTP head, a Gemma 4 assistant
with `--drafter`; `--spec dflash|dspine`: a block drafter checkpoint for Qwen3.8-27B with `--drafter`;
docs/design-spec-decode.md). One sequence at a time. Embedding models behind `POST /v1/embeddings` (all-MiniLM-L6-v2
and bge-small-en-v1.5: BERT; nomic-embed-text-v1.5: NomicBERT; docs/decisions/0017), checked against
sentence-transformers on cpu-ref.

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
build/src/xpu-serve bench <hf-model-dir> --device GPU --prompt-tokens 1024 --gen-tokens 128
build/src/xpu-serve run   <decision-model-dir> --device GPU --request request.json   # a System One request's answers
build/src/xpu-serve run   <embedding-model-dir> --device GPU --prompt "text"        # its embedding, a JSON array
```

- icpx turns the SYCL backend on (`XS_WITH_SYCL`); `-DXS_SYCL_TARGETS=intel_gpu_bmg_g21` compiles its kernels ahead
  of time for the Arc Pro B60. Any other C++20 compiler builds only the CPU reference backend (the test oracle).
- Useful flags: `--backend sycl|cpu-ref` (default: sycl when built), `--device GPU|GPU.0|GPU.1`, `--precision f16|f32`,
  `--kv-dtype auto|i8`, `--ctx N`, `--prefill-chunk N`, `--backend-opt attention=auto|rows|gemm`,
  `--backend-opt gemv=auto|vector|xmx`, `--backend-opt few_rows=N`, `--backend-opt block=auto|f16`,
  `--spec mtp|dflash|dspine --spec-k N`,
  `--readout letters|openjev` (a language model served as a decision model),
  `--drafter <dir>`,
  `--draft-shortlist N`, `--no-vision`,
  `--log-level debug`. `xpu-serve <command> --help` lists all of them;
  `src/cli/main.cpp` is the source of truth.
- Run one test case: `build/tests/xs_tests -tc='model matches*'`. Environment for the tests that need files:
  `XS_TEST_MODEL_DIR` (a Hugging Face model directory), `XS_PARITY_BACKEND` (`cpu-ref` or `sycl`),
  `XS_PARITY_DEVICE`, `XS_PARITY_PRECISION`, `XS_PARITY_FIXTURES`, `XS_PARITY_KV_DTYPE`, `XS_TEST_DRAFTER_DIR` (a
  drafter checkpoint beside the model); `XS_QWEN35_DIR` (a Qwen3.5 checkpoint, for tokenizer parity; with `XS_GGUF_FILE`, a qwen35 GGUF of it, for the GGUF import's real-file checks); `XS_GEMMA4_DIR` (with `XS_GGUF_FILE`, a gemma4 GGUF: the small files of the checkpoint it was converted from); `XS_GGUF_DRAFTER_FILE` (a gemma4-assistant GGUF) with `XS_GEMMA4_ASSISTANT_DIR` (the assistant checkpoint's config.json); `XS_CLEF_DIR` (with `XS_GGUF_FILE`, a clef GGUF: the Clef checkpoint it was converted from, its config.json and joint_head_config.json, and joint_head.safetensors and tokenizer.json when there); `XS_LFM3_DIR`
  (an LFM3 checkpoint, for its tokenizer and chat template, which are not kept in this repository);
  `XS_WORDPIECE_DIRS` (WordPiece checkpoints for tokenizer parity, separated by `:`, each directory named as its
  repository: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5`); `XS_LAYA_DIR` and `XS_JULIA_DIR`
  (the convaiinnovations/laya and SupersonicLabs/Julia-1 checkpoints, for the tokenizers in their `tokenizer/`
  folders: Laya's three, Julia-1's one);
  `XS_TEST_DECISION_DIR` (a decision model) with `XS_DECISION_FIXTURES` (its reference answers) and
  `XS_DECISION_CHUNK` (tokens a step feeds); `XS_TEST_EMBEDDING_DIR` (an embedding model) with
  `XS_EMBEDDING_FIXTURES` (its reference embeddings) and `XS_EMBEDDING_TEXTS=1` (also tokenize the texts).
  `tests/parity/model/make_tiny.py` writes a tiny random checkpoint of an architecture (real layer pattern and tensor
  names; with the checkpoint's own modeling code for one transformers does not ship, like LFM3), or of a block
  drafter for one (`--drafter`), or an encoder in a sentence-transformers checkpoint (`--arch bert|nomic_bert`), to run
  the reference scripts and the parity tests on in seconds.
- `XS_SYCL_PROFILE=1` logs the SYCL backend's per-op device time per step (barrier timestamps between ops, no host
  round trip; each barrier stalls the GPU briefly, so small ops read high: rank by it, measure with `bench`);
  `XS_SYCL_SYNC=1` waits after
  every op and names the op a device error comes from (`XS_SYCL_TRACE=1` also prints each op before it runs).
- The GPU suites ("device ops", "device attention", "device session", "device vs cpu-ref", ...) run when
  `XS_DEVICE_TEST_DEVICE=GPU.0` is set (`XS_DEVICE_TEST_PRECISION=f32` for f32; `XS_DEVICE_BENCH=1` adds the decode
  timing case).

## 3. Repository map

```text
xpu-serve/
├── CMakeLists.txt          options XS_WITH_SYCL, XS_SYCL_TARGETS, XS_BUILD_TESTS, XS_WERROR; C++20; warnings target;
│                           `cmake --install` puts the one binary in <prefix>/bin (src/CMakeLists.txt)
├── README.md               user-facing intro and quick start
├── CLAUDE.md               this file
├── DESIGN.md               the design (why), roadmap, open questions
├── TODO.md                 phase checklists; [x] = done and tested
├── docs/decisions/         one short record per decision (NNNN-title.md)
├── docs/gguf_models.md     what was measured with GGUF files: every variant of three language models and Clef-Flash that fits a 24 GB card
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
| `util/` | xs_core | `error.h` (`xs::Error`, `fail(...)`, `XS_CHECK`, `str_cat`), `log`, `timer.h`, `mmap_file`, `half` (f16/bf16 conversions), `base64` |
| `tensor/` | xs_core | `DType` (incl. `Q8` / `Q4` pack weights, `F8`, an FP8 checkpoint's weight, and `Block`, a weight in one of GGUF's block formats kept as the file stores it), `block_format` (those formats: names, bytes and weights per block), `TensorView` (non-owning, e.g. into an mmap; `quant` points at a pack weight's scales and biases, `fp8` at an FP8 weight's runs of rows, `block` names a Block weight's format), `HostTensor` (owning), `TensorMap` (name -> HostTensor) |
| `io/` | xs_core | `safetensors`: `WeightStore` interface, `SafetensorsWeights::open_dir` (single file or sharded index, mmapped; a pack's quantized weights appear as one Q8 / Q4 view each, or one Block view over their U8 rows of blocks, an FP8 checkpoint's `F8_E4M3` matrices as one F8 view each over codes and block scales; `join_rows` shows F8 tensors stored in pieces as one weight, without a copy; `open_file` for a head kept in a file of its own); `safetensors_writer` (streamed writing); `joined_weights` (a drafter checkpoint's weights under a prefix beside the target's); `gguf` (a GGUF file read in place: metadata of every type, the tensor table, each tensor's bytes; refusals by name) |
| `quant/` | xs_core | round-to-nearest quantization into the pack formats (DESIGN.md 6.3) and dequantization; a tensor `[..., K]` is a matrix of `numel / K` rows (expert banks `[E, N, K]` included); `read_rows`, a float weight's rows whatever it is stored as (what backends and `convert` read through); `fp8` (the E4M3 codes and block scales of FP8 checkpoints); `block_decode` (GGUF block formats decoded on the host, bit for bit as ggml decodes them) and `block_tables.h` (the codebooks of their i-quant formats, ggml's own: plain constexpr data that device kernels include too); `sign_sketch`, the one-bit copy of a weight a drafter's shortlist scores with |
| `convert/` | xs_core | `convert_model`: checkpoint (language model, decision model, policy, or a drafter beside its model with `--target`) -> pack; the model's own graphs decide which weights to quantize (`plan_graph_formats`: bandwidth-bound matmuls, or all with `--scope all`); `gguf/`: `import_gguf`, a GGUF file -> pack (`gguf_import.cpp`: block format tensors copied byte for byte, float tensors as stored or through a transform, streamed; config.json, tokenizer, chat template, manifest), `architecture.h` (what an architecture's importer provides: config.json from the metadata and any other JSON file of the pack, the tensor name map, float transforms, the named chat templates it has no use for, and a model's MTP head from a file of its own, `--mtp`: `plan_with_mtp`, refused by default), `layout` (the tensor layout an architecture describes and the checks every file goes through: each tensor of the file and the pack tensors it holds, fused ones cut into runs of rows, ones the converter computed from the config checked and left out, a shape's leading dims of 1 not counted), one file per architecture (`qwen35.cpp`, `gemma4.cpp` for `gemma4` and `gemma4-assistant`, `clef.cpp`), `qwen35.h` (the Qwen3.5 decoder's reading under any metadata namespace: `qwen35`'s, and `clef`'s, which adds a decision head), `tokenizer` (tokenizer.json and tokenizer_config.json from the file's vocabulary and merges, the normalizer, pre-tokenizer and BPE settings from a table keyed by `tokenizer.ggml.model` and, for gpt2, `tokenizer.ggml.pre`, with the type exceptions and end-of-turn tokens of each) |
| `image/` | xs_core | `Image` (8-bit RGB), `decode` (stb_image), `resize_bicubic` (bit-exact with torchvision), `content_hash` |
| `graph/` | xs_core | the step graph IR and builder (`graph.h` holds the op spec) |
| `models/` | xs_core | `model.h` (the `Model` interface: language models), `policy.h` (the `Policy` interface: robot policies, one static program per call shape: camera subset, step count, padded prompt length), `decision.h` (the `DecisionModel` interface: decision models, which lay out a request's prompt and name the steps to run), `embedding.h` (the `EmbeddingModel` interface: embedding models, one stateless encoder program and a spec: dimension, longest input, pooling, normalize), `registry.cpp` (model_type / LeRobot type / a head beside the decoder -> code, `model_kind`), `config_util` (checked JSON readers), `common/blocks` (the shared building blocks: gated MLPs, mixture of experts, RMSNorm variants, RoPE frequencies, attention, a drafter's head over a shortlist), `common/py_text` (a JSON value as Python's `json.dumps` and `str` write it: what reference code puts in a prompt), `common/sentence_transformers` (the pipeline a sentence-transformers checkpoint lists: pooling mode, Normalize, the longest input; the rest refused by name), `common/encoder` (BERT's post-norm layer and the `EmbeddingModel` around an encoder graph), one folder per architecture (`pi05/`: π0.5; `clef/` and `pplx_decider/`: decision models on the Qwen3.5 decoder; `bert/` and `nomic_bert/`: encoders of embedding models), `letter_readout/` (any language model as a decision model: OpenJev), `drafters/dflash` (block drafters that serve any target they were trained for: DFlash, DFlash 2, DSpine) |
| `engine/` | xs_core | `Engine` (one sequence: prefix reuse, chunked prefill, image encoding, decode loop, speculative rounds: draft, verify, accept, roll back), `Sampler` (and rejection sampling of a draft), request/result types (`MediaInput`), `media` (placeholder expansion), `service.h` (the `InferenceService` interface the server uses), `spec/` (the drafter side of speculative decoding: `drafter.h` is what the engine drives and the session's program order; `chain_drafter` feeds a token-a-run drafter the target's hidden states and chains its drafts; `block_drafter` runs a block drafter's program and follows its candidates) |
| `backend/` | xs_backends | `backend.h` (Backend/Session interfaces), `registry.cpp`, `cpu_ref/`, `sycl/` |
| `tokenizer/` | xs_tokenizer | Hugging Face `tokenizer.json`, natively: `model` (the model interface and its vocabulary: `bpe_model`, `wordpiece_model`), normalizers (Replace, NFC, BertNormalizer, from generated Unicode tables), pre-tokenizers (Split with hand-written GPT-style regexes, ByteLevel with or without the GPT-2 regex, Metaspace, BertPreTokenizer; pieces may be rewritten, so they are owned strings), `post_processor` (TemplateProcessing's special tokens around one sequence: `Tokenizer::with_special_tokens`), byte fallback, byte-level or WordPiece decoding, added tokens, streaming decoder; tokenizer.json's truncation and padding are not applied |
| `chat/` | xs_chat | `ChatTemplate` (renders the model's Jinja template with vendored jinja.cpp, in the request's key order), output parsers per dialect: `gemma4`, `qwen3_5` (thinking, tool calls; LFM3 writes the same format), `plain` |
| `server/` | xs_server | HTTP (cpp-httplib), OpenAI request parsing and response formatting, stop strings, streaming; the policy API; the System One API (`decision_handlers`); the embeddings API (`embedding_handlers`) |
| `service/` | xs_service | `EngineService`: loads tokenizer + template + model, compiles on a backend, owns the engine thread |
| `policy/` | xs_policy | `PolicyRunner`: a policy, its tokenizer and one compiled program per call shape; warm-up; per-call timings (`xpu-serve bench <policy-dir>`) |
| `decision/` | xs_decision | `DecisionRunner`: a decision model, its tokenizer and chat template (`PromptTools`), its programs in one session, a request's steps run as the model names them; `answers` (probabilities -> choice, score, confidence); `DecisionService` (one device thread, requests in arrival order) |
| `embedding/` | xs_embedding | `EmbeddingRunner`: an embedding model and its one stateless session, a run per token sequence, timings; `pooling` (mean or CLS, then the L2 norm, on the host); `EmbeddingService` (the tokenizer for text, one device thread) |
| `cli/` | (exe) | `main.cpp` (commands serve, run, bench, convert, info), `args` (flag parser) |

### 4.1 Files that matter most

| File | Role |
|---|---|
| `src/graph/graph.h` | **The op contract.** Every op's semantics are written next to `OpKind`; every backend must implement exactly that |
| `src/models/model.h` | What a model provides: `info()`, `graph()`, `weights()`, `prepare_inputs()`, and optionally `vision()` (a `VisionEncoder`: preprocessing, encoder graph, prompt layout of an image) |
| `src/models/registry.cpp` | Dispatch on `config.json` `model_type`; the list of supported types |
| `src/models/gemma4/gemma4_model.cpp` | Gemma 4 text model: graph builder + per-step host inputs; experts beside the MLP and K = V global layers (26B-A4B); an assistant checkpoint as the drafter (query-only layers over the target's KV through `kv_read`, the centroid head) |
| `src/models/gemma4/gemma4_config.cpp` | Gemma 4 `config.json` parsing; refuses unsupported variants by name |
| `src/models/gemma4/gemma4_vision.cpp` | Gemma 4 vision: image processor (resize, patches, pooling map) and vision tower graph |
| `src/models/qwen3_5/qwen3_5_model.cpp` | Qwen3.5: Gated DeltaNet layers (causal conv + gated delta rule) and gated full attention; dense MLP or `blocks::moe` (`qwen3_5_moe`: fused gate_up expert banks, or an FP8 release's per-expert tensors joined at load; gated shared expert); the MTP head (`mtp.*`) as the drafter, built from the same layer builders, or a block drafter checkpoint in its place |
| `src/models/qwen3_5/qwen3_5_decoder.cpp` | `qwen3_5::Decoder`: the decoder's layer builders over one checkpoint's weights, shared by the language model and the decision models built on it; full-attention layers causal through a KV slot, or over the whole step; linear attention's value heads in the checkpoint's order (`linear_value_head_order`: grouped as Hugging Face stores them, or tiled as GGUF files do, moved to the recurrence's order and back around GatedDelta) |
| `src/convert/gguf/qwen35.cpp` | GGUF import of the Qwen3.5 family: metadata -> config.json, `blk.<i>.*` -> Hugging Face names, the converter's transforms undone on float tensors (norms stored as 1 + w, `ssm_a` = -exp(A_log)), value heads left tiled (`linear_value_head_order`), merged input projections cut by rows, the MTP block -> `mtp.*`, also from the head's own file (`--mtp`: llama.cpp's draft model, its metadata checked against the model file's, its own embedding, output head and final norm checked and left out) |
| `src/convert/gguf/clef.cpp` | GGUF import of Clef (Clef-Flash, Clef): the decoder read as a `qwen35` file's under `clef.*` (`read_qwen35_decoder`, `qwen35.h`), the head's `dec.blk.<i>` (evidence layers, then decoder layers) and `decision.*` tensors mapped back to the checkpoint's names under `joint_head.`, each attention's q / k / v kept apart as a pack of the checkpoint holds them, `decision.scales` (the values the head multiplies by) kept as `joint_head.scales`, joint_head_config.json from the metadata and the head's tensor shapes; llama.cpp's `systemone` prompt template left out (the model code writes the prompt) |
| `src/convert/gguf/gemma4.cpp` | GGUF import of Gemma 4 E2B / E4B as a text-only checkpoint, and of its assistant (`gemma4-assistant`) as a drafter pack (config.json and weights; no centroid head in the file, so `use_ordered_embeddings: false`, the whole-vocabulary head llama.cpp computes): metadata (per-layer head counts and MLP widths, the sliding pattern, KV sharing) -> config.json, names mapped back with no value changed, a fused `attn_qkv` cut by rows, `rope_freqs` read as the proportional RoPE's `partial_rotary_factor` and left out; refuses a layout whose KV sharing llama.cpp reads otherwise than Gemma 4 |
| `src/models/decision.h` | What a decision model provides: `spec()`, `programs()`, `weights()`, and `start(request, tools, limits)` -> a `DecisionRun` that hands the runtime one step at a time and ends with each question's probabilities |
| `src/models/clef/` | Clef: `clef_prompt.cpp` (the reference's record encoding: token ids and the spans of each instruction and option), `clef_model.cpp` (the encode program, which keeps what the head's attentions read of the sequence in KV slots, the route, fields and score programs, and the host work between them; the head's three scalars as the checkpoint's parameters, or as a GGUF import's `joint_head.scales`) |
| `src/models/pplx_decider/` | pplx-decider (protocol version 3): `pplx_prompt.cpp` (the chat-template prompt with answer codes and a marker per question, special tokens in user text neutralized, a long request split by question), `pplx_model.cpp` (one program: the decoder with whole-step full attention and the read-out on every row; temperature and softmax on the host) |
| `src/models/letter_readout/letter_readout.cpp` | A `Model` behind the `DecisionModel` interface (OpenJev's helper): a prompt per question through the chat template, the logits of the option letters at the first output position, temperature and yes / no calibration (`Settings`, presets by name), more options than letters in parts and a pass over their winners; the state's part of a prompt read once and each question's part undone for the next |
| `src/decision/runner.cpp` | Wiring from a decision model directory to answers: compiles the programs, runs a request's steps, times them |
| `src/server/decision_handlers.cpp` | `POST /v1/systemone`: the request schema (422 bodies naming the field), the answers, `Server-Timing` |
| `src/models/embedding.h` | What an embedding model provides: `spec()`, `weights()`, `graph()` (the encoder, its final hidden states as "hidden") and `prepare_inputs(ids)` |
| `src/models/common/sentence_transformers.cpp` | A sentence-transformers checkpoint's pipeline (`modules.json`, the Pooling config in both spellings, `sentence_bert_config.json`), as sentence-transformers 6 reads it; every other module, pooling mode or setting refused by name |
| `src/server/embedding_handlers.cpp` | `POST /v1/embeddings`: text, token ids or lists of them; float or base64; 400s naming the field (`input[2]`), over-long inputs refused |
| `src/models/lfm3/lfm3_decoder.cpp` | `lfm3::Decoder`: LFM3's layer builders over one checkpoint's weights (under `model.`, or `model.language_model.` in an `lfm3_vl` checkpoint, whose `text_config` `lfm3::text_config` reads), shared by the language model and d1: gated short convolutions (`causal_conv` between two gates of one projection), attention with q/k norms and a sigmoid gate per head (full layers rotate half of each head, sliding layers all of it within their window; full layers causal through a KV slot, or over the whole step), every operator and MLP output normed again before the residual add |
| `src/models/lfm3/lfm3_model.cpp` | LFM3 as a language model: the decoder, its last row through the final norm and the head tied to the embedding |
| `src/models/drafters/dflash.cpp` | Block drafters (DFlash, DFlash 2, DSpine): their config (refusals by name), the context a target's prefill and verify programs append for them, and the block program (DFlash 2's convolutions and candidate selector, DSpine's code bridge and refined last write) |
| `src/backend/backend.h` | `Backend::compile(graphs, weights, options) -> Session` (programs sharing weights by name and state slots by label); `Session::run(program, inputs, outputs, only)` (every output, or just the ones asked for), `reset`, `truncate` (inside a step of up to `rewind_tokens` tokens too) |
| `src/backend/cpu_ref/ops.cpp` | f32/f64 reference kernels, one per op: the accuracy oracle |
| `src/backend/sycl/sycl_backend.cpp` | SYCL session: device dtypes, fusions, per-T memory plan (one arena), op dispatch, KV and state buffers |
| `src/backend/sycl/sycl_kernels.cpp` | SYCL kernels: elementwise, norms, RoPE, gather, copies, causal conv, gated delta, decode attention, GEMV, top-k (rows as wide as a vocabulary by blocks), SignMatMul (bit planes and popcounts) |
| `src/backend/sycl/sycl_matmul.cpp` | MatMul (GEMV up to 8 rows, oneMKL GEMM beyond; quantized: `quant_path` picks the GEMV or its matrix-engine version, quant_gemm, quant_gemm_tiled or dequantize + oneMKL per device and shape) and Attention (per-row kernel with split-K, the fused kernel, or GEMMs) |
| `src/backend/sycl/sycl_flash_attention.cpp` | Fused attention on the matrix engines (joint_matrix; head_dim up to 128; blocks of tokens in one launch) |
| `src/backend/sycl/sycl_quant_gemm.cpp` | Quantized matmuls on the matrix engines (DPAS and 2D block load builtins): gemv_quant_xmx (up to 8 rows at about the cost of one: a speculative verify; x in f16 and its group sums from one small launch before), quant_gemm (9 to 64 rows, the weight read once, 64 bytes of codes a lane and 32 rows of x a load; weights that read the same x share a launch) and quant_gemm_tiled (many rows: each weight tile dequantized once per work-group of 512 rows into local memory, no f16 copy of the weight) |
| `src/backend/sycl/sycl_xmx.h` | What the kernels written against the matrix engines directly share: the DPAS and 2D block load builtins, their operand types, and `xmx_prep` (x in f16 and its sums per group of columns: the A operand of the GEMVs on the matrix engines) |
| `src/backend/sycl/sycl_block_layout.cpp` | GGUF block-format weights for the device: a block's fields regrouped into aligned streams with codes in column order (the same bits, nothing re-encoded); Q4_0, Q4_1 and Q8_0 rewritten as a pack's q4 / q8 with a group of 32 (the same values), which the pack kernels then read |
| `src/backend/sycl/sycl_block_kernels.cpp` | Kernels over those streams, one decoder per format (`Fmt<F>`: K-quants read their codes without an int-to-float conversion, codebook formats look their magnitudes up in a table on the device): the GEMV (up to 8 rows, each weight read once) and its matrix-engine version (`gemv_block_xmx`: each lane writes its own output feature's codes or levels, exact in f16, into DPAS B operands, so the cost hardly grows with the rows: a speculative verify), the weight written out as f16 / f32 for a GEMM, rows gathered by id. `QuantWeight::block` routes a weight to them through the pack kernels' entry points |
| `src/engine/engine.cpp` | Generation loop for one sequence; encodes images and splices their rows into prefill chunks; speculative rounds (`spec_round`) |
| `src/models/model.h` (`Speculation`) | What a model with a drafter adds: prefill, verify and draft programs, and its way of drafting (`ChainDrafting`: a token a run from the target's `hidden`; `BlockDrafting`: a block a run from state the target's programs write) |
| `src/service/engine_service.cpp` | Wiring from a model directory to a ready service |
| `src/server/completion_handlers.cpp` | `/v1/chat/completions` and `/v1/completions` |
| `src/cli/main.cpp` | Commands, flags and their defaults |

## 5. How a request flows

```text
HTTP POST /v1/chat/completions                                     server/completion_handlers.cpp
  -> parse + validate the OpenAI body (400 names the field);       server/openai_request.cpp
     image_url data: URLs come out as bytes
  -> prepare_image(bytes): decode, resize, patches -> MediaInput   image/, models/<arch>/ (VisionEncoder::prepare)
  -> ChatTemplate::render(messages, tools)                         chat/chat_template.cpp (+ third_party/jinja_cpp)
  -> Tokenizer::encode(prompt)                                     tokenizer/
  -> expand_media: each image placeholder -> its token run         engine/media.cpp
  -> InferenceService::generate(request, on_token)                 service/engine_service.cpp (queues to engine thread)
       Engine::generate                                            engine/engine.cpp
         reuse_prefix: keep the KV of the longest common prefix with the previous request (images by content hash)
         prefill in chunks (--prefill-chunk); an image's rows come from one vision-session run (cached by hash)
         -> decode one token per step:
           Model::prepare_inputs(tokens, start_pos, rows) -> TensorMap   models/<arch>/
           Session::run(inputs) -> logits                          backend/<name>/
           Sampler::sample(logits)                                 engine/sampler.cpp
         on_token(token) per generated token
  -> StreamDecoder (text deltas) -> OutputParser (content / reasoning / tool calls)   tokenizer/, chat/
  -> stop strings, SSE chunks or one JSON body                     server/
```

Threads: HTTP threads parse, render and tokenize; one engine thread owns the device and runs generations one at a
time in arrival order.

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
sequence. Model code builds it once at load time; a backend compiles it once.

- Two symbolic dims: `kT` (tokens fed this step) and `kL` (tokens in the KV cache after this step). All other dims
  are static. No batch dim.
- Values are logically f32; weights keep their stored dtype and backends choose the compute precision.
- MatMul weights use the Hugging Face layout `[out, in]`: `matmul(x, w) = x @ w^T`.
- Ops: Input, Weight, Constant, Gather, MatMul, Add, Mul, Div, Scale, Tanh, GeluTanh, Gelu, Softcap, Sigmoid, Silu, Softplus,
  Clamp, RmsNorm, Softmax, TopK, ReduceSum, Reshape, Slice, Transpose, PrevRow (each row of a step replaced by the row
  before it), Rope (1-D or axial, optionally partial), KvCache, KvRead (another program's KV slot, read only),
  Attention (causal with optional window, or non-causal: every key, the keys of the query's own block of tokens, or
  a step that sees itself whole and a window before it), CausalConv and GatedDelta (stateful, for linear attention),
  ExpertMatMul (each token's chosen experts from a stacked `[E, N, K]` bank, for mixture of experts), SignMatMul (a
  matmul by a one-bit weight: the cheap score of every row that a shortlist starts from). The exact semantics
  are the comments in `graph.h`.
- KV: `add_kv_slot(...)` declares per-sequence state; `kv_cache(slot, k, v)` appends this step's K/V and returns the
  whole cache. A slot can be read by later layers (Gemma 4's KV-shared layers reuse an earlier layer's outputs).
- Graph inputs are filled per step by `Model::prepare_inputs` (token ids, positions, and model-specific host inputs
  such as Gemma 4's per-layer embedding rows, and the image rows that replace soft-token embeddings).
- Conv and recurrent slots (`add_conv_slot`, `add_recurrent_slot`) hold fixed-size per-sequence state for
  `causal_conv` and `gated_delta`. They summarize the whole sequence, so a session cannot cut them back to a prefix,
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
| `sycl` | main GPU backend, the default when built (icpx turns it on) | Runs the graph op by op on an in-order Level Zero queue with its own kernels; oneMKL for prefill GEMMs. Weights and KV f16 (f32 at f32 precision), activations f32; `--kv-dtype i8` stores KV as int8 with an f32 scale per (token, KV head): decode attention reads the codes, prefill dequantizes the keys in view to f16 for its GEMMs. Memory: one arena per step, planned per T from value lifetimes; KV slots grow by doubling; conv/recurrent state double-buffered so a failed step changes nothing. Fuses act(x)*gate and conv+SiLU. Decode attention reads only the sliding window and splits long caches across work-groups. Options `attention=auto|rows|gemm`, `gemv=auto|vector|xmx` (quantized matmuls of up to 8 rows: the GEMV, or its matrix-engine version from 4 rows, from 2 for a GGUF block format), `few_rows=N` (quantized matmuls of 9 to N rows on quant_gemm, 64 by default; 0: dequantize + GEMM). GGUF block formats keep the file's bits, regrouped for aligned loads (`block=f16` decodes them to f16 instead: the reference the kernels are tested against). See docs/decisions/0007 |

## 7. Tests (`tests/`)

One test binary, `xs_tests` (doctest). Every `tests/unit/**/*.cpp` is compiled in automatically.

| Path | What it checks | Against |
|---|---|---|
| `tests/unit/test_graph.cpp`, `test_half.cpp`, `test_base64.cpp` | IR shape inference and validation; f16/bf16 conversions; base64 | hand-written expectations, RFC 4648 vectors |
| `tests/unit/image/` | decoding, resize, hashing | Pillow-written images; torchvision resize outputs in `tests/fixtures/image/` (bit-exact) |
| `tests/unit/backend/cpu_*.cpp` | each op on cpu-ref (`cpu_encoder_ops.cpp`: the encoder ops and stateless sessions), KV sessions, attention masks | fp64 references in `reference_ops.h`, with planted-bug controls |
| `tests/unit/backend/device_*.cpp` | each op on the GPU backend (`device_encoder_ops.cpp` likewise), sessions, agreement with cpu-ref (also with an int8 KV cache on both), decode timing and quantized matmul timing by rows, format and path (`XS_DEVICE_BENCH=1`, `device_quant_timing.cpp`) | the same references; skip unless `XS_DEVICE_TEST_DEVICE` is set |
| `tests/unit/backend/*_moe_ops.cpp` | Div, Softmax, TopK (ties to the lower index), ReduceSum, ExpertMatMul and the `moe` block (softmax and sigmoid routers, shared expert), on cpu-ref and the GPU | fp64 references and an fp64 composition of the block (`moe_ops_cases.h`) |
| `tests/unit/backend/*_shortlist_ops.cpp` | SignMatMul (words loaded whole and one by one, rows that fill no sub-group) and the shortlist head: the best rows by sketch, their exact logits, the head's pick, on cpu-ref and the GPU | fp64 references (`shortlist_cases.h`) |
| `tests/unit/backend/*_programs.cpp` | sessions of several programs: KV slots shared by label, KvRead, truncate inside a short step (conv, gated delta, KV) equals never running the dropped tokens, the last step undone whole equals never running it; a run asked for some outputs returns those and advances the state all the same | the same session without them (`session_cases.h`) |
| `tests/unit/engine/` | speculative decoding emits exactly the greedy tokens (a stateful tiny model, a partly wrong drafter that drafts a token a run or a block a run, prefix reuse, stops); the same tokens when greedy decoding reads each row's `argmax` output instead of the logits; rejection sampling draws the target's distribution | plain greedy decoding; exact distributions |
| `tests/unit/backend/*_recurrent_ops.cpp` | sigmoid/silu/softplus, partial rope, CausalConv and GatedDelta over several steps (state carried), on cpu-ref and the GPU | one fp64 pass over the whole sequence (`recurrent_ops_cases.h`) |
| `tests/unit/tokenizer/` | encode/decode, streaming, the post-processor's special tokens; NFC and regex Split hermetically (`test_tokenizer_qwen35.cpp`); WordPiece, BertNormalizer, BertPreTokenizer and the WordPiece decoder on a hand-written vocabulary, and the BERT normalizer and pre-tokenizer over every code point (`test_tokenizer_wordpiece.cpp`); the GPT-2 regex, ByteLevel and Metaspace, the regex over every code point (`test_tokenizer_laya_julia.cpp`); whole tokenizers of Gemma 4, Qwen3.5, LFM3, the WordPiece embedding models and the Laya and Julia-1 decision models (ModernBERT's and mmBERT's) (`parity_cases.h`; each needs its checkpoint's `tokenizer.json`; `scratch_model_dir.h` writes a hand-written one to load) | Hugging Face `tokenizers` outputs in `tests/fixtures/tokenizer/` (`tests/parity/tokenizer/gen_cases.py`, `gen_qwen35_cases.py`, `gen_lfm3_cases.py`, `gen_wordpiece_cases.py`, `gen_laya_julia_cases.py`; tables from `gen_unicode_tables.py`) |
| `tests/unit/chat/` | template rendering (LFM3's template comes from `XS_LFM3_DIR`), Gemma 4 and Qwen3.5 output parsing, the latter on LFM3's prompts too | `apply_chat_template` outputs in `tests/fixtures/chat/` (`tests/parity/chat/gen_cases.py`) |
| `tests/unit/server/` | request parsing, response format, stop strings, HTTP end to end | a fake `InferenceService` (`fake_service.h`) |
| `tests/unit/io/gguf_reader.cpp` | the GGUF reader: every metadata value type, nested arrays, a vocabulary-sized string array, tensors at the file's alignment, versions 2 and 3, values compared exactly (`same_as`); refusals (big-endian, split files, tensor types it does not read, truncation, misalignment) | files written by `gguf_test_util.h` (which also encodes Q8_0 and Q4_0 blocks as ggml does) |
| `tests/unit/quant/test_block_decode.cpp` | every GGUF block format decoded on the host, on random and edge blocks; the block layouts and codebook tables; planted wrong layouts | ggml's own `dequantize_row_*`, bit for bit, in `tests/fixtures/quant/blocks/` (`tests/parity/quant/gen_block_cases.py`) |
| `tests/unit/quant/test_block_weights.cpp` | weights in GGUF block formats: rows read whole, in part and of 3-D banks, a pack's block format entries and their refusals, the sign sketch of one | values written from the formats' fields; the same weight decoded whole |
| `tests/unit/backend/cpu_block_ops.cpp` | MatMul, Gather and ExpertMatMul reading block format weights on cpu-ref | the same graph over the decoded f32 weights, bit for bit |
| `tests/unit/backend/device_block_ops.cpp` | MatMul (every row count the GEMV's variants take, and past them through a GEMM, with a residual folded in) and Gather reading block format weights on the GPU, and the same with `block=f16`; timing by format with `XS_DEVICE_BENCH=1` (`device_block_timing.cpp`) | fp64 over the values the host decodes; blocks of random bytes |
| `tests/unit/convert/gguf_import.cpp` | GGUF import end to end: a tiny qwen35 GGUF written as llama.cpp's converter writes one (`tiny_gguf.h`: Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, tiled value heads, merged input projections, an MTP head, a byte-level BPE vocabulary): its weights, logits over prefill and decode steps, greedy tokens and drafts on cpu-ref, its tokenizer, chat template, stop token and sampling defaults; the same model as a file without the head and the head's own file in unsloth's layout (`--mtp`, `split_tiny_gguf`): the same pack byte for byte and the same chained drafts; refusals by name (both ways in) | the twin checkpoint its blocks decode to, in the Hugging Face layout; the original tokenizer; the import of the file that holds the head; planted: the pack read in the grouped head order, the head of another model |
| `tests/unit/convert/gguf_real_file.cpp` | a real qwen35 GGUF (`XS_GGUF_FILE`) against its checkpoint (`XS_QWEN35_DIR`): tokenizer parity (`parity_cases.h`) and token-by-token equality, the config fields the model code reads, the chat template renders, the import loads | Hugging Face's tokenizer and config.json |
| `tests/unit/convert/gguf_import_gemma4.cpp` | GGUF import of Gemma 4 end to end: a tiny gemma4 GGUF written as llama.cpp's converter writes one (`tiny_gemma4_gguf.h`, from the tiny Gemma 4 of `tiny_checkpoints.h`: per-layer embeddings, KV-shared layers with a double-wide MLP, per-layer KV heads, a fused `attn_qkv`, `rope_freqs`, Q8_0 / Q4_0 / F16 / BF16 / F32 tensors, a Gemma-style vocabulary with the converter's token types): its weights, logits over prefill and decode steps (to the bit) and greedy tokens on cpu-ref, its tokenizer, chat template, stop tokens and sampling defaults; a tiny gemma4-assistant GGUF beside it: the drafter's weights, draft logits and chained drafts; refusals by name | the twin checkpoints their blocks decode to; the original tokenizer; planted: the full-attention RoPE read wrong (target and drafter) |
| `tests/unit/convert/gguf_gemma4_real_file.cpp` | a real gemma4 GGUF (`XS_GGUF_FILE`) against its checkpoint's small files (`XS_GEMMA4_DIR`): tokenizer parity and equality over the whole vocabulary (text, special flag, encoding, decoding), the config fields and stop tokens the model code reads, the chat template (thinking, tools, a tool round trip, the checkpoint's renderings), the pack loads and answers on cpu-ref; a real gemma4-assistant GGUF (`XS_GGUF_DRAFTER_FILE`): its config against the assistant's (`XS_GEMMA4_ASSISTANT_DIR`), and beside the target it loads and drafts the target's next token | Hugging Face's tokenizer, config.json, generation_config.json and `tests/fixtures/chat/gemma4_cases.json`; the target's greedy tokens |
| `tests/unit/convert/gguf_import_clef.cpp` | GGUF import of Clef end to end: a tiny clef GGUF written as llama.cpp's converter writes one (`tiny_clef_gguf.h`: `tiny_gguf.h`'s decoder under `clef.*`, a joint schema head as `dec.blk.<i>` and `decision.*`, each attention's in_proj as three tensors, the scales computed from the parameters, the head's matrices Q8_0 / Q4_0 / BF16 / F16, `scorer_out` stored 1-D as llama-quantize writes it or [1, width] as the converter does): the pack's files, names, weights and joint_head_config.json, its answers on cpu-ref (and the GPU against cpu-ref); refusals by name | the tiny Clef checkpoint its blocks decode to, in the original layout; planted: query and key projections exchanged, the parameters taken for the scales |
| `tests/unit/convert/gguf_clef_real_file.cpp` | a real clef GGUF (`XS_GGUF_FILE`) against its checkpoint (`XS_CLEF_DIR`): the config fields the model code reads and joint_head_config.json, the head's tensors by name, shape and value (float ones exactly, block formats within their rounding, the scales from the parameters), the tokenizer token by token, the pack loads as a decision model | the checkpoint's config.json, joint_head_config.json, joint_head.safetensors and tokenizer.json |
| `tests/unit/test_fp8.cpp` | FP8 checkpoints: the E4M3 code table, codes paired with their block scales, tensors joined row-wise, rows read whole and in part, refusals | values written from the codes' fields and the blocks' indices; planted swapped grid |
| `tests/unit/model/fp8_checkpoint.cpp` | a tiny Qwen3.5 (dense; mixture of experts stored expert by expert) written the way an FP8 release stores it: weights, logits (cpu-ref and GPU) and `convert`'s packs; `quantization_config` refusals | the same checkpoint with the decoded values in the usual layout (`tiny_checkpoints.h`) |
| `tests/unit/model/gemma4_config.cpp`, `qwen3_5_config.cpp`, `lfm3_config.cpp` | config parsing and refusals (LFM3: also an `lfm3_vl` checkpoint's `text_config`) | |
| `tests/unit/model/qwen3_5_value_heads.cpp` | linear attention's value heads in the tiled order: the same logits over a prefill and decode steps, no ops added for a grouped checkpoint or when the orders coincide | the tiny checkpoint in the grouped order; planted: the tiled weights read as grouped |
| `tests/unit/model/model_parity.cpp` | whole model: last-token logits and greedy continuations; its drafter (MTP head, or an assistant with `XS_TEST_DRAFTER_DIR`): logits and chained drafts after each prompt | Hugging Face fp32, from `tests/parity/model/hf_reference.py` (`--assistant` for a Gemma 4 assistant; a checkpoint that names its own modeling code, like LFM3, runs with it) |
| `tests/unit/model/block_drafter_parity.cpp` | a block drafter (`XS_TEST_DRAFTER_DIR`) over rounds whose context grows by several tokens: drafts, candidates and scores | the drafters' own PyTorch code, from `tests/parity/model/block_drafter_reference.py` (clones of z-lab/dflash and Tencent/DSpine) |
| `tests/unit/model/block_drafters.cpp` | block drafters on tiny checkpoints the test writes (`tiny_checkpoints.h`): a context fed in chunks or cut back after a verify, a causal block stopping after the drafts asked for, a drafter's pack, config refusals | the same drafter fed differently; its unquantized checkpoint |
| `tests/unit/model/policy_parity.cpp` | a robot policy end to end: prompt ids and the action chunk, all cameras and one camera (`XS_TEST_POLICY_DIR`, `XS_POLICY_FIXTURES`) | LeRobot fp32 with fixed noise, from `tests/parity/policy/lerobot_reference.py` |
| `tests/unit/backend/*_policy_ops.cpp` | LayerNorm, Concat (also of computed values that share a part), attention with a key bias over static queries (head dims 72, 256) and within blocks of tokens | fp64 references (`policy_ops_cases.h`) |
| `tests/unit/model/gemma4_vision_parity.cpp` | image input: pixels, vision features, expanded prompt, logits, greedy (whole and chunked prefill), prefix reuse by image | Hugging Face fp32, from `tests/parity/model/hf_vision_reference.py` |
| `tests/unit/model/decision_parity.cpp` | a decision model (`XS_TEST_DECISION_DIR`): each request's prompt tokens and every option's probability | the checkpoint's own Python code in fp32, from `tests/parity/decision/reference.py` over `requests.json` (`make_tiny.py` writes a tiny checkpoint in a released one's layout) |
| `tests/unit/model/decision_models.cpp` | decision models on tiny checkpoints the test writes (`decision_test_util.h`). Clef: a prompt fed in steps, a choice's options reordered, questions decided jointly, a request over the limit refused, the head's scales stored as a GGUF file stores them. pplx-decider: answers that depend on later questions, a noul's code order, a long request answered in parts, special tokens neutralized, refusals. Letter read-out (a tiny language model): each answer equal to its question asked alone, the state read once per request and not again by the next request about it, more options than letters, the calibration settings, refusals. All: the GPU against cpu-ref. Clef and pplx-decider: a q8 pack from `convert` loads without the head's file and answers nearly alike | the same model run differently; its unquantized checkpoint |
| `tests/unit/model/py_text.cpp` | Python's text for a JSON value: `json.dumps` (key order kept or sorted, both separators, an indent), `str()`, float `repr` | what Python wrote |
| `tests/unit/decision/` | answers from probabilities: the API's confidence formulas, each question type | the API's documented example and formulas |
| `tests/unit/server/test_systemone_http.cpp` | the System One API: the schema refused field by field (422 bodies), answers of each type, 400 / 500, concurrent clients | a fake decision function |
| `tests/unit/model/embedding_config.cpp` | BERT's and NomicBERT's config.json, the sentence-transformers pipeline (both pooling spellings, Normalize, the longest input, `do_lower_case`), the model kind; every refusal by name | hand-written expectations |
| `tests/unit/model/embedding_models.cpp` | encoders on tiny checkpoints the test writes (`embedding_test_util.h`): runs that keep nothing, every token seeing every other, BERT's absolute and NomicBERT's relative positions; the GPU against cpu-ref | the same model run differently |
| `tests/unit/model/embedding_parity.cpp` | an embedding model (`XS_TEST_EMBEDDING_DIR`): hidden states, pooled vector and embedding of each case's ids, up to the model's longest input; with `XS_EMBEDDING_TEXTS=1` each text's ids too | sentence-transformers fp32, from `tests/parity/embedding/reference.py` (`make_tiny.py --arch bert|nomic_bert` writes tiny checkpoints) |
| `tests/unit/embedding/` | pooling; the runner (a run per sequence, the limits it refuses, `--ctx` below the model's); the service (text through a tokenizer, clients on several threads) | values computed by hand; the runner on its own |
| `tests/unit/server/test_embeddings_http.cpp` | the embeddings API: every input form, each 400 naming its field, float and base64 (decoded), models / health / no chat routes, concurrent clients, a tiny model and its tokenizer end to end | a fake embedding function; the runner |
| `tests/e2e/` | a running chat server: 23 tool-calling cases with simulated tools, one turn and several (`tool_calls.py`, graded against `tool_calls_rubric.md`; an original checkpoint and its GGUF import compared call by call). A running decision server: 25 decisions with one accepted answer each (`decisions25.json`), the API's shape, latency (`systemone_eval.py`, also against reference answers or another run's saved ones: what a pack or a setting changes); `make_mmlu_pro.py` writes a larger, harder set from MMLU-Pro (not kept here) | accepted answers; the model's reference code; another run |

- `tests/parity/**/*.py` generate fixtures from Hugging Face (run with a Python venv that has `transformers`,
  `torch`). Small fixtures are committed; large ones go to `tests/fixtures/generated/` (gitignored).
  `tests/parity/quant/gen_block_cases.py` builds ggml's own dequantization (`ggml_dequantize.c` with the
  `ggml-quants.c` of a llama.cpp checkout) to write the block format fixtures.
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
   graph. Parity test modeled on `gemma4_vision_parity.cpp` + `hf_vision_reference.py`.
7. Chat: if the model's output format needs parsing (thinking markers, tool-call syntax), add a dialect in `src/chat/`
   and set `ModelInfo::dialect`.
8. Quantized packs need nothing more: `convert` reads the new model's graph to find its matmul weights and gather
   tables, and the loaded pack hands the same graph Q8 / Q4 weights. Only weights the model reads on the host (like
   Qwen3.5's norm weights or Gemma 4's per-layer embedding table) must stay out of matmuls and gathers, which they
   already are. Check quality with the parity test on a pack (`XS_TEST_MODEL_DIR=<pack>`).
9. Update TODO.md, this file's map, and README if users see the change.

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
- Tests: the engine's equivalence tests cover every drafter; add the drafter's parity against its reference
  (`model_parity.cpp` or `block_drafter_parity.cpp`, fixtures from `tests/parity/model/`).

### 8.6 Add a decision model

1. Read the checkpoint's reference code: how a request becomes a prompt (to the token), what is read off the decoder
   and where, how scores become probabilities.
2. `src/models/<arch>/`: a class implementing `DecisionModel` (`models/decision.h`). Build its programs once, from
   the decoder's builders (`qwen3_5::Decoder` for a Qwen3.5 decoder) plus the read-out. `start()` lays out the prompt
   with `PromptTools` (the tokenizer and chat template) and returns a `DecisionRun` whose `next()` names each step:
   the program, its inputs, the outputs to bring back, and how much of the sequence's state to keep. Rows whose
   count varies per request (options, questions) are a program's `kT`; what has a per-request shape beyond that is
   host work between steps. Refuse what the model cannot take with `DecisionRefused` (the client sees the message).
   `models/clef/` (a head of several programs), `models/pplx_decider/` (one program) and `models/letter_readout/`
   (a language model's own step graph, a prompt per question) are the examples.
3. Write prompts exactly as the reference does (`models/common/py_text.h` for JSON the Python way), and give
   probabilities in `DecisionQuestion::options` order whatever order the prompt uses.
4. Register it in `registry.cpp` (`model_kind`, `load_decision_model`, `supported_decision_models`).
5. Tests: `tests/parity/decision/reference.py` learns to run the checkpoint's code, `make_tiny.py` to write a tiny
   one; `decision_parity.cpp` then checks tokens and probabilities, on cpu-ref and the GPU; invariants on a tiny
   checkpoint the test writes go in `decision_models.cpp`. End to end: `tests/e2e/systemone_eval.py --reference`.
6. Update TODO.md, this file's map and the README's decision models table.

### 8.7 Add an embedding model

1. An encoder in `src/models/<arch>/`: its config (refusals by name) and a graph from `models/common/encoder.h` (BERT's
   post-norm layer, attention over the whole sequence) and `blocks`, made into a model with `encoder::embedding_spec`
   and `encoder::make_model`; register the `model_type` in `registry.cpp` (`supported_embedding_models`,
   `load_embedding_model`).
2. A pipeline beyond Transformer, Pooling (mean, CLS) and Normalize (another pooling mode, a Dense module, a prompt)
   is refused today: teach `models/common/sentence_transformers` and `embedding/pooling` first.
3. Tests: `make_tiny.py` learns to write a tiny one and `tests/parity/embedding/reference.py` runs it;
   `embedding_parity.cpp` then compares hidden states and embeddings; refusals in `embedding_config.cpp`.
4. Update TODO.md, this file's map and the README.

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
   the pack (`ImportPlan::files`), as `clef.cpp` does.
3. Transforms: a float tensor can go through a `FloatMap` (value by value, written as F32). A tensor in a block format
   is never decoded, re-encoded or reordered (a K-quant block spans 256 columns): it can only be cut into runs of whole
   rows. When the converter reordered a quantized weight's rows or columns, the model code computes in the file's
   order instead, behind a config.json key the importer writes (as `linear_value_head_order` for Qwen3.5); add that
   key to the model's config parser and graph first, with a test that a reordered checkpoint gives the same logits.
   When the converter stored a value computed from parameters (Clef's `decision.scales`), the model reads the stored
   values under a name of their own rather than the import computing the parameters back, which rounds again.
4. The tokenizer: a `tokenizer.ggml.model` (or, for gpt2, a `tokenizer.ggml.pre`) the table in
   `convert/gguf/tokenizer.cpp` does not have needs an entry with the original tokenizer.json's normalizer,
   pre-tokenizer, post-processor, decoder and BPE settings, where the converter's token types depart from the
   original's special flags (read the converter's `set_vocab`), and the end-of-turn tokens llama.cpp stops at by text
   (`llama-vocab.cpp`) that the file's eos_token_id does not name.
5. Tests: a tiny GGUF of the architecture written as the converter writes one, imported and compared with the
   checkpoint its blocks decode to (modeled on `tests/unit/convert/gguf_import.cpp`), with a planted control for every
   reordering; real-file checks behind an environment variable (`gguf_real_file.cpp`).
6. Update this file (map, tests, section 10) and the README's GGUF line.

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

- Three model families: Gemma 4 E2B/E4B (text and images) and 26B-A4B (text), Qwen3.5 dense and MoE (text; its
  vision tower is not loaded; MoE checkpoints must use fused expert banks, as the released ones do), and LFM3 (text:
  `lfm3`; the vision and audio variants are other model types and are refused). No audio or video.
  Images only as base64 data: URLs.
- Speculative decoding drafts greedily (rejection sampling keeps sampled output exact). Drafters: the MTP heads, the
  Gemma 4 assistants, and block drafters for Qwen3.5-family targets (checkpoints exist for Qwen3.8-27B). An MTP head
  picks each draft among 64 tokens (`--draft-shortlist`; a one-bit sketch of the output layer chooses them, the exact
  layer ranks them): a pick the shortlist misses costs a draft. A verify of more than 8 tokens leaves the fast path
  (a block drafter drafts 7 by default).
- Models with conv or recurrent state (Qwen3.5's linear attention, LFM3's convolutions) re-prefill when a new prompt
  does not extend the cached sequence (no state snapshots): with thinking on, a template that drops earlier turns'
  reasoning makes every new user turn such a prompt.
- An FP8 checkpoint (`quantization_config`: fp8, e4m3, one scale per 128 x 128 block) is the float model its codes
  decode to: weights go to the device as f16, like the original's, and activations are not rounded to FP8 as the
  reference runtime does. It is half the download; `convert` makes the pack that is smaller and faster to serve.
  Other quantized checkpoints (bitsandbytes, AWQ, GPTQ) are refused by name.
- GGUF files are imported, not served as they are: `convert <file.gguf> <out-dir>` writes a pack whose tensors keep
  the file's block formats byte for byte. Architectures `qwen35` (the Qwen3.5 family's dense text models) and `gemma4`
  (Gemma 4 E2B / E4B, the text model: llama.cpp keeps the vision and audio towers in a separate mmproj file, which is
  not read, so images need the original checkpoint), and `gemma4-assistant`, its drafter (`--drafter <pack> --spec
  mtp`): the file has no centroid head, so the imported assistant scores the whole vocabulary, as llama.cpp does,
  where the original picks among its top centroids' tokens (drafts can differ; the output is the target's either
  way). A `qwen35` file without the MTP head (unsloth's smallest Qwen3.8-27B files) imports with the head's own file,
  llama.cpp's draft model (`convert <file.gguf> <out-dir> --mtp <mtp.gguf>`), into the pack a file that held the head
  gives: the draft model's own token_embd, output and output_norm (low-bit copies of the model's) are checked against
  the model's shapes and left out, since the head reads the model's embedding and lm_head; its metadata must equal
  the model file's but for the head's block. `clef` (Clef-Flash and Clef, decision models) imports into the pack
  `convert` makes of the checkpoint, but for the head's three scalars, kept as the file stores them (the values they
  give, `joint_head.scales`); llama.cpp's `systemone` prompt template is not imported (the model code writes Clef's
  prompt) and its mmproj file (the vision tower) is not read. `qwen35moe`, a Gemma 4 with experts (26B-A4B) and every
  other architecture are refused by name. Tokenizers of the importer's
  table: byte-level BPE (`tokenizer.ggml.model` gpt2) with `tokenizer.ggml.pre` `qwen35`, and `gemma4`. The pack's chat
  template and stop tokens are the file's: a publisher's changed template is what it renders, and a stop token only
  the original checkpoint's config names (Qwen3.5's `<|endoftext|>`) is not one; for `gemma4` the import adds the
  end-of-turn tokens llama.cpp stops at (`<eos>`, `<turn|>`, `<|tool_response>`, the original's), and the
  tokenizer_config.json eos_token is the file's (unsloth's `<turn|>`, Google's `<eos>`). A file that is one part of a
  split model, or holds tensor types beyond F32, F16, BF16 and the block formats, is refused by name.
- A pack can hold weights in GGUF's block formats (`DType::Block`, kept as the file stores them): the host decodes all
  19 as ggml does, so cpu-ref runs them. The SYCL backend has kernels for all of them, but refuses an expert bank in one by
  name; `--backend-opt block=f16` decodes block formats to f16 as it uploads them (also an expert bank, at f16's
  size). Up to 8 rows a block format's matmul is a GEMV, from 2 rows its matrix-engine version (x read in f16,
  a cost that hardly grows with the rows: a speculative verify), past them the weight is dequantized for a GEMM. Q4_0
  takes a ninth more device memory than its file (its q4 form stores a bias per group).
- Quantization is round-to-nearest only (no AWQ / GPTQ yet). Gemma 4's per-layer embedding table (read on the host)
  and vision towers stay bf16 in a pack. Prefill dequantizes each quantized weight to f16 for oneMKL's GEMM, except
  on integrated GPUs up to 512 rows (quant_gemm / quant_gemm_tiled dequantize on the way; sycl_be::quant_path).
- Decision models: text states only (`assets`, `images` and box questions are refused by name); a decision's
  latency is the prefill of its prompt, so a pack of one is smaller but slower on a discrete GPU, and q4 changes
  answers the model is unsure of (docs/decisions/0015); `convert` rounds to nearest for them (no AWQ).
  pplx-decider reads a prompt in one step (its attention looks ahead), which costs more per token past a few
  thousand tokens. A language model read with `--readout` serves decisions only (no chat on the same server).
- Embedding models: BERT and NomicBERT encoders of sentence-transformers checkpoints, mean or CLS pooling; a request's
  inputs run one at a time (no batching); an input over the limit is refused, not cut; `dimensions` only as the
  model's own; no rerank or score API; no packs (`convert` refuses them).
- One sequence at a time; no batching.
- Sliding-window layers keep every key in memory (the kernels read only the window; memory still grows with context).
