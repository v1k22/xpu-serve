# PROJECT.md: how this repo is laid out

The orientation guide for anyone (person or AI) about to change this code. Read this first, then
[DESIGN.md](DESIGN.md) for the reasoning and [TODO.md](TODO.md) for what is planned. Keep this file current: a change
that moves, adds or renames a file updates the map below in the same commit.

## 1. What xpu-serve is

A C++20 LLM inference engine and OpenAI-compatible server for Intel GPUs. One repo, one binary (`xpu-serve`):
model code, engine, tokenizer, chat templates and HTTP server all live here, and the math runs through a backend
(OpenVINO today, SYCL planned, plus a CPU reference used as the test oracle).

The rule that shapes everything: **one feature = one PR in this repo.** Supporting a model or a feature must never
need changes in an exporter, a separate pipeline library and a separate server.

How that works:

- **Model code is C++ in this repo** (`src/models/<arch>/`). It reads the Hugging Face `config.json`, and builds a
  backend-neutral **step graph** out of ~20 ops (matmul, rms_norm, rope, attention, ...).
- **Weights are read from Hugging Face safetensors at load time** (memory-mapped). There is no exported IR and no
  conversion step yet; a `convert` command for quantized packs is planned (DESIGN.md section 6).
- **A backend compiles the step graph** for a device and runs it. Backends know nothing about models.

Status (see TODO.md): Gemma 4 E2B/E4B, text and image input, served end to end on OpenVINO (CPU, Arc B390, Arc Pro
B60), matching Hugging Face fp32 logits (text: cosine >= 0.999999 in f16 on GPU). One sequence at a time, no
quantization yet.

## 2. Build, run, test

```bash
source /opt/intel/openvino_2026.2.1/setupvars.sh          # OpenVINO 2026.2 runtime
cmake -S . -B build -G Ninja -DOpenVINO_DIR=$INTEL_OPENVINO_DIR/runtime/cmake
cmake --build build
build/tests/xs_tests                                      # every unit test (doctest); model tests skip without a model
build/src/xpu-serve info                                  # backends and devices
build/src/xpu-serve run   <hf-model-dir> --device GPU --prompt "Hi"
build/src/xpu-serve run   <hf-model-dir> --device GPU --image photo.jpg --prompt "Describe this image."
build/src/xpu-serve serve <hf-model-dir> --device GPU     # http://127.0.0.1:8080/v1
build/src/xpu-serve bench <hf-model-dir> --device GPU --prompt-tokens 1024 --gen-tokens 128
```

- Without OpenVINO, CMake builds only the CPU reference backend (`-DXS_WITH_OPENVINO=OFF`).
- Useful flags: `--backend openvino|cpu-ref`, `--device CPU|GPU|GPU.1`, `--precision f16|f32`, `--ctx N`,
  `--prefill-chunk N`, `--backend-opt key=value` (OpenVINO: `attention=auto|sdpa|matmul`,
  `activations_scale_factor=F`), `--no-vision`, `--log-level debug`. `xpu-serve <command> --help` lists all of them;
  `src/cli/main.cpp` is the source of truth.
- Run one test case: `build/tests/xs_tests -tc='gemma4 matches*'`. Environment for the tests that need files:
  `XS_TEST_MODEL_DIR` (a Hugging Face model directory), `XS_PARITY_BACKEND` (`cpu-ref` or `openvino`),
  `XS_PARITY_DEVICE`, `XS_PARITY_PRECISION`, `XS_PARITY_FIXTURES`.
- Remote GPU machines: `scripts/dev/remote_build.sh <system> <tag>` copies this tree to `~/xpu-serve/src-<tag>` on a
  machine reachable through the `remote` CLI and builds it there; `scripts/dev/remote_env.sh` sets up OpenVINO and
  `XS_TEST_MODEL_DIR` on that machine.

## 3. Repository map

```text
xpu-serve/
├── CMakeLists.txt          options XS_WITH_OPENVINO, XS_BUILD_TESTS, XS_WERROR; C++20; warnings target
├── README.md               user-facing intro and quick start
├── PROJECT.md              this file
├── DESIGN.md               the design (why), roadmap, open questions
├── TODO.md                 phase checklists; [x] = done and tested
├── docs/decisions/         one short record per decision (NNNN-title.md)
├── scripts/dev/            remote_build.sh, remote_env.sh (remote GPU test machines)
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
| `tensor/` | xs_core | `DType`, `TensorView` (non-owning, e.g. into an mmap), `HostTensor` (owning), `TensorMap` (name -> HostTensor) |
| `io/` | xs_core | `safetensors`: `WeightStore` interface, `SafetensorsWeights::open_dir` (single file or sharded index, mmapped) |
| `image/` | xs_core | `Image` (8-bit RGB), `decode` (stb_image), `resize_bicubic` (bit-exact with torchvision), `content_hash` |
| `graph/` | xs_core | the step graph IR and builder (`graph.h` holds the op spec) |
| `models/` | xs_core | `model.h` (the `Model` interface), `registry.cpp` (model_type -> code), `config_util` (checked JSON readers), one folder per architecture |
| `engine/` | xs_core | `Engine` (one sequence: prefix reuse, chunked prefill, image encoding, decode loop), `Sampler`, request/result types (`MediaInput`), `media` (placeholder expansion), `service.h` (the `InferenceService` interface the server uses) |
| `backend/` | xs_backends | `backend.h` (Backend/Session interfaces), `registry.cpp`, `cpu_ref/`, `openvino/` |
| `tokenizer/` | xs_tokenizer | Hugging Face `tokenizer.json` BPE, natively (normalizer, pre-tokenizer, added tokens, byte fallback, streaming decoder) |
| `chat/` | xs_chat | `ChatTemplate` (renders the model's Jinja template with vendored jinja.cpp), output parsers per dialect (thinking, tool calls) |
| `server/` | xs_server | HTTP (cpp-httplib), OpenAI request parsing and response formatting, stop strings, streaming |
| `service/` | xs_service | `EngineService`: loads tokenizer + template + model, compiles on a backend, owns the engine thread |
| `cli/` | (exe) | `main.cpp` (commands serve, run, bench, info), `args` (flag parser) |

### 4.1 Files that matter most

| File | Role |
|---|---|
| `src/graph/graph.h` | **The op contract.** Every op's semantics are written next to `OpKind`; every backend must implement exactly that |
| `src/models/model.h` | What a model provides: `info()`, `graph()`, `weights()`, `prepare_inputs()`, and optionally `vision()` (a `VisionEncoder`: preprocessing, encoder graph, prompt layout of an image) |
| `src/models/registry.cpp` | Dispatch on `config.json` `model_type`; the list of supported types |
| `src/models/gemma4/gemma4_model.cpp` | Gemma 4 text model: graph builder + per-step host inputs |
| `src/models/gemma4/gemma4_config.cpp` | Gemma 4 `config.json` parsing; refuses unsupported variants by name |
| `src/models/gemma4/gemma4_vision.cpp` | Gemma 4 vision: image processor (resize, patches, pooling map) and vision tower graph |
| `src/backend/backend.h` | `Backend::compile(graph, weights, options) -> Session`; `Session::run/reset/truncate` |
| `src/backend/cpu_ref/ops.cpp` | f32/f64 reference kernels, one per op: the accuracy oracle |
| `src/backend/openvino/ov_translate.cpp` | Step graph -> `ov::Model` (one lowering per op), KV as stateful variables |
| `src/backend/openvino/ov_attention.cpp` | Attention lowering: mask, GQA by broadcast, SDPA or matmul path |
| `src/engine/engine.cpp` | Generation loop for one sequence; encodes images and splices their rows into prefill chunks |
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

## 6. The step graph and the backends

**The step graph** (`src/graph/graph.h`) describes one engine step (a prefill chunk or one decode token) for one
sequence. Model code builds it once at load time; a backend compiles it once.

- Two symbolic dims: `kT` (tokens fed this step) and `kL` (tokens in the KV cache after this step). All other dims
  are static. No batch dim.
- Values are logically f32; weights keep their stored dtype and backends choose the compute precision.
- MatMul weights use the Hugging Face layout `[out, in]`: `matmul(x, w) = x @ w^T`.
- Ops: Input, Weight, Constant, Gather, MatMul, Add, Mul, Scale, Tanh, GeluTanh, Softcap, Clamp, RmsNorm, Reshape,
  Slice, Transpose, Rope (1-D or axial), KvCache, Attention (causal with optional window, or non-causal). The exact
  semantics are the comments in `graph.h`.
- KV: `add_kv_slot(...)` declares per-sequence state; `kv_cache(slot, k, v)` appends this step's K/V and returns the
  whole cache. A slot can be read by later layers (Gemma 4's KV-shared layers reuse an earlier layer's outputs).
- Graph inputs are filled per step by `Model::prepare_inputs` (token ids, positions, and model-specific host inputs
  such as Gemma 4's per-layer embedding rows, and the image rows that replace soft-token embeddings).
- A graph without KV slots is stateless (every run stands alone): that is how vision encoders run, one image per run,
  compiled as a second session next to the text one.

**Backends** (`src/backend/`):

| Backend | Status | Notes |
|---|---|---|
| `cpu-ref` | test oracle | f32 values, f64 accumulation, threaded matmul. Slow by design; correctness only |
| `openvino` | main backend today | Translates the graph to an `ov::Model` with the OpenVINO C++ opset API; KV as ReadValue/Assign state; bf16 weights converted to f16 (or f32) at compile; runs on CPU and Intel GPUs |
| `sycl` | planned (DESIGN.md 8.7) | |

## 7. Tests (`tests/`)

One test binary, `xs_tests` (doctest). Every `tests/unit/**/*.cpp` is compiled in automatically.

| Path | What it checks | Against |
|---|---|---|
| `tests/unit/test_graph.cpp`, `test_half.cpp`, `test_base64.cpp` | IR shape inference and validation; f16/bf16 conversions; base64 | hand-written expectations, RFC 4648 vectors |
| `tests/unit/image/` | decoding, resize, hashing | Pillow-written images; torchvision resize outputs in `tests/fixtures/image/` (bit-exact) |
| `tests/unit/backend/cpu_*.cpp` | each op on cpu-ref (`cpu_encoder_ops.cpp`: the encoder ops and stateless sessions), KV sessions, attention masks | fp64 references in `reference_ops.h`, with planted-bug controls |
| `tests/unit/backend/ov_*.cpp` | each op on OpenVINO (`ov_encoder_ops.cpp` likewise), sessions, cross-backend agreement, decode timing | the same references; skip without OpenVINO |
| `tests/unit/tokenizer/` | encode/decode, streaming | Hugging Face `tokenizers` outputs in `tests/fixtures/tokenizer/` |
| `tests/unit/chat/` | template rendering, Gemma 4 output parsing | `apply_chat_template` outputs in `tests/fixtures/chat/` |
| `tests/unit/server/` | request parsing, response format, stop strings, HTTP end to end | a fake `InferenceService` (`fake_service.h`) |
| `tests/unit/model/gemma4_config.cpp` | config parsing and refusals | |
| `tests/unit/model/gemma4_parity.cpp` | whole model: last-token logits and greedy continuations | Hugging Face fp32, from `tests/parity/model/hf_reference.py` |
| `tests/unit/model/gemma4_vision_parity.cpp` | image input: pixels, vision features, expanded prompt, logits, greedy (whole and chunked prefill), prefix reuse by image | Hugging Face fp32, from `tests/parity/model/hf_vision_reference.py` |

- `tests/parity/**/*.py` generate fixtures from Hugging Face (run with a Python venv that has `transformers`,
  `torch`). Small fixtures are committed; large ones go to `tests/fixtures/generated/` (gitignored).
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
     node names, and fill every graph input in `prepare_inputs`. The last-token logits output must be named `logits`.
3. Register the `model_type` in `src/models/registry.cpp` (`load_model` and `supported_model_types`).
4. If the model needs an op the IR does not have, add the op first (8.2).
5. Tests: a config test (`tests/unit/model/<arch>_config.cpp`) and a parity test against Hugging Face fp32 modeled
   on `gemma4_parity.cpp` + `tests/parity/model/hf_reference.py`.
6. Images (optional): implement `VisionEncoder` (see `gemma4_vision.cpp`): `prepare` reproduces the Hugging Face image
   processor on an `image::Image`, `graph()` is the tower ending in an `embeddings` output, and `token_run` lays out
   one image in the prompt. Return it from `Model::vision()` and add the `media` / `media_keep` inputs to the text
   graph. Parity test modeled on `gemma4_vision_parity.cpp` + `hf_vision_reference.py`.
7. Chat: if the model's output format needs parsing (thinking markers, tool-call syntax), add a dialect in `src/chat/`
   and set `ModelInfo::dialect`.
8. Update TODO.md, this file's map, and README if users see the change.

### 8.2 Add a graph op

Touch, in this order:

1. `src/graph/graph.h`: `OpKind` entry with its exact semantics in the comment; attrs struct if any; builder method.
2. `src/graph/graph.cpp`: `op_name`, builder with operand checks and shape inference; extend `tests/unit/test_graph.cpp`.
3. `tests/unit/backend/reference_ops.h`: an fp64 reference.
4. `src/backend/cpu_ref/ops.{h,cpp}` + the switch in `cpu_backend.cpp`; test in `tests/unit/backend/cpu_ops.cpp`.
5. `src/backend/openvino/ov_translate.cpp` (and `ov_attention.cpp` for attention); test in
   `tests/unit/backend/ov_ops.cpp` on CPU, then run the same test on both GPUs.

### 8.3 Add a backend option

`SessionOptions::extra` carries `--backend-opt key=value`. Each backend parses its own keys and refuses unknown ones by
name (`parse_extra` in `ov_backend.cpp`).

### 8.4 Add an API field

Parse and validate it in `src/server/openai_request.cpp` (a bad value is a 400 naming the field), carry it in
`GenerateRequest` or the handler, format it in `openai_format.cpp`, and test it in `tests/unit/server/`.

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

- One model family (Gemma 4 E2B/E4B); text and images, no audio or video yet. Images only as base64 data: URLs.
- No `convert`/quantization: weights are converted from bf16 to f16 at load.
- One sequence at a time; no batching. KV lives in OpenVINO state; truncation copies through the host.
- Sliding-window layers keep every key (memory grows with context).
