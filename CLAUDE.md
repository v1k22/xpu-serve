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
- **Weights are read from Hugging Face safetensors at load time** (memory-mapped). There is no exported IR and no
  conversion step yet; a `convert` command for quantized packs is planned (DESIGN.md section 6).
- **A backend compiles the step graph** for a device and runs it. Backends know nothing about models.

Status (see TODO.md): Gemma 4 E2B/E4B (text and images) and the Qwen3.5 family (Qwen3.5, Qwen3.6, Qwen3.8; text; a
hybrid of Gated DeltaNet linear attention and full attention, dense or mixture of experts), served end to end on the SYCL backend (Arc Pro B60),
matching Hugging Face fp32 logits and greedy output; q8 / q4 packs from `convert`. One sequence at a time.

## 2. Build, run, test

```bash
source /opt/intel/oneapi/setvars.sh                       # oneAPI 2026.2: icpx + oneMKL (SYCL backend)
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
build/tests/xs_tests                                      # every unit test (doctest); model tests skip without a model
build/src/xpu-serve info                                  # backends and devices
build/src/xpu-serve run   <hf-model-dir> --device GPU --prompt "Hi"
build/src/xpu-serve run   <hf-model-dir> --device GPU --image photo.jpg --prompt "Describe this image."
build/src/xpu-serve serve <hf-model-dir> --device GPU     # http://127.0.0.1:8080/v1
build/src/xpu-serve bench <hf-model-dir> --device GPU --prompt-tokens 1024 --gen-tokens 128
```

- icpx turns the SYCL backend on (`XS_WITH_SYCL`); `-DXS_SYCL_TARGETS=intel_gpu_bmg_g21` compiles its kernels ahead
  of time for the Arc Pro B60. Any other C++20 compiler builds only the CPU reference backend (the test oracle).
- Useful flags: `--backend sycl|cpu-ref` (default: sycl when built), `--device GPU|GPU.0|GPU.1`, `--precision f16|f32`,
  `--kv-dtype auto|i8`, `--ctx N`, `--prefill-chunk N`, `--backend-opt attention=auto|rows|gemm`, `--no-vision`, `--log-level debug`. `xpu-serve <command> --help` lists all of them;
  `src/cli/main.cpp` is the source of truth.
- Run one test case: `build/tests/xs_tests -tc='model matches*'`. Environment for the tests that need files:
  `XS_TEST_MODEL_DIR` (a Hugging Face model directory), `XS_PARITY_BACKEND` (`cpu-ref` or `sycl`),
  `XS_PARITY_DEVICE`, `XS_PARITY_PRECISION`, `XS_PARITY_FIXTURES`, `XS_PARITY_KV_DTYPE`; `XS_QWEN35_DIR` (a Qwen3.5
  checkpoint, for tokenizer parity). `tests/parity/model/make_tiny.py` writes a tiny random checkpoint of an
  architecture (real layer pattern and tensor names) to run `hf_reference.py` and the parity test on in seconds.
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
├── CMakeLists.txt          options XS_WITH_SYCL, XS_SYCL_TARGETS, XS_BUILD_TESTS, XS_WERROR; C++20; warnings target
├── README.md               user-facing intro and quick start
├── CLAUDE.md               this file
├── DESIGN.md               the design (why), roadmap, open questions
├── TODO.md                 phase checklists; [x] = done and tested
├── docs/decisions/         one short record per decision (NNNN-title.md)
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
| `tensor/` | xs_core | `DType` (incl. `Q8` / `Q4` pack weights), `TensorView` (non-owning, e.g. into an mmap; `quant` points at a pack weight's scales and biases), `HostTensor` (owning), `TensorMap` (name -> HostTensor) |
| `io/` | xs_core | `safetensors`: `WeightStore` interface, `SafetensorsWeights::open_dir` (single file or sharded index, mmapped; a pack's quantized weights appear as one Q8 / Q4 view each); `safetensors_writer` (streamed writing) |
| `quant/` | xs_core | round-to-nearest quantization into the pack formats (DESIGN.md 6.3) and dequantization; a tensor `[..., K]` is a matrix of `numel / K` rows (expert banks `[E, N, K]` included) |
| `convert/` | xs_core | `convert_model`: checkpoint (language model or policy) -> pack; the model's own graph decides which weights to quantize (`plan_graph_formats`: bandwidth-bound matmuls, or all with `--scope all`) |
| `image/` | xs_core | `Image` (8-bit RGB), `decode` (stb_image), `resize_bicubic` (bit-exact with torchvision), `content_hash` |
| `graph/` | xs_core | the step graph IR and builder (`graph.h` holds the op spec) |
| `models/` | xs_core | `model.h` (the `Model` interface: language models), `policy.h` (the `Policy` interface: robot policies, one static program per call shape: camera subset, step count, padded prompt length), `registry.cpp` (model_type / LeRobot type -> code, `model_kind`), `config_util` (checked JSON readers), `common/blocks` (the shared building blocks: gated MLPs, mixture of experts, RMSNorm variants, RoPE frequencies, attention), one folder per architecture (`pi05/`: π0.5) |
| `engine/` | xs_core | `Engine` (one sequence: prefix reuse, chunked prefill, image encoding, decode loop), `Sampler`, request/result types (`MediaInput`), `media` (placeholder expansion), `service.h` (the `InferenceService` interface the server uses) |
| `backend/` | xs_backends | `backend.h` (Backend/Session interfaces), `registry.cpp`, `cpu_ref/`, `sycl/` |
| `tokenizer/` | xs_tokenizer | Hugging Face `tokenizer.json` BPE, natively: normalizers (Replace, NFC with generated Unicode tables), pre-tokenizers (Split with hand-written GPT-style regexes, ByteLevel), byte fallback or byte-level decoding, added tokens, streaming decoder |
| `chat/` | xs_chat | `ChatTemplate` (renders the model's Jinja template with vendored jinja.cpp, in the request's key order), output parsers per dialect: `gemma4`, `qwen3_5` (thinking, tool calls), `plain` |
| `server/` | xs_server | HTTP (cpp-httplib), OpenAI request parsing and response formatting, stop strings, streaming |
| `service/` | xs_service | `EngineService`: loads tokenizer + template + model, compiles on a backend, owns the engine thread |
| `policy/` | xs_policy | `PolicyRunner`: a policy, its tokenizer and one compiled program per call shape; warm-up; per-call timings (`xpu-serve bench <policy-dir>`) |
| `cli/` | (exe) | `main.cpp` (commands serve, run, bench, convert, info), `args` (flag parser) |

### 4.1 Files that matter most

| File | Role |
|---|---|
| `src/graph/graph.h` | **The op contract.** Every op's semantics are written next to `OpKind`; every backend must implement exactly that |
| `src/models/model.h` | What a model provides: `info()`, `graph()`, `weights()`, `prepare_inputs()`, and optionally `vision()` (a `VisionEncoder`: preprocessing, encoder graph, prompt layout of an image) |
| `src/models/registry.cpp` | Dispatch on `config.json` `model_type`; the list of supported types |
| `src/models/gemma4/gemma4_model.cpp` | Gemma 4 text model: graph builder + per-step host inputs |
| `src/models/gemma4/gemma4_config.cpp` | Gemma 4 `config.json` parsing; refuses unsupported variants by name |
| `src/models/gemma4/gemma4_vision.cpp` | Gemma 4 vision: image processor (resize, patches, pooling map) and vision tower graph |
| `src/models/qwen3_5/qwen3_5_model.cpp` | Qwen3.5: Gated DeltaNet layers (causal conv + gated delta rule) and gated full attention; dense MLP or `blocks::moe` (`qwen3_5_moe`: fused gate_up expert banks, gated shared expert) |
| `src/backend/backend.h` | `Backend::compile(graph, weights, options) -> Session`; `Session::run/reset/truncate` |
| `src/backend/cpu_ref/ops.cpp` | f32/f64 reference kernels, one per op: the accuracy oracle |
| `src/backend/sycl/sycl_backend.cpp` | SYCL session: device dtypes, fusions, per-T memory plan (one arena), op dispatch, KV and state buffers |
| `src/backend/sycl/sycl_kernels.cpp` | SYCL kernels: elementwise, norms, RoPE, gather, copies, causal conv, gated delta, decode attention, GEMV |
| `src/backend/sycl/sycl_matmul.cpp` | MatMul (GEMV up to 8 rows, oneMKL GEMM beyond; quantized: quant_gemm up to 64 rows) and Attention (per-row kernel with split-K, the fused kernel, or GEMMs) |
| `src/backend/sycl/sycl_flash_attention.cpp` | Fused attention on the matrix engines (joint_matrix; head_dim up to 128) |
| `src/backend/sycl/sycl_quant_gemm.cpp` | Quantized matmul of 9 to 64 rows reading the q8 / q4 weight once (DPAS and 2D block load builtins) |
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
- Ops: Input, Weight, Constant, Gather, MatMul, Add, Mul, Div, Scale, Tanh, GeluTanh, Softcap, Sigmoid, Silu, Softplus,
  Clamp, RmsNorm, Softmax, TopK, ReduceSum, Reshape, Slice, Transpose, Rope (1-D or axial, optionally partial), KvCache,
  Attention (causal with optional window, or non-causal), CausalConv and GatedDelta (stateful, for linear attention),
  ExpertMatMul (each token's chosen experts from a stacked `[E, N, K]` bank, for mixture of experts). The exact semantics
  are the comments in `graph.h`.
- KV: `add_kv_slot(...)` declares per-sequence state; `kv_cache(slot, k, v)` appends this step's K/V and returns the
  whole cache. A slot can be read by later layers (Gemma 4's KV-shared layers reuse an earlier layer's outputs).
- Graph inputs are filled per step by `Model::prepare_inputs` (token ids, positions, and model-specific host inputs
  such as Gemma 4's per-layer embedding rows, and the image rows that replace soft-token embeddings).
- Conv and recurrent slots (`add_conv_slot`, `add_recurrent_slot`) hold fixed-size per-sequence state for
  `causal_conv` and `gated_delta`. They summarize the whole sequence, so a session cannot cut them back to a prefix:
  the engine reuses a cached prefix for such models only when the new prompt extends it, else it starts over.
- A graph without slots is stateless (every run stands alone): that is how vision encoders run, one image per run,
  compiled as a second session next to the text one.

**Backends** (`src/backend/`):

| Backend | Status | Notes |
|---|---|---|
| `cpu-ref` | test oracle | f32 values, f64 accumulation, threaded matmul. Slow by design; correctness only |
| `sycl` | main GPU backend, the default when built (icpx turns it on) | Runs the graph op by op on an in-order Level Zero queue with its own kernels; oneMKL for prefill GEMMs. Weights and KV f16 (f32 at f32 precision), activations f32; `--kv-dtype i8` stores KV as int8 with an f32 scale per (token, KV head): decode attention reads the codes, prefill dequantizes the keys in view to f16 for its GEMMs. Memory: one arena per step, planned per T from value lifetimes; KV slots grow by doubling; conv/recurrent state double-buffered so a failed step changes nothing. Fuses act(x)*gate and conv+SiLU. Decode attention reads only the sliding window and splits long caches across work-groups. Option `attention=auto|rows|gemm`. See docs/decisions/0007 |

## 7. Tests (`tests/`)

One test binary, `xs_tests` (doctest). Every `tests/unit/**/*.cpp` is compiled in automatically.

| Path | What it checks | Against |
|---|---|---|
| `tests/unit/test_graph.cpp`, `test_half.cpp`, `test_base64.cpp` | IR shape inference and validation; f16/bf16 conversions; base64 | hand-written expectations, RFC 4648 vectors |
| `tests/unit/image/` | decoding, resize, hashing | Pillow-written images; torchvision resize outputs in `tests/fixtures/image/` (bit-exact) |
| `tests/unit/backend/cpu_*.cpp` | each op on cpu-ref (`cpu_encoder_ops.cpp`: the encoder ops and stateless sessions), KV sessions, attention masks | fp64 references in `reference_ops.h`, with planted-bug controls |
| `tests/unit/backend/device_*.cpp` | each op on the GPU backend (`device_encoder_ops.cpp` likewise), sessions, agreement with cpu-ref (also with an int8 KV cache on both), decode timing | the same references; skip unless `XS_DEVICE_TEST_DEVICE` is set |
| `tests/unit/backend/*_moe_ops.cpp` | Div, Softmax, TopK (ties to the lower index), ReduceSum, ExpertMatMul and the `moe` block (softmax and sigmoid routers, shared expert), on cpu-ref and the GPU | fp64 references and an fp64 composition of the block (`moe_ops_cases.h`) |
| `tests/unit/backend/*_recurrent_ops.cpp` | sigmoid/silu/softplus, partial rope, CausalConv and GatedDelta over several steps (state carried), on cpu-ref and the GPU | one fp64 pass over the whole sequence (`recurrent_ops_cases.h`) |
| `tests/unit/tokenizer/` | encode/decode, streaming; NFC and regex Split hermetically (`test_tokenizer_qwen35.cpp`) | Hugging Face `tokenizers` outputs in `tests/fixtures/tokenizer/` (`tests/parity/tokenizer/gen_cases.py`, `gen_qwen35_cases.py`; tables from `gen_unicode_tables.py`) |
| `tests/unit/chat/` | template rendering, Gemma 4 and Qwen3.5 output parsing | `apply_chat_template` outputs in `tests/fixtures/chat/` |
| `tests/unit/server/` | request parsing, response format, stop strings, HTTP end to end | a fake `InferenceService` (`fake_service.h`) |
| `tests/unit/model/gemma4_config.cpp`, `qwen3_5_config.cpp` | config parsing and refusals | |
| `tests/unit/model/model_parity.cpp` | whole model: last-token logits and greedy continuations | Hugging Face fp32, from `tests/parity/model/hf_reference.py` |
| `tests/unit/model/policy_parity.cpp` | a robot policy end to end: prompt ids and the action chunk, all cameras and one camera (`XS_TEST_POLICY_DIR`, `XS_POLICY_FIXTURES`) | LeRobot fp32 with fixed noise, from `tests/parity/policy/lerobot_reference.py` |
| `tests/unit/backend/*_policy_ops.cpp` | LayerNorm, Concat, attention with a key bias over static queries (head dims 72, 256) | fp64 references (`policy_ops_cases.h`) |
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
     Build layers from `models/common/blocks.h` (`attention`, `gated_mlp`, `moe`, `rms_norm_one_plus`,
     `gated_rms_norm`, `rope_inv_freq`): the model maps its weight names onto a block's projections and norms; the math lives once.
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

- Two model families: Gemma 4 E2B/E4B (text and images) and Qwen3.5 dense and MoE (text; its vision tower and MTP
  head are not loaded; MoE checkpoints must use fused expert banks, as the released ones do). No audio or video. Images only as base64 data: URLs.
- Linear-attention models re-prefill when a new prompt does not extend the cached sequence (no state snapshots).
- Quantization is round-to-nearest only (no AWQ / GPTQ yet). Gemma 4's per-layer embedding table (read on the host)
  and vision towers stay bf16 in a pack. Prefill dequantizes each quantized weight to f16 before its GEMM.
- One sequence at a time; no batching.
- Sliding-window layers keep every key in memory (the kernels read only the window; memory still grows with context).
