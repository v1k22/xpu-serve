<p align="center">
  <img src="docs/images/logo.png" alt="xpu-serve: one repo, one binary, Intel GPUs" width="900">
</p>

# xpu-serve
Experimental engine to run LLMs on Intel platforms.

One repo, one binary: the model code, the engine, the tokenizer, chat templates and an OpenAI-compatible server all
live here, and the math runs on Intel GPUs through our SYCL backend (our own kernels, oneMKL for large matmuls). Read
[DESIGN.md](DESIGN.md) for the why and [TODO.md](TODO.md) for the plan.

**Status: v0.4.** Language models, decision models, embedding models and a robot policy, one request at a time, on
Intel GPUs. **[model_recipes.md](model_recipes.md) has the commands for every model (the reference setting and the
optimized one) and what was measured**; this page says what is supported and how the APIs look.
[CLAUDE.md](CLAUDE.md) maps the code.

> 🚀 **A 27B model's full context on one 24 GB card.** Qwen3.8-27B (its Q4_0 GGUF file, with its MTP drafter) reads all
> 262,144 tokens it can take on a single Intel Arc Pro B60, with the KV cache in TurboQuant's `turbo4`. With an f16
> cache the same card stops at 64k tokens, with int8 at 128k.
>
> ```bash
> xpu-serve serve ~/models/Qwen3.8-27B-Q4_0 --device GPU --spec mtp --spec-k 4 --ctx 262144 --kv-dtype turbo4
> ```
>
> The format is lossy and opt-in; what it costs and what was measured:
> [model_recipes.md](model_recipes.md#measured-on-qwen38-27b-a-gguf-pack).

## What is supported

- ✅ **OpenAI-compatible server**: `/v1/chat/completions` (streaming, tool calls, thinking), `/v1/completions`,
  `/v1/models`
- ✅ **Language models**: Gemma 4 (E2B, E4B, 26B-A4B), Qwen3.5 / Qwen3.6 / Qwen3.8 (dense and mixture of experts),
  LFM3
- ✅ **Image input** (Gemma 4 E2B / E4B), as base64 `data:` URLs
- ✅ **Quantized packs**: q8 and q4 from `xpu-serve convert` (round to nearest; AWQ calibration opt-in)
- ✅ **FP8 checkpoints** (Qwen3.8-27B-FP8, Qwen3.6-35B-A3B-FP8), served as they are or converted to a pack
- ✅ **GGUF import**: `qwen35`, `gemma4`, `gemma4-assistant`, `clef` and the `modern-bert` files of Laya and Julia-1,
  in all 19 of GGUF's block formats, kept as the file stores them
- ✅ **Speculative decoding**: Qwen's MTP heads, Gemma 4's assistants, DFlash 2 and DSpine block drafters
  (Qwen3.8-27B)
- ✅ **int8 KV cache** (`--kv-dtype i8`)
- ✅ **TurboQuant KV cache** (`--kv-dtype turbo4 | turbo3 | turbo2`, or int8 keys with TurboQuant values): a cache
  3.9 to 7.5 times smaller than f16
- ✅ **Decision models** behind the System One API (`POST /v1/systemone`): Clef-Flash, Clef, pplx-decider-flash,
  OpenJev, d1, Laya, Julia-1, or any chat model read at the option letters
- ✅ **Embedding models** behind `/v1/embeddings`: all-MiniLM-L6-v2, bge-small-en-v1.5, nomic-embed-text-v1.5
- ✅ **Robot policy** (π0.5) behind the policy API
- ✅ **Prefix reuse**: a request keeps the KV cache of the longest prefix it shares with the one before
- ✅ **Intel Arc GPUs**, discrete (Arc Pro B60) and integrated (Arc B390), through SYCL

Not yet:

- Several sequences at once (batching): one request runs at a time.
- One model across two GPUs.
- Audio and video input; images for Qwen3.5 (its vision tower is not loaded) and Gemma 4 26B-A4B; images from
  remote URLs.
- A mixture of experts from a GGUF file (`qwen35moe`, Gemma 4 26B-A4B).
- `response_format` (JSON mode), `n` above 1, a rerank API.

## Supported models

Each model's recipe has its commands and numbers.

| Model | Kind | Checkpoint | Runs from | Recipe |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | language, images | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` | original, q8 / q4 pack, GGUF (text) | [E2B](model_recipes.md#gemma-4-e2b), [E4B](model_recipes.md#gemma-4-e4b), [GGUF](model_recipes.md#gemma-4-e2b-gguf) |
| Gemma 4 26B-A4B | language, MoE | `google/gemma-4-26B-A4B-it` | q4 pack | [recipe](model_recipes.md#gemma-4-26b-a4b) |
| Qwen3.5-4B | language | `Qwen/Qwen3.5-4B` | original, q8 / q4 pack, GGUF | [recipe](model_recipes.md#qwen35-4b), [GGUF](model_recipes.md#qwen35-4b-gguf) |
| Qwen3.5-9B | language | `Qwen/Qwen3.5-9B` | original, q8 / q4 pack | [recipe](model_recipes.md#qwen35-9b) |
| Qwen3.6-35B-A3B | language, MoE | `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-35B-A3B-FP8` | q4 pack | [recipe](model_recipes.md#qwen36-35b-a3b) |
| Qwen3.8-27B | language | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-FP8` | q4 pack, GGUF | [recipe](model_recipes.md#qwen38-27b), [GGUF](model_recipes.md#qwen38-27b-gguf) |
| LFM3-1.7B | language | `LiquidAI/LFM3-1.7B` | original, q8 / q4 pack | [recipe](model_recipes.md#lfm3-17b) |
| Clef-Flash | decision | `Cloudflare/clef-flash` | original, q8 / q4 pack, GGUF | [recipe](model_recipes.md#clef-flash) |
| Clef | decision | `ggml-org/Clef-GGUF` (of `Cloudflare/clef`) | GGUF Q4_K_M | [recipe](model_recipes.md#clef) |
| pplx-decider-flash-v1.2 | decision | `perplexity-ai/pplx-decider-flash-v1.2` | original, q8 / q4 pack | [recipe](model_recipes.md#pplx-decider-flash-v12) |
| OpenJev | decision (a language model read at the option letters) | `openjev/openjev` | q4 pack | [recipe](model_recipes.md#openjev) |
| d1-2B | decision | `LiquidAI/d1-2B` | original, q8 / q4 pack | [recipe](model_recipes.md#d1-2b) |
| Laya | decision | `convaiinnovations/laya` | original, GGUF | [recipe](model_recipes.md#laya) |
| Julia-1 | decision | `SupersonicLabs/Julia-1` | original, GGUF | [recipe](model_recipes.md#julia-1) |
| all-MiniLM-L6-v2 | embedding | `sentence-transformers/all-MiniLM-L6-v2` | original | [recipe](model_recipes.md#all-minilm-l6-v2) |
| bge-small-en-v1.5 | embedding | `BAAI/bge-small-en-v1.5` | original | [recipe](model_recipes.md#bge-small-en-v15) |
| nomic-embed-text-v1.5 | embedding | `nomic-ai/nomic-embed-text-v1.5` | original | [recipe](model_recipes.md#nomic-embed-text-v15) |
| π0.5 | robot policy | `lerobot/pi05_base` | original, q8 pack | [recipe](model_recipes.md#π05) |

"Original" is the Hugging Face checkpoint at f16; a model listed without it is larger than a 24 GB card at f16 and
was run from the form named. Some checkpoints carry their own license (OpenJev's weights: CC BY-NC 4.0; d1's
weights, and LFM3's tokenizer and chat template: the LFM Open License v1.0); none of them is in this repository.

A model is added in `src/models/<arch>/` (language models: [CLAUDE.md](CLAUDE.md); policies:
[docs/design-vla.md](docs/design-vla.md); decision models:
[docs/decisions/0015](docs/decisions/0015-decision-models.md)); this table lists what has been enabled and measured.

## Build

Needs CMake 3.22+, Ninja and the oneAPI toolkit 2026.2 (DPC++ compiler `icpx` and oneMKL), and an Intel GPU driver
with Level Zero.

```bash
source /opt/intel/oneapi/setvars.sh                      # oneAPI 2026.2
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
cmake --install build --prefix ~/.local                  # the xpu-serve command, in ~/.local/bin
build/tests/xs_tests                                     # unit tests (XS_DEVICE_TEST_DEVICE=GPU.0 adds the GPU suites)
```

The build leaves the binary at `build/src/xpu-serve`; the install step copies it to `<prefix>/bin` (`/usr/local/bin`
without `--prefix`, which needs `sudo`). With that directory on `PATH` the command is `xpu-serve`, as in the examples
below. It finds the oneAPI runtime through the environment, so run it in a shell that has sourced `setvars.sh`.

Add `-DXS_SYCL_TARGETS=intel_gpu_bmg_g21` to compile the SYCL kernels ahead of time for the Arc Pro B60 (otherwise
they compile at first use). Without oneAPI, any C++20 compiler (g++ 11 or newer) builds everything but the GPU
backend: the CPU reference backend, which is the test oracle and far too slow to serve.

## Run

Any Hugging Face directory of a supported model works, e.g. `google/gemma-4-E2B-it` or `Qwen/Qwen3.5-4B`:

```bash
xpu-serve info                                           # backends and devices (the first is the default)
xpu-serve run   ~/models/gemma-4-E2B-it --device GPU     # chat in the terminal
xpu-serve run   ~/models/gemma-4-E2B-it --device GPU --image photo.jpg --prompt "What is in this picture?"
xpu-serve serve ~/models/gemma-4-E2B-it --device GPU     # http://127.0.0.1:8080/v1
xpu-serve bench ~/models/gemma-4-E2B-it --device GPU --prompt-tokens 1024 --gen-tokens 128
```

That is the reference setting: the checkpoint's own weights at f16, an f16 KV cache. The steps below make a model
smaller and faster; [model_recipes.md](model_recipes.md) has the combination to use for each model.
`xpu-serve <command> --help` lists every option.

### Quantized packs (`convert`)

`convert` writes a pack: the checkpoint with its matmul weights quantized, everything else copied. Every command
takes a pack wherever it takes a model directory; the model code is the same.

```bash
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q8 --quant q8
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
xpu-serve serve ~/models/Qwen3.5-4B-q8 --device GPU
```

- `q8` is symmetric int8; `q4` is 4-bit with a scale and bias per group of 32 columns (`--group 64` or `128` reads
  fewer bytes). Both round to nearest by default; `--method awq` adds an AWQ-style search calibrated on the model
  itself (on the GPU; `--calib file` brings your own text), which helps Qwen3.5 but not yet Gemma 4
  ([docs/decisions/0009](docs/decisions/0009-quantized-packs.md)).
- Embedding tables are q8 in either case, small matrices and non-matmul tensors stay as they are, and `--q8 name,...`
  keeps chosen weights (parts of their names) at q8. The format of every tensor is in the pack's `xpu_pack.json`.
- Which matmul weights are quantized follows one rule for every model, read off the model's own graph: a weight whose
  matmuls read it more than they compute (a step-dependent number of rows, as in decode, or at most 64 rows per call)
  is quantized, because a smaller weight makes those matmuls faster; weights of compute-bound matmuls (an image
  encoder, a policy's prefix over hundreds of tokens) stay as they are. `--scope all` quantizes every matmul weight
  instead (the smallest pack). For π0.5 the rule picks the action expert, which runs 50 tokens per flow step.
- The source can be the model's FP8 release instead of the original (`convert ~/models/Qwen3.8-27B-FP8 --out ...`):
  half the download, the same pack layout and speed, and logits as close to Hugging Face's
  ([docs/decisions/0014](docs/decisions/0014-fp8-checkpoints.md)). An FP8 checkpoint also serves as it is, as f16 on
  the device like the original, so it fits where the original fits.
- A decision model converts like a language model; embedding models, Laya and Julia-1 have no pack.

### From a GGUF file

`convert` also imports a GGUF file (llama.cpp's format) into a pack. Each quantized tensor stays exactly as the file
stores it (Q4_K, Q6_K, IQ3_S, ...: nothing is decoded or quantized again), so the model has the file's weights and
takes the file's size in GPU memory:

```bash
xpu-serve convert ~/models/Qwen3.5-4B-Q4_K_M.gguf ~/models/Qwen3.5-4B-Q4_K_M
xpu-serve serve ~/models/Qwen3.5-4B-Q4_K_M --device GPU
```

- The architecture comes from the file (`general.architecture`). Supported: `qwen35`, the Qwen3.5 / Qwen3.6 / Qwen3.8
  dense text models; `gemma4`, Gemma 4 E2B and E4B (the text model: a GGUF file holds no vision or audio tower, so
  images need the original checkpoint), with its assistant (`gemma4-assistant`, the `MTP/mtp-*.gguf` files) as the
  drafter; `clef`, the Clef decision models; and `modern-bert` with a Laya read-out, the files of Laya and Julia-1.
  Others are refused by name, as is a mixture of experts in a block format.
- A Qwen3.5-family file that comes without its MTP head (the smallest files of Qwen3.8-27B) takes the head from its
  `MTP/mtp-*.gguf` file with `--mtp`: the pack is the one a file with the head gives, and `--spec mtp` drafts with it.
- config.json, the tokenizer and the chat template are written from the file's metadata (on Qwen3.5-4B and Gemma 4
  E2B the tokenizer equals the original's token by token), and the template is the file's, which a publisher may have
  changed from the original checkpoint's. Thinking, tool calls and streaming work as with the original checkpoint.
  The quantization options do not apply.
- Both backends read all 19 of GGUF's block formats (Q4_0 to Q8_0, the K-quants, IQ1 to IQ4), each decoded to the
  values llama.cpp's own code gives (checked bit for bit on the host). A file is usually a mix of them.

Commands per model: [model_recipes.md](model_recipes.md#importing-a-gguf-file). Every file's numbers:
[docs/gguf_models.md](docs/gguf_models.md).

### Speculative decoding

`--spec` (on `run`, `serve` and `bench`) drafts with the model's drafter and verifies the drafts in one step; greedy
output stays the model's own and sampling keeps its distribution (rejection sampling).

| `--spec` | Drafter | Where it comes from |
|---|---|---|
| `mtp` | the MTP head of a Qwen3.5-family checkpoint | inside the model (and its pack) |
| `mtp` | a Gemma 4 assistant | `--drafter <google/gemma-4-*-it-assistant>` |
| `dflash` | a DFlash or DFlash 2 block drafter | `--drafter <z-lab/Qwen3.8-27B-DFlash2>` |
| `dspine` | a DSpine block drafter | `--drafter <tencent/Qwen3.8-27B-DSpine-b16>` |

```bash
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec mtp --spec-k 4
xpu-serve run   ~/models/gemma-4-E2B-it --device GPU --spec mtp --spec-k 1 --drafter ~/models/gemma-4-E2B-it-assistant
```

A drafter is trained for one model, so a technique works only where a drafter for that model exists (elsewhere the
option is refused by name). `--spec-k` is the number of drafts per round; which model takes which technique, the
value to pass and the speed it gives: [model_recipes.md](model_recipes.md#speculative-decoding). The design:
[docs/design-spec-decode.md](docs/design-spec-decode.md).

### KV cache formats

`--kv-dtype` (on `serve`, `run` and `bench`) stores the KV cache in fewer bits, for longer contexts in the same
memory: `i8` (int8 with one scale per token and KV head), or TurboQuant's `turbo4`, `turbo3` and `turbo2` (4, 3 or 2
bits a value); `<keys>:<values>` sets the two apart, e.g. `i8:turbo4`. All of them are lossy and opt-in; the default,
`auto`, is f16. What each costs and saves: [model_recipes.md](model_recipes.md#kv-cache-formats); the designs:
[docs/decisions/0010](docs/decisions/0010-int8-kv-cache.md) (int8) and
[0020](docs/decisions/0020-turboquant-kv-cache.md) (TurboQuant).

## APIs

`xpu-serve serve <dir>` serves whatever the directory holds: a language model behind the OpenAI API, an embedding
model behind `/v1/embeddings`, a decision model behind the System One API, a robot policy behind the policy API.
`--host` and `--port` set the address (127.0.0.1:8080 by default), `--api-key` asks for `Authorization: Bearer <key>`,
`--model-id` names the model in `/v1/models`. `GET /health` answers once the server is up. A request that names
something unsupported is refused with the field's name, never served differently.

### Chat completions

`POST /v1/chat/completions`, `POST /v1/completions` and `GET /v1/models`, so any OpenAI client can point at the
server:

```bash
xpu-serve serve ~/models/Qwen3.5-4B --device GPU
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

- **Streaming** (`"stream": true`, server-sent events; `stream_options.include_usage` adds the usage chunk).
- **Tool calls**: `tools` with `tool_choice` `auto` or `none`; calls come back as `tool_calls`, also streamed.
- **Thinking**: a model's reasoning comes back as `reasoning_content`, apart from `content`;
  `chat_template_kwargs` reaches the chat template (`{"enable_thinking": false}` turns thinking off where a model
  thinks by default).
- **Images**: `image_url` parts with a base64 `data:` URL (PNG, JPEG, BMP or GIF); the server does not fetch remote
  URLs.
- **Sampling**: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repetition_penalty`, `seed`, `stop`, `max_tokens` / `max_completion_tokens`.
- A response carries `usage` and `timings`. One choice per request (`n` is 1); `response_format` other than text and
  audio or video parts are refused by name.

### Embeddings

A sentence-transformers checkpoint of a BERT or NomicBERT encoder is served behind OpenAI's embeddings API, the one
vLLM and llama.cpp serve too:

```bash
xpu-serve serve ~/models/bge-small-en-v1.5 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["What is a GPU?", "A graphics processing unit runs many threads at once."]}'
```

```json
{"object": "list", "model": "bge-small-en-v1.5",
 "data": [{"object": "embedding", "index": 0, "embedding": [0.0113, 0.0305, -0.0081, ...]},
          {"object": "embedding", "index": 1, "embedding": [-0.0246, -0.0259, 0.0025, ...]}],
 "usage": {"prompt_tokens": 20, "total_tokens": 20}}
```

(bge-small-en-v1.5's answer on an Arc Pro B60, 384 values each, shortened and rounded here.)

- `input` is a string, a list of strings, a list of token ids or a list of such lists (ids are read as given).
  `encoding_format` is `float` or `base64` (float32, little-endian: what the OpenAI Python client asks for).
- The pipeline is the checkpoint's own (`modules.json`): its pooling (mean for MiniLM and nomic, CLS for bge) and L2
  normalization only when it lists a Normalize module (MiniLM, bge; not nomic), as sentence-transformers and vLLM do;
  llama.cpp normalizes every model.
- An input longer than the model reads (256, 512 and 8,192 tokens) is refused with its token count, not cut;
  `dimensions` is accepted only at the model's own size. Each input is one run of the encoder.
- nomic-embed-text-v1.5 expects its task prefixes in the text (`search_query: `, `search_document: `).

`xpu-serve bench <dir>` reports the latency of one text at several lengths and the throughput of a batch; `run <dir>
--prompt "..."` prints one embedding. Decision [0017](docs/decisions/0017-embedding-models.md) has the details.

### Decisions: the System One API

A decision model answers typed questions about a state instead of writing text: a yes / no (`noul`), a `choice`
among options you name, or a `score` on levels you order, each as probabilities read off the model in one pass, with
nothing to parse. `serve` puts one behind TypeSafe's System One API, which clients written for Jev, Clef or vLLM's
`/v1/systemone` already speak:

```bash
xpu-serve serve ~/models/clef-flash --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef-flash",
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route":   {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges and refunds", "shipping": "Delivery and tracking", "technical": null}},
    "angry":   {"type": "noul", "instructions": "Is the customer angry?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today"]}
  }
}'
```

```json
{"model": "clef-flash",
 "answers": {
  "route":   {"type": "choice", "choice": "billing", "confidence": 0.966,
              "probabilities": {"billing": 0.977, "shipping": 0.014, "technical": 0.009}},
  "angry":   {"type": "noul", "noul": 0.423},
  "urgency": {"type": "score", "score": 1.621, "confidence": 0.431, "legend": {"0": "can wait", "1": "this week", "2": "today"},
              "probabilities": {"0": 0.048, "1": 0.284, "2": 0.669}}},
 "usage": {"input_tokens": 320, "output_tokens": 0}}
```

(Clef-Flash's answer, rounded here to three digits.)

- The `state` and every `instructions` and description may be a string or any JSON. `confidence` follows the API's
  formulas for every model; probabilities are not rounded.
- A body that does not follow the schema gets the API's 422 with the path of the field, one longer than `--ctx` its
  400 (`max_tokens_exceeded`): nothing is cut to fit. Each response has its timings in a `Server-Timing` header.
- Images (`assets`, box questions) are refused by name: the Qwen3.5 vision tower is not loaded yet.
- The models: Clef-Flash and Clef (every question of a request in one pass, a transformer head scores the options),
  pplx-decider-flash (one pass, each question read at a marker), d1 (questions packed after one state), Laya and
  Julia-1 (small encoders, a sequence per question), and OpenJev or any chat model read at the option letters
  (`--readout openjev | letters`). How each decides, its command and its latency:
  [model_recipes.md](model_recipes.md#decision-models).
- Without a server, `xpu-serve run <dir> --request request.json` answers one request file and `bench` times one.

### Robot policy

A robot policy runs from its LeRobot directory (with the tokenizer as `tokenizer.json` beside it); `bench` times
calls on synthetic camera images after a warm-up, for the cameras present and the flow steps asked for, and `serve`
puts it behind the policy API:

```bash
xpu-serve bench ~/models/pi05_base --device GPU.0 --precision f16 --cameras 3 --steps 10 --runs 10
xpu-serve serve ~/models/pi05_base --device GPU.0 --cameras 2    # warm up for the first 2 cameras
```

`GET /v1/policy/info` describes the policy (cameras and their sizes, state and action sizes, flow steps);
`POST /v1/policy/act` takes the cameras (base64 PNG or JPEG, or raw RGB), the state and the instruction (optionally a
seed, the flow steps or fixed noise) and returns the action chunk with per-stage timings; `GET /ready` answers 200 once
warmed up. Calls run one at a time and the latest observation wins: a call still waiting when a newer one arrives is
answered 409. Requests fail by name (an unknown camera, a wrong state length, an image of the wrong size).

## Correctness

Every layer is tested against a reference:

| What | Reference | Where |
|---|---|---|
| Tokenizer (Gemma BPE, byte-level BPE with NFC, LFM3's, BERT WordPiece, ModernBERT's and mmBERT's) | Hugging Face `tokenizers`, exact ids and text | `tests/unit/tokenizer/` |
| Chat template | `apply_chat_template`, byte-exact | `tests/unit/chat/` |
| Image resize | torchvision's antialiased bicubic, bit-exact | `tests/unit/image/` |
| Every graph op, on cpu-ref and on the GPU | fp64 references, with planted-bug controls | `tests/unit/backend/` |
| Whole model (Gemma 4, Qwen3.5, LFM3) | Hugging Face fp32 logits and greedy output | `tests/unit/model/model_parity.cpp` |
| Image input | Hugging Face fp32 pixels, vision features, logits and greedy output | `tests/unit/model/gemma4_vision_parity.cpp` |
| Robot policy (π0.5) | LeRobot fp32 actions and prompt token ids, 3 cameras and 1 | `tests/unit/model/policy_parity.cpp` |
| Decision models | the checkpoint's own Python code in fp32: prompt token ids and every option's probability | `tests/unit/model/decision_parity.cpp` |
| Embedding models | sentence-transformers in fp32: token ids, hidden states and embeddings | `tests/unit/model/embedding_parity.cpp` |
| GGUF block formats | ggml's own dequantization, bit for bit | `tests/unit/quant/test_block_decode.cpp` |
| System One API | the API's schema and error bodies; 25 decisions with accepted answers through a running server | `tests/unit/server/test_systemone_http.cpp`, `tests/e2e/` |

What each model was checked against is in its recipe. The model parity test needs the model and generated fixtures:

```bash
python -I tests/parity/model/hf_reference.py --model-dir ~/models/gemma-4-E2B-it          # needs transformers
python -I tests/parity/model/hf_vision_reference.py --model-dir ~/models/gemma-4-E2B-it   # image cases
XS_TEST_MODEL_DIR=~/models/gemma-4-E2B-it XS_PARITY_BACKEND=sycl XS_PARITY_DEVICE=GPU.0 build/tests/xs_tests -tc='model*'
```

## More

- [model_recipes.md](model_recipes.md): the commands and the measured numbers, model by model.
- [docs/gguf_models.md](docs/gguf_models.md): every GGUF file that was measured.
- [DESIGN.md](DESIGN.md) and [docs/decisions/](docs/decisions/): why things are the way they are.
- [CLAUDE.md](CLAUDE.md): the map of the code, and how to add a model, an op or an API field.
- [TODO.md](TODO.md): what is planned.
