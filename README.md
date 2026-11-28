<p align="center">
  <img src="docs/images/logo.png" alt="xpu-serve" width="360">
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
> xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --spec mtp --spec-k 4 --ctx 262144 --kv-dtype turbo4
> ```
>
> The format is lossy and opt-in; what it costs and what was measured:
> [model_recipes.md](model_recipes.md#measured-on-qwen38-27b-a-gguf-pack). Ornith-1.5-35B-A3B, a 35B mixture of
> experts, does the same from its Q4_K_M GGUF file (22.8 GiB at 262,144 tokens;
> [its recipe](model_recipes.md#ornith-15-35b-a3b-gguf)). Inkling-2-Mini, a 26B mixture of experts, reads all
> 1,048,576 tokens it takes on the same card, and on an Arc B390 integrated GPU, from its UD-Q4_K_M file in `turbo3`
> (perplexity +2.0 %; [its recipe](model_recipes.md#inkling-2-mini)).

## What is supported

- ✅ **Models straight from the Hugging Face Hub**: `-hf org/name`, or `-hf org/name:QUANT` for a GGUF file, into
  the Hugging Face cache (shared with huggingface_hub), resumable and checked
- ✅ **OpenAI-compatible server**: `/v1/chat/completions` (streaming, tool calls, thinking, structured outputs,
  log-probabilities), `/v1/completions`, `/v1/models`; and Anthropic's Messages API, `/v1/messages` (content blocks,
  tools, thinking, streamed events)
- ✅ **Language models**: Gemma 4 (E2B, E4B, 26B-A4B), Qwen3.5 / Qwen3.6 / Qwen3.8 (dense and mixture of experts),
  Ornith-1.5-35B-A3B, SenseNova-Loop-9B (layers run in a loop inside hyper-connections), LFM3, Inkling 2 Mini (a
  looped mixture of experts without rotary positions)
- ✅ **Image input** (Gemma 4 E2B / E4B, Qwen3.5 / Qwen3.6 / Qwen3.8, Qwen3-Omni, Inkling 2 Mini), as base64 `data:`
  URLs; from a GGUF file with its vision tower's file (`mmproj-*.gguf`) too
- ✅ **Audio and images in chat**: Qwen3-Omni-30B-A3B's thinker (text, audio and images in, text and tool calls
  out), as `input_audio` and `image_url` parts
- ✅ **Speech recognition** (Qwen3-ASR) behind `/v1/audio/transcriptions` (json, text or streamed), OpenAI's
  realtime API (`/v1/realtime`: audio streamed over a WebSocket, turns found by voice activity) and as `input_audio`
  in chat: WAV, MP3 or FLAC, long audio cut at quiet points
- ✅ **Quantized packs**: q8 and q4 from `xpu-serve convert` (round to nearest; AWQ calibration opt-in)
- ✅ **FP8 checkpoints** (Qwen3.8-27B-FP8, Qwen3.6-35B-A3B-FP8, Ornith-1.5-35B-A3B-FP8: a scale per 128 x 128 block,
  or compressed-tensors' scale per channel), served as they are or converted to a pack
- ✅ **GGUF files**, served by their path (`serve model.gguf`, imported once into a cache) or with `-hf`, split
  files by their first part: `qwen35`, `qwen35moe`, `gemma4`, `gemma4-assistant`, `inkling2_loop` and `inkling2`
  (Inkling 2 in mainline llama.cpp's layout and Unsloth's, with its DSpark drafter's file in either format and its
  vision tower's), `clef` and the `modern-bert` files of Laya and Julia-1, in all 19 of GGUF's block formats and
  Unsloth's NVFP4, MXFP4 and Q2_0 (expert banks included), kept as the file stores them
- ✅ **Speculative decoding**: Qwen's MTP heads, Gemma 4's assistants, DFlash 2 and DSpine block drafters
  (Qwen3.8-27B), a DFlash block drafter (Ornith-1.5-35B-A3B), Inkling 2's DSpark drafter, and prompt lookup
  (`--spec ngram`: drafts copied from the text itself, for any language model); drafts per round chosen by measured
  acceptance and cost (`--spec-adaptive`)
- ✅ **LoRA adapters** in llama.cpp's GGUF form (`--lora`): each change added where the model reads the weight, which
  stays as stored (quantized ones included)
- ✅ **int8 KV cache** (`--kv-dtype i8`)
- ✅ **TurboQuant KV cache** (`--kv-dtype turbo4 | turbo3 | turbo2`, or int8 keys with TurboQuant values): a cache
  3.9 to 7.5 times smaller than f16
- ✅ **Decision models** behind the System One API (`POST /v1/systemone`): Clef-Flash, Clef, pplx-decider-flash,
  OpenJev, d1, Laya, Julia-1, or any chat model read at the option letters
- ✅ **Embedding models** behind `/v1/embeddings`: all-MiniLM-L6-v2, bge-small-en-v1.5, nomic-embed-text-v1.5
- ✅ **Robot policy** (π0.5) behind the policy API
- ✅ **Several requests at once** (`--sequences N`): each its own cached prompt, their decode steps run as one
  ([how it scales](model_recipes.md#several-requests-at-once))
- ✅ **Prefix reuse**: a request keeps the KV cache of the longest prefix it shares with the one before on its
  sequence
- ✅ **Intel Arc GPUs**, discrete (Arc Pro B60) and integrated (Arc B390), through SYCL

Not yet:

- A system prompt shared between users of a model with recurrent state (a prefix is copied between requests up to
  where a conversation's messages ended: decision 0029); with `--spec`, the drafts of several requests in one step
  (their verifies already are).
- One model across two GPUs.
- Timestamps and translation of speech; speech output; video input; images for Gemma 4 26B-A4B and SenseNova Loop,
  and for a Qwen3.5-family model beside a block drafter (`--spec dflash|dspine`); images from remote URLs.
- Gemma 4 26B-A4B from a GGUF file (a `gemma4` file with experts).
- The prompt's log-probabilities (`echo`, `prompt_logprobs`), `best_of` and beam search, a rerank API.

## Supported models

Each model's recipe has its commands and numbers.

| Model | Kind | Checkpoint | Runs from | Recipe |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | language, images | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` | original, q8 / q4 pack, GGUF (text) | [E2B](model_recipes.md#gemma-4-e2b), [E4B](model_recipes.md#gemma-4-e4b), [GGUF](model_recipes.md#gemma-4-e2b-gguf) |
| Gemma 4 26B-A4B | language, MoE | `google/gemma-4-26B-A4B-it` | q4 pack | [recipe](model_recipes.md#gemma-4-26b-a4b) |
| Qwen3.5-4B | language | `Qwen/Qwen3.5-4B` | original, q8 / q4 pack, GGUF | [recipe](model_recipes.md#qwen35-4b), [GGUF](model_recipes.md#qwen35-4b-gguf) |
| Qwen3.5-9B | language | `Qwen/Qwen3.5-9B` | original, q8 / q4 pack | [recipe](model_recipes.md#qwen35-9b) |
| SenseNova-Loop-9B | language | `sensenova/SenseNova-Loop-9B` (gated), `-Preview` | original, q8 / q4 pack | [recipe](model_recipes.md#sensenova-loop-9b) |
| Qwen3.6-35B-A3B | language, MoE | `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-35B-A3B-FP8` | q4 pack | [recipe](model_recipes.md#qwen36-35b-a3b) |
| Qwen3.8-27B | language | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-FP8` | q4 pack, GGUF | [recipe](model_recipes.md#qwen38-27b), [GGUF](model_recipes.md#qwen38-27b-gguf) |
| Ornith-1.5-35B-A3B | language, MoE | `ornith-ai/Ornith-1.5-35B-A3B`, `-FP8`, `-GGUF` | q4 pack, GGUF Q4_K_M | [recipe](model_recipes.md#ornith-15-35b-a3b), [GGUF](model_recipes.md#ornith-15-35b-a3b-gguf) |
| LFM3-1.7B | language | `LiquidAI/LFM3-1.7B` | original, q8 / q4 pack | [recipe](model_recipes.md#lfm3-17b) |
| Inkling-2-Mini | language, MoE | `thinkingmachines/Inkling-2-Mini`, `unsloth/Inkling-2-Mini-GGUF` | q8 / q4 pack, GGUF | [recipe](model_recipes.md#inkling-2-mini), [GGUF](model_recipes.md#inkling-2-mini-gguf) |
| Inkling-2-Mini Uncensored | language, MoE | `AtomicChat/Inkling-2-Mini-Uncensored-GGUF` | GGUF Q4_K, IQ3_S, IQ2_M | [GGUF](model_recipes.md#inkling-2-mini-uncensored-gguf) |
| Clef-Flash | decision | `Cloudflare/clef-flash` | original, q8 / q4 pack, GGUF | [recipe](model_recipes.md#clef-flash) |
| Clef | decision | `ggml-org/Clef-GGUF` (of `Cloudflare/clef`) | GGUF Q4_K_M | [recipe](model_recipes.md#clef) |
| pplx-decider-flash-v1.2 | decision | `perplexity-ai/pplx-decider-flash-v1.2` | original, q8 / q4 pack | [recipe](model_recipes.md#pplx-decider-flash-v12) |
| OpenJev | decision (a language model read at the option letters) | `openjev/openjev` | q4 pack | [recipe](model_recipes.md#openjev) |
| d1-2B | decision | `LiquidAI/d1-2B` | original, q8 / q4 pack | [recipe](model_recipes.md#d1-2b) |
| Laya | decision | `convaiinnovations/laya` | original, GGUF | [recipe](model_recipes.md#laya) |
| Julia-1 | decision | `SupersonicLabs/Julia-1` | original, GGUF | [recipe](model_recipes.md#julia-1) |
| Qwen3-Omni-30B-A3B-Instruct | language, audio and images in, MoE | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | q4 pack | [recipe](model_recipes.md#qwen3-omni-30b-a3b-instruct) |
| Qwen3-ASR-1.7B / 0.6B | speech recognition | `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | original, q8 / q4 pack | [1.7B](model_recipes.md#qwen3-asr-17b), [0.6B](model_recipes.md#qwen3-asr-06b) |
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

Needs CMake 3.22+, Ninja, OpenSSL 3 (`libssl-dev`, for downloads over HTTPS) and the oneAPI toolkit 2026.2 (DPC++
compiler `icpx` and oneMKL), and an Intel GPU driver with Level Zero.

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

Name a model on the Hugging Face Hub with `-hf`, or give the directory or GGUF file of one you have:

```bash
xpu-serve info                                                # backends and devices (the first is the default)
xpu-serve run   -hf google/gemma-4-E2B-it --device GPU        # downloads it, then chat in the terminal
xpu-serve run   -hf google/gemma-4-E2B-it --device GPU --image photo.jpg --prompt "What is in this picture?"
xpu-serve serve -hf Qwen/Qwen3.5-4B --device GPU              # http://127.0.0.1:8080/v1
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --spec mtp --spec-k 4   # one GGUF file of a repository
xpu-serve bench ~/models/gemma-4-E2B-it --device GPU --prompt-tokens 1024 --gen-tokens 128   # a local directory
xpu-serve serve ~/models/Qwen3.5-4B-Q4_K_M.gguf --device GPU   # a GGUF file you have
```

That is the reference setting: the checkpoint's own weights at f16, an f16 KV cache. The steps below make a model
smaller and faster; [model_recipes.md](model_recipes.md) has the combination to use for each model.
`xpu-serve <command> --help` lists every option.

### Downloading models (`-hf`)

`-hf org/name` (or `--hf`) on `serve`, `run`, `bench` and `convert` takes the place of a model directory, and `-hfd`
(`--hf-drafter`) the place of `--drafter <dir>`:

- **Into the Hugging Face cache**, in its own layout (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE` / `HF_HOME`), so a
  model downloaded by huggingface_hub, `hf download` or another engine is found there and not downloaded again, and
  theirs find the ones xpu-serve downloaded. The second run downloads nothing.
- **Only what xpu-serve reads**: the safetensors and the files beside them, not the PyTorch, ONNX, OpenVINO or
  TensorFlow copies of the weights, pictures, or subfolders that are other variants of the model.
- **A GGUF file**: `-hf org/name:QUANT` names one file of a GGUF repository by its quantization (`Q4_K_M` by default;
  `Q4_0`, `UD-Q4_K_XL`, `Q8_0`, ...). It is imported once into a pack in xpu-serve's cache (`~/.cache/xpu-serve`, or
  `XS_CACHE_DIR`), so it takes its size twice on disk. A Qwen3.5-family file published without its MTP head gets the
  head from the repository's `mtp-*.gguf`, so `--spec mtp` works. A file split into parts (`*-00001-of-00003.gguf`)
  comes down whole, every part. `-hfd org/name` picks a GGUF repository's drafter file (`mtp-*.gguf`, `Q8_0` by
  default).
- **Fast and resumable**: files come in ranges over eight connections, an interrupted download continues where it
  stopped, and every file is checked against its SHA-256 before it is used.
- **Gated and private repositories** need `HF_TOKEN` (or the token file `hf auth login` writes) from an account that
  accepted the model's terms. π0.5 reads PaliGemma's tokenizer, which `-hf lerobot/pi05_base` fetches from the gated
  `google/paligemma-3b-pt-224`.
- **The rest as huggingface_hub reads it**: `@revision` after the name (a branch, tag or commit), `HF_ENDPOINT` for a
  mirror, `HF_HUB_OFFLINE=1` for the cache alone (also used, with a warning, when the Hub cannot be reached), the
  proxies of `https_proxy` and `no_proxy`.
- `serve` names the model `org/name` (or `org/name:QUANT`) in `/v1/models`; `--model-id` changes that.

The design: [docs/decisions/0021](docs/decisions/0021-hugging-face-downloads.md).

### Quantized packs (`convert`)

`convert` writes a pack: the checkpoint with its matmul weights quantized, everything else copied. Every command
takes a pack wherever it takes a model directory; the model code is the same.

```bash
xpu-serve convert -hf Qwen/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q8 --quant q8
xpu-serve convert -hf Qwen/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
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
- The source can be the model's FP8 release instead of the original (`convert -hf Qwen/Qwen3.8-27B-FP8 --out ...`):
  half the download, the same pack layout and speed, and logits as close to Hugging Face's
  ([docs/decisions/0014](docs/decisions/0014-fp8-checkpoints.md)). An FP8 checkpoint also serves as it is, as f16 on
  the device like the original, so it fits where the original fits.
- A decision model converts like a language model; embedding models, Laya and Julia-1 have no pack.

### From a GGUF file

A GGUF file (llama.cpp's format) is imported into a pack, which is then served like any model directory. Each
quantized tensor stays exactly as the file stores it (Q4_K, Q6_K, IQ3_S, ...: nothing is decoded or quantized again),
so the model has the file's weights and takes the file's size in GPU memory:

```bash
xpu-serve serve ~/models/Qwen3.5-4B-Q4_K_M.gguf --device GPU               # a file you have, imported once
xpu-serve serve -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --device GPU            # a file of a repository, likewise
xpu-serve convert ~/models/Qwen3.5-4B-Q4_K_M.gguf ~/models/Qwen3.5-4B-Q4_K_M   # the pack in a directory you choose
```

- `serve`, `run` and `bench` take a GGUF file by its path and import it into xpu-serve's cache (`~/.cache/xpu-serve`,
  or `XS_CACHE_DIR`) the first time; the next start uses that pack, and a file that changed (its size or time) is
  imported again.
- The files beside a GGUF file are read as its repository: a model whose import reads a file of its own gets the one
  `-hf` would take, its MTP head or drafter (`MTP/mtp-*.gguf`, `dspark-*.gguf`) and its vision tower
  (`mmproj-*.gguf`); `--mtp <file>` and `--mmproj <file>` name others.
- The architecture comes from the file (`general.architecture`). Supported: `qwen35` and `qwen35moe`, the Qwen3.5 /
  Qwen3.6 / Qwen3.8 models, dense and mixture of experts (Ornith-1.5-35B-A3B's files are `qwen35moe`), with their
  vision tower's file (`mmproj-*.gguf`, fetched by `-hf` and found beside a file) for images; `gemma4`, Gemma 4 E2B
  and E4B (the text model: a GGUF file holds no vision or audio tower, so images need the original checkpoint), with
  its assistant (`gemma4-assistant`, the `MTP/mtp-*.gguf` files) as the drafter; `inkling2_loop`, Inkling 2 Mini in
  mainline llama.cpp's layout (Unsloth's files, and AtomicChat's uncensored ones), with its drafter's file
  (`dspark-*.gguf`, in mainline's format or Unsloth's earlier one) and its vision tower's (`mmproj-*.gguf`), both
  fetched by `-hf`, and AtomicChat's LoRA adapter file with `--lora`; `inkling2`, the same model in Unsloth's earlier
  layout (the `Old_Support/` copies of its repository); `clef`, the Clef decision models; and `modern-bert` with a
  Laya read-out, the files of Laya and Julia-1. Others are refused by name.
- A file split into parts (llama.cpp's gguf-split: `*-00001-of-00003.gguf` and on) is named by its first part, the
  others found beside it; it gives the pack the whole file gives.
- A Qwen3.5-family file that comes without its MTP head (the smallest files of Qwen3.8-27B) takes the head from its
  `MTP/mtp-*.gguf` file (found beside it, or named with `--mtp`): the pack is the one a file with the head gives, and
  `--spec mtp` drafts with it.
- config.json, the tokenizer and the chat template are written from the file's metadata (on Qwen3.5-4B and Gemma 4
  E2B the tokenizer equals the original's token by token), and the template is the file's, which a publisher may have
  changed from the original checkpoint's. Thinking, tool calls and streaming work as with the original checkpoint.
  The quantization options do not apply.
- Both backends read all 19 of GGUF's block formats (Q4_0 to Q8_0, the K-quants, IQ1 to IQ4), and the three more of
  Unsloth's llama.cpp builds (NVFP4, MXFP4, Q2_0), each decoded to the values llama.cpp's own code gives (checked
  bit for bit on the host). A file is usually a mix of them.

Commands per model: [model_recipes.md](model_recipes.md#importing-a-gguf-file). Every file's numbers:
[docs/gguf_models.md](docs/gguf_models.md).

### Speculative decoding

`--spec` (on `run`, `serve` and `bench`) drafts with the model's drafter and verifies the drafts in one step; greedy
output stays the model's own and sampling keeps its distribution (rejection sampling).

| `--spec` | Drafter | Where it comes from |
|---|---|---|
| `mtp` | the MTP head of a Qwen3.5-family checkpoint | inside the model (and its pack) |
| `mtp` | a Gemma 4 assistant | `--drafter <google/gemma-4-*-it-assistant>` |
| `dflash` | a DFlash or DFlash 2 block drafter | `--drafter <z-lab/Qwen3.8-27B-DFlash2>`, `--drafter <ornith-ai/Ornith-1.5-35B-A3B-DFlash>` |
| `dspine` | a DSpine block drafter | `--drafter <tencent/Qwen3.8-27B-DSpine-b16>` |
| `dspark` | Inkling 2's DSpark block drafter | inside the checkpoint and its packs; `dspark-*.gguf` of its GGUF repository |
| `ngram` | prompt lookup: the tokens that followed the first earlier occurrence of the text's last tokens | the text itself, any language model (`--ngram-min`, `--ngram-max`: 5 by default) |

```bash
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec mtp --spec-k 4
xpu-serve run   -hf google/gemma-4-E2B-it --device GPU --spec mtp --spec-k 1 -hfd google/gemma-4-E2B-it-assistant
```

A drafter is trained for one model, so a technique works only where a drafter for that model exists (elsewhere the
option is refused by name); prompt lookup needs none, and pays where the output repeats its input (code edits,
extraction, quoting). `--spec-k` is the number of drafts per round, and with `--spec-adaptive` the most a round asks
for: the engine then picks each round's count from the request's acceptance so far and what verifies of each width
cost on the device. Which model takes which technique, the value to pass and the speed it gives:
[model_recipes.md](model_recipes.md#speculative-decoding). The design:
[docs/design-spec-decode.md](docs/design-spec-decode.md), [docs/decisions/0034](docs/decisions/0034-prompt-lookup.md).

### KV cache formats

`--kv-dtype` (on `serve`, `run` and `bench`) stores the KV cache in fewer bits, for longer contexts in the same
memory: `i8` (int8 with one scale per token and KV head), or TurboQuant's `turbo4`, `turbo3` and `turbo2` (4, 3 or 2
bits a value); `<keys>:<values>` sets the two apart, e.g. `i8:turbo4`. All of them are lossy and opt-in; the default,
`auto`, is f16. What each costs and saves: [model_recipes.md](model_recipes.md#kv-cache-formats); the designs:
[docs/decisions/0010](docs/decisions/0010-int8-kv-cache.md) (int8) and
[0020](docs/decisions/0020-turboquant-kv-cache.md) (TurboQuant).

`--ctx` is reserved when the model loads: the KV cache of the whole context and the memory of the largest step (a
prefill chunk, a verify). A context that does not fit the GPU is refused there, with what each part takes and the
most tokens that fit, rather than running out of memory halfway into a long prompt.

## APIs

`xpu-serve serve <dir>` serves whatever the directory holds: a language model behind the OpenAI API, a speech
recognition model behind `/v1/audio/transcriptions` and `/v1/realtime`, an embedding model behind `/v1/embeddings`, a decision model behind the System One API, a robot policy behind the policy API.
`--host` and `--port` set the address (127.0.0.1:8080 by default), `--api-key` asks for `Authorization: Bearer <key>`,
`--model-id` names the model in `/v1/models`. `GET /health` answers once the server is up. A request that names
something unsupported is refused with the field's name, never served differently.

### Chat completions

`POST /v1/chat/completions`, `POST /v1/completions` and `GET /v1/models`, so any OpenAI client can point at the
server:

```bash
xpu-serve serve -hf Qwen/Qwen3.5-4B --device GPU
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

- **Streaming** (`"stream": true`, server-sent events; `stream_options.include_usage` adds the usage chunk).
- **Tool calls**: `tools` with `tool_choice` `auto`, `none`, `required` or a named function
  (`{"type": "function", "function": {"name": ...}}`), and `parallel_tool_calls`; calls come back as `tool_calls`,
  also streamed. A forced call is held to the model's own call syntax token by token (Qwen 3.5's XML parameters,
  Qwen 3's JSON, Gemma 4's arguments, Inkling 2's messages), after its reasoning when the prompt leaves it thinking; a
  strict tool's arguments (`"strict": true`) follow its `parameters` schema, also under `auto` once the model opens a
  call; a named function ends with `finish_reason` `stop`, as OpenAI's does. docs/decisions/0035.
- **Thinking**: a model's reasoning comes back as `reasoning_content`, apart from `content`;
  `chat_template_kwargs` reaches the chat template (`{"enable_thinking": false}` turns thinking off where a model
  thinks by default), and so does `reasoning_effort` under its own name (Inkling 2's template reads it).
- **Images**: `image_url` parts with a base64 `data:` URL (PNG, JPEG, BMP or GIF); the server does not fetch remote
  URLs. `--image-max-pixels` bounds the size Qwen3-Omni's processor resizes an image to (its tokens and memory).
- **Audio**: `input_audio` parts (`{"data": <base64>, "format": "wav" | "mp3" | "flac"}`), for a model that takes
  audio (Qwen3-Omni; Qwen3-ASR, whose reply is `language <Name><asr_text><transcript>`).
- **Sampling**: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`,
  `repetition_penalty`, `seed`, `stop`, `max_tokens` / `max_completion_tokens`.
- **Structured outputs**: `response_format` `{"type": "json_object"}` (any JSON object) or `{"type": "json_schema",
  "json_schema": {"name", "schema", "strict"}}`, on `/v1/chat/completions` and `/v1/completions`: the answer is held
  to the schema token by token, and can end only once it is whole. With thinking on, the reasoning is free and the
  answer after it is held. Schemas in OpenAI's strict subset (types, properties in their order, required,
  `additionalProperties: false`, items, enum, const, anyOf, `$ref` / `$defs`, recursive ones too); any other keyword
  is refused with a 400 naming its path (`response_format.json_schema.schema.properties.age.minimum`). Beside `tools`
  with `tool_choice` `auto` the reply is tool calls or the formatted answer; beside a forced call it is refused.
  docs/decisions/0033.
- **Log-probabilities**: `logprobs: true` with `top_logprobs` (0 to 20) on chat completions, `logprobs` (0 to 20) on
  text completions, in OpenAI's shapes, streamed with each chunk's tokens. Each is the log-softmax of the model's
  row before penalties and temperature (vLLM's default), after a `response_format` grammar masked it (a masked
  token's is -9999); speculating or not, they are the same. A token that leaves a character unfinished reads as
  `""` and the one that finishes it carries the whole character, as vLLM writes them; `bytes` are the token's own.
  The prompt's (`echo`, `prompt_logprobs`) are refused by name.
- **Several choices** (`n`, up to 128, on chat and text completions): each sampled on its own (with a `seed`, choice
  i draws with seed + i), its own parser, grammar and log-probabilities; the first is prefilled and the others copy its
  prompt (with `--sequences` enough they decode together). Streamed chunks carry their choice's `index`; the usage
  counts the prompt once. As vLLM, `n` above 1 needs sampling (`temperature` 0 is refused).
- A response carries `usage` and `timings`. Audio or video parts are refused by
  name.

### Anthropic messages

`POST /v1/messages` and `POST /v1/messages/count_tokens` answer Anthropic's Messages API, so its clients can point at
the server too (`--api-key` is then read from `x-api-key`):

```bash
curl -s http://127.0.0.1:8080/v1/messages -H 'Content-Type: application/json' -d '{
  "model": "local", "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

- The reply is a message of content blocks: `thinking` (the model's reasoning), `text` and `tool_use`, with
  `stop_reason` (`end_turn`, `max_tokens`, `stop_sequence` naming the sequence, `tool_use`) and `usage` (cached
  prompt tokens as `cache_read_input_tokens`).
- **Streaming** (`"stream": true`): `message_start`, each block's `content_block_start`, `_delta` (`text_delta`,
  `thinking_delta`, `input_json_delta`) and `_stop`, then `message_delta` and `message_stop`.
- **Requests**: `system` (a string or text blocks); `text`, base64 `image`, `tool_use`, `tool_result` and `thinking`
  blocks; `tools` with `input_schema` (and `strict`), `tool_choice` `auto`, `any`, `tool` (with `name`) or `none` and
  `disable_parallel_tool_use` (forced calls held as on the OpenAI route); `thinking` `enabled` or `disabled` (off the
  way the model's template turns it off); `temperature`, `top_p`, `top_k`, `stop_sequences`, `max_tokens`
  (required). Server tools, documents and image URLs are refused by name, in the API's error shape.

### Transcriptions

A speech recognition model is served behind OpenAI's transcription API:

```bash
xpu-serve serve -hf Qwen/Qwen3-ASR-1.7B --device GPU
curl -s http://127.0.0.1:8080/v1/audio/transcriptions -F file=@speech.wav -F language=en
```

```json
{"text": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
 "usage": {"type": "tokens", "input_tokens": 94, "output_tokens": 22, "total_tokens": 116}}
```

(Qwen3-ASR-1.7B's answer on an Arc Pro B60 for LibriSpeech's first clip, 5.9 s.)

- `file`: WAV (integer PCM or float, any channels and sample rate), MP3 or FLAC. Other containers (Ogg, Opus, M4A,
  WebM) are refused with an ffmpeg command line that converts them.
- `language`: an ISO 639-1 code or a name (`en`, `English`, `yue`); without it the model finds the language.
  `prompt` becomes the model's system turn (context such as names and terms); `temperature` 0 to 1.
- `response_format` `json` or `text`; `stream=true` sends `transcript.text.delta` events as the text is decoded and a
  `transcript.text.done` event with the whole text and its usage.
- Audio longer than the model's piece (20 minutes for Qwen3-ASR) is cut at its quietest moments and the pieces'
  texts joined.
- Refused by name: timestamps (`verbose_json`, `srt`, `vtt`, `timestamp_granularities[]`), `diarized_json`, and
  `/v1/audio/translations` (the model transcribes only).

Decision [0023](docs/decisions/0023-speech-recognition.md) has the details.

### Realtime transcription

`GET /v1/realtime` is OpenAI's Realtime API with a transcription session: a WebSocket that takes audio as it is
recorded and sends back each turn's text as it is decoded. The events are OpenAI's (GA), so its clients work:

```text
client: {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
           "transcription": {"language": "en"},
           "turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}}}}
client: {"type": "input_audio_buffer.append", "audio": "<base64: 16-bit PCM, mono, 24 kHz>"}   (every ~100 ms)
server: input_audio_buffer.speech_started, ... .speech_stopped, input_audio_buffer.committed,
        conversation.item.added, conversation.item.input_audio_transcription.delta ...,
        conversation.item.input_audio_transcription.completed {"transcript", "usage"}, conversation.item.done
```

- Turns: with `server_vad` (the default) the server ends a turn after `silence_duration_ms` of silence and
  transcribes it; with `turn_detection: null` the client ends one with `input_audio_buffer.commit`.
  `input_audio_buffer.clear` drops what is buffered. A turn is transcribed whole, as `/v1/audio/transcriptions` would
  transcribe it; turns are transcribed one after another while audio keeps coming.
- The server's VAD measures frame energy against a noise floor it tracks (`threshold` 0 to 1 sets the absolute level
  speech must reach, `prefix_padding_ms` the audio kept before speech); it is not a model.
- Refused by name, as `error` events: `semantic_vad`, noise reduction, logprobs, formats other than `audio/pcm` at
  24 kHz (G.711 included), and any event a transcription session does not take (responses, conversation items).
- A browser cannot set headers on a WebSocket: with `--api-key`, the key goes as the subprotocol
  `openai-insecure-api-key.<key>`, as OpenAI's own browser clients send it.

On an Arc Pro B60 (Qwen3-ASR-1.7B, q8) a 3 to 7 s turn's whole transcript comes 130 to 250 ms after the VAD ends it
([recipe](model_recipes.md#qwen3-asr-17b)). Decision [0024](docs/decisions/0024-realtime-transcription.md) has the
details.

### Embeddings

A sentence-transformers checkpoint of a BERT or NomicBERT encoder is served behind OpenAI's embeddings API, the one
vLLM and llama.cpp serve too:

```bash
xpu-serve serve -hf BAAI/bge-small-en-v1.5 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["What is a GPU?", "A graphics processing unit runs many threads at once."]}'
```

```json
{"object": "list", "model": "BAAI/bge-small-en-v1.5",
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
xpu-serve serve -hf Cloudflare/clef-flash --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "Cloudflare/clef-flash",
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
{"model": "Cloudflare/clef-flash",
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

A robot policy runs from its LeRobot checkpoint and PaliGemma's tokenizer, which `-hf` fetches beside it (from the
gated `google/paligemma-3b-pt-224`: set `HF_TOKEN`; a local directory needs `tokenizer.json` in it); `bench` times
calls on synthetic camera images after a warm-up, for the cameras present and the flow steps asked for, and `serve`
puts it behind the policy API:

```bash
xpu-serve bench -hf lerobot/pi05_base --device GPU.0 --precision f16 --cameras 3 --steps 10 --runs 10
xpu-serve serve -hf lerobot/pi05_base --device GPU.0 --cameras 2    # warm up for the first 2 cameras
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
| Whole model (Gemma 4, Qwen3.5, LFM3, SenseNova Loop) | Hugging Face fp32 logits and greedy output (SenseNova Loop: its own modeling code) | `tests/unit/model/model_parity.cpp` |
| Image input | Hugging Face fp32 pixels, vision features, logits and greedy output | `tests/unit/model/vision_parity.cpp` |
| Audio decoding, resampling and log-mel features | libsndfile, torchaudio and transformers' feature extractor | `tests/unit/audio/` |
| Speech recognition (Qwen3-ASR) | transformers fp32 log-mel, encoder rows, prompt, logits and greedy text | `tests/unit/model/audio_parity.cpp` |
| Realtime transcription | OpenAI's GA event shapes over a real WebSocket; turns at the samples the VAD should find | `tests/unit/server/test_realtime_*.cpp`, `tests/unit/audio/test_vad.cpp` |
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
