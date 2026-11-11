<p align="center">
  <img src="docs/images/logo.png" alt="xpu-serve: one repo, one binary, Intel GPUs" width="380">
</p>

# xpu-serve
Experimental engine to run LLMs on Intel platforms.

One repo, one binary: the model code, the engine, the tokenizer, chat templates and an OpenAI-compatible server all
live here, and the math runs on Intel GPUs through our SYCL backend (our own kernels, oneMKL for large matmuls). Read
[DESIGN.md](DESIGN.md) for the why and [TODO.md](TODO.md) for the plan.

**Status: v0.4.** Language models (Gemma 4, Qwen3.5 / Qwen3.6 / Qwen3.8, LFM3) and a robot policy (π0.5), from the original
checkpoint or a q8 / q4 pack written by `xpu-serve convert`, one request at a time, on Intel GPUs through the SYCL
backend; speculative decoding with each model's drafter (Qwen's MTP head, Gemma 4's assistant checkpoints, DFlash 2
and DSpine block drafters for Qwen3.8-27B). [CLAUDE.md](CLAUDE.md) maps the code.

## Supported models

| Model | Kind | Weights | Speed | Measured on |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | language, images | f16, q8, q4 | 61.5 / 92 tok/s decode (f16 / q8); q4: 140, 174 with its assistant | Arc Pro B60 |
| Gemma 4 26B-A4B | language, MoE | q4 | 67 tok/s decode, 91 with its assistant | Arc Pro B60 |
| Qwen3.5 4B | language | f16, q8, q4 | 42 / 71 tok/s decode (f16 / q8) | Arc Pro B60 |
| Qwen3.6-35B-A3B | language, MoE | q4 | 62 tok/s decode, 94 with MTP; 1,550 tok/s prefill | Arc Pro B60 |
| Qwen3.8-27B | language | q4 | 22 tok/s decode, 50 with MTP, 61 with a DFlash 2 drafter (53 and 62 at q4 g128); 830 tok/s prefill; on Arc B390 6.9 tok/s decode, 18 with MTP, 20 with a DFlash 2 drafter (20 and 23 at q4 g128), 460 prefill | Arc Pro B60, Arc B390 |
| LFM3-1.7B | language | f16, q8, q4 | 99 / 159 / 206 tok/s decode (f16 / q8 / q4); 12,000 tok/s prefill | Arc Pro B60 |
| π0.5 | robot policy | f16, q8 | 215 / 198 ms per call (f16 / q8); 115 / 113 ms on the Arc Pro B60 | Arc B390, 3 cameras |

| Model | Checkpoint (`model_type`) | Checked against |
|---|---|---|
| Gemma 4 E2B / E4B | `google/gemma-4-E2B-it` (`gemma4`) | Hugging Face fp32 logits, greedy text, image features; its assistant's drafts |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` (`gemma4`) | tiny-checkpoint parity of its layout (experts beside the MLP, K = V global layers) |
| Qwen3.5 4B | `Qwen/Qwen3.5-4B` (`qwen3_5`) | Hugging Face fp32 logits, greedy text |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B` (`qwen3_5_moe`) | tiny-checkpoint parity; MMLU-Pro (25 questions): 84 % |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` (`qwen3_5`) | tiny-checkpoint parity (its head ratios and RoPE, its MTP head; DFlash, DFlash 2 and DSpine drafters against their PyTorch code); MMLU-Pro (25 questions): 76 % |
| LFM3-1.7B | `LiquidAI/LFM3-1.7B` (`lfm3`) | Hugging Face fp32 logits, greedy text (the checkpoint's own modeling code); MMLU-Pro (25 questions): 60 % |
| π0.5 | `lerobot/pi05_base` (LeRobot `pi05`) | LeRobot fp32 actions: max error 5e-6 (cpu-ref), 1e-3 (GPU f16) |

Qwen3.5's q4 can be calibrated with AWQ; Qwen3.6-35B-A3B and Qwen3.8-27B at q4 fit one 24 GB card; LFM3 thinks by
default (`"chat_template_kwargs": {"enable_thinking": false}` in a request turns it off), was run to 64k tokens
of context, and is best served from its q8 pack (at q4 its reasoning runs long more often); π0.5 maps camera images, the
robot state and an instruction to 50 x 32 actions in 10 flow steps, and its q8 pack quantizes the action expert.

A model is added in `src/models/<arch>/` (language models: [CLAUDE.md](CLAUDE.md); policies:
[docs/design-vla.md](docs/design-vla.md)); these tables list what has been enabled and measured.

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

Speculative decoding (`--spec` on `run`, `serve` and `bench`) drafts with the model's drafter and verifies the
drafts in one step; greedy output stays the model's own and sampling keeps its distribution (rejection sampling).

| `--spec` | Drafter | Where it comes from |
|---|---|---|
| `mtp` | the MTP head of a Qwen3.5-family checkpoint | inside the model (and its pack) |
| `mtp` | a Gemma 4 assistant | `--drafter <google/gemma-4-*-it-assistant>` |
| `dflash` | a DFlash or DFlash 2 block drafter | `--drafter <z-lab/Qwen3.8-27B-DFlash2>` |
| `dspine` | a DSpine block drafter | `--drafter <tencent/Qwen3.8-27B-DSpine-b16>` |

Which model takes which technique (a drafter is trained for one model, so a technique works only where a drafter for
that model exists; anything else is refused by name):

| Model | `mtp` | `dflash` | `dspine` | Decode with it (tok/s, off → on) |
|---|---|---|---|---|
| Gemma 4 E2B | yes: `google/gemma-4-E2B-it-assistant` | no | no | 140 → 174 (q4, Arc Pro B60) |
| Gemma 4 E4B | same code as E2B; its assistant has not been run | no | no | not measured |
| Gemma 4 26B-A4B | yes: `google/gemma-4-26B-A4B-it-assistant` | no | no | 67 → 91 (q4, Arc Pro B60) |
| Qwen3.5 4B | yes: the MTP head in its checkpoint | no | no | not measured |
| Qwen3.6-35B-A3B | yes: the MTP head in its checkpoint | no | no | 62 → 94 (q4, Arc Pro B60) |
| Qwen3.8-27B | yes: the MTP head in its checkpoint | yes: `z-lab/Qwen3.8-27B-DFlash2` | yes: `tencent/Qwen3.8-27B-DSpine-b16` | Arc Pro B60: 22 → 50 (`mtp`), 61 (`dflash`), 51 (`dspine`); Arc B390: 6.6 → 18, 20, 17.5 |
| LFM3-1.7B | no (no drafter exists) | no | no | |

DFlash 2 and DSpine checkpoints are released for Qwen3.8-27B only. The original DFlash format loads through the same
code (`--spec dflash`), but no DFlash checkpoint exists for a model in this table.

`--spec-k` sets the drafts per round: 3 by default (dense Qwen3.8 does best at 4 or 5 with its MTP head, the MoE
models at 2 or 3), 7 for a block drafter, which proposes its whole block in one pass
([docs/design-spec-decode.md](docs/design-spec-decode.md)). A drafter checkpoint is served as it is, or from a pack
of its own, which is smaller and drafts faster: `convert <drafter> --target <model> --out <dir>`. An MTP head picks
each draft among 64 tokens that a one-bit sketch of the output layer shortlists (`--draft-shortlist`, 0 to read the
whole layer). `bench --prompts file` measures over real prompts, since acceptance depends on the text, and prints
where a round's time goes:

```bash
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec mtp --spec-k 4
xpu-serve run   ~/models/gemma-4-E2B-it --device GPU --spec mtp --spec-k 1 --drafter ~/models/gemma-4-E2B-it-assistant
xpu-serve convert ~/models/Qwen3.8-27B-DFlash2 --target ~/models/Qwen3.8-27B-q4 --out ~/models/Qwen3.8-27B-DFlash2-q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dflash --drafter ~/models/Qwen3.8-27B-DFlash2-q4
```

The server speaks the OpenAI API (`/v1/chat/completions` with streaming, tool calls and images, `/v1/completions`,
`/v1/models`), so any OpenAI client can point at it. Images go in as `image_url` parts with a base64 `data:` URL
(PNG, JPEG, BMP or GIF); the server does not fetch remote URLs.

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

## Quantize (`convert`)

`convert` writes a pack: the checkpoint with its matmul weights quantized, everything else copied. Every command
takes a pack wherever it takes a model directory; the model code is the same.

```bash
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q8 --quant q8
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
xpu-serve serve ~/models/Qwen3.5-4B-q8 --device GPU
```

`q8` is symmetric int8; `q4` is 4-bit with a scale and bias per group of 32 columns (`--group 128` reads 15% fewer
bytes: Qwen3.8-27B decodes 13% faster on both the Arc Pro B60 and the Arc B390, and scores 80% on 25 MMLU-Pro questions
against 76% at 32). Both round to nearest by default; `--method awq` adds an AWQ-style search calibrated on the model
itself (on the GPU, about a minute; `--calib file` brings your own text), which helps Qwen3.5 but not yet Gemma 4 (see
the decision record). Embedding tables are q8 in either case, small matrices and non-matmul tensors stay as they are,
and `--q8 name,...` keeps chosen weights (parts of their names) at q8. The format of every tensor is in the pack's
`xpu_pack.json`.

Which matmul weights are quantized follows one rule for every model, read off the model's own graph: a weight whose
matmuls read it more than they compute (a step-dependent number of rows, as in decode, or at most 64 rows per call)
is quantized, because a smaller weight makes those matmuls faster; weights of compute-bound matmuls (an image encoder,
a policy's prefix over hundreds of tokens) stay as they are. `--scope all` quantizes every matmul weight instead (the
smallest pack). For π0.5 the rule picks the action expert, which runs 50 tokens per flow step:

```bash
xpu-serve convert ~/models/pi05_base --out ~/models/pi05_base-q8 --quant q8
```

On the Arc B390 that pack runs a 3-camera call in 198 ms instead of 215 ms with actions within cosine 0.99999 of
LeRobot's; q4 is faster still (194 ms) but misses the 0.999 accuracy gate (cosine 0.998), so q8 is the setting for
policies. (These are times for calls with pauses between them; a laptop that runs the policy back to back for
longer than its short-term power budget lasts slows down, see [docs/design-vla.md](docs/design-vla.md) §6.1.)

On the Arc Pro B60, q8 decodes 1.5 to 1.7 times faster than f16 with no measurable quality loss; q4 decodes about
twice as fast and keeps MMLU-Pro scores, with some loss of logit fidelity on small models (Qwen3.5-4B logits cosine
0.90 to 0.96 at groups of 32, 0.92 to 0.96 with AWQ). Numbers: [docs/decisions/0009](docs/decisions/0009-quantized-packs.md).

## Long context: int8 KV cache

`--kv-dtype i8` (on `serve`, `run` and `bench`) stores the KV cache as int8 with one scale per token and KV head,
which halves its memory (Qwen3.5-4B: 32 KiB to 16.3 KiB per token). On the Arc Pro B60 it decodes 2 to 9 % faster
at 16k to 32k tokens and prefills 1 to 8 % slower. Logits stay as close to f16 as with q8 weights, and Qwen3.5-4B
keeps its MMLU-Pro score. Gemma 4 E2B changes a few unsure answers, so the option stays opt-in for now. Numbers:
[docs/decisions/0010](docs/decisions/0010-int8-kv-cache.md).

## Correctness

Every layer is tested against a reference:

| What | Reference | Where |
|---|---|---|
| Tokenizer (Gemma BPE, byte-level BPE with NFC, LFM3's) | Hugging Face `tokenizers`, exact ids and text | `tests/unit/tokenizer/` |
| Chat template | `apply_chat_template`, byte-exact | `tests/unit/chat/` |
| Image resize | torchvision's antialiased bicubic, bit-exact | `tests/unit/image/` |
| Every graph op, on cpu-ref and on the GPU | fp64 references, with planted-bug controls | `tests/unit/backend/` |
| Whole model (Gemma 4, Qwen3.5, LFM3) | Hugging Face fp32 logits and greedy output | `tests/unit/model/model_parity.cpp` |
| Image input | Hugging Face fp32 pixels, vision features, logits and greedy output | `tests/unit/model/gemma4_vision_parity.cpp` |
| Robot policy (π0.5) | LeRobot fp32 actions and prompt token ids, 3 cameras and 1 | `tests/unit/model/policy_parity.cpp` |

The model parity test needs the model and generated fixtures:

```bash
python -I tests/parity/model/hf_reference.py --model-dir ~/models/gemma-4-E2B-it          # needs transformers
python -I tests/parity/model/hf_vision_reference.py --model-dir ~/models/gemma-4-E2B-it   # image cases
XS_TEST_MODEL_DIR=~/models/gemma-4-E2B-it XS_PARITY_BACKEND=sycl XS_PARITY_DEVICE=GPU.0 build/tests/xs_tests -tc='model*'
```
