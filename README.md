# xpu-serve
Experimental engine to run LLMs on Intel platforms.

One repo, one binary: the model code, the engine, the tokenizer, chat templates and an OpenAI-compatible server all
live here, and the math runs on Intel GPUs through our SYCL backend (our own kernels, oneMKL for large matmuls). Read
[DESIGN.md](DESIGN.md) for the why and [TODO.md](TODO.md) for the plan.

**Status: v0.4.** Language models (Gemma 4, Qwen3.5 / Qwen3.6) and a robot policy (π0.5), from the original
checkpoint or a q8 / q4 pack written by `xpu-serve convert`, one request at a time, on Intel GPUs through the SYCL
backend. [PROJECT.md](PROJECT.md) maps the code.

## Supported models

| Model | Kind | Checkpoint (`model_type` / LeRobot `type`) | Weights | Checked against | Measured |
|---|---|---|---|---|---|
| Gemma 4 E2B / E4B | language, images | `google/gemma-4-E2B-it` (`gemma4`) | f16, q8, q4 | Hugging Face fp32 logits, greedy text, image features | Arc Pro B60: 61.5 tok/s decode (f16), 92 tok/s (q8) |
| Qwen3.5 4B | language | `Qwen/Qwen3.5-4B` (`qwen3_5`) | f16, q8, q4 (AWQ) | Hugging Face fp32 logits, greedy text | Arc Pro B60: 42 tok/s decode (f16), 71 tok/s (q8) |
| Qwen3.6-35B-A3B | language, mixture of experts | `Qwen/Qwen3.6-35B-A3B` (`qwen3_5_moe`) | q4 (fits one 24 GB card) | tiny-checkpoint parity, MMLU-Pro (25 questions): 84 % | Arc Pro B60: 48 tok/s decode, 1,250 tok/s prefill |
| π0.5 | robot policy (cameras, state, instruction -> 50 x 32 actions) | `lerobot/pi05_base` (`pi05`) | f32 / f16, q8 (action expert) | LeRobot f32 actions: max error 5e-6 (cpu-ref), 2e-3 (GPU f16) | Arc B390, 3 cameras, 10 steps: 245 ms (f16), 231 ms (q8) |

A model is added in `src/models/<arch>/` (language models: [PROJECT.md](PROJECT.md); policies:
[docs/design-vla.md](docs/design-vla.md)); this table lists what has been enabled and measured.

## Build

Needs CMake 3.22+, Ninja and the oneAPI toolkit 2026.2 (DPC++ compiler `icpx` and oneMKL), and an Intel GPU driver
with Level Zero.

```bash
source /opt/intel/oneapi/setvars.sh                      # oneAPI 2026.2
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
build/tests/xs_tests                                     # unit tests (XS_DEVICE_TEST_DEVICE=GPU.0 adds the GPU suites)
```

Add `-DXS_SYCL_TARGETS=intel_gpu_bmg_g21` to compile the SYCL kernels ahead of time for the Arc Pro B60 (otherwise
they compile at first use). Without oneAPI, any C++20 compiler (g++ 11 or newer) builds everything but the GPU
backend: the CPU reference backend, which is the test oracle and far too slow to serve.

## Run

Any Hugging Face directory of a supported model works, e.g. `google/gemma-4-E2B-it` or `Qwen/Qwen3.5-4B`:

```bash
build/src/xpu-serve info                                           # backends and devices (the first is the default)
build/src/xpu-serve run   ~/models/gemma-4-E2B-it --device GPU     # chat in the terminal
build/src/xpu-serve run   ~/models/gemma-4-E2B-it --device GPU --image photo.jpg --prompt "What is in this picture?"
build/src/xpu-serve serve ~/models/gemma-4-E2B-it --device GPU     # http://127.0.0.1:8080/v1
build/src/xpu-serve bench ~/models/gemma-4-E2B-it --device GPU --prompt-tokens 1024 --gen-tokens 128
```

The server speaks the OpenAI API (`/v1/chat/completions` with streaming, tool calls and images, `/v1/completions`,
`/v1/models`), so any OpenAI client can point at it. Images go in as `image_url` parts with a base64 `data:` URL
(PNG, JPEG, BMP or GIF); the server does not fetch remote URLs.

A robot policy runs from its LeRobot directory (with the tokenizer as `tokenizer.json` beside it); `bench` times
calls on synthetic camera images after a warm-up, for the cameras present and the flow steps asked for:

```bash
build/src/xpu-serve bench ~/models/pi05_base --device GPU.0 --precision f16 --cameras 3 --steps 10 --runs 10
```

## Quantize (`convert`)

`convert` writes a pack: the checkpoint with its matmul weights quantized, everything else copied. Every command
takes a pack wherever it takes a model directory; the model code is the same.

```bash
build/src/xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q8 --quant q8
build/src/xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
build/src/xpu-serve serve ~/models/Qwen3.5-4B-q8 --device GPU
```

`q8` is symmetric int8; `q4` is 4-bit with a scale and bias per group of 32 columns. Both round to nearest by
default; `--method awq` adds an AWQ-style search calibrated on the model itself (on the GPU, about a minute; `--calib
file` brings your own text), which helps Qwen3.5 but not yet Gemma 4 (see the decision record).
Embedding tables are q8 in either case, small matrices and non-matmul tensors stay as they are, and `--q8 name,...`
keeps chosen weights (parts of their names) at q8. The format of every tensor is in the pack's `xpu_pack.json`.

Which matmul weights are quantized follows one rule for every model, read off the model's own graph: a weight whose
matmuls read it more than they compute (a step-dependent number of rows, as in decode, or at most 64 rows per call)
is quantized, because a smaller weight makes those matmuls faster; weights of compute-bound matmuls (an image encoder,
a policy's prefix over hundreds of tokens) stay as they are. `--scope all` quantizes every matmul weight instead (the
smallest pack). For π0.5 the rule picks the action expert, which runs 50 tokens per flow step:

```bash
build/src/xpu-serve convert ~/models/pi05_base --out ~/models/pi05_base-q8 --quant q8
```

On the Arc B390 that pack runs a 3-camera call in 231 ms instead of 245 ms with actions within cosine 0.99999 of
LeRobot's; q4 is faster still (227 ms) but misses the 0.999 accuracy gate, so q8 is the setting for policies.

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
| Tokenizer (Gemma BPE, byte-level BPE with NFC) | Hugging Face `tokenizers`, exact ids and text | `tests/unit/tokenizer/` |
| Chat template | `apply_chat_template`, byte-exact | `tests/unit/chat/` |
| Image resize | torchvision's antialiased bicubic, bit-exact | `tests/unit/image/` |
| Every graph op, on cpu-ref and on the GPU | fp64 references, with planted-bug controls | `tests/unit/backend/` |
| Whole model (Gemma 4, Qwen3.5) | Hugging Face fp32 logits and greedy output | `tests/unit/model/model_parity.cpp` |
| Image input | Hugging Face fp32 pixels, vision features, logits and greedy output | `tests/unit/model/gemma4_vision_parity.cpp` |
| Robot policy (π0.5) | LeRobot fp32 actions and prompt token ids, 3 cameras and 1 | `tests/unit/model/policy_parity.cpp` |

The model parity test needs the model and generated fixtures:

```bash
python -I tests/parity/model/hf_reference.py --model-dir ~/models/gemma-4-E2B-it          # needs transformers
python -I tests/parity/model/hf_vision_reference.py --model-dir ~/models/gemma-4-E2B-it   # image cases
XS_TEST_MODEL_DIR=~/models/gemma-4-E2B-it XS_PARITY_BACKEND=sycl XS_PARITY_DEVICE=GPU.0 build/tests/xs_tests -tc='model*'
```
