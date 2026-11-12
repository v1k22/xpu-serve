<p align="center">
  <img src="docs/images/logo.png" alt="xpu-serve: one repo, one binary, Intel GPUs" width="380">
</p>

# xpu-serve
Experimental engine to run LLMs on Intel platforms.

One repo, one binary: the model code, the engine, the tokenizer, chat templates and an OpenAI-compatible server all
live here, and the math runs on Intel GPUs through our SYCL backend (our own kernels, oneMKL for large matmuls). Read
[DESIGN.md](DESIGN.md) for the why and [TODO.md](TODO.md) for the plan.

**Status: v0.4.** Language models (Gemma 4, Qwen3.5 / Qwen3.6 / Qwen3.8, LFM3), a robot policy (π0.5) and decision
models (Clef-Flash, pplx-decider-flash, OpenJev, behind the System One API), from the
original checkpoint, its FP8 release where there is one, or a q8 / q4 pack written by `xpu-serve convert`, one request
at a time, on Intel GPUs through the SYCL backend; speculative decoding with each model's drafter (Qwen's MTP head,
Gemma 4's assistant checkpoints, DFlash 2 and DSpine block drafters for Qwen3.8-27B). [CLAUDE.md](CLAUDE.md) maps the
code.

## Supported models

| Model | Kind | Weights | Speed | Measured on |
|---|---|---|---|---|
| Gemma 4 E2B / E4B | language, images | f16, q8, q4 | E2B: 61.5 / 92 tok/s decode (f16 / q8); q4: 140, 174 with its assistant. E4B at q4: 84, 126 with its assistant | Arc Pro B60 |
| Gemma 4 26B-A4B | language, MoE | q4 | 67 tok/s decode, 91 with its assistant | Arc Pro B60 |
| Qwen3.5 4B | language | f16, q8, q4 | 43 / 77 / 101 tok/s decode (f16 / q8 / q4); q4: 149 with MTP; 5,000 tok/s prefill | Arc Pro B60 |
| Qwen3.5 9B | language | f16, q8, q4 | 21 / 38 / 62 tok/s decode (f16 / q8 / q4); q4: 108 with MTP; 3,200 tok/s prefill | Arc Pro B60 |
| Qwen3.6-35B-A3B | language, MoE | q4 | 62 tok/s decode, 94 with MTP; 1,550 tok/s prefill | Arc Pro B60 |
| Qwen3.8-27B | language | q4 | 22 tok/s decode, 50 with MTP, 61 with a DFlash 2 drafter (53 and 62 at q4 g128); 830 tok/s prefill; on Arc B390 6.9 tok/s decode, 18 with MTP, 20 with a DFlash 2 drafter (20 and 23 at q4 g128), 460 prefill | Arc Pro B60, Arc B390 |
| LFM3-1.7B | language | f16, q8, q4 | 99 / 159 / 206 tok/s decode (f16 / q8 / q4); 12,000 tok/s prefill | Arc Pro B60 |
| π0.5 | robot policy | f16, q8 | 215 / 198 ms per call (f16 / q8); 115 / 113 ms on the Arc Pro B60 | Arc B390, 3 cameras |
| Clef-Flash | decision | f16, q8, q4 | 89 ms per decision (p50; 109 ms p95) on prompts of 150 to 262 tokens; 323 ms at 1,024 tokens | Arc Pro B60 |
| pplx-decider-flash-v1.2 | decision | f16, q8, q4 | 39 ms per decision (p50; 57 ms p95) on prompts of 92 to 230 tokens; 214 ms at 1,024 tokens | Arc Pro B60 |
| OpenJev | decision (a language model read at the option letters) | q4 | 266 ms per decision (p50; 478 ms p95) on prompts of 67 to 198 tokens, 143 ms for another question about the state just read; 1.46 s for three questions on a 1,024-token prompt | Arc Pro B60 |

| Model | Checkpoint (`model_type`) | Checked against |
|---|---|---|
| Gemma 4 E2B / E4B | `google/gemma-4-E2B-it` (`gemma4`) | Hugging Face fp32 logits, greedy text, image features; its assistant's drafts |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` (`gemma4`) | tiny-checkpoint parity of its layout (experts beside the MLP, K = V global layers) |
| Qwen3.5 4B | `Qwen/Qwen3.5-4B` (`qwen3_5`) | Hugging Face fp32 logits, greedy text |
| Qwen3.5 9B | `Qwen/Qwen3.5-9B` (`qwen3_5`) | Hugging Face fp32 logits, greedy text, its MTP head's drafts; MMLU-Pro (25 questions): 72 % |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-35B-A3B-FP8` (`qwen3_5_moe`) | tiny-checkpoint parity; its q4 pack against Hugging Face fp32 logits of the checkpoint (cosine 0.92 to 0.98); MMLU-Pro (25 questions): 80 to 84 % |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-FP8` (`qwen3_5`) | tiny-checkpoint parity (its head ratios and RoPE, its MTP head; DFlash, DFlash 2 and DSpine drafters against their PyTorch code); MMLU-Pro (25 questions): 76 % |
| LFM3-1.7B | `LiquidAI/LFM3-1.7B` (`lfm3`) | Hugging Face fp32 logits, greedy text (the checkpoint's own modeling code); MMLU-Pro (25 questions): 60 % |
| π0.5 | `lerobot/pi05_base` (LeRobot `pi05`) | LeRobot fp32 actions: max error 5e-6 (cpu-ref), 1e-3 (GPU f16) |
| Clef-Flash | `Cloudflare/clef-flash` (`qwen3_5` with a joint schema head) | its own Python code in fp32: the same prompt tokens, probabilities within 0.0007 (GPU f16); 25 of 25 test decisions right and equal to that code's |
| pplx-decider-flash-v1.2 | `perplexity-ai/pplx-decider-flash-v1.2` (`qwen3_5` with a read-out per question) | its own Python package in fp32: the same prompt tokens, probabilities within 0.0009 (GPU f16); 25 of 25 test decisions right and equal to that code's |
| OpenJev | `openjev/openjev` (`qwen3_5`, Qwen3.8-27B; weights CC BY-NC 4.0) | its own helper code over transformers in fp32: tiny-checkpoint parity (the same prompts, probabilities within 0.0024 on the GPU at f16); the q4 pack against the checkpoint in fp32: 25 of 25 test decisions right and equal, probabilities within 0.0095 |

Qwen3.5's q4 can be calibrated with AWQ; Qwen3.6-35B-A3B and Qwen3.8-27B at q4 fit one 24 GB card, and their FP8
releases (half the download) load and convert like the originals; LFM3 thinks by
default (`"chat_template_kwargs": {"enable_thinking": false}` in a request turns it off), was run to 64k tokens
of context, and is best served from its q8 pack (at q4 its reasoning runs long more often); π0.5 maps camera images, the
robot state and an instruction to 50 x 32 actions in 10 flow steps, and its q8 pack quantizes the action expert.

A model is added in `src/models/<arch>/` (language models: [CLAUDE.md](CLAUDE.md); policies:
[docs/design-vla.md](docs/design-vla.md); decision models:
[docs/decisions/0015](docs/decisions/0015-decision-models.md)); these tables list what has been enabled and measured.

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

Which model takes which technique. A drafter is trained for one model, so a technique works only where a drafter
for that model exists (`–`: none does, and the option is refused by name). Each cell is the decode speed in tok/s
without → with that technique, and the `--spec-k` (drafts per round) that gave it, which is the one to use:

| Model | Measured on | `mtp` | `dflash` | `dspine` |
|---|---|---|---|---|
| Gemma 4 E2B | q4, Arc Pro B60 | 140 → 174, k = 1 | – | – |
| Gemma 4 E4B | q4, Arc Pro B60 | 84 → 126, k = 2 | – | – |
| Gemma 4 26B-A4B | q4, Arc Pro B60 | 67 → 91, k = 2 | – | – |
| Qwen3.5 4B | q4, Arc Pro B60 | 104 → 149, k = 2 | – | – |
| Qwen3.5 9B | q4, Arc Pro B60 | 62 → 108, k = 4 | – | – |
| Qwen3.6-35B-A3B | q4, Arc Pro B60 | 62 → 94, k = 3 | – | – |
| Qwen3.8-27B | q4, Arc Pro B60 | 22 → 50, k = 4 | 22 → 61, k = 7 | 22 → 51, k = 7 |
| Qwen3.8-27B | q4, Arc B390 | 6.6 → 18, k = 5 | 6.6 → 20, k = 7 | 6.6 → 17.5, k = 7 |
| LFM3-1.7B | | – | – | – |

Greedy decode over six chat prompts of up to 256 new tokens (`bench --prompts`); the drafters as q4 packs, the Gemma 4
assistants as their checkpoints. DFlash 2 and DSpine checkpoints are released for Qwen3.8-27B only. The original
DFlash format loads through the same code (`--spec dflash`), but no DFlash checkpoint exists for a model in this
table.

`--spec-k` is 3 by default and 7 for a block drafter, which proposes its whole block in one pass
([docs/design-spec-decode.md](docs/design-spec-decode.md)); the table has the value to pass per model. A drafter
checkpoint is served as it is, or from a pack of its own, which is smaller and drafts faster:
`convert <drafter> --target <model> --out <dir>`. An MTP head picks each draft among 64 tokens that a one-bit sketch
of the output layer shortlists (`--draft-shortlist`, 0 to read the whole layer). `bench --prompts file` measures over
real prompts, since acceptance depends on the text, and prints where a round's time goes:

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

## Decision models (the System One API)

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

(Clef-Flash's answer, rounded here to three digits; it took 117 ms on an Arc Pro B60.) The `state` and every
`instructions` and
description may be a string or any JSON. `confidence` follows the API's formulas for every model; probabilities are
not rounded. A body that does not follow the schema gets the API's 422 with the path of the field, one longer than
`--ctx` its 400 (`max_tokens_exceeded`): nothing is cut to fit. Each response has its timings in a `Server-Timing`
header. Images (`assets`, box questions) are refused by name: the Qwen3.5 vision tower is not loaded yet.

| Model | Checkpoint | How it decides | On the Arc Pro B60 |
|---|---|---|---|
| Clef-Flash | `Cloudflare/clef-flash` (9B) | every question of a request in one pass; a transformer head scores the options | f16, 15.0 GiB; 89 ms per decision at about 180 tokens |
| pplx-decider-flash-v1.2 | `perplexity-ai/pplx-decider-flash-v1.2` (4B) | every question of a request in one pass, each read at a marker after its options; calibrated with the checkpoint's temperature | f16, 7.8 GiB; 39 ms per decision at about 110 tokens |
| OpenJev | `openjev/openjev` (27B, a tuned Qwen3.8-27B; weights CC BY-NC 4.0) | a prompt per question; the output layer's scores of the option letters at the first output position, calibrated as its helper does | q4 pack, 16.2 GiB; 266 ms per decision at about 90 tokens, 143 ms for another question about the same state |

The per-decision times are medians over 25 test decisions, one question a request, all of which the three models
answer right. Questions sent together share one pass in Clef and pplx-decider and can move each other's answers;
Perplexity's card recommends one question per request for its model unless the questions are facets of one decision.
A pplx-decider request longer than `--ctx` is answered in parts (its questions split, each part with the whole
state), as its own code does.

OpenJev is a language model, so its directory serves as one unless `--readout` asks for decisions, and a 24 GB card
takes it as a q4 pack:

```bash
xpu-serve convert ~/models/openjev --out ~/models/openjev-q4 --quant q4
xpu-serve serve ~/models/openjev-q4 --device GPU --ctx 16384 --readout openjev
```

`--readout openjev` reads the letters with the calibration of OpenJev's serving guide (temperature 0.85, the yes / no
slope 1.829074); `--readout letters` reads any chat model the same way without calibration, and
`--readout-temperature`, `--readout-noul-temperature` and `--readout-noul-bias` set the numbers. Each question is a
prompt of its own, so its answer does not depend on the other questions; the state is read once per request, and a
request about the state of the one before reads only its questions. A question with more than 52 options takes
several passes, as in the helper. OpenJev needs `instructions` in every question.

Latency is the time to read the prompt, so it grows with the state (three questions, p50; for OpenJev, whose three
questions are three prompts, also the time when the state was read by the request before):

| Prompt tokens | sample | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,000 |
|---|---|---|---|---|---|---|---|
| Clef-Flash | 123 ms (325 tokens) | 165 ms | 323 ms | 643 ms | 1.34 s | 2.74 s | 5.89 s |
| pplx-decider-flash-v1.2 | 46 ms (170 tokens) | 104 ms | 214 ms | 456 ms | 1.07 s | 2.53 s | 7.31 s |
| OpenJev (q4), a new state | 538 ms (84 tokens) | 1.03 s | 1.46 s | 2.54 s | 4.75 s | 9.47 s | 19.6 s |
| OpenJev (q4), the same state again | 429 ms | 434 ms | 434 ms | 440 ms | 455 ms | 475 ms | 523 ms |

`convert` writes packs of the two flash checkpoints as of any model (`convert ~/models/clef-flash --out ... --quant
q8`). On this card a pack saves memory, not time: reading a prompt from quantized weights dequantizes them at every
step, 50 to 65 ms a step for Clef-Flash, which a language model wins back in decode and a decision model never does. What a pack changes in the answers, on 300 MMLU-Pro questions (one choice of up to ten options each, which the
models get right about two times in three) against the f16 answers:

| | Clef-Flash f16 | q8 | q4 | pplx-decider f16 | q8 | q4 |
|---|---|---|---|---|---|---|
| On the device | 15.0 GiB | 7.7 GiB | 5.1 GiB | 7.8 GiB | 4.0 GiB | 2.7 GiB |
| Latency p50, the 25 test decisions | 89 ms | 145 ms | 138 ms | 39 ms | 66 ms | 62 ms |
| Right, of 300 | 198 | 194 | 193 | 192 | 193 | 184 |
| Same answer as f16, of 300 | | 292 | 262 | | 298 | 265 |
| Changed among answers f16 gives 0.9 or more | | 0 of 41 | 0 of 41 | | 0 of 119 | 3 of 119 |
| Largest change of a probability per question: mean, 95th percentile | | 0.005, 0.02 | 0.06, 0.20 | | 0.007, 0.02 | 0.08, 0.31 |

q8 keeps the answers: the few that change are ones the model itself is unsure of. q4 (groups of 32, round to
nearest) moves one answer in eight on questions this hard and shifts probabilities by 0.06 to 0.08 on average, though
answers the model is confident of stay; on the 25 easy test decisions every pack gives the f16 answers (probabilities
within 0.003 at q8, 0.05 at q4). So f16 where it fits, q8 where memory is short, and q4 only where an unsure answer
is treated as unsure anyway. An AWQ-calibrated q4 of the decoder narrows the shift by a quarter to a third and
changes four fewer answers of the 300: better, not q8
([docs/decisions/0015](docs/decisions/0015-decision-models.md)). OpenJev's q4 pack, the form a 24 GB card takes it
in, against its checkpoint's float32 answers on the same questions: 265 of 300 the same and 218 right against 222, with
1 of the 138 answers held at 0.9 or more changed.

Without a server, `run` answers one request file and `bench` times one (its own sample by default, `--state-tokens N`
to grow the state):

```bash
xpu-serve run   ~/models/clef-flash --device GPU --request request.json
xpu-serve bench ~/models/clef-flash --device GPU --ctx 16384 --state-tokens 4096 --runs 7
python3 tests/e2e/systemone_eval.py --url http://127.0.0.1:8080      # 25 decisions: answers, API shape, latency
```

## Quantize (`convert`)

`convert` writes a pack: the checkpoint with its matmul weights quantized, everything else copied. Every command
takes a pack wherever it takes a model directory; the model code is the same.

```bash
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q8 --quant q8
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
xpu-serve serve ~/models/Qwen3.5-4B-q8 --device GPU
```

The source can be the model's FP8 release instead of the original (`convert ~/models/Qwen3.8-27B-FP8 --out ...`):
half the download, the same pack layout and speed, and logits as close to Hugging Face's (numbers:
[docs/decisions/0014](docs/decisions/0014-fp8-checkpoints.md)). An FP8 checkpoint also serves as it is, as f16 on the
device like the original, so it fits where the original fits.

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

### From a GGUF file

`convert` also imports a GGUF file (llama.cpp's format) into a pack. Each quantized tensor stays exactly as the file
stores it (Q4_K, Q6_K, IQ2_XS, ...: nothing is decoded or quantized again), so the pack holds the file's weights at
the file's size:

```bash
xpu-serve convert ~/models/Qwen3.5-4B-Q4_K_M.gguf ~/models/Qwen3.5-4B-Q4_K_M
xpu-serve run ~/models/Qwen3.5-4B-Q4_K_M --backend cpu-ref --prompt "Hi"
```

The architecture comes from the file (`general.architecture`): `qwen35`, the Qwen3.5 / Qwen3.6 / Qwen3.8 dense text
models, is supported; others are refused by name. config.json, the tokenizer and the chat template are written from
the file's metadata (on Qwen3.5-4B the tokenizer equals the original's token by token), and the template is the
file's, which a publisher may have changed from the original checkpoint's. The quantization options do not apply.
The CPU reference backend reads every block format; the GPU backend does not read them yet.

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
| Decision models (Clef-Flash, pplx-decider-flash, OpenJev) | the checkpoint's own Python code in fp32: prompt token ids and every option's probability | `tests/unit/model/decision_parity.cpp` |
| System One API | the API's schema and error bodies; 25 decisions with accepted answers through a running server | `tests/unit/server/test_systemone_http.cpp`, `tests/e2e/` |

The model parity test needs the model and generated fixtures:

```bash
python -I tests/parity/model/hf_reference.py --model-dir ~/models/gemma-4-E2B-it          # needs transformers
python -I tests/parity/model/hf_vision_reference.py --model-dir ~/models/gemma-4-E2B-it   # image cases
XS_TEST_MODEL_DIR=~/models/gemma-4-E2B-it XS_PARITY_BACKEND=sycl XS_PARITY_DEVICE=GPU.0 build/tests/xs_tests -tc='model*'
```
