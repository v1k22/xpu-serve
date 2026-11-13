# GGUF files: what was measured

Every result behind xpu-serve's GGUF support, in one place: two model families, every file their publisher offers,
on one Intel Arc Pro B60 (24 GB). [Decision 0016](decisions/0016-gguf.md) has the design; the README's GGUF section
has the short version.

## How it was measured

- **Import**: `xpu-serve convert <file.gguf> <dir>` (2 to 15 seconds a file), then the pack is used like any model.
- **GPU memory**: the weights the backend holds on the device, as `serve` logs them.
- **Decode and prefill**: `xpu-serve bench <dir> --device GPU --prompt-tokens 1024 --gen-tokens 128`, the last of its
  three runs.
- **Answers**: the model served with `--readout letters` (a language model read at the option letters, through the
  System One API) on two question sets: the 25 test decisions of `tests/e2e/decisions25.json`, and 300 MMLU-Pro
  questions (`tests/e2e/make_mmlu_pro.py`). "Right" counts correct answers. "Same as f16" counts the questions on
  which a file gives the answer the original checkpoint gives at f16, and "shift" is the mean total variation
  between the two answer distributions (0: identical, 1: disjoint). On 300 questions a count of right answers moves
  by about 8 from chance alone, so the agreement with f16 is the sharper measure of what a format costs.
- **Baselines**: the original Hugging Face checkpoint at f16 and xpu-serve's own q8 and q4 packs of it (round to
  nearest, groups of 32).

## Qwen3.5-4B

Files: [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), all 22.

| Variant | File | GPU memory | Decode | Prefill | Test decisions right (25) | MMLU-Pro right (300) | Same answer as f16 (300) | Shift |
|---|---|---|---|---|---|---|---|---|
| UD-IQ2_XXS | 1.52 GB | 1.40 GiB | 91 tok/s | 4,235 tok/s | 16 | 61 | 55 | 0.662 |
| UD-IQ2_M | 1.76 GB | 1.63 GiB | 85 tok/s | 4,216 tok/s | 21 | 87 | 90 | 0.550 |
| UD-Q2_K_XL | 1.94 GB | 1.80 GiB | 95 tok/s | 4,205 tok/s | 24 | 106 | 142 | 0.399 |
| UD-IQ3_XXS | 1.95 GB | 1.80 GiB | 82 tok/s | 4,204 tok/s | 24 | 121 | 140 | 0.373 |
| Q3_K_S | 2.11 GB | 1.95 GiB | 86 tok/s | 4,158 tok/s | 24 | 140 | 180 | 0.294 |
| Q3_K_M | 2.29 GB | 2.12 GiB | 90 tok/s | 4,179 tok/s | 24 | 135 | 198 | 0.219 |
| UD-Q3_K_XL | 2.44 GB | 2.26 GiB | 83 tok/s | 4,151 tok/s | 23 | 129 | 198 | 0.259 |
| IQ4_XS | 2.48 GB | 2.30 GiB | 67 tok/s | 4,143 tok/s | 24 | 135 | 217 | 0.182 |
| IQ4_NL | 2.58 GB | 2.39 GiB | 70 tok/s | 4,166 tok/s | 24 | 132 | 217 | 0.172 |
| Q4_K_S | 2.59 GB | 2.40 GiB | 93 tok/s | 4,146 tok/s | 24 | 138 | 248 | 0.146 |
| our q4 pack of it | n/a | 2.49 GiB | 109 tok/s | 4,543 tok/s | 23 | 134 | 194 | 0.260 |
| Q4_K_M | 2.74 GB | 2.54 GiB | 90 tok/s | 4,158 tok/s | 24 | 139 | 251 | 0.143 |
| Q4_0 | 2.58 GB | 2.58 GiB | 99 tok/s | 4,215 tok/s | 24 | 121 | 227 | 0.178 |
| Q4_1 | 2.78 GB | 2.58 GiB | 99 tok/s | 4,216 tok/s | 24 | 138 | 238 | 0.148 |
| UD-Q4_K_XL | 2.91 GB | 2.70 GiB | 84 tok/s | 4,149 tok/s | 24 | 148 | 252 | 0.120 |
| Q5_K_S | 3.02 GB | 2.81 GiB | 83 tok/s | 4,126 tok/s | 24 | 136 | 272 | 0.076 |
| Q5_K_M | 3.14 GB | 2.92 GiB | 84 tok/s | 4,136 tok/s | 24 | 135 | 271 | 0.067 |
| UD-Q5_K_XL | 3.25 GB | 3.02 GiB | 85 tok/s | 4,138 tok/s | 24 | 136 | 274 | 0.074 |
| Q6_K | 3.53 GB | 3.27 GiB | 81 tok/s | 4,118 tok/s | 24 | 139 | 269 | 0.047 |
| UD-Q6_K_XL | 4.15 GB | 3.85 GiB | 74 tok/s | 4,157 tok/s | 24 | 142 | 280 | 0.036 |
| our q8 pack of it | n/a | 4.04 GiB | 77 tok/s | 4,441 tok/s | 24 | 141 | 291 | 0.019 |
| Q8_0 | 4.48 GB | 4.16 GiB | 74 tok/s | 4,146 tok/s | 24 | 141 | 294 | 0.018 |
| UD-Q8_K_XL | 5.95 GB | 5.53 GiB | 58 tok/s | 4,244 tok/s | 24 | 143 | 295 | 0.015 |
| BF16 | 8.42 GB | 7.83 GiB | 43 tok/s | 4,622 tok/s | 24 | 144 | 300 | 0.001 |
| original checkpoint, f16 | n/a | 7.83 GiB | 44 tok/s | 4,990 tok/s | 24 | 144 | 300 | 0.000 |

- A file's GPU memory is its size (2.74 GB is 2.55 GiB), except Q4_0, which takes 7 % more.
- The BF16 file gives the original's answer on all 300 questions: the import (names, transforms, value-head order,
  tokenizer, template) is exact.
- From 5 bits up a file answers like the original on 9 questions in 10 or more; the 4-bit K-quants on 5 in 6, where
  our round-to-nearest q4 pack of the same size does on 2 in 3; the 3-bit K-quants on 3 in 5 to 2 in 3; UD-IQ3_XXS
  and the 2-bit files on fewer than half.
- Our q4 pack is the fastest at its size (109 tok/s against 90 to 99 for the 4-bit files).

What each file holds (formats with at least 0.5 % of the weights; a file name is an average):

| Variant | File | Formats | Share of the weights |
|---|---|---|---|
| UD-IQ2_XXS | 1.52 GB | 8 | IQ2_XXS 64 %, Q5_K 15 %, IQ3_XXS 6 %, Q4_K 6 %, IQ3_S 3 %, IQ2_S 3 %, Q2_K 2 % |
| UD-IQ2_M | 1.76 GB | 5 | IQ2_S 58 %, Q5_K 21 %, IQ3_XXS 13 %, IQ3_S 8 % |
| UD-Q2_K_XL | 1.94 GB | 10 | Q2_K 39 %, Q6_K 15 %, Q3_K 15 %, IQ3_S 8 %, IQ3_XXS 8 %, Q5_K 6 %, IQ2_S 6 %, IQ4_XS 3 % |
| UD-IQ3_XXS | 1.95 GB | 7 | IQ3_XXS 71 %, Q5_K 15 %, Q6_K 6 %, IQ4_XS 3 %, IQ2_S 3 %, IQ3_S 2 % |
| Q3_K_S | 2.11 GB | 4 | Q3_K 79 %, Q6_K 15 %, Q4_K 6 % |
| Q3_K_M | 2.29 GB | 5 | Q3_K 46 %, Q4_K 37 %, Q6_K 15 %, Q5_K 1 % |
| UD-Q3_K_XL | 2.44 GB | 7 | Q4_K 28 %, Q3_K 27 %, Q6_K 21 %, IQ4_XS 13 %, IQ3_S 6 %, Q5_K 4 % |
| IQ4_XS | 2.48 GB | 4 | IQ4_XS 78 %, Q6_K 15 %, Q5_K 6 % |
| IQ4_NL | 2.58 GB | 4 | IQ4_NL 78 %, Q6_K 15 %, Q5_K 6 % |
| Q4_0 | 2.58 GB | 5 | Q4_0 77 %, Q6_K 15 %, Q5_K 6 %, Q4_1 2 % |
| Q4_K_S | 2.59 GB | 4 | Q4_K 76 %, Q6_K 15 %, Q5_K 8 % |
| Q4_K_M | 2.74 GB | 4 | Q4_K 58 %, Q6_K 24 %, Q5_K 18 % |
| Q4_1 | 2.78 GB | 4 | Q4_1 79 %, Q6_K 15 %, Q5_K 6 % |
| UD-Q4_K_XL | 2.91 GB | 6 | Q4_K 36 %, Q5_K 28 %, Q6_K 24 %, Q8_0 6 %, IQ4_XS 6 % |
| Q5_K_S | 3.02 GB | 3 | Q5_K 79 %, Q6_K 21 % |
| Q5_K_M | 3.14 GB | 3 | Q5_K 58 %, Q6_K 42 % |
| UD-Q5_K_XL | 3.25 GB | 5 | Q6_K 49 %, Q5_K 39 %, Q8_0 6 %, Q4_K 6 % |
| Q6_K | 3.53 GB | 2 | Q6_K 94 %, Q8_0 6 % |
| UD-Q6_K_XL | 4.15 GB | 3 | Q6_K 57 %, Q8_0 37 %, F16 6 % |
| Q8_0 | 4.48 GB | 1 | Q8_0 100 % |
| UD-Q8_K_XL | 5.95 GB | 2 | Q8_0 63 %, F16 37 % |
| BF16 | 8.42 GB | 1 | BF16 100 % |

The kernels against the same pack with every block format decoded to f16 as it is uploaded
(`--backend-opt block=f16`), on the same questions:

| Variant | Same answer (300) | Mean shift | Largest shift | Same answer (25 decisions) |
|---|---|---|---|---|
| IQ4_XS | 299 | 0.0011 | 0.0058 | 25 |
| Q3_K_S | 300 | 0.0010 | 0.0061 | 25 |
| Q4_K_M | 298 | 0.0009 | 0.0045 | 25 |
| Q6_K | 300 | 0.0010 | 0.0074 | 25 |
| UD-IQ2_M | 298 | 0.0024 | 0.0085 | 25 |
| UD-IQ2_XXS | 299 | 0.0020 | 0.0082 | 25 |
| UD-IQ3_XXS | 299 | 0.0013 | 0.0066 | 25 |
| UD-Q2_K_XL | 298 | 0.0013 | 0.0058 | 25 |

## Gemma 4 E2B

Files: [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF), all 21, and its three
drafter files. Text only: a GGUF file holds no vision tower, and the original checkpoint and our packs were measured
without theirs (`--no-vision`).

| Variant | File | GPU memory | Decode | Prefill | Test decisions right (25) | MMLU-Pro right (300) | Same answer as f16 (300) | Shift |
|---|---|---|---|---|---|---|---|---|
| UD-IQ2_M | 2.29 GB | 0.83 GiB | 114 tok/s | 6,325 tok/s | 22 | 61 | 122 | 0.524 |
| UD-IQ3_XXS | 2.37 GB | 0.90 GiB | 111 tok/s | 6,316 tok/s | 22 | 68 | 137 | 0.494 |
| UD-Q2_K_XL | 2.40 GB | 0.93 GiB | 122 tok/s | 5,837 tok/s | 22 | 56 | 121 | 0.528 |
| Q3_K_S | 2.45 GB | 0.97 GiB | 117 tok/s | 6,234 tok/s | 21 | 63 | 167 | 0.393 |
| Q3_K_M | 2.54 GB | 1.06 GiB | 116 tok/s | 6,056 tok/s | 22 | 69 | 199 | 0.296 |
| UD-Q3_K_XL | 2.92 GB | 1.14 GiB | 114 tok/s | 5,941 tok/s | 23 | 69 | 195 | 0.284 |
| IQ4_XS | 2.98 GB | 1.20 GiB | 98 tok/s | 6,182 tok/s | 22 | 58 | 247 | 0.168 |
| IQ4_NL | 3.04 GB | 1.25 GiB | 97 tok/s | 5,747 tok/s | 22 | 61 | 249 | 0.158 |
| Q4_K_S | 3.04 GB | 1.25 GiB | 124 tok/s | 6,175 tok/s | 22 | 64 | 239 | 0.179 |
| Q4_K_M | 3.11 GB | 1.31 GiB | 120 tok/s | 6,156 tok/s | 23 | 63 | 237 | 0.170 |
| Q4_0 | 3.04 GB | 1.36 GiB | 127 tok/s | 6,296 tok/s | 21 | 51 | 195 | 0.301 |
| Q4_1 | 3.15 GB | 1.36 GiB | 126 tok/s | 6,089 tok/s | 23 | 60 | 240 | 0.178 |
| UD-Q4_K_XL | 3.18 GB | 1.38 GiB | 117 tok/s | 6,198 tok/s | 23 | 65 | 233 | 0.184 |
| our q4 pack of it | n/a | 1.43 GiB | 121 tok/s | 4,792 tok/s | 22 | 64 | 227 | 0.229 |
| Q5_K_S | 3.32 GB | 1.51 GiB | 111 tok/s | 6,160 tok/s | 23 | 66 | 262 | 0.099 |
| Q5_K_M | 3.36 GB | 1.54 GiB | 110 tok/s | 5,546 tok/s | 23 | 65 | 261 | 0.092 |
| UD-Q5_K_XL | 4.29 GB | 1.59 GiB | 110 tok/s | 6,239 tok/s | 23 | 61 | 260 | 0.104 |
| Q6_K | 4.50 GB | 1.79 GiB | 109 tok/s | 6,011 tok/s | 23 | 65 | 276 | 0.071 |
| UD-Q6_K_XL | 4.71 GB | 1.98 GiB | 100 tok/s | 6,210 tok/s | 23 | 65 | 281 | 0.056 |
| our q8 pack of it | n/a | 2.23 GiB | 91 tok/s | 5,377 tok/s | 23 | 66 | 293 | 0.023 |
| Q8_0 | 5.05 GB | 2.29 GiB | 91 tok/s | 6,268 tok/s | 23 | 65 | 292 | 0.019 |
| UD-Q8_K_XL | 5.28 GB | 2.49 GiB | 85 tok/s | 6,319 tok/s | 23 | 65 | 293 | 0.020 |
| BF16 | 9.31 GB | 4.25 GiB | 61 tok/s | 6,676 tok/s | 23 | 62 | 300 | 0.000 |
| original checkpoint, f16 | n/a | 4.25 GiB | 61 tok/s | 6,220 tok/s | 23 | 62 | 300 | 0.000 |

- The BF16 file gives the original's answer on all 300 questions.
- GPU memory is under half the file: the per-layer embedding table (2.35 billion weights) is read on the host, a row
  per token, as it is from the original checkpoint.
- This model gets about one MMLU-Pro question in five right through the letter read-out at any precision, so only
  the agreement with f16 tells the files apart.
- Q4_0 agrees with f16 on 195 questions where Q4_1 and Q4_K_M agree on about 240: the file's own quality (the check
  below gives the same answers from its decoded weights).

What each file holds:

| Variant | File | Formats | Share of the weights |
|---|---|---|---|
| UD-IQ2_M | 2.29 GB | 8 | Q4_K 51 %, IQ2_S 31 %, Q3_K 9 %, IQ3_S 4 %, IQ3_XXS 3 %, IQ4_XS 1 %, F32 1 % |
| UD-IQ3_XXS | 2.37 GB | 9 | Q4_K 51 %, IQ3_XXS 32 %, Q3_K 9 %, IQ3_S 3 %, IQ2_S 2 %, IQ4_XS 1 %, F32 1 % |
| UD-Q2_K_XL | 2.40 GB | 10 | Q4_K 60 %, Q2_K 17 %, Q3_K 11 %, IQ3_S 5 %, IQ2_S 4 %, IQ4_XS 1 %, F32 1 % |
| Q3_K_S | 2.45 GB | 4 | Q4_K 51 %, Q3_K 49 %, F32 1 % |
| Q3_K_M | 2.54 GB | 5 | Q4_K 64 %, Q3_K 34 %, F32 1 % |
| UD-Q3_K_XL | 2.92 GB | 8 | Q5_K 52 %, Q4_K 24 %, Q3_K 17 %, IQ3_S 4 %, Q6_K 1 %, F32 1 % |
| IQ4_XS | 2.98 GB | 5 | Q5_K 51 %, IQ4_XS 39 %, Q4_K 9 %, F32 1 % |
| IQ4_NL | 3.04 GB | 5 | Q5_K 51 %, IQ4_NL 39 %, Q4_K 9 %, F32 1 % |
| Q4_0 | 3.04 GB | 6 | Q5_K 51 %, Q4_0 39 %, Q4_K 9 %, Q4_1 1 %, F32 1 % |
| Q4_K_S | 3.04 GB | 4 | Q5_K 51 %, Q4_K 48 %, F32 1 % |
| Q4_K_M | 3.11 GB | 5 | Q5_K 51 %, Q4_K 43 %, Q6_K 6 %, F32 1 % |
| Q4_1 | 3.15 GB | 5 | Q5_K 51 %, Q4_1 40 %, Q4_K 9 %, F32 1 % |
| UD-Q4_K_XL | 3.18 GB | 5 | Q5_K 64 %, Q4_K 29 %, Q6_K 5 %, F32 1 % |
| Q5_K_S | 3.32 GB | 3 | Q5_K 99 %, F32 1 % |
| Q5_K_M | 3.36 GB | 4 | Q5_K 93 %, Q6_K 6 %, F32 1 % |
| UD-Q5_K_XL | 4.29 GB | 6 | Q8_0 52 %, Q5_K 27 %, Q6_K 16 %, Q4_K 4 %, F32 1 % |
| Q6_K | 4.50 GB | 4 | Q8_0 51 %, Q6_K 49 %, F32 1 % |
| UD-Q6_K_XL | 4.71 GB | 4 | Q8_0 69 %, Q6_K 30 %, F32 1 % |
| Q8_0 | 5.05 GB | 3 | Q8_0 99 %, F32 1 % |
| UD-Q8_K_XL | 5.28 GB | 4 | Q8_0 94 %, F16 5 %, F32 1 % |
| BF16 | 9.31 GB | 1 | BF16 100 % |

The kernels against the weights decoded to f16 on upload:

| Variant | Same answer (300) | Mean shift | Largest shift | Same answer (25 decisions) |
|---|---|---|---|---|
| IQ4_XS | 300 | 0.0010 | 0.0082 | 25 |
| Q3_K_S | 300 | 0.0010 | 0.0074 | 25 |
| Q4_0 | 299 | 0.0011 | 0.0052 | 25 |
| Q4_K_M | 299 | 0.0010 | 0.0118 | 25 |
| Q6_K | 300 | 0.0010 | 0.0084 | 25 |
| UD-IQ2_M | 299 | 0.0011 | 0.0389 | 25 |

### The drafter files

`MTP/mtp-gemma-4-E2B-it-<Q8_0|BF16|F16>.gguf` (llama.cpp's `gemma4-assistant`) import as a drafter for
`--spec mtp --drafter <dir>`. They are Google's assistant checkpoint without its centroid head: the drafter scores
the whole vocabulary with its tied embedding, as llama.cpp does. With a target of Q4_K_M or Q8_0 and a drafter of
Q8_0 or BF16, greedy output is the same text with and without the drafter. Decode speed on `bench`'s generated text
(tok/s):

| Target | No drafter | Imported drafter, Q8_0 | Imported drafter, BF16 | Google's assistant (our q8 pack of it) |
|---|---|---|---|---|
| Q4_0 file | 127.6 | 242.9 | 239.5 | |
| Q4_K_M file | 118.7 | 182.6 | 179.8 | |
| Q8_0 file | 90.6 | 218.8 | 210.8 | |
| BF16 file | 61.3 | 179.5 | 176.7 | |
| our q4 pack of the original | 123.3 | 252.3 | | 267.7 |

A K-quant target gains less: a verify step of several rows has no matrix-engine kernel over block formats yet.

## Tool calls and chat templates

A pack imported from a GGUF renders the file's own chat template, which a publisher may have edited. `tests/e2e/
tool_calls.py` runs 23 cases against a server with simulated tools: 6 easy (one tool, one call), 11 medium (ten
tools offered, two calls at once, no tool needed, none that fits, a system prompt, thinking on, a streamed call) and
6 conversations of two or three user turns that need an earlier turn's tool result. A stronger model (Claude Sonnet
5.5) graded every transcript against `tests/e2e/tool_calls_rubric.md`, with one verdict for form (well-formed calls
to offered tools, the right finish reason, no markup leaked into the text, later turns consistent with earlier
ones: what the template and the output parser must get right) and one for the task (the right tool, arguments and
answer).

| Model | Run | Well-formed | Task right | Easy | Medium | Multi-turn |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | original checkpoint, f16 | 23 / 23 | 23 / 23 | 6 / 6 | 11 / 11 | 6 / 6 |
| Qwen3.5-4B | BF16 GGUF | 23 / 23 | 23 / 23 | 6 / 6 | 11 / 11 | 6 / 6 |
| Qwen3.5-4B | Q4_K_M GGUF | 23 / 23 | 23 / 23 | 6 / 6 | 11 / 11 | 6 / 6 |
| Gemma 4 E2B | original checkpoint, f16 | 23 / 23 | 22 / 23 | 6 / 6 | 11 / 11 | 5 / 6 |
| Gemma 4 E2B | BF16 GGUF | 23 / 23 | 22 / 23 | 6 / 6 | 11 / 11 | 5 / 6 |
| Gemma 4 E2B | Q4_K_M GGUF | 23 / 23 | 22 / 23 | 6 / 6 | 11 / 11 | 5 / 6 |

- The BF16 GGUF made the original's tool calls, name and arguments, in all 23 cases of both models. Gemma's whole
  transcript is identical to the original's (every answer's text and token count); Qwen's answer text differs in
  one case.
- The Q4_K_M files made the same calls as the originals too, apart from one optional argument left out and one
  email body worded differently (Qwen); their answers are worded differently in 12 cases of each model.
- Gemma's one miss is the same in all three runs: asked "should I take an umbrella or sunglasses?" after a weather
  call, it asks for more information instead of advising.
- No run had a request error, a template error or markup in its text; with thinking on, the reasoning came back
  apart from the answer.

## Kernel rates

Weights read per second (billions) by a chain of sixteen 2560 x 9216 matmuls with one, four and eight rows of x
(`XS_DEVICE_BENCH=1`, `tests/unit/backend/device_block_timing.cpp`). One row is a decode step; several are a
speculative verify.

| Format | Bits a weight | Kernel | 1 row | 4 rows | 8 rows |
|---|---|---|---|---|---|
| q4 | 5 (a pack) | pack q4 | 394 | 331 | 326 |
| q8 | 8.5 (a pack) | pack q8 | 274 | 228 | 220 |
| Q4_0 | 4.5 | pack q4 (same values) | 395 | 312 | 295 |
| Q4_1 | 5 | pack q4 (same values) | 396 | 315 | 297 |
| Q5_0 | 5.5 | block | 369 | 152 | 88 |
| Q5_1 | 6 | block | 359 | 156 | 89 |
| Q8_0 | 8.5 | pack q8 (same values) | 277 | 230 | 229 |
| Q2_K | 2.625 | block | 465 | 162 | 91 |
| Q3_K | 3.4375 | block | 353 | 152 | 88 |
| Q4_K | 4.5 | block | 393 | 159 | 89 |
| Q5_K | 5.5 | block | 339 | 144 | 86 |
| Q6_K | 6.5625 | block | 333 | 154 | 88 |
| IQ1_S | 1.5625 | block | 268 | 165 | 93 |
| IQ1_M | 1.75 | block | 272 | 131 | 80 |
| IQ2_XXS | 2.0625 | block | 385 | 167 | 99 |
| IQ2_XS | 2.3125 | block | 375 | 161 | 92 |
| IQ2_S | 2.5625 | block | 346 | 159 | 93 |
| IQ3_XXS | 3.0625 | block | 335 | 151 | 89 |
| IQ3_S | 3.4375 | block | 336 | 138 | 83 |
| IQ4_NL | 4.5 | block | 269 | 155 | 93 |
| IQ4_XS | 4.25 | block | 255 | 151 | 89 |

No format is bound by memory at this size, so decode speed follows the one-row column, not the bits. With several
rows the pack kernels have a matrix-engine GEMV and the block kernels do not.

## What it took

Lines added, from `git diff --numstat`.

The first model (Qwen3.5), which brought everything a GGUF needs:

| Area | Lines | Main files |
|---|---|---|
| Weight type and host decoders, all 19 formats | 736 | `quant/block_decode.cpp`, `tensor/block_format.h` |
| GPU kernels and backend | 1,347 | `backend/sycl/sycl_block_kernels.cpp`, `sycl_block_layout.cpp` |
| GGUF reader and importer, any model | 1,388 | `io/gguf.cpp`, `convert/gguf/gguf_import.cpp`, `tokenizer.cpp` |
| Qwen3.5 itself | 407 | `convert/gguf/qwen35.cpp` (369), the value-head order in `models/qwen3_5/` (38) |
| Command line | 54 | `cli/main.cpp` |
| Tests | 2,733 | 15 files |
| ggml's codebook tables (data) | 1,196 | `third_party/ggml/` |

The second architecture (Gemma 4, with its drafter):

| Area | Lines | Main files |
|---|---|---|
| Gemma 4 itself | 433 | `convert/gguf/gemma4.cpp` |
| Shared importer | 431 added, 233 removed | what Qwen3.5's file shared moved to `convert/gguf/layout`; a tokenizer kind (BPE with scores) |
| Model code, kernels, backend | 0 | |
| Tests | 1,497 | 8 files |
