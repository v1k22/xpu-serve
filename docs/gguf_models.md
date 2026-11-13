# GGUF files: what was measured

Every result behind xpu-serve's GGUF support, in one place: three language models and the decision models
Clef-Flash, Clef, Laya and Julia-1, every file their publisher offers that fits one Intel Arc Pro B60 (24 GB).
[Decision 0016](decisions/0016-gguf.md) has the design; the README's GGUF section has the short version.

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
(tok/s), measured before the block formats had a matrix-engine GEMV for the verify:

| Target | No drafter | Imported drafter, Q8_0 | Imported drafter, BF16 | Google's assistant (our q8 pack of it) |
|---|---|---|---|---|
| Q4_0 file | 127.6 | 242.9 | 239.5 | |
| Q4_K_M file | 118.7 | 182.6 | 179.8 | |
| Q8_0 file | 90.6 | 218.8 | 210.8 | |
| BF16 file | 61.3 | 179.5 | 176.7 | |
| our q4 pack of the original | 123.3 | 252.3 | | 267.7 |

With it (`gemv_block_xmx`, see Qwen3.8-27B below) a K-quant target gains more. The builds before and after it side by
side in one session, two rounds each, the Q8_0 drafter: the Q4_K_M file 182 and 183 tok/s before, 234 and 231 after;
the Q8_0 file, which no block format kernel reads, 218 and 220 before, 219 and 216 after. At these speeds a drafter's
decode varies by 10 to 15 % between sessions and cards, so only runs side by side compare.

## Qwen3.8-27B

Files: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), the 20 that fit the card, and its
MTP file. The rest hold more weights than a 24 GB card has room for beside the KV cache and activations: UD-Q6_K_L
(24.2 GB of weights, the card 24.4 GB), UD-Q6_K_XL, Q8_0, UD-Q8_K_L, UD-Q8_K_XL, and the BF16 file (also split in
two, which the import refuses by name).

The original checkpoint (54 GB in bf16) does not fit one card either, so the reference is the original in float32 on
the CPU through the read-out's reference code (`tests/parity/decision/reference.py`, which runs OpenJev's own helper:
the prompt, letters and settings of `--readout openjev`), and every GPU run here reads with `--readout openjev`.
"Same answer as the reference" and "shift" compare with it. MTP: greedy decode of 256 tokens over the six chat prompts
of [the spec decode measurements](design-spec-decode.md) (`bench --prompts`), without and with the model's MTP head at
k = 1 to 5 drafts a round; the five files without the head take it from the MTP file (`convert --mtp`).

| Variant | File | GPU memory | Decode | Prefill | Test decisions right (25) | MMLU-Pro right (300) | Same answer as the reference (300) | Shift | Decode with MTP (best k) |
|---|---|---|---|---|---|---|---|---|---|
| UD-IQ1_S | 6.19 GB | 5.75 GiB | 14.7 tok/s | 925 tok/s | 18 | 61 | 52 | 0.760 | 24.8 tok/s (k = 5), 1.65x |
| UD-IQ1_M | 6.73 GB | 6.25 GiB | 15.9 tok/s | 928 tok/s | 23 | 96 | 92 | 0.632 | 28.5 tok/s (k = 5), 1.76x |
| UD-IQ2_XXS | 7.21 GB | 6.70 GiB | 17.4 tok/s | 935 tok/s | 24 | 154 | 174 | 0.352 | 31.5 tok/s (k = 5), 1.78x |
| UD-IQ2_XS | 7.63 GB | 7.09 GiB | 18.0 tok/s | 937 tok/s | 24 | 149 | 180 | 0.346 | 33.0 tok/s (k = 4), 1.79x |
| UD-IQ2_S | 8.42 GB | 7.83 GiB | 18.3 tok/s | 935 tok/s | 24 | 167 | 198 | 0.303 | 35.8 tok/s (k = 4), 1.92x |
| UD-Q2_K_XL | 9.95 GB | 8.93 GiB | 18.1 tok/s | 933 tok/s | 24 | 171 | 219 | 0.224 | 38.7 tok/s (k = 5), 2.09x |
| UD-IQ3_XXS | 10.93 GB | 9.84 GiB | 17.5 tok/s | 929 tok/s | 24 | 182 | 236 | 0.186 | 36.7 tok/s (k = 5), 2.06x |
| UD-IQ3_S | 12.04 GB | 10.87 GiB | 17.0 tok/s | 925 tok/s | 23 | 173 | 229 | 0.216 | 39.2 tok/s (k = 5), 2.28x |
| UD-Q3_K_XL | 13.15 GB | 11.90 GiB | 15.8 tok/s | 922 tok/s | 24 | 178 | 240 | 0.148 | 37.8 tok/s (k = 5), 2.36x |
| UD-IQ4_XS | 14.35 GB | 13.03 GiB | 14.6 tok/s | 920 tok/s | 24 | 187 | 259 | 0.101 | 41.3 tok/s (k = 5), 2.79x |
| UD-Q4_K_S | 15.36 GB | 13.96 GiB | 15.4 tok/s | 916 tok/s | 24 | 192 | 270 | 0.090 | 39.9 tok/s (k = 5), 2.54x |
| Q4_0 | 16.06 GB | 16.01 GiB | 21.3 tok/s | 929 tok/s | 24 | 180 | 251 | 0.133 | 47.4 tok/s (k = 5), 2.20x |
| UD-Q4_K_M | 16.56 GB | 15.09 GiB | 15.5 tok/s | 914 tok/s | 24 | 184 | 279 | 0.073 | 38.2 tok/s (k = 5), 2.43x |
| Q4_1 | 17.54 GB | 16.01 GiB | 21.4 tok/s | 926 tok/s | 24 | 184 | 265 | 0.087 | 46.7 tok/s (k = 5), 2.16x |
| UD-Q4_K_XL | 17.66 GB | 16.11 GiB | 16.1 tok/s | 912 tok/s | 24 | 187 | 280 | 0.063 | 37.9 tok/s (k = 5), 2.33x |
| UD-Q5_K_S | 18.76 GB | 17.14 GiB | 16.8 tok/s | 910 tok/s | 24 | 185 | 278 | 0.059 | 35.6 tok/s (k = 5), 2.08x |
| UD-Q5_K_M | 19.87 GB | 18.17 GiB | 17.1 tok/s | 908 tok/s | 24 | 189 | 279 | 0.050 | 33.4 tok/s (k = 5), 1.93x |
| UD-Q5_K_XL | 20.98 GB | 19.20 GiB | 16.8 tok/s | 907 tok/s | 24 | 193 | 288 | 0.039 | 32.9 tok/s (k = 4), 1.94x |
| UD-Q6_K | 21.98 GB | 20.13 GiB | 16.8 tok/s | 906 tok/s | 24 | 191 | 289 | 0.032 | 32.5 tok/s (k = 4), 1.91x |
| UD-Q6_K_M | 23.09 GB | 21.16 GiB | 16.1 tok/s | 906 tok/s | 24 | 193 | 293 | 0.026 | 33.1 tok/s (k = 5), 2.03x |
| our q4 pack of it | n/a | 16.21 GiB | 22.3 tok/s | 974 tok/s | 24 | 185 | 245 | 0.144 | 50.5 tok/s (k = 5), 2.25x |
| *original, f32 on the CPU* | | | | | 24 | 190 | 300 | 0 | |

- **Accuracy**: the 4- to 6-bit K-quant and i-quant files give the reference's answer on 259 to 293 questions,
  UD-Q4_K_XL on 280; the 3-bit files on 229 to 240, the 2-bit ones on 174 to 219, the 1-bit ones lose most. Q4_0 (251) and our
  round-to-nearest q4 pack (245) keep fewer than the 4-bit K-quant files (270 to 280).
- **Decode** follows the kernels, not the bytes: the K-quant and i-quant files run at 15 to 18 tok/s whatever their
  size, Q4_0 and Q4_1 (read by a pack's q4 kernel) at 21, our q4 pack at 22.
- **MTP** speeds every file up 1.65 to 2.79 times, to 25 to 47 tok/s; the head keeps 3.0 to 3.4 tokens a verify at
  k = 5 in every file. A file whose one-row GEMV is slow gains most (UD-IQ4_XS: 14.8 to 41.3 tok/s, IQ4's levels come
  from a table in the verify's kernel).
- Greedy text with MTP (k = 4) equals greedy text without on three prompts each for UD-Q4_K_XL and Q4_0.
- One MTP sweep of UD-Q6_K_M, the largest file that fits (1.5 GiB left beside its weights), lost the device on its
  first run; the eight runs of the same bench after it did not.

Decode with MTP by drafts a round, in tok/s, with the tokens a verify keeps in parentheses:

| Variant | Without | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|---|
| UD-IQ1_S | 15.0 | 16.7 (1.78) | 21.1 (2.36) | 23.3 (2.71) | 24.0 (2.86) | 24.8 (3.06) |
| UD-IQ1_M | 16.2 | 19.3 (1.80) | 23.9 (2.38) | 27.0 (2.76) | 27.5 (2.97) | 28.5 (3.17) |
| UD-IQ2_XXS | 17.7 | 22.1 (1.82) | 27.4 (2.37) | 30.2 (2.74) | 31.2 (2.95) | 31.5 (3.08) |
| UD-IQ2_XS | 18.4 | 23.5 (1.80) | 29.2 (2.36) | 31.8 (2.70) | 33.0 (2.91) | 32.4 (3.01) |
| UD-IQ2_S | 18.6 | 24.5 (1.82) | 30.3 (2.38) | 34.2 (2.80) | 35.8 (3.02) | 35.6 (3.19) |
| UD-Q2_K_XL | 18.5 | 26.8 (1.86) | 33.4 (2.46) | 37.2 (2.86) | 38.1 (3.08) | 38.7 (3.26) |
| UD-IQ3_XXS | 17.8 | 25.6 (1.81) | 32.5 (2.43) | 35.3 (2.76) | 36.6 (3.02) | 36.7 (3.14) |
| UD-IQ3_S | 17.2 | 25.9 (1.85) | 33.2 (2.50) | 36.7 (2.88) | 38.7 (3.19) | 39.2 (3.36) |
| UD-Q3_K_XL | 16.0 | 25.7 (1.82) | 32.5 (2.45) | 36.4 (2.85) | 37.5 (3.10) | 37.8 (3.27) |
| UD-IQ4_XS | 14.8 | 26.8 (1.86) | 34.0 (2.50) | 38.3 (2.93) | 41.0 (3.29) | 41.3 (3.44) |
| UD-Q4_K_S | 15.7 | 26.4 (1.86) | 34.1 (2.50) | 37.9 (2.95) | 39.3 (3.23) | 39.9 (3.41) |
| Q4_0 | 21.5 | 34.0 (1.85) | 40.5 (2.48) | 45.6 (2.93) | 47.4 (3.21) | 47.4 (3.37) |
| UD-Q4_K_M | 15.7 | 25.5 (1.84) | 32.2 (2.46) | 35.9 (2.88) | 37.2 (3.11) | 38.2 (3.32) |
| Q4_1 | 21.6 | 33.9 (1.84) | 41.1 (2.48) | 45.3 (2.85) | 45.9 (3.05) | 46.7 (3.25) |
| UD-Q4_K_XL | 16.3 | 24.9 (1.87) | 32.2 (2.54) | 35.7 (2.94) | 37.6 (3.27) | 37.9 (3.41) |
| UD-Q5_K_S | 17.1 | 23.6 (1.84) | 29.8 (2.45) | 33.8 (2.89) | 35.6 (3.21) | 35.6 (3.32) |
| UD-Q5_K_M | 17.3 | 23.3 (1.84) | 29.0 (2.46) | 31.9 (2.85) | 33.0 (3.11) | 33.4 (3.25) |
| UD-Q5_K_XL | 17.0 | 22.8 (1.84) | 28.5 (2.43) | 32.0 (2.85) | 32.9 (3.14) | 32.8 (3.26) |
| UD-Q6_K | 17.0 | 22.7 (1.85) | 28.9 (2.46) | 31.8 (2.89) | 32.5 (3.15) | 32.4 (3.28) |
| UD-Q6_K_M | 16.3 | 22.6 (1.84) | 28.3 (2.48) | 32.1 (2.92) | 32.8 (3.23) | 33.1 (3.40) |
| our q4 pack of it | 22.4 | 35.8 (1.85) | 41.9 (2.46) | 48.1 (2.92) | 49.9 (3.20) | 50.5 (3.35) |

Before the matrix-engine GEMV over block formats (a verify of k + 1 rows ran the GEMV, whose cost grows with every
row), MTP gained 1.2 to 1.4 times on the K-quant and i-quant files at their best k, which was 1 (UD-Q4_K_XL 16.2 to
22.0 tok/s, UD-Q2_K_XL 18.4 to 22.7, UD-IQ3_XXS 17.8 to 21.6), and Q4_0 1.98 times. A verify of 5 rows took 246 ms
over UD-Q4_K_XL and 74 over Q4_0 (its Q5_K and Q6_K tensors on the GEMV); it now takes 75 and 56, where a decode
step takes 61 and 47.

On an Arc B390 (Panther Lake, 30 GB of memory shared with the system) the Q4_0 file decodes at 6.4 tok/s, and with
MTP (the same prompts; k = 1 to 5 the mean of two runs):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| tok/s (tokens a verify keeps) | 11.1 (1.85) | 14.3 (2.48) | 15.9 (2.93) | 16.5 (3.21) | 17.2 (3.37) | 17.4 (3.50) | 16.9 (3.58) | 10.6 (3.65) |

A verify there costs about a decode step up to 8 rows (162 ms at 2 rows, 173 at 8, against 156 for a decode step), so
MTP gains more than on the B60: 2.7 times at k = 6. At k = 8 the verify's 9 rows leave the GEMV (302 ms). The greedy
text is the same with MTP. Loading the 16 GiB pack right after another run lost the device in 2 of 8 loads, while the
GPU driver still held the last run's pages; these numbers come from loads with that memory released first (TODO.md).

What each file holds:

| Variant | Weights | Formats | MTP head | Share of the weights |
|---|---|---|---|---|
| UD-IQ1_S | 6.18 GB | 8 | none: `--mtp` | IQ1_S 68 %, IQ2_XXS 11 %, Q2_K 6 %, Q3_K 5 %, IQ1_M 4 %, IQ2_XS 3 %, IQ2_S 2 %, IQ3_XXS 1 % |
| UD-IQ1_M | 6.72 GB | 9 | none: `--mtp` | IQ1_S 44 %, IQ2_XXS 24 %, IQ1_M 8 %, IQ2_S 6 %, Q2_K 5 %, Q3_K 5 %, IQ2_XS 4 %, IQ3_XXS 3 % |
| UD-IQ2_XXS | 7.20 GB | 9 | none: `--mtp` | IQ2_XXS 42 %, IQ1_S 30 %, IQ2_S 8 %, IQ2_XS 6 %, IQ3_XXS 6 %, Q3_K 5 %, IQ3_S 1 % |
| UD-IQ2_XS | 7.61 GB | 10 | none: `--mtp` | IQ2_XXS 40 %, IQ1_S 21 %, IQ2_XS 11 %, IQ2_S 10 %, IQ3_XXS 8 %, Q3_K 5 %, IQ3_S 4 %, IQ4_XS 1 % |
| UD-IQ2_S | 8.41 GB | 10 | none: `--mtp` | IQ2_XXS 29 %, IQ3_XXS 15 %, IQ1_S 13 %, IQ2_S 13 %, IQ2_XS 10 %, Q2_K 6 %, IQ3_S 5 %, Q4_K 5 %, IQ4_XS 2 %, IQ1_M 1 % |
| UD-Q2_K_XL | 9.94 GB | 11 | in the file | IQ3_XXS 27 %, IQ2_XXS 15 %, IQ2_S 14 %, IQ2_XS 9 %, IQ3_S 8 %, Q2_K 8 %, IQ4_XS 7 %, IQ1_S 6 %, Q4_K 5 %, Q6_K 2 % |
| UD-IQ3_XXS | 10.92 GB | 12 | in the file | IQ3_XXS 29 %, IQ3_S 22 %, IQ4_XS 10 %, IQ2_S 10 %, IQ2_XXS 7 %, Q2_K 6 %, Q4_K 6 %, IQ2_XS 4 %, IQ1_S 3 %, Q6_K 2 %, Q3_K 1 %, IQ1_M 1 % |
| UD-IQ3_S | 12.03 GB | 12 | in the file | IQ3_S 32 %, IQ4_XS 18 %, IQ3_XXS 18 %, Q3_K 8 %, IQ2_S 7 %, Q5_K 5 %, IQ2_XXS 4 %, IQ2_XS 4 %, Q6_K 2 %, Q4_K 2 %, Q2_K 1 %, IQ1_S 1 % |
| UD-Q3_K_XL | 13.14 GB | 12 | in the file | IQ4_XS 34 %, IQ3_S 30 %, IQ3_XXS 9 %, Q3_K 7 %, Q5_K 6 %, IQ2_S 5 %, Q4_K 4 %, Q6_K 2 %, IQ2_XS 1 %, Q2_K 1 %, IQ2_XXS 1 % |
| UD-IQ4_XS | 14.34 GB | 11 | in the file | IQ4_XS 56 %, Q5_K 11 %, IQ3_S 11 %, Q4_K 6 %, IQ3_XXS 6 %, Q3_K 6 %, IQ2_S 2 %, Q6_K 2 %, IQ2_XS 1 % |
| UD-Q4_K_S | 15.35 GB | 10 | in the file | IQ4_XS 43 %, Q4_K 19 %, Q5_K 13 %, Q3_K 9 %, Q6_K 7 %, IQ3_S 5 %, IQ3_XXS 2 %, IQ4_NL 2 % |
| Q4_0 | 16.05 GB | 4 | in the file | Q4_0 87 %, Q5_K 6 %, Q6_K 5 %, Q4_1 3 % |
| UD-Q4_K_M | 16.55 GB | 8 | in the file | IQ4_XS 39 %, Q5_K 28 %, Q4_K 20 %, Q6_K 8 %, IQ4_NL 3 %, Q3_K 1 %, IQ3_S 1 % |
| Q4_1 | 17.53 GB | 3 | in the file | Q4_1 90 %, Q5_K 6 %, Q6_K 5 % |
| UD-Q4_K_XL | 17.65 GB | 7 | in the file | Q5_K 45 %, IQ4_XS 27 %, Q6_K 12 %, Q4_K 12 %, IQ4_NL 3 %, Q8_0 1 % |
| UD-Q5_K_S | 18.75 GB | 6 | in the file | Q5_K 55 %, Q6_K 19 %, IQ4_XS 13 %, Q4_K 8 %, IQ4_NL 2 %, Q8_0 2 % |
| UD-Q5_K_M | 19.86 GB | 6 | in the file | Q5_K 50 %, Q6_K 33 %, Q4_K 7 %, IQ4_XS 5 %, Q8_0 4 %, IQ4_NL 1 % |
| UD-Q5_K_XL | 20.97 GB | 6 | in the file | Q5_K 46 %, Q6_K 40 %, Q8_0 9 %, IQ4_XS 4 %, Q4_K 1 % |
| UD-Q6_K | 21.97 GB | 3 | in the file | Q6_K 62 %, Q5_K 29 %, Q8_0 9 % |
| UD-Q6_K_M | 23.08 GB | 3 | in the file | Q6_K 62 %, Q8_0 20 %, Q5_K 18 % |
| UD-Q6_K_L (does not fit) | 24.17 GB | 3 | in the file | Q6_K 60 %, Q8_0 31 %, Q5_K 9 % |
| MTP/mtp-Qwen3.8-27B-Q4_0 | 1.36 GB | 3 | the head alone | Q3_K 86 % (its own embedding and output head), Q4_K 11 %, Q6_K 4 % |

## Clef-Flash and Clef (decision models)

Files: [ggml-org/Clef-Flash-GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF), all three (BF16, Q8_0, Q4_K_M), and
Clef's Q4_K_M from [ggml-org/Clef-GGUF](https://huggingface.co/ggml-org/Clef-GGUF) (Clef is Qwen3.8-27B with the same head; its Q8_0,
28.7 GB, and BF16, 54.1 GB, do not fit the card).
llama.cpp's `clef` architecture is the Qwen3.5 decoder under the `clef.` namespace plus Clef's joint schema head
(`dec.blk.*`, `decision.*`); `convert` imports it into the pack the Clef model serves from the original checkpoint
(decision 0016). The vision files (`mmproj-*`) are not read.

The reference is the original checkpoint's own code (`joint_schema_model.py`) in float32 on the CPU
(`tests/parity/decision/reference.py`), on the 25 test decisions and the 300 MMLU-Pro questions; each pack is served
through the System One API and scored against it (`tests/e2e/systemone_eval.py --reference`). "Same as the
reference" counts the questions answered as the reference answers them; the probability differences are per option.
Latency is the p50 of the 25 test decisions as served (about 180 tokens each), and `bench` (3 questions, 325
tokens).

| Variant | File | GPU memory | 25 decisions right | MMLU-Pro right (300) | Same answer as the reference (300) | Probability difference, largest / p95 | Latency p50 | `bench` p50 |
|---|---|---|---|---|---|---|---|---|
| BF16 | 18.2 GB | 15.01 GiB | 25 | 198 | 299 | 0.0019 / 0.0006 | 96 ms | 129 ms |
| Q8_0 | 9.7 GB | 7.98 GiB | 25 | 197 | 295 | 0.035 / 0.017 | 147 ms | 191 ms |
| Q4_K_M | 6.7 GB | 5.25 GiB | 25 | 195 | 287 | 0.39 / 0.12 | 150 ms | 185 ms |
| *original at f16* | | 15.01 GiB | 25 | 198 | 299 | 0.0016 / 0.0005 | 90 ms | 124 ms |
| *original in f32 on the CPU* | | | 25 | 198 | 300 | 0 | | |
| Clef Q4_K_M | 19.5 GB | 16.93 GiB | 25 | 200 | 281 | 0.19 / 0.11 | 443 ms | 549 ms |
| *Clef, original in f32 on the CPU* | | | 25 | 203 | 300 | 0 | | |

- **The import is exact**: on each file the config and the head's shapes are the original's, the BF16 file's head
  holds the original's values, its Q8_0 matrices (in both quantized files) are within 0.42 % of them, and the scales
  the file stores are exp and sigmoid of the original's parameters (`gguf_clef_real_file.cpp`).
- **BF16 is the original**: it answers as the original at f16 does. Q8_0 changes 5 answers of 300 and Q4_K_M 13.
- **Clef's Q4_K_M** (compared with its own original's code, which at 27B does not fit the card at f16) changes 19
  answers of 300 and fits with 5.8 GiB to spare; a decision costs about 0.44 s, five times Clef-Flash's.
- **The quantized files are slower**: a decision is one prefill over the prompt, which dequantizes every quantized
  weight for its GEMM (as 0015 measured for q8 and q4 packs), so they cost about 50 ms more than f16.

## Laya and Julia-1 (decision models)

Files: [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF) and
[ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF), both files each (BF16, Q8_0). llama.cpp's
`modern-bert` architecture with a Laya read-out (`decision.type = "laya"`) holds the encoder, the two-layer head and
the scorer; `convert` imports either file into the original checkpoint's layout, telling the two models apart by the
prompt template the converter writes (decision 0018). The BF16 files are roundings of the originals (F16 for Laya,
F32 for Julia-1), not copies. Measured as Clef-Flash above, against each original's own code in fp32; Julia-1 refuses
what its strict policy refuses (8 of the 25 test decisions, 5 of the 300 questions), exactly as its own code does.

| Variant | File | GPU memory | 25 decisions: right, same as the reference | MMLU-Pro: right, same as the reference | Probability difference, largest / p95 | Latency p50 |
|---|---|---|---|---|---|---|
| Laya-BF16 | 0.84 GB | 0.78 GiB | 19, 25 | 38, 299 of 300 | 0.055 / 0.006 | 8 ms |
| Laya-Q8_0 | 0.45 GB | 0.42 GiB | 19, 25 | 38, 293 of 300 | 0.052 / 0.023 | 9 ms |
| *Laya, original at f16* | | 0.78 GiB | 19, 25 | 37, 300 of 300 | 0.016 / 0.001 | 8 ms |
| Julia-1-BF16 | 0.30 GB | 0.27 GiB | 8, 17 of 17 | 42, 292 of 295 | 0.074 / 0.034 | 5 ms |
| Julia-1-Q8_0 | 0.17 GB | 0.14 GiB | 9, 16 of 17 | 41, 271 of 295 | 0.55 / 0.22 | 8 ms |
| *Julia-1, original at f16* | | 0.27 GiB | 8, 17 of 17 | 43, 294 of 295 | 0.029 / 0.008 | 7 ms |

- **The import keeps the files' values**: the tokenizers equal the originals' on every token (50,368 and 256,000), the
  configs and head are the originals' (`gguf_modern_bert_real_file.cpp`), and every request becomes the same tokens.
- Both are small classifiers: on MMLU-Pro's ten-option questions their references get 37 and 43 of 300 right, and the
  engine gets the same. Julia-1's Q8_0 file moves its answers the most (24 of 295 changed).

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

Weights read per second (billions) by a chain of sixteen 2560 x 9216 matmuls with 1 to 8 rows of x
(`XS_DEVICE_BENCH=1`, `tests/unit/backend/device_block_timing.cpp`). One row is a decode step; several are a
speculative verify, which a block format runs on its matrix-engine GEMV from 2 rows (`gemv_block_xmx`; a pack's q4
from 4 rows). The last column is the vector GEMV at 8 rows (`--backend-opt gemv=vector`), what a verify ran before.

| Format | Bits a weight | Kernel | 1 row | 2 rows | 4 rows | 8 rows | 8 rows, vector GEMV |
|---|---|---|---|---|---|---|---|
| q4 | 5 (a pack) | pack q4 | 391 | 362 | 317 | 296 | 196 |
| q8 | 8.5 (a pack) | pack q8 (no matrix-engine GEMV) | 277 | 270 | 235 | 228 | 228 |
| Q4_0 | 4.5 | pack q4 (same values) | 393 | 364 | 319 | 301 | 195 |
| Q4_1 | 5 | pack q4 (same values) | 392 | 361 | 319 | 299 | 196 |
| Q5_0 | 5.5 | block | 372 | 257 | 253 | 237 | 89 |
| Q5_1 | 6 | block | 361 | 233 | 227 | 219 | 89 |
| Q8_0 | 8.5 | pack q8 (same values) | 277 | 271 | 235 | 229 | 224 |
| Q2_K | 2.625 | block | 468 | 316 | 308 | 289 | 92 |
| Q3_K | 3.4375 | block | 363 | 260 | 257 | 246 | 87 |
| Q4_K | 4.5 | block | 393 | 295 | 288 | 270 | 89 |
| Q5_K | 5.5 | block | 350 | 239 | 236 | 227 | 87 |
| Q6_K | 6.5625 | block | 340 | 235 | 231 | 222 | 87 |
| IQ1_S | 1.5625 | block | 277 | 193 | 189 | 179 | 91 |
| IQ1_M | 1.75 | block | 274 | 187 | 183 | 175 | 80 |
| IQ2_XXS | 2.0625 | block | 389 | 305 | 299 | 282 | 96 |
| IQ2_XS | 2.3125 | block | 380 | 297 | 295 | 280 | 90 |
| IQ2_S | 2.5625 | block | 350 | 285 | 278 | 262 | 92 |
| IQ3_XXS | 3.0625 | block | 337 | 269 | 267 | 255 | 87 |
| IQ3_S | 3.4375 | block | 341 | 262 | 256 | 244 | 83 |
| IQ4_NL | 4.5 | block | 273 | 236 | 264 | 250 | 92 |
| IQ4_XS | 4.25 | block | 266 | 278 | 275 | 262 | 88 |

No format is bound by memory at this size, so decode speed follows the one-row column, not the bits. From 2 rows on
the matrix-engine GEMV costs about the same at every row count: a verify of 8 rows reads its weights 2.0 to 3.1 times
faster than the vector GEMV did. At 2 rows of this size the vector GEMV is still ahead for Q5_K, Q6_K, Q5_0, Q5_1
and IQ1 (by 5 to 24 %); at Qwen3.8-27B's MLP size only for Q5_1 and IQ1. A chain of four 5120 x 17408
matmuls, the formats of its files:

| Format | 1 row | 2 rows | 4 rows | 8 rows | 8 rows, vector GEMV |
|---|---|---|---|---|---|
| q4 (pack) | 400 | 376 | 372 | 348 | 196 |
| Q4_K | 419 | 356 | 345 | 321 | 94 |
| Q5_K | 372 | 282 | 269 | 249 | 90 |
| Q6_K | 341 | 290 | 279 | 267 | 90 |
| Q3_K | 320 | 304 | 296 | 277 | 90 |
| IQ4_XS | 266 | 331 | 322 | 306 | 93 |
| IQ4_NL | 270 | 318 | 316 | 297 | 89 |
| IQ3_S | 363 | 305 | 287 | 279 | 86 |
| IQ3_XXS | 362 | 327 | 311 | 288 | 91 |
| IQ2_XXS | 411 | 369 | 357 | 331 | 96 |
| IQ1_S | 279 | 198 | 194 | 181 | 93 |

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

The third model (Qwen3.8-27B) needed no architecture code. Its five smallest files come without the MTP head, and a
verify over block formats was slow; both are general:

| Area | Lines | Main files |
|---|---|---|
| A matrix-engine GEMV over block formats | 481 added, 163 removed | `backend/sycl/sycl_block_kernels.cpp`; the DPAS builtins and the preparation of x moved from `sycl_quant_gemm.cpp` to `sycl_xmx.h`; the path in `sycl_matmul.cpp` |
| An MTP head from a file of its own (`convert --mtp`) | 240 added, 45 removed | `convert/gguf/qwen35.cpp` (121), `gguf_import.cpp`, `cli/main.cpp` |
| Qwen3.8-27B itself | 0 | |
| Tests | 340 added, 55 removed | 8 files |
