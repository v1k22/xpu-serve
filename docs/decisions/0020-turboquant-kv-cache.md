# 0020: TurboQuant KV cache formats (`--kv-dtype turbo4 | turbo3 | turbo2`)

Status: **accepted (opt-in)** · 2026-11-14

## Context

The int8 KV cache (0010) halves the cache. TurboQuant (arXiv 2504.19874) goes to 4, 3 or 2 bits a value: a row of a
cache is rotated so that its coordinates spread like a Gaussian's, each coordinate is rounded to the nearest of a few
fixed centroids, and one norm is kept per rotated group. The formats taken here are the ones
[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) settled on and its llama.cpp fork stores
(`turbo4`, `turbo3`, `turbo2`: no QJL stage, a rotation and one norm per 128 values, the norm corrected so a decoded
group has the row's own norm). vLLM merged the same idea as `--kv-cache-dtype turboquant_*`, upstream llama.cpp its
rotation.

## Decision

- **A KV cache has a format for its keys and one for its values** (`quant::KvFormats`, `SessionOptions::kv`;
  `quant/kv_format.h`): `auto` (the precision's float), `i8`, `turbo4`, `turbo3`, `turbo2`. `--kv-dtype <format>` sets
  both, `<keys>:<values>` each. The SYCL backend takes both alike, or `i8` keys with TurboQuant values; cpu-ref takes
  any pair. This replaced `SessionOptions::kv_dtype`.
- **The formats** (`quant/kv_turbo.h`): a row (one token's head) is cut into groups of 128 values; a group is sign
  flipped, Walsh-Hadamard transformed and sign flipped again (the fork's sign patterns), each coordinate coded against
  the Lloyd-Max centroids of a Gaussian of variance 1/128 (the fork's tables), the codes packed as the fork packs
  them, and an f16 norm stored beside them: **66, 50 and 34 bytes per 128 values (4.125, 3.125 and 2.125 bits a
  value; 3.88, 5.12 and 7.53 times smaller than f16)**. A head_dim that is not a multiple of 128 is refused by name.
- **One oracle**: `quant::round_trip_kv` is the value a format stores for a row, in the original domain. cpu-ref's
  cache holds exactly that, and every device path is tested against attention over those values.
- **Host and device store the same bits.** The code of a coordinate is decided without a division or a square root:
  with w the unnormalized transform of the group and n its squared norm, the coordinate lies beyond an edge e when
  `w * w >= (128 e^2) * n`. The sums behind n and the norm are added in one fixed order (each lane's 8 values, then
  the 16 lanes pairwise) with an explicit fused multiply-add, on the host as in the kernel. The reference
  implementation divides by the norm first, so a coordinate that sits on an edge could round the other way; on the 43
  fixture rows of its own quantizer (`tests/fixtures/quant/turbo`: Gaussian, uniform, spiky, constant and zero rows)
  all 5,504 codes of each format are the same and every norm is within one f16 step.
- **Attention never rotates a key back.** The rotation is orthogonal, so the query is rotated once and meets the
  decoded keys where they are; the weighted sum of decoded values is rotated back once.
  - Decode (the per-row kernel) reads the codes: a lane loads 8 codes of a group in one or two loads, looks each
    centroid up in a small table, and rotates its query and its sum in registers (three butterfly stages inside a
    lane, four between lanes by sub-group shuffles).
  - Prefill (GEMMs or the fused kernel) decodes the keys and values in view to f16 as they are stored, rotated,
    rotates the chunk's queries in one launch, runs unchanged, and rotates its output back in one launch.
- **The per-row kernel reads a cache through a reader per format** (`sycl_kv_rows.h`: floats, int8, TurboQuant),
  one for the keys and one for the values, and serves the query heads that share a KV head in one work-group: each
  key and value is read and decoded once for up to four heads instead of once a head, which is what a format whose
  decoding costs arithmetic needs at long context (the float and int8 caches gain or lose a few percent by it).

## What was measured

Qwen3.5-4B (the original checkpoint, f16 compute; 8 attention layers of 4 KV heads of 256 values) on an Intel Arc
Pro B60, in a container, 2026-11-14. The model's other 24 layers keep a fixed-size state, so its cache is small for
its size: 32 KiB a token at f16.

**Memory.** The cache takes exactly the formats' bytes:

| `--kv-dtype` | Cache bytes a token | Against f16 | Cache at 32k tokens | at 128k tokens |
|---|---|---|---|---|
| `auto` (f16) | 32,768 | 1.00x | 1.00 GiB | 4.00 GiB |
| `i8` | 16,640 | 1.97x | 0.51 GiB | 2.03 GiB |
| `i8:turbo4` | 12,544 | 2.61x | 0.38 GiB | 1.53 GiB |
| `turbo4` | 8,448 | 3.88x | 0.26 GiB | 1.03 GiB |
| `turbo3` | 6,400 | 5.12x | 0.20 GiB | 0.78 GiB |
| `turbo2` | 4,352 | 7.53x | 0.13 GiB | 0.53 GiB |

A prefill adds its scratch: one layer's keys and values in view decoded to f16 (0.5 GiB at 128k tokens for this
model), as with int8. The peaks of the whole process, about 8.7 GiB of which is the model and its working buffers, are in the
speed table.

**Accuracy.** Against Hugging Face fp32 on the six parity prompts (5 to 1,691 tokens): the worst cosine of the
last-token logits, and on how many prompts 32 greedy tokens are the reference's. Against the f16 cache on wikitext-2's
test split (`tests/unit/model/kv_format_quality.cpp`): four windows of 8,192 tokens, the next-token distribution at the
last 256 positions of each (1,024 positions, each attending to about 8,000 rounded keys and values).

| `--kv-dtype` | Logits cosine, worst of 6 | Greedy equal, of 6 | Perplexity (change) | Mean KL from f16 | Same top token |
|---|---|---|---|---|---|
| `auto` (f16) | 1.00000 | 6 | 9.655 | 0 | 100 % |
| `i8` | 0.99949 | 6 | 9.657 (+0.02 %) | 0.00008 | 99.6 % |
| `i8:turbo4` | 0.99347 | 5 | 9.638 (-0.18 %) | 0.0020 | 98.0 % |
| `turbo4` | 0.99158 | 5 | 9.683 (+0.29 %) | 0.0037 | 97.4 % |
| `i8:turbo3` | 0.97940 | 3 | 9.528 (-1.32 %) | 0.0095 | 95.5 % |
| `turbo3` | 0.96621 | 3 | 9.595 (-0.63 %) | 0.030 | 93.6 % |
| `i8:turbo2` | 0.89668 | 2 | 9.890 (+2.43 %) | 0.044 | 91.4 % |
| `turbo2` | 0.87628 | 2 | 10.279 (+6.46 %) | 0.092 | 89.4 % |

Over 1,024 positions a perplexity moves by about a percent from the choice of positions alone, so the KL divergence
and the top token are the sharper columns. Over 16 windows of 512 tokens (4,096 positions) the perplexity rises by
0.76 % (turbo4), 1.84 % (turbo3) and 8.07 % (turbo2), and stays where it was with int8 (-0.01 %).

Answers: 300 MMLU-Pro questions read at the option letters, and a fact planted in texts of 8k, 33k and 101k tokens
and asked for (`tests/e2e/needle.py`, six probes):

| `--kv-dtype` | MMLU-Pro right, of 300 | Same answer as f16 | Planted facts found |
|---|---|---|---|
| `auto` (f16) | 144 | | 6 of 6 |
| `i8` | 143 | 295 | 6 of 6 |
| `i8:turbo4` | 144 | 276 | 6 of 6 |
| `turbo4` | 140 | 266 | 6 of 6 |
| `turbo3` | 133 | 233 | 6 of 6 |
| `turbo2` | 115 | 162 | 6 of 6 |

**Speed** (`xpu-serve bench`: a prompt that fills the context but 80 tokens, then 64 tokens generated). Prefill is
within 4 % of f16's at every length (turbo4: 4,738, 2,805 and 944 tok/s at 4k, 32k and 128k tokens against 4,764,
2,826 and 962). Decode, tok/s:

| `--kv-dtype` | at 4k tokens | at 32k | at 128k | Peak GPU memory at 128k |
|---|---|---|---|---|
| `auto` (f16) | 42.1 | 36.3 | 25.7 | 12.99 GiB |
| `i8` | 42.2 | 36.8 | 26.5 | 11.57 GiB |
| `i8:turbo4` | 42.0 | 36.8 | 26.3 | 11.07 GiB |
| `turbo4` | 42.0 | 36.6 | 24.8 | 10.55 GiB |
| `turbo3` | 42.1 | 35.6 | 24.2 | 10.31 GiB |
| `turbo2` | 42.2 | 36.5 | 25.2 | 10.05 GiB |

**A GGUF import, Qwen3.8-27B Q4_0 with its MTP head** (16.3 GiB of weights on the 24 GB card; 64 KiB of cache a
token at f16). The formats apply unchanged, since a cache does not depend on how the weights are stored. An f16 cache
stops at 64k tokens (22.3 GiB at the peak; 96k runs out of memory) and int8 at 128k; with turbo4 the pack read all
262,144 tokens the model takes (23.5 GiB at the peak, prefill 141 tok/s, decode 6.3 tok/s with MTP), and 128k at
20.8 GiB (turbo3: 20.2). At 64k the process peaks at 20.5 (int8), 19.5 (turbo4) and 18.9 GiB (turbo2). Decode with
MTP at 64k: 23.5 tok/s with f16, 22.4 with int8, 22.9 with int8 keys and turbo4 values, 19.7, 17.9 and 19.7 with
turbo4, turbo3 and turbo2. Against its f16 cache over 2,048 positions of 512-token windows: KL 0.00005 (int8), 0.0015
(i8:turbo4), 0.0026 (turbo4), 0.0091 (turbo3), 0.037 (turbo2, +3.8 % perplexity): 0.6 times what the same formats
cost Qwen3.5-4B.

[model_recipes.md](../../model_recipes.md#kv-cache-formats) carries the full tables.

## Findings

- **The formats deliver their sizes**: 3.88, 5.12 and 7.53 times less cache than f16, twice to nearly four times
  less than int8.
- **Accuracy follows the bits, and matches what TurboQuant+ reports**: turbo4 costs a third of a percent of
  perplexity (theirs: +0.23 %), turbo2 six and a half (theirs: +6.48 %). int8 stays the near-lossless step (KL
  0.00008); turbo4 moves the next-token distribution about fifty times as far as int8, turbo3 eight times as far as
  turbo4, turbo2 three times as far again. The most likely token changes on 2.6 %, 6.4 % and 10.6 % of positions.
- **What is lost is close calls, not retrieval.** Every format finds every planted fact up to 101k tokens. On
  MMLU-Pro, where this model is right half the time and often unsure, turbo4 changes one answer in nine (about what a
  Q5_K_M file of the weights does), turbo3 one in four or five, turbo2 nearly half, with 29 fewer right.
- **int8 keys help at every width**: with the keys at int8 the KL divergence falls by 40 to 70 % and turbo4 keeps
  276 MMLU-Pro answers instead of 266, for 1.5 to 2.4 times the cache. On short prompts the values matter as much as
  the keys (the worst logit cosine barely moves with int8 keys); over 8,000 tokens the keys matter more.
- **A quantized cache amplifies what happens upstream.** A coordinate near an edge takes the other code after the
  slightest change in a key, and a code is a whole centroid apart. A tiny decoder whose weights are moved by 5e-4 moves
  its own logits by 0.013 with a float cache, 0.03 with int8 and 0.10 to 0.22 with the TurboQuant formats. This is why
  greedy text diverges early and why the cross-backend test of these formats runs at f32 only.
- **Decode speed is a matter of the kernel, not of the bytes.** The first kernel decoded each code by selects and
  each key once per query head: turbo4 decoded at 16.6 tok/s at 128k tokens, f16 at 24.4. A table for the centroids
  (35.1 against 31.1 tok/s at 32k) and one read of a key for the four heads that share it (24.8 against 22.0 tok/s at
  128k) brought turbo4 to f16's speed. The float and int8 caches gained or lost under 5 % from the shared read: at
  these lengths every format ends within a few percent of the others, so what bounds decode attention now is neither
  the cache's bytes nor its decoding.
- **On a large model the formats buy context.** Qwen3.8-27B's weights leave 7 GiB of a 24 GB card: 64k tokens of
  f16 cache, 128k of int8, and with turbo4 the whole 262,144 the model reads. A verify step of five tokens decodes
  each key for five queries, so with a drafter the TurboQuant formats decode at 0.76 to 0.84 of f16's speed at 64k,
  where int8 keys with turbo4 values keep f16's.
- **Opt-in, with int8 as the first step.** `--kv-dtype i8` is the setting to try first; `i8:turbo4` and `turbo4` when
  the cache has to shrink further, `turbo3` and `turbo2` when memory decides and the answers are checked.

## Not done

- A rotation of 64 for head dims that are not multiples of 128; a format per layer (TurboQuant+ keeps the first and
  last layers' values at 8 bits under turbo2); skipping the values of keys with negligible weight in decode.
- Prefill still decodes one layer's keys and values in view to f16 at a time, so its peak is the compressed cache
  plus one layer's f16 copy.
