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
model), as with int8.

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
and the top token are the sharper columns.

**Speed.** Prefill is within 3 % of f16's at every length (turbo4: 4,750, 2,760 and 946 tok/s at 4k, 32k and 128k
tokens against 4,780, 2,835 and 961). Decode, tok/s, one run each:

| `--kv-dtype` | at 4k tokens | at 32k | at 128k |
|---|---|---|---|
| `auto` (f16) | 42.1 | 36.3 | 25.7 |
| `i8` | 42.2 | 36.8 | 26.5 |
| `turbo4` | 41.8 | 36.4 | 24.8 |
| `turbo2` | | 37.0 | |

[model_recipes.md](../../model_recipes.md#kv-cache-formats) carries the full tables, the long-context retrieval and
MMLU-Pro checks, and a GGUF pack of Qwen3.8-27B.

## Findings

- **The formats deliver their sizes**: 3.88, 5.12 and 7.53 times less cache than f16, twice to nearly four times
  less than int8.
- **Accuracy follows the bits, and matches what TurboQuant+ reports**: turbo4 costs a third of a percent of
  perplexity (theirs: +0.23 %), turbo2 six and a half (theirs: +6.48 %). int8 stays the near-lossless step (KL
  0.00008); turbo4 moves the next-token distribution about fifty times as far as int8, turbo3 eight times as far as
  turbo4, turbo2 three times as far again. The most likely token changes on 2.6 %, 6.4 % and 10.6 % of positions.
- **int8 keys help at every width**: with the keys at int8 the KL divergence halves (turbo4, turbo2) or drops to a
  third (turbo3), for 1.5 to 2.4 times the cache. On short prompts the values matter as much as the keys (the worst
  logit cosine barely moves with int8 keys); over 8,000 tokens the keys matter more.
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
- **Opt-in, with int8 as the first step.** `--kv-dtype i8` is the setting to try first; `i8:turbo4` and `turbo4` when
  the cache has to shrink further, `turbo3` and `turbo2` when memory decides and the answers are checked.

## Not done

- A rotation of 64 for head dims that are not multiples of 128; a format per layer (TurboQuant+ keeps the first and
  last layers' values at 8 bits under turbo2); skipping the values of keys with negligible weight in decode.
- Prefill still decodes one layer's keys and values in view to f16 at a time, so its peak is the compressed cache
  plus one layer's f16 copy.
