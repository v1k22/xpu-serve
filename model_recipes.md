# Model recipes

For every model xpu-serve runs: where it comes from, the command that runs it as the reference (the original
checkpoint, f16 weights, f16 KV cache: what you would measure accuracy against), the command that runs it fast and
small, and what was measured. Start from the reference command. When it does not fit the device or is too slow, step
down: a pack, the model's drafter, a smaller KV cache. The [README](README.md) says what is supported and how the
APIs look; the numbers live here.

## Contents

- [How to read a recipe](#how-to-read-a-recipe)
- [KV cache formats](#kv-cache-formats)
- [Language models](#language-models)
  - [Gemma 4 E2B](#gemma-4-e2b)
  - [Gemma 4 E4B](#gemma-4-e4b)
  - [Gemma 4 26B-A4B](#gemma-4-26b-a4b)
  - [Qwen3.5-4B](#qwen35-4b)
  - [Qwen3.5-9B](#qwen35-9b)
  - [Qwen3.6-35B-A3B](#qwen36-35b-a3b)
  - [Qwen3.8-27B](#qwen38-27b)
  - [LFM3-1.7B](#lfm3-17b)
- [GGUF imports](#gguf-imports)
  - [Importing a GGUF file](#importing-a-gguf-file)
  - [Qwen3.5-4B GGUF](#qwen35-4b-gguf)
  - [Gemma 4 E2B GGUF](#gemma-4-e2b-gguf)
  - [Qwen3.8-27B GGUF](#qwen38-27b-gguf)
- [Drafters](#drafters)
  - [Speculative decoding](#speculative-decoding)
  - [Gemma 4 assistants](#gemma-4-assistants)
  - [DFlash 2 and DSpine drafters for Qwen3.8-27B](#dflash-2-and-dspine-drafters-for-qwen38-27b)
- [Decision models](#decision-models)
  - [The System One request](#the-system-one-request)
  - [Clef-Flash](#clef-flash)
  - [Clef](#clef)
  - [pplx-decider-flash-v1.2](#pplx-decider-flash-v12)
  - [OpenJev](#openjev)
  - [d1-2B](#d1-2b)
  - [Laya](#laya)
  - [Julia-1](#julia-1)
  - [Decision models: latency by prompt length, and what a pack changes](#decision-models-latency-by-prompt-length-and-what-a-pack-changes)
- [Embedding models](#embedding-models)
  - [all-MiniLM-L6-v2](#all-minilm-l6-v2)
  - [bge-small-en-v1.5](#bge-small-en-v15)
  - [nomic-embed-text-v1.5](#nomic-embed-text-v15)
- [Robot policy](#robot-policy)
  - [π0.5](#π05)

## How to read a recipe

- `~/models/<name>` is the model's Hugging Face repository downloaded as it is (for example
  `huggingface-cli download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B`). `xpu-serve` is the installed binary
  ([README, Build](README.md#build)).
- **Reference** is the original checkpoint with nothing lossy: weights and KV cache in f16, no drafter. It needs
  about two bytes of GPU memory per weight; where a model is larger than a 24 GB card, the recipe says so.
- **Optimized** is the setting the measurements below recommend, in this order of steps:
  1. **A pack.** `xpu-serve convert` writes the checkpoint with its matmul weights quantized. On the Arc Pro B60, q8
     decodes 1.5 to 1.7 times faster than f16 with no measurable quality loss; q4 decodes about twice as fast and
     keeps MMLU-Pro scores, with some loss of logit fidelity on small models (Qwen3.5-4B logits cosine 0.90 to 0.96
     at groups of 32, 0.92 to 0.96 with AWQ). Numbers: [docs/decisions/0009](docs/decisions/0009-quantized-packs.md).
     An FP8 release converts like the original and gives the same pack
     ([docs/decisions/0014](docs/decisions/0014-fp8-checkpoints.md)).
  2. **The model's drafter** (`--spec`): the output stays the model's own, at a higher speed
     ([Speculative decoding](#speculative-decoding)).
  3. **A smaller KV cache** (`--kv-dtype`): longer contexts in the same memory
     ([KV cache formats](#kv-cache-formats)).
- A [GGUF file](#importing-a-gguf-file) of the model is another way down: the file's own quantization, at the file's
  size.
- `--ctx N` sets the context length (prompt and output, 8192 by default); the KV cache grows with what a request
  uses, up to it.
- Numbers are from an Intel Arc Pro B60 (a 24 GB discrete GPU) unless a line names the Arc B390 (an integrated GPU),
  at f16 compute. "MMLU-Pro (25 questions)" is a small sample (one answer is 4 points); "of 300" counts are the
  sharper measure.

## KV cache formats

`--kv-dtype` (on `serve`, `run` and `bench`) chooses how the KV cache is stored. Every format but `auto` is lossy and
opt-in.

| `--kv-dtype` | What is stored | Bits a value |
|---|---|---|
| `auto` (default) | f16 | 16 |
| `i8` | int8 codes and one scale per token and KV head | 8 |
| `turbo4` | TurboQuant: rows rotated, a 4-bit code a value, an f16 norm per 128 values | 4.125 |
| `turbo3` | TurboQuant with 3-bit codes | 3.125 |
| `turbo2` | TurboQuant with 2-bit codes | 2.125 |
| `<keys>:<values>` | keys and values apart; supported with `i8` keys, e.g. `i8:turbo4` | between the two |

- The TurboQuant formats need a head_dim that is a multiple of 128 (Qwen3.5-4B's 256, Gemma 4 E2B's 256 and 512); a
  model with another is refused by name.
- The order from the reference down: `auto`, `i8`, `i8:turbo4`, `turbo4`, `turbo3`, `turbo2`. Each step is smaller
  and further from the f16 answers. **Start with `i8`**: it halves the cache and changes next to nothing. Go further
  down only when the cache still does not fit, and check the answers you care about.
- The formats work the same over an original checkpoint, a pack and a GGUF import: the cache is the same whatever
  the weights are stored as.
- Designs and details: [docs/decisions/0010](docs/decisions/0010-int8-kv-cache.md) (int8) and
  [0020](docs/decisions/0020-turboquant-kv-cache.md) (TurboQuant).

### Measured on Qwen3.5-4B

The original checkpoint at f16 on an Arc Pro B60. Its cache is 32 KiB a token at f16 (8 attention layers of 4 KV
heads; the other 24 layers keep a fixed-size state).

<!-- KV-Q35-TABLES -->
Memory. The cache takes exactly the formats' bytes. Peak GPU memory is the whole process as the driver reports it
during `xpu-serve bench`: the model, the cache, and a prefill's scratch (for a quantized cache, one layer's keys and
values decoded to f16).

| `--kv-dtype` | Cache bytes a token | Against f16 | Cache at 32k tokens | Cache at 128k tokens | Peak GPU memory at 4k | Peak GPU memory at 32k | Peak GPU memory at 128k |
|---|---|---|---|---|---|---|---|
| `auto` (f16) | 32,768 | 1.00x | 1.00 GiB | 4.00 GiB | 8.84 GiB | 9.99 GiB | 12.99 GiB |
| `i8` | 16,640 | 1.97x | 0.51 GiB | 2.03 GiB | 8.79 GiB | 9.64 GiB | 11.57 GiB |
| `i8:turbo4` | 12,544 | 2.61x | 0.38 GiB | 1.53 GiB | 8.77 GiB | 9.52 GiB | 11.07 GiB |
| `turbo4` | 8,448 | 3.88x | 0.26 GiB | 1.03 GiB | 8.77 GiB | 9.42 GiB | 10.55 GiB |
| `turbo3` | 6,400 | 5.12x | 0.20 GiB | 0.78 GiB | 8.76 GiB | 9.35 GiB | 10.31 GiB |
| `turbo2` | 4,352 | 7.53x | 0.13 GiB | 0.53 GiB | 8.75 GiB | 9.29 GiB | 10.05 GiB |

Speed (`xpu-serve bench`: a prompt that fills the context but 80 tokens, then 64 tokens generated):

| `--kv-dtype` | Decode at 4k tokens | Decode at 32k tokens | Decode at 128k tokens | Prefill at 4k | Prefill at 32k | Prefill at 128k |
|---|---|---|---|---|---|---|
| `auto` (f16) | 42.1 tok/s | 36.3 tok/s | 25.7 tok/s | 4,764 tok/s | 2,826 tok/s | 962 tok/s |
| `i8` | 42.2 tok/s | 36.8 tok/s | 26.5 tok/s | 4,742 tok/s | 2,719 tok/s | 940 tok/s |
| `i8:turbo4` | 42.0 tok/s | 36.8 tok/s | 26.3 tok/s | 4,753 tok/s | 2,766 tok/s | 942 tok/s |
| `turbo4` | 42.0 tok/s | 36.6 tok/s | 24.8 tok/s | 4,738 tok/s | 2,805 tok/s | 944 tok/s |
| `turbo3` | 42.1 tok/s | 35.6 tok/s | 24.2 tok/s | 4,749 tok/s | 2,784 tok/s | 946 tok/s |
| `turbo2` | 42.2 tok/s | 36.5 tok/s | 25.2 tok/s | 4,739 tok/s | 2,774 tok/s | 947 tok/s |

Accuracy, in the next-token distribution. Against the f16 cache on wikitext-2's test split
(`tests/unit/model/kv_format_quality.cpp`): the distribution at the last half of 16 windows of 512 tokens (4,096
positions) and at the last 256 positions of 4 windows of 8,192 tokens (1,024 positions, where a perplexity moves by a
percent from the choice of positions alone), every position attending to keys and values the format rounded:

| `--kv-dtype` | 512-token windows: perplexity (change) | mean KL from f16 | same top token | 8,192-token windows: perplexity (change) | mean KL from f16 | same top token |
|---|---|---|---|---|---|---|
| `auto` (f16) | 10.747 | 0 | 100 % | 9.655 | 0 | 100 % |
| `i8` | 10.746 (-0.01 %) | 0.00015 | 99.4 % | 9.657 (+0.02 %) | 0.00008 | 99.6 % |
| `i8:turbo4` | 10.797 (+0.46 %) | 0.0025 | 97.7 % | 9.638 (-0.18 %) | 0.0020 | 98.0 % |
| `turbo4` | 10.829 (+0.76 %) | 0.0044 | 97.0 % | 9.683 (+0.29 %) | 0.0037 | 97.4 % |
| `i8:turbo3` | 10.850 (+0.96 %) | 0.0090 | 95.5 % | 9.528 (-1.32 %) | 0.0095 | 95.5 % |
| `turbo3` | 10.944 (+1.84 %) | 0.015 | 94.2 % | 9.595 (-0.63 %) | 0.030 | 93.6 % |
| `i8:turbo2` | 11.313 (+5.27 %) | 0.037 | 90.6 % | 9.890 (+2.43 %) | 0.044 | 91.4 % |
| `turbo2` | 11.614 (+8.07 %) | 0.063 | 87.5 % | 10.279 (+6.46 %) | 0.092 | 89.4 % |

Accuracy, in answers. Against Hugging Face fp32 on six prompts of 5 to 1,691 tokens (the logits after the prompt, and
32 greedy tokens); 300 MMLU-Pro questions read at the option letters (`serve --readout letters`); and
`tests/e2e/needle.py`, a fact planted at a quarter and at three quarters of texts of 8k, 33k and 101k tokens, then
asked for:

| `--kv-dtype` | Logits cosine against Hugging Face fp32, worst of 6 prompts | Greedy text equal, of 6 | MMLU-Pro right, of 300 | Same answer as f16, of 300 | Planted facts found |
|---|---|---|---|---|---|
| `auto` (f16) | 1.00000 | 6 | 144 |  | 6 of 6 |
| `i8` | 0.99949 | 6 | 143 | 295 | 6 of 6 |
| `i8:turbo4` | 0.99347 | 5 | 144 | 276 | 6 of 6 |
| `turbo4` | 0.99158 | 5 | 140 | 266 | 6 of 6 |
| `i8:turbo3` | 0.97940 | 3 |  |  |  |
| `turbo3` | 0.96621 | 3 | 133 | 233 | 6 of 6 |
| `i8:turbo2` | 0.89668 | 2 |  |  |  |
| `turbo2` | 0.87628 | 2 | 115 | 162 | 6 of 6 |
<!-- /KV-Q35-TABLES -->

What the numbers say:

- **Memory.** The cache shrinks by exactly the format's factor: 3.88, 5.12 and 7.53 times against f16 for `turbo4`,
  `turbo3` and `turbo2`, twice to nearly four times against `i8`. At 128k tokens the whole process peaks at 13.0 GiB
  with f16, 11.6 with `i8`, 10.6 with `turbo4` and 10.1 with `turbo2` (about 8.7 GiB of each is the model and its
  working buffers).
- **Speed.** Prefill stays within 4 % of f16's and decode within 6 %, at every length and for every format.
- **`i8` is next to lossless**: 295 of 300 MMLU-Pro answers stay, and the next-token distribution hardly moves.
- **The TurboQuant formats cost accuracy by their bits.** `turbo4` raises perplexity by 0.3 to 0.8 %, changes the
  most likely token at 3 % of positions and keeps 266 of 300 MMLU-Pro answers: about what a Q5_K_M file of the
  weights costs this model (271 of 300, [Qwen3.5-4B GGUF](#qwen35-4b-gguf)). `turbo3` (+1.8 % at 512 tokens, 233 of
  300) is near a Q4_0 file's 227. `turbo2` (+6.5 to 8 %, 162 of 300, 29 fewer right answers) is between a Q3 file and
  a 2-bit one. Greedy text diverges early with any of the three. The perplexity costs are the ones TurboQuant+
  reports for its formats on other models (+0.23 %, +1.06 % and +6.48 %), a little higher on this small Qwen.
- **Long-context retrieval survives all of them**: every format finds the planted fact in all six texts, up to 101k
  tokens. What the low-bit formats lose is close calls, not the ability to find what is in the context.
- **int8 keys buy accuracy back**: with the keys at `i8`, `i8:turbo4` keeps 276 answers against `turbo4`'s 266, and
  the divergence from f16 falls by 40 to 70 % at every width, for 1.5 to 2.4 times the cache.
- **So**: `i8` first. `i8:turbo4` or `turbo4` when the cache must shrink further and a small loss is acceptable;
  `turbo3` and `turbo2` when memory decides, with the answers checked on the task.

<!-- KV-Q38-TABLES -->

## Language models

All of them are served behind the OpenAI API ([README](README.md#chat-completions)). Where a model thinks by
default, `"chat_template_kwargs": {"enable_thinking": false}` in a request turns that off.

### Gemma 4 E2B

`google/gemma-4-E2B-it` (`gemma4`). Text and images. Drafter: `google/gemma-4-E2B-it-assistant`.

Reference:

```bash
xpu-serve serve ~/models/gemma-4-E2B-it --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/gemma-4-E2B-it --out ~/models/gemma-4-E2B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-E2B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 1 --drafter ~/models/gemma-4-E2B-it-assistant
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy ([KV cache formats](#kv-cache-formats)).
- `--quant q8` is the pack with no measurable loss. AWQ (`--method awq`) does not help Gemma 4 yet
  ([docs/decisions/0009](docs/decisions/0009-quantized-packs.md)).
- With an int8 KV cache Gemma 4 E2B changes a few answers it is unsure of
  ([docs/decisions/0010](docs/decisions/0010-int8-kv-cache.md)); leave `--kv-dtype` out where that matters.
- Images work from the original and from a pack (a pack keeps the vision tower as it is).
- Its GGUF files: [Gemma 4 E2B GGUF](#gemma-4-e2b-gguf).

Measured:

| | f16 | q8 | q4 | q4 with its assistant (k = 1) |
|---|---|---|---|---|
| Decode | 61.5 tok/s | 92 tok/s | 140 tok/s | 174 tok/s |

Checked against Hugging Face fp32 logits, greedy text and image features, and its assistant's drafts.

### Gemma 4 E4B

`google/gemma-4-E4B-it` (`gemma4`). Text and images. Drafter: `google/gemma-4-E4B-it-assistant`.

Reference:

```bash
xpu-serve serve ~/models/gemma-4-E4B-it --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/gemma-4-E4B-it --out ~/models/gemma-4-E4B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-E4B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 2 --drafter ~/models/gemma-4-E4B-it-assistant
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy ([KV cache formats](#kv-cache-formats)).
- The notes of [Gemma 4 E2B](#gemma-4-e2b) apply: the same model code.

Measured: at q4, decode 84 tok/s, and 126 tok/s with its assistant (k = 2).

### Gemma 4 26B-A4B

`google/gemma-4-26B-A4B-it` (`gemma4`, a mixture of experts). Text only for now. Drafter:
`google/gemma-4-26B-A4B-it-assistant`.

Reference (its f16 weights do not fit a 24 GB card; there the q4 pack below is the way to run it):

```bash
xpu-serve serve ~/models/gemma-4-26B-A4B-it --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/gemma-4-26B-A4B-it --out ~/models/gemma-4-26B-A4B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-26B-A4B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 2 --drafter ~/models/gemma-4-26B-A4B-it-assistant
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- A GGUF file of it (a Gemma 4 with experts) is refused by name.

Measured: at q4, decode 67 tok/s, and 91 tok/s with its assistant (k = 2).

Checked against Hugging Face on a tiny checkpoint of its layout (experts beside the MLP, K = V global layers).

### Qwen3.5-4B

`Qwen/Qwen3.5-4B` (`qwen3_5`). Text; its MTP head is the drafter, inside the checkpoint and its packs.

Reference:

```bash
xpu-serve serve ~/models/Qwen3.5-4B --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
xpu-serve serve ~/models/Qwen3.5-4B-q4 --device GPU --spec mtp --spec-k 2 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy ([KV cache formats](#kv-cache-formats), measured on this model).
- `--quant q8` is the pack with no measurable loss. `--method awq` calibrates q4 on the model itself (on the GPU,
  about a minute), which helps Qwen3.5; without it q4 rounds to nearest.
- Its GGUF files: [Qwen3.5-4B GGUF](#qwen35-4b-gguf).

Measured:

| | f16 | q8 | q4 | q4 with MTP (k = 2) |
|---|---|---|---|---|
| Decode | 43 tok/s | 77 tok/s | 101 tok/s | 149 tok/s |

Prefill 5,000 tok/s. Over six chat prompts the MTP head takes q4 from 104 to 149 tok/s. With an int8 KV cache it
keeps its MMLU-Pro score.

Checked against Hugging Face fp32 logits and greedy text.

### Qwen3.5-9B

`Qwen/Qwen3.5-9B` (`qwen3_5`). Text; its MTP head is the drafter.

Reference:

```bash
xpu-serve serve ~/models/Qwen3.5-9B --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/Qwen3.5-9B --out ~/models/Qwen3.5-9B-q4 --quant q4
xpu-serve serve ~/models/Qwen3.5-9B-q4 --device GPU --spec mtp --spec-k 4 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- `--quant q8` is the pack with no measurable loss.

Measured:

| | f16 | q8 | q4 | q4 with MTP (k = 4) |
|---|---|---|---|---|
| Decode | 21 tok/s | 38 tok/s | 62 tok/s | 108 tok/s |

Prefill 3,200 tok/s. MMLU-Pro (25 questions): 72 %.

Checked against Hugging Face fp32 logits, greedy text and its MTP head's drafts.

### Qwen3.6-35B-A3B

`Qwen/Qwen3.6-35B-A3B`, or its FP8 release `Qwen/Qwen3.6-35B-A3B-FP8` (half the download) (`qwen3_5_moe`, a mixture
of experts). Text; its MTP head is the drafter.

Reference (its f16 weights do not fit a 24 GB card; an FP8 checkpoint also runs as f16 on the device):

```bash
xpu-serve serve ~/models/Qwen3.6-35B-A3B --device GPU
```

Optimized (a q4 pack fits one 24 GB card):

```bash
xpu-serve convert ~/models/Qwen3.6-35B-A3B-FP8 --out ~/models/Qwen3.6-35B-A3B-q4 --quant q4
xpu-serve serve ~/models/Qwen3.6-35B-A3B-q4 --device GPU --spec mtp --spec-k 3 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- A GGUF file of it (`qwen35moe`) is refused by name.

Measured: at q4, decode 62 tok/s, and 94 tok/s with MTP (k = 3); prefill 1,550 tok/s.

Checked against Hugging Face on a tiny checkpoint; its q4 pack against Hugging Face fp32 logits of the checkpoint
(cosine 0.92 to 0.98); MMLU-Pro (25 questions): 80 to 84 %.

### Qwen3.8-27B

`Qwen/Qwen3.8-27B`, or its FP8 release `Qwen/Qwen3.8-27B-FP8` (half the download) (`qwen3_5`). Text. Drafters: its
MTP head (inside the checkpoint and its packs), and the block drafters `z-lab/Qwen3.8-27B-DFlash2` and
`tencent/Qwen3.8-27B-DSpine-b16`.

Reference (its f16 weights do not fit a 24 GB card; an FP8 checkpoint also runs as f16 on the device):

```bash
xpu-serve serve ~/models/Qwen3.8-27B --device GPU
```

Optimized (a q4 pack fits one 24 GB card):

```bash
xpu-serve convert ~/models/Qwen3.8-27B-FP8 --out ~/models/Qwen3.8-27B-q4 --quant q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec mtp --spec-k 4 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- `--group 128` on `convert` reads 15% fewer bytes: the pack decodes 13% faster on both the Arc Pro B60 and the Arc
  B390, and scores 80% on 25 MMLU-Pro questions against 76% at the default group of 32.
- On the Arc B390, `--spec-k 5`.
- Faster still on prose: a DFlash 2 drafter
  ([DFlash 2 and DSpine drafters](#dflash-2-and-dspine-drafters-for-qwen38-27b)).
- Its GGUF files, 6 to 23 GB: [Qwen3.8-27B GGUF](#qwen38-27b-gguf).

Measured, at q4:

| | Decode | With MTP | With a DFlash 2 drafter | With a DSpine drafter | Prefill |
|---|---|---|---|---|---|
| Arc Pro B60 | 22 tok/s | 50 tok/s (k = 4) | 61 tok/s (k = 7) | 51 tok/s (k = 7) | 830 tok/s |
| Arc Pro B60, `--group 128` | | 53 tok/s | 62 tok/s | | |
| Arc B390 | 6.9 tok/s | 18 tok/s (k = 5) | 20 tok/s (k = 7) | 17.5 tok/s (k = 7) | 460 tok/s |
| Arc B390, `--group 128` | | 20 tok/s | 23 tok/s | | |

MMLU-Pro (25 questions): 76 %.

Checked against Hugging Face on a tiny checkpoint (its head ratios and RoPE, its MTP head), and the DFlash, DFlash 2
and DSpine drafters against their PyTorch code.

### LFM3-1.7B

`LiquidAI/LFM3-1.7B` (`lfm3`; its tokenizer and chat template come under the LFM Open License v1.0, which limits
commercial use). Text; no drafter exists for it.

Reference:

```bash
xpu-serve serve ~/models/LFM3-1.7B --device GPU
```

Optimized (its q8 pack: at q4 its reasoning runs long more often):

```bash
xpu-serve convert ~/models/LFM3-1.7B --out ~/models/LFM3-1.7B-q8 --quant q8
xpu-serve serve ~/models/LFM3-1.7B-q8 --device GPU --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- It thinks by default; `"chat_template_kwargs": {"enable_thinking": false}` in a request turns that off.

Measured:

| | f16 | q8 | q4 |
|---|---|---|---|
| Decode | 99 tok/s | 159 tok/s | 206 tok/s |

Prefill 12,000 tok/s. Run to 64k tokens of context. MMLU-Pro (25 questions): 60 %.

Checked against Hugging Face fp32 logits and greedy text (the checkpoint's own modeling code).

## GGUF imports

### Importing a GGUF file

`convert` imports a GGUF file (llama.cpp's format) into a pack in a few seconds. Each quantized tensor stays exactly
as the file stores it (nothing is decoded or quantized again), so the model has the file's weights and takes the
file's size in GPU memory. The pack is then served like any model, with the same `--spec` and `--kv-dtype` options.

```bash
xpu-serve convert <file.gguf> <pack-dir>
xpu-serve convert <file.gguf> <pack-dir> --mtp <mtp.gguf>     # a Qwen3.5-family file that comes without its MTP head
xpu-serve serve <pack-dir> --device GPU
```

What is supported, and what a pack takes from the file: [README, From a GGUF file](README.md#from-a-gguf-file).
**[docs/gguf_models.md](docs/gguf_models.md) has every file's numbers**, what each file holds, the checks behind them
and the kernel rates; the sections below carry a few of them. Measured with every file of
[unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) (22) and
[unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) (21, and its drafter files), and the
20 files of [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) that fit the card. "Same as
f16" is how many of 300 MMLU-Pro questions get the answer the original checkpoint gives at f16.

What the measurements say:

- **Memory is the file's.** A Qwen file takes its size on the GPU (2.74 GB is 2.55 GiB); a Gemma 4 E2B file under
  half of it, because its per-layer embedding table is read on the host.
- **The import is exact.** Both BF16 files give the original checkpoint's answer on all 300 questions.
- **The kernels keep the file's values.** Read by the block kernels or decoded to f16 on upload
  (`--backend-opt block=f16`), a file gives the same answer on 298 to 300 of 300.
- **Accuracy follows the bits**, and is the file's: from 5 bits up a file answers like the original on about 9
  questions in 10, the 4-bit K-quants on about 4 in 5 (Qwen3.8-27B's on 9 in 10), the 2-bit files on fewer than half
  (Qwen3.8-27B's on about 3 in 5).
- **Speed.** Files decode at 1.3 to 2.3 times f16's speed. No format is bound by memory at these sizes, so a smaller
  file is not a faster one; the IQ4 formats are the slowest per weight.
- **Drafters.** A verify of several rows over a block format runs on the matrix engines (from 2 rows), so MTP pays
  over any file: 1.65 to 2.8 times on Qwen3.8-27B's, with the greedy text unchanged.
- **Tool calls and chat templates.** A pack renders the GGUF's own template. On the 23 tool-calling cases of
  `tests/e2e/tool_calls.py` (one turn and several, graded by a stronger model against `tool_calls_rubric.md`), the
  original checkpoint, its BF16 GGUF and its Q4_K_M GGUF score alike: Qwen3.5-4B 23 of 23 well-formed and right,
  Gemma 4 E2B 23 well-formed and 22 right, and the BF16 GGUF makes the original's calls in every case.

The decision models' files are in their own sections: [Clef-Flash](#clef-flash), [Clef](#clef), [Laya](#laya),
[Julia-1](#julia-1).

### Qwen3.5-4B GGUF

[unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), all 22 files (2 to 16 bits a weight).

Reference (the BF16 file is the original checkpoint):

```bash
xpu-serve convert ~/models/Qwen3.5-4B-BF16.gguf ~/models/Qwen3.5-4B-BF16
xpu-serve serve ~/models/Qwen3.5-4B-BF16 --device GPU
```

Optimized:

```bash
xpu-serve convert ~/models/Qwen3.5-4B-Q4_K_M.gguf ~/models/Qwen3.5-4B-Q4_K_M
xpu-serve serve ~/models/Qwen3.5-4B-Q4_K_M --device GPU --kv-dtype i8
```

- Less VRAM: a smaller file, and `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2`
  ([KV cache formats](#kv-cache-formats)).
- A file that holds the model's MTP head drafts with it: add `--spec mtp --spec-k 2`.
- A file is usually a mix of formats: `Q4_K_M` of Qwen3.5-4B is 57 % Q4_K, 24 % Q6_K and 18 % Q5_K, and a
  "dynamic" (`UD-`) file mixes up to fifteen.

Measured (decode 58 to 99 tok/s over all its files):

| Variant | File | GPU memory | Decode | Same as f16 |
|---|---|---|---|---|
| BF16 | 8.42 GB | 7.83 GiB | 43 tok/s | 300 |
| Q8_0 | 4.48 GB | 4.16 GiB | 74 tok/s | 294 |
| Q6_K | 3.53 GB | 3.27 GiB | 81 tok/s | 269 |
| Q5_K_M | 3.14 GB | 2.92 GiB | 84 tok/s | 271 |
| Q4_K_M | 2.74 GB | 2.54 GiB | 90 tok/s | 251 |
| Q4_0 | 2.58 GB | 2.58 GiB | 99 tok/s | 227 |
| Q3_K_M | 2.29 GB | 2.12 GiB | 90 tok/s | 198 |
| UD-IQ2_M | 1.76 GB | 1.63 GiB | 85 tok/s | 90 |
| *original, f16* | | 7.83 GiB | 44 tok/s | 300 |
| *our q4 pack* | | 2.49 GiB | 109 tok/s | 194 |

### Gemma 4 E2B GGUF

[unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF), all 21 files, and its drafter
files (`MTP/mtp-*.gguf`, the assistant). The text model only: a GGUF file holds no vision or audio tower, so images
need the original checkpoint. Gemma 4 E4B's files import the same way.

Reference (the BF16 file is the original checkpoint's text model):

```bash
xpu-serve convert ~/models/gemma-4-E2B-it-BF16.gguf ~/models/gemma-4-E2B-it-BF16
xpu-serve serve ~/models/gemma-4-E2B-it-BF16 --device GPU
```

Optimized, with its drafter:

```bash
xpu-serve convert ~/models/gemma-4-E2B-it-Q4_K_M.gguf ~/models/gemma-4-E2B-it-Q4_K_M
xpu-serve convert ~/models/mtp-gemma-4-E2B-it-Q8_0.gguf ~/models/gemma-4-E2B-it-mtp
xpu-serve serve ~/models/gemma-4-E2B-it-Q4_K_M --device GPU --kv-dtype i8 \
    --spec mtp --drafter ~/models/gemma-4-E2B-it-mtp
```

- Less VRAM: a smaller file, and `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2`
  ([KV cache formats](#kv-cache-formats)).
- The imported assistant scores the whole vocabulary, as llama.cpp does (the file has no centroid head), so its
  drafts can differ from the original assistant's; the output is the model's either way.

Measured:

| Variant | File | GPU memory | Decode | Same as f16 |
|---|---|---|---|---|
| BF16 | 9.31 GB | 4.25 GiB | 61 tok/s | 300 |
| Q8_0 | 5.05 GB | 2.29 GiB | 91 tok/s | 292 |
| Q6_K | 4.50 GB | 1.79 GiB | 109 tok/s | 276 |
| Q5_K_M | 3.36 GB | 1.54 GiB | 110 tok/s | 261 |
| Q4_K_M | 3.11 GB | 1.31 GiB | 120 tok/s | 237 |
| Q4_0 | 3.04 GB | 1.36 GiB | 127 tok/s | 195 |
| Q3_K_M | 2.54 GB | 1.06 GiB | 116 tok/s | 199 |
| UD-IQ2_M | 2.29 GB | 0.83 GiB | 114 tok/s | 122 |
| *original, f16* | | 4.25 GiB | 61 tok/s | 300 |
| *our q4 pack* | | 1.43 GiB | 121 tok/s | 227 |

With the drafter from its `MTP/` files the text is the same, at 230 to 270 tok/s on the Q4_0 file (125 without) and
200 to 235 on Q4_K_M (117), from session to session.

### Qwen3.8-27B GGUF

[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF): the 20 files that fit a 24 GB card (1.8
to 6.8 bits a weight, 6 to 23 GB). The original does not fit one card, so a file is the closest to the reference
that does: the larger the file, the closer.

Closest to the reference on one card:

```bash
xpu-serve convert ~/models/Qwen3.8-27B-UD-Q6_K_M.gguf ~/models/Qwen3.8-27B-UD-Q6_K_M
xpu-serve serve ~/models/Qwen3.8-27B-UD-Q6_K_M --device GPU
```

Optimized (the fastest file with its MTP head):

```bash
xpu-serve convert ~/models/Qwen3.8-27B-Q4_0.gguf ~/models/Qwen3.8-27B-Q4_0
xpu-serve serve ~/models/Qwen3.8-27B-Q4_0 --device GPU --spec mtp --spec-k 4 --kv-dtype i8
```

The smallest files come without the MTP head; `--mtp` takes it from the head's own file (`MTP/mtp-*.gguf`), and the
pack is the one a file with the head gives:

```bash
xpu-serve convert ~/models/Qwen3.8-27B-UD-IQ2_XXS.gguf ~/models/Qwen3.8-27B-UD-IQ2_XXS \
    --mtp ~/models/mtp-Qwen3.8-27B-Q4_0.gguf
xpu-serve serve ~/models/Qwen3.8-27B-UD-IQ2_XXS --device GPU --spec mtp
```

- Less VRAM: a smaller file, and `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2`
  ([KV cache formats](#kv-cache-formats)).

Measured. "Same as the reference" counts the answers the original gives in float32 on the CPU (190 of 300 of them
right). MTP is greedy decode over six chat prompts with the file's MTP head at the best of 1 to 5 drafts a round.
Over all its files: 15 to 21 tok/s decode, 25 to 47 with MTP.

| Variant | File | GPU memory | Decode | With MTP | Same as the reference | MMLU-Pro right (300) |
|---|---|---|---|---|---|---|
| UD-Q6_K_M | 23.09 GB | 21.16 GiB | 16 tok/s | 33 tok/s | 293 | 193 |
| UD-Q5_K_XL | 20.98 GB | 19.20 GiB | 17 tok/s | 33 tok/s | 288 | 193 |
| UD-Q4_K_XL | 17.66 GB | 16.11 GiB | 16 tok/s | 38 tok/s | 280 | 187 |
| Q4_0 | 16.06 GB | 16.01 GiB | 21 tok/s | 47 tok/s | 251 | 180 |
| UD-IQ4_XS | 14.35 GB | 13.03 GiB | 15 tok/s | 41 tok/s | 259 | 187 |
| UD-IQ3_XXS | 10.93 GB | 9.84 GiB | 18 tok/s | 37 tok/s | 236 | 182 |
| UD-Q2_K_XL | 9.95 GB | 8.93 GiB | 18 tok/s | 39 tok/s | 219 | 171 |
| UD-IQ2_XXS (head from `--mtp`) | 7.21 GB | 6.70 GiB | 17 tok/s | 32 tok/s | 174 | 154 |
| *our q4 pack* | | 16.21 GiB | 22 tok/s | 51 tok/s | 245 | 185 |

On an Arc B390 the Q4_0 file decodes at 6.4 tok/s and at 17.4 with MTP (k = 6, 2.7 times; 17.2 at k = 5): a verify
of up to 8 rows costs about one decode step there.

## Drafters

### Speculative decoding

`--spec` (on `run`, `serve` and `bench`) drafts with the model's drafter and verifies the drafts in one step; greedy
output stays the model's own and sampling keeps its distribution (rejection sampling).

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

- A drafter checkpoint is served as it is, or from a pack of its own, which is smaller and drafts faster:
  `convert <drafter> --target <model> --out <dir>`.
- `--spec-k` is 3 by default and 7 for a block drafter, which proposes its whole block in one pass
  ([docs/design-spec-decode.md](docs/design-spec-decode.md)); the table has the value to pass per model.
- An MTP head picks each draft among 64 tokens that a one-bit sketch of the output layer shortlists
  (`--draft-shortlist`, 0 to read the whole layer).
- `bench --prompts file` measures over real prompts, since acceptance depends on the text, and prints where a round's
  time goes.

### Gemma 4 assistants

`google/gemma-4-E2B-it-assistant`, `google/gemma-4-E4B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant`: each
drafts for its own model, from its checkpoint as it is.

```bash
xpu-serve run ~/models/gemma-4-E2B-it --device GPU --spec mtp --spec-k 1 --drafter ~/models/gemma-4-E2B-it-assistant
```

`--spec-k`: 1 for E2B, 2 for E4B and 26B-A4B ([the table above](#speculative-decoding)).

### DFlash 2 and DSpine drafters for Qwen3.8-27B

`z-lab/Qwen3.8-27B-DFlash2` (blocks of 8) and `tencent/Qwen3.8-27B-DSpine-b16` (blocks of 16). A drafter checkpoint is
served as it is, or from a pack of its own, which is smaller and drafts faster:

```bash
xpu-serve convert ~/models/Qwen3.8-27B-DFlash2 --target ~/models/Qwen3.8-27B-q4 --out ~/models/Qwen3.8-27B-DFlash2-q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dflash --drafter ~/models/Qwen3.8-27B-DFlash2-q4

xpu-serve convert ~/models/Qwen3.8-27B-DSpine-b16 --target ~/models/Qwen3.8-27B-q4 --out ~/models/Qwen3.8-27B-DSpine-q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dspine --drafter ~/models/Qwen3.8-27B-DSpine-q4
```

Measured in [Qwen3.8-27B](#qwen38-27b): 22 → 61 tok/s with DFlash 2 and 22 → 51 with DSpine on the Arc Pro B60, at the
default of 7 drafts a round.

## Decision models

A decision model answers typed questions about a state instead of writing text. `serve` puts one behind the System
One API ([README](README.md#decisions-the-system-one-api) has the schema's rules and errors). Every model below takes
the request of the next section; where a model wants something else of it, its section says so.

### The System One request

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

(Clef-Flash's answer, rounded here to three digits; it took 117 ms on an Arc Pro B60.) Every model answers in this
shape: a `choice` with the chosen option and each option's probability, a `noul` with the probability of yes, a
`score` with the expected level and each level's probability.

Without a server, `run` answers one request file and `bench` times one (its own sample by default, `--state-tokens N`
to grow the state):

```bash
xpu-serve run   ~/models/clef-flash --device GPU --request request.json
xpu-serve bench ~/models/clef-flash --device GPU --ctx 16384 --state-tokens 4096 --runs 7
python3 tests/e2e/systemone_eval.py --url http://127.0.0.1:8080      # 25 decisions: answers, API shape, latency
```

The per-decision times below are medians over 25 test decisions, one question a request.

### Clef-Flash

`Cloudflare/clef-flash` (9B: a Qwen3.5 decoder with a joint schema head). Every question of a request is answered in
one pass; a transformer head scores the options. Questions sent together can move each other's answers.

Reference:

```bash
xpu-serve serve ~/models/clef-flash --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "clef-flash", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory (f16 where it fits: on a discrete GPU a pack is smaller but slower):

```bash
xpu-serve convert ~/models/clef-flash --out ~/models/clef-flash-q8 --quant q8
xpu-serve serve ~/models/clef-flash-q8 --device GPU --ctx 16384
```

Or one of its GGUF files, [ggml-org/Clef-Flash-GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF) (BF16, Q8_0,
Q4_K_M):

```bash
xpu-serve convert ~/models/Clef-Flash-Q8_0.gguf ~/models/Clef-Flash-Q8_0
xpu-serve serve ~/models/Clef-Flash-Q8_0 --device GPU --ctx 16384
```

Measured:

| | f16 | q8 pack | q4 pack | GGUF BF16 | GGUF Q8_0 | GGUF Q4_K_M |
|---|---|---|---|---|---|---|
| On the device | 15.0 GiB | 7.7 GiB | 5.1 GiB | 15.0 GiB | 8.0 GiB | 5.3 GiB |
| Per decision (p50) | 89 ms | 145 ms | 138 ms | 96 ms | 147 ms | 150 ms |

- 89 ms per decision (p50; 109 ms p95) on prompts of 150 to 262 tokens; 323 ms at 1,024 tokens
  ([by prompt length](#decision-models-latency-by-prompt-length-and-what-a-pack-changes)).
- Its GGUF files give the same answer as the original's own code on 299, 295 and 287 of 300 MMLU-Pro questions at
  BF16, Q8_0 and Q4_K_M ([docs/gguf_models.md](docs/gguf_models.md)); its q8 and q4 packs the f16 answer on 292 and
  262 of 300.

Checked against its own Python code in fp32: the same prompt tokens, probabilities within 0.0007 (GPU f16); 25 of 25
test decisions right and equal to that code's.

### Clef

`Cloudflare/clef` (27B: Qwen3.8-27B with Clef-Flash's head), served from its GGUF file
[ggml-org/Clef-GGUF](https://huggingface.co/ggml-org/Clef-GGUF) Q4_K_M. The original at f16 does not fit a 24 GB
card and has not been run here; the Q4_K_M file is the form that fits (its Q8_0 and BF16 files do not).

```bash
xpu-serve convert ~/models/Clef-Q4_K_M.gguf ~/models/Clef-Q4_K_M
xpu-serve serve ~/models/Clef-Q4_K_M --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "Clef-Q4_K_M", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Measured: 443 ms per decision (p50) on prompts of about 180 tokens; 16.9 GiB on the device.

Checked against the original's own code in fp32: 25 of 25 test decisions right and equal; 281 of 300 MMLU-Pro answers
equal, probabilities within 0.11 (p95).

### pplx-decider-flash-v1.2

`perplexity-ai/pplx-decider-flash-v1.2` (4B: a Qwen3.5 decoder with a read-out per question). Every question of a
request is answered in one pass, each read at a marker after its options, calibrated with the checkpoint's
temperature. Questions sent together can move each other's answers: Perplexity's card recommends one question per
request unless the questions are facets of one decision.
A request longer than `--ctx` is answered in parts (its questions split, each part with the whole state), as its own
code does.

Reference:

```bash
xpu-serve serve ~/models/pplx-decider-flash-v1.2 --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "pplx-decider-flash-v1.2", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory (f16 where it fits: on a discrete GPU a pack is smaller but slower):

```bash
xpu-serve convert ~/models/pplx-decider-flash-v1.2 --out ~/models/pplx-decider-flash-v1.2-q8 --quant q8
xpu-serve serve ~/models/pplx-decider-flash-v1.2-q8 --device GPU --ctx 16384
```

Measured:

| | f16 | q8 pack | q4 pack |
|---|---|---|---|
| On the device | 7.8 GiB | 4.0 GiB | 2.7 GiB |
| Per decision (p50) | 39 ms | 66 ms | 62 ms |

39 ms per decision (p50; 57 ms p95) on prompts of 92 to 230 tokens; 214 ms at 1,024 tokens
([by prompt length](#decision-models-latency-by-prompt-length-and-what-a-pack-changes)).

Checked against its own Python package in fp32: the same prompt tokens, probabilities within 0.0009 (GPU f16); 25 of
25 test decisions right and equal to that code's.

### OpenJev

`openjev/openjev` (27B, a tuned Qwen3.8-27B; weights CC BY-NC 4.0). A language model: its directory serves as one
unless `--readout` asks for decisions. A prompt per question; the output layer's scores of the option letters at the
first output position, calibrated as its helper does.

Reference (its f16 weights do not fit a 24 GB card):

```bash
xpu-serve serve ~/models/openjev --device GPU --ctx 16384 --readout openjev
```

Optimized (the q4 pack, the form a 24 GB card takes it in):

```bash
xpu-serve convert ~/models/openjev --out ~/models/openjev-q4 --quant q4
xpu-serve serve ~/models/openjev-q4 --device GPU --ctx 16384 --readout openjev
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "openjev-q4", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

- `--readout openjev` reads the letters with the calibration of OpenJev's serving guide (temperature 0.85, the yes /
  no slope 1.829074); `--readout letters` reads any chat model the same way without calibration, and
  `--readout-temperature`, `--readout-noul-temperature` and `--readout-noul-bias` set the numbers.
- Each question is a prompt of its own, so its answer does not depend on the other questions; the state is read once
  per request, and a request about the state of the one before reads only its questions. A question with more than
  52 options takes several passes, as in the helper.
- OpenJev needs `instructions` in every question.

Measured (q4 pack, 16.2 GiB on the device): 266 ms per decision (p50; 478 ms p95) on prompts of 67 to 198 tokens,
143 ms for another question about the state just read; 1.46 s for three questions on a 1,024-token prompt
([by prompt length](#decision-models-latency-by-prompt-length-and-what-a-pack-changes)). Against its checkpoint's
float32 answers on 300 MMLU-Pro questions: 265 of 300 the same and 218 right against 222, with 1 of the 138 answers
held at 0.9 or more changed.

Checked against its own helper code over transformers in fp32: tiny-checkpoint parity (the same prompts,
probabilities within 0.0024 on the GPU at f16); the q4 pack against the checkpoint in fp32: 25 of 25 test decisions
right and equal, probabilities within 0.0095.

### d1-2B

`LiquidAI/d1-2B` (2B, LFM3-VL's text model; weights under the LFM Open License v1.0, commercial use only below $10M
annual revenue). Up to 16 questions are packed after one state, as its own code packs them: on the full-attention
layers the state sees everything and each question the state and itself; the output layer's scores of each option's
code tokens at its answer, calibrated (T = 1.3374).

Reference:

```bash
xpu-serve serve ~/models/d1-2B --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "d1-2B", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"},
                     "cause": {"type": "choice", "instructions": "What is the likely cause?",
                               "criteria": {"flaky": "A test that fails at random", "bug": "A defect in the change"}}}}'
```

Optimized for memory:

```bash
xpu-serve convert ~/models/d1-2B --out ~/models/d1-2B-q8 --quant q8
xpu-serve serve ~/models/d1-2B-q8 --device GPU
```

- `--set-size N` changes how many questions share a pack; `--set-size 1` reads each question alone, as llama.cpp
  does. A pack longer than `--ctx` is refused.

Measured:

| | f16 | q8 pack | q4 pack |
|---|---|---|---|
| On the device | 3.2 GiB | 1.7 GiB | 1.1 GiB |
| Same answer as its own code, of 300 MMLU-Pro questions | 299 | 295 | 257 |

14 ms per decision (p50; 72 ms p95) on prompts of 35 to 209 tokens; 15.5 ms for three questions packed after one
state (124 tokens).

Checked against its own code in fp32, up to 16 questions packed after one state as it packs them (and one at a
time): the same prompt tokens, probabilities within 0.0005 (GPU f16) and 5e-6 (f32), all 43 packed answers equal; 25
of 25 test decisions right and equal.

### Laya

`convaiinnovations/laya` (421M: ModernBERT-large and a scoring head; Apache-2.0). A sequence per question: the
question, each option at a mask token, then the state (cut to 512 tokens, as its own code cuts it); a two-layer head
scores each mask, calibrated by the number of options.

Reference:

```bash
xpu-serve serve ~/models/laya --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "laya", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory, from its GGUF files [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF) (BF16,
Q8_0); `convert` writes no q8 or q4 pack of it:

```bash
xpu-serve convert ~/models/Laya-Q8_0.gguf ~/models/Laya-Q8_0
xpu-serve serve ~/models/Laya-Q8_0 --device GPU
```

Measured: 8 ms per decision (p50) on the 25 test decisions; 19 ms for three questions (158 tokens in all); 0.78 GiB
on the device at f16.

Checked against its own code in fp32: the same token sequences, probabilities within 0.004 (GPU f16) and 1e-5 (f32);
the 25 test decisions answered as it answers them (19 right, as it gets); all 300 MMLU-Pro answers equal; its GGUF
files: 299 (BF16) and 293 (Q8_0).

### Julia-1

`SupersonicLabs/Julia-1` (144M: mmBERT-small and Laya's head; Apache-2.0). Laya's design, each option read as its
description; what its own code would cut (a long state or option) is refused, as its policy says.

Reference (every option of a choice needs a description):

```bash
xpu-serve serve ~/models/Julia-1 --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "Julia-1",
  "state": "Customer message: I was charged twice for my order last week and nobody has replied.",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges and refunds", "shipping": "Delivery and tracking",
                           "technical": "Bugs, errors and outages"}},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"}
  }
}'
```

Optimized for memory, from its GGUF files [ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF)
(BF16, Q8_0); `convert` writes no q8 or q4 pack of it:

```bash
xpu-serve convert ~/models/Julia-1-BF16.gguf ~/models/Julia-1-BF16
xpu-serve serve ~/models/Julia-1-BF16 --device GPU
```

Measured: 7 ms per decision (p50) on the 25 test decisions; 9 ms for three questions (128 tokens in all); 0.27 GiB
on the device at f16.

Checked against its own code in fp32 under its strict policy: the same refusals, probabilities within 0.0039 (GPU
f16) and 7e-6 (f32); the 17 test decisions it takes answered as it answers them (8 right, as it gets); 294 of 295
MMLU-Pro answers equal; its GGUF files: 292 (BF16) and 271 (Q8_0).

### Decision models: latency by prompt length, and what a pack changes

Latency is the time to read the prompt, so it grows with the state (three questions, p50; for OpenJev, whose three
questions are three prompts, also the time when the state was read by the request before):

| Prompt tokens | sample | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,000 |
|---|---|---|---|---|---|---|---|
| Clef-Flash | 123 ms (325 tokens) | 165 ms | 323 ms | 643 ms | 1.34 s | 2.74 s | 5.89 s |
| pplx-decider-flash-v1.2 | 46 ms (170 tokens) | 104 ms | 214 ms | 456 ms | 1.07 s | 2.53 s | 7.31 s |
| OpenJev (q4), a new state | 538 ms (84 tokens) | 1.03 s | 1.46 s | 2.54 s | 4.75 s | 9.47 s | 19.6 s |
| OpenJev (q4), the same state again | 429 ms | 434 ms | 434 ms | 440 ms | 455 ms | 475 ms | 523 ms |

`convert` writes packs of the two flash checkpoints as of any model. On the Arc Pro B60 a pack saves memory, not
time: reading a prompt from quantized weights dequantizes them at every step, 50 to 65 ms a step for Clef-Flash,
which a language model wins back in decode and a decision model never does. What a pack changes in the answers, on
300 MMLU-Pro questions (one choice of up to ten options each, which the models get right about two times in three)
against the f16 answers:

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
([docs/decisions/0015](docs/decisions/0015-decision-models.md)).

## Embedding models

A sentence-transformers checkpoint, served behind OpenAI's embeddings API
([README](README.md#embeddings)). f16 only: `convert` writes no pack of an embedding model. `xpu-serve bench <dir>`
reports the latency of one text at several lengths and the throughput of a batch; `run <dir> --prompt "..."` prints
one embedding.

### all-MiniLM-L6-v2

`sentence-transformers/all-MiniLM-L6-v2` (`bert`). Mean pooling, normalized; inputs of up to 256 tokens.

```bash
xpu-serve serve ~/models/all-MiniLM-L6-v2 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["What is a GPU?", "A graphics processing unit runs many threads at once."]}'
```

Measured: 1.9 ms per text (16 to 256 tokens); 651 texts/s of 128 tokens.

Checked against sentence-transformers in fp32 on 15 texts up to 256 tokens: the same token ids from the text;
embeddings within 1 - cosine 3.4e-7 (GPU f16), 4e-13 (GPU f32).

### bge-small-en-v1.5

`BAAI/bge-small-en-v1.5` (`bert`). CLS pooling, normalized; inputs of up to 512 tokens.

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

(Its answer on an Arc Pro B60, 384 values each, shortened and rounded here.)

Measured: 1.9 ms per text of 16 tokens, 2.1 ms at 128, 5.3 ms at 512; 471 texts/s of 128 tokens.

Checked as all-MiniLM-L6-v2 on 15 texts up to 512 tokens: 6.9e-8 (f16), 5e-13 (f32).

### nomic-embed-text-v1.5

`nomic-ai/nomic-embed-text-v1.5` (`nomic_bert`). Mean pooling, not normalized; inputs of up to 8,192 tokens. It
expects its task prefixes in the text (`search_query: `, `search_document: `).

```bash
xpu-serve serve ~/models/nomic-embed-text-v1.5 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["search_query: What is a GPU?", "search_document: A graphics processing unit runs many threads at once."]}'
```

Measured: 2.0 ms per text of 16 tokens, 3.1 ms at 128, 9.6 ms at 512, 627 ms at 8,192; 306 texts/s of 128 tokens.

Checked as all-MiniLM-L6-v2, with its own modeling code, on 16 texts up to 8,192 tokens: 5.4e-7 (f16), 1.1e-10
(f32).

## Robot policy

### π0.5

`lerobot/pi05_base` (LeRobot `pi05`), with the tokenizer as `tokenizer.json` beside it. It maps camera images, the
robot state and an instruction to 50 x 32 actions in 10 flow steps, behind the policy API
([README](README.md#robot-policy)).

Reference:

```bash
xpu-serve bench ~/models/pi05_base --device GPU.0 --precision f16 --cameras 3 --steps 10 --runs 10
xpu-serve serve ~/models/pi05_base --device GPU.0 --cameras 2    # warm up for the first 2 cameras
```

Optimized (its q8 pack quantizes the action expert, which runs 50 tokens per flow step):

```bash
xpu-serve convert ~/models/pi05_base --out ~/models/pi05_base-q8 --quant q8
xpu-serve serve ~/models/pi05_base-q8 --device GPU.0
```

Measured, a 3-camera call:

| | f16 | q8 pack |
|---|---|---|
| Arc B390 | 215 ms | 198 ms |
| Arc Pro B60 | 115 ms | 113 ms |

On the Arc B390 the q8 pack's actions are within cosine 0.99999 of LeRobot's; q4 is faster still (194 ms) but misses
the 0.999 accuracy gate (cosine 0.998), so q8 is the setting for policies. (These are times for calls with pauses
between them; a laptop that runs the policy back to back for longer than its short-term power budget lasts slows
down, see [docs/design-vla.md](docs/design-vla.md) §6.1.)

Checked against LeRobot fp32 actions: max error 5e-6 (cpu-ref), 1e-3 (GPU f16).
