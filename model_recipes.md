# Model recipes

For every model xpu-serve runs: where it comes from, the command that runs it as the reference (the original
checkpoint, f16 weights, f16 KV cache: what you would measure accuracy against), the command that runs it fast and
small, and what was measured. Start from the reference command. When it does not fit the device or is too slow, step
down: a pack, the model's drafter, a smaller KV cache. The [README](README.md) says what is supported and how the
APIs look; the numbers live here.

## Contents

- [How to read a recipe](#how-to-read-a-recipe)
- [KV cache formats](#kv-cache-formats)
- [Several requests at once](#several-requests-at-once)
- [Language models](#language-models)
  - [Gemma 4 E2B](#gemma-4-e2b)
  - [Gemma 4 E4B](#gemma-4-e4b)
  - [Gemma 4 26B-A4B](#gemma-4-26b-a4b)
  - [Qwen3.5-4B](#qwen35-4b)
  - [Qwen3.5-9B](#qwen35-9b)
  - [SenseNova-Loop-9B](#sensenova-loop-9b)
  - [Qwen3.6-35B-A3B](#qwen36-35b-a3b)
  - [Ornith-1.5-35B-A3B](#ornith-15-35b-a3b)
  - [Qwen3.8-27B](#qwen38-27b)
  - [LFM3-1.7B](#lfm3-17b)
  - [Inkling-2-Mini](#inkling-2-mini)
- [GGUF imports](#gguf-imports)
  - [Importing a GGUF file](#importing-a-gguf-file)
  - [Qwen3.5-4B GGUF](#qwen35-4b-gguf)
  - [Gemma 4 E2B GGUF](#gemma-4-e2b-gguf)
  - [Qwen3.8-27B GGUF](#qwen38-27b-gguf)
  - [Ornith-1.5-35B-A3B GGUF](#ornith-15-35b-a3b-gguf)
  - [Inkling-2-Mini GGUF](#inkling-2-mini-gguf)
  - [Inkling-2-Mini Uncensored GGUF](#inkling-2-mini-uncensored-gguf)
- [Drafters](#drafters)
  - [Speculative decoding](#speculative-decoding)
  - [Gemma 4 assistants](#gemma-4-assistants)
  - [DFlash 2 and DSpine drafters for Qwen3.8-27B](#dflash-2-and-dspine-drafters-for-qwen38-27b)
  - [The DFlash drafter for Ornith-1.5-35B-A3B](#the-dflash-drafter-for-ornith-15-35b-a3b)
  - [DSpark, Inkling 2's drafter](#dspark-inkling-2s-drafter)
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
- [Speech recognition](#speech-recognition)
  - [Qwen3-ASR-1.7B](#qwen3-asr-17b)
  - [Qwen3-ASR-0.6B](#qwen3-asr-06b)
- [Omni models](#omni-models)
  - [Qwen3-Omni-30B-A3B-Instruct](#qwen3-omni-30b-a3b-instruct)
- [Embedding models](#embedding-models)
  - [all-MiniLM-L6-v2](#all-minilm-l6-v2)
  - [bge-small-en-v1.5](#bge-small-en-v15)
  - [nomic-embed-text-v1.5](#nomic-embed-text-v15)
- [Robot policy](#robot-policy)
  - [π0.5](#π05)

## How to read a recipe

- The commands download what they need with `-hf <org/name>`: the checkpoint goes into the Hugging Face cache, where
  a model huggingface_hub or `hf download` fetched before is found and not downloaded again; `-hf <org/name:QUANT>`
  names one GGUF file of a repository ([README, Downloading models](README.md#downloading-models--hf)). A local
  directory works wherever `-hf` is shown (`xpu-serve serve <dir>`), and `~/models/<name>` below is a pack `convert`
  writes. `xpu-serve` is the installed binary ([README, Build](README.md#build)).
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
### Measured on Qwen3.8-27B (a GGUF pack)

The formats over a GGUF import: the Q4_0 file of [Qwen3.8-27B GGUF](#qwen38-27b-gguf) with its MTP head at 4 drafts
a round (`--spec mtp --spec-k 4`), on an Arc Pro B60. The weights take 16.3 GiB of the card's 24. The cache is 64 KiB
a token at f16 (16 attention layers of 4 KV heads of 256 values), and the MTP head keeps one more layer's worth.

Memory (peak GPU memory of the whole process during `xpu-serve bench`):

| `--kv-dtype` | Cache bytes a token | Against f16 | Cache at 64k tokens | Peak GPU memory at 8k | Peak GPU memory at 64k |
|---|---|---|---|---|---|
| `auto` (f16) | 65,536 | 1.00x | 4.00 GiB | 18.62 GiB | 22.28 GiB |
| `i8` | 33,280 | 1.97x | 2.03 GiB | 18.35 GiB | 20.48 GiB |
| `i8:turbo4` | 25,088 | 2.61x | 1.53 GiB | 18.27 GiB | 19.96 GiB |
| `turbo4` | 16,896 | 3.88x | 1.03 GiB | 18.22 GiB | 19.46 GiB |
| `turbo3` | 12,800 | 5.12x | 0.78 GiB | 18.19 GiB | 19.19 GiB |
| `turbo2` | 8,704 | 7.53x | 0.53 GiB | 18.15 GiB | 18.93 GiB |

Speed (a prompt that fills the context but 80 tokens, then 64 tokens generated with MTP; the bench's prompt is easy to
draft for, 4.85 tokens a verify, so these decode speeds are above a chat's):

| `--kv-dtype` | Decode at 8k tokens | Decode at 64k tokens | Prefill at 8k | Prefill at 64k |
|---|---|---|---|---|
| `auto` (f16) | 57.8 tok/s | 23.5 tok/s | 816 tok/s | 440 tok/s |
| `i8` | 59.5 tok/s | 22.4 tok/s | 807 tok/s | 435 tok/s |
| `i8:turbo4` | 59.0 tok/s | 22.9 tok/s | 811 tok/s | 435 tok/s |
| `turbo4` | 56.2 tok/s | 19.7 tok/s | 810 tok/s | 434 tok/s |
| `turbo3` | 53.9 tok/s | 17.9 tok/s | 815 tok/s | 433 tok/s |
| `turbo2` | 56.3 tok/s | 19.7 tok/s | 818 tok/s | 436 tok/s |

The longest contexts that ran on the one 24 GB card:

| `--kv-dtype` | Context | Peak GPU memory | Prefill | Decode with MTP |
|---|---|---|---|---|
| `auto` (f16) | 64k tokens (out of memory at 96k) | 22.28 GiB | 440 tok/s | 23.5 tok/s |
| `i8` | 128k tokens (out of memory at 192k) | 22.90 GiB | 255 tok/s | 13.0 tok/s |
| `turbo3` | 128k tokens | 20.22 GiB | 257 tok/s | 10.4 tok/s |
| `turbo4` | 128k tokens | 20.76 GiB | 256 tok/s | 11.5 tok/s |
| `turbo4` | 256k tokens, all the model reads | 23.47 GiB | 141 tok/s | 6.3 tok/s |

Accuracy against the f16 cache, the next-token distribution at the last half of 8 windows of 512 tokens of
wikitext-2 (2,048 positions):

| `--kv-dtype` | 512-token windows: perplexity (change) | mean KL from f16 | same top token |
|---|---|---|---|
| `auto` (f16) | 6.394 | 0 | 100 % |
| `i8` | 6.393 (-0.01 %) | 0.00005 | 99.7 % |
| `i8:turbo4` | 6.384 (-0.17 %) | 0.0015 | 97.8 % |
| `turbo4` | 6.378 (-0.26 %) | 0.0026 | 97.0 % |
| `turbo3` | 6.439 (+0.70 %) | 0.0091 | 95.6 % |
| `turbo2` | 6.636 (+3.78 %) | 0.037 | 90.2 % |
<!-- /KV-Q38-TABLES -->

What the numbers say:

- **The formats work over a GGUF import as over any model**: the cache does not depend on how the weights are
  stored.
- **Context is what they buy here.** With an f16 cache this pack stops at 64k tokens (22.3 GiB; 96k runs out of
  memory) and with `i8` at 128k. With `turbo4` it reads **all 262,144 tokens the model takes, on one 24 GB card**
  (23.5 GiB at the peak, the card's limit; a `turbo3` cache is a gigabyte smaller at that length).
- **Speed.** At 8k tokens every format decodes within 7 % of f16. At 64k `i8` and `i8:turbo4` stay within 5 %, and
  `turbo4`, `turbo3` and `turbo2` decode at 0.84, 0.76 and 0.84 of f16's speed: a verify step of five tokens
  decodes every key for five queries. Prefill is the same for every format.
- **Accuracy.** The 27B tolerates the formats better than Qwen3.5-4B: `turbo4` moves the next-token distribution
  0.6 times as far (KL 0.0026 against 0.0044), `turbo3` and `turbo2` likewise, and `i8` leaves it where it was.
- **So, for this pack**: `--kv-dtype i8` up to 128k tokens; `i8:turbo4` for f16's speed with 2.3 GiB less at 64k;
  `turbo4` or `turbo3` past 128k.

The whole context on one 24 GB card:

```bash
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --spec mtp --spec-k 4 --ctx 262144 --kv-dtype turbo4
```

## Several requests at once

`--sequences N` (on `serve`) keeps up to N requests in flight, each on a sequence of its own (its own KV cache, its
prompt kept for its next turn), and runs the decode steps of all that decode as one step: the weights are read once
for the next token of each ([docs/decisions/0028](docs/decisions/0028-batched-decode.md)). Prefill chunks run a request
at a time between those steps. The KV cache is billed for each sequence at load, so `--ctx` is a sequence's context
(on a 24 GB card, Qwen3.8-27B in Q4_0 takes eight of 8,192 tokens in f16; `--kv-dtype i8` about doubles that). With
`--spec` the requests decode a round each in turn instead.

```bash
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --sequences 8 --ctx 8192
```

Arc Pro B60, N requests sent at once (the six chat prompts in turn, thinking off, 256 greedy tokens each at most):
the tokens of all of them over the wall time from the first request to the last reply ("in all", prefills included),
and the server's decode speed of each request:

| Model | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---|---|---|---|
| Qwen3.8-27B, Q4_0 pack (dense) | 21.6 tok/s | 37.4 in all, 19.0 each | 64.1 in all, 16.9 each | 106.8 in all, 14.6 each |
| Inkling-2-Mini, AtomicChat's AD-Q4_K (mixture of experts) | 42.6 (45.9 decoding) | 61.5 in all, 32.9 each | 74.0 in all, 21.0 each | 88.2 in all, 13.0 each |

A dense model reads the same weights for every token, so eight requests cost little more than one: 4.9 times the
tokens. A mixture of experts reads the experts each token picks, more of them as requests are added, and its steps
hold more stateful ops (Inkling 2: a convolution beside every sublayer and attention in each of 48 layer runs), each a
launch a request: 2.1 times at eight.

## Language models

All of them are served behind the OpenAI API ([README](README.md#chat-completions)). Where a model thinks by
default, `"chat_template_kwargs": {"enable_thinking": false}` in a request turns that off.

### Gemma 4 E2B

`google/gemma-4-E2B-it` (`gemma4`). Text and images. Drafter: `google/gemma-4-E2B-it-assistant`.

Reference:

```bash
xpu-serve serve -hf google/gemma-4-E2B-it --device GPU
```

Optimized:

```bash
xpu-serve convert -hf google/gemma-4-E2B-it --out ~/models/gemma-4-E2B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-E2B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 1 -hfd google/gemma-4-E2B-it-assistant
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
xpu-serve serve -hf google/gemma-4-E4B-it --device GPU
```

Optimized:

```bash
xpu-serve convert -hf google/gemma-4-E4B-it --out ~/models/gemma-4-E4B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-E4B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 2 -hfd google/gemma-4-E4B-it-assistant
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
xpu-serve serve -hf google/gemma-4-26B-A4B-it --device GPU
```

Optimized:

```bash
xpu-serve convert -hf google/gemma-4-26B-A4B-it --out ~/models/gemma-4-26B-A4B-it-q4 --quant q4
xpu-serve serve ~/models/gemma-4-26B-A4B-it-q4 --device GPU --kv-dtype i8 \
    --spec mtp --spec-k 2 -hfd google/gemma-4-26B-A4B-it-assistant
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
xpu-serve serve -hf Qwen/Qwen3.5-4B --device GPU
```

Optimized:

```bash
xpu-serve convert -hf Qwen/Qwen3.5-4B --out ~/models/Qwen3.5-4B-q4 --quant q4 --method awq
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
xpu-serve serve -hf Qwen/Qwen3.5-9B --device GPU
```

Optimized:

```bash
xpu-serve convert -hf Qwen/Qwen3.5-9B --out ~/models/Qwen3.5-9B-q4 --quant q4
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

### SenseNova-Loop-9B

`sensenova/SenseNova-Loop-9B` (`sensenova_loop`; Apache 2.0; gated on the Hub: accept its terms on the model page,
then give `-hf` a token in `HF_TOKEN`), or its open preview `sensenova/SenseNova-Loop-9B-Preview`. A Qwen3.5-9B
decoder whose layers 4 to 27 run twice, with the residual carried as two streams through them (manifold-constrained
hyper-connections; [docs/decisions/0022](docs/decisions/0022-sensenova-loop.md)): 56 layer runs a token, so it
decodes at about 0.55 times Qwen3.5-9B's speed from the same bytes of weights. Text (the vision tower is not loaded).
A reasoning model: thinking is on unless a request turns it off. Its MTP head is the drafter, inside the checkpoint
and its packs.

Reference (16.7 GiB of weights on the device; 64k tokens of context fit a 24 GB card):

```bash
xpu-serve serve -hf sensenova/SenseNova-Loop-9B --device GPU
```

Optimized (56 tok/s over six chat prompts on the Arc Pro B60; 9.4 GiB at 64k tokens):

```bash
xpu-serve convert -hf sensenova/SenseNova-Loop-9B --out ~/models/SenseNova-Loop-9B-q4 --quant q4
xpu-serve serve ~/models/SenseNova-Loop-9B-q4 --device GPU --spec mtp --spec-k 2 --kv-dtype i8 --ctx 65536
```

Full context (the release's 512K tokens by its model card), the q4 pack with an int8 KV cache and no drafter:

```bash
xpu-serve convert -hf sensenova/SenseNova-Loop-9B --out ~/models/SenseNova-Loop-9B-q4 --quant q4
xpu-serve serve ~/models/SenseNova-Loop-9B-q4 --device GPU --kv-dtype i8 --ctx 524288
```

- Estimated, not measured: the cache is 14.2 GiB at 524,288 tokens, about 23 GiB in all with the weights and the
  prefill's scratch, at the edge of a 24 GB card. If it runs out of memory, `--kv-dtype i8:turbo4` (10.7 GiB of
  cache, about 20 GiB in all) or `turbo4` (7.2 GiB, about 17 GiB).
- `--ctx` is capped at the checkpoint's `max_position_embeddings`: 262,144 in the Preview's config. The release's
  config was not read here (gated).

- Closer to the original: `--quant q8` keeps 290 of the original's 300 answers below (q4: 229), at two thirds of
  q4's speed. `--method awq` on the q4 pack keeps 252 at 11 % less decode speed, and its MTP rounds are slower still
  (42 tok/s instead of 56).
- Less VRAM: `--kv-dtype i8` stores 28 KiB a token instead of 56 (both at 64k below), `turbo4` 14.4 KiB
  ([KV cache formats](#kv-cache-formats)).
- The MTP head's second draft is accepted less often than Qwen3.5's, so `--spec-k 2` is the best on q4 and q8 (k = 3
  equals it at f16).

Measured on `sensenova/SenseNova-Loop-9B-Preview` (the release has its architecture; the gated weights were not
measured), Arc Pro B60, one GPU. Peak GPU memory of the process, prefill of a prompt of that length, then 64 greedy
tokens after it:

| Context | f16: memory | prefill | decode | q8: memory | prefill | decode | q4: memory | prefill | decode |
|---|---|---|---|---|---|---|---|---|---|
| 1k | 17.5 GiB | 1,597 tok/s | 12.2 tok/s | 9.5 GiB | 1,362 | 21.1 | 6.5 GiB | 1,384 | 32.6 |
| 2k | 17.7 GiB | 1,584 | 12.2 | 9.6 GiB | 1,355 | 21.1 | 6.7 GiB | 1,376 | 32.6 |
| 4k | 18.1 GiB | 1,559 | 12.2 | 10.0 GiB | 1,336 | 21.1 | 7.0 GiB | 1,349 | 32.6 |
| 16k | 18.9 GiB | 1,364 | 11.7 | 10.8 GiB | 1,190 | 19.7 | 7.8 GiB | 1,207 | 29.4 |
| 32k | 19.7 GiB | 1,134 | 11.2 | 11.8 GiB | 1,013 | 18.2 | 8.7 GiB | 1,026 | 26.3 |
| 64k | 21.5 GiB | 794 | 10.4 | 13.4 GiB | 732 | 16.1 | 10.5 GiB | 739 | 22.1 |

Weights on the device: 16.7 GiB (f16), 8.6 GiB (q8), 5.6 GiB (q4); packs on disk 10.4 GB and 7.1 GB, from 19.3 GB.
The KV cache takes 56 KiB a token (14 full-attention runs of 4 KV heads of 256): 3.5 GiB at 64k. The q4 pack at
64k with `--kv-dtype i8`: 9.0 GiB, decode 22.8 tok/s; with `turbo4`: 8.2 GiB, 21.7 tok/s.

Over six chat prompts (thinking off, 256 greedy tokens each), decode without → with its MTP head:

| | Without | k = 1 | k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|
| f16 | 12.4 tok/s | 20.6 | 24.2 | 24.4 | 22.9 |
| q8 | 21.5 tok/s | 34.3 | 37.3 | 36.7 | 35.1 |
| q4 | 33.6 tok/s | 51.1 | 56.0 | 55.2 | 51.8 |

2.2 tokens a verify at k = 2. The optimized command as written (q4, k = 2, `i8`) decodes them at 56.4 tok/s.

Accuracy, against the checkpoint's own code in float32 on the CPU, 300 MMLU-Pro questions read at the option letters
with thinking off (`--readout openjev`, as for [Ornith-1.5-35B-A3B](#ornith-15-35b-a3b)):

| | Right (300) | The original's answer kept (300) | Shift |
|---|---|---|---|
| *original, f32 on the CPU* | 98 | 300 | 0 |
| f16 | 98 | 300 | 0.001 |
| q8 pack | 95 | 290 | 0.019 |
| q4 pack | 96 | 229 | 0.184 |
| q4 pack, AWQ | 112 | 252 | 0.140 |

Checked against the checkpoint's own code in fp32 (transformers 5.18): last-token logits cosine 0.999997 or better at
f16, 32 greedy tokens and the MTP head's drafts equal on six prompts of up to 1,691 tokens (q8: cosine 0.9996, q4:
0.95 to 0.997); thinking, tool calls, streaming and prefix reuse through the server.

### Qwen3.6-35B-A3B

`Qwen/Qwen3.6-35B-A3B`, or its FP8 release `Qwen/Qwen3.6-35B-A3B-FP8` (half the download) (`qwen3_5_moe`, a mixture
of experts). Text; its MTP head is the drafter.

Reference (its f16 weights do not fit a 24 GB card; an FP8 checkpoint also runs as f16 on the device):

```bash
xpu-serve serve -hf Qwen/Qwen3.6-35B-A3B --device GPU
```

Optimized (a q4 pack fits one 24 GB card):

```bash
xpu-serve convert -hf Qwen/Qwen3.6-35B-A3B-FP8 --out ~/models/Qwen3.6-35B-A3B-q4 --quant q4
xpu-serve serve ~/models/Qwen3.6-35B-A3B-q4 --device GPU --spec mtp --spec-k 3 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy; they need a head_dim that is a multiple of 128
  ([KV cache formats](#kv-cache-formats)).
- Its GGUF files (`qwen35moe`) import like Ornith-1.5-35B-A3B's ([Ornith-1.5-35B-A3B GGUF](#ornith-15-35b-a3b-gguf)).

Measured: at q4, decode 62 tok/s, and 94 tok/s with MTP (k = 3); prefill 1,550 tok/s.

Checked against Hugging Face on a tiny checkpoint; its q4 pack against Hugging Face fp32 logits of the checkpoint
(cosine 0.92 to 0.98); MMLU-Pro (25 questions): 80 to 84 %.

### Ornith-1.5-35B-A3B

`ornith-ai/Ornith-1.5-35B-A3B`, or its FP8 release `ornith-ai/Ornith-1.5-35B-A3B-FP8` (39 GB to download instead of
72) (`qwen3_5_moe`: Qwen3.6-35B-A3B's shape, 3B of its 35B parameters active a token, trained further by Ornith; MIT
license). Text. A reasoning model: thinking is on unless a request turns it off. Drafters: its MTP head (inside the
checkpoint and its packs) and the DFlash drafter `ornith-ai/Ornith-1.5-35B-A3B-DFlash`
([its section](#the-dflash-drafter-for-ornith-15-35b-a3b)).

Reference (its f16 weights do not fit a 24 GB card; the FP8 release also runs as f16 on the device):

```bash
xpu-serve serve -hf ornith-ai/Ornith-1.5-35B-A3B --device GPU
```

Optimized (a q4 pack fits one 24 GB card, with the drafter's pack beside it):

```bash
xpu-serve convert -hf ornith-ai/Ornith-1.5-35B-A3B-FP8 --out ~/models/Ornith-1.5-35B-A3B-q4 --quant q4
xpu-serve convert -hf ornith-ai/Ornith-1.5-35B-A3B-DFlash --target ~/models/Ornith-1.5-35B-A3B-q4 \
    --out ~/models/Ornith-1.5-35B-A3B-DFlash-q4
xpu-serve serve ~/models/Ornith-1.5-35B-A3B-q4 --device GPU --spec dflash \
    --drafter ~/models/Ornith-1.5-35B-A3B-DFlash-q4 --spec-k 3 --kv-dtype i8
```

- Less VRAM: `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2` store the KV cache in fewer bits, each at a
  growing cost in accuracy ([KV cache formats](#kv-cache-formats)). Long contexts are measured on the GGUF file:
  [Ornith-1.5-35B-A3B GGUF](#ornith-15-35b-a3b-gguf).
- On the Arc B390 the MTP head is the faster drafter: `--spec mtp --spec-k 2`.
- Closer to the original: round-to-nearest q4 moves this model's answers more than it moves Qwen3.6's when it answers
  without thinking (below). A pack that keeps everything but the expert banks at q8
  (`--q8 linear_attn,self_attn,shared_expert,lm_head` on `convert`, 24.2 GB), or the GGUF Q4_K_M file keeps more of the
  original's answers there, at about a tenth less speed without a drafter (with the MTP head the q8 one decodes as fast,
  83 tok/s); with thinking on, the three scored alike on 50 questions (34 to 39, within what 50 questions can tell).
- Its GGUF files: [Ornith-1.5-35B-A3B GGUF](#ornith-15-35b-a3b-gguf).

Measured over six chat prompts (thinking off, 256 greedy tokens each; drafters at the best `--spec-k`):

| | GPU memory | Decode | With MTP | With the DFlash drafter | Prefill (1k / 4k tokens) |
|---|---|---|---|---|---|
| Arc Pro B60, q4 pack | 20.4 GiB | 61 tok/s | 83 tok/s (k = 2) | 89 tok/s (k = 3) | 1,465 / 1,608 tok/s |
| Arc Pro B60, q4 with the rest at q8 | 21.2 GiB | 54 tok/s | 83 tok/s (k = 2) | 81 tok/s (k = 3) | 1,474 tok/s (1k) |
| Arc B390, q4 pack | 20.4 GiB | 36 tok/s | 55 tok/s (k = 2) | 48 tok/s (k = 3) | 639 / 728 tok/s |

The optimized command as written (the drafter at k = 3, an `i8` cache) decodes the six prompts at 89 tok/s. With
thinking on, the drafter does better: answering MMLU-Pro questions with the DFlash drafter at k = 3, the q4 pack
decoded at 102 tok/s on the B60 (the median over 50 answers of 2,700 tokens on average; the pack with the rest at q8,
95 tok/s; the GGUF Q4_K_M, 73 tok/s).

Accuracy, against the original checkpoint's own answers in float32 on the CPU:

| | MMLU-Pro, thinking on (50 questions) | MMLU-Pro read at the option letters, right (300) | The original's answer kept (300) | Shift |
|---|---|---|---|---|
| *original, f32 on the CPU* | | 125 | 300 | 0 |
| GGUF Q4_K_M | 39 (5 ran past 8,192 tokens) | 114 | 213 | 0.280 |
| q4 with the rest at q8 | 34 (8 ran past 8,192 tokens) | 111 | 198 | 0.308 |
| q4 pack | 36 (9 ran past 8,192 tokens) | 113 | 163 | 0.388 |

"Read at the option letters" is the letter read-out with thinking off (`--readout openjev`, the same 300 questions
as [docs/gguf_models.md](docs/gguf_models.md)): a reasoning model answers worse without thinking, and its answer
distributions there are wide, so they move more with rounding than Qwen3.8-27B's (shift 0.07 to 0.13 for its 4-bit
files). The q4 packs from the FP8 release and from the bf16 original hold the same tensors to cosine 0.9945 and keep
163 and 168 of the 300 ([docs/decisions/0014](docs/decisions/0014-fp8-checkpoints.md)).

Checked: the engine's prompts are the reference's token for token, and on one pack the GPU's answers differ from the
reference as much as the CPU backend's at f32 do (the difference is the rounding of the weights). The model code is
Qwen3.6-35B-A3B's (tiny-checkpoint parity with Hugging Face); the FP8 release's 30,880 FP8 tensors against the bf16
original at cosine 0.99964 or better.

### Qwen3.8-27B

`Qwen/Qwen3.8-27B`, or its FP8 release `Qwen/Qwen3.8-27B-FP8` (half the download) (`qwen3_5`). Text. Drafters: its
MTP head (inside the checkpoint and its packs), and the block drafters `z-lab/Qwen3.8-27B-DFlash2` and
`tencent/Qwen3.8-27B-DSpine-b16`.

Reference (its f16 weights do not fit a 24 GB card; an FP8 checkpoint also runs as f16 on the device):

```bash
xpu-serve serve -hf Qwen/Qwen3.8-27B --device GPU
```

Optimized (a q4 pack fits one 24 GB card):

```bash
xpu-serve convert -hf Qwen/Qwen3.8-27B-FP8 --out ~/models/Qwen3.8-27B-q4 --quant q4
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
xpu-serve serve -hf LiquidAI/LFM3-1.7B --device GPU
```

Optimized (its q8 pack: at q4 its reasoning runs long more often):

```bash
xpu-serve convert -hf LiquidAI/LFM3-1.7B --out ~/models/LFM3-1.7B-q8 --quant q8
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

### Inkling-2-Mini

`thinkingmachines/Inkling-2-Mini` (`inkling2_loop`; Apache 2.0): a mixture of experts of 26.4B parameters (128
experts, 8 a token) whose layers 6 to 17 run three times, 48 layer runs a token, without rotary positions
([docs/decisions/0026](docs/decisions/0026-inkling2.md)). Text and images: a pack of the checkpoint keeps its vision
tower (an image is one token per 40 x 40 pixels, not resized), and so does a GGUF file's import, from the
repository's `mmproj-BF16.gguf`, which `-hf` fetches with it. Its drafter (DSpark) is
inside the checkpoint and its packs, and a file of its GGUF repository ([DSpark](#dspark-inkling-2s-drafter)). Its
thinking comes back as `reasoning_content` and its tool calls as `tool_calls`; a request's `reasoning_effort` (none,
minimal, low, medium, high, xhigh, max) sets how much it thinks, 0.9 (high) when left out.

Reference: the release stores its experts in NVFP4 and MXFP8, which a backend holds as f16 (49 GiB): more than a 24 GB
card or an Arc B390 has. The release's own values on one device are its lossless GGUF file, UD-Q8_K_XL (21.3 GiB: the
NVFP4 experts as stored, the rest in bf16; [Inkling-2-Mini GGUF](#inkling-2-mini-gguf)).

Two settings per device, from the measurements below: the closest to the release that fits, and the balance of
accuracy, speed and context. UD-Q4_K_M scores within 0.3 % of the lossless file's perplexity at 15.4 GiB.

| | Best accuracy: UD-Q8_K_XL | Balanced: UD-Q4_K_M |
|---|---|---|
| Arc Pro B60 (24 GB) | 35.5 tok/s, 39.3 with DSpark; text only: 63,488 tokens with an `i8` cache, 186,368 in `turbo3` | 43.2 tok/s, 47.1 with DSpark; 369,664 tokens in `i8`, 993,280 in `turbo3` (all 1,048,576 text only) |
| Arc B390 (30 GB of system memory) | 16.3 tok/s, 18.1 with DSpark; 262,144 tokens in `i8`, 524,288 in `turbo3` | 23.1 tok/s, 24.9 with DSpark; 524,288 tokens in `i8`, all 1,048,576 in `turbo3` |

Arc Pro B60:

```bash
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q8_K_XL --device GPU --spec dspark --spec-k 1 \
    --kv-dtype i8 --ctx 28672 --no-vision                                            # best accuracy
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M --device GPU --spec dspark --spec-k 1 \
    --kv-dtype i8 --ctx 262144                                                       # balanced
```

Arc B390:

```bash
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q8_K_XL --device GPU --spec dspark --spec-k 1 \
    --kv-dtype i8 --ctx 131072                                                       # best accuracy
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M --device GPU --spec dspark --spec-k 1 \
    --kv-dtype i8 --ctx 262144                                                       # balanced
```

`--ctx` is reserved when the model loads, and a context that does not fit is refused there with the most tokens that
do. On the Arc Pro B60 (strict bill: the KV cache of the whole context, each program's largest step, the vision
tower's arena):

| File, KV cache | Text and images | With DSpark | Text only (`--no-vision`) | Text only, with DSpark |
|---|---|---|---|---|
| UD-Q8_K_XL, `i8` | 1,024 | none | 63,488 | 28,672 |
| UD-Q8_K_XL, `turbo3` | 7,168 | none | 186,368 | 91,136 |
| UD-Q4_K_M, `i8` | 369,664 | 300,032 | 437,248 | 360,448 |
| UD-Q4_K_M, `turbo3` | 993,280 | 808,960 | all 1,048,576 | 967,680 |

The vision tower's arena holds a whole image's patches, which the lossless file cannot spare beside the drafter: its
best-accuracy setting on the B60 is text only. The drafter adds its weights (0.45 GiB from the repository's file, whose
head is folded through the output layer: [DSpark](#dspark-inkling-2s-drafter)), a ninth KV slot that grows
with the context and a verify's step; a long document decodes faster without it (below). The B390's contexts are the
longest run there, its memory shared with the system, before the load-time reservation: not yet checked against it.

Fastest (a q4 pack of the checkpoint with its drafter, perplexity +1.6 %: 57.7 tok/s over six chat prompts on the Arc
Pro B60, 26.7 on the Arc B390):

```bash
xpu-serve convert -hf thinkingmachines/Inkling-2-Mini --out ~/models/Inkling-2-Mini-q4 --quant q4
xpu-serve serve ~/models/Inkling-2-Mini-q4 --device GPU --spec dspark --spec-k 1 --kv-dtype i8
```

- Long contexts: forty of the 48 layer runs attend within 1,024 tokens and keep only that window, so the KV cache
  grows by 32 KiB a token at f16 (16.5 KiB in `i8`, 8.25 KiB in `turbo4`, 6.25 KiB in `turbo3`;
  [KV cache formats](#kv-cache-formats)). `i8` costs next to nothing here (perplexity +0.23 % on the lossless file),
  `turbo4` +0.28 % and `turbo3` +2.0 % (on UD-Q4_K_M, KL 0.010 and 0.024), more than the file's own +0.29 %: past
  65,536 tokens on the B60 the balanced file in `i8` or `turbo4` keeps more of the model than the lossless one in
  `turbo3`.
- Closer to the original than UD-Q4_K_M without the lossless file's size: the q8 pack (25.1 GiB, larger than the
  release's NVFP4 experts) fits the Arc B390 only, where it decodes at 16.5 tok/s.
- The drafter gains about a tenth: every draft verified reads the experts of its own token, and the first draft is
  accepted 63 % of the time, so one draft a round is the best.

Measured over six chat prompts (greedy; 256 tokens each on the B60, 128 on the B390). Perplexity over 4,096 positions of
wikitext-2 (16 windows of 512 tokens, the last 256 of each scored; `tests/unit/model/kv_format_quality.cpp`); against
Hugging Face fp32 (transformers' `Inkling2Loop`), the last-token logits' cosine (lowest and mean of six prompts of up to
1,775 tokens) and how many of 32 greedy tokens stay equal, per prompt:

| | GPU memory | Decode | With DSpark | Prefill (512 tokens) | Perplexity | Logits cosine | Greedy tokens kept |
|---|---|---|---|---|---|---|---|
| Arc Pro B60, GGUF UD-Q8_K_XL | 21.3 GiB | 35.5 tok/s | 39.3 tok/s | 612 tok/s | 8.108 | 0.9999 / 1.0000 | 32 on all six |
| Arc Pro B60, GGUF UD-Q4_K_M | 15.4 GiB | 43.2 tok/s | 47.1 tok/s | 589 tok/s | 8.131 (+0.29 %) | 0.9995 / 0.9999 | 32, 31, 32, 26, 32, 32 |
| Arc Pro B60, q4 pack | 15.4 GiB | 51.9 tok/s | 57.7 tok/s | 604 tok/s | 8.240 (+1.63 %) | 0.9981 / 0.9991 | 32, 31, 22, 26, 4, 32 |
| Arc B390, GGUF UD-Q8_K_XL | 21.3 GiB | 16.3 tok/s | 18.1 tok/s | 216 tok/s | | 0.9999 / 1.0000 | 32, 31, 32, 32, 4, 32 |
| Arc B390, GGUF UD-Q4_K_M | 15.4 GiB | 23.1 tok/s | 24.9 tok/s | 220 tok/s | | 0.9997 / 0.9999 | 32, 32, 32, 26, 32, 32 |
| Arc B390, q4 pack | 15.4 GiB | 24.1 tok/s † | 26.7 tok/s † | 230 tok/s | | 0.9981 / 0.9991 | 32, 31, 22, 26, 4, 32 |
| Arc B390, q8 pack | 25.1 GiB | 16.5 tok/s † | 17.3 tok/s † | 208 tok/s | | 0.9998 / 0.9999 | 32, 32, 32, 32, 4, 32 |

† Measured before the decode fixes of [docs/decisions/0026](docs/decisions/0026-inkling2.md) (the routers' top-k,
RmsNorm, the block residual's transpose), which made a decode step 15 to 23 % faster on the B60 and 8 to 11 % on the
B390 where measured again.

DSpark at one draft a round. The fifth prompt parts at its fifth token on several variants (the lossless file on the
B390 too): two tokens there are within f16's rounding of each other, and the lossless file in float32 on the CPU keeps
all 32. Every GGUF file's numbers: [Inkling-2-Mini GGUF](#inkling-2-mini-gguf).

**Long contexts.** Each context filled with a prompt of that length, then 64 tokens decoded without a drafter
(measured before the decode fixes †):

| Device, file | Context | KV cache | Prefill | Decode |
|---|---|---|---|---|
| Arc Pro B60, UD-Q8_K_XL | 32,768 | `i8` | 732 tok/s | 25.0 tok/s |
| | 65,536 | `i8` | 673 tok/s | 22.1 tok/s |
| | 98,304 | `turbo4` | 606 tok/s | 17.1 tok/s |
| | 131,072 | `turbo3` | 557 tok/s | 16.0 tok/s |
| | 196,608 | `turbo3` | 479 tok/s | 13.1 tok/s |
| Arc Pro B60, UD-Q4_K_M | 32,768 | `i8` | 716 tok/s | 28.5 tok/s |
| | 131,072 | f16 | 569 tok/s | 19.0 tok/s |
| | 262,144 | `i8` | 418 tok/s | 14.0 tok/s |
| | 524,288 | `turbo4` | 280 tok/s | 6.5 tok/s |
| | 1,048,576 | `turbo3` | 169 tok/s | 4.0 tok/s |
| Arc B390, UD-Q8_K_XL | 131,072 | `i8` | 266 tok/s | 8.9 tok/s |
| | 262,144 | `i8` | 217 tok/s | 6.4 tok/s |
| | 524,288 | `turbo3` | 154 tok/s | 3.3 tok/s |
| Arc B390, UD-Q4_K_M | 131,072 | `i8` | 249 tok/s | 10.7 tok/s |
| | 524,288 | `i8` | 151 tok/s | 4.0 tok/s |
| | 524,288 | `turbo4` | 152 tok/s | 4.5 tok/s |
| | 1,048,576 | `turbo3` | 98 tok/s | 1.9 tok/s |

On the B60 the lossless file runs out of device memory at 262,144 tokens in `turbo3`. On the B390 the GPU's memory is
the system's (30.3 GiB): the most in use during a run, the system's own included, was 29.6 GiB for the lossless file at
262,144 tokens in `i8` and 28.4 at 524,288 in `turbo3`, 29.5 for the balanced one at 524,288 in `i8` and 27.9 at
1,048,576 in `turbo3`; with DSpark, 28.1 for the lossless file at 131,072 and 26.7 for the balanced one at 262,144 (both
`i8`).

DSpark at length (one draft a round, `i8`; before the decode fixes †): at 32,768 tokens on the B60 the lossless
file decodes at 30.0 tok/s with it (25.0 without) and the balanced one at 33.3 (28.5); at 131,072 the balanced one at
18.4 (19.8) and at 262,144 at 11.4 (14.0). Measured again on the balanced file with the decode fixes and a verify's
rows sharing their reads of the cache: at 32,768 tokens 38.7 tok/s with it (35.6 without), at 126,000 24.3 (28.4). On the B390 the lossless file decodes at 8.5 tok/s at 131,072 (8.9 without), the balanced one at 5.8 at 262,144.
Each row of a verify, and the drafter's block, reads the whole cache: between 32,768 and 131,072 tokens the drafter
turns from a gain into a loss. Prefill is the same with it.

Checked against transformers' `Inkling2Loop` in fp32: on a tiny checkpoint (cosine 0.999996 or better at f16, greedy
tokens equal, on cpu-ref and the GPU), and with the released weights through the lossless GGUF (cosine 1 and 32 of
32 greedy tokens on all six prompts on cpu-ref in f32). The drafter against a float64 port of its math
([DSpark](#dspark-inkling-2s-drafter)).

## GGUF imports

### Importing a GGUF file

A GGUF file (llama.cpp's format) is imported into a pack in a few seconds. Each quantized tensor stays exactly as
the file stores it (nothing is decoded or quantized again), so the model has the file's weights and takes the file's
size in GPU memory. The pack is then served like any model, with the same `--spec` and `--kv-dtype` options.

`-hf <org/name:QUANT>` names a file of a repository by its quantization (`Q4_K_M` when none is given): it is downloaded
into the Hugging Face cache, imported once into a pack in xpu-serve's cache (`~/.cache/xpu-serve/packs`, or
`XS_CACHE_DIR`) and run from there, so it takes the file's size twice on disk. A Qwen3.5-family file that comes without
its MTP head gets the one its repository publishes apart (`MTP/mtp-*.gguf`), so `--spec mtp` works on it.

```bash
xpu-serve serve -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --device GPU
xpu-serve convert -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M <pack-dir>       # the pack in a directory of your choice
```

A file on disk imports with `convert`:

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
xpu-serve serve -hf unsloth/Qwen3.5-4B-GGUF:BF16 --device GPU
```

Optimized:

```bash
xpu-serve serve -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --device GPU --kv-dtype i8
```

- Less VRAM: a smaller file, and `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2`
  ([KV cache formats](#kv-cache-formats)).
- These files hold no MTP head and the repository publishes none (`-hf` says so when it imports one), so
  `--spec mtp` needs the original checkpoint or a pack of it ([Qwen3.5-4B](#qwen35-4b)).
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
xpu-serve serve -hf unsloth/gemma-4-E2B-it-GGUF:BF16 --device GPU
```

Optimized, with its drafter (`-hfd` of the repository takes its `MTP/mtp-*.gguf` file, Q8_0 by default, imported
as a drafter pack):

```bash
xpu-serve serve -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --device GPU --kv-dtype i8 \
    --spec mtp -hfd unsloth/gemma-4-E2B-it-GGUF
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
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_M --device GPU
```

Optimized (the fastest file with its MTP head):

```bash
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --spec mtp --spec-k 4 --kv-dtype i8
```

The smallest files come without the MTP head; `-hf` takes it from the head's own file (`MTP/mtp-*.gguf`), and the
pack is the one a file with the head gives:

```bash
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS --device GPU --spec mtp
```

From files on disk, `convert <file.gguf> <pack-dir> --mtp <mtp-file.gguf>` does the same.

- Less VRAM: a smaller file, and `--kv-dtype i8:turbo4`, then `turbo4`, `turbo3` and `turbo2`
  ([KV cache formats](#kv-cache-formats)).
- Long contexts (`--ctx N`): an f16 cache fits 64k tokens beside this file on a 24 GB card, `--kv-dtype i8` 128k, and
  `--kv-dtype turbo4` the model's whole 262,144
  ([measured on this pack](#measured-on-qwen38-27b-a-gguf-pack)):

```bash
xpu-serve serve -hf unsloth/Qwen3.8-27B-GGUF:Q4_0 --device GPU --spec mtp --spec-k 4 --ctx 262144 --kv-dtype turbo4
```

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
| UD-IQ2_XXS (head from its own file) | 7.21 GB | 6.70 GiB | 17 tok/s | 32 tok/s | 174 | 154 |
| *our q4 pack* | | 16.21 GiB | 22 tok/s | 51 tok/s | 245 | 185 |

On an Arc B390 the Q4_0 file decodes at 6.4 tok/s and at 17.4 with MTP (k = 6, 2.7 times; 17.2 at k = 5): a verify
of up to 8 rows costs about one decode step there.

### Ornith-1.5-35B-A3B GGUF

[ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (`qwen35moe`): five
files, each with the MTP head inside. Only Q4_K_M (21.7 GB) fits a 24 GB card; Q5_K_M (25.3 GB), Q6_K, Q8_0 and BF16
hold more weights than the card has memory. The mmproj file (the vision tower) is not read.

Closest to the reference on one card, and with its whole context (`turbo4` cache):

```bash
xpu-serve serve -hf ornith-ai/Ornith-1.5-35B-A3B-GGUF:Q4_K_M --device GPU --spec mtp --spec-k 2 \
    --ctx 262144 --kv-dtype turbo4
```

- Its MTP head drafts as well as the DFlash drafter over this file (71 tok/s either way on the B60; the command above,
  with its `turbo4` cache, 75 tok/s in a later run), so no second download is needed. The DFlash drafter:
  `--spec dflash -hfd ornith-ai/Ornith-1.5-35B-A3B-DFlash --spec-k 3`.
- With a drafter the greedy text can part from plain decoding where two tokens are close: a verify of several tokens
  reads the activations in f16 (the matrix-engine GEMV over block formats), a single-token step in f32. With
  `--backend-opt gemv=vector` the two agree token for token (checked on both GPUs).
- Q4_K_M here is Q4_K for 82 % of the weights and Q6_K for 18 % (half the layers' down-projection expert banks, the
  output head, part of the attention), imported byte for byte: GPU memory is the file's size.

Measured, thinking off, six chat prompts of 256 greedy tokens (drafters at their best `--spec-k`):

| | GPU memory | Decode | With MTP | With the DFlash drafter | Prefill (1k / 4k tokens) |
|---|---|---|---|---|---|
| Arc Pro B60 | 19.7 GiB | 55 tok/s | 71 tok/s (k = 2) | 71 tok/s (k = 3) | 1,211 / 1,321 tok/s |
| Arc B390 | 19.7 GiB | 34 tok/s | 51 tok/s (k = 3) | 47 tok/s (k = 3) | 603 / 675 tok/s |

Accuracy: in [Ornith-1.5-35B-A3B](#ornith-15-35b-a3b), beside the packs (this file keeps 213 of the original's 300
answers, the q4 pack 163).

**Long contexts.** The KV cache of the model's ten full-attention layers takes 20,480 bytes a token at f16, 10,400 in
`i8` and 5,280 in `turbo4` (the other thirty layers keep a fixed-size state). On one Arc Pro B60, each context filled
with a prompt of that length and then 64 tokens decoded (no drafter):

| `--kv-dtype` | 32k | 64k | 128k | 256k (262,144, the model's whole context) |
|---|---|---|---|---|
| f16 | 21.6 GiB, 1,063 / 40.9 | 22.2 GiB, 871 / 34.7 | 23.5 GiB, 589 / 26.1 | out of device memory |
| `i8` | 21.4 GiB, 1,078 / 43.2 | 21.8 GiB, 873 / 35.7 | 22.5 GiB, 584 / 27.0 | 23.9 GiB, 321 / 2.3 (the card is full) |
| `turbo4` | 21.2 GiB, 898 / 40.3 | 21.5 GiB, 744 / 32.0 | 21.9 GiB, 532 / 24.3 | **22.8 GiB, 342 / 16.7** |

Each cell: peak GPU memory, then prefill and decode in tok/s. With `turbo4` the whole 262,144-token context fits a
24 GB card with about 1.2 GiB to spare (its prefill takes 12.8 minutes); an `i8` cache of that length fills the card,
and decode slows to a crawl. With the MTP head (`--spec mtp --spec-k 2`, the command above) in `turbo4`:
22.6 GiB and decode 29.4 tok/s at 128k, 23.5 GiB and 17.5 tok/s at 256k (2.86 tokens a round; at that length each
round's attention over the whole cache is most of its cost).

On the Arc B390 (its memory is the system's: 30 GB in the machine measured), `turbo4` also takes the whole context:
prefill 526, 271 and 150 tok/s and decode 26.4, 16.7 and 11.4 tok/s at 32k, 128k and 256k (that prefill takes 29
minutes).

### Inkling-2-Mini GGUF

[unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`inkling2_loop`): all ten files of
the model (UD-IQ1_S to UD-Q8_K_XL, 7.8 to 22.9 GB) and the drafter's (`dspark-Inkling-2-Mini-Q8_0.gguf`, 500 MB).
`-hf` fetches the drafter's file with the model's and imports both into one pack, so `--spec dspark` works on every
file. The 4-bit and larger files keep the release's NVFP4 expert banks of 20 layers as they are (77 % of the weights)
and differ in the rest; the smaller ones quantize more of the banks further (NVFP4 is 57 % of UD-Q3_K_XL, 1 % of
UD-IQ1_S). `-hf` also fetches the vision tower's file (`mmproj-BF16.gguf`), so images
work from every file. Since 2026-11-23 the repository's files are in mainline llama.cpp's layout (`inkling2_loop`, as
[AtomicChat's](#inkling-2-mini-uncensored-gguf) are: the same tensors, byte for byte, under other names) and its
drafter in mainline's format (its head folded through the output layer; [DSpark](#dspark-inkling-2s-drafter)). The
earlier files, in Unsloth's layout (`inkling2`), are in `Old_Support/` as `.gguf_file` (the drafter also as
`dspark-Inkling-2-Mini-Q8_0_Old.gguf`) and import too, but not through `-hf`: downloaded and renamed to `.gguf`,
`convert <file.gguf> <pack-dir> --mtp <drafter.gguf> --mmproj <mmproj.gguf>`, with either drafter. The two layouts
of a file are the same model: every file measured again from the repository's new layout gave the perplexity below
within 0.13 % (f16's rounding) and the speeds below.

Closest to the reference, on either device (the release's own values; on a 24 GB card text only, with the drafter
28,672 tokens of context, 63,488 without; [what fits](#inkling-2-mini)):

```bash
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q8_K_XL --device GPU --spec dspark --spec-k 1 --kv-dtype i8 \
    --ctx 28672 --no-vision
```

Balanced (perplexity +0.29 %, two thirds of the memory; long contexts in [Inkling-2-Mini](#inkling-2-mini)):

```bash
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M --device GPU --spec dspark --spec-k 1 --kv-dtype i8
```

Measured: perplexity over 4,096 positions of wikitext-2 as for [Inkling-2-Mini](#inkling-2-mini), the logits against
Hugging Face fp32 (the lowest and mean cosine of six prompts; greedy tokens kept of 6 x 32), and greedy decode over
six chat prompts without and with the drafter (k = 1, the best of 1 to 3 on every file):

| Variant | File | GPU memory | Perplexity | Logits cosine | Greedy kept (192) | B60 decode | With DSpark | B390 decode | With DSpark |
|---|---|---|---|---|---|---|---|---|---|
| UD-Q8_K_XL | 22.9 GB | 21.27 GiB | 8.108 | 0.9999 / 1.0000 | 192 | 35.5 tok/s | 39.3 tok/s | 16.3 tok/s | 18.1 tok/s |
| UD-Q4_K_XL | 17.6 GB | 16.34 GiB | 8.129 (+0.27 %) | 0.9994 / 0.9999 | 163 | 43.1 | 47.3 | 20.7 † | 23.4 † |
| UD-Q4_K_M | 16.6 GB | 15.39 GiB | 8.131 (+0.29 %) | 0.9995 / 0.9999 | 185 | 43.2 | 47.1 | 23.1 | 24.9 |
| UD-IQ4_XS | 15.7 GB | 14.57 GiB | 8.136 (+0.34 %) | 0.9997 / 0.9999 | 163 | 43.3 | 46.2 | 21.6 † | 24.5 † |
| UD-Q3_K_XL | 13.6 GB | 12.62 GiB | 8.176 (+0.84 %) | 0.9908 / 0.9964 | 131 | 43.1 | 45.4 | 22.4 † | 25.3 † |
| UD-IQ3_XXS | 11.7 GB | 10.78 GiB | 8.306 (+2.44 %) | 0.9937 / 0.9965 | 128 | 43.5 | 45.8 | 23.6 † | 26.3 † |
| UD-Q2_K_XL | 9.14 GB | 8.43 GiB | 8.551 (+5.47 %) | 0.9766 / 0.9879 | 98 | 45.4 | 48.4 | 24.9 † | 28.6 † |
| UD-IQ2_XXS | 8.49 GB | 7.82 GiB | 8.724 (+7.60 %) | 0.9636 / 0.9851 | 100 | 44.8 | 48.8 | 25.6 † | 29.3 † |
| UD-IQ1_M | 8.02 GB | 7.39 GiB | 8.799 (+8.53 %) | 0.9952 / 0.9973 | 139 | 44.9 | 48.1 | 26.0 † | 29.9 † |
| UD-IQ1_S | 7.76 GB | 7.15 GiB | 8.862 (+9.31 %) | 0.9815 / 0.9882 | 48 | 45.6 | 48.8 | 26.1 † | 30.4 † |
| *our q4 pack* | | 15.44 GiB | 8.240 (+1.63 %) | 0.9981 / 0.9991 | 147 | 51.9 | 57.7 | 24.1 † | 26.7 † |

† Measured before the decode fixes of [Inkling-2-Mini](#inkling-2-mini), which made the files measured again 8 to 11 %
faster on the B390. The B60's columns are the files of the repository's new layout, with its new drafter.

Prefill of 512 tokens: 568 to 596 tok/s on the B60 and 215 to 232 on the B390 for every file (UD-Q8_K_XL: 590 on the
B60 and 216 on the B390, its four layers of bf16 experts read by the grouped GEMM). Perplexity is the sharper
measure: six prompts decide little between files whose cosine is 0.999 or better, and greedy tokens part at the first
near tie. Between the 4-bit files the differences (0.07 %) are within what f16 rounding moves (the attention kernels'
order alone moved the lossless file's by 0.1 %); an `i8` KV cache moves each file's by 0.3 % or less.

- **Speed.** A decode step is bound by the device, not by its launches: on the B60 two thirds of it reads the
  weights a token goes through, the rest is some 3,400 small kernels of 1 to 3 us
  ([docs/decisions/0026](docs/decisions/0026-inkling2.md)). So a smaller file is faster by less than its size: 35.5
  tok/s for UD-Q8_K_XL, 43.2 for UD-Q4_K_M, 45.6 for UD-IQ1_S at a third of the lossless file's size. On the B390,
  whose memory is slower, size counts for more.
- **Memory is the file's**, the drafter's (0.45 GiB) and the vision tower's when they are loaded; the contexts each
  file takes are in [Inkling-2-Mini](#inkling-2-mini).

### Inkling-2-Mini Uncensored GGUF

[AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF)
(`inkling2_loop`, mainline llama.cpp's layout; Apache 2.0): Inkling-2-Mini with a rank-1 edit that removes each
layer's refusal direction (attention outputs, routed experts and the embedding), merged into the weights, so it
answers requests the original refuses. Three files (AD-Q4_K 16.5 GB, AD-IQ3_S 13.3 GB, AD-IQ2_M 8.67 GB), the
drafter's (`dspark-Inkling-2-Mini-Q8_0.gguf`, in mainline llama.cpp's format) and the vision tower's, which `-hf`
fetches with any of them: images and `--spec dspark` work. Its adapter file (`Inkling-2-Mini-Uncensored-LoRA.gguf`,
44 MB: the same edit for any GGUF of the model) applies with `--lora`, each change added where the model reads the
weight it changes, which stays as the file stores it: Unsloth's files then keep their NVFP4 experts.

Arc Pro B60:

```bash
xpu-serve serve -hf AtomicChat/Inkling-2-Mini-Uncensored-GGUF:Q4_K --device GPU --spec dspark --spec-k 1 \
    --ctx 131072                                                                     # closest to the model
xpu-serve serve -hf AtomicChat/Inkling-2-Mini-Uncensored-GGUF:IQ2_M --device GPU --spec dspark --spec-k 1 \
    --kv-dtype i8                                                                    # smallest
```

Measured on the Arc Pro B60: perplexity over 4,096 positions of wikitext-2 as for [Inkling-2-Mini](#inkling-2-mini),
against the original's lossless file (8.108), so it holds the edit's change and the file's rounding together; what an
`i8` KV cache changes in it; greedy decode over six chat prompts of 256 tokens without and with the drafter (k = 1);
prefill of 512 tokens:

| File | Size | GPU memory | Perplexity | With an `i8` cache | Decode | With DSpark | Prefill (512 tokens) |
|---|---|---|---|---|---|---|---|
| AD-Q4_K | 16.5 GB | 15.36 GiB | 8.177 (+0.85 %) | +0.09 %, KL 0.003 | 45.8 tok/s | 51.9 tok/s | 600 tok/s |
| AD-IQ3_S | 13.3 GB | 12.40 GiB | 8.321 (+2.6 %) | −0.16 %, KL 0.003 | 39.8 tok/s | 42.2 tok/s | 597 tok/s |
| AD-IQ2_M | 8.67 GB | 8.04 GiB | 9.014 (+11.2 %) | +0.10 %, KL 0.003 | 46.0 tok/s | 50.1 tok/s | 630 tok/s |

- **Answers.** Through the server on AD-IQ3_S and AD-IQ2_M: 17 x 23, why Canberra and not Sydney, a tool call and the
  answer from its result, the shapes and colours of a picture and the text in another, a 150-word story; on AD-Q4_K
  with the drafter, 17 x 23, a picture, a tool call and the story. Every answer was right, the tool calls well formed
  and the stories coherent (a story thinks long at the default effort: give it room, 6,000 tokens or so).
- **Their experts are i-quants.** The edit changes every routed expert, and these files store them as IQ3_S and
  IQ4_NL (AD-IQ3_S) or IQ2_XXS, Q2_0 and IQ4_NL (AD-IQ2_M), where Unsloth's files keep the release's NVFP4. AD-IQ3_S
  scores below UD-Q3_K_XL (13.6 GB, +0.84 %, its experts mostly NVFP4); how much of that is the edit's own change
  these measurements cannot tell.
- **Contexts.** AD-Q4_K with images and the drafter holds 131,072 tokens in f16 on the B60 (0.7 GiB to spare); the
  smaller files take less memory than UD-Q4_K_M, so they hold at least its [contexts](#inkling-2-mini).
- **The adapter** on Unsloth's UD-Q4_K_M (Arc Pro B60, 512 + 128 tokens): decode 40.8 to 39.1 tok/s, prefill 588 to
  515 tok/s (in the repository's new layout, the one the adapter was written for; 38.8 to 37.2 and 589 to 514 on the
  earlier one, where the model then answered what it declined without it).

```bash
curl -LO https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF/resolve/main/Inkling-2-Mini-Uncensored-LoRA.gguf
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_M --device GPU --kv-dtype i8 \
    --lora Inkling-2-Mini-Uncensored-LoRA.gguf
```

## Drafters

### Speculative decoding

`--spec` (on `run`, `serve` and `bench`) drafts with the model's drafter and verifies the drafts in one step; greedy
output stays the model's own and sampling keeps its distribution (rejection sampling).

| `--spec` | Drafter | Where it comes from |
|---|---|---|
| `mtp` | the MTP head of a Qwen3.5-family checkpoint | inside the model (and its pack) |
| `mtp` | a Gemma 4 assistant | `-hfd google/gemma-4-*-it-assistant` (or `--drafter <dir>`) |
| `dflash` | a DFlash or DFlash 2 block drafter | `-hfd z-lab/Qwen3.8-27B-DFlash2` (or `--drafter <dir>`) |
| `dspine` | a DSpine block drafter | `-hfd tencent/Qwen3.8-27B-DSpine-b16` (or `--drafter <dir>`) |
| `dspark` | Inkling 2's DSpark block drafter | inside the checkpoint and its packs; the GGUF repository's `dspark-*.gguf`, which `-hf` imports with the model's file |

Which model takes which technique. A drafter is trained for one model, so a technique works only where a drafter
for that model exists (`–`: none does, and the option is refused by name). Each cell is the decode speed in tok/s
without → with that technique, and the `--spec-k` (drafts per round) that gave it, which is the one to use:

| Model | Measured on | `mtp` | `dflash` | `dspine` | `dspark` |
|---|---|---|---|---|---|
| Gemma 4 E2B | q4, Arc Pro B60 | 140 → 174, k = 1 | – | – | – |
| Gemma 4 E4B | q4, Arc Pro B60 | 84 → 126, k = 2 | – | – | – |
| Gemma 4 26B-A4B | q4, Arc Pro B60 | 67 → 91, k = 2 | – | – | – |
| Qwen3.5 4B | q4, Arc Pro B60 | 104 → 149, k = 2 | – | – | – |
| Qwen3.5 9B | q4, Arc Pro B60 | 62 → 108, k = 4 | – | – | – |
| SenseNova-Loop-9B | q4, Arc Pro B60 | 34 → 56, k = 2 | – | – | – |
| Qwen3.6-35B-A3B | q4, Arc Pro B60 | 62 → 94, k = 3 | – | – | – |
| Ornith-1.5-35B-A3B | q4, Arc Pro B60 | 61 → 83, k = 2 | 61 → 89, k = 3 | – | – |
| Ornith-1.5-35B-A3B | q4, Arc B390 | 36 → 55, k = 2 | 36 → 48, k = 3 | – | – |
| Ornith-1.5-35B-A3B | GGUF Q4_K_M, Arc Pro B60 | 55 → 71, k = 2 | 55 → 71, k = 3 | – | – |
| Qwen3.8-27B | q4, Arc Pro B60 | 22 → 50, k = 4 | 22 → 61, k = 7 | 22 → 51, k = 7 | – |
| Qwen3.8-27B | q4, Arc B390 | 6.6 → 18, k = 5 | 6.6 → 20, k = 7 | 6.6 → 17.5, k = 7 | – |
| LFM3-1.7B | | – | – | – | – |
| Inkling-2-Mini | q4, Arc Pro B60 | – | – | – | 51.9 → 57.7, k = 1 |
| Inkling-2-Mini | q4, Arc B390 | – | – | – | 24.1 → 26.7, k = 1 (before the decode fixes) |
| Inkling-2-Mini | GGUF UD-Q8_K_XL, Arc Pro B60 | – | – | – | 35.5 → 39.3, k = 1 |
| Inkling-2-Mini | GGUF UD-Q8_K_XL, Arc B390 | – | – | – | 16.3 → 18.1, k = 1 |

Greedy decode over six chat prompts of up to 256 new tokens (`bench --prompts`); the drafters as q4 packs, the Gemma 4
assistants as their checkpoints. DFlash 2 and DSpine checkpoints are released for Qwen3.8-27B; a DFlash one
(the original format) for Ornith-1.5-35B-A3B. With thinking on a drafter does better than in this table (reasoning
text is easier to predict): Ornith-1.5-35B-A3B's q4 pack with its DFlash drafter decodes on the B60 while it
reasons (102 tok/s, the median over 50 answers).

- A drafter checkpoint is served as it is (`-hfd <org/name>`), or from a pack of its own, which is smaller and drafts
  faster: `convert -hf <drafter> --target <model> --out <dir>`, then `--drafter <dir>`.
- `--spec-k` is 3 by default and 7 for a block drafter, which proposes up to its whole block in one pass and runs only
  the rows a round asks for ([docs/design-spec-decode.md](docs/design-spec-decode.md)); the table has the value to
  pass per model.
- An MTP head picks each draft among 64 tokens that a one-bit sketch of the output layer shortlists
  (`--draft-shortlist`, 0 to read the whole layer).
- `bench --prompts file` measures over real prompts, since acceptance depends on the text, and prints where a round's
  time goes.

### Gemma 4 assistants

`google/gemma-4-E2B-it-assistant`, `google/gemma-4-E4B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant`: each
drafts for its own model, from its checkpoint as it is.

```bash
xpu-serve run -hf google/gemma-4-E2B-it --device GPU --spec mtp --spec-k 1 -hfd google/gemma-4-E2B-it-assistant
```

`--spec-k`: 1 for E2B, 2 for E4B and 26B-A4B ([the table above](#speculative-decoding)).

### DFlash 2 and DSpine drafters for Qwen3.8-27B

`z-lab/Qwen3.8-27B-DFlash2` (blocks of 8) and `tencent/Qwen3.8-27B-DSpine-b16` (blocks of 16), for the q4 pack of
[Qwen3.8-27B](#qwen38-27b) (`~/models/Qwen3.8-27B-q4`). A drafter checkpoint is served as it is:

```bash
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dflash -hfd z-lab/Qwen3.8-27B-DFlash2
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dspine -hfd tencent/Qwen3.8-27B-DSpine-b16
```

or from a pack of its own, which is smaller and drafts faster:

```bash
xpu-serve convert -hf z-lab/Qwen3.8-27B-DFlash2 --target ~/models/Qwen3.8-27B-q4 \
    --out ~/models/Qwen3.8-27B-DFlash2-q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dflash --drafter ~/models/Qwen3.8-27B-DFlash2-q4

xpu-serve convert -hf tencent/Qwen3.8-27B-DSpine-b16 --target ~/models/Qwen3.8-27B-q4 \
    --out ~/models/Qwen3.8-27B-DSpine-q4
xpu-serve serve ~/models/Qwen3.8-27B-q4 --device GPU --spec dspine --drafter ~/models/Qwen3.8-27B-DSpine-q4
```

Measured in [Qwen3.8-27B](#qwen38-27b): 22 → 61 tok/s with DFlash 2 and 22 → 51 with DSpine on the Arc Pro B60, at the
default of 7 drafts a round.

### The DFlash drafter for Ornith-1.5-35B-A3B

`ornith-ai/Ornith-1.5-35B-A3B-DFlash` (blocks of 16; five sliding-window layers under a full-attention one), for
[Ornith-1.5-35B-A3B](#ornith-15-35b-a3b) and its GGUF file. As it is, or as a pack of its own (231 MB at q4):

```bash
xpu-serve serve ~/models/Ornith-1.5-35B-A3B-q4 --device GPU --spec dflash --spec-k 3 \
    -hfd ornith-ai/Ornith-1.5-35B-A3B-DFlash
xpu-serve convert -hf ornith-ai/Ornith-1.5-35B-A3B-DFlash --target ~/models/Ornith-1.5-35B-A3B-q4 \
    --out ~/models/Ornith-1.5-35B-A3B-DFlash-q4
```

Over the six chat prompts with the q4 target on the B60: 89, 86 and 79 tok/s at k = 3, 5 and 7 (2.46, 2.75 and 2.85
tokens a round); the checkpoint as it is drafts a little slower (86, 83, 77). Its acceptance matches its reference:
z-lab's own code over the float32 target accepts 2.79 tokens a round on the same prompts at its whole block, this
engine 2.87 over the q4 target, and the drafter proposes 74 to 80 % of the reference's drafts round by round.

### DSpark, Inkling 2's drafter

Inkling-2-Mini's own block drafter: three decoder layers that read four of the target's hidden states fill a block of
8 rows in one pass, and a recurrent head picks the drafts from it one after another, each from the token before it
([docs/decisions/0026](docs/decisions/0026-inkling2.md)). It is inside the checkpoint (`mtp.safetensors`) and every
pack `convert` makes of it, and a file of its own in the GGUF repositories, which `-hf` imports with any model file. The
file is in mainline llama.cpp's format since 2026-11-23 (its head's correction folded through the output layer:
`dspark-Inkling-2-Mini-Q8_0.gguf` of Unsloth's and AtomicChat's repositories), or Unsloth's earlier one
(`dspark-Inkling-2-Mini-Q8_0_Old.gguf`); both import:

```bash
xpu-serve serve ~/models/Inkling-2-Mini-q4 --device GPU --spec dspark --spec-k 1
xpu-serve serve -hf unsloth/Inkling-2-Mini-GGUF:UD-Q8_K_XL --device GPU --spec dspark --spec-k 1
```

From files on disk: `convert <model.gguf> <pack-dir> --mtp dspark-Inkling-2-Mini-Q8_0.gguf`.

Over the six chat prompts with the q4 pack on the Arc Pro B60 (41.7 tok/s without, a decode step 24 ms; before the
decode fixes of [Inkling-2-Mini](#inkling-2-mini), after which k = 1 gives 51.9 → 57.7):

| `--spec-k` | 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|
| Decode | 50.1 tok/s | 50.3 | 46.9 | 43.2 | 34.5 |
| Tokens a verify | 1.63 | 1.98 | 2.16 | 2.26 | 2.38 |
| A verify | 30.1 ms | 36.1 | 41.8 | 47.2 | 61.9 |

Each row of a verify costs about 5.4 ms more: its token picks 8 experts of its own, whose weights the verify reads
too, where a dense model's rows share every weight. Past the second draft, a draft is accepted less often than its
row costs; the default is 2 drafts, one is as fast and the better choice on the B390 and the GGUF files. The head
reads the whole output layer for each draft: the MTP heads' shortlist of 64 rows lowered the first draft's acceptance
from 63 % to 56 %. With the folded head of mainline's file, the output layer reads the block's rows once and each
draft reads the fold (98 MiB): on UD-Q4_K_M (Arc Pro B60, the same target, two runs each) one draft gives 47.6 tok/s
against 48.0 with the earlier file, two 45.0 against 44.4, the drafts accepted the same (612 of 917 at one).

Checked against a float64 port of the drafter's math over transformers' target on a tiny checkpoint: the same drafts
on all 1,050 of them on cpu-ref and the GPU in f32 (no public module exists to compare with). On the released weights
the drafter reads the target's states its config names (the stream entering layer runs 5, 23 and 41, and the final
normed state): k = 2 accepts 49 % of its drafts, and 43 to 44 % with any of them moved one layer run.

## Decision models

A decision model answers typed questions about a state instead of writing text. `serve` puts one behind the System
One API ([README](README.md#decisions-the-system-one-api) has the schema's rules and errors). Every model below takes
the request of the next section; where a model wants something else of it, its section says so.

### The System One request

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

(Clef-Flash's answer, rounded here to three digits; it took 117 ms on an Arc Pro B60.) Every model answers in this
shape: a `choice` with the chosen option and each option's probability, a `noul` with the probability of yes, a
`score` with the expected level and each level's probability.

Without a server, `run` answers one request file and `bench` times one (its own sample by default, `--state-tokens N`
to grow the state):

```bash
xpu-serve run   -hf Cloudflare/clef-flash --device GPU --request request.json
xpu-serve bench -hf Cloudflare/clef-flash --device GPU --ctx 16384 --state-tokens 4096 --runs 7
python3 tests/e2e/systemone_eval.py --url http://127.0.0.1:8080      # 25 decisions: answers, API shape, latency
```

The per-decision times below are medians over 25 test decisions, one question a request.

### Clef-Flash

`Cloudflare/clef-flash` (9B: a Qwen3.5 decoder with a joint schema head). Every question of a request is answered in
one pass; a transformer head scores the options. Questions sent together can move each other's answers.

Reference:

```bash
xpu-serve serve -hf Cloudflare/clef-flash --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "Cloudflare/clef-flash", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory (f16 where it fits: on a discrete GPU a pack is smaller but slower):

```bash
xpu-serve convert -hf Cloudflare/clef-flash --out ~/models/clef-flash-q8 --quant q8
xpu-serve serve ~/models/clef-flash-q8 --device GPU --ctx 16384
```

Or one of its GGUF files, [ggml-org/Clef-Flash-GGUF](https://huggingface.co/ggml-org/Clef-Flash-GGUF) (BF16, Q8_0,
Q4_K_M):

```bash
xpu-serve serve -hf ggml-org/Clef-Flash-GGUF:Q8_0 --device GPU --ctx 16384
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
xpu-serve serve -hf ggml-org/Clef-GGUF:Q4_K_M --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "ggml-org/Clef-GGUF:Q4_K_M", "state": "The build failed twice on the same test.",
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
xpu-serve serve -hf perplexity-ai/pplx-decider-flash-v1.2 --device GPU --ctx 16384
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "perplexity-ai/pplx-decider-flash-v1.2", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory (f16 where it fits: on a discrete GPU a pack is smaller but slower):

```bash
xpu-serve convert -hf perplexity-ai/pplx-decider-flash-v1.2 --out ~/models/pplx-decider-flash-v1.2-q8 --quant q8
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
xpu-serve serve -hf openjev/openjev --device GPU --ctx 16384 --readout openjev
```

Optimized (the q4 pack, the form a 24 GB card takes it in):

```bash
xpu-serve convert -hf openjev/openjev --out ~/models/openjev-q4 --quant q4
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
xpu-serve serve -hf LiquidAI/d1-2B --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "LiquidAI/d1-2B", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"},
                     "cause": {"type": "choice", "instructions": "What is the likely cause?",
                               "criteria": {"flaky": "A test that fails at random", "bug": "A defect in the change"}}}}'
```

Optimized for memory:

```bash
xpu-serve convert -hf LiquidAI/d1-2B --out ~/models/d1-2B-q8 --quant q8
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
xpu-serve serve -hf convaiinnovations/laya --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model": "convaiinnovations/laya", "state": "The build failed twice on the same test.",
       "questions": {"retry": {"type": "noul", "instructions": "Is it worth retrying the build?"}}}'
```

Optimized for memory, from its GGUF files [ggml-org/Laya-GGUF](https://huggingface.co/ggml-org/Laya-GGUF) (BF16,
Q8_0); `convert` writes no q8 or q4 pack of it:

```bash
xpu-serve serve -hf ggml-org/Laya-GGUF:Q8_0 --device GPU
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
xpu-serve serve -hf SupersonicLabs/Julia-1 --device GPU
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "SupersonicLabs/Julia-1",
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
xpu-serve serve -hf ggml-org/Julia-1-GGUF:BF16 --device GPU
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

## Speech recognition

A speech recognition model is served behind OpenAI's transcription API ([README](README.md#transcriptions)), its
realtime API ([README](README.md#realtime-transcription)) and as `input_audio` parts in chat. WAV, MP3 and FLAC at any
sample rate. Audio longer than a piece is cut at quiet points and transcribed piece by piece: the model's own limit
(20 minutes for Qwen3-ASR), or less where the server's `--ctx` holds less (at the default 8,192 tokens, about 5
minutes of Qwen3-ASR's audio with room for its longest reply).

### Qwen3-ASR-1.7B

`Qwen/Qwen3-ASR-1.7B` (`qwen3_asr`): an audio encoder (convolutions over the log-mel features, attention within
windows of 8 seconds) and a Qwen3 decoder; 30 languages and 22 Chinese dialects, found by the model or forced with
`language`. No timestamps (they need Qwen3-ForcedAligner, not supported yet).

Reference:

```bash
xpu-serve serve -hf Qwen/Qwen3-ASR-1.7B --device GPU
curl -s http://127.0.0.1:8080/v1/audio/transcriptions -F file=@speech.wav -F language=en
```

Optimized (its q8 pack: 1.5 times as fast as f16 at nearly the same word error rate; larger prefill steps for long
audio):

```bash
xpu-serve convert -hf Qwen/Qwen3-ASR-1.7B --out ~/models/Qwen3-ASR-1.7B-q8 --quant q8
xpu-serve serve ~/models/Qwen3-ASR-1.7B-q8 --device GPU --prefill-chunk 8192
```

- `"stream": "true"` in the form sends the text as it is decoded (`transcript.text.delta` events).
- q4 is faster still and transcribed every clip here, but in a 10-minute piece (`--ctx 32768`) it missed the end of
  the speech and began the clip again, until the reply's limit of 4,096 tokens; f16 and q8 stopped where the speech
  did. q8 is the safer pack for long pieces.

Measured on an Arc Pro B60 with the commands above (`--prefill-chunk 8192`, the default `--ctx`), through the API,
each request a clip the server had not seen (no cached prompt). Word error rate on LibriSpeech's 73 clips of
`hf-internal-testing/librispeech_asr_dummy` (481 s, 1,169 words; Whisper's English text normalizer on both sides);
the longer clips are those clips joined:

| | f16 | q8 | q4 |
|---|---|---|---|
| Word error rate | 3.34 % | 3.51 % | 4.11 % |
| A clip of 6.6 s on average (median) | 235 ms | 157 ms | 118 ms |
| All 73 clips, one after another | 20.2 s (24x real time) | 13.2 s (36x) | 10.2 s (47x) |
| 33 s clip | 1.42 s | 0.92 s | 0.70 s |
| 120 s clip | 4.79 s | 3.01 s | 2.31 s |
| 600 s clip (two pieces) | 24.6 s (24x) | 15.8 s (38x) | 12.2 s (49x) |
| First streamed text: 33 s / 120 s / 600 s | 149 / 351 / 926 ms | 160 / 361 / 986 ms | 157 / 365 / 985 ms |
| Decode (a 120 s clip's reply) | 79 tok/s | 133 tok/s | 182 tok/s |
| GPU memory, the 600 s clip | 7.6 GB | 5.8 GB | 5.1 GB |

Where a 600 s clip's time goes (q8): the encoder 0.52 s (75 windows of 8 s: 8 a run, then one a run), prefill of
7,837 prompt tokens 0.86 s, decoding 1,786 tokens 14.1 s, the log-mel features about 0.08 s on the CPU. The first text
comes after the first piece's encoder and prefill.

Checked against transformers' Qwen3-ASR in fp32: log-mel features, the encoder's rows, the prompt, last-token logits
and greedy text, on LibriSpeech clips of 6 to 120 s (one to fifteen windows) and synthetic ones; f32 on the GPU:
every greedy token equal; f16: logits cosine 0.999997 or more, every greedy token equal.

Realtime (`/v1/realtime`, q8, LibriSpeech read aloud at real time in 100 ms pieces): with the server's VAD a turn ends
500 ms (`silence_duration_ms`) after the speech does; from then, the first text about 50 ms and the whole
transcript 130 to 250 ms later for turns of 3 to 7 s. With the client committing at the end of the speech instead:
first text 55 to 150 ms after the commit, the whole transcript 145 to 345 ms after it for 5 to 12 s of speech, 780 ms
for 29 s.

### Qwen3-ASR-0.6B

`Qwen/Qwen3-ASR-0.6B` (`qwen3_asr`): the same model, smaller (a 1,024-wide decoder), served the same way.

```bash
xpu-serve serve -hf Qwen/Qwen3-ASR-0.6B --device GPU --prefill-chunk 8192
```

Measured as Qwen3-ASR-1.7B, at f16: word error rate 3.93 %; a clip of 6.6 s in 107 ms (median), all 73 clips in 9.4 s
(51x real time); 33 s in 0.68 s, 120 s in 2.30 s, 600 s in 12.0 s; first streamed text after 111 / 266 / 776 ms;
decode 167 tok/s. At f16 it is about as fast as the 1.7B's q4, with a lower word error rate.

## Omni models

### Qwen3-Omni-30B-A3B-Instruct

`Qwen/Qwen3-Omni-30B-A3B-Instruct` (`qwen3_omni_moe`): its thinker, Qwen's audio encoder, Qwen3-VL's vision tower
and a Qwen3 mixture-of-experts decoder (48 layers, 128 experts, 8 a token, 3B active). Text, audio (`input_audio`
parts: WAV, MP3 or FLAC) and images (`image_url` data URLs) in, text out, tool calls (JSON, the `qwen3` dialect).
Video and speech out are not supported yet: the talker the checkpoint holds is not read ([decision
0025](docs/decisions/0025-qwen3-omni.md)). The other releases of the thinker (`-Thinking`, `-Captioner`) are the same
model type.

The checkpoint is 70.5 GB in bf16: on a 24 GB card it runs as a q4 pack, its encoders at q8 (`--scope all`):

```bash
xpu-serve convert -hf Qwen/Qwen3-Omni-30B-A3B-Instruct --out ~/models/Qwen3-Omni-30B-A3B-q4 --quant q4 --scope all
xpu-serve serve ~/models/Qwen3-Omni-30B-A3B-q4 --device GPU --image-max-pixels 1048576
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": "'"$(base64 -w0 speech.flac)"'", "format": "flac"}},
    {"type": "text", "text": "Describe the speaker and summarize what they say."}]}]
}'
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}},
    {"type": "text", "text": "What is in this picture?"}]}]
}'
```

- The device holds 19.6 GB of weights: the thinker (18.3 GB), the vision tower (0.67 GB) and the audio encoder
  (0.66 GB), the encoders at q8. The pack (26 GB) also keeps the talker and Code2Wav, for when they are read.
- `--image-max-pixels 1048576` resizes an image to at most a megapixel, about 1,000 tokens: the checkpoint's own
  bound is 12.8 megapixels (12,544 tokens), which a 24 GB card does not hold beside the weights and whose vision tower
  takes seconds (its attention spans every patch of the image). At the default `--ctx 8192` and that bound, a
  2048 x 2048 photo peaked at 23.3 GB. Without the encoders' q8 (the default `--scope`, encoders in f16) a megapixel
  image peaked at 24.1 GB and the same photo at its full size ran the card out of memory. The f16 tower is the more
  faithful one: its rows against transformers' at cosine 0.99999, the q8 tower's at 0.991 to 0.997 (the answers
  below were right with both).
- Text and audio alone: `--no-vision` frees the vision tower, and up to `--ctx 16384` fits.
- Sampling defaults: greedy (its generation_config.json sets only the talker's).

Measured on an Arc Pro B60 (q4):

| Prompt | 512 | 2,048 | 4,096 | 8,000 | 16,000 |
|---|---|---|---|---|---|
| Prefill | 1,028 tok/s | 1,546 tok/s | 1,473 tok/s | 1,282 tok/s | 1,003 tok/s |
| Decode | 64 tok/s | 65 tok/s | 58 tok/s | 50 tok/s | 41 tok/s |

| Image | Tokens | Vision tower | The whole answer |
|---|---|---|---|
| a 640 x 360 invoice: "Read the invoice number and the total." | 239 | 70 ms | 0.95 s, both read right |
| the Hugging Face docs' cat photo | 648 | 0.30 s | 1.4 s: "a Pallas's cat, also known as a manul" |
| the same at 2048 x 2048, resized to a megapixel | 1,043 | 0.73 s | 3.0 s |
| the same at 2048 x 2048, not resized (no `--image-max-pixels`) | 4,114 | 9.1 s | 13.5 s |

A short chat decodes at about 70 tok/s. Asked about LibriSpeech's first clip ("What is said in this audio?") it
quoted it word for word in 1.6 s; asked to describe a 12 s clip's speaker and summarize it, it wrote 85 tokens in
2.0 s. A prompt of a few hundred tokens takes 0.5 to 0.75 s to prefill, about what 512 tokens do: a mixture of
experts reads nearly every expert for any step of more than a few tokens.

Checked against transformers' thinker on the original weights, asked about two LibriSpeech clips and a text prompt:
on cpu-ref in f32, the log-mel features, the encoder's rows (cosine 1, within 8.6e-7) and every greedy token equal;
last logits within 1.5e-5 but for the 12 s clip's, cosine 0.99939, the amount transformers' own float32 differs from
its float64 on that prompt (a router's top-8 decided within float32 rounding). Asked to describe the cat photo and the
invoice: the processor's patches within 6e-8, the vision tower's rows and the last logits at cosine 1, every greedy
token equal. The q4 pack on the GPU: the audio encoder's rows cosine 1, last logits cosine 0.967 to 0.986 with the same
top token; its answers are worded otherwise from the first tokens on (both correct).

## Embedding models

A sentence-transformers checkpoint, served behind OpenAI's embeddings API
([README](README.md#embeddings)). f16 only: `convert` writes no pack of an embedding model. `xpu-serve bench -hf
<org/name>` reports the latency of one text at several lengths and the throughput of a batch; `run -hf <org/name>
--prompt "..."` prints one embedding.

### all-MiniLM-L6-v2

`sentence-transformers/all-MiniLM-L6-v2` (`bert`). Mean pooling, normalized; inputs of up to 256 tokens.

```bash
xpu-serve serve -hf sentence-transformers/all-MiniLM-L6-v2 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["What is a GPU?", "A graphics processing unit runs many threads at once."]}'
```

Measured: 1.9 ms per text (16 to 256 tokens); 651 texts/s of 128 tokens.

Checked against sentence-transformers in fp32 on 15 texts up to 256 tokens: the same token ids from the text;
embeddings within 1 - cosine 3.4e-7 (GPU f16), 4e-13 (GPU f32).

### bge-small-en-v1.5

`BAAI/bge-small-en-v1.5` (`bert`). CLS pooling, normalized; inputs of up to 512 tokens.

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

(Its answer on an Arc Pro B60, 384 values each, shortened and rounded here.)

Measured: 1.9 ms per text of 16 tokens, 2.1 ms at 128, 5.3 ms at 512; 471 texts/s of 128 tokens.

Checked as all-MiniLM-L6-v2 on 15 texts up to 512 tokens: 6.9e-8 (f16), 5e-13 (f32).

### nomic-embed-text-v1.5

`nomic-ai/nomic-embed-text-v1.5` (`nomic_bert`). Mean pooling, not normalized; inputs of up to 8,192 tokens. It
expects its task prefixes in the text (`search_query: `, `search_document: `).

```bash
xpu-serve serve -hf nomic-ai/nomic-embed-text-v1.5 --device GPU
curl -s http://127.0.0.1:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["search_query: What is a GPU?", "search_document: A graphics processing unit runs many threads at once."]}'
```

Measured: 2.0 ms per text of 16 tokens, 3.1 ms at 128, 9.6 ms at 512, 627 ms at 8,192; 306 texts/s of 128 tokens.

Checked as all-MiniLM-L6-v2, with its own modeling code, on 16 texts up to 8,192 tokens: 5.4e-7 (f16), 1.1e-10
(f32).

## Robot policy

### π0.5

`lerobot/pi05_base` (LeRobot `pi05`). It maps camera images, the robot state and an instruction to 50 x 32 actions
in 10 flow steps, behind the policy API ([README](README.md#robot-policy)). It reads PaliGemma's tokenizer, which its
repository does not publish: `-hf` puts `tokenizer.json` of `google/paligemma-3b-pt-224` beside it. That repository
is gated, so set `HF_TOKEN` to an access token of an account that accepted PaliGemma's terms on the Hub; without one
the download stops before the weights. (A local directory needs that `tokenizer.json` beside the checkpoint.)

Reference:

```bash
xpu-serve bench -hf lerobot/pi05_base --device GPU.0 --precision f16 --cameras 3 --steps 10 --runs 10
xpu-serve serve -hf lerobot/pi05_base --device GPU.0 --cameras 2    # warm up for the first 2 cameras
```

Optimized (its q8 pack quantizes the action expert, which runs 50 tokens per flow step):

```bash
xpu-serve convert -hf lerobot/pi05_base --out ~/models/pi05_base-q8 --quant q8
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
