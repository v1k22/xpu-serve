# 0014: FP8 checkpoints

Status: **accepted** · 2026-11-11

## Context

Qwen releases its larger models a second time as FP8 (`Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.6-35B-A3B-FP8`; there is
none for the Qwen3.5 sizes): the same model in about half the download (31 GB instead of 56, 37 instead of 72). The
layout is transformers' fine-grained FP8, which DeepSeek-V3 introduced:

- Every projection of the decoder and of the MTP head is stored as `F8_E4M3`, a byte per weight, beside
  `<name>_scale_inv`: one bf16 scale per block of 128 rows and columns. A weight is its code's value times its
  block's scale.
- Embeddings, the head, norms, routers, the small projections of linear attention and the vision tower stay bf16.
- `config.json` names the scheme under `quantization_config` (`quant_method: fp8`, `fmt: e4m3`,
  `weight_block_size: [128, 128]`, `activation_scheme: dynamic`).
- The mixture-of-experts release stores each expert's three projections as tensors of their own
  (`mlp.experts.<e>.gate_proj.weight`: 768 tensors and as many scale grids per layer), where the bf16 release ships
  two fused banks per layer.

Enabling the bf16 releases had not enabled these. The safetensors reader refused the dtype by name, and the
mixture-of-experts one lacked the banks the model code reads.

## Decision

- **FP8 is one more way a float weight is stored, beside bf16.** `DType::F8`, and a `TensorView` whose `fp8` lists
  runs of rows (codes and their scale grid). The weight store (`SafetensorsWeights`) pairs each `F8_E4M3` matrix
  with its scales when it opens a directory; the scale tensors are not weights of their own. Nothing is decoded or
  copied at load: the codes stay in the memory-mapped files.
- **One reader for a weight's values.** `quant::read_rows` gives the float rows of a weight however it is stored
  (f32, f16, bf16, a pack's q8 / q4, f8). The CPU reference backend, the SYCL upload, `convert` and the sign sketch
  read through it, so model code and graphs do not know the format.
- **On the device an FP8 weight is f16** (f32 at f32 precision), decoded as it is uploaded, a piece of rows at a
  time. There are no FP8 kernels: an FP8 checkpoint costs on the device what its bf16 original costs, and the pack
  is the format for speed and size (q8 is a byte per weight too, with kernels that read it).
- **A bank stored expert by expert is joined, not copied.** `SafetensorsWeights::join_rows` shows F8 tensors as one
  weight whose rows are theirs, one after another; the Qwen loader names the pieces (experts in order, an expert's
  gate rows before its up rows, the layout of the fused banks). A model of 35 B parameters would otherwise need a
  30 GB copy to load. Only F8 tensors join: a dense tensor would have to be copied, no such checkpoint exists, and
  it is refused by name.
- **`convert` reads an FP8 checkpoint as it reads any other** and writes the same pack layout (fused banks, the
  same tensor names), so a pack from the FP8 release serves wherever one from the bf16 release does, drafter packs
  included. What a pack keeps of an FP8 weight is written as its f32 values; a pack holds no F8.
- **Activations are not rounded to FP8.** With `activation_scheme: dynamic` the reference runtime rounds each
  projection's input to E4M3 as it goes. Here the weights are the decoded ones and activations are f16 or f32, which
  is the float model the codes stand for and closer to the bf16 release than the reference FP8 runtime is.
- **Everything else is refused by name** (`cfg::check_quantization`, the weight store): another `quant_method`
  (bitsandbytes, AWQ, GPTQ), another block size or code format, a static activation scheme, codes without scales or
  with a grid of another shape.

## What was measured

**Tiny checkpoints against Hugging Face.** `tests/parity/model/to_fp8.py` rewrites a checkpoint in this layout
(torch rounds to `float8_e4m3fn`) and writes the model those bytes decode to; the reference outputs of the second
are the fixtures for the first. Tiny Qwen3.5, dense and mixture of experts, each with an MTP head (32 and 152 FP8
tensors): on the CPU reference backend at f32 the logits match to 1.4e-6 with greedy output and MTP drafts equal;
the Arc Pro B60 passes at f32 and f16. The control fails as it must: the original tiny checkpoint against the
decoded fixtures has cosine 0.992.

**The releases against the bf16 ones**, tensor by tensor with torch alone (code times block scale against the
original's values; an expert's projection against its rows of the fused bank):

| Release | FP8 tensors | Cosine with the original (min) | Relative error | Other tensors |
|---|---|---|---|---|
| Qwen3.8-27B-FP8 | 407 | 0.999648 | 2.65 % | 792, all equal |
| Qwen3.6-35B-A3B-FP8 | 31,745 | 0.999645 | 2.66 % | 706, all equal |

2.65 % is what three mantissa bits leave. The pairing of an expert's tensors with the bank is the one that fits:
against the same expert's up rows a gate projection has cosine -0.01, against the next expert's gate rows 0.0005.

**q4 packs (g32, round to nearest) from the FP8 release and from the bf16 release**, Arc Pro B60:

| | Qwen3.8-27B, from bf16 | from FP8 | Qwen3.6-35B-A3B, from bf16 | from FP8 |
|---|---|---|---|---|
| `convert` | | 64 s, 18.59 GB | | 68 s, 23.34 GB |
| Decode, six chat prompts | 22.5 tok/s | 22.5 | 61.8 | 61.8 |
| With the MTP head (k = 4; 3 for the MoE) | 50.5 (3.20 tokens per verify) | 50.8 (3.20) | 94.3 (2.79) | 95.6 (2.83) |
| With a DFlash 2 drafter (k = 7) | 60.7 (3.88) | 60.5 (3.85) | | |
| Last-token logits against Hugging Face fp32, cosine over six prompts | 0.67 to 0.996 | 0.71 to 0.996 | 0.92 to 0.98 | 0.91 to 0.98 |
| MMLU-Pro, 25 questions, thinking on, the MTP head drafting | 76 % | 76 % | 80 % | 76 % |

One MMLU-Pro question is 4 points: the 35B-A3B packs are one question apart, and the pack from bf16 gave 84 % on
the same questions in an earlier build with a g64 pack, so this sample does not separate the two sources. Three
answers of the 27B pack from bf16 and one of the pack from FP8 ran into the 4,096-token limit.

The two packs of a model hold the same tensors in the same formats (1,199 for the 27B, 1,045 for the 35B-A3B). The
unquantized ones are identical (456 and 370), and the quantized ones (410 and 342, the expert banks among them) agree
to cosine 0.9945 or better; a misplaced expert or a wrong scale grid would show as none. The drafter packs made for
the bf16-derived 27B pack serve the FP8-derived one unchanged. Against Hugging Face a q4 pack's greedy text holds for
32 tokens on some prompts and leaves early on others, whichever release it was made from (35B-A3B: four of six
prompts from bf16, two of six from FP8, with 31, 17 and 8 tokens on three others).

**An FP8 checkpoint served as it is, at real size.** Neither release fits a 24 GB card unquantized, so Qwen3.5-4B
was rewritten with `to_fp8.py` (207 FP8 tensors; 5.3 GB instead of 8.8). On the Arc Pro B60 at f16, against Hugging
Face fp32 on the decoded twin: last-token logits cosine 0.999997 or better on six prompts of up to 1,691 tokens,
32 greedy tokens and the MTP head's drafts equal on all six. The bf16 original fails against those fixtures (cosine
0.987 to 0.9995), as it must. Decode is 43.5 tok/s with 7.8 GiB on the device, as for the original; loading takes
14 s instead of 9, the decode on the host.

Tests: `tests/unit/test_fp8.cpp` (the code table, pairing, joining, refusals), `tests/unit/model/fp8_checkpoint.cpp`
(a tiny checkpoint written both ways: equal weights, equal logits on cpu-ref and bit for bit on the GPU at f16 and
f32, equal packs). The GPU suites pass at f16 and f32 (339 cases).

## Consequences

- An FP8 checkpoint needs no step of its own: `run`, `serve`, `bench` and `convert` take its directory. Whether it
  fits a device is decided as for the bf16 release (the two above do not fit a 24 GB card unquantized), so in
  practice the FP8 release is the smaller download to convert from.
- Loading one directly decodes every weight on the host once per load.
- An FP8 weight with native kernels (a byte per weight on the device, no conversion) is not planned: q8 covers it.
