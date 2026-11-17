# 0022: SenseNova Loop

Status: **accepted** · 2026-11-17

## Context

SenseNova-Loop-9B (`sensenova/SenseNova-Loop-9B`, `model_type` `sensenova_loop`; its preview
`sensenova/SenseNova-Loop-9B-Preview`) is a Qwen3.5-9B text decoder (32 layers, three Gated DeltaNet layers then one
full-attention layer, hidden 4,096, an MTP head) with two changes, both written in the checkpoint's own
`modeling_sensenova_loop.py` (transformers 5.18; 5.23 no longer has a class it imports):

- **A loop** (`loop_settings`): layers 0 to 3 run once, layers 4 to 27 run `eval_count` times (2), then layers 28 to
  31. A layer's weights serve every run of it; every run has a cache of its own (its KV, convolution and recurrent
  state). 56 layer runs a token: 14 of them full attention.
- **Manifold-constrained hyper-connections** (`mhc`) over layers 4 to 27: the residual is two streams. Each of a
  layer's two branches (attention, MLP) has a router (`mhc_attn`, `mhc_ffn`: an fp32 `[8, 8192]` weight and a bias)
  over the RMS-normed, flattened streams. Its outputs give read weights `sigmoid(z[:2])` (the branch's input is the
  weighted sum of the streams), write weights `2 sigmoid(z[2:4])` (each stream takes that share of the branch's
  output) and a 2 x 2 stream mix made doubly stochastic by 20 Sinkhorn iterations in log space. The first layer copies
  the hidden state into both streams; the last averages them. These are the H_pre, H_post and H_res maps of DeepSeek's
  mHC.

## Decision

- **Part of the Qwen3.5 model code, not a folder of its own.** `qwen3_5::Config` gains the run order (`plan`, from
  `loop_settings`; empty for every other checkpoint) and the hyper-connections (`Mhc`); `Decoder::layers` walks the
  plan, a layer's later runs keeping their state slots under `layers.<i>.pass<n>`; `Decoder::mhc_layer` wraps the
  same attention and MLP builders in the routers. The language model, its MTP head, packs and the server need
  nothing else; `sensenova_loop` dispatches to `qwen3_5::load`.
- **Two graph ops.** `MhcRoute` turns a router's output (and its bias) into the three maps in one kernel: read and
  write weights and the Sinkhorn mix. `StreamMix` computes `m @ x` per token, plus the outer product of the write
  weights and the branch's output when given: the mix of the streams, and with one row of read weights the branch's
  input. The router's RMS norm and matmul stay the usual ops.
- **`convert` keeps the routers as stored.** A matmul whose output an MhcRoute reads is a router, like a mixture of
  experts' (a 64k-element matrix would stay unquantized anyway; `--min-elements 0` no longer changes that).
- **Refused by name**: `fixed_identity` (no stream mix), a loop count other than one fixed count, settings crossing
  the loop's bounds, unknown `mhc` keys, `loop_settings` or `mhc` on another model type, and a block drafter beside a
  looped model (a drafter reads the target's layers by index).
- **The vision tower is not loaded**, as for every Qwen3.5 checkpoint.

## Alternatives

- **The router's tail from existing ops plus a Sinkhorn op** (three slices, two sigmoids, a scale, the Sinkhorn): the
  first version. A decode step ran 96 Sinkhorn launches of 49 us each (one lane, a matrix indexed at run time spills)
  and four small kernels per router. `MhcRoute`, with the stream count a template parameter so the matrix stays in
  registers, took q4 decode from 29.4 to 32.6 tok/s and the graph from 3,836 to 3,068 nodes.
- **The router's matmul inside the op as well**: two fewer launches per branch (about 5 % of a q4 step by kernel
  count), but an op that reads a weight outside MatMul, with its own GEMV. Not worth the specialization now.

## What was measured

Against the checkpoint's own code in fp32 on the CPU (`tests/parity/model/hf_reference.py`), six prompts of 5 to
1,691 tokens, last-token logits, 32 greedy tokens and the MTP head's drafts (`tests/unit/model/model_parity.cpp`):

| Checkpoint | Backend | Logits cosine | Max logit error | Greedy tokens equal | MTP drafts |
|---|---|---|---|---|---|
| tiny (12 layers, 4 to 7 twice) | cpu-ref f32 | 1 | 2.2e-6 | 32 of 32, all prompts | equal |
| tiny | Arc Pro B60 f32 | 1 | 2.6e-6 | 32 of 32 | equal |
| tiny | Arc Pro B60 f16 | 0.999999 | 0.0022 | 32 of 32 | equal |
| SenseNova-Loop-9B-Preview | Arc Pro B60 f16 | 0.999997 to 1 | 0.054 | 32 of 32 (29 of 29) | equal |
| Preview, q8 pack | Arc Pro B60 f16 | 0.9996 to 0.99998 | 0.43 | 32 of 32 on five prompts, 8 on one | equal |
| Preview, q4 pack | Arc Pro B60 f16 | 0.951 to 0.997 | 3.7 | 2 to 29 | 1 to 3 of 3 |

The tiny checkpoint comes from the model's own classes (`make_tiny.py --arch sensenova_loop`), its routers as large
as trained ones so that read, write and mix weights vary by token and stream.

On SenseNova-Loop-9B-Preview, Arc Pro B60 (model_recipes.md has every number): decode 12.2 / 21.1 / 32.6 tok/s at 1k
tokens (f16 / q8 / q4), about 0.55 times Qwen3.5-9B's (56 layer runs a token instead of 32, the routers' small
kernels the rest); 64k tokens fit one 24 GB card at f16 (21.5 GiB). Read at the option letters over 300 MMLU-Pro
questions, f16 keeps all 300 of the reference's answers, q8 290, q4 229 (AWQ 252). Its MTP head takes q4 from 34 to
56 tok/s at k = 2.
