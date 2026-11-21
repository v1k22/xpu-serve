# 0026: Inkling 2 Mini

Status: **accepted** for the text decoder; fused kernels, the drafter, GGUF files and images follow · 2026-11-21

## Context

thinkingmachines/Inkling-2-Mini (`inkling2_loop`, 26.4B parameters, 19.5 GB as released) is a 24-layer mixture-of-
experts decoder whose layers 6 to 17 run three times with the same weights: 48 layer applications a token. It has no
rotary positions. What it has instead, per layer:

- Attention whose scores add a bias by how far back a key sits: the query's own projection gives each head a
  16-wide state, which mixes 16 stored profiles of bias over 1,024 distances. The softmax has one more logit per
  head, a sink without a value, so a head can attend to nothing. Queries and keys are clipped to an L2 norm, values
  are normed. Twenty layers attend within 1,024 tokens, every sixth over the whole context.
- A short causal convolution (four taps, depthwise, added to its input) on the key and value projections and after
  each sublayer. In the full-attention layers the taps of keys and values are per token.
- Block residuals: a layer's input is a mix, by learned queries, of the residual stream and the states at the start
  of each block of six layers. A block's first layer hands its input to those states instead of adding it back.
- Experts behind a sigmoid router with a bias that only chooses, a softcapped SwiGLU, gate and up rows alternating
  in one fused bank.
- The loop: each pass restarts from the state before the first, with KV and convolution state of its own; passes
  after the first hear the pass before through learned queries and low-rank gates on the query, key, value and
  expert inputs.

The release stores the experts of 20 layers in NVFP4 (E2M1 codes, an E4M3 scale per 16 columns, a factor per
expert) and of 4 layers in MXFP8 (E4M3 codes, a power of two per 32 columns); the rest is bf16. The reference is
transformers' `Inkling2Loop` classes.

## Decision

- **Three additions to the step graph**, each where composing existing ops would cost a kernel per term:
  - Attention takes a relative bias `[T, heads, extent]` and a sink `[T, heads]` (causal only). The bias is a table
    lookup by distance inside the softmax, so it is an operand, not a matrix of `[T, L]` scores the model builds.
  - L2Clip: a vector longer than a threshold scaled down to it (used three times a layer).
  - DynamicConv: a causal convolution with taps per token and group, over a conv slot. Its taps are indexed by how
    far back a token sits and its weight is `[taps, channels]`, as the checkpoint stores both, so the model reads
    them as they are; CausalConv keeps the torch layout it had.
- **Everything else from ops the graph had**: a block residual is a matmul by the queries, a softmax and a
  StreamMix over the stacked states; a loop pass is the same layer builders with state slots of its own
  (`layers.<i>.pass<n>`, as SenseNova Loop); the position scaling of the full layers is a host input.
- **The released experts are read by the model's loader, not by the weight store's rule for FP8 checkpoints**: the
  store opens the files as stored, and the loader pairs each bank with its companions (`pair_nvfp4`, a new `F4`
  dtype; `pair_mxfp8`, an F8 weight with powers of two as scales). A pack `convert` makes of the checkpoint holds
  plain quantized banks and needs neither.
- **The mixture-of-experts block grew three options** rather than a second block: a selection bias, alternating
  gate and up rows, and a hidden state the model computes.
- **The tokenizer is o200k_base's**: its Split regex tells letters apart by case, so the Unicode tables gained the
  cased letters (Lu and Lt, Ll), from the same generator and regex engine as the rest.

## What the tiny test model taught

The first tiny checkpoint matched transformers in f32 (cosine 1) and failed at f16 (cosine 0.6 to 0.99) with no bug
in the engine: transformers itself moved as much when its weights were rounded to f16. Its random weights amplified
an error of one part in a million a thousand times. Halving the initializer range and the spread of the zero-
initialized parameters, and a route scale of 2, brought the gain to under ten; the model then matches at f16
(cosine 0.999996 or better, greedy tokens equal). A trained checkpoint is not that sensitive; a random one has to be
made well conditioned on purpose (`make_tiny.py --arch inkling2_loop`).

## Open

- A step launches some 4,700 small kernels, which bounds decode speed: the block residual, the message gate, the
  convolution with its residual and the router are each one kernel's worth of work.
- The checkpoint's own block drafter (`mtp.safetensors`), the GGUF files (which keep NVFP4 as a block type), images,
  and KV memory for the windowed layers (40 of the 48 applications read only their last 1,024 keys).
