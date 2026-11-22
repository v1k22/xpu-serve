# 0026: Inkling 2 Mini

Status: **accepted** for the text decoder, its GGUF files and its drafter; fused small kernels and images follow ·
2026-11-21

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

## The GGUF files

Unsloth's files (`general.architecture` `inkling2`, their llama.cpp builds' layout) import into a pack like any other
GGUF (`src/convert/gguf/inkling2.cpp`). What was particular:

- **Metadata per layer visit.** The file counts 48 blocks of metadata (a looped layer once per pass) over 24 blocks
  of tensors. The importer reads the loop back out of those arrays and refuses a file whose passes disagree.
- **Float tensors the converter moved.** It transposes the relative bias projection and stores the dynamic
  convolutions' taps flipped and transposed. The importer puts the values back (`FloatReorder`: a whole float tensor
  in another order, written as F32), so the model code reads one layout.
- **Expert banks stay in the file's format**, NVFP4 included (a block format of its own: 36 bytes for 64 weights).
  Two things differ from the checkpoint and cannot be undone on a quantized tensor, so the model reads either form:
  the fused gate and up bank holds the gate rows and then the up rows (`gate_up_rows: halves` in config.json), and
  a bank's factor per expert is a tensor beside it (`w13_weight_factor`, `w2_weight_factor`) that multiplies the
  expert's output (`MoeParts::gate_up_scale`, `expert_scale`).
- **The lossless file is the proof.** UD-Q8_K_XL holds the checkpoint's own values; its pack matches Hugging Face
  fp32 as the checkpoint does (cosine 1, greedy tokens equal), which checks every name, shape and reorder.

## The drafter (DSpark)

The checkpoint ships a block drafter (`mtp.safetensors`, `mtp_config`): three decoder layers that fill a block of 8
(the pending token and 7 mask rows) in one pass, and a recurrent head that then picks the 7 drafts one after
another. It is not DFlash's shape, so it is a drafter of its own (`src/models/drafters/dspark.cpp`) behind the same
`BlockDrafting` interface:

- **Context per layer.** Each drafter layer mixes four of the target's states with its own weights (the softmax of
  its row of `fuse_weight`), norms the mix and projects it to keys and values with projections kept for that. The
  states are the stream entering three layer visits (a looped layer has a visit per pass) and the state the output
  layer reads; the target's prefill and verify programs append them, as for DFlash.
- **A window anchored at the block.** A local layer's block sees itself and the 1,024 positions before its first
  row, the same keys for every row. No setting of the per-row window gives that set, so attention gained a mode
  (`AttentionAttrs::anchored`).
- **One row through the host.** The head also reads its own mix of the target's states at the last position the
  target ran. That row depends on which drafts a verify accepted, so the target's programs output it for every row
  (`draft_context`) and the engine's block runner hands the last kept one to the draft program
  (`BlockDrafting::target_rows`).
- **The head in the graph.** Each draft needs the token before it, so the steps are unrolled: a gated recurrent
  step, a correction added to the block's row, the target's output layer, an argmax. Each step reads the whole
  output layer: the MTP heads' shortlist (64 rows a one-bit sketch picks) lowered the first draft's acceptance from
  63 % to 56 % on the six chat prompts, more than its time saved.
- **The block is always 8 rows.** Its layers see the whole block, so a shorter one would be other hidden states;
  `--spec-k` below 7 runs fewer head steps over the same block.
- **No public module to compare with.** The reference is the math written out in float64 from llama.cpp's and
  vLLM's ports (`tests/parity/model/dspark_reference.py`) over transformers' target; on the released weights the
  check is the acceptance rate.

In a GGUF repository the drafter is a file of its own (`dflash` with `dflash.dspark.markov_rank`); `-hf` fetches it
with the model file and the import puts its tensors beside the model's, as a pack of the checkpoint holds them.

## Images

The vision tower (transformers' Inkling2VisionModel; `src/models/inkling2/inkling2_vision.cpp`) is a plain pre-norm
encoder: patches of 20 pixels through a linear layer, a position table interpolated to the image's grid, 2-D rotary
attention, layer scale, and a two-layer projection of each square of four patches to one row of the decoder. Its
processor does not resize: an image is padded to whole squares of 40 pixels, one token each. Two things were
particular:

- **A class token beside the patches.** The other towers run over an image's tokens with their patches as rows, and
  a class token is no token's patch. It is one more block of rows, the class token in its first and padding in the
  rest, which no query sees: attention over rows of several a token takes a key bias a row. The block comes last, so
  the program's leading rows are the image's.
- **Rotary pairs as neighbours.** The tower rotates channels (2 i, 2 i + 1); the graph's Rope pairs c with c + D / 2.
  Queries and keys go through it with their channels in that order, which their products do not see.

A GGUF repository keeps the tower in a file of its own (llama.cpp's mmproj: `clip`, projector type `inkling2v`).
`convert --mmproj` adds it to the model's pack under the checkpoint's names, and `-hf` fetches it with the model
file; its tensors are the checkpoint's, the patch embedding stored as a convolution's weight and put back.

An image's rows stand in place of its tokens' embeddings after the embedding norm, as the reference scatters them.
On a tiny checkpoint the processor's pixels are equal, the tower's rows and the prompt's logits match at cosine 1
(cpu-ref and the GPU), and greedy output is the reference's.

## KV memory

Forty of the 48 layer applications attend within 1,024 tokens, yet their KV slots grew with the context. A session
with conv or recurrent state can never be cut back before its last step, so no later step reads a windowed slot's
older keys: the SYCL backend now keeps only the window for such slots (docs: CLAUDE.md section 6). Inkling 2's KV
then grows by 8 full-attention slots a token instead of 48 (and one more with the drafter).

## Kernels for its files and its long contexts

Three backend changes the measurements asked for, each general:

- **Float expert banks on the grouped GEMM.** The lossless file keeps four layers' experts in bf16 (the release's
  MXFP8 layers), which the device holds as f16. An f16 bank took one GEMV per (token, expert) pair at every size,
  reading an expert's weights once for each of its tokens; from 256 pairs it now takes the grouped GEMM the quantized
  banks take, its rows read in place. The file's 512-token prefill went from 361 to 590 tok/s on the Arc Pro B60 and
  from 112 to 210 on the Arc B390.
- **The relative bias and the sink in the fused attention kernel.** They took the GEMM path, whose scores for a chunk
  of queries grow with the context. The flash kernel adds the bias within its extent and starts a row's online
  softmax at the sink (in the first split of the keys alone). A 130,816-token prefill went from 422 to 592 tok/s.
- **A long quantized cache by chunks.** For the matrix paths every key and value in view was decoded to f16 first:
  4 GiB at a million tokens, more than a `turbo3` cache of that length. Past 65,536 keys the fused kernel now reads
  one decoded chunk at a time and merges each chunk's partial softmax into the rows'.

## What was measured

The variants, the commands and every number are in model_recipes.md (Inkling-2-Mini, Inkling-2-Mini GGUF, DSpark).
In short: the lossless file matches Hugging Face fp32 (cosine 0.9999) and is the closest to the release on either
device; UD-Q4_K_M is within 0.3 % of its perplexity at 15.4 GiB; the q4 pack of the checkpoint is the fastest (+1.6 %).
With the windowed slots and the chunked cache, the model's whole context (1,048,576 tokens) fits the B60 and the
B390 from UD-Q4_K_M in `turbo3`, which costs 2.0 % of perplexity (`i8` 0.15 %, `turbo4` 0.28 %).
Accuracy is perplexity over wikitext-2 against the lossless file: `--readout` reads no answer letter from this
model (below), so the multiple-choice read-out the other GGUF files were scored with is not available.

## Open

- A step launches some 4,700 small kernels, which bounds decode speed: the block residual, the message gate, the
  convolution with its residual and the router are each one kernel's worth of work. On the B60 every GGUF file
  decodes at 35 to 37 tok/s whatever its size.
- `--readout`: the template ends without a generation prompt and the model writes its own `<|message_model|>` and
  content tag, so the read-out sees a structural token, not an answer letter. (Chat is parsed: the `inkling2`
  dialect splits the model's messages into reasoning, content and tool calls.)
- A verify of several rows, and the drafter's block, read the whole KV cache once per row at long contexts: DSpark
  gains at 32k tokens and loses at 128k.
- The files in mainline llama.cpp's layout (`inkling2_loop`), which are other names for the same tensors.
