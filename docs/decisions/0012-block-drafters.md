# 0012: block drafters (DFlash, DFlash 2, DSpine)

Status: **accepted** · 2026-11-11

## Context

An MTP head drafts one token per run, each from the one before. A block drafter is a second small model (five
decoder layers of the target's width for Qwen3.8-27B, about 2 B parameters) that fills a block of mask tokens after
the pending token in one pass, reading the target through the hidden states of a few of its layers. Three released
kinds share that backbone: DFlash (z-lab), DFlash 2 (z-lab: dynamic convolutions and a candidate selector) and DSpine
(Tencent: a code bridge between positions and a refined last write). Checkpoints exist for Qwen3.8-27B
(`z-lab/Qwen3.8-27B-DFlash2`, block 8; `tencent/Qwen3.8-27B-DSpine-b16`, block 16).

They do not fit the drafter interface the MTP heads and Gemma assistants use (pairs of token and hidden state in,
one token out, chained), and they are not a part of any one model: the same drafter code serves whatever target a
checkpoint was trained for.

## Decision

- **Two ways of drafting behind one engine interface.** `Speculation` (models/model.h) offers `chain()`
  (`ChainDrafting`: MTP heads, assistants) or `block()` (`BlockDrafting`). The engine drives `spec::Drafter`
  (engine/spec/drafter.h), with one runner per way (`ChainDrafter`, `BlockDrafter`); the loop, the verify, the
  acceptance and the rollback are the same for all drafters.
- **The target's programs write the drafter's context.** The context (the target's layer states through `fc` and
  `hidden_norm`, then each drafter layer's key and value projections) depends on the target alone. The prefill and
  verify programs compute it for the tokens they run and append it to KV slots that the draft program shares by
  label. Nothing is handed over on the host, and cutting a verify back to its accepted tokens cuts the context with
  it (`Session::truncate` on the verify program). The draft program appends its block to the same slots, attends,
  and the runner truncates them back.
- **The drafter lives in `models/drafters/dflash.{h,cpp}`**, built from `models/common/blocks`. A model that takes one
  (Qwen3.5 so far) passes it its layer states, its embedding table and its head. `LoadOptions::drafter` names the
  checkpoint; its weights sit under `drafter.` beside the target's (`JoinedWeights`).
- **What the block program gives.** DFlash: `drafts` [n]. DFlash 2 and DSpine choose each draft given the one before:
  the program gives `candidates` [n, K] and `scores` [n, K, K] (the score of candidate k of a draft for each
  candidate j the draft before could have been), and the host follows the best of each row in turn. That keeps the
  serial choice (a handful of comparisons) off the device while every tensor is computed in one run.
- **A causal block runs only the rows asked for.** DSpine's rows do not see the rows after them, so a round of k
  drafts runs k + 1 rows. A block that sees itself whole (DFlash, DFlash 2) always runs the rows it was trained
  with, and the first k drafts are used.
- **The drafts default to 7** for a block drafter: a verify of up to 8 tokens runs on the matrix-engine GEMV, whose
  cost hardly grows with the rows; a verify of 9 or more does not (a 16-row verify of Qwen3.8-27B on the Arc Pro
  B60: 117 ms, against 57 ms at 8 rows).
- **A drafter checkpoint converts to a pack of its own**: `convert <drafter> --target <model> --out <dir>` loads the
  target with the drafter and lets the programs that read the drafter's weights decide their formats, by the rule
  every pack follows (docs/decisions/0009). A block is bound by reading the drafter: at q4 it is 1.3 to 1.4 GB
  instead of 4 GB.
- **Two additions to the graph**: `PrevRow` (row t is row t - 1 of a step: a block's convolution taps and the code a
  position receives from the one before), and attention that is not causal with a sliding window (a block that sees
  itself whole and a window of the context before it).

## What was measured

Against the references (z-lab's torch model; DSpine's own plain-torch bridge and refinement references), on tiny
random checkpoints in the released layout: every draft and candidate equal, scores within 1e-5 (f32), over four
rounds per prompt with contexts that grow by several tokens a round and pass the drafter's window
(`block_drafter_parity.cpp`). One thing the released configs never use came out of it: Hugging Face's hidden state
after the last layer is the normed one.

On the real checkpoints the references (fp32 target on a CPU, 128 tokens per prompt over the six bench prompts)
accept 3.81 tokens per round for DFlash 2 and 3.46 for DSpine; this engine with the q4 target accepts 3.92 (7
drafts) and 3.53 (15 drafts). DSpine's published lead over DFlash 2 is on reasoning-heavy and code benchmarks with
thinking on; on these prompts (thinking off) its longer block does not pay for itself.

Speeds are in docs/design-spec-decode.md, section 6.2.

## Consequences

- A drafter must match its target (hidden size, vocabulary, layer count); a mismatch is refused when the model loads.
- `convert` keeps DSpine's stacked bridge matrices as they are (they are read one matrix at a time); its codebook and
  last read weight follow the rule (q8 as a gather table, q4 as a matmul weight) with no measured loss of acceptance.
- Sampling uses the greedy block as the proposal with rejection sampling, as for every drafter; the references'
  sampled chains (a draw per position from the candidates) are not implemented.
- A verify of 9 to 16 rows needs the matrix-engine GEMV extended past 8 rows before a DSpine block of 16 pays off.
