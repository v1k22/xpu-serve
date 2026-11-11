# Design: speculative decoding with a model's drafter

Status: **implemented** · 2026-11-11 (measured speeds in section 6)

Speculative decoding for every model that has a drafter: the MTP head inside Qwen3.5-family checkpoints
(Qwen3.8-27B, Qwen3.6-35B-A3B), the Gemma 4 "assistant" drafters (`google/gemma-4-*-assistant`, one per Gemma 4
size), and block drafters trained for a target (DFlash, DFlash 2, DSpine: section 1.1). One loop in the engine, one
drafter interface, the model-specific wiring in each model's folder (DESIGN.md section 10). This document records how
the backend, the models and the engine changed.

## 1. The two drafter families

| | Qwen3.5 family MTP head | Gemma 4 assistant |
|---|---|---|
| Where | `mtp.*` in the target checkpoint (`mtp_num_hidden_layers: 1`) | separate checkpoint `<target>-assistant` (`gemma4_assistant`) |
| Input | `fc(cat(norm_e(embed(t)), norm_h(h)))`, 2H -> H; `h` = the target's hidden after its final norm | `pre_projection(cat(embed(t) * sqrt(H), h))`, 2H -> 1024; same `h` |
| Layers | one full-attention decoder layer of the target's kind (gated attention; dense MLP on 27B, MoE on 35B-A3B) | 4 layers (3 sliding, 1 full), **query-only attention over the target's KV** (the last sliding and the last full layer), GeGLU MLP |
| Own KV | yes: one slot over the positions it has seen | none |
| Head | the target's `lm_head` | its own tied head through a centroid lookup (top 32 of 2048 centroids, then those tokens' logits) |
| Next hidden (chained drafts) | its own output after `norm` | `post_projection(norm(x))`, back to the target's width |
| Reference | vLLM `qwen3_5_mtp.py` (transformers skips `mtp.*`) | transformers `Gemma4AssistantForCausalLM` + `generate(assistant_model=...)` |

### 1.1 Block drafters

A block drafter is a separate checkpoint of a few decoder layers at the target's width. One pass fills a block of
mask tokens after the pending token; the layers attend to the block and to a context made of the target's hidden
states (the states after a few of its layers, concatenated, through `fc` and a norm). docs/decisions/0012 records the
design.

| | DFlash | DFlash 2 | DSpine |
|---|---|---|---|
| Checkpoint for Qwen3.8-27B | none released | `z-lab/Qwen3.8-27B-DFlash2`, 3.85 GB bf16 | `tencent/Qwen3.8-27B-DSpine-b16`, 4.05 GB bf16 |
| Block | 16 | 8 (7 drafts) | 16 (15 drafts) |
| A block row sees | the whole block | the whole block; context within a window of 2,048 | the rows before it; a window of 8,192 |
| Beside the layers | | a two-tap convolution over the block's rows before and after attention and MLP, its taps predicted per row | after every layer each row receives a code of what the row before predicts (the code bridge) |
| Choosing the drafts | each row's argmax | 16 candidates a row; a low-rank term scores each against the token chosen before | 16 candidates a row; the last layer's write is redone with the token actually chosen before |
| Reference | `github.com/z-lab/dflash` | the same | `github.com/Tencent/DSpine` |

## 2. What the references accept

Greedy, 6 prompts (code, explanation, word problem, story, summary, translation; chat template, no thinking), 256 new
tokens each, on the Arc Pro B60 host's CPU. Tokens per round = tokens emitted per target verify (accepted drafts + the
target's own token), the number speculation multiplies decode by before costs.

- Qwen: the target's greedy decode in transformers (bf16; fp32 for the MoE, whose bf16 path on the CPU is slow), its
  post-norm hidden states from one pass, and the MTP head built from transformers' own Qwen3.5 layer classes with the
  `mtp.*` weights, wired as vLLM does. Greedy decoding emits the target's tokens whatever the drafts are, so drafting k
  tokens at every round start and accepting the longest prefix equal to the target's tokens gives the acceptance exactly.
- Gemma: transformers' `generate(assistant_model=...)` with a constant k (and Google's heuristic schedule), fp32;
  acceptance from the number of target forwards. Every assisted output equalled plain greedy decoding.

| Model | Drafter | k = 1 | k = 2 | k = 3 | k = 4 | per prompt at k = 3 |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | MTP head (1 layer) | 1.85 | 2.46 | 2.86 | 3.15 | 2.22 (story) to 3.75 (code) |
| Qwen3.6-35B-A3B | MTP head (1 MoE layer) | 1.86 | 2.49 | 2.91 | 3.19 | 2.28 (story) to 3.70 (code) |
| Gemma 4 E2B | assistant (4 layers, 256 wide) | 1.62 | 2.02 | 2.27 | 2.36 | |
| Gemma 4 26B-A4B | assistant (4 layers, 1024 wide) | 1.81 | 2.43 | 2.84 | 3.15 | 2.26 (story) to 3.70 (word problem) |

The first draft is right 73 to 97 % of the time on Qwen3.8 depending on the text; chained drafts degrade slowly (the one
MTP layer is reused for every depth). Google's heuristic schedule settled at k = 4 on these prompts.

## 3. What has to change, and why it is a refactor

Three things the current architecture cannot express:

1. **Several programs over one sequence.** A `Session` today runs one graph and owns that graph's state. Speculative
   decoding runs three programs over the same sequence: the target's step (prefill, decode), the target's verify
   (logits for every position plus the final hidden states) and the drafter. They share weights (the MTP head reads the
   target's embedding table and `lm_head`), and the Gemma assistant reads the target's KV cache.
2. **Rolling back recurrent state.** `Session::truncate` refuses graphs with conv or recurrent state. Both Qwen targets
   have Gated DeltaNet layers: after a verify of k+1 tokens accepts j + 1 of them, the delta-rule state and the conv
   window must be those after token j + 1, not after k + 1.
3. **A program reading another's KV without appending** (the assistant's query-only attention).

### 3.1 Backend: a session holds programs and shared state

```cpp
// backend.h
std::unique_ptr<Session> Backend::compile(std::span<const graph::Graph* const> programs, const WeightStore& weights,
                                          const SessionOptions& options);
void Session::run(size_t program, const TensorMap& inputs, TensorMap& outputs);
void Session::truncate(size_t program, int64_t n);
```

- Weights are uploaded once per session and shared by name across its programs.
- State slots are shared **by label**: programs that declare a slot with the same label use the same state (a shape
  mismatch is refused at compile). Each KV slot keeps its own length; a program's kL is the length of the slots it
  appends to plus T (or, for a program that only reads, the length of the slot it reads).
- New op `KvRead(slot)` -> `(k_all, v_all)`: another program's KV slot, read only (the assistant's attention). A graph
  reads all its KV slots or appends to all of them.
- The single-graph `compile` stays as the one-program case; nothing else changes for models without a drafter.

### 3.2 Truncating inside a short step

- `SessionOptions::rewind_tokens` (= k + 1 when speculating): for a step of up to that many tokens, the conv and
  recurrent ops keep the step's inputs (conv: x; gated delta: k, v, g, beta), and the state from before the step is
  the other half of the double buffer the backend already keeps.
- `Session::truncate(program, n)` with n inside that step: KV lengths shrink; conv and recurrent states are rebuilt
  from the state before the step by replaying the first n - base kept inputs (the delta rule over m tokens, the conv
  window from the kept rows). Every slot is checked before any changes, so a refused count changes nothing. Memory:
  (k + 1) rows of each op's inputs.

### 3.3 Models

- `Model::speculation()` returns a `Speculation` (models/model.h) when the model was loaded with
  `LoadOptions::speculation` and has a drafter: the target's prefill program ("logits" of the last row), its verify
  program ("logits" of every row), the draft program, and the drafter's way of drafting, one of two. A chain drafter
  (`ChainDrafting`: MTP heads, assistants) proposes a token a run from pairs of a token and the target's post-norm
  "hidden" state, which the prefill and verify programs then also output: `prepare_draft(tokens, hidden, start_pos,
  depth)`, `drafter_keeps_state()`. A block drafter (`BlockDrafting`) proposes a block a run:
  `prepare_block(token, position)`; it is handed nothing, because the target's programs append what it reads of them
  to its own state slots.
- Qwen3.5 (`qwen3_5_model.cpp`): one set of layer builders (weight prefix and state label as parameters) makes the
  step, prefill and verify graphs and the MTP head, so dense and MoE heads come out of the same code; the head's KV
  slot is `mtp.layers.0`. A pair (h_i, token i + 1) runs at position i; chained drafts advance one position each.
- Block drafters (`models/drafters/dflash.cpp`, one class for the three kinds): `append_context` adds the context's
  keys and values to a prefill or verify graph (the model passes its layer states), `build_block` builds the draft
  program over the model's embedding table and head. It outputs "drafts", or "candidates" and "scores" for the kinds
  that choose each draft given the one before. Qwen3.5 takes one with `--drafter` in place of its MTP head; its
  weights sit under `drafter.` beside the target's.
- Gemma 4 (`gemma4_model.cpp`): the assistant from its checkpoint (`--drafter <dir>`, its weights under `assistant.`
  through `JoinedWeights`): query-only layers reading the target's last sliding / full KV slots through `KvRead`,
  the centroid head built from TopK, an i32 Gather of the cluster's tokens, Gather and MatMul (outputs "logits" over
  the candidates and "candidates"). Every draft of a round runs at the position of the token it starts from.

### 3.4 Engine

- `engine/spec/drafter.h`: the drafter side as the engine sees it (`spec::Drafter`: what it reads of the target's
  outputs, `add`, `propose`, `truncate`), and the session's program order (`spec::Program`).
- `engine/spec/chain_drafter`: pairs wait until the drafter takes them; a drafter that keeps state takes every pair
  once, in order (prefill chunks flush, a round's first step catches up on the pairs the last verify accepted), its
  chained drafts are truncated away after each round.
- `engine/spec/block_drafter`: one run of the draft program at the pending token's position, its rows truncated
  away; the drafts are the program's, or the path through its candidates that takes the best score of each row.
- `Engine::spec_round`: k drafts, one verify of the pending token and the drafts, acceptance through
  `Sampler::verify` (greedy: the argmax; sampling: rejection sampling against the point-mass draft, then the
  residual, so the output distribution is exactly the target's), truncation of the target to the accepted point.
  Rounds that would pass the context or the requested length draft fewer tokens.
- Stats: verify rounds, drafted and accepted tokens in `GenerateStats` (`timings` in API responses, `run` / `bench`).

### 3.5 Flags and packs

- `--spec off|mtp|dflash|dspine` (default `off`; `dflash` takes a DFlash or DFlash 2 checkpoint), `--spec-k N`
  (default 3, or 7 of a block drafter's block), `--drafter <dir>` on `run`, `serve` and `bench`; `bench --prompts
  <file>` measures over real prompts (acceptance depends on the text).
- `convert` plans the drafter's graph too: `mtp.*` is quantized by the same rule as the target's matmuls. A drafter
  checkpoint beside a model converts to a pack of its own, `convert <drafter> --target <model> --out <dir>`: the
  programs that read it decide its formats (a block drafter's matmul weights q4, its token tables q8).

## 4. Prerequisite: Gemma 4 26B-A4B

Enabled for text: its MoE blocks (`enable_moe_block`: 128 experts, top 8, beside the dense MLP; the router reads the
normed residual and scales each expert's weight after normalizing, `blocks::moe` `router_x` / `expert_scale`),
`attention_k_eq_v` (global layers' values are the key projection, `AttentionParts::v` empty) with their own KV heads
(`num_global_key_value_heads`, or transformers' newer `per_layer_config`). Image tokens attend both ways in this model
(`use_bidirectional_attention: vision`), not supported yet: it loads text only. Checked against Hugging Face on a tiny
checkpoint of its layout (`make_tiny.py --arch gemma4_moe`).

## 5. Tests

- Sessions (cpu-ref and SYCL, `session_cases.h`): truncate inside a short step equals a session that never ran the
  dropped tokens (CausalConv, GatedDelta, KV); refusals change nothing; programs share KV by label; `KvRead`.
- Engine (`tests/unit/engine/spec_engine.cpp`): a tiny model with every kind of state and a drafter that is right
  about half the time: speculation emits exactly the greedy tokens, through prefix reuse, chunked prefill, stop tokens
  and rounds cut short. `sampler.cpp`: rejection sampling draws the target's distribution, whatever the draft.
- Drafters against PyTorch (`model_parity.cpp`, fixtures from `hf_reference.py`): the MTP heads of tiny `qwen3_5`
  dense and MoE checkpoints (`make_tiny.py` adds them) and the real Gemma 4 E2B assistant (`--assistant`): logits and
  the three chained drafts after each prompt.
- Block drafters against PyTorch (`block_drafter_parity.cpp`, fixtures from `block_drafter_reference.py`): tiny
  DFlash, DFlash 2 and DSpine checkpoints for the tiny `qwen3_5` (`make_tiny.py --drafter`), four rounds per prompt
  with contexts that pass the drafter's window: drafts, candidates and scores. Without a reference
  (`block_drafters.cpp`, checkpoints written by the test): a context fed in chunks or cut back after a verify is the
  same context, a causal block stops after the drafts asked for, a drafter's pack drafts like its checkpoint.
- Real models: greedy output with speculation equals output without (`run`, then the bench prompts).

## 6. Measured speed (Arc Pro B60, batch 1; the Arc B390 in 6.1, block drafters in 6.2)

Greedy decode over the six prompts of section 2 (chat template, thinking off, up to 256 new tokens each, stopping at
the end of the turn), `xpu-serve bench <pack> --prompts <file> --spec off|mtp --spec-k k`; decode tok/s counts the
tokens after the first. q4 packs (g32, round to nearest; the MTP heads quantized like the target); the Gemma
assistants run from their bf16 checkpoints.

| Model | spec off | k = 1 | k = 2 | k = 3 | k = 4 | best |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 22.5 | 35.7 (1.85) | 42.3 (2.46) | 47.6 (2.92) | **49.5** (3.20) | 2.20x |
| Qwen3.6-35B-A3B | 61.6 | 84.9 (1.84) | 94.1 (2.43) | **94.4** (2.83) | 92.5 (3.09) | 1.53x |
| Gemma 4 E2B + assistant | 139.9 | **173.7** (1.65) | 169.5 (2.02) | 169.2 (2.27) | 163.1 (2.40) | 1.24x |
| Gemma 4 26B-A4B + assistant | 67.2 | 83.6 (1.81) | **91.4** (2.41) | 82.6 (2.83) | 82.4 (3.17) | 1.36x |
| Qwen3.5-9B | 62.1 | 91.8 (1.80) | 101.9 (2.36) | 106.3 (2.71) | **107.7** (2.91) | 1.73x |
| Qwen3.5-4B | 104.0 | 137.1 (1.79) | **149.1** (2.30) | 142.0 (2.63) | 135.7 (2.77) | 1.43x |
| Gemma 4 E4B + assistant | 84.2 | 118.5 (1.66) | **126.4** (2.07) | 126.5 (2.33) | 123.2 (2.51) | 1.50x |

tok/s, with tokens per verify in parentheses; they follow the references of section 2. Qwen3.8 at k = 5 / 6: 49.1 /
48.6 (3.35 / 3.48). The last three rows were added on 2026-11-11 (Qwen3.5-9B at k = 5: 99.5); a single run of this
bench moves by about 4 %, so the best k and its neighbours are medians of three runs there, and E4B's k = 2 and k = 3
are a tie. Greedy output with k = 2 equals plain decoding prompt by prompt on
Qwen3.8, Qwen3.6 and E2B (6 of 6 each) and on 5 of 6 for 26B-A4B: its story diverges after 673 of 902 bytes ("It
wasn't painting a perfect circle" / "It wasn't a perfect circle"), the kind of near tie a wider verify (another
kernel path at f16) can flip (DESIGN.md 10.4; the margin was not measured).

`bench` prints where a round's time goes (a draft, a verify, a round). Qwen3.8 at k = 4: a draft 2.6 ms, a verify
53.4 ms, a round 64.6 ms. What the round costs, and what brought it there (each from its own measurement):

- A verify of T tokens costs more than a decode step (44.4 ms). The GEMV's arithmetic grows with the rows; its
  matrix-engine version (`gemv_quant_xmx`, from 4 rows) costs about the same at any row count. Its sub-groups used to
  convert x to f16 and sum it per group (a DPAS by ones) themselves: one small launch before now does both for every
  weight that reads the same x (an isolated chain of MLP-sized matmuls at 5 rows, Arc B390: 8.71 to 8.22 ms, the
  one-row GEMV 7.66).
- Greedy decoding read every verify row's logits back and searched them on the host: 5 MB over PCIe and five scans
  of 248k floats, about 10 ms a verify. The target programs now also output each row's `argmax`, a run returns just
  the outputs it is asked for (`Session::run(..., only)`) and skips what neither they nor the state need: greedy
  decoding without penalties reads a token per row (Qwen3.8 k = 4: a verify 62.7 to 53.4 ms; spec off 22.2 to 22.5,
  E2B 127 to 140).
- A draft step of an MTP head was mostly the target's lm_head (0.79 GB over 248k tokens). It now scores every row
  with a one-bit sketch of the head and computes the exact logits of the 64 best (docs/decisions/0011): 3.6 to 2.6
  ms a draft, with 1,038 of 1,876 drafts accepted against 1,040 of 1,872.
- A verify keeps its conv and recurrent inputs so that a rejected draft can be rolled back: one launch a slot (the
  driver's memcpy per input was 5 ms a verify on the Arc B390).
- MoE targets (Qwen3.6, 26B-A4B): a verify of k + 1 tokens reads the union of their experts, so it grows faster with
  k than a dense model's (Qwen3.6: 19.4 / 21.8 / 24.2 / 26.0 ms at k = 1..4). Up to 64 (token, expert) pairs (a
  verify of up to 8 tokens with eight experts each) run one GEMV per pair.
- The E2B target's step is short (about 7 ms), so its rounds are bound by per-program host overhead (several programs
  per round, each waiting for its outputs); k = 1 is best there.
- Adaptive k (DESIGN.md 10.1) is not done: `--spec-k` is fixed (3 by default); dense Qwen3.8 does best at 4 or 5, the
  MoE targets at 2 or 3, E2B at 1.

### 6.1 Arc B390

Qwen3.8-27B on the Arc B390 (integrated; reads stream at about 115 GB/s, measured, of 153.6 theoretical), the same
prompts and bench; tok/s with tokens per verify in parentheses:

| Pack | spec off | k = 3 | k = 4 | k = 5 | best |
|---|---|---|---|---|---|
| q4 g32 | 6.9 | | 17.6 (3.20) | **18.0** to 18.4 (3.35) | 2.6x |
| q4 g128 | 7.8 | 18.7 (2.77) | 19.6 (3.03) | **19.8** (3.14) | 2.5x |

(Before the work of this section's list: 15.9 at g32, 16.2 at g128. The laptop's numbers move 2 to 3% with how long
it has been under load: the driver reports the GPU throttled by its sustained power limit, 2.0 to 2.2 of 2.5 GHz.)

- A decode step runs at the memory's speed (16.1 GB of g32 matmul weights in 144 ms is 112 GB/s): every further
  gain is fewer bytes read per emitted token, or less time not reading. A round at g32, k = 5 (181 ms for 3.35
  tokens): a verify of 6 rows 158 ms, five drafts of 4.3 ms each.
- The verify's 14 ms over a decode step: about 7 ms the matrix-engine GEMV (its launches cost 13 to 29 us more than
  the one-row GEMV's, of which the launch that prepares x is 7), 3 ms the rows in the other ops, 1 ms the kept
  inputs. Sleeping through a step instead of polling for its end left the GPU more of the power budget (2.36 GHz, 40%
  of a core instead of 103%) and changed nothing: the step waits on the memory.
- A draft: the MTP layer's 0.26 GB (2.2 ms), the sketch's 159 MB (1.5 ms), the top 64 (0.3 ms). Host time per
  draft is 0.1 ms.
- g128 reads 15% fewer bytes than g32 and decodes 13% faster without drafts; its drafts are accepted less often (3.03
  against 3.20 tokens per verify at k = 4). That comes from the target, not from the drafter's own weights: the g128
  target with the g32 pack's MTP head accepts the same (3.04). AWQ at g128 does not bring it back (2.96).

### 6.2 Block drafters

Qwen3.8-27B (q4 g32 pack) with the released block drafters, as their bf16 checkpoints and as q8 and q4 packs
(`convert <drafter> --target <model>`), the same prompts and bench. Arc Pro B60, tok/s with tokens per verify in
parentheses; spec off is 22.6:

| Drafter | Weights | k = 3 | k = 5 | k = 7 | best |
|---|---|---|---|---|---|
| MTP head | in the pack, q4 | 48.2 (2.92) | 49.8 (3.35) | 47.3 (3.55) | 2.21x (49.9 at k = 4) |
| DFlash 2 | bf16, 3.85 GB | 45.2 (3.07) | 51.0 (3.58) | 54.0 (3.92) | 2.39x |
| DFlash 2 | q8, 1.98 GB | 48.3 (3.07) | 54.1 (3.58) | 57.7 (3.93) | 2.55x |
| DFlash 2 | q4, 1.26 GB | 50.6 (3.03) | 57.3 (3.55) | **60.7** (3.88) | 2.69x |
| DSpine | bf16, 4.05 GB | 41.2 (2.70) | 44.5 (3.07) | 45.0 (3.27) | 1.99x |
| DSpine | q8, 2.12 GB | 43.1 (2.70) | 46.4 (3.07) | 47.2 (3.26) | 2.09x |
| DSpine | q4, 1.43 GB | 45.4 (2.73) | 49.3 (3.09) | **50.6** (3.31) | 2.24x |

With the q4 g128 target pack (spec off 25.5; it reads fewer bytes a step and its drafts are accepted less often,
section 6.1): the MTP head 51.5 at k = 5 (3.14), DFlash 2 at q4 62.1 at k = 7 (3.69), DSpine at q4 54.2 (3.12).

By prompt at k = 7 (q4 drafters, the g32 target), tokens per verify and tok/s:

| Prompt | MTP head | DFlash 2 | DSpine |
|---|---|---|---|
| Python function with tests | 6.38, 84.5 | 6.89, 109.1 | **7.29, 114.3** |
| word problem, with reasoning | 5.10, 67.4 | 5.20, 79.8 | **5.31, 81.3** |
| how attention works, step by step | 3.59, 48.2 | **3.86, 60.6** | 3.86, 59.2 |
| five-bullet history summary | 2.97, 39.7 | **3.70, 57.1** | 2.45, 37.5 |
| translation, then grammar notes | 3.11, 42.2 | **3.49, 54.3** | 2.80, 42.6 |
| short story | 2.45, 32.1 | **2.46, 37.5** | 2.09, 31.5 |

- A round with a block drafter is one verify and one block: DFlash 2 at q4 and k = 7, a verify 55.8 ms and the block
  7.1 ms (63.8 ms a round); the MTP head at k = 4, a verify 53.3 ms and four drafts of 2.5 ms (64.0 ms). The block
  costs less than four chained drafts and proposes seven.
- A block is bound by reading the drafter, also on the B60: bf16 15 ms (DFlash 2, 8 rows), q8 10.7, q4 7.1. Of the
  q4 block, the target's head over the block's rows is 2 ms, the drafter's MLPs 2.6, its attention projections 1.
  Acceptance does not move with the drafter's format (3.92, 3.93, 3.88).
- DSpine drafts best where the text is predictable far ahead (code, a worked calculation) and worst on prose, on
  these prompts with thinking off; its references behave the same (3.46 tokens a round for DSpine and 3.81 for
  DFlash 2 over the six prompts, fp32 target). Its block of 16 does not pay here: 9, 11 and 15 drafts accept 3.39,
  3.45 and 3.53 tokens a verify, and a verify of more than 8 rows leaves the matrix-engine GEMV (117 ms at 16 rows,
  26.4 tok/s).
- Greedy text with a block drafter equals plain decoding on 6 of 6 prompts (DFlash 2 and DSpine at 7 drafts, DSpine
  at 15 with its 16-row verify; 5,968 bytes).

Arc B390, the q4 drafter packs, the q4 g32 target (spec off 6.6 in this session):

| Drafter | k = 3 | k = 5 | k = 7 | best |
|---|---|---|---|---|
| MTP head | | 18.3 (3.35) | | 2.8x (17.8 at k = 4) |
| DFlash 2, q4 | 16.8 (3.03) | 19.3 (3.55) | **20.1** (3.88) | 3.0x |
| DSpine, q4 | 14.2 (2.73) | 16.4 (3.09) | 17.5 (3.31) | 2.7x |

With the q4 g128 target pack (spec off 7.8): the MTP head 19.8 at k = 5 (3.14), DFlash 2 **22.8** at k = 7 (3.69),
DSpine 18.3 (3.11).

- A round of DFlash 2 at k = 7: a verify of 8 rows 167 ms, the block 23 ms (193 ms for 3.88 tokens); the MTP head at
  k = 5: a verify of 6 rows 156 ms, five drafts of 4.9 ms (183 ms for 3.35 tokens). The block reads the drafter's
  1.26 GB and the target's head (0.79 GB) once; five chained drafts read the head's sketch and the MTP layer five
  times.
- With a drafter pack beside the target's 16.6 GB of weights, about 7 GB of the machine's 30 GB stay available
  during a run. The bf16 drafters (2.7 GB more) were not run here.
- DSpine at 15 drafts: 11.1 tok/s. A verify of 16 rows takes 277 ms here too (`quant_gemm`), so a long block needs
  the matrix-engine GEMV past 8 rows on both GPUs.

## 7. Next

- The matrix-engine GEMV past 8 rows, so that a verify of a whole DSpine block (16 rows) costs about what one of 8
  does; then a round's drafts chosen from what the text has been accepting.
- A block drafter's head over a shortlist (as the MTP head's), where reading the head bounds a block.
- Block drafters for Gemma 4 and the MoE targets when checkpoints exist; sampled chains from the candidates.
- A recurrent state that lags a step behind: a verify would apply the tokens the previous step kept and then its own
  without storing them, instead of storing all and replaying the kept ones after a rejection (2.6 ms a round on
  the Arc B390).
- x prepared for the matrix-engine GEMV by the kernel that produces it (RMSNorm, the fused activation) instead of a
  launch of its own.
- MoE verify that reads each expert once per verify; q8 weights on the matrix-engine GEMV (q4 only so far).
- Adaptive k.
- Prompt lookup drafts (free drafts where the reply copies the prompt).
