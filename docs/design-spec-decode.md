# Design: speculative decoding with MTP drafters

Status: **implemented** · 2026-11-09 (measured speeds in section 6)

Speculative decoding for every model that ships a multi-token-prediction drafter: the MTP head inside Qwen3.5-family
checkpoints (Qwen3.8-27B, Qwen3.6-35B-A3B) and the Gemma 4 "assistant" drafters (`google/gemma-4-*-assistant`, one
per Gemma 4 size). One loop in the engine, one drafter interface, the model-specific wiring in each model's folder
(DESIGN.md section 10). This document records how the backend, the models and the engine changed.

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
  `LoadOptions::speculation` and has a drafter: the target's prefill program ("logits" of the last row, "hidden" of
  every row), its verify program ("logits" and "hidden" of every row), the draft program, `prepare_draft(tokens,
  hidden, start_pos, depth)` and `drafter_keeps_state()`. "hidden" is the target's post-norm hidden state, what both
  drafter families take.
- Qwen3.5 (`qwen3_5_model.cpp`): one set of layer builders (weight prefix and state label as parameters) makes the
  step, prefill and verify graphs and the MTP head, so dense and MoE heads come out of the same code; the head's KV
  slot is `mtp.layers.0`. A pair (h_i, token i + 1) runs at position i; chained drafts advance one position each.
- Gemma 4 (`gemma4_model.cpp`): the assistant from its checkpoint (`--drafter <dir>`, its weights under `assistant.`
  through `JoinedWeights`): query-only layers reading the target's last sliding / full KV slots through `KvRead`,
  the centroid head built from TopK, an i32 Gather of the cluster's tokens, Gather and MatMul (outputs "logits" over
  the candidates and "candidates"). Every draft of a round runs at the position of the token it starts from.

### 3.4 Engine

- `engine/spec/draft_runner`: the drafter side. Pairs wait until the drafter takes them; a drafter that keeps state
  takes every pair once, in order (prefill chunks flush, a round's first step catches up on the pairs the last verify
  accepted), its chained drafts are truncated away after each round. The session's program order is `spec::Program`.
- `Engine::spec_round`: k drafts, one verify of the pending token and the drafts, acceptance through
  `Sampler::verify` (greedy: the argmax; sampling: rejection sampling against the point-mass draft, then the
  residual, so the output distribution is exactly the target's), truncation of the target to the accepted point.
  Rounds that would pass the context or the requested length draft fewer tokens.
- Stats: verify rounds, drafted and accepted tokens in `GenerateStats` (`timings` in API responses, `run` / `bench`).

### 3.5 Flags and packs

- `--spec off|mtp` (default `off`), `--spec-k N` (default 3), `--drafter <dir>` on `run`, `serve` and `bench`;
  `bench --prompts <file>` measures over real prompts (acceptance depends on the text).
- `convert` plans the drafter's graph too: `mtp.*` is quantized by the same rule as the target's matmuls. A Gemma
  assistant is read from its own checkpoint (bf16); packing it with the target is not done yet.

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
- Real models: greedy output with speculation equals output without (`run`, then the bench prompts).

## 6. Measured speed (Arc Pro B60, batch 1)

Greedy decode over the six prompts of section 2 (chat template, thinking off, up to 256 new tokens each, stopping at
the end of the turn), `xpu-serve bench <pack> --prompts <file> --spec off|mtp --spec-k k`; decode tok/s counts the
tokens after the first. q4 packs (g32, round to nearest; the MTP heads quantized like the target); the Gemma
assistants run from their bf16 checkpoints.

| Model | spec off | k = 1 | k = 2 | k = 3 | k = 4 | best |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 22.2 | 35.0 (1.85) | **39.6** (2.47) | 37.5 (2.92) | 35.8 (3.21) | 1.78x |
| Qwen3.6-35B-A3B | 59.9 | 80.6 (1.84) | **85.6** (2.44) | 83.5 (2.82) | 64.9 (3.06) | 1.43x |
| Gemma 4 E2B + assistant | 127.3 | **164.5** (1.65) | 144.5 (2.02) | 147.2 (2.27) | 132.4 (2.40) | 1.29x |
| Gemma 4 26B-A4B + assistant | 65.6 | 81.7 (1.81) | **84.2** (2.41) | 79.2 (2.86) | 61.0 (3.20) | 1.28x |

tok/s, with tokens per verify in parentheses; they follow the references of section 2. Greedy output with k = 2
equals plain decoding prompt by prompt on Qwen3.8, Qwen3.6 and E2B (6 of 6 each) and on 5 of 6 for 26B-A4B: its story
diverges after 673 of 902 bytes ("It wasn't painting a perfect circle" / "It wasn't a perfect circle"), the kind of
near tie a wider verify (another kernel path at f16) can flip (DESIGN.md 10.4; the margin was not measured).

Where the time goes, and what would move it:

- A verify of T tokens costs more than a decode step. Qwen3.8 at T = 1 / 2 / 3 / 4: 44.8 / 48.2 / 52.4 / 61.9 ms (the
  q4 GEMV re-tuned for 2 and 3 rows took T = 2 from 60.2 ms). A verify as cheap as a decode step would take Qwen3.8
  at k = 3 to about 50 tok/s.
- MoE targets (Qwen3.6, 26B-A4B): a verify of k + 1 tokens reads the union of their experts, so it grows faster with
  k than a dense model's; past 4 (token, expert) pairs per expert bank the expert path also changes kernels. Best k
  is 2.
- Drafter steps: Qwen3.8's MTP step is about 2.5 ms (its lm_head over 248k tokens alone 1.8 ms). The E2B target's
  step is short (about 7 ms), so its rounds are bound by per-program host overhead (several programs per round, each
  waiting for its outputs); k = 1 is best there.
- Adaptive k (DESIGN.md 10.1) is not done: `--spec-k` is fixed, 2 is the best default on these four.

## 7. Next

- Verify cost at 2 to 5 rows (a DPAS path for the quantized matmuls at few rows); MoE verify that reads each expert
  once per verify.
- Fewer host round trips per round (the drafter's chained steps without waiting for each output).
- Adaptive k; packing a Gemma assistant with its target (`convert --with-drafter`); the assistant at q8.
