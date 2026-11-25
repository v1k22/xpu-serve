# 0028: The decode steps of several requests as one

Status: **accepted** for decode steps; speculative rounds and KV blocks shared between sequences still to come ·
2026-11-25

## Context

Decision 0027 put several requests in flight, each on a sequence of its own, their steps taking turns: a second user
no longer waits for the first to finish, but N streams share one stream's speed. A decode step of a dense model is
bound by reading its weights (Qwen3.8-27B in q4: 15 GB a token on an Arc Pro B60, about 22 tok/s), and reading them
once for the next token of several requests costs little more than for one. That needs a step whose rows belong to
several sequences, which the step graph's contract (one run advances one sequence) did not allow.

## Decision

- **A run's rows can be segments of several sequences** (`Session::run_segments`): the next tokens of each sequence,
  one segment after another. Such a run gives what running each segment alone on its sequence gives, its outputs'
  rows in the same order; every sequence advances by its tokens, and a failed run leaves them all as they were. That
  definition is the session's own implementation (each segment run in turn), which cpu-ref keeps as the oracle.
- **Only a graph whose rows stay independent may run so** (`Graph::allow_segments`), which `validate()` checks: every
  input and output has the token axis first and no value has it elsewhere (so no matmul reads across it), no PrevRow
  or KvRead, no Slice or ReduceSum along it, no reduction over a last axis that is it, no Gather from a table of the
  step's rows, and attention only causal over the keys and values of a KvCache of the graph, which feed nothing
  else. The stateful ops (the KV appends, attention over them, the convolutions, the gated delta rule) then act on
  each segment with its sequence's state; everything else is per row.
- **The GPU runs every other op once over all the rows**, and a stateful op a segment at a time over the segment's
  rows and its sequence's buffers, through the kernels it already has: the program's launch reads a segment's view
  (its first row and count) for the values with the token axis first, and the sequence's slots are made current for
  that launch. The weights are read once a step; attention and the state ops are a launch a segment.
- **A model gives a batch graph** (`Model::batch_graph`): its step graph with every row's logits and argmax instead
  of the last row's, built when the server holds several sequences (`LoadOptions::batch`). Every language model
  has one but Qwen3-ASR's (transcription runs a request at a time).
- **The engine decodes the requests in flight together** (`Engine::advance_decoding`): each emits its pending token,
  then the next tokens of all that go on run as one step of the batch graph, each request sampling its own row
  (from the argmax when every request's sampler reads no more). Prefill chunks still run a request at a time, the
  service sharing the device's time between the two sides as before. It is on whenever `--sequences` is above 1,
  except with speculative decoding, whose rounds stay a request's own.

## Measured

Arc Pro B60, N requests sent at once (`--sequences N`; six chat prompts in turn, thinking off, 256 greedy tokens each
at most): the tokens of all of them over the wall time ("in all", prefills included), and each request's decode speed.

| Model | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---|---|---|---|
| Qwen3.8-27B, Q4_0 pack (dense) | 21.6 tok/s | 37.4 in all, 19.0 each | 64.1 in all, 16.9 each | 106.8 in all, 14.6 each |
| Inkling-2-Mini, AD-Q4_K (mixture of experts) | 42.6 (45.9 decoding) | 61.5 in all, 32.9 each | 74.0 in all, 21.0 each | 88.2 in all, 13.0 each |

The dense model reads its weights once a step whatever the rows: 4.9 times the tokens at eight. The mixture of experts
reads the experts each token picks, more as requests are added, and its steps hold many stateful ops, each a launch a
segment: 2.1 times.

## Consequences

- No new flag: `--sequences N` now buys throughput, not only turns. Its KV cache is billed per sequence (0027), and
  the batch graph's step is billed at N tokens.
- With `--spec`, requests in flight still decode in turn: a round's verify is a segment of k + 1 rows, which the same
  mechanism runs once the drafters keep state per sequence in one step. That is the next step, then KV blocks shared
  between sequences (a prefix cache across them, DESIGN.md 9.4).
- A stateful op costs a launch a segment, so attention's share of a step grows with N; one kernel over the segments'
  rows (each reading its sequence's cache) would remove it.
