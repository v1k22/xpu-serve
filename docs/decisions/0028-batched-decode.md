# 0028: The decode steps of several requests as one

Status: **accepted** for decode steps and the verifies of speculative rounds; the drafters' steps and KV blocks shared
between sequences still to come · 2026-11-25

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
- **The GPU runs every op once over all the rows.** A stateful op reads each segment's rows and its sequence's state
  through a table of the step's segments (an entry a segment: its first row, its tokens, its sequence's buffers),
  uploaded once a step: the convolutions, the gated delta rule, the KV appends and decode attention, over a cache of
  any format, are one launch each. What has no such kernel (the gated delta rule at a key width its lanes kernel does
  not take, attention over more rows than the per-row kernel's, an f32 cache's appends, a step that keeps its inputs
  for truncate) runs a segment at a time: the launch reads the segment's view (its first row and count) with the
  sequence's slots made current. The weights are read once a step.
- **A model gives a batch graph** (`Model::batch_graph`): its step graph with every row's logits and argmax instead
  of the last row's, built when the server holds several sequences (`LoadOptions::batch`). Every language model
  has one but Qwen3-ASR's (transcription runs a request at a time).
- **The engine decodes the requests in flight together** (`Engine::advance_decoding`): each emits its pending token,
  then the next tokens of all that go on run as one step of the batch graph, each request sampling its own row
  (from the argmax when every request's sampler reads no more). Prefill chunks still run a request at a time, the
  service sharing the device's time between the two sides as before. It is on whenever `--sequences` is above 1.
- **Speculating, the verifies of the requests' rounds run as one step**: each request drafts with its own drafter
  (its draft program, its sequence current), then one step of the verify graph (which takes segments when the model
  is loaded with `LoadOptions::batch`) runs every request's pending token and drafts as a segment; each request
  accepts from its own rows and cuts its sequence back inside its segment (truncate with it current: the session
  keeps a segment's inputs for that, as it does a short step's). A request emits every token its last round gave
  before the next step, so all of them run their rounds in the same steps. The requests of a step share a budget of
  rows (`EngineOptions::verify_rows`, 8): each drafts up to 8 / N - 1, none from nine requests on. On a discrete GPU a
  step of up to 8 rows costs about one row's; past it the rows add up.

## Measured

Arc Pro B60, N requests sent at once (`--sequences N`; six chat prompts in turn, thinking off, 256 greedy tokens each
at most): the tokens of all of them over the wall time ("in all", prefills included), and each request's decode speed.

| Model | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---|---|---|---|
| Qwen3.8-27B, Q4_0 pack (dense) | 21.7 tok/s | 38.7 in all, 19.6 each | 67.4 in all, 17.5 each | 113.9 in all, 15.4 each |
| Inkling-2-Mini, AD-Q4_K (mixture of experts) | 43.1 (46.6 decoding) | 64.9 in all, 34.8 each | 82.7 in all, 23.9 each | 103.0 in all, 15.9 each |

The dense model reads its weights once a step whatever the rows: 5.2 times the tokens at eight. The mixture of experts
reads the experts each token picks, more as requests are added: 2.4 times. Measured with the stateful ops run as one
launch over the segments and the matmul kernels of the same day (q8 and the GGUF block formats on the matrix engines,
an expert's pairs as the rows of one GEMV); with a launch a segment the same sweep gave 106.8 and 88.2 tok/s at eight,
before those kernels 111.6 and 96.1.

Speculating (`--spec`, the same requests; MTP's three drafts a round, DSpark's two, fewer as the budget asks):

| Model, drafter | N = 1 | N = 2 | N = 4 | N = 8 |
|---|---|---|---|---|
| Qwen3.8-27B, Q4_0 pack, its MTP head | 59.2 (60.9 decoding) | 78.9 in all, 44.7 each | 94.2 in all, 25.6 each | 112.4 in all, 15.2 each |
| Inkling-2-Mini, AD-Q4_K, DSpark | 52.4 (57.6 decoding) | 69.9 in all, 39.0 each | 77.8 in all, 23.1 each | 101.3 in all, 15.6 each |

The dense model gains over plain batching at two and four requests (2.0 and 1.4 times) and loses 1 % at eight, where
no draft fits. The mixture of experts gains 8 % at two and loses 6 % at four and 2 % at eight: each draft's row reads
experts of its own.
A verify's stateful ops run as one launch over the segments too, each segment keeping its inputs in its sequence's
slot for the truncate (`StateSegment::keep`); with a launch a segment, eight requests gave 103.6 and 84.8.

## Consequences

- No new flag: `--sequences N` now buys throughput, not only turns. Its KV cache is billed per sequence (0027), and
  the batch graph's step is billed at N tokens.
- Speculating, the drafters still run a request at a time (a chain drafter's k runs, a block drafter's one): one
  draft program over the segments needs every row's outputs (a chain drafter's gives its last pair's) and, for a block
  drafter, attention within each segment's block. Then KV blocks shared between sequences (a prefix cache across
  them, DESIGN.md 9.4).
- With more than two requests on a mixture of experts, speculation costs more than it gains: serve them without
  `--spec`.
- A stateful op without a kernel over the segments still costs a launch a segment, its share of a step growing with
  N; none of the served models meets one in a batched decode at f16.
