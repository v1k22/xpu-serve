# 0002: v0.1 scope, and OpenVINO stateful KV as a stepping stone

Date: 2026-11-06 · Status: accepted for v0.1

**Scope of v0.1:** google/gemma-4-E2B-it, text only, weights converted from the checkpoint's bf16 at load (no
`convert` step yet), one sequence at a time, OpenVINO backend on Arc B390 and Arc Pro B60, no speculative decoding.

**KV cache:** DESIGN.md P4 says the engine owns KV memory (paged pools, block tables). v0.1 does not do that yet: the
step graph's `KvCache` op has per-slot state that the backend session owns, and the OpenVINO backend implements it with
`ReadValue`/`Assign` variables, the pattern its GPU plugin already optimizes. This keeps v0.1 small and fast while the
graph-level contract (`KvCache` appends and returns the full K/V; `Attention` reads it) is already the one paged
attention will implement. Moving to engine-owned paged KV changes the backends, not the model code.

Consequences: no continuous batching yet; prefix reuse works by truncating the session state to the common prefix;
sliding-window layers keep all keys in v0.1 (the mask hides the old ones).
