# 0006: Hybrid linear attention (Qwen3.5 Gated DeltaNet)

Date: 2026-11-07 · Status: accepted

Qwen3.5 interleaves Gated DeltaNet layers (three of every four) with full attention. A Gated DeltaNet layer keeps a
fixed-size state per sequence instead of a growing KV cache: the last `kernel - 1` inputs of a causal depthwise conv,
and a `[heads, key_dim, value_dim]` matrix updated by the gated delta rule for every token.

**Two stateful graph ops.** `CausalConv` and `GatedDelta` carry their state in conv and recurrent slots, the same
way `KvCache` uses KV slots: model code stays backend-neutral and a session owns the state. Their exact semantics are
in `graph.h`, and both backends are tested over several steps against one fp64 pass over the whole sequence
(`tests/unit/backend/recurrent_ops_cases.h`).

**OpenVINO lowering.** OpenVINO 2026.2 runs the gated delta rule as internal ops whose headers are not in the
release archive: `GatedDeltaNet` (state as a variable; its GPU kernel is a reference implementation, about 2 ms per
layer per decoded token on the Arc Pro B60) and `PagedGatedDeltaNet` (optimized GPU kernel, 0.35 ms per decoded token
for all 24 layers). The backend uses the paged op by default with one block per layer: a state table the session
allocates on the device, binds once and the kernel updates in place, with the block indices pointing reads and writes
at block 0. The headers are vendored unchanged and the build insists on the matching OpenVINO version
(`third_party/openvino_dev/PROVENANCE.md`). `--backend-opt gdn=ref` selects the reference op. The causal conv is
one depthwise GroupConvolution over `[conv state; x]`, with the state in a variable.

**Grouped-query attention on GPU.** Qwen3.5's full-attention layers have 4 KV heads for 16 query heads. The OpenVINO
GPU plugin runs that in its fused SDPA kernel only when kv is repeated in the exact form its
UnsqueezeBroadcastReshapeSDPAFusion matches (Unsqueeze, bidirectional Broadcast with a constant target, Reshape);
any other form left SDPA decomposed into kernels built per new sequence length, about 85 ms of host time per decoded
token on the B60. `ov_attention.cpp` emits that form.

**Prefix reuse.** Fixed-size state cannot be cut back to a prefix, so `Session::truncate` only keeps all or nothing
for such graphs and the engine re-prefills when a new prompt does not extend the cached sequence. Snapshotting the
state at the end of each prompt (cheap: tens of MB) would restore multi-turn reuse; the paged op's `cache_interval`
is built for exactly that.
