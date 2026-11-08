# TODO

The working plan for [DESIGN.md](DESIGN.md). Phases match DESIGN.md section 17.
**Backends: OpenVINO first (Phases 1 to 3), then SYCL (Phase 4). 2026-11-07: after the first comparison SYCL became
the GPU backend (`docs/decisions/0007`) and the OpenVINO backend was removed (`docs/decisions/0008`). Open
OpenVINO-only items were dropped; the rest live in Phase 4. OpenVINO GenAI stays as an external speed baseline.**

**v0.1 (2026-11-06):** Gemma 4 E2B text, served end to end on OpenVINO, weights loaded from the HF checkpoint (convert
skipped for now, by decision), KV as OpenVINO state rather than engine-owned pages (`docs/decisions/0002`). Ticked items
below are done and tested; the parity numbers are in `tests/unit/model/model_parity.cpp`.

**v0.3 (2026-11-07):** Qwen3.5 dense text (Gated DeltaNet + gated full attention, `docs/decisions/0006`), byte-level BPE
tokenizer with NFC, Qwen3.5 chat template parity and output parser.

**v0.2 (2026-11-07):** image input for Gemma 4 E2B/E4B (OpenAI `image_url` data URLs), checked against Hugging Face
end to end (`docs/decisions/0005`, `tests/unit/model/gemma4_vision_parity.cpp`). `PROJECT.md` maps the code.

Legend: `[x]` done · `[ ]` to do · **Done when** = the check that closes the item.

---

## Bootstrap

- [x] Repo created with MIT license
- [x] README
- [x] DESIGN.md and TODO.md
- [x] Review of DESIGN.md (approved 2026-11-06); decisions so far in `docs/decisions/0001`–`0004`

---

## Phase 0: Foundations

Everything that does not depend on a GPU backend. All tests run on the CPU reference backend.

### 0.1 Skeleton
- [x] CMake project (C++20), one `xpu-serve` binary, options `XS_WITH_SYCL` (on with icpx), `XS_SYCL_TARGETS`
- [ ] CLI dispatcher: `convert`, `serve`, `run`, `bench`, `check`, `info` (stubs that refuse by name)
- [x] Vendored deps in `third_party/`: nlohmann/json, cpp-httplib; `NOTICE` started
- [ ] CTest wiring + GitHub Actions job (Linux, CPU): build + unit tests
- [ ] `.clang-format`, `.gitignore`
- **Done when:** a clean checkout builds and `ctest` passes in CI.

### 0.2 Measurement setup (before writing any kernel)
- [ ] `bench/hardware.md`: Arc B390 (Panther Lake) and Arc Pro B60: driver, OS, kernel, memory, theoretical bandwidth
- [ ] Bandwidth probe on both devices (large read kernel) → measured roofline numbers
- [ ] OpenVINO GenAI baseline for Gemma 4 26B-A4B int4: decode tok/s and TTFT at 1k / 8k / 32k / 64k, batch 1,
      with and without the MTP drafter → `bench/baselines.md` with full setup
- [ ] Fixed prompt sets in `bench/prompts/` (code, tool calling, chat) at each context length
- **Done when:** every later perf number has a baseline and a ceiling to compare with.

### 0.3 Safetensors and packs
- [x] Safetensors reader (mmap, header JSON, dtypes incl. BF16/F16/U8/I8/F32), sharded index
- [x] Safetensors writer (streamed, sharding ≤ 4 GB, index file)
- [x] Pack format documented in DESIGN.md 6.3 / 6.4: `q8` (symmetric), `q4` (affine, f16 scale + bias), manifest
- [x] Pack reader: manifest, per-tensor formats as Q8 / Q4 views, refuses unknown formats and versions by name
- **Done when:** round-trip tests (write → read → identical bytes) pass for every format.

### 0.4 Quantizers
- [x] RTN `q8` (symmetric) and `q4` (affine), groups 32 / 64 / 128 (`src/quant/`)
- [x] Dequantization used by tests and the CPU reference backend
- [x] Unit tests: known vectors, nibble order (planted bug), all-zero and constant groups, bad shapes refused by name
- **Done when:** quantize → dequantize error matches the analytic bound for each format.

### 0.5 `convert`
- [ ] Hugging Face download: resumable, `HF_TOKEN`, pinned revision, sha256 recorded
- [x] Local-directory input
- [x] What to quantize comes from the model's own step graph (matmul weights; gather tables at q8), no per-arch code;
      `--q8 name,...` keeps chosen weights at q8
- [x] Copy config / tokenizer / chat template / generation config unchanged; write `xpu_pack.json`
- [x] End to end on a tiny Qwen3.5 written to disk (plan, layout, exact values, logits, GPU vs cpu-ref)
- [ ] Quantize Gemma 4's per-layer embedding table (read on the host) and vision towers
- [ ] `--with-drafter` → `drafters/<kind>/`
- [ ] Reproducibility test: same input + options → same bytes
- **Done when:** `xpu-serve convert google/gemma-4-26B-A4B-it --quant q4_sym` produces a pack that `xpu-serve check`
  accepts, with a correct memory bill.

### 0.6 Step graph and CPU reference backend
- [x] `graph/`: nodes, symbolic dims (tokens, sequences), builder API, outputs, topological order
- [x] Backend interface (`backend/backend.h`): `upload`, `make_kv_pool`, `compile`, `Program::run`, capabilities
- [x] CPU reference backend: every op in DESIGN.md 8.4, fp32 (fp64 in contracts)
- [x] Kernel contract framework: fp64 reference, shape sweeps, dtype tolerances, **planted-bug controls that must fail**
- **Done when:** every op has a passing contract on the CPU reference backend, and every planted bug is caught.

### 0.7 Tokenizer and chat templates
- [x] `tokenizer.json` reader: BPE with byte fallback (Gemma), byte-level BPE (Qwen, Llama 3), normalizers,
      pre-tokenizer regex, added/special tokens
- [ ] Decide regex engine (PCRE2 vs port) → `docs/decisions/`
- [x] Parity corpus vs Hugging Face (code, CJK, emoji, whitespace runs, special tokens)
- [x] Jinja renderer (minja candidate) → `docs/decisions/`; parity vs `apply_chat_template` incl. tools
- **Done when:** both parity suites pass for Gemma 4 (and Qwen 3 as the second family).

### 0.8 Gemma 4 model code
- [x] `models/gemma4/config.cpp`: parse (`text_config` first, then root), validate, memory bill
- [x] `models/common/`: attention, GeGLU MLP, MoE, RMSNorm, RoPE variants (`blocks.h`; Gemma 4 and Qwen3.5 built
      from them; MoE = router + top-k + ExpertMatMul over a stacked bank, cpu-ref and SYCL)
- [x] `models/gemma4/model.cpp`: sliding + global attention, K=V global layers, MoE + dense MLP, tied head, softcap
- [ ] `models/gemma4/README.md`: quirks, references, tested packs
- [ ] `tests/parity/make_tiny.py`: tiny random Gemma 4 (sliding + global + MoE + K=V) → weights + fp32 logits and
      hidden states; also a "fake-quant" reference run with dequantized weights
- **Done when:** tiny Gemma 4 on the CPU reference backend matches Hugging Face: bf16 cos ≥ 0.9999 and same top-1;
  q8/q4 against the fake-quant reference within contract tolerance.

### 0.9 Minimal generate loop
- [x] Engine skeleton: one slot, contiguous KV, greedy sampling, stop on EOS
- [x] `xpu-serve run` on the CPU reference backend with the tiny model
- **Done when:** the tiny model generates the same greedy tokens as Hugging Face.

---

## Phase 1: first backend (OpenVINO, since removed), Gemma 4 end to end

- [x] Step graph → `ov::Model` (opset), dynamic token dims; weights as Constants over the mmapped safetensors
- [ ] Embedding gather from the quantized table; tied LM head; logits only for the rows we sample from
- [ ] Fill in the capabilities table (DESIGN.md 8.5) with what was actually verified
- [x] Real parity vs Hugging Face fp32 — done for Gemma 4 **E2B** (6 prompts, logits + 32-token greedy) on cpu-ref and
      OpenVINO CPU; 26B-A4B needs MoE + K=V support first
- [ ] Bench Gemma 4 26B-A4B q4 on B390 and B60 vs the GenAI baseline

---

## Phase 2: Server and engine

- [x] Engine thread + request queue + per-request token channel; cancel on disconnect
- [ ] Paged KV: block allocator, pools per attention kind (sliding / global), sliding-window block release,
      KV-shared layers read their source pool
- [x] int8 KV option (`--kv-dtype i8`): int8 codes + an f32 scale per (token, KV head), opt-in (docs/decisions/0010)
- [ ] Chunked prefill that yields to decode at chunk boundaries
- [ ] Continuous batching (after batch 1 is solid)
- [ ] Prefix cache: hash-chained full blocks, refcounts, LRU; validity rule for sliding layers (DESIGN.md 9.4)
- [ ] Memory bill at load; refuse by name when it does not fit
- [x] Sampling: temperature, top-k, top-p, min-p, penalties, seed; device-side argmax / top-k
- [x] OpenAI API: `/v1/chat/completions` (stream + non-stream, tools, reasoning), `/v1/completions`, `/v1/models`,
      `/health`, `/metrics`
- [x] Gemma 4 tool-call and reasoning dialect (`chat/dialects/`)
- [x] One summary log line per request
- [x] E2E tests with an OpenAI client: streaming, tools, stop strings, cancellation, concurrency
- **Done when:** E2E suite passes; a multi-turn agent session shows prefix-cache hits; 64k context runs on B390
  without OOM.

---

## Phase 3: Speculative decoding

- [ ] `engine/spec/`: drafter interface, verify step (k+1 positions), greedy acceptance, rejection sampling, KV rollback
- [ ] Prompt lookup (n-gram) drafter
- [ ] Gemma 4 MTP assistant (`google/gemma-4-26B-A4B-it-assistant`): convert with `--with-drafter`, shared target KV,
      centroid LM head (`models/gemma4/drafter.cpp`)
- [ ] DFlash drafter (chain): hidden-state capture outputs from chosen target layers; drafter program
- [ ] Adaptive k from measured acceptance and per-k step cost (MoE verify cost)
- [ ] Spec equivalence tests: greedy spec on vs off (near-tie rule, DESIGN.md 10.4); sampling distribution test
- [ ] Spec stats in logs and `/metrics`
- [ ] Bench at 32k / 64k on B390 and B60 vs the GenAI baseline with the same drafter
- **Done when:** equivalence tests pass and decode tok/s at 32k/64k beats the GenAI baseline with the same drafter.

---

## Phase 4: SYCL backend (same model, own kernels)

- [x] Toolchain: oneAPI DPC++ build in CMake; device selection; USM allocation; in-order queues (2026-11-07, oneAPI 2026.2)
- [ ] Bandwidth probe in SYCL (matches 0.2)
- [x] int4 / int8 GEMV for decode; dequantize + GEMM for prefill; quantized gather (device suites, groups 32 / 128)
- [ ] int4 / int8 GEMV at ≥ 85% of measured bandwidth; prefill without the f16 copy of each weight
- [x] MoE expert matmuls over q8 / q4 banks: one GEMV per (token, expert) up to 32 pairs (decode), pairs grouped by
      expert up to 256, grouped oneMKL GEMMs on dequantized banks beyond (prefill)
- [ ] MoE prefill without the f16 copy of each expert bank (a GEMM that dequantizes in registers); the expert union
      at verify width k+1
- [ ] Paged decode attention: head dims 256 / 512, GQA, sliding window, chain and **tree** masks
- [x] Prefill: oneMKL GEMM (f16 x f16 -> f32); prefill attention through two batched GEMMs per query chunk
- [ ] Prefill attention on XMX without the score matrix (flash attention, e.g. SYCL*TLA)
- [ ] Gated delta prefill in chunks (WY form, GEMMs) instead of the per-token loop
- [x] Norms, RoPE, activations, embedding gather; f16 GEMV for decode; split-K decode attention (sampling stays on host)
- [ ] SYCL command-graph capture/replay per shape bucket; eager fallback; measure per-step overhead
- [x] Tiny + real parity on SYCL: op suites at f32 and f16; Gemma 4 E2B and Qwen3.5-4B match HF (f32 cosine 1.0, greedy exact)
- [ ] Multi-GPU: layer split and tensor split across two Arc Pro B60s in one process
- [x] End-to-end comparison with OpenVINO 2026.4.1 (docs/decisions/0007); per-op profile with `XS_SYCL_PROFILE=1`
- [ ] Tree drafts end to end (capability only SYCL has at first)
- **Done when:** every contract and parity test passes on SYCL, and the per-op comparison is published.

---

## Phase 5: Performance on both backends, then the backend decision

- [x] int8 KV vs f16 KV (speed and accuracy): decode +2 to 9 % at 16k to 32k, prefill -1 to 8 %, logits about
      q8-close, MMLU-Pro unchanged on Qwen3.5 and within noise on Gemma 4 (docs/decisions/0010)
- [ ] int8 KV: a larger eval on Gemma 4 (more questions, long-context tasks); per-channel key scales if it loses
- [ ] K=V stored once on Gemma 4 global layers (where the backend allows)
- [ ] Fusions found by profiling (residual + norm, router + top-k, PLE tail for E-series later)
- [ ] Adaptive k tuning; tree drafts vs chain drafts
- [ ] Per-step overhead budget (host side) measured and minimized
- [ ] Run the decision experiments (DESIGN.md 8.2 criteria) on B390 and B60 → `docs/decisions/NNNN-main-backend.md`
- **Done when:** the main backend is chosen with numbers, and every optimization shows its gain against the roofline.

---

## Phase 6: More models and formats

- [x] Image input for Gemma 4 E-series: `image_url` data URLs, Hugging Face-exact preprocessing (torchvision-exact
      resize), the vision tower as a stateless graph (Clamp, Transpose, axial Rope, non-causal Attention), image rows
      spliced into prefill chunks, prefix cache keyed by image content
- [ ] Vision towers with `standardize` (Gemma 4 26B / 31B); Qwen-VL style towers
- [ ] Audio input (Gemma 4 E-series audio tower, mel features) and video input
- [ ] Remote image URLs behind an explicit opt-in (size and host limits)
- [ ] Gemma 4 dense (12B, 31B) and E-series (E2B, E4B: PLE, KV-shared tail)
- [x] Qwen3.5 dense (text): Gated DeltaNet via CausalConv + GatedDelta graph ops, OpenVINO PagedGatedDeltaNet,
      byte-level BPE + NFC tokenizer, chat template parity, `qwen3_5` output parser; HF parity on CPU and B60
- [x] Qwen3.5 MoE (`qwen3_5_moe`): HF parity on a tiny random checkpoint (cpu-ref exact, SYCL f16 / f32); q8 / q4
      expert banks in packs (routers stay unquantized); SYCL expert kernels: one GEMV per (token, expert) for few
      pairs, pairs grouped by expert on the device for prefill
- [ ] Qwen3.5: state snapshots at prompt ends for multi-turn prefix reuse; vision tower; MTP head; per-expert
      checkpoint layout (`experts.<e>.gate_proj`)
- [ ] Qwen 3 dense and MoE: the "add a model in one PR" test; write `docs/adding-a-model.md` from the experience
- [ ] Llama 3
- [ ] Import MLX-affine packs; import GGUF (Q4_0 / Q8_0 / K-quants)
- [x] Data-aware quantization: AWQ-style scale and clip search in `convert`, calibrated on our own engine
- [ ] GPTQ; AWQ with the full activation covariance (not per channel); larger calibration sets
- [ ] Anthropic `/v1/messages`, Ollama `/api/chat` and `/api/generate`
- [ ] Constrained decoding (JSON schema / grammar), logprobs
- **Done when:** each item lands as one PR with its parity and E2E tests.

### Model enablement checklist

Every model below is enabled the same way (PROJECT.md 8.1); a model is done when every box is ticked:

1. Config parsed, unsupported variants refused by name (`models/<arch>/<arch>_config.cpp`, config test)
2. Graph built from shared blocks (`models/common/`), new ops added to the IR, cpu-ref and SYCL first
3. Weights load from the Hugging Face checkpoint (single and sharded)
4. Parity vs Hugging Face fp32: logits cosine and greedy output on cpu-ref (f32) and SYCL (f16)
5. Tokenizer and chat template byte-exact vs Hugging Face; output parser (thinking, tool calls) if the format needs one
6. Packs: `convert` q8 and q4 (AWQ), quality on the parity prompts and MMLU-Pro (25 questions) through the server
7. Fits and runs on the target device (Arc Pro B60, Panther Lake iGPU); bench decode / TTFT at 1k / 4k / 16k
8. Docs: TODO, PROJECT.md map, README supported list

| # | Model | Hugging Face ID | Notes |
|---|---|---|---|
| 1 | Qwen3.8-27B | `Qwen/Qwen3.8-27B` | |
| 2 | Qwen3.6-35B-A3B (preferred) | `Qwen/Qwen3.6-35B-A3B` | MoE (3B active). **Runs** (text): tiny-model HF parity; q4 g64 pack 18.9 GiB on one Arc Pro B60, decode ~48 tok/s, prefill ~1,250 tok/s, MMLU-Pro 84 % (25 q). To do: Panther Lake, AWQ, vision, MTP |
| | Qwen3.5-35B-A3B (older) | `Qwen/Qwen3.5-35B-A3B` | MoE; the Qwen3.5 Gated DeltaNet hybrid |
| 3 | Gemma 4 31B | `google/gemma-4-31B-it` | vision tower uses `standardize` |
| 4 | Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | MoE (4B active) |
| 5 | Nemotron 3.5 Lightning (30B-A3B) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | MoE (3B active) |
| | Nemotron 3.5 Super (120B-A12B, larger) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | MoE + vision; int4 and two GPUs |
| 6 | Muse Glimmer 30B | `meta-models/Muse-Glimmer-30B` | |
| 7 | Gemma 4 12B | `google/gemma-4-12B-it` | |
| 8 | Liquid LFM3-1.7B (main text model) | `LiquidAI/LFM3-1.7B` | small |
| 9 | Flint | `RecursiveSI/flint` | |

---

## Robot policies (VLA) — docs/design-vla.md

- [x] Policy kind: `models/policy.h`, `model_kind`, `load_policy`; `src/policy/` runtime (programs per camera subset
      and flow steps, warm-up, timings); `xpu-serve bench <policy-dir>`
- [x] π0.5 (`lerobot/pi05_base`): parity with LeRobot f32 (cpu-ref 5e-6, Arc B390 f16 1e-3), 1 / 2 / 3 cameras
- [x] π0.5 latency, first pass: f16 operands, fusions, fused flash attention (XMX) up to head_dim 128 with SigLIP
      per camera, Concat as one kernel (Arc B390, 3 cameras: 428.5 -> 302.2 ms; docs/design-vla.md §6)
- [ ] π0.5 latency, next: the prefix down
      projection's GEMM (~30 TFLOPS against ~50 for gate / up), in-place expert K/V, gate and up in one GEMM,
      record / replay of the program
- [ ] π0.5 int8 weights (opt-in, lossy): the action expert re-reads ~620 MB of f16 weights per flow step (~60 ms of
      a 3-camera call at the memory floor); int8 halves it. Needs a GEMM that dequantizes in registers (50 rows)
- [ ] Flash attention at head_dim 256 (Gemma): 8 rows per sub-group fill the registers, so every K/V tile feeds one
      DPAS and oneMKL's GEMMs match it; needs K/V or P shared across sub-groups through local memory
- [ ] Low precision on Panther Lake: the B390 compiles fp8 (E4M3 / E5M2) DPAS but returns wrong results; int8 DPAS is
      exact at ~122 TOPS (2x f16). A w8a8 path is the low-precision option there (lossy, opt-in)
- [ ] Normalization statistics (quantiles, mean/std) and resize-with-pad for other image sizes
- [ ] Policy service and API: `/v1/policy/info`, `/v1/policy/act` (JSON + msgpack), latest-wins queue, `/ready` after
      warm-up; then the openpi-compatible WebSocket; RTC (previous chunk + delay)
- [ ] More policies: π0, SmolVLA, GR00T N1.6 / N1.7, π0-FAST (through the language engine), OpenVLA-OFT

## World models

- [ ] `WorldModel` kind and `world/` runtime; first model: V-JEPA 2-AC (planning, batch axis) or Matrix-Game
      (interactive streaming)
- [ ] Ops: Conv2d / Conv3d, GroupNorm, Upsample, 3-D RoPE, block-causal masks, a batch dim

## Audio models

- [ ] `AudioModel` kind, `audio/` (decode, resample, log-mel), `/v1/audio/transcriptions` and `/v1/audio/speech`
- [ ] Speech recognition (Whisper-style encoder-decoder; streaming encoder state)
- [ ] Speech synthesis (codec-token TTS: token loop + codec decoder, streaming output)
- [ ] Audio in chat (an audio encoder in front of an LLM, like images)

---

## Phase 7: Packaging and release

- [ ] Linux release build (bundled runtimes), Windows release build
- [ ] Docker image
- [ ] Install and quick-start docs
- [ ] Versioning, CHANGELOG, release checklist
- **Done when:** a clean machine goes from download to a served model with two commands.
