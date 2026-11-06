# TODO

The working plan for [DESIGN.md](DESIGN.md). Phases match DESIGN.md section 17.
**Backend order: OpenVINO first (Phases 1 to 3), then SYCL (Phase 4). The main backend is chosen in Phase 5,
after experiments.**

Legend: `[x]` done · `[ ]` to do · **Done when** = the check that closes the item.

---

## Bootstrap

- [x] Repo created with MIT license
- [x] README
- [x] DESIGN.md and TODO.md
- [ ] Review of DESIGN.md; answers to its section 19 recorded in `docs/decisions/0001-*.md` and later

---

## Phase 0: Foundations

Everything that does not depend on a GPU backend. All tests run on the CPU reference backend.

### 0.1 Skeleton
- [ ] CMake project (C++20), one `xpu-serve` binary, options `XPU_BACKEND_OPENVINO` / `XPU_BACKEND_SYCL`
- [ ] CLI dispatcher: `convert`, `serve`, `run`, `bench`, `check`, `info` (stubs that refuse by name)
- [ ] Vendored deps in `third_party/`: nlohmann/json, cpp-httplib; `NOTICE` started
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
- [ ] Safetensors reader (mmap, header JSON, dtypes incl. BF16/F16/U8/I8/F32), sharded index
- [ ] Safetensors writer (sharding ≤ 4 GB, index file)
- [ ] `docs/pack-format.md`: formats `bf16`, `f16`, `q8_sym`, `q4_sym`, `q4_asym`; packing order; manifest schema
- [ ] Pack reader: manifest, per-tensor formats, refuses unknown formats by name
- **Done when:** round-trip tests (write → read → identical bytes) pass for every format.

### 0.4 Quantizers
- [ ] RTN `q8_sym` (per group, per channel), `q4_sym`, `q4_asym`, group 32 / 64 / 128
- [ ] Dequant reference (fp64) used by tests and the CPU backend
- [ ] Unit tests: known vectors, edge cases (all-zero group, outliers, odd `in` sizes refused by name)
- **Done when:** quantize → dequantize error matches the analytic bound for each format.

### 0.5 `convert`
- [ ] Hugging Face download: resumable, `HF_TOKEN`, pinned revision, sha256 recorded
- [ ] Local-directory input
- [ ] Per-arch plan hook (`models/<arch>/convert.cpp`): what to quantize, renames
- [ ] Copy config / tokenizer / chat template / generation config unchanged; write `xpu_pack.json`
- [ ] `--with-drafter` → `drafters/<kind>/`
- [ ] Reproducibility test: same input + options → same bytes
- **Done when:** `xpu-serve convert google/gemma-4-26B-A4B-it --quant q4_sym` produces a pack that `xpu-serve check`
  accepts, with a correct memory bill.

### 0.6 Step graph and CPU reference backend
- [ ] `graph/`: nodes, symbolic dims (tokens, sequences), builder API, outputs, topological order
- [ ] Backend interface (`backend/backend.h`): `upload`, `make_kv_pool`, `compile`, `Program::run`, capabilities
- [ ] CPU reference backend: every op in DESIGN.md 8.4, fp32 (fp64 in contracts)
- [ ] Kernel contract framework: fp64 reference, shape sweeps, dtype tolerances, **planted-bug controls that must fail**
- **Done when:** every op has a passing contract on the CPU reference backend, and every planted bug is caught.

### 0.7 Tokenizer and chat templates
- [ ] `tokenizer.json` reader: BPE with byte fallback (Gemma), byte-level BPE (Qwen, Llama 3), normalizers,
      pre-tokenizer regex, added/special tokens
- [ ] Decide regex engine (PCRE2 vs port) → `docs/decisions/`
- [ ] Parity corpus vs Hugging Face (code, CJK, emoji, whitespace runs, special tokens)
- [ ] Jinja renderer (minja candidate) → `docs/decisions/`; parity vs `apply_chat_template` incl. tools
- **Done when:** both parity suites pass for Gemma 4 (and Qwen 3 as the second family).

### 0.8 Gemma 4 model code
- [ ] `models/gemma4/config.cpp`: parse (`text_config` first, then root), validate, memory bill
- [ ] `models/common/`: attention, GeGLU MLP, MoE, RMSNorm, RoPE variants
- [ ] `models/gemma4/model.cpp`: sliding + global attention, K=V global layers, MoE + dense MLP, tied head, softcap
- [ ] `models/gemma4/README.md`: quirks, references, tested packs
- [ ] `tests/parity/make_tiny.py`: tiny random Gemma 4 (sliding + global + MoE + K=V) → weights + fp32 logits and
      hidden states; also a "fake-quant" reference run with dequantized weights
- **Done when:** tiny Gemma 4 on the CPU reference backend matches Hugging Face: bf16 cos ≥ 0.9999 and same top-1;
  q8/q4 against the fake-quant reference within contract tolerance.

### 0.9 Minimal generate loop
- [ ] Engine skeleton: one slot, contiguous KV, greedy sampling, stop on EOS
- [ ] `xpu-serve run` on the CPU reference backend with the tiny model
- **Done when:** the tiny model generates the same greedy tokens as Hugging Face.

---

## Phase 1: OpenVINO backend, Gemma 4 end to end

- [ ] Decide the OpenVINO pin: a release, or pinned source + `third_party/patches/openvino/` →
      `scripts/build-openvino.sh` → `docs/decisions/`
- [ ] Step graph → `ov::Model` (opset), dynamic token dims; weights as Constants over the mmapped safetensors
- [ ] Quantized weights via `Convert → Subtract(zp) → Multiply(scale) → MatMul`; confirm the GPU plugin fuses it
      (compressed matmul shows up in the perf counters)
- [ ] Embedding gather from the quantized table; tied LM head; logits only for the rows we sample from
- [ ] PagedAttention adapter (`backend/openvino/paged_attention.cpp`): engine-owned KV tensors (remote context),
      block tables, lengths; sliding window; GQA; head dims 256 and 512
- [ ] Model cache (`CACHE_DIR`) for compile time
- [ ] Fill in the capabilities table (DESIGN.md 8.5) with what was actually verified
- [ ] Kernel contracts on OpenVINO (one-op graphs) for every op
- [ ] Tiny parity on OpenVINO GPU (bf16, q8, q4)
- [ ] Real parity: Gemma 4 26B-A4B bf16 vs Hugging Face fp32 (8 prompts, 64 tokens)
- [ ] Bench Gemma 4 26B-A4B q4 on B390 and B60 vs the GenAI baseline
- **Done when:** parity passes and decode tok/s is within 5% of OpenVINO GenAI at 1k / 32k / 64k (same kernels, so a
  gap means our overhead; find it before moving on).

---

## Phase 2: Server and engine

- [ ] Engine thread + request queue + per-request token channel; cancel on disconnect
- [ ] Paged KV: block allocator, pools per attention kind (sliding / global), sliding-window block release,
      KV-shared layers read their source pool
- [ ] int8 KV option (`--kv-dtype i8`)
- [ ] Chunked prefill that yields to decode at chunk boundaries
- [ ] Continuous batching (after batch 1 is solid)
- [ ] Prefix cache: hash-chained full blocks, refcounts, LRU; validity rule for sliding layers (DESIGN.md 9.4)
- [ ] Memory bill at load; refuse by name when it does not fit
- [ ] Sampling: temperature, top-k, top-p, min-p, penalties, seed; device-side argmax / top-k
- [ ] OpenAI API: `/v1/chat/completions` (stream + non-stream, tools, reasoning), `/v1/completions`, `/v1/models`,
      `/health`, `/metrics`
- [ ] Gemma 4 tool-call and reasoning dialect (`chat/dialects/`)
- [ ] One summary log line per request
- [ ] E2E tests with an OpenAI client: streaming, tools, stop strings, cancellation, concurrency
- **Done when:** E2E suite passes; a multi-turn agent session shows prefix-cache hits; 64k context runs on B390
  without OOM.

---

## Phase 3: Speculative decoding (on OpenVINO)

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

- [ ] Toolchain: oneAPI DPC++ build in CMake; device selection; USM allocation; in-order queues
- [ ] Bandwidth probe in SYCL (matches 0.2)
- [ ] int4 / int8 GEMV for decode (target ≥ 85% of measured bandwidth), contracts for every format and group size
- [ ] MoE grouped GEMV (top-8 of 128) incl. the expert union at verify width k+1
- [ ] Paged decode attention: head dims 256 / 512, GQA, sliding window, chain and **tree** masks
- [ ] Prefill: oneDNN GEMM through SYCL interop; prefill attention
- [ ] Norms, RoPE, activations, embedding gather, sampling helpers
- [ ] SYCL command-graph capture/replay per shape bucket; eager fallback; measure per-step overhead
- [ ] Tiny + real parity on SYCL
- [ ] Per-op comparison with OpenVINO (same graph, same inputs) → `bench/`
- [ ] Tree drafts end to end (capability only SYCL has at first)
- **Done when:** every contract and parity test passes on SYCL, and the per-op comparison is published.

---

## Phase 5: Performance on both backends, then the backend decision

- [ ] int8 KV vs f16 KV (speed and accuracy)
- [ ] K=V stored once on Gemma 4 global layers (where the backend allows)
- [ ] Fusions found by profiling (residual + norm, router + top-k, PLE tail for E-series later)
- [ ] Adaptive k tuning; tree drafts vs chain drafts
- [ ] Per-step overhead budget (host side) measured and minimized
- [ ] Run the decision experiments (DESIGN.md 8.2 criteria) on B390 and B60 → `docs/decisions/NNNN-main-backend.md`
- **Done when:** the main backend is chosen with numbers, and every optimization shows its gain against the roofline.

---

## Phase 6: More models and formats

- [ ] Gemma 4 dense (12B, 31B) and E-series (E2B, E4B: PLE, KV-shared tail)
- [ ] Qwen 3 dense and MoE: the "add a model in one PR" test; write `docs/adding-a-model.md` from the experience
- [ ] Llama 3
- [ ] Import MLX-affine packs; import GGUF (Q4_0 / Q8_0 / K-quants)
- [ ] Data-aware quantization (AWQ-style scale search, GPTQ) in `convert`, running on our own engine
- [ ] Anthropic `/v1/messages`, Ollama `/api/chat` and `/api/generate`
- [ ] Constrained decoding (JSON schema / grammar), logprobs
- **Done when:** each item lands as one PR with its parity and E2E tests.

---

## Phase 7: Packaging and release

- [ ] Linux release build (bundled runtimes), Windows release build
- [ ] Docker image
- [ ] Install and quick-start docs
- [ ] Versioning, CHANGELOG, release checklist
- **Done when:** a clean machine goes from download to a served model with two commands.
