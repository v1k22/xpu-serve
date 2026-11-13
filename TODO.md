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
end to end (`docs/decisions/0005`, `tests/unit/model/gemma4_vision_parity.cpp`). `CLAUDE.md` maps the code.

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
- [x] A drafter checkpoint to a pack of its own: `convert <drafter> --target <model>`
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
- [x] WordPiece (BERT-family encoders: BertNormalizer, BertPreTokenizer, WordPiece decoder) and TemplateProcessing's
      special tokens (`with_special_tokens`); parity on all-MiniLM-L6-v2, bge-small-en-v1.5, nomic-embed-text-v1.5
- [x] ByteLevel with the GPT-2 regex (ModernBERT) and Metaspace (mmBERT, Gemma's vocabulary): the tokenizers of Laya
      (three) and Julia-1
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
- [ ] Refuse a `--ctx` whose KV cache and largest step cannot fit the device, at load and by name: Qwen3.8-27B Q4_0
      on an Arc Pro B60 runs 64k tokens with an f16 cache and 128k with int8, but a 96k f16 prompt ran out of device
      memory in the middle of its prefill and the uncaught error ended the process
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

- [x] `engine/spec/` + `Engine::spec_round`: verify of k+1 positions, greedy acceptance, rejection sampling, rollback
      of KV and of conv / recurrent state (`Session::truncate` inside a step of `rewind_tokens`), sessions of several
      programs sharing weights and slots, `KvRead` (docs/design-spec-decode.md)
- [x] Qwen3.5-family MTP head (`mtp.*`, dense and MoE) as the drafter; `convert` quantizes it
- [ ] Prompt lookup (n-gram) drafter
- [x] Gemma 4 assistants (E2B and 26B-A4B, `--drafter <dir>`): query-only layers over the target's KV, centroid
      LM head; drafts equal transformers' after every parity prompt (E2B)
- [x] A drafter checkpoint as a pack of its own (`convert <drafter> --target <model>`: an assistant, a block
      drafter); Gemma 4 26B-A4B with its assistant at q4: 88.5 to 96.2 tok/s on the B60, the same drafts accepted
- [ ] Adaptive k from measured acceptance and per-k step cost (MoE verify cost)
- [x] Spec equivalence tests: greedy spec on vs off on a stateful tiny model with a partly wrong drafter; the real
      models' greedy output; rejection sampling's distribution
- [x] Verify cost: the q4 GEMV on the matrix engines from 4 rows (`gemv_quant_xmx`) costs about one row's (Qwen3.8
      on the B60, T = 4 / 8: 55.1 / 59.0 ms against 44.8 at T = 1); Qwen3.8 with MTP 42.1 tok/s (B60), 15.9 (B390); MoE
      verify on the direct path up to 64 (token, expert) pairs
- [x] An MTP head's drafts over a shortlist of the vocabulary: a one-bit sketch of the lm_head scores every row,
      the 64 best get their exact logits (docs/decisions/0011): a draft step 9.8 to about 4.5 ms on the B390, the
      same acceptance
- [x] Greedy decoding reads each row's argmax, not its logits (`Session::run(..., only)`; Qwen3.8 with MTP on the
      B60 42.8 to 49.5 tok/s); a verify's inputs kept for rollback with one launch a slot
- [ ] Pending experiment: the default group of q4 (32 or 128). Qwen3.8-27B at g128 reads 15% fewer bytes (B390 with
      MTP 19.8 against 18.0 tok/s, B60 52.9 against 49.5) and loses 5% of its accepted drafts, which come from the
      target and not the drafter (design-spec-decode 6.1); 25 MMLU-Pro questions cannot tell the two apart (80% / 76%).
      Measure Gemma and a larger evaluation before changing `convert`'s default
- [ ] A recurrent state that lags a step behind (no replay after a rejected draft); x prepared for the matrix-engine
      GEMV by the kernel that produces it; MoE verify reading each expert once; q8 on the matrix-engine GEMV; fewer
      host round trips per round (E2B is bound by them)
- [x] Block drafters: DFlash, DFlash 2 and DSpine checkpoints beside a Qwen3.5-family target (`--spec dflash|dspine
      --drafter <dir>`; `convert <drafter> --target <model>` for a q4 / q8 drafter pack), a block of drafts from one
      drafter pass over state the target's programs write; parity with the drafters' PyTorch code on tiny checkpoints
      (docs/decisions/0012). Qwen3.8-27B q4 with DFlash 2 at q4, 7 drafts: 60.7 tok/s on the B60 (MTP 49.9), 20.1 on
      the B390 (MTP 18.3; 22.8 and 19.8 with the q4 g128 target); DSpine 50.6 and 17.5
- [ ] The matrix-engine GEMV past 8 rows (a verify of a whole 16-token block); a block drafter's head over a
      shortlist; sampled chains from a block drafter's candidates
- [ ] Spec stats in logs and `/metrics`
- [ ] Bench at 32k / 64k on B390 and B60 vs the GenAI baseline with the same drafter
- **Done when:** equivalence tests pass and decode tok/s at 32k/64k beats the GenAI baseline with the same drafter.

---

## Phase 4: SYCL backend (same model, own kernels)

- [x] Toolchain: oneAPI DPC++ build in CMake; device selection; USM allocation; in-order queues (2026-11-07, oneAPI 2026.2)
- [ ] Bandwidth probe in SYCL (matches 0.2)
- [x] int4 / int8 GEMV for decode; dequantize + GEMM for prefill; quantized gather (device suites, groups 32 / 128)
- [x] int4 / int8 GEMV at ≥ 85% of measured bandwidth (B60: a plain read streams 426 to 453 GB/s; the q4 GEMV
      410 to 447 GB/s on 17408 x 5120 and larger, 357 to 412 on 6144 to 12288 rows): codes read as 32-bit words, q4
      decoded from its bits (no int-to-float conversion), two rows per sub-group at M = 1; quantized matmuls that
      read the same x run as one launch (gate + up, q + k + v); float GEMVs under 512 rows split K
- [x] Quantized prefill (2026-11-09): the f16 copy of a weight written with coalesced stores (B60: 220 to about
      400 GB/s; a work item per 32 columns wrote 64 bytes apart); quant_gemm_tiled, a matrix-engine GEMM that
      dequantizes each weight tile once per 512-row work-group into local memory, runs on integrated GPUs up to 512
      rows (B390, 17408 x 5120 q4 at 512 rows: 3.1 ms against 4.0 for dequantize + oneMKL); on the B60 dequantize +
      oneMKL ties it at 512 rows and wins elsewhere, so discrete cards keep it (sycl_be::quant_path). Qwen3.8-27B q4
      on the B60: 630 to 838 tok/s at 512 tokens, 763 to 945 at 4,096. The bar: OpenVINO GenAI 1,255 tok/s at 512
      with its default int8 activations (1,086 with f16 activations, as ours)
- [ ] A matrix-engine GEMM over q4 weights at oneMKL's efficiency: oneMKL's f16 GEMM reaches 91 TFLOPS on the B60 at
      512 rows; quant_gemm_tiled 60, of which the structure alone (no dequantization) 74. ESIMD (explicit registers,
      no unpacking moves around packed f16 math) dequantized faster but needs the large register file per kernel,
      which ESIMD kernels ignore as a property (a global build flag only)
- [ ] Per-regime planning in the SYCL backend: the f16 value plan (plan_half_values) and the residual-GEMM fold
      (mm_add_) need static shapes, so a language model's prefill (a symbolic T) keeps every value in f32 and every
      residual add separate. Planned per T regime (the GEMV's rows, the GEMM's) they would reach LLM prefill: about
      30 ms of a 512-token Qwen3.8 prefill on the B60 (the f32 traffic of SwiGLU's Mul, the adds, conv and slices)
- [x] MoE expert matmuls over q8 / q4 banks: one GEMV per (token, expert) up to 64 pairs (decode, verify), pairs grouped by
      expert up to 256, grouped oneMKL GEMMs on dequantized banks beyond (prefill)
- [ ] MoE prefill without the f16 copy of each expert bank (a GEMM that dequantizes in registers); the expert union
      at verify width k+1
- [ ] Paged decode attention: head dims 256 / 512, GQA, sliding window, chain and **tree** masks
- [x] Prefill: oneMKL GEMM (f16 x f16 -> f32); prefill attention through two batched GEMMs per query chunk
- [ ] Prefill attention on XMX without the score matrix (flash attention, e.g. SYCL*TLA)
- [x] Gated delta at any length: a sub-group per (v head, 16 state columns), a lane per column with all of its rows
      in registers, no barrier between tokens (Qwen3.8-27B on the B60: 99 to 33 ms of a 512-token prefill; decode as
      fast as the per-(head, columns) work-group kernel, which stays for 256-row heads)
- [ ] Gated delta prefill in chunks (WY form, matrix engines): the per-token chain is now 33 ms of a 512-token
      Qwen3.8 prefill on the B60
- [x] Gated delta decode (up to 8 tokens): a work-group per (v head, 16 state columns), state rows coalesced
      (Qwen3.8-27B: 41 to 19 us per layer on the B60)
- [ ] Decode beyond the GEMVs (Qwen3.8-27B on the B60, measured by skipping op kinds: of a 44.8 ms token the norms
      take 2.1 ms, gated delta 1.4, the Mul, Softplus and Sigmoid kernels 1.4; the q4 GEMVs stream at 96 % of the
      bandwidth). A one-row RmsNorm costs 12 us in the engine where an isolated kernel with a sub-group reduction and
      one barrier takes 4: unexplained so far
- [x] Norms, RoPE, activations, embedding gather; f16 GEMV for decode; split-K decode attention (sampling stays on host)
- [x] SYCL command-graph replay measured (decode, the ops between stateful ones replayed): host submission of a
      Qwen3.8-27B step 4.0 to 1.3 ms, device time unchanged (45 ms, device-bound; a dependent small kernel costs the
      GPU under 1 us). Not adopted; revisit for small models or if the host becomes the bound
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
- [x] FP8 checkpoints (`Qwen/Qwen3.8-27B-FP8`, `Qwen/Qwen3.6-35B-A3B-FP8`): E4M3 codes with a scale per 128 x 128
      block read as one more stored format of a float weight (`DType::F8`, `quant::read_rows`), expert banks stored
      expert by expert joined without a copy; `run`, `serve`, `bench` and `convert` take the directory
      (`docs/decisions/0014`)
- [ ] Qwen3.5: state snapshots at prompt ends for multi-turn prefix reuse; vision tower; a bf16 checkpoint with
      per-expert tensors (`experts.<e>.gate_proj`; only FP8 ones are joined)
- [x] LFM3 (`lfm3`, text): gated short convolutions and gated attention from the shared blocks (a per-head gate added
      to `blocks::attention`; no new op), its tokenizer and chat template byte-exact, the `qwen3_5` output parser;
      HF parity on cpu-ref and the B60, q8 / q4 packs (`docs/decisions/0013`)
- [ ] LFM3: the vision and audio variants (`lfm3_vl`, `lfm3_audio`, `lfm3_omni`); conv-window snapshots for prefix
      reuse after a turn with reasoning; the small ops of a q4 decode step (119 norms and 48 residual adds rank
      right after the matmuls under `XS_SYCL_PROFILE`)
- [ ] Qwen 3 dense and MoE: the "add a model in one PR" test; write `docs/adding-a-model.md` from the experience
- [ ] Llama 3
- [ ] Import MLX-affine packs
- [x] Import GGUF files (`convert <file.gguf> <out-dir>`): architecture `qwen35`, tensors kept in their block formats
- [x] GGUF: architecture `gemma4` (Gemma 4 E2B / E4B, text): per-layer embeddings and KV-shared layers, the `gemma4`
      tokenizer (BPE with byte fallback) and its end-of-turn stop tokens; equal to the original token by token and in
      every tensor of unsloth's BF16 file
- [x] GGUF block formats on the GPU: all 19, kept in the file's bits (decision 0016); every file of
      unsloth/Qwen3.5-4B-GGUF runs at its size
- [x] GGUF: the Gemma 4 assistant (`gemma4-assistant`) as a drafter, with the whole-vocabulary head llama.cpp
      computes (the file has no centroid head)
- [x] GGUF: a `qwen35` file without its MTP head, with the head's own file (`convert --mtp`, llama.cpp's draft
      model): the pack a file with the head gives, byte for byte
- [x] GGUF: architecture `clef` (ggml-org/Clef-Flash-GGUF, ggml-org/Clef-GGUF): the decoder read as a `qwen35`
      file's under `clef.*`, the joint schema head as a pack of the checkpoint holds it, its scales as the file stores
      them; a tiny file answers as the checkpoint its blocks decode to
- [ ] GGUF: Clef's files measured end to end (decision parity and the 25 test decisions against the checkpoint, on
      the GPU; results in docs/gguf_models.md)
- [ ] GGUF: more architectures (`qwen35moe`, Gemma 4 26B-A4B (`gemma4` with experts), LFM3)
- [ ] GGUF: a Gemma 4 vision tower from llama.cpp's mmproj file
- [ ] GGUF: expert banks in block formats (the expert kernel reads q8 / q4 only)
- [x] GGUF: a matrix-engine GEMV over block formats (`gemv_block_xmx`, from 2 rows: a speculative verify)
- [x] GGUF: Qwen3.8-27B's files that fit a 24 GB card (20 of unsloth's), with MTP; results in docs/gguf_models.md
- [ ] GGUF: one six-prompt bench of Qwen3.8-27B UD-Q6_K_M (1.5 GiB left beside its weights) lost the device once and
      not in eight runs after; check what a step allocates when the card is that full
- [ ] Integrated GPUs: drop a pack's file pages from the page cache as its weights upload (on a 30 GB Arc B390 machine
      a 16 GiB pack lost the device in 2 of 8 loads while the driver still held the previous run's pages)
- [ ] GGUF: the 9 to 64 row kernel over block formats (past 8 rows the weight is dequantized for a GEMM)
- [ ] GGUF: Q4_0 at its file's size (its q4 form stores a bias per group it does not need)
- [ ] GGUF: `serve <file.gguf>` (an import into a cache), split files
- [ ] GGUF: the tiled value-head order inside GatedDelta (four small transposes per linear layer now: 1 % of a
      decode step of Qwen3.5-4B at f16)
- [x] Data-aware quantization: AWQ-style scale and clip search in `convert`, calibrated on our own engine
- [ ] GPTQ; AWQ with the full activation covariance (not per channel); larger calibration sets
- [ ] Anthropic `/v1/messages`, Ollama `/api/chat` and `/api/generate`
- [ ] Constrained decoding (JSON schema / grammar), logprobs
- **Done when:** each item lands as one PR with its parity and E2E tests.

### Model enablement checklist

Every model below is enabled the same way (CLAUDE.md 8.1); a model is done when every box is ticked:

1. Config parsed, unsupported variants refused by name (`models/<arch>/<arch>_config.cpp`, config test)
2. Graph built from shared blocks (`models/common/`), new ops added to the IR, cpu-ref and SYCL first
3. Weights load from the Hugging Face checkpoint (single and sharded)
4. Parity vs Hugging Face fp32: logits cosine and greedy output on cpu-ref (f32) and SYCL (f16)
5. Tokenizer and chat template byte-exact vs Hugging Face; output parser (thinking, tool calls) if the format needs one
6. Packs: `convert` q8 and q4 (AWQ), quality on the parity prompts and MMLU-Pro (25 questions) through the server
7. Fits and runs on the target device (Arc Pro B60, Panther Lake iGPU); bench decode / TTFT at 1k / 4k / 16k
8. Docs: TODO, CLAUDE.md map, README supported list

| # | Model | Hugging Face ID | Notes |
|---|---|---|---|
| 1 | Qwen3.8-27B | `Qwen/Qwen3.8-27B` | Dense `qwen3_5`. **Runs** (text): tiny-checkpoint HF parity with its ratios (cosine 1, greedy exact on cpu-ref and SYCL); q4 g32 pack 19.2 GB (16.2 GiB on device) on one Arc Pro B60: decode 22 tok/s, prefill 831 tok/s at 512 tokens and 945 at 4,096 (1,058 with --prefill-chunk 4096; 630 / 763 before the 2026-11-09 prefill work), MMLU-Pro 76 % (25 q; 3 answers hit the 4,096-token cap); q4 g64: 23.8 tok/s; with its MTP head (`--spec mtp --spec-k 4`): 49.5 tok/s over six chat prompts (3.20 tokens per verify; 52.9 at q4 g128, k = 5). Arc B390: decode 6.9 tok/s, 18.0 with MTP (k = 5; 19.8 at q4 g128), prefill 460 tok/s at 512 tokens (307 before), 270 to 385 at 4,096 (noisy, run to run). q4 (RTN) vs HF fp32 last-token cosine 0.67 to 0.99, as Qwen3.5-4B's q4 (0.87 to 0.99). Against OpenVINO 2026.4.1 GenAI (optimum-intel int4: asymmetric, group 128, int8 head; `llm_bench`, no MTP, prefix cache off), our q4 at the same group size decodes 24.9 vs 23.4 tok/s on the B60 (24.1 vs 23.0 at 4k) and 7.6 vs 7.65 on the B390; OpenVINO prefills 1.5x faster on the B60 at 512 tokens (1,255 vs 831 tok/s; with its int8 activations: 1,086 with f16 activations, as ours) and 1.3x at 4,096 (1,251 vs 945), 1.6x on the B390 at 512. Also from `Qwen/Qwen3.8-27B-FP8` (`docs/decisions/0014`): the same pack layout, speed and MMLU-Pro score. To do: prefill (below), AWQ, vision |
| 2 | Qwen3.6-35B-A3B (preferred) | `Qwen/Qwen3.6-35B-A3B` | MoE (3B active). **Runs** (text): tiny-model HF parity; q4 g64 pack 18.9 GiB on one Arc Pro B60, decode ~58 tok/s, prefill ~1,250 tok/s, MMLU-Pro 84 % (25 q); with its MTP head: 85.6 tok/s at k = 2 (59.9 without); the q4 g32 pack against HF fp32 on the real checkpoint (2026-11-11): last-token cosine 0.92 to 0.98 on six prompts, MMLU-Pro 80 %. Also from `Qwen/Qwen3.6-35B-A3B-FP8` (per-expert FP8 tensors; `docs/decisions/0014`): the same pack layout and speed, MMLU-Pro 76 %. To do: Panther Lake, AWQ, vision |
| | Qwen3.5-35B-A3B (older) | `Qwen/Qwen3.5-35B-A3B` | MoE; the Qwen3.5 Gated DeltaNet hybrid |
| | Qwen3.5-9B | `Qwen/Qwen3.5-9B` | Dense `qwen3_5` (32 layers of 4,096, an untied head). **Runs** (text) with no code change: HF fp32 parity of the real checkpoint on six prompts of up to 1,691 tokens (cpu-ref f32: cosine 1, max logit error 5e-5, 32 greedy tokens and the MTP head's drafts equal on all six; Arc Pro B60 f16: cosine 0.999998 or better, greedy equal); 16.7 GiB on the device at f16. Packs: q8 10.4 GB (cosine 0.997 to 0.99996, greedy equal on six of six), q4 g32 7.1 GB (0.90 to 0.99, greedy equal on three of six). Arc Pro B60: decode 21.4 / 38.2 / 61.5 tok/s (f16 / q8 / q4), prefill 3,250 tok/s at 1,024 tokens (f16); with its MTP head at q4 62 to 108 tok/s (k = 4). MMLU-Pro (25 q, thinking off): 72 / 76 / 76 % (f16 / q8 / q4). To do: Panther Lake, AWQ, vision |
| 3 | Gemma 4 31B | `google/gemma-4-31B-it` | vision tower uses `standardize` |
| 4 | Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it` | MoE (4B active). **Runs** (text): tiny-checkpoint HF parity of its layout (experts beside the MLP with per-expert scales, K = V global layers with 2 KV heads; cosine 1 on cpu-ref and SYCL f32); q4 g32 pack 17.3 GB on one Arc Pro B60: decode 65.6 tok/s, 84.2 with its assistant at k = 2. q4 vs HF fp32 last-token cosine 0.95 to 0.9997. To do: images (bidirectional attention over image tokens), AWQ |
| 5 | Nemotron 3.5 Lightning (30B-A3B) | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | MoE (3B active) |
| | Nemotron 3.5 Super (120B-A12B, larger) | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | MoE + vision; int4 and two GPUs |
| 6 | Muse Glimmer 30B | `meta-models/Muse-Glimmer-30B` | |
| 7 | Gemma 4 12B | `google/gemma-4-12B-it` | |
| 8 | LFM3-1.7B | `LiquidAI/LFM3-1.7B` | Dense `lfm3` (13 gated short convolutions, 6 full and 5 sliding attention layers). **Runs** (text): HF fp32 parity of the real checkpoint on cpu-ref and the Arc Pro B60 (f32 cosine 1 and greedy exact on six prompts up to 1,770 tokens; f16 max logit error 0.013; int8 KV cosine 0.99998) and of a tiny checkpoint; tokenizer and chat template byte-exact; thinking, tool calls and streaming through the server. Arc Pro B60: decode 99 / 159 / 206 tok/s (f16 / q8 / q4 g32), prefill 13,500 tok/s at 1k tokens, 10,000 at 16k, 5,200 at 64k (decode 68 to 107 tok/s there); packs 3.46 GB to 1.78 (q8) and 1.20 (q4). q8 vs HF fp32 last-token cosine 0.9998, q4 0.98 to 0.995 (AWQ the same). MMLU-Pro (25 q, thinking on): 60 % f16, 64 % q8, 52 % q4, 60 % q4 AWQ, with 1 / 0 / 4 / 5 answers at the 4,096-token cap: q8 is the pack to serve. To do: Panther Lake |
| 8 | Liquid LFM3-1.7B (main text model) | `LiquidAI/LFM3-1.7B` | small |
| 9 | Flint | `RecursiveSI/flint` | |

---

## Robot policies (VLA) — docs/design-vla.md

- [x] Policy kind: `models/policy.h`, `model_kind`, `load_policy`; `src/policy/` runtime (programs per camera subset
      and flow steps, warm-up, timings); `xpu-serve bench <policy-dir>`
- [x] π0.5 (`lerobot/pi05_base`): parity with LeRobot f32 (cpu-ref 5e-6, Arc B390 f16 1e-3), 1 / 2 / 3 cameras
- [x] π0.5 latency, first pass: f16 operands, fusions, fused flash attention (XMX) up to head_dim 128 with SigLIP
      per camera, Concat as one kernel, a cheaper GELU, prompt padded to 32-token buckets (Arc B390, 3 cameras:
      428.5 -> 243.9 ms; design-vla.md §6)
- [x] π0.5 fusions (Arc B390, 3 cameras): residual adds folded into the prefix's GEMMs, written in the residual's
      memory (beta 1; 35 passes over [928, 2048] gone, -4 ms), quant_gemm inputs kept in f16 by their producers (~900
      conversion kernels gone, -2 ms), the expert's K / V concats read in place by the attention GEMMs (about even:
      the copies were cached, the GEMMs doubled). q8 action expert 231 -> 225 ms, f16 244 -> 239 ms
- [x] π0.5 latency, second pass (design-vla.md §6.1; a second Arc B390 laptop, 3 cameras: f16 227.4 -> 215.2 ms,
      q8 expert 216.9 -> 198.0; Arc Pro B60: 124.0 -> 114.7 and 131.2 -> 113.1): the last prefix layer stops at its
      keys and values, a plain GEMM for one KV head, attention in cache-sized chunks of queries on integrated GPUs,
      SigLIP's biases where they cost no pass (key bias dropped, value bias into the output projection's, residual
      biases added by the layer norms), RoPE's angle once per token and pair, the cameras' attention in one launch
      (`AttentionAttrs::block`), and for the q8 expert the few-row kernel's loads and sibling launches. Tried and
      dropped: gate and up in one quant_gemm with the activation, and the gated residual in its epilogue (720
      kernels fewer per call, within noise: the Arc B390 pipelines small kernels, under 1 us each unprofiled;
      XS_SYCL_PROFILE's barriers make them look 10x dearer); the gated MLP in column blocks (5.5 ms slower on the
      Arc B390); the expert's keys and values made in place beside the prefix's (within noise)
- [ ] π0.5 latency, next: the prefix MLP's intermediates (gate and up write 30 MB each that the product reads back;
      a GEMM that multiplies them in registers), its down projection's GEMM (~30 TFLOPS against ~50 for gate / up;
      oneMKL split-K is slower), attention at head_dim 256 without scores in memory, the fixed cost of a few-row
      launch (8 to 12 us on the Arc Pro B60). The Arc B390 is bound by bytes moved (about 105 GB/s; the f16 weights
      are 10.7 GB a call)
- [x] Generic weight quantization by a graph rule (`convert --scope bandwidth|all`): weights of matmuls with a
      step-dependent row count or at most 64 rows per call; policies convert too. π0.5 q8 (the rule picks the action
      expert): 245 -> 231 ms at 3 cameras on Arc B390, actions cosine 0.99999; q4 misses the 0.999 gate
- [x] SYCL quant_gemm: quantized matmuls of 9 to 64 rows read the q8 / q4 weight once and dequantize in registers
      (DPAS and 2D block load builtins), 2.5 to 4x the dequantize + GEMM path
- [x] quant_gemm, second pass (Arc B390, M = 9 to 64): K split between the sub-groups of a work-group (partial sums
      added in local memory instead of written out), scales loaded once per group, q8 codes to f16 two at a time,
      one 16-feature column at 8 row tiles (no spills): q4 1.1-2.7x, q8 at 64 rows 1.4x, q8 at 9 rows 1.1x. Staging
      x in local memory and prefetching codes a step ahead measured no faster (x tiles were not the limit)
- [x] quant_gemm, third pass: 64 bytes of codes a lane and 32 rows of x a load, B built a slice at a time, one
      feature tile past 32 rows (q4 no longer spills at 7 row tiles: 60 -> 8.5 ms for 0.9 GB at 50 rows on the Arc
      Pro B60; 9 to 32 rows 15 to 18 % faster, q8 8 to 13 %), up to four weights of one x in a launch, and used up to
      64 rows on discrete cards too (`--backend-opt few_rows=N`)
- [ ] quant_gemm's arithmetic: at q8 the dequantization (not the loads) bounds a sub-group, about 1 us per 16
      columns of 32 features on the Arc Pro B60; the GEMV's trick (codes as 1024 + c, scales applied to group sums)
      would remove most of it. A launch's fixed cost (8 to 12 us) is the other half at a policy's sizes
- [ ] q4 for policies: AWQ-style calibration on policy activations (or q4 with q8 for the most sensitive matmuls)
- [ ] Flash attention at head_dim 256 (Gemma): 8 rows per sub-group fill the registers, so every K/V tile feeds one
      DPAS and oneMKL's GEMMs match it; needs K/V or P shared across sub-groups through local memory. Tried on Arc
      B390 (pi05, 2026-11-09): the joint_matrix kernel with one 8-row tile per sub-group and the 8 query heads of a
      KV head in one work-group spilled 2 KB (Q in registers), 450-580 bytes with Q read from local memory (32 or 16
      keys per block), and ran 10-15 ms slower per call than the GEMM path; it needs hand-placed DPAS (as quant_gemm)
- [ ] Low precision on Panther Lake: the B390 compiles fp8 (E4M3 / E5M2) DPAS but returns wrong results; int8 DPAS is
      exact at ~122 TOPS (2x f16). A w8a8 path is the low-precision option there (lossy, opt-in)
- [ ] Normalization statistics (quantiles, mean/std) and resize-with-pad for other image sizes
- [x] Policy service and API (JSON): `xpu-serve serve <policy-dir>` mounts `/v1/policy/info`, `/v1/policy/act` (base64
      PNG / JPEG or raw RGB, fixed noise optional, requests fail by name), a latest-wins one-slot queue (a waiting call
      is answered 409 when a newer one arrives) and `/ready` after warm-up (`--cameras N`). pi05 on Arc B390: actions
      identical to policy-eval, 199-218 ms per call at 2 cameras (~20-30 ms of it HTTP and JSON)
- [ ] Policy API next: msgpack bodies (no base64), the openpi-compatible WebSocket, RTC (previous chunk + delay)
- [ ] More policies: π0, SmolVLA, GR00T N1.6 / N1.7, π0-FAST (through the language engine), OpenVLA-OFT

## Decision models (the System One API) — docs/decisions/0015

- [x] `DecisionModel` as a third model kind; the runtime runs the steps a model names (`src/decision/`)
- [x] `POST /v1/systemone` (TypeSafe's System One, OpenAPI 0.2.0): noul, choice, score; the API's 422 / 400 bodies;
      `GET /v1/models` in its shape and OpenAI's; `Server-Timing`
- [x] Clef-Flash (`Cloudflare/clef-flash`, Qwen3.5-9B + a joint schema head): its prompt token for token, the head as
      four programs of the decoder's session; parity with its Python code (tiny and released checkpoints), 25 test
      decisions end to end on the Arc Pro B60
- [x] `run --request`, `bench` (latency, `--state-tokens`), `tests/e2e/systemone_eval.py`
- [x] pplx-decider-flash-v1.2 (`perplexity-ai/pplx-decider-flash-v1.2`, Qwen3.5-4B): whole-prompt attention in the
      full-attention layers, a read-out at a marker per question, the checkpoint's temperature; long requests split
      by question; parity with its package (tiny and released checkpoints), 25 test decisions end to end
- [ ] pplx-decider past a few thousand tokens: its single step is slower per token than a chunked prefill
- [x] OpenJev (`openjev/openjev`, Qwen3.8-27B) and any chat model, `--readout openjev|letters`: a prompt per question,
      the logits of the option letters, its helper's calibration and many-option composition; the state read once
      (the last step undone whole: `Session::truncate`); parity with the helper on a tiny checkpoint, the q4 pack
      against the checkpoint in fp32 on 25 test decisions
- [ ] Decisions and chat from one server for a language model read with `--readout`
- [ ] OpenJev's question parts of 65 to about 200 tokens take about 345 ms on the Arc Pro B60 where parts of up to
      64 take about 140 ms: above `few_rows` (64 at most) a q4 step dequantizes every weight for its GEMMs
- [ ] Images (`assets`, box questions): needs the Qwen3.5 vision tower
- [x] Packs of decision checkpoints (`convert`): q8 keeps the answers, q4 changes one in eight on 300 MMLU-Pro
      questions (answers the model is unsure of); slower than f16 on the Arc Pro B60
- [ ] AWQ for decision checkpoints (calibrating on decision prompts): tried through the language-model path, a
      quarter to a third less probability shift at q4
- [ ] A prefill from quantized weights that does not dequantize every weight each step: it costs a Clef-Flash pack
      50 to 65 ms a step on the Arc Pro B60, which makes packs of decision models slower than f16
- [ ] Batching several requests into one step
- **Done when:** each model answers as its reference code does and a System One client cannot tell the servers apart.

---

## Embedding models (`POST /v1/embeddings`) — docs/decisions/0017

- [x] `EmbeddingModel` as a fourth model kind (a sentence-transformers checkpoint: `modules.json` with a Pooling
      module); the pipeline read as sentence-transformers reads it (mean or CLS pooling, Normalize when listed, the
      longest input), everything else refused by name; the runtime and its service (`src/embedding/`)
- [x] BERT (all-MiniLM-L6-v2, bge-small-en-v1.5) and NomicBERT (nomic-embed-text-v1.5, read to 8192 tokens after
      training on 2048): parity with sentence-transformers on cpu-ref (tiny and released checkpoints, up to 8192 tokens)
- [x] `POST /v1/embeddings` as OpenAI's (text, token ids, float or base64, OpenAI's 400s naming the field; `dimensions`
      only the model's own; over-long inputs refused); `run --prompt`, `bench` (latency by length, throughput)
- [x] Text through the checkpoints' WordPiece tokenizers: each text's ids are sentence-transformers'
      (`embedding_parity.cpp` with `XS_EMBEDDING_TEXTS=1`; bge's `do_lower_case` as sentence-transformers 6 applies it)
- [ ] Parity and timing on the Arc Pro B60 (f16 and f32)
- [ ] Several inputs of a request in one step (padded with a key bias, or Attention's `block` for inputs of one length)
- [ ] Shorter embeddings (`dimensions`) for checkpoints that declare them (Matryoshka)
- [ ] The rerank and score APIs (cross-encoders)
- [ ] Packs of embedding models (`convert` refuses them by name)
- **Done when:** a client of OpenAI's embeddings API gets sentence-transformers' vectors from the GPU at a speed
  measured against vLLM's.

---

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
