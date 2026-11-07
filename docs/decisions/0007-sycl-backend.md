# 0007: SYCL backend (experimental) and the first comparison with OpenVINO

Date: 2026-11-07 · Status: proposed (SYCL as the main GPU backend; OpenVINO kept for CPU, NPU and as a second opinion)

**What it is.** `src/backend/sycl/` runs the step graph op by op on an in-order Level Zero queue (oneAPI 2026.2:
DPC++ and oneMKL). Our own kernels cover every op: elementwise, norms, RoPE, gather, copies, causal conv, the gated
delta rule, decode attention and an f16 GEMV for decode; oneMKL's XMX GEMMs take matmuls of more than 8 rows and the
two products of prefill attention. Weights and the KV cache are f16 (f32 at f32 precision), activations f32. Memory is
one arena per step, planned once per T from value lifetimes; KV slots grow by doubling; conv and recurrent state are
double-buffered so a failed step leaves the sequence as it was. The model code and the graph did not change.

**Correctness.** The backend op suites (fp64 references, planted-bug controls) pass on the Arc Pro B60 at f32 and f16.
Gemma 4 E2B and Qwen3.5-4B match Hugging Face fp32: at f32 cosine 1.0 and greedy output identical on every prompt; at
f16 cosine >= 0.99999, greedy identical.

**Speed.** Intel Arc Pro B60 discrete GPU, f16, `xpu-serve bench`, prefill in 1024-token chunks, steady-state run of
three; OpenVINO 2026.4.1 against the SYCL backend. TTFT is the prefill time; decode in tokens per second.

| Model | Prompt | TTFT OpenVINO | TTFT SYCL | Decode OpenVINO | Decode SYCL |
|---|---|---|---|---|---|
| Gemma 4 E2B | 128 | 58 ms | 35 ms | 57.5 | 61.5 |
| | 1024 | 178 ms | 153 ms | 53.9 | 59.8 |
| | 4096 | 861 ms | 693 ms | 44.3 | 58.7 |
| | 16384 | 6.29 s | 3.21 s | 26.3 | 51.5 |
| Qwen3.5-4B | 128 | 63 ms | 44 ms | 39.2 | 41.9 |
| | 1024 | 193 ms | 222 ms | 38.8 | 41.5 |
| | 4096 | 755 ms | 924 ms | 38.1 | 41.1 |
| | 16384 | 3.86 s | 4.50 s | 34.3 | 38.7 |

Where the differences come from:
- Gemma's sliding-window layers (28 of 35, window 512): the SYCL kernels read only the keys in the window, in decode
  and in prefill (each query chunk's GEMMs cover just its window); our OpenVINO graph hands attention the whole cache
  and masks it. Long caches on the global layers are split across work-groups (flash decoding).
- Decode GEMV reads f16 weights at 340-450 GB/s (the B60's peak is about 456 GB/s); what is left per step is mostly
  small ops, which the act(x)*gate and conv+SiLU fusions cut.
- Qwen3.5 prefill still trails: the gated delta rule runs token by token (about 49 ms of a 1024-token chunk) and
  prefill attention materializes the score matrix. A chunked delta rule and XMX flash attention are the next steps.

**Runtime lessons** (each guarded in code):
- oneMKL orders its kernels by the dependency list, not by the queue being in-order: every call gets a barrier
  before (as its dependency) and after (`ordered()` in `sycl_matmul.cpp`). Without it the GEMMs raced the kernels
  around them and the device was lost.
- A single 1.6 GB copy from pageable host memory lost the device; weights upload in 256 MiB pieces.
- The memory planner must keep every graph input live from the start of the step: inputs are uploaded before the
  first op runs.
- `XS_SYCL_SYNC=1` (wait after every op, name the failing one) found each of these in minutes.

**Why SYCL as the main GPU backend.** Measured speed above; full control over kernels (every OpenVINO issue so far
needed a workaround tied to plugin internals: the slow internal GatedDeltaNet op, the one GQA pattern the GPU plugin
fuses, per-length kernel rebuilds); multi-GPU (OpenVINO's GPU plugin has no tensor parallelism and its HETERO pipeline
split is fragile, while a SYCL process can schedule both GPUs of a dual-B60 system itself); and DPC++, Level Zero and
oneMKL are the layer the rest of Intel's stack is built on. Costs: kernels are ours to write (XMX attention,
quantized GEMV), and the Panther Lake iGPU is not validated on SYCL yet.

Triton for Intel GPUs (intel-xpu-backend-for-triton) was considered for kernels: it is a Python JIT whose ahead-of-time
path for C++ hosts is unfinished, so it would add Python and an unstable kernel ABI to the build. SYCL*TLA (CUTLASS
for SYCL, header-only C++, flash attention tuned for Battlemage) is the candidate for the XMX kernels instead.
