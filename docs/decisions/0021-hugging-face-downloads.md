# 0021: Models straight from the Hugging Face Hub (`-hf`)

Status: **accepted** · 2026-11-14

## Context

Every command took a model directory, so a user first downloaded the checkpoint (`hf download`, git, a script),
found which of its files mattered, and pointed xpu-serve at the folder; a GGUF file also went through `convert`.
llama.cpp's `-hf org/name:QUANT` and vLLM's `serve org/name` take a repository name instead. The machines this runs on
are often behind a proxy, and one connection through it can be slow (3.7 MB/s on the machine this was measured on,
where eight in parallel got 54 MB/s).

## Decision

- **`-hf org/name[@revision][:QUANT]`** on `serve`, `run`, `bench` and `convert`, in place of `<model-dir>`, and
  `-hfd` (`--hf-drafter`) in place of `--drafter <dir>`. `--hf` and `--hf-drafter` are the long spellings.
- **The Hugging Face cache, in its own layout** (`src/hub/hf_hub.cpp`): `models--org--name/blobs/<hash>` (the SHA-256
  of a file kept in LFS or Xet, the git object id of one kept in git), `snapshots/<commit>/<path>` as relative links,
  `refs/<revision>`, and huggingface_hub's lock files. A model huggingface_hub, `hf download` or another engine
  downloaded is found there, and one xpu-serve downloaded is theirs: `scan_cache_dir` reads it without a warning.
  Where the cache is follows huggingface_hub (HF_HUB_CACHE, HF_HOME, XDG_CACHE_HOME), as do the token (HF_TOKEN, the
  token file), the endpoint (HF_ENDPOINT) and offline use (HF_HUB_OFFLINE=1). When the Hub cannot be reached and the
  cache has the snapshot, it is used with a warning.
- **Downloads in ranges over eight connections** (`src/hub/download.cpp`): ranges of at most 64 MiB, at least eight
  a file (8 MiB at the least), handed out across all the files of a request; a file in progress is `<blob>.xs-partial` beside a list of the ranges written, so an interrupted
  download resumes with the ranges it lacks; a file is moved into the cache only when its size and SHA-256 are right.
  The names differ from huggingface_hub's `.incomplete`, which it would resume as a prefix. Proxies come from
  `https_proxy` / `http_proxy` / `no_proxy`. The token goes to the Hub's origin only, never to where a redirect
  leads (the CDN's signed URLs need none). TLS is OpenSSL's, through cpp-httplib: OpenSSL 3 is now a build
  dependency.
- **Which files** (`src/hub/select.cpp`): a checkpoint's safetensors and everything beside them but other runtimes'
  weights (PyTorch, ONNX, OpenVINO, TensorFlow, GGUF), pictures and media, and subfolders that hold weights of their
  own when the top does (other variants: Laya's `multilingual/`). A repository of GGUF files (no safetensors, or a
  `:QUANT`): the one file whose name ends in that tag, Q4_K_M by default (a drafter's `mtp-*.gguf`: Q8_0), the plain
  file before unsloth's dynamic `UD-` one of the same tag, never a vision projector or an importance matrix; a file
  split into parts is refused, as the import refuses it. Every refusal lists what the repository has.
- **A GGUF file is imported once** (`src/hub/model_source.cpp`) into a pack in xpu-serve's own cache
  (`~/.cache/xpu-serve/packs/<repo folder>/<commit>/<file>`, or XS_CACHE_DIR), built in a partial directory and renamed
  when whole. A `qwen35` file without its MTP head gets the head the repository publishes in a file of its own
  (`mtp-<the file's name>.gguf`, else the only one), as `convert --mtp` would.
- **Files a checkpoint reads from another repository** are a fact of the model: `missing_companion_files` in the
  model registry names them (π0.5 reads PaliGemma's tokenizer.json, which `lerobot/pi05_base` does not publish). The
  model's directory is then a folder of links in xpu-serve's cache: the snapshot's files and the companion. A
  checkpoint's weights download last, so a companion that cannot be had (a gated repository without a token) is
  refused before them.
- `serve -hf` names the model `org/name` (or `org/name:QUANT`) in `/v1/models`; any snapshot of the Hugging Face cache
  is named `org/name` by `cfg::model_id`.

## Measured

On the Arc Pro B60 machine, behind its proxy, from an empty cache (2026-11-14), every kind of model:

| `-hf` | What came down | Then |
|---|---|---|
| `Qwen/Qwen3.5-4B`, stopped after 45 s and run again | 1.11 GB, then the 8.25 GB it lacked in 209 s (39 MB/s) | `run` answers |
| `unsloth/Qwen3.5-4B-GGUF:Q4_K_M` | 2.74 GB in 84 s, imported | `run` answers |
| `BAAI/bge-small-en-v1.5`, `nomic-ai/nomic-embed-text-v1.5` | 135 MB; 547 MB in 20 s (27 MB/s) | embeddings |
| `convaiinnovations/laya` | 846 MB of the repository's 2.37 GB (its variants left) | `bench`: 19 ms |
| `ggml-org/Julia-1-GGUF:Q8_0` | 168 MB, imported | `bench`: 16 ms |
| `LiquidAI/d1-2B`, `LiquidAI/LFM3-1.7B` | 3.86 GB, 3.48 GB | `bench` 16 ms; `run` answers |
| `unsloth/gemma-4-E2B-it-GGUF:Q4_K_M` with `-hfd` of the same repository | 3.11 GB and the 98 MB drafter, both imported | 121 tok/s with the drafter |
| `unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS` | 7.21 GB in 215 s, its MTP head's 1.37 GB file, imported together | 51.7 tok/s with `--spec mtp` |
| `lerobot/pi05_base`, no token | 44 KB of small files | refused: PaliGemma's tokenizer is gated |

`serve -hf Qwen/Qwen3.5-4B` reported `Qwen/Qwen3.5-4B` in `/v1/models`; `HF_HUB_OFFLINE=1` ran it from the cache;
huggingface_hub 1.33 read the cache xpu-serve wrote (10 repositories, 32.3 GB) with no warning, and its
`snapshot_download(local_files_only=True)` found the snapshot. One connection through that proxy gets about 4 MB/s,
so a file is cut into at least eight ranges of 8 MiB or more (the first measurement of a 133 MB file in three 64 MiB
ranges took 25 s).

## Consequences

- A GGUF file takes its size twice on disk: in the Hugging Face cache, and as its pack.
- No `hf_xet` chunk deduplication: files are fetched whole through the Hub's resolve URLs, as huggingface_hub does
  without `hf_xet`.
- Tests: `tests/unit/hub/` runs the client against a Hub in the test process (`fake_hub.h`: the listing API, resolve
  URLs redirecting to a second server for the CDN, byte ranges, tokens, gated repositories, faults on demand).
