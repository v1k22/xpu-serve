# 0031: GGUF files by their path

Status: **accepted** · 2026-11-27

## Context

`serve`, `run` and `bench` took a model directory or `-hf`. A GGUF file on disk went through `convert` into a
directory of the user's first (decision 0016 left `serve <file.gguf>` as "a thin layer on top"), where llama.cpp's
`-m file.gguf` serves the file itself. `-hf` already imported a repository's file into xpu-serve's cache, with the
companion files its import reads (an MTP head or drafter, a vision tower), on a path of its own.

## Decision

- **One path from a model spec to a directory** (`src/hub/model_source.cpp`), for every command: a spec (a directory,
  a `.gguf` file, or `-hf`, with `--mtp` and `--mmproj`) becomes the files it names (`source_files`: a checkpoint's
  directory, or a GGUF file and its companions), and they become the directory the command reads (`model_dir`).
  `serve`, `run`, `bench`, `--drafter` and convert's source all go through it; `convert` writes a GGUF file's files
  into the directory it is given instead of the cache.
- **A file on disk is imported into xpu-serve's cache in `-hf`'s layout**: `packs/local/<path>/<name>` where `-hf` has
  `packs/models--<org>--<name>/<commit>/<name>`. `<path>` is 12 hex digits of the SHA-256 of the model file's
  canonical path; `<name>` is the model's (the file name without `.gguf` or a part's `-00001-of-00003`), with
  `+<head>` and `+<tower>` for the companions, as `-hf` names its packs.
- **The key: canonical path, size and modification time.** The pack holds a stamp (`.source`) of every part of every
  file imported into it (docs/decisions/0030): its canonical path, size and modification time. The pack is current
  while the files have them still; otherwise the file is imported again, in the same place (in a partial directory
  renamed over the old one, under the lock `-hf` takes). A start costs a stat a part. A rewrite of a file moves its
  time and a cut changes its size; the canonical path makes a link or a relative path to a file the same pack, and
  resolves a file of the Hugging Face cache to its blob, which is named by its SHA-256. Not chosen:
  - a hash of the contents: every byte of a file of many gigabytes read at each start, longer than the import;
  - a hash of the header and the size: a few megabytes of vocabulary read, and blind to a file quantized again in the
    same layout (another importance matrix gives other bytes in the same tensors, of the same sizes, under a header
    that need not differ);
  - the key in the directory's name: every version of a file would keep a pack of its own, each the file's size.

  What the key misses: a file rewritten in place at the same size with its time set back (`touch -r`, `cp -p` over
  it). A process still serving the old pack keeps reading its files, which the new import unlinked.
- **`-hf`'s packs stay under the commit**, with no stamp: a commit's files never change. When the user names a
  companion with `-hf`, the files are not the repository's alone and the pack is filed as a file on disk's.
- **Companions as `-hf` takes them.** `--mtp` and `--mmproj` name a GGUF model's MTP head (or drafter) and vision
  tower, as `convert` takes them. Unnamed, a companion comes from the folder of the file and the folders in it, read
  as the repository it came from (`MTP/mtp-*.gguf`, `dspark-*.gguf`, `mmproj-*.gguf`): the same rules over that
  listing (`select_mtp_head`, `select_projector`), asked only when the file's import reads such a file
  (`lacks_mtp_head`, `reads_projector`, the predicates `-hf` asks). `convert <file.gguf>` takes them too, so it writes
  the pack `serve <file.gguf>` serves.

## Consequences

- A file on disk takes its size twice, the file and its pack, as `-hf`'s files do; a changed file's old pack is
  replaced, not kept.
- A folder that mixes several models' files can hand one model another's drafter or tower (by `-hf`'s rules, the only
  one there is taken). Its import refuses it by name (the head of another model, a tower that is not the model's);
  `--mtp` and `--mmproj` name the right one.
- The companion rules stay `-hf`'s: a model whose importer comes to read a file of its own (a vision tower) is found
  beside a file on disk by the same change that has `-hf` fetch it.
- Checked on an Arc Pro B60 with unsloth's Qwen3.5-4B Q4_K_M file: `serve <file>` imports it (2 s) and serves it, a
  second start does not import it again, a touched part of its split copy imports it again; the file by its path, its
  split copy and `-hf` give one pack, every file the same, and the same greedy answer; `run` and `bench` take the
  file, and Gemma 4 E2B's file drafts with its assistant's file given as `--drafter <file.gguf>`.
- Tests: `tests/unit/hub/test_model_source.cpp` (a file by its path: imports, companions, changes, split files,
  refusals) and `tests/unit/hub/test_hub_download.cpp` (`-hf` with a companion the user names).
