# 0030: GGUF files split into parts

Status: **accepted** · 2026-11-27

## Context

Large models are published split into parts, by llama.cpp's gguf-split or its converter's `--split-max-size`:
`<name>-00001-of-00003.gguf` and on, each part with `split.no`, `split.count` and `split.tensors.count`, the first one
holding the metadata (with tensors, or with none under `--no-tensor-first-split`), the tensors spread over all. The
reader, the import and `-hf` refused them: unsloth's BF16 file of Qwen3.8-27B is two parts, and so is every larger
quantization of a bigger model.

## Decision

- **A split model is one file to the reader** (`gguf::File`): opened by its first part, it maps every part, found
  beside the first by llama.cpp's naming (`llama_split_path`). The metadata is the first part's; the tensors are those
  of all parts in order, each with its part, its offset in that part's data section, and that part's alignment (a
  later part has only the split keys, so the default one, unless it says another).
- **Refused by name**: a later part given in place of the first (the error names the first), a first part not named
  as one (its other parts could not be found), a missing part, and parts that disagree: `split.no`, `split.count`,
  `split.tensors.count` and the tensors they hold in all, any other value of a later part that the first part does
  not hold the same, a tensor in two parts.
- **Nothing after the reader knows.** The import of the parts is the import of the whole file, byte for byte: the
  same tensors, config, tokenizer and template; its manifest names the file the parts were split from
  (`<name>.gguf`), as the whole file's does.
- **`-hf` downloads every part** of a split file of the quantization asked for, the first part being the file; a part
  the listing lacks, or parts that number the model differently, are refused by name. The pack in xpu-serve's cache is
  named without the part's `-00001-of-00003`.

## Alternatives considered

- **Merge the parts on disk first** (`gguf-split --merge`): a third copy of the weights beside the parts and the pack,
  for what reading them where they are gives.

## Consequences

- A split file and its whole make the same pack, so everything measured on one holds for the other. Checked on
  unsloth's Qwen3.5-4B Q4_K_M file split into seven parts of 64 tensors: every file of the pack equal to the whole
  file's, `gguf_real_file.cpp` passing over the parts, and on an Arc Pro B60 the whole file's greedy answer.
- Tests: `tests/unit/io/gguf_reader.cpp` (parts as gguf-split writes them, `GgufWriter::split`, and every refusal),
  `tests/unit/convert/gguf_split.cpp` (the pack of the parts and of the whole file, byte for byte, with an MTP head's
  file split too), `tests/unit/hub/test_hub_select.cpp` and `test_hub_download.cpp` (`-hf` of a split file).
