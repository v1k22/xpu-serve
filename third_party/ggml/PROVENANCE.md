# ggml quantization tables provenance

- **Upstream:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-common.h` (MIT,
  `LICENSE` here), at commit `2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c`.
- **Taken:** the tables GGUF's block formats index when they are decoded: `kmask_iq2xs`, `ksigns_iq2xs`,
  `kvalues_iq4nl`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`; nothing else.
- **How:** `quant_tables.h` is written by `extract_tables.py` (`python3 -I extract_tables.py <llama.cpp checkout> >
  quant_tables.h`): each `GGML_TABLE_BEGIN(type, name, size)` table becomes `inline constexpr std::array<type, size>
  name` in namespace `ggml`, its value lines copied unchanged. `tests/unit/quant/test_block_decode.cpp` checks every
  value against the tables ggml's own code was compiled with (`tests/fixtures/quant/blocks/tables.bin`).
- **Used through:** `src/quant/block_tables.h`, the one place the engine includes them from (host and device code).
