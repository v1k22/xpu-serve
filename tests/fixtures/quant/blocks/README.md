# GGUF block fixtures

Written by `tests/parity/quant/gen_block_cases.py` with ggml's own C code as the oracle: `ggml_dequantize.c` built
with `ggml/src/ggml-quants.c` of llama.cpp at commit `2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c` (gcc,
`-O2 -ffp-contract=off`). A second build with fused multiply-adds (`-ffp-contract=fast -mfma`) gave the same bytes for
every format.

| File | What |
|---|---|
| `<FORMAT>.blocks` | 40 blocks of the format, as a GGUF file stores them: 32 of random bits (f16 fields kept finite), then the edge blocks below |
| `<FORMAT>.values` | f32, little-endian: what ggml's `dequantize_row_<format>` gives those blocks, a block's weights after another's |
| `layout.txt` | per format: its name, then ggml's bytes and weights per block (`sizeof(block_<format>)`, `QK`) |
| `tables.bin` | the codebook tables ggml-quants.c is built with, one after another, little-endian: `kmask_iq2xs`, `ksigns_iq2xs`, `kvalues_iq4nl`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid` |

The edge blocks, blocks 32 to 39 in this order (codes: every byte but the f16 fields and the sub-block scales; IQ1_M's
f16 is the top nibbles of its scale words):

1. every byte zero but the f16 fields (random, finite)
2. every byte 0xFF but the f16 fields
3. codes all zero
4. codes all ones
5. f16 fields zero
6. f16 fields negative
7. f16 fields 65504, the largest f16, every other bit set
8. f16 fields -2^-24, the smallest subnormal
