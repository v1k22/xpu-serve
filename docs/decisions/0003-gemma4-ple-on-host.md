# 0003: Gemma 4 per-layer embedding rows are gathered on the host

Date: 2026-11-06 · Status: accepted

Gemma 4 E-series models have a per-layer embedding (PLE) table of vocab x (layers x 256): 262144 x 8960 for E2B,
4.7 GB in bf16, larger than the rest of the decoder's per-token reads and above the 4 GB single-allocation limit some
drivers apply. Only one row per token is ever read.

The model therefore gathers PLE rows on the host from the memory-mapped checkpoint (`Model::prepare_inputs`) and feeds
them to the step graph as the `ple` input. The device never holds the table; the cost is ~35 KB of input per token.
