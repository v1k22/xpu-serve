# Third-party code

Pinned copies. Update one dependency per commit and note the new version here.

| Directory | Project | Version | License |
|---|---|---|---|
| `nlohmann/` | [nlohmann/json](https://github.com/nlohmann/json) single header | v3.11.3 (matches the copy the Jinja engine was written against) | MIT |
| `httplib/` | [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.57.1 | MIT |
| `doctest/` | [doctest/doctest](https://github.com/doctest/doctest) (tests only) | v2.5.3 | MIT |
| `stb/` | [nothings/stb](https://github.com/nothings/stb) `stb_image.h` (image decoding; JPEG, PNG, BMP, GIF enabled in `stb_image.cpp`) | v2.30, identical to upstream master (sha256 `594c2fe3...4200b3`) | public domain or MIT |
| `dr_libs/` | [mackron/dr_libs](https://github.com/mackron/dr_libs) `dr_mp3.h` and `dr_flac.h` (MP3 and FLAC decoding, compiled in `dr_libs.cpp`: from memory, MP3 straight to float, FLAC without Ogg) | dr_mp3 v0.7.4 (tag `mp3-0.7.4`, sha256 `f5c8020e...ee915`) and dr_flac v0.13.4 (tag `flac-0.13.4`, sha256 `b8078eeb...c9d24`), identical to upstream | public domain or MIT-0 |
| `ggml/` | the codebook tables of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `ggml/src/ggml-common.h` that GGUF's block formats index (data only, rewritten as `constexpr` arrays: `ggml/PROVENANCE.md`) | `2a4daf0e` | MIT |
| `jinja_cpp/` | [wangzhaode/jinja.cpp](https://github.com/wangzhaode/jinja.cpp) as vendored and fixed in [mlx-serve](https://github.com/ddalcu/mlx-serve) `lib/jinja_cpp` | mlx-serve `1f1332d7` + 4 fixes (see `jinja_cpp/PROVENANCE.md`) | Apache-2.0 |

Fixes we make to vendored code are kept small, and each one is listed in that directory's notes.
