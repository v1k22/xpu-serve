# Third-party code

Pinned copies. Update one dependency per commit and note the new version here.

| Directory | Project | Version | License |
|---|---|---|---|
| `nlohmann/` | [nlohmann/json](https://github.com/nlohmann/json) single header | v3.11.3 (matches the copy the Jinja engine was written against) | MIT |
| `httplib/` | [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.57.1 | MIT |
| `doctest/` | [doctest/doctest](https://github.com/doctest/doctest) (tests only) | v2.5.3 | MIT |
| `stb/` | [nothings/stb](https://github.com/nothings/stb) `stb_image.h` (image decoding; JPEG, PNG, BMP, GIF enabled in `stb_image.cpp`) | v2.30, identical to upstream master (sha256 `594c2fe3...4200b3`) | public domain or MIT |
| `jinja_cpp/` | [wangzhaode/jinja.cpp](https://github.com/wangzhaode/jinja.cpp) as vendored and fixed in [mlx-serve](https://github.com/ddalcu/mlx-serve) `lib/jinja_cpp` | mlx-serve `1f1332d7` + 4 fixes (see `jinja_cpp/PROVENANCE.md`) | Apache-2.0 |

Fixes we make to vendored code are kept small, and each one is listed in that directory's notes.
