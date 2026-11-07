# Jinja engine provenance

- **Upstream:** [wangzhaode/jinja.cpp](https://github.com/wangzhaode/jinja.cpp) (Apache-2.0, `LICENSE` here). The
  sources follow the lexer/parser/runtime/value layout of llama.cpp's `common/jinja` engine, which that project tracks.
- **Copied from:** [mlx-serve](https://github.com/ddalcu/mlx-serve) `lib/jinja_cpp/` at commit
  `1f1332d788105a9a4b0d06f662ec63acd5faec07`, which vendors upstream with its own fixes (its git history:
  `|min`/`|max` over arrays, `set` scoping inside `for`, and others) and renders Gemma 4's template in production.
- **Taken:** `caps.*`, `jinja_string.*`, `lexer.*`, `log.h`, `parser.*`, `runtime.*`, `utils.h`, `value.*`, `LICENSE`.
- **Left out:** `libjinja.a` and `obj/` (prebuilt Mach-O objects), `nlohmann/` (identical to our
  `third_party/nlohmann/json.hpp` 3.11.3), and `jinja_wrapper.*` (mlx-serve's C API; we call the C++ API from
  `src/chat/chat_template.cpp`).

## Changes made here (each marked `xpu-serve:` in the source)

| File | Change | Why |
|---|---|---|
| `value.cpp` | `dictsort` honors `case_sensitive` (default false: string keys compare lowercased) and sorts stably | Jinja2's default; Gemma 4 sorts tool parameters with `dictsort`, so `URL`/`Limit` ordered differently from Hugging Face |
| `value.cpp` | `map('<filter>', ...)` applies the named filter to each item (was "not implemented") | Gemma 4's template renders `items.type` lists with `map('upper')` |
| `value.h`, `value.cpp` | Floats print as Python `repr` (`python_float_repr`: shortest round-trip, `1e-05`, `1e+20`) instead of `std::to_string` with trailing zeros trimmed | Byte parity with Hugging Face for numeric tool-call arguments (`2.5e-07` printed as `0.0`) |
| `value.cpp` | `tojson` (and `\|string` on a float, which goes through it) writes floats with `python_float_repr` instead of `ostream <<` (6 significant digits, `3` for `3.0`) | Python's `json.dumps` and `str(float)` write `repr`; Qwen 3.5 renders tool-call arguments with `\|string` and tool schemas with `tojson` |
| `runtime.cpp` | Runtime errors put the engine's message first, on one line, then the template line and column | The message used to start with a newline, so callers that show the first line showed nothing |
| `CMakeLists.txt` | New: static library `xs_jinja` | Our build |
