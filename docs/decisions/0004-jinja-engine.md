# 0004: Chat templates are rendered by the vendored jinja.cpp

Date: 2026-11-06 · Status: accepted

Models ship their own Jinja chat templates, and output must match Hugging Face's `apply_chat_template` byte for byte.
We vendor wangzhaode/jinja.cpp (Apache-2.0) in the copy mlx-serve already uses to serve Gemma 4, plus three small
fixes needed for parity (listed in `third_party/jinja_cpp/PROVENANCE.md`). minja was the other candidate; it has not
been updated since 2025-09 and nothing showed it renders Gemma 4 correctly. The parity suite
(`tests/fixtures/chat/gemma4_cases.json`) is the gate for any future engine change.
