"""Writes quant_tables.h from ggml/src/ggml-common.h of a llama.cpp checkout.

Usage: python3 -I extract_tables.py <llama.cpp checkout> > quant_tables.h

Each GGML_TABLE_BEGIN(type, name, size) table named in TABLES becomes `inline constexpr std::array<type, size> name`
in namespace ggml, its value lines copied as they are.
"""

import re
import sys
from pathlib import Path

TABLES = [
    "kmask_iq2xs",
    "ksigns_iq2xs",
    "kvalues_iq4nl",
    "iq2xxs_grid",
    "iq2xs_grid",
    "iq2s_grid",
    "iq3xxs_grid",
    "iq3s_grid",
    "iq1s_grid",
]


def main() -> None:
    root = Path(sys.argv[1])
    text = (root / "ggml/src/ggml-common.h").read_text()
    defines = dict(re.findall(r"^#define (\w+) (\d+)\s*$", text, re.M))
    tables = {}
    for m in re.finditer(r"GGML_TABLE_BEGIN\((\w+), (\w+), (\w+)\)\n(.*?)GGML_TABLE_END\(\)", text, re.S):
        ctype, name, size, body = m.groups()
        tables.setdefault(name, (ctype, int(defines.get(size, size)), body))
    out = [
        "// The codebooks of ggml's quantization formats, from ggml/src/ggml-common.h of llama.cpp (MIT License,",
        "// Copyright (c) 2023-2026 The ggml authors: LICENSE). Written by extract_tables.py (PROVENANCE.md); the values",
        "// are ggml's, unchanged. Plain constexpr data, so device code can include it too.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace ggml {",
    ]
    for name in TABLES:
        ctype, size, body = tables[name]
        values = [v for v in re.split(r"[,\s]+", body) if v]
        if len(values) != size:
            sys.exit(f"{name}: {len(values)} values, expected {size}")
        out.append("")
        out.append(f"inline constexpr std::array<{ctype}, {size}> {name} = {{")
        out.extend(line.rstrip() for line in body.strip("\n").split("\n"))
        out.append("};")
    out.append("")
    out.append("}  // namespace ggml")
    print("\n".join(out))


if __name__ == "__main__":
    main()
