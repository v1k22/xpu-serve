"""Writes src/tokenizer/unicode_tables.cpp: the Unicode data behind the NFC normalizer and the Split regex matcher.

Run: python -I tests/parity/tokenizer/gen_unicode_tables.py

The data comes from Python's unicodedata, aligned with what Hugging Face tokenizers actually uses, because parity with
it is the goal and its Rust dependencies carry other Unicode versions:
- Regex classes (\\p{L}, \\p{M}, \\p{N}) are read from the Oniguruma build inside `tokenizers` (newer than
  unicodedata), and checked against unicodedata for every code point unicodedata assigns.
- NFC data (canonical decompositions, combining classes, compositions) is unicodedata's, minus the entries that the
  normalizer inside `tokenizers` (an older Unicode) does not know, found by asking it.
"""

import sys
import unicodedata
from pathlib import Path

import tokenizers
from tokenizers import Regex, normalizers, pre_tokenizers

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "src/tokenizer/unicode_tables.cpp"
MAX_CP = 0x110000

# Unicode White_Space: the regex \s of Oniguruma and Rust's char::is_whitespace (utf8::is_whitespace in C++).
WHITE_SPACE = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
WHITE_SPACE |= set(range(0x2000, 0x200B))

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28


def code_points():
    return (cp for cp in range(MAX_CP) if not 0xD800 <= cp < 0xE000)


def regex_members(pattern: str) -> set:
    """Code points the tokenizers regex engine matches with `pattern` (a one-character class)."""
    split = pre_tokenizers.Split(Regex(pattern), behavior="removed")
    cps = list(code_points())
    members = set()
    for start in range(0, len(cps), 4096):
        chunk = "".join(chr(cp) for cp in cps[start : start + 4096])
        left = {ord(c) for piece, _ in split.pre_tokenize_str(chunk) for c in piece}
        members.update(cp for cp in cps[start : start + 4096] if cp not in left)
    return members


def category_classes() -> dict:
    """cp -> 1 (L), 2 (M) or 3 (N), per the tokenizers regex engine."""
    classes = {}
    for code, pattern in ((1, r"\p{L}"), (2, r"\p{M}"), (3, r"\p{N}")):
        for cp in regex_members(pattern):
            if cp in classes:
                sys.exit(f"U+{cp:04X} is in two of L, M, N")
            classes[cp] = code
    expected = {"L": 1, "M": 2, "N": 3}
    for cp in code_points():
        category = unicodedata.category(chr(cp))
        if category == "Cn":
            continue
        if classes.get(cp, 0) != expected.get(category[0], 0):
            sys.exit(f"U+{cp:04X}: regex class {classes.get(cp, 0)} disagrees with unicodedata category {category}")
    space = regex_members(r"\s")
    if space != WHITE_SPACE:
        sys.exit(f"regex \\s differs from White_Space: {sorted(space ^ WHITE_SPACE)}")
    return classes


def ranges(values: dict):
    """Collapses {cp: value} (value != 0) into (first, last, value) runs."""
    out = []
    for cp in sorted(values):
        v = values[cp]
        if out and out[-1][1] == cp - 1 and out[-1][2] == v:
            out[-1][1] = cp
        else:
            out.append([cp, cp, v])
    return out


def nfc_data():
    hf_nfd = normalizers.NFD()
    hf_nfc = normalizers.NFC()
    nfd = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731
    nfc = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731

    ccc = {}
    for cp in code_points():
        cls = unicodedata.combining(chr(cp))
        if cls == 0:
            continue
        # Put the mark on the wrong side of a mark of another class; a normalizer that knows its class reorders.
        probe = "́" + chr(cp) if cls == 1 else chr(cp) + "̴"
        if hf_nfd.normalize_str(probe) == nfd(probe):
            ccc[cp] = cls
        elif hf_nfd.normalize_str(probe) != probe:
            sys.exit(f"U+{cp:04X}: tokenizers NFD reorders it, but not by combining class {cls}")
    unknown_ccc = sum(1 for cp in code_points() if unicodedata.combining(chr(cp))) - len(ccc)

    decomposition = {}  # cp -> one-level canonical decomposition (tuple of 1 or 2 code points)
    unknown_decomp = 0
    for cp in code_points():
        if S_BASE <= cp < S_BASE + L_COUNT * V_COUNT * T_COUNT:
            continue  # Hangul syllables decompose algorithmically
        fields = unicodedata.decomposition(chr(cp)).split()
        if not fields or fields[0].startswith("<"):
            continue
        if hf_nfd.normalize_str(chr(cp)) == nfd(chr(cp)):
            decomposition[cp] = tuple(int(f, 16) for f in fields)
        elif hf_nfd.normalize_str(chr(cp)) == chr(cp):
            unknown_decomp += 1
        else:
            sys.exit(f"U+{cp:04X}: tokenizers NFD disagrees with unicodedata")
    if any(len(d) > 2 for d in decomposition.values()):
        sys.exit("a canonical decomposition has more than two code points")

    primary = set()
    for cp, d in decomposition.items():
        if len(d) == 2 and nfc("".join(map(chr, d))) == chr(cp):
            if hf_nfc.normalize_str("".join(map(chr, d))) != chr(cp):
                sys.exit(f"U+{cp:04X}: tokenizers NFC does not compose its decomposition")
            primary.add(cp)
        elif hf_nfc.normalize_str(chr(cp)) != nfc(chr(cp)):
            sys.exit(f"U+{cp:04X}: tokenizers NFC disagrees with unicodedata")

    # NFC quick check is not Yes, or the combining class is not 0: text around such a character may change.
    unsafe = set(ccc)
    unsafe |= {decomposition[cp][1] for cp in primary}
    unsafe |= set(range(V_BASE, V_BASE + V_COUNT)) | set(range(T_BASE + 1, T_BASE + T_COUNT))
    unsafe |= {cp for cp in decomposition if cp not in primary}
    changed = True
    while changed:
        changed = False
        for cp, d in decomposition.items():
            if cp not in unsafe and d[0] in unsafe:
                unsafe.add(cp)
                changed = True
    return ccc, decomposition, primary, unsafe, unknown_ccc, unknown_decomp


def fmt_rows(items) -> str:
    """Joins items into lines of at most 120 columns."""
    lines = ["   "]
    for item in items:
        if len(lines[-1]) + 1 + len(item) > 120:
            lines.append("   ")
        lines[-1] += " " + item
    return "\n".join(lines)


def main() -> None:
    classes = category_classes()
    ccc, decomposition, primary, unsafe, unknown_ccc, unknown_decomp = nfc_data()

    cat_ranges = ranges(classes)
    ccc_ranges = ranges(ccc)
    unsafe_ranges = ranges({cp: 1 for cp in unsafe})
    decomp_rows = []
    for cp in sorted(decomposition):
        d = decomposition[cp]
        second = d[1] if len(d) == 2 else 0
        decomp_rows.append(f"{{0x{cp:X},0x{d[0]:X},0x{second:X},{int(cp in primary)}}},")

    text = f"""// Generated by tests/parity/tokenizer/gen_unicode_tables.py; do not edit.
// Regex classes: the Oniguruma inside Hugging Face tokenizers {tokenizers.__version__}, checked against Unicode
// {unicodedata.unidata_version} (Python unicodedata) on every code point that version assigns.
// NFC: Unicode {unicodedata.unidata_version} (Python unicodedata) minus what the tokenizers {tokenizers.__version__} normalizer
// does not know: {unknown_ccc} combining classes and {unknown_decomp} decomposition(s) of newer characters.

#include "tokenizer/unicode_tables.h"

namespace xs::tok::unicode::tables {{

// clang-format off
const CategoryRange kCategoryRanges[] = {{
{fmt_rows([f"{{0x{a:X},0x{b:X},{v}}}," for a, b, v in cat_ranges])}
}};
const size_t kCategoryRangeCount = sizeof(kCategoryRanges) / sizeof(kCategoryRanges[0]);

const CombiningRange kCombiningRanges[] = {{
{fmt_rows([f"{{0x{a:X},0x{b:X},{v}}}," for a, b, v in ccc_ranges])}
}};
const size_t kCombiningRangeCount = sizeof(kCombiningRanges) / sizeof(kCombiningRanges[0]);

const Decomposition kDecompositions[] = {{
{fmt_rows(decomp_rows)}
}};
const size_t kDecompositionCount = sizeof(kDecompositions) / sizeof(kDecompositions[0]);

const CodePointRange kNfcUnsafeRanges[] = {{
{fmt_rows([f"{{0x{a:X},0x{b:X}}}," for a, b, _ in unsafe_ranges])}
}};
const size_t kNfcUnsafeRangeCount = sizeof(kNfcUnsafeRanges) / sizeof(kNfcUnsafeRanges[0]);
// clang-format on

}}  // namespace xs::tok::unicode::tables
"""
    OUT.write_text(text, encoding="utf-8")
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): {len(cat_ranges)} category ranges, {len(ccc_ranges)} combining "
        f"ranges, {len(decomp_rows)} decompositions ({len(primary)} compose), {len(unsafe_ranges)} NFC-unsafe ranges"
    )


if __name__ == "__main__":
    main()
