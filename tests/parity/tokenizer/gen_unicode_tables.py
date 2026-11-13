"""Writes src/tokenizer/unicode_tables.cpp: the Unicode data behind the NFC normalizer, the Split regex matcher and the
BERT normalizer and pre-tokenizer.

Run: python -I tests/parity/tokenizer/gen_unicode_tables.py

The data comes from Python's unicodedata, aligned with what Hugging Face tokenizers actually uses, because parity with
it is the goal and its Rust dependencies carry other Unicode versions:
- Regex classes (\\p{L}, \\p{M}, \\p{N}) are read from the Oniguruma build inside `tokenizers` (newer than
  unicodedata), and checked against unicodedata for every code point unicodedata assigns.
- NFC data (canonical decompositions, combining classes, compositions) is unicodedata's, minus the entries that the
  normalizer inside `tokenizers` (an older Unicode) does not know, found by asking it. NFD is the same data.
- The BERT classes (control characters, the CJK blocks, nonspacing marks, punctuation) and the lowercase mapping are
  read from the BertNormalizer and BertPreTokenizer inside `tokenizers`, one code point at a time: their Unicode data
  (the unicode_categories crate, older than unicodedata, and Rust's char::to_lowercase, newer) is no Python version's.
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

# unicode::BertClass bits (unicode.h).
BERT_CONTROL, BERT_CHINESE, BERT_NONSPACING_MARK, BERT_PUNCTUATION = 1, 2, 4, 8


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


def bert_data():
    """cp -> BertClass bits, and cp -> its lowercase (when it changes), per tokenizers' BERT components."""

    def bert(clean=False, chinese=False, strip=False, lower=False):
        return normalizers.BertNormalizer(
            clean_text=clean, handle_chinese_chars=chinese, strip_accents=strip, lowercase=lower
        )

    clean, chinese, strip, lower = bert(clean=True), bert(chinese=True), bert(strip=True), bert(lower=True)
    hf_nfd = normalizers.NFD()
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    classes, lowercase, decomposed = {}, {}, []
    for cp in code_points():
        c = chr(cp)
        bits = 0
        # clean_text drops U+0000, U+FFFD and control characters, then maps whitespace to a space.
        cleaned = clean.normalize_str(c)
        if cleaned == "" and cp != 0xFFFD:
            bits |= BERT_CONTROL
        elif cleaned not in ("", " " if cp in WHITE_SPACE else c):
            sys.exit(f"U+{cp:04X}: BertNormalizer clean_text gives {cleaned!r}")
        spaced = chinese.normalize_str(c)
        if spaced == f" {c} ":
            bits |= BERT_CHINESE
        elif spaced != c:
            sys.exit(f"U+{cp:04X}: BertNormalizer handle_chinese_chars gives {spaced!r}")
        # strip_accents is NFD, then nonspacing marks removed: a character NFD leaves alone tells its own class.
        if hf_nfd.normalize_str(c) == c:
            stripped = strip.normalize_str(c)
            if stripped == "":
                bits |= BERT_NONSPACING_MARK
            elif stripped != c:
                sys.exit(f"U+{cp:04X}: BertNormalizer strip_accents gives {stripped!r}")
        else:
            decomposed.append(cp)
        lowered = lower.normalize_str(c)
        if lowered != c:
            lowercase[cp] = lowered
        pieces = [piece for piece, _ in pre_tokenizer.pre_tokenize_str(f"a{c}b")]
        if pieces == ["a", c, "b"]:
            bits |= BERT_PUNCTUATION
        elif pieces != (["a", "b"] if cp in WHITE_SPACE else [f"a{c}b"]):
            sys.exit(f"U+{cp:04X}: BertPreTokenizer splits 'a', it, 'b' into {pieces}")
        if bits:
            classes[cp] = bits
    for cp in decomposed:
        kept = "".join(x for x in hf_nfd.normalize_str(chr(cp)) if not classes.get(ord(x), 0) & BERT_NONSPACING_MARK)
        if strip.normalize_str(chr(cp)) != kept:
            sys.exit(f"U+{cp:04X}: BertNormalizer strip_accents is not NFD without nonspacing marks")
    return classes, lowercase


def lowercase_runs(lowercase: dict):
    """Single-character mappings as (first, last, delta, stride) runs, and the longer ones as (cp, lower)."""
    runs, expansions = [], []
    for cp in sorted(lowercase):
        lowered = lowercase[cp]
        if len(lowered) > 1:
            if len(lowered) > 3:
                sys.exit(f"U+{cp:04X} lowercases to more than three characters")
            expansions.append((cp, lowered))
            continue
        delta = ord(lowered) - cp
        if runs and runs[-1][2] == delta:
            first, last, _, stride = runs[-1]
            if (first == last and cp - last in (1, 2)) or cp - last == stride:
                runs[-1] = [first, cp, delta, cp - last]
                continue
        runs.append([cp, cp, delta, 1])
    return runs, expansions


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
    bert_classes, lowercase = bert_data()

    cat_ranges = ranges(classes)
    ccc_ranges = ranges(ccc)
    unsafe_ranges = ranges({cp: 1 for cp in unsafe})
    decomp_rows = []
    for cp in sorted(decomposition):
        d = decomposition[cp]
        second = d[1] if len(d) == 2 else 0
        decomp_rows.append(f"{{0x{cp:X},0x{d[0]:X},0x{second:X},{int(cp in primary)}}},")
    bert_ranges = ranges(bert_classes)
    case_runs, case_expansions = lowercase_runs(lowercase)
    expansion_rows = []
    for cp, lowered in case_expansions:
        chars = ",".join(f"0x{ord(x):X}" for x in lowered.ljust(3, chr(0)))
        expansion_rows.append(f"{{0x{cp:X},{{{chars}}}}},")

    text = f"""// Generated by tests/parity/tokenizer/gen_unicode_tables.py; do not edit.
// Regex classes: the Oniguruma inside Hugging Face tokenizers {tokenizers.__version__}, checked against Unicode
// {unicodedata.unidata_version} (Python unicodedata) on every code point that version assigns.
// NFC: Unicode {unicodedata.unidata_version} (Python unicodedata) minus what the tokenizers {tokenizers.__version__} normalizer
// does not know: {unknown_ccc} combining classes and {unknown_decomp} decomposition(s) of newer characters.
// BERT classes and lowercase: the BertNormalizer and BertPreTokenizer of tokenizers
// {tokenizers.__version__}, asked about every code point.

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

const BertClassRange kBertClassRanges[] = {{
{fmt_rows([f"{{0x{a:X},0x{b:X},{v}}}," for a, b, v in bert_ranges])}
}};
const size_t kBertClassRangeCount = sizeof(kBertClassRanges) / sizeof(kBertClassRanges[0]);

const LowercaseRun kLowercaseRuns[] = {{
{fmt_rows([f"{{0x{a:X},0x{b:X},{d},{st}}}," for a, b, d, st in case_runs])}
}};
const size_t kLowercaseRunCount = sizeof(kLowercaseRuns) / sizeof(kLowercaseRuns[0]);

const LowercaseExpansion kLowercaseExpansions[] = {{
{fmt_rows(expansion_rows)}
}};
const size_t kLowercaseExpansionCount = sizeof(kLowercaseExpansions) / sizeof(kLowercaseExpansions[0]);
// clang-format on

}}  // namespace xs::tok::unicode::tables
"""
    OUT.write_text(text, encoding="utf-8")
    print(
        f"wrote {OUT} ({OUT.stat().st_size} bytes): {len(cat_ranges)} category ranges, {len(ccc_ranges)} combining "
        f"ranges, {len(decomp_rows)} decompositions ({len(primary)} compose), {len(unsafe_ranges)} NFC-unsafe ranges, "
        f"{len(bert_ranges)} BERT class ranges, {len(case_runs)} lowercase runs, {len(expansion_rows)} expansion(s)"
    )


if __name__ == "__main__":
    main()
