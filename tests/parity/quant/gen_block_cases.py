"""Writes tests/fixtures/quant/blocks: blocks of every GGUF block format and the values ggml's own code gives them.

Run: python3 -I tests/parity/quant/gen_block_cases.py <llama.cpp checkout>

Compiles ggml_dequantize.c with the checkout's ggml/src/ggml-quants.c (cc; -ffp-contract=off, so each product is
rounded on its own, as the C source says), then writes per format:
  <FORMAT>.blocks  32 blocks of random bits, then the edge blocks of EDGES in that order
  <FORMAT>.values  f32, little-endian: ggml's dequantize_row_<format> of those blocks
and layout.txt (ggml's bytes and weights per block) and tables.bin (its codebook tables). The same values come out of a
second build with fused multiply-adds allowed and FMA instructions on (-ffp-contract=fast -mfma), or the script stops.
"""

import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "tests/fixtures/quant/blocks"
RANDOM_BLOCKS = 32

# name: (bytes, weights, offsets of the f16 fields, byte ranges of sub-block scales). The other bytes are codes (grid
# indexes, signs, high bits, and for IQ1_S, IQ2_XXS and IQ3_XXS the scales packed among them). IQ1_M's f16 is the top
# nibbles of bytes 49, 51, 53 and 55.
FORMATS = {
    "Q4_0": (18, 32, [0], []),
    "Q4_1": (20, 32, [0, 2], []),
    "Q5_0": (22, 32, [0], []),
    "Q5_1": (24, 32, [0, 2], []),
    "Q8_0": (34, 32, [0], []),
    "Q2_K": (84, 256, [80, 82], [(0, 16)]),
    "Q3_K": (110, 256, [108], [(96, 108)]),
    "Q4_K": (144, 256, [0, 2], [(4, 16)]),
    "Q5_K": (176, 256, [0, 2], [(4, 16)]),
    "Q6_K": (210, 256, [208], [(192, 208)]),
    "IQ1_S": (50, 256, [0], []),
    "IQ1_M": (56, 256, [], [(48, 56)]),
    "IQ2_XXS": (66, 256, [0], []),
    "IQ2_XS": (74, 256, [0], [(66, 74)]),
    "IQ2_S": (82, 256, [0], [(74, 82)]),
    "IQ3_XXS": (98, 256, [0], []),
    "IQ3_S": (110, 256, [0], [(106, 110)]),
    "IQ4_NL": (18, 32, [0], []),
    "IQ4_XS": (136, 256, [0], [(2, 8)]),
}

# The edge blocks after the random ones: (name, codes, f16 fields); codes "zero" / "max" set every code byte to 0x00 /
# 0xFF ("all" every byte but the f16 fields), None leaves them random; an f16 of None is random and finite.
EDGES = [
    ("all bytes zero", "all_zero", None),
    ("all bytes 0xFF", "all_max", None),
    ("codes zero", "zero", None),
    ("codes all ones", "max", None),
    ("scale zero", None, 0x0000),
    ("scale negative", None, "negative"),
    ("scale the largest f16, every other bit set", "all_max", 0x7BFF),
    ("scale the smallest f16 subnormal, negative", None, 0x8001),
]


def finite_f16(rng: random.Random) -> int:
    h = rng.getrandbits(16)
    return h ^ 0x0400 if (h >> 10) & 0x1F == 0x1F else h


def set_f16(block: bytearray, name: str, h: int) -> None:
    if name == "IQ1_M":
        for k in range(4):
            block[49 + 2 * k] = (block[49 + 2 * k] & 0x0F) | (((h >> (4 * k)) & 0xF) << 4)
        return
    for at in FORMATS[name][2]:
        block[at : at + 2] = struct.pack("<H", h)


def make_block(name: str, rng: random.Random, codes=None, f16=None) -> bytes:
    size, _, f16_at, scale_ranges = FORMATS[name]
    block = bytearray(rng.getrandbits(8) for _ in range(size))
    if codes in ("all_zero", "all_max"):
        block = bytearray([0 if codes == "all_zero" else 0xFF] * size)
    elif codes is not None:
        fixed = set(f16_at) | {a + 1 for a in f16_at}
        for lo, hi in scale_ranges:
            fixed |= set(range(lo, hi))
        for i in range(size):
            if i not in fixed:
                block[i] = 0 if codes == "zero" else 0xFF
    if f16 is None:
        set_f16(block, name, finite_f16(rng))
    elif f16 == "negative":
        set_f16(block, name, finite_f16(rng) | 0x8000)
    else:
        set_f16(block, name, f16)
    return bytes(block)


def build(llama: Path, work: Path, flags: list) -> Path:
    exe = work / ("ggml_dequantize" + "".join(flags).replace("-", "_").replace("=", "_"))
    src = llama / "ggml/src"
    cmd = ["cc", "-std=c11", "-O2", *flags, "-I", str(llama / "ggml/include"), "-I", str(src),
           "-I", str(src / "ggml-cpu"), str(HERE / "ggml_dequantize.c"), str(src / "ggml-quants.c"), "-lm",
           "-o", str(exe)]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return exe


def main() -> None:
    llama = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        exact = build(llama, work, ["-ffp-contract=off"])
        fused = build(llama, work, ["-ffp-contract=fast", "-mfma"])
        if OUT.exists():
            for f in OUT.iterdir():
                if f.suffix in (".blocks", ".values", ".txt", ".bin"):
                    f.unlink()
        OUT.mkdir(parents=True, exist_ok=True)

        layout = subprocess.run([str(exact), "layout"], check=True, capture_output=True, text=True).stdout
        for line in layout.splitlines():
            name, size, weights = line.split()
            if FORMATS[name][:2] != (int(size), int(weights)):
                sys.exit(f"{name}: ggml's block is {size} bytes of {weights} weights, not {FORMATS[name][:2]}")
        (OUT / "layout.txt").write_text(layout)
        subprocess.run([str(exact), "tables", str(OUT / "tables.bin")], check=True)

        for name in FORMATS:
            rng = random.Random(f"gguf blocks {name}")
            blocks = [make_block(name, rng) for _ in range(RANDOM_BLOCKS)]
            blocks += [make_block(name, rng, codes, f16) for _, codes, f16 in EDGES]
            (OUT / f"{name}.blocks").write_bytes(b"".join(blocks))
            subprocess.run([str(exact), name, str(OUT / f"{name}.blocks"), str(OUT / f"{name}.values")], check=True)
            subprocess.run([str(fused), name, str(OUT / f"{name}.blocks"), str(work / "fused.values")], check=True)
            if (OUT / f"{name}.values").read_bytes() != (work / "fused.values").read_bytes():
                sys.exit(f"{name}: ggml's values change with fused multiply-adds")
            print(f"{name}: {len(blocks)} blocks")


if __name__ == "__main__":
    main()
