"""Writes tests/fixtures/quant/turbo: rows of 128 values and the blocks TurboQuant's reference code stores them as.

Run: python3 -I tests/parity/quant/gen_turbo_cases.py <TheTom/llama-cpp-turboquant checkout>

Compiles turbo_quantize.c with the checkout's ggml/src/ggml-turbo-quant.c (cc; -ffp-contract=off), then writes
  rows.f32        f32, little-endian: the rows of ROWS in that order
  turbo<N>.blocks the checkout's block_turbo<N>_0 of each row: an f16 norm, then the codes (N = 4, 3, 2)
and layout.txt (bits, bytes and values of each block).
"""

import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "tests/fixtures/quant/turbo"
GROUP = 128
BLOCK_BYTES = {4: 66, 3: 50, 2: 34}


def rows() -> list:
    rng = random.Random("turbo kv rows")
    out = []
    for scale in (1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 30.0, 30.0, 1e-4, 300.0):
        for _ in range(3):
            out.append([rng.gauss(0.0, scale) for _ in range(GROUP)])
    for _ in range(4):
        out.append([rng.uniform(-2.0, 2.0) for _ in range(GROUP)])
    for spikes in (1, 2, 5, 16):  # a few large values over small noise: far from a Gaussian before the rotation
        row = [rng.gauss(0.0, 0.05) for _ in range(GROUP)]
        for i in rng.sample(range(GROUP), spikes):
            row[i] = rng.choice((-1.0, 1.0)) * rng.uniform(20.0, 40.0)
        out.append(row)
    out.append([0.0] * GROUP)
    out.append([0.0] * 77 + [3.5] + [0.0] * 50)
    out.append([0.75] * GROUP)
    out.append([(-1.0) ** i * 1.25 for i in range(GROUP)])
    out.append([rng.gauss(4.0, 0.5) for _ in range(GROUP)])  # a large mean
    return out


def main() -> None:
    fork = Path(sys.argv[1]).resolve()
    src = fork / "ggml/src"
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "turbo_quantize"
        subprocess.run(["cc", "-std=c11", "-O2", "-ffp-contract=off", "-I", str(fork / "ggml/include"), "-I", str(src),
                        "-I", str(src / "ggml-cpu"), str(HERE / "turbo_quantize.c"), str(src / "ggml-turbo-quant.c"),
                        "-lm", "-o", str(exe)], check=True, stderr=subprocess.DEVNULL)
        layout = subprocess.run([str(exe), "layout"], check=True, capture_output=True, text=True).stdout
        for line in layout.splitlines():
            bits, size, values = (int(x) for x in line.split())
            if (size, values) != (BLOCK_BYTES[bits], GROUP):
                sys.exit(f"turbo{bits}: the checkout's block is {size} bytes of {values} values")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "layout.txt").write_text(layout)
        data = rows()
        (OUT / "rows.f32").write_bytes(b"".join(struct.pack(f"<{GROUP}f", *r) for r in data))
        for bits in BLOCK_BYTES:
            subprocess.run([str(exe), str(bits), str(OUT / "rows.f32"), str(OUT / f"turbo{bits}.blocks")], check=True)
            print(f"turbo{bits}: {len(data)} rows, {(OUT / f'turbo{bits}.blocks').stat().st_size} bytes")


if __name__ == "__main__":
    main()
