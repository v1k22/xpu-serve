#!/usr/bin/env python3
"""Per-bank KL sensitivities from layer flips (branch exp/manual-quant), for allocate.py --sens.

    python sens_fit.py --sweep sweeps/ex --both sens_both.jsonl [--split sens_gu.jsonl] --out sens.json

sens.sh lines: {"layer", "bank": "both" | "gate_up_proj", "from", "to", "kl", "base"}: a layer's experts (both banks,
or gate_up alone) moved from one format to another, everything else held. With dE a bank's output error change
(sweep.py errors, summed over experts):
  split layers:  s_gate_up = dKL_gate_up / dE_gate_up,  s_down = (dKL_both - dKL_gate_up) / dE_down
  other layers:  the ratio s_down / s_gate_up interpolated from the split layers, dKL_both shared accordingly.
Writes {"<tensor name>": KL per unit of output error}.
"""

import argparse
import json

import numpy as np

from allocate import load

NAME = "model.language_model.layers.{}.mlp.experts.{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--both", required=True)
    ap.add_argument("--split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--errors", default="err", choices=["err", "err_rtn"])
    args = ap.parse_args()
    table = load(args.sweep, key=args.errors)
    de = lambda li, bank, a, b: table[NAME.format(li, bank)][b][0].sum() - table[NAME.format(li, bank)][a][0].sum()
    both = {r["layer"]: r for r in map(json.loads, open(args.both))}
    split = {r["layer"]: r for r in map(json.loads, open(args.split))} if args.split else {}
    ratio = {}
    for li, r in split.items():
        b = both[li]
        s_gu = (r["kl"] - r["base"]) / de(li, "gate_up_proj", r["from"], r["to"])
        s_dn = (b["kl"] - b["base"] - (r["kl"] - r["base"])) / de(li, "down_proj", b["from"], b["to"])
        ratio[li] = max(s_dn, 1e-12) / max(s_gu, 1e-12)
        print(f"layer {li}: s_gate_up {s_gu:.3e}, s_down {s_dn:.3e}, ratio {ratio[li]:.2f}")
    known = sorted(ratio)
    out = {}
    for li, b in sorted(both.items()):
        kappa = np.interp(li, known, [ratio[k] for k in known]) if known else 1.0
        d_gu, d_dn = de(li, "gate_up_proj", b["from"], b["to"]), de(li, "down_proj", b["from"], b["to"])
        s_gu = max(b["kl"] - b["base"], 1e-6) / (d_gu + kappa * d_dn)
        out[NAME.format(li, "gate_up_proj")] = s_gu
        out[NAME.format(li, "down_proj")] = kappa * s_gu
        print(f"layer {li}: dKL {b['kl'] - b['base']:+.4f}, s_gate_up {s_gu:.3e}, s_down {kappa * s_gu:.3e}")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
