#!/usr/bin/env python3
"""Chooses a format per expert bank (branch exp/manual-quant) from sweep.py errors and measured sensitivities.

    python allocate.py --sweep sweeps/ex --sens sens.json --budget-gib 11 --out alloc.json
    python allocate.py --sweep sweeps/ex --sens sens.json --frontier      # predicted KL against size

Model: the KL a bank adds is its output error (sweep.py, summed over experts) times the bank's sensitivity, KL per
unit of output error (sens.json from sens_fit.py: measured layer flips). KL is taken as additive over banks (and,
with --per-expert, over experts); the budget is met with a Lagrange multiplier (each bank or expert minimizes
s * err + lambda * bytes on its own), bisected on lambda.
"""

import argparse
import json
import os
import re

import numpy as np

EXPERT = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def bytes_of(shape, bits, group):
    rows, k = int(np.prod(shape[:-1])), shape[-1]
    return rows * (k * bits // 8 + (k // group) * 2 * (1 if bits == 8 else 2))


def load(sweep, configs=None):
    """-> {name: {config: (err array per expert, bytes)}}."""
    table = {}
    for c in sorted(configs or os.listdir(sweep)):
        p = os.path.join(sweep, c, "errors.json")
        if not os.path.exists(p):
            continue
        man = json.load(open(os.path.join(sweep, c, "xpu_pack.json")))["tensors"]
        errs = json.load(open(p))
        for name, e in errs.items():
            m = man[name]
            table.setdefault(name, {})[c] = (np.array(e["err"]), bytes_of(m["shape"], int(c[1]), m["group"]))
    return table


def sensitivities(table, sens):
    """sens: {name: KL per unit of output error} (sens_fit.py). Banks it lacks take the nearest layer's of their kind."""
    out = {}
    for name in table:
        if name in sens:
            out[name] = sens[name]
            continue
        li, kind = EXPERT.search(name).groups()
        cands = [(abs(int(EXPERT.search(n).group(1)) - int(li)), v) for n, v in sens.items()
                 if EXPERT.search(n).group(2) == kind]
        out[name] = min(cands)[1] if cands else 1.0
    return out


def choose(table, s, lam, per_expert=False):
    """-> ({name: config or [config per expert]}, predicted dKL, bytes)."""
    alloc, kl, size = {}, 0.0, 0
    for name, cfgs in table.items():
        names = list(cfgs)
        err = np.stack([cfgs[c][0] for c in names])  # [configs, experts]
        byt = np.array([cfgs[c][1] for c in names], dtype=np.float64)
        if per_expert:
            cost = s[name] * err + lam * (byt / err.shape[1])[:, None]
            pick = cost.argmin(0)
            alloc[name] = [names[i] for i in pick]
            kl += s[name] * err[pick, np.arange(err.shape[1])].sum()
            size += (byt[pick] / err.shape[1]).sum()
        else:
            cost = s[name] * err.sum(1) + lam * byt
            i = int(cost.argmin())
            alloc[name] = names[i]
            kl += s[name] * err[i].sum()
            size += byt[i]
    return alloc, kl, size


def solve(table, s, budget, per_expert=False):
    lo, hi = 1e-20, 1e-3
    best = None
    for _ in range(100):
        lam = (lo * hi) ** 0.5
        alloc, kl, size = choose(table, s, lam, per_expert)
        if size <= budget:
            best, hi = (alloc, kl, size), lam
        else:
            lo = lam
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--sens", required=True)
    ap.add_argument("--budget-gib", type=float)
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--per-expert", action="store_true", help="a format per expert (a qx bank; one group size)")
    ap.add_argument("--configs", default="", help="comma-separated formats to choose from (default: all swept)")
    ap.add_argument("--out")
    args = ap.parse_args()
    table = load(args.sweep, args.configs.split(",") if args.configs else None)
    s = sensitivities(table, json.load(open(args.sens)))
    if args.frontier:
        for gib in np.arange(8.5, 17.5, 0.5):
            a = solve(table, s, gib * 2**30)
            b = solve(table, s, gib * 2**30, per_expert=True)
            if a:
                print(f"{gib:5.1f} GiB experts: predicted dKL per bank {a[1]:.4f}" +
                      (f", per expert {b[1]:.4f}" if b else ""))
        return
    alloc, kl, size = solve(table, s, args.budget_gib * 2**30, args.per_expert)
    counts = {}
    for v in alloc.values():
        for c in (v if isinstance(v, list) else [v]):
            counts[c] = counts.get(c, 0) + 1
    print(f"predicted dKL {kl:.4f}, experts {size / 2**30:.2f} GiB, {counts}")
    if args.out:
        path = lambda c: os.path.join(args.sweep, c)
        json.dump({n: [path(x) for x in c] if isinstance(c, list) else path(c) for n, c in sorted(alloc.items())},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
