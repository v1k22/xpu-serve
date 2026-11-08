#!/usr/bin/env python3
"""Action error of policy-eval outputs against a reference (branch exp/manual-quant).

    python policy_compare.py <reference actions> <actions> [<actions> ...] --cases 64

Per file: the mean over cases of the relative L2 error ||a - ref|| / ||ref|| of the action chunk, its 90th
percentile, and the cosine similarity, plus the worst absolute difference.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("actions", nargs="+")
    ap.add_argument("--cases", type=int, required=True)
    args = ap.parse_args()
    ref = np.fromfile(args.reference, dtype=np.float32).reshape(args.cases, -1).astype(np.float64)
    for path in args.actions:
        a = np.fromfile(path, dtype=np.float32).reshape(args.cases, -1).astype(np.float64)
        rel = np.linalg.norm(a - ref, axis=1) / np.linalg.norm(ref, axis=1)
        cos = (a * ref).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(ref, axis=1))
        print(f"{os.path.basename(path)}: relative L2 error mean {rel.mean():.4f}, p90 {np.percentile(rel, 90):.4f}, "
              f"cosine mean {cos.mean():.5f}, max |diff| {np.abs(a - ref).max():.4f}")


if __name__ == "__main__":
    main()
