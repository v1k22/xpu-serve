#!/usr/bin/env python3
"""Writes decisions from MMLU-Pro for systemone_eval.py: questions a decision model gets wrong often enough to show a
change (needs the `datasets` package; the data is not kept in this repository).

    python3 tests/e2e/make_mmlu_pro.py --count 300 --out mmlu_pro_300.json

Each request is one multiple-choice question: the question as the state, its options (up to ten) as a choice's
criteria under the letters A, B, ..., and the correct letter as the accepted answer. The sample is drawn with a fixed
seed over the whole test split, so two runs ask the same questions.
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset

    rows = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    picked = sorted(random.Random(args.seed).sample(range(len(rows)), args.count))
    requests, expected = [], []
    for index in picked:
        row = rows[index]
        options = [o for o in row["options"] if o != "N/A"]
        letters = [chr(ord("A") + i) for i in range(len(options))]
        requests.append({"model": "decision-model", "state": row["question"],
                         "questions": {"answer": {"type": "choice",
                                                  "instructions": "Which option answers the question correctly?",
                                                  "criteria": dict(zip(letters, options))}}})
        expected.append({"answer": letters[row["answer_index"]]})
    with open(args.out, "w") as f:
        json.dump({"description": f"{args.count} MMLU-Pro test questions (seed {args.seed}), one choice each",
                   "requests": requests, "expected": expected}, f, ensure_ascii=False)
    print(f"{len(requests)} questions -> {args.out}")


if __name__ == "__main__":
    main()
