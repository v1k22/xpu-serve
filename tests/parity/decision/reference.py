#!/usr/bin/env python3
"""Reference answers of a decision model, from the checkpoint's own Python code in float32 on the CPU (test-only).

    python -I tests/parity/decision/reference.py --model-dir <checkpoint> --requests tests/parity/decision/requests.json
        --out <fixtures.json>

For each request of the file (System One request bodies): the token ids of every sequence the model reads it as, and
each question's probability per option, keyed by the option's name in the request (a noul's by "true" and "false").
decision_parity.cpp checks the engine against them: the same tokens, the same probabilities.

Clef (joint_head_config.json): joint_schema_model.py's encode_record and ClefModel, a softmax per question.
pplx-decider (decision_config.json): the package under source/src: Decider's plan and reference path, the raw
read-out logits over the saved temperature through a softmax.
"""

import argparse
import json
import os
import sys

import torch


def clef(model_dir, requests):
    sys.path.insert(0, model_dir)
    import joint_schema_model as code

    model, processor = code.load_release_model(model_dir, device="cpu", dtype=torch.float32)
    tokenizer = processor.tokenizer
    cases = []
    for request in requests:
        encoded = code.encode_record(tokenizer, request, max_length=1 << 30, processor=processor)
        batch = code.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
        with torch.inference_mode():
            logits = model(batch)[0]
        probabilities = {}
        for question, values in zip(encoded.questions, logits):
            p = values.double().softmax(-1).tolist()
            probabilities[question.question_id] = dict(zip(question.option_ids, p))
        cases.append({"request": request, "input_ids": [list(encoded.input_ids)], "probabilities": probabilities})
    return cases


def pplx(model_dir, requests):
    sys.path.insert(0, os.path.join(model_dir, "source", "src"))
    from pplx_decider import Decider, options

    decider = Decider.from_pretrained(model_dir, device="cpu")
    cases = []
    for request in requests:
        chunks = decider.encoder.plan(request, decider.max_length)
        logits = decider.logits([request])[0]
        probabilities = {}
        for name, question in request["questions"].items():
            p = (torch.tensor(logits[name], dtype=torch.float64) / decider.temperature).softmax(-1).tolist()
            probabilities[name] = dict(zip(options(question)[0], p))
        cases.append({"request": request, "input_ids": [chunk.input_ids.tolist() for chunk in chunks],
                      "probabilities": probabilities})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    with open(args.requests) as f:
        requests = json.load(f)["requests"]
    model_dir = os.path.abspath(args.model_dir)
    if os.path.exists(os.path.join(model_dir, "joint_head_config.json")):
        kind, cases = "clef", clef(model_dir, requests)
    elif os.path.exists(os.path.join(model_dir, "decision_config.json")):
        kind, cases = "pplx_decider", pplx(model_dir, requests)
    else:
        ap.error(f"{model_dir} holds no decision model this script knows")
    with open(args.out, "w") as f:
        json.dump({"kind": kind, "cases": cases}, f, ensure_ascii=False)
    tokens = sum(len(ids) for c in cases for ids in c["input_ids"])
    print(f"{kind}: {len(cases)} requests, {tokens} tokens -> {args.out}")


if __name__ == "__main__":
    main()
