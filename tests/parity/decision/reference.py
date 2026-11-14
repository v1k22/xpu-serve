#!/usr/bin/env python3
"""Reference answers of a decision model, from the checkpoint's own Python code in float32 on the CPU (test-only).

    python -I tests/parity/decision/reference.py --model-dir <checkpoint> --requests tests/parity/decision/requests.json
        --out <fixtures.json> [--set-size N]

For each request of the file (System One request bodies): the token ids of every sequence the model reads it as, and
each question's probability per option, keyed by the option's name in the request (a noul's by "true" and "false").
decision_parity.cpp checks the engine against them: the same tokens, the same probabilities. A request the
reference code refuses is left out (the engine refuses it too).

Clef (joint_head_config.json): joint_schema_model.py's encode_record and ClefModel, a softmax per question.
pplx-decider (decision_config.json): the package under source/src: Decider's plan and reference path, the raw
read-out logits over the saved temperature through a softmax.
OpenJev (helper/shim.py): the helper's own code with the settings of its serving guide. Only its client is replaced:
the one call it makes to a server (the log-probabilities of the option letters at the first output position) is
answered by the model in transformers, and the answers are left unrounded.
d1 (config.json's architectures: D1Model): the checkpoint's own model class and runner (AutoModel with its code): the
sequences its runner lays out and the probabilities it reads, through calibration.json's temperature. --set-size is
the runner's set_size, the questions read together after one state: the checkpoint's own (d1_set_size) by default, 1
for each question alone; the fixtures record it ("set_size"), and decision_parity.cpp reads the model at it.
Laya (rl_agent_config.json): rl_agent_api.py's RLAgent.system_one, its probabilities left unrounded; a question
without instructions is passed with null ones (the request schema does not tell them apart; the reference writes
null as "null").
Julia-1 (julia_config.json): julia's load_model and predict_typed with inference-policy.json's limits and encoding.
The encoder's inference-only forward (julia/router/encoder.py) is written against transformers 5.0's ModernBERT
internals: with a later transformers it is taken off, and ModernBERT's own forward computes the same.

A request the reference refuses is a case with "refused" (its message) instead of tokens and probabilities; the
engine must refuse it too.
"""

import argparse
import importlib.util
import json
import os
import sys
import types

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


def letters(model_dir, requests):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    # The helper reads its settings from the environment when imported: serve/SERVE.md's.
    os.environ.update(READOUT_T="0.85", READOUT_NOUL_T="1.829074", READOUT_NOUL_BIAS="0", READOUT_TARGETED="1",
                      READOUT_INSTR_STYLE="pyrepr", TOKENIZER=model_dir)
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=lambda **kwargs: None)  # its import; no server is called
    spec = importlib.util.spec_from_file_location("shim", os.path.join(model_dir, "helper", "shim.py"))
    shim = importlib.util.module_from_spec(spec)
    sys.modules["shim"] = shim
    spec.loader.exec_module(shim)
    shim.round = lambda value, digits=None: value

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    prompts = []

    def create(messages=None, extra_body=None, **kwargs):
        wanted = extra_body["logprob_token_ids"]
        assert extra_body["chat_template_kwargs"] == {"enable_thinking": False} and isinstance(messages[0]["content"], str)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        prompts.append(list(ids))
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1].double()
        scores = logits.log_softmax(-1)
        top = [types.SimpleNamespace(token=f"token_id:{i}", logprob=float(scores[i])) for i in wanted]
        content = [types.SimpleNamespace(top_logprobs=top)]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(logprobs=types.SimpleNamespace(content=content))],
                                     usage=types.SimpleNamespace(prompt_tokens=len(ids)))

    shim.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    cases = []
    for request in requests:
        prompts.clear()
        state = shim.with_image(request["state"])
        probabilities = {}
        for name, question in request["questions"].items():
            if question.get("instructions") is None:  # the helper needs instructions; the engine refuses these too
                probabilities = None
                break
            answer, _ = shim.ANSWER[question["type"]](state, question)
            if question["type"] == "noul":
                probabilities[name] = {"true": answer["noul"], "false": 1 - answer["noul"]}
            else:
                probabilities[name] = answer["probabilities"]
        if probabilities is not None:
            cases.append({"request": request, "input_ids": [list(p) for p in prompts], "probabilities": probabilities})
    return cases


def d1(model_dir, requests, set_size):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32)
    engine = model.engine
    cases = []
    for request in requests:
        try:
            questions = list(engine.named(request["questions"]).values())
        except ValueError as e:
            cases.append({"request": request, "refused": str(e)})
            continue
        asked = [(request["state"], questions, [])]
        sequences = engine._sequences(asked, set_size=set_size)
        probs = engine.run(asked, set_size=set_size)[0][0]
        probabilities = {}
        for (name, question), p in zip(request["questions"].items(), probs):
            names = (["true", "false"] if question["type"] == "noul" else
                     list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(p))])
            probabilities[name] = dict(zip(names, p))
        cases.append({"request": request, "input_ids": [list(s[2]) for s in sequences],
                      "probabilities": probabilities})
    return cases, engine.set_size if set_size is None else set_size


def is_d1(model_dir):
    path = os.path.join(model_dir, "config.json")  # Laya's variant folders have none
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return json.load(f).get("architectures") == ["D1Model"]




def laya(model_dir, requests):
    sys.path.insert(0, model_dir)
    import rl_agent_api

    agent = rl_agent_api.RLAgent(model_dir, device="cpu")
    rl_agent_api.round = lambda value, digits=None: value  # system_one rounds to 4 digits
    sequences = []

    def build(*args, **kwargs):
        ids, markers = build_sequence(*args, **kwargs)
        sequences.append(list(ids))
        return ids, markers

    build_sequence = rl_agent_api.build_sequence
    rl_agent_api.build_sequence = build
    cases = []
    for request in requests:
        sequences.clear()
        questions = {name: dict(q, instructions=q.get("instructions")) for name, q in request["questions"].items()}
        try:
            answers = agent.system_one(request["state"], questions)["answers"]
        except (ValueError, TypeError) as e:
            cases.append({"request": request, "refused": str(e)})
            continue
        probabilities = {}
        for name, a in answers.items():
            if a["type"] == "noul":
                probabilities[name] = {"true": a["noul"], "false": 1 - a["noul"]}
            else:
                probabilities[name] = a["probabilities"]
        cases.append({"request": request, "input_ids": list(sequences), "probabilities": probabilities})
    return cases


def julia(model_dir, requests):
    sys.path.insert(0, model_dir)
    import julia.router.engine as engine_module
    from julia import load_model

    with open(os.path.join(model_dir, "inference-policy.json")) as f:
        policy = json.load(f)
    engine = load_model(model_dir, device="cpu", max_length=policy["max_length"], head_length=policy["head_length"],
                        strict_encoding=policy["strict_encoding"])
    torch.set_float32_matmul_precision("highest")  # its configure() allows reduced-precision float32 matmuls
    encoder = engine.model.encoder
    if not hasattr(type(encoder), "_update_attention_mask") and "forward" in vars(encoder):
        del encoder.forward  # the specialized forward needs transformers 5.0 (see above)
    sequences = []

    def sequence(*args, **kwargs):
        encoded = encode(*args, **kwargs)
        sequences.append(list(encoded["ids"]))
        return encoded

    encode = engine_module.sequence
    engine_module.sequence = sequence
    cases = []
    for request in requests:
        sequences.clear()
        engine.clear_cache()
        try:
            answers = engine.predict(state=request["state"], questions=request["questions"])["answers"]
        except ValueError as e:
            cases.append({"request": request, "refused": str(e)})
            continue
        probabilities = {name: a["probabilities"] for name, a in answers.items()}
        cases.append({"request": request, "input_ids": list(sequences), "probabilities": probabilities})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--set-size", type=int, default=None, help="d1: questions read together (default: the model's)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    with open(args.requests) as f:
        requests = json.load(f)["requests"]
    model_dir = os.path.abspath(args.model_dir)
    set_size = None
    if is_d1(model_dir):
        kind, (cases, set_size) = "d1", d1(model_dir, requests, args.set_size)
    elif os.path.exists(os.path.join(model_dir, "joint_head_config.json")):
        kind, cases = "clef", clef(model_dir, requests)
    elif os.path.exists(os.path.join(model_dir, "decision_config.json")):
        kind, cases = "pplx_decider", pplx(model_dir, requests)
    elif os.path.exists(os.path.join(model_dir, "helper", "shim.py")):
        kind, cases = "letter_readout", letters(model_dir, requests)
    elif os.path.exists(os.path.join(model_dir, "rl_agent_config.json")):
        kind, cases = "laya", laya(model_dir, requests)
    elif os.path.exists(os.path.join(model_dir, "julia_config.json")):
        kind, cases = "julia", julia(model_dir, requests)
    else:
        ap.error(f"{model_dir} holds no decision model this script knows")
    fixtures = {"kind": kind, "cases": cases}
    if kind == "letter_readout":
        fixtures["preset"] = "openjev"  # the settings the helper ran with
    if set_size is not None:
        fixtures["set_size"] = set_size
    with open(args.out, "w") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    tokens = sum(len(ids) for c in cases for ids in c.get("input_ids", []))
    refused = sum("refused" in c for c in cases)
    print(f"{kind}: {len(cases)} requests ({refused} refused), {tokens} tokens -> {args.out}")


if __name__ == "__main__":
    main()
