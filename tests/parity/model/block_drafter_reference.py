#!/usr/bin/env python3
"""Reference rounds of a block drafter (DFlash, DFlash 2, DSpine) on a target checkpoint (test-only).

    python -I tests/parity/model/block_drafter_reference.py --model-dir <target> --drafter <drafter checkpoint>
        --dflash-repo <clone of github.com/z-lab/dflash> [--dspine-repo <clone of github.com/Tencent/DSpine>]
        [--out tests/fixtures/generated/<model_type>] [--rounds 4] [--accept 2,0,5,1] [--prompts chat_short,...]
        [--prompts-file <one user message a line> [--max-tokens 128]]

Target and drafter run in float32 on the CPU, without caches: every round is one forward over the whole sequence.
The drafter is z-lab's torch model (dflash/model.py). A DSpine checkpoint is that model's layers with the code bridge
written after each of them and the last write redone per candidate, both by DSpine's own plain-torch references
(tests/ref_utils.py: ref_bridge_forward, ref_refine_table).

Writes <out>/<variant>.json (variant: dflash, dflash2 or dspine). Per prompt, a list of rounds, each one block drafted
after the tokens committed so far:
  anchor      the pending token the block starts from
  drafts      the block's proposal
  accepted    how many drafts the next round counts as committed: the target's own verdict, or --accept's pattern
              (a random tiny checkpoint accepts nothing, and a context that grows by several rows a round is the case
              worth testing)
  candidates  [drafts][K] token ids and scores [drafts][K senders][K candidates] (<variant>.<prompt>.r<i>.scores.bin,
              float32): DFlash 2's selector and DSpine's refinement. Sender j of a draft is candidate j of the draft
              before (the anchor for the first). DSpine's scores are logits of the normalized state, so they equal
              unnormalized ones up to a positive factor per sender.
  margins     DFlash only: per draft, the gap between its two best logits over the logit range.

With --prompts-file the prompts are that file's lines as user messages (thinking off), and a prompt's rounds also end
at a stop token or after --max-tokens tokens: the reference's tokens per round on a real checkpoint, printed at the end.
"""

import argparse
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_reference import build_prompts  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_of(config):
    arch = config["architectures"][0]
    if arch == "DFlash2DraftModel":
        return "dflash2"
    if arch == "DFlashDraftModel":
        return "dspine" if "selected_prefix_code_bridge" in config["dflash_config"] else "dflash"
    raise SystemExit(f"unsupported drafter architecture {arch}")


class Bridge:
    """A DSpine checkpoint's code bridge, as ref_bridge_forward reads it."""

    def __init__(self, path, config, block_size):
        from safetensors.torch import load_file

        t = {k: v.float() for k, v in load_file(os.path.join(path, "model.safetensors")).items() if v.is_floating_point()}
        b = "selected_prefix_bridge."
        cfg = config["dflash_config"]["selected_prefix_code_bridge"]
        self.block_size = block_size
        self.rank, self.code_size, self.eps = int(cfg["rank"]), int(cfg["code_size"]), float(cfg["eps"])
        self.codebook = t[b + "codebook"]
        self.read = [*t[b + "early_read_weights"], t[b + "read_weight"]]
        self.feature_write_weights = t[b + "feature_write_weights"]
        self.gate = SimpleNamespace(weight=t[b + "gate.weight"], bias=t[b + "gate.bias"])
        self.predecessor_gate = t[b + "predecessor_gate.weight"]
        self.prev = t[b + "prev.weight"]
        self.packed_weight = torch.cat([self.gate.weight, self.predecessor_gate, self.prev])

    def layer_read_weight(self, layer):
        return self.read[layer]


def dspine_block(model, bridge, ref, context, noise, position_ids, block_ids):
    """The block through z-lab's layers with DSpine's write after each: the final normed rows and the last write's
    stash (raw, rms, packed)."""
    ctx = model.hidden_norm(model.fc(context))
    pe = model.rotary_emb(noise, position_ids)
    residual = noise
    x = model.layers[0].input_layernorm(noise)
    last = len(model.layers) - 1
    stash = None
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn(hidden_states=x, target_hidden=ctx, position_embeddings=pe, attention_mask=None)[0]
        h = residual + attn
        out = h + layer.mlp(layer.post_attention_layernorm(h))
        stash = {} if i == last else None
        next_norm = model.layers[i + 1].input_layernorm if i < last else model.norm
        x, residual = ref.ref_bridge_forward(bridge, out[0], None, block_ids, i, next_norm, stash)
        x, residual = x[None], residual[None]
    return x[0], stash


def walk(table, candidates):
    """The greedy path: each draft's best candidate given the one the draft before took."""
    drafts, j = [], 0
    for i in range(table.shape[0]):
        j = int(table[i, j].argmax())
        drafts.append(int(candidates[i, j]))
    return drafts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--dflash-repo", required=True)
    ap.add_argument("--dspine-repo")
    ap.add_argument("--out", help="default: tests/fixtures/generated/<model_type>")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--accept", help="drafts counted as accepted, round by round (e.g. 2,0,5,1); default: the target's")
    ap.add_argument("--prompts", help="comma-separated prompt names; default: all but the long one")
    ap.add_argument("--prompts-file", help="one user message a line, instead of the built-in prompts")
    ap.add_argument("--max-tokens", type=int, default=0, help="end a prompt's rounds after this many tokens (0: never)")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    with open(os.path.join(args.model_dir, "config.json")) as f:
        target_config = json.load(f)
    with open(os.path.join(args.drafter, "config.json")) as f:
        config = json.load(f)
    variant = variant_of(config)
    args.out = args.out or os.path.join("tests", "fixtures", "generated", target_config["model_type"])
    os.makedirs(args.out, exist_ok=True)
    forced = [int(x) for x in args.accept.split(",")] if args.accept else None

    dflash = load_module("dflash_model", os.path.join(args.dflash_repo, "dflash", "model.py"))
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    auto = AutoModelForImageTextToText if "vision_config" in target_config else AutoModelForCausalLM
    target = auto.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    cls = dflash.DFlash2DraftModel if variant == "dflash2" else dflash.DFlashDraftModel
    model = cls.from_pretrained(args.drafter, dtype=torch.float32).eval()
    head = target.get_output_embeddings()
    embed = target.get_input_embeddings().weight
    block_size, mask = int(model.block_size), int(model.mask_token_id)
    bridge = ref = top_k = None
    if variant == "dspine":
        if not args.dspine_repo:
            raise SystemExit("a DSpine drafter needs --dspine-repo (its tests/ref_utils.py)")
        ref = load_module("dspine_ref_utils", os.path.join(args.dspine_repo, "tests", "ref_utils.py"))
        bridge = Bridge(args.drafter, config, block_size)
        top_k = int(config["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"])

    if args.prompts_file:
        with open(args.prompts_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        prompts = {f"line{i}": tok.apply_chat_template([{"role": "user", "content": line}], tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=False)
                   for i, line in enumerate(lines)}
        prompts = {k: list(v["input_ids"] if hasattr(v, "keys") else v) for k, v in prompts.items()}
        names = list(prompts)
    else:
        prompts = build_prompts(tok)
        names = args.prompts.split(",") if args.prompts else [n for n in prompts if n != "chat_long"]
    eos = target.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos]) if args.prompts_file else set()
    emitted = verifies = 0
    manifest = {"drafter_dir": os.path.abspath(args.drafter), "variant": variant, "block_size": block_size, "cases": []}
    for name in names:
        committed = list(prompts[name])
        with torch.no_grad():
            out = target(input_ids=torch.tensor([committed]), output_hidden_states=True)
        anchor = int(out.logits[0, -1].argmax())
        features = dflash.extract_context_feature(out.hidden_states, model.target_layer_ids)
        rounds = []
        for r in range(args.rounds):
            start = len(committed)
            if anchor in stops or (args.max_tokens and start + 1 - len(prompts[name]) >= args.max_tokens):
                break
            block = torch.tensor([[anchor] + [mask] * (block_size - 1)])
            positions = torch.arange(start + block_size)[None]
            noise = F.embedding(block, embed)
            entry = {"anchor": anchor}
            with torch.no_grad():
                if variant == "dspine":
                    rows, stash = dspine_block(model, bridge, ref, features, noise, positions, block)
                    cand = torch.topk(head(rows[1:]), top_k).indices.sort(-1).values
                    table = ref.ref_refine_table(
                        stash["raw"], stash["rms"], stash["packed"][:, : bridge.rank], torch.tensor([anchor]), cand,
                        bridge.codebook, bridge.predecessor_gate, bridge.prev, bridge.gate.bias,
                        bridge.feature_write_weights[-1], model.norm.weight, head.weight,
                        float(model.norm.variance_epsilon), block_size)
                    drafts = walk(table, cand)
                else:
                    hidden = model(target_hidden=features, noise_embedding=noise, position_ids=positions)[:, 1:, :]
                    logits = model.compute_logits(hidden, head)
                    if variant == "dflash2":
                        path, cand, _ = model.propose(hidden, block[:, 0], head, 0.0)
                        drafts, cand = path[0].tolist(), cand[0]
                        sel = model.candidate_selector
                        senders = torch.cat([block[:, :1].expand(1, cand.shape[1]), cand[:-1]])
                        table = logits[0].gather(-1, cand)[:, None, :] + torch.einsum(
                            "njr,nr,nkr->njk", sel.predecessor_codebook(senders), sel.hidden_projection(hidden)[0],
                            sel.successor_codebook(cand))
                        assert walk(table, cand) == drafts, "the score table does not reproduce the selector's path"
                    else:
                        drafts = logits[0].argmax(-1).tolist()
                        top2 = torch.topk(logits[0], 2).values
                        span = logits[0].max(-1).values - logits[0].min(-1).values
                        entry["margins"] = ((top2[:, 0] - top2[:, 1]) / span).tolist()
                        table = None
                if table is not None:
                    file = f"{variant}.{name}.r{r}.scores.bin"
                    table.numpy().astype(np.float32).tofile(os.path.join(args.out, file))
                    entry.update({"candidates": cand.tolist(), "scores": file})
                sequence = committed + [anchor] + drafts
                out = target(input_ids=torch.tensor([sequence]), output_hidden_states=True)
            posterior = out.logits[0, start:].argmax(-1).tolist()
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == posterior[accepted]:
                accepted += 1
            entry["target_accepts"] = accepted
            if forced is not None:
                accepted = min(forced[r % len(forced)], len(drafts))
            entry.update({"drafts": drafts, "accepted": accepted})
            rounds.append(entry)
            committed = sequence[: start + 1 + accepted]
            emitted += 1 + accepted
            verifies += 1
            features = dflash.extract_context_feature(out.hidden_states, model.target_layer_ids)[:, : len(committed)]
            anchor = posterior[accepted]
        manifest["cases"].append({"name": name, "input_ids": list(prompts[name]), "rounds": rounds})
        if args.prompts_file:
            taken = sum(1 + e["accepted"] for e in rounds)
            print(f"{name}: {taken} tokens in {len(rounds)} rounds, {taken / max(len(rounds), 1):.2f} per round", flush=True)
        else:
            print(f"{name}: " + "; ".join(f"{tok.decode(e['drafts'])!r} ({e['target_accepts']} accepted)" for e in rounds),
                  flush=True)
    if args.prompts_file:
        print(f"{variant}: {emitted} tokens in {verifies} rounds, {emitted / max(verifies, 1):.2f} per round")
    with open(os.path.join(args.out, f"{variant}.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
