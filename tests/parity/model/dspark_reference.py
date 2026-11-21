#!/usr/bin/env python3
"""Reference rounds of Inkling 2's block drafter (DSpark) for tests/unit/model/dspark_parity.cpp.

No PyTorch module of this drafter is public: this is its math written out in float64, as llama.cpp (common/dspark.cpp
in Unsloth's builds) and vLLM's port compute it, over states read from transformers' Inkling 2 with hooks. It reads
the cases hf_reference.py wrote for the same checkpoint (<fixtures>/manifest.json: a prompt and the target's greedy
continuation) and drafts round after round along that continuation, each round keeping the drafts the target agrees
with.

  python tests/parity/model/dspark_reference.py --model-dir <checkpoint> --fixtures <dir of manifest.json>

Writes <fixtures>/dspark.json: per case its rounds (the pending token, the drafts, how many the target accepts, and
each draft's lead over the runner-up, in logits).
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, positions, theta):
    """x [T, heads, D], rotated by halves at each token's position."""
    half = x.shape[-1] // 2
    inv = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    angle = positions.double()[:, None, None] * inv
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()], -1)


class Drafter:
    def __init__(self, model_dir, config):
        self.c = config["mtp_config"]
        with safe_open(os.path.join(model_dir, "mtp.safetensors"), "pt") as f:
            self.w = {k: f.get_tensor(k).double() for k in f.keys()}
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            where = json.load(f)["weight_map"]
        self.target = {}
        for name in ("embed.weight", "embed_norm.weight", "unembed.weight"):
            with safe_open(os.path.join(model_dir, where["model.llm." + name]), "pt") as f:
                self.target[name] = f.get_tensor("model.llm." + name)
        self.tokens = config["text_config"]["unpadded_vocab_size"]
        self.alpha = torch.softmax(self.w["fuse_weight"], 1)

    def mix(self, taps, row, norm):
        """taps [P, states, H] -> the mix of one reader, normed: [P, H]."""
        return rms((taps * self.alpha[row][None, :, None]).sum(1), self.w[norm], self.c["rms_norm_eps"])

    def context(self, taps):
        """Each layer's context keys and values over every position, and the head's context rows."""
        c, positions = self.c, torch.arange(taps.shape[0])
        shape = (-1, c["num_key_value_heads"], c["head_dim"])
        keys, values = [], []
        for l in range(c["num_hidden_layers"]):
            a = f"layers.{l}.self_attn."
            x = self.mix(taps, l, f"layers.{l}.dflare_norm.weight")
            k = rms((x @ self.w[a + "k_proj_ctx.weight"].T).view(shape), self.w[a + "k_norm_ctx.weight"], c["rms_norm_eps"])
            keys.append(rope(k, positions, c["rope_theta"]))
            values.append((x @ self.w[a + "v_proj_ctx.weight"].T).view(shape))
        return keys, values, self.mix(taps, c["num_hidden_layers"], "markov_head.dflare_norm.weight")

    def block(self, anchor, at, keys, values, context):
        """The drafts after `anchor`, which sits at position `at`, and each draft's lead over the runner-up."""
        c, w, eps = self.c, self.w, self.c["rms_norm_eps"]
        rows, heads, kv, dim = c["block_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        first = rms(self.target["embed.weight"][anchor].double(), self.target["embed_norm.weight"].double(), eps)
        x = torch.stack([first] + [w["mask_embed"]] * (rows - 1))
        positions = at + torch.arange(rows)
        for l in range(c["num_hidden_layers"]):
            a = f"layers.{l}.self_attn."
            y = rms(x, w[f"layers.{l}.input_layernorm.weight"], eps)
            q = rope(rms((y @ w[a + "q_proj.weight"].T).view(rows, heads, dim), w[a + "q_norm.weight"], eps), positions,
                     c["rope_theta"])
            k = rope(rms((y @ w[a + "k_proj.weight"].T).view(rows, kv, dim), w[a + "k_norm.weight"], eps), positions,
                     c["rope_theta"])
            v = (y @ w[a + "v_proj.weight"].T).view(rows, kv, dim)
            # a local layer sees the window before the block's start, the same keys for every row
            lo = max(0, at - c["sliding_window_size"]) if l in c["local_layer_ids"] else 0
            all_k = torch.cat([keys[l][lo:at], k]).repeat_interleave(heads // kv, 1)
            all_v = torch.cat([values[l][lo:at], v]).repeat_interleave(heads // kv, 1)
            scores = torch.einsum("ihd,jhd->hij", q, all_k) / dim ** 0.5
            o = torch.einsum("hij,jhd->ihd", scores.softmax(-1), all_v).reshape(rows, heads * dim)
            x = x + o @ w[a + "o_proj.weight"].T
            z = rms(x, w[f"layers.{l}.post_attention_layernorm.weight"], eps)
            gate = torch.nn.functional.silu(z @ w[f"layers.{l}.mlp.gate_proj.weight"].T)
            x = x + (gate * (z @ w[f"layers.{l}.mlp.up_proj.weight"].T)) @ w[f"layers.{l}.mlp.down_proj.weight"].T
        h = rms(x, w["norm.weight"], eps)

        rank = c["markov_rank"]
        state, previous, drafts, leads = torch.zeros(rank, dtype=torch.float64), anchor, [], []
        head = self.target["unembed.weight"][: self.tokens]
        for i in range(1, rows):
            z = w["markov_head.rnn_proj.weight"] @ torch.cat([state, w["markov_head.markov_w1.weight"][previous], h[i],
                                                             context]) + w["markov_head.rnn_proj.bias"]
            gate, candidate, latent = z[:rank].sigmoid(), z[rank : 2 * rank].tanh(), z[2 * rank :].tanh()
            state = gate * state + (1 - gate) * candidate
            corrected = h[i] + w["markov_head.hidden_out.weight"] @ latent + w["markov_head.hidden_out.bias"]
            logits = (head.float() @ corrected.float()).double()
            top = logits.topk(2)
            previous = int(top.indices[0])
            drafts.append(previous)
            leads.append(float(top.values[0] - top.values[1]))
        return drafts, leads


def target_states(model, ids, taps):
    """The stream entering each layer visit of `taps` (the last one: the state the output layer reads): [P, 4, H]."""
    text = model.model.llm if hasattr(model.model, "llm") else model.model.language_model
    entered, hooks = [], []
    for layer in text.layers:
        hooks.append(layer.register_forward_pre_hook(lambda m, args, kwargs: entered.append(args[0][0].double()),
                                                     with_kwargs=True))
    final = []
    hooks.append(text.norm.register_forward_hook(lambda m, args, out: final.append(out[0].double())))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    for h in hooks:
        h.remove()
    assert len(entered) == taps[-1], f"{len(entered)} layer visits, the drafter counts {taps[-1]}"
    return torch.stack([entered[t] for t in taps[:-1]] + [final[0]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--fixtures", required=True, help="the directory hf_reference.py wrote for this checkpoint")
    ap.add_argument("--cases", help="comma-separated case names; default: all")
    ap.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.fixtures, "manifest.json")) as f:
        cases = json.load(f)["cases"]
    wanted = args.cases.split(",") if args.cases else None
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.float32,
                                                        attn_implementation="eager").eval()
    drafter = Drafter(args.model_dir, config)
    out = {"block_size": drafter.c["block_size"], "cases": []}
    for case in cases:
        if wanted and case["name"] not in wanted:
            continue
        prompt, greedy = case["input_ids"], case["greedy"]
        sequence = prompt + greedy
        keys, values, context = drafter.context(target_states(model, sequence, drafter.c["target_layer_ids"]))
        rounds, at = [], len(prompt)
        while at + drafter.c["block_size"] <= len(sequence):
            drafts, leads = drafter.block(sequence[at], at, keys, values, context[at - 1])
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == sequence[at + 1 + accepted]:
                accepted += 1
            rounds.append({"anchor": sequence[at], "drafts": drafts, "accepted": accepted, "leads": leads})
            at += accepted + 1
        out["cases"].append({"name": case["name"], "input_ids": prompt, "rounds": rounds})
        drafted = sum(len(r["drafts"]) for r in rounds)
        print(f"{case['name']}: {len(rounds)} rounds, {sum(r['accepted'] for r in rounds)} of {drafted} drafts accepted")
    with open(os.path.join(args.fixtures, "dspark.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
