#!/usr/bin/env python3
"""Experiment (not part of the engine): rewrites a Qwen3.5-family GGUF as an xpu-serve pack, to measure what loading
GGUF natively would give. Every tensor is mapped back to its Hugging Face name and layout, undoing what llama.cpp's
converter did (norm weights + 1, A_log -> -exp, value heads tiled instead of grouped, the MTP block as blk.<n>).

    python gguf_to_pack.py --gguf X.gguf --side <dir with config.json, tokenizer files> --out <pack dir> [--hf <ckpt>]

4-bit blocks with one scale (and offset) per 32 weights become the pack's q4 with groups of 32 without loss: Q4_0,
Q4_1. Q8_0 becomes q8 with groups of 32 without loss, and so does a Q4_0 embedding table (the engine gathers from
q8). Q5_K and Q6_K have no counterpart in the pack formats: they are decoded and rounded again to q8 (groups of 32).
With --hf every mapped tensor is compared with the original checkpoint's.
"""
import argparse, collections, json, os, shutil, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_gguf import parse  # noqa: E402

GROUP = 32


def f16(x):
    return np.asarray(x, np.float32).astype(np.float16)


def halves(raw, at):
    return raw[:, at:at + 2].copy().view(np.float16).astype(np.float32)[:, 0]


# ---- GGUF blocks -> (codes [blocks, 32] as small ints, scale, bias) or float values

def q4_0(raw):
    b = raw.reshape(-1, 18)
    d = halves(b, 0)
    q = np.concatenate([b[:, 2:] & 15, b[:, 2:] >> 4], axis=1)
    return q, d, -8.0 * d


def q4_1(raw):
    b = raw.reshape(-1, 20)
    q = np.concatenate([b[:, 4:] & 15, b[:, 4:] >> 4], axis=1)
    return q, halves(b, 0), halves(b, 2)


def q8_0(raw):
    b = raw.reshape(-1, 34)
    return b[:, 2:].view(np.int8), halves(b, 0)


def k_scales(s):
    s = s.astype(np.int32)
    sc = np.zeros((len(s), 8), np.int32)
    mn = np.zeros((len(s), 8), np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def q5_k(raw):
    b = raw.reshape(-1, 176)
    d, dmin = halves(b, 0), halves(b, 2)
    sc, mn = k_scales(b[:, 4:16])
    qh, ql = b[:, 16:48], b[:, 48:]
    q = np.zeros((len(b), 8, 32), np.float32)
    for pair in range(4):
        part = ql[:, pair * 32:(pair + 1) * 32]
        q[:, 2 * pair] = (part & 15) + np.where(qh & (1 << (2 * pair)), 16, 0)
        q[:, 2 * pair + 1] = (part >> 4) + np.where(qh & (2 << (2 * pair)), 16, 0)
    return ((d[:, None] * sc)[:, :, None] * q - (dmin[:, None] * mn)[:, :, None]).astype(np.float32).reshape(-1)


def q6_k(raw):
    b = raw.reshape(-1, 210)
    ql, qh = b[:, :128], b[:, 128:192]
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = halves(b, 208)
    y = np.zeros((len(b), 256), np.float32)
    for n in range(2):
        l, h = ql[:, n * 64:(n + 1) * 64].astype(np.int32), qh[:, n * 32:(n + 1) * 32].astype(np.int32)
        qs = [(l[:, :32] & 15) | ((h & 3) << 4), (l[:, 32:] & 15) | (((h >> 2) & 3) << 4),
              (l[:, :32] >> 4) | (((h >> 4) & 3) << 4), (l[:, 32:] >> 4) | (((h >> 6) & 3) << 4)]
        for k, q in enumerate(qs):
            scale = np.repeat(sc[:, n * 8 + 2 * k:n * 8 + 2 * k + 2], 16, axis=1)  # a scale per 16 weights
            y[:, n * 128 + k * 32:n * 128 + (k + 1) * 32] = d[:, None] * scale * (q - 32)
    return y.reshape(-1)


# ---- pack components

def pack_q4(q, scale, bias, rows, cols):
    codes = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8).reshape(rows, cols // 2)
    return {"": codes, ".scales": f16(scale).reshape(rows, cols // GROUP), ".biases": f16(bias).reshape(rows, cols // GROUP)}


def pack_q8(codes, scale, rows, cols):
    return {"": np.ascontiguousarray(codes, np.int8).reshape(rows, cols), ".scales": f16(scale).reshape(rows, cols // GROUP)}


def round_q8(w):
    """Round to nearest, symmetric int8 with a scale per group of 32 (the pack's q8), against the f16 scale."""
    rows, cols = w.shape
    g = w.reshape(-1, GROUP)
    s = f16(np.abs(g).max(1) / 127).astype(np.float32)
    codes = np.clip(np.rint(g / np.where(s == 0, 1, s)[:, None]), -127, 127)
    return pack_q8(codes, s, rows, cols)


def values(parts, fmt, rows, cols):
    """What the engine reads of a packed weight."""
    if fmt == "q4":
        c = parts[""]
        q = np.stack([c & 15, c >> 4], axis=2).reshape(-1, GROUP).astype(np.float32)
        return (parts[".scales"].astype(np.float32).reshape(-1, 1) * q + parts[".biases"].astype(np.float32).reshape(-1, 1)).reshape(rows, cols)
    return (parts[""].reshape(-1, GROUP).astype(np.float32) * parts[".scales"].astype(np.float32).reshape(-1, 1)).reshape(rows, cols)


def grouped(x, unit=1):
    """Value heads from llama.cpp's tiled order back to grouped by key head, along axis 0 (16 key heads, 3 each)."""
    return x.reshape(3, 16, unit, *x.shape[1:]).swapaxes(0, 1).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf")
    args = ap.parse_args()
    from safetensors.numpy import save_file

    os.makedirs(args.out, exist_ok=True)
    head = os.path.join(args.out, "gguf.head")  # the header alone: metadata and the tensor table
    with open(args.gguf, "rb") as f, open(head, "wb") as h:
        h.write(f.read(64 << 20))
    version, kv, tensors, data_start = parse(head)
    os.remove(head)
    blocks = kv["qwen35.block_count"] - kv.get("qwen35.nextn_predict_layers", 0)
    data = np.memmap(args.gguf, dtype=np.uint8, mode="r")
    P = "model.language_model."

    hf = None
    if args.hf:
        from safetensors import safe_open
        index = json.load(open(os.path.join(args.hf, "model.safetensors.index.json")))["weight_map"]
        opened = {}

        def hf(name):
            shard = index[name]
            if shard not in opened:
                opened[shard] = safe_open(os.path.join(args.hf, shard), framework="pt")
            return opened[shard].get_tensor(name).float().numpy()

    BYTES = {"F32": (4, 1), "Q4_0": (18, 32), "Q4_1": (20, 32), "Q8_0": (34, 32), "Q5_K": (176, 256), "Q6_K": (210, 256)}
    out, manifest, errors = {}, {}, collections.defaultdict(list)

    def emit(name, parts, fmt, shape):
        for suffix, a in parts.items():
            out[name + suffix] = a
        manifest[name] = {"format": fmt, "group": GROUP, "shape": list(shape)}

    for t in tensors:
        gname, kind = t["name"], t["type"]
        dims = t["dims"][::-1]  # GGUF lists dimensions innermost first
        n = int(np.prod(dims))
        per, weights = BYTES[kind]
        raw = np.asarray(data[data_start + t["offset"]: data_start + t["offset"] + n // weights * per])

        # the Hugging Face name
        if gname.startswith("blk."):
            _, i, rest = gname.split(".", 2)
            i = int(i)
            L = P + f"layers.{i}." if i < blocks else f"mtp.layers.{i - blocks}."
            table = {"attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
                     "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
                     "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
                     "attn_output.weight": "self_attn.o_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
                     "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
                     "ssm_alpha.weight": "linear_attn.in_proj_a.weight", "ssm_beta.weight": "linear_attn.in_proj_b.weight",
                     "ssm_a": "linear_attn.A_log", "ssm_dt.bias": "linear_attn.dt_bias", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
                     "ssm_norm.weight": "linear_attn.norm.weight", "ssm_out.weight": "linear_attn.out_proj.weight"}
            nextn = {"nextn.eh_proj.weight": "mtp.fc.weight", "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
                     "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight", "nextn.shared_head_norm.weight": "mtp.norm.weight"}
            name = nextn[rest] if rest in nextn else L + table[rest]
        else:
            rest = gname
            name = {"token_embd.weight": P + "embed_tokens.weight", "output.weight": "lm_head.weight", "output_norm.weight": P + "norm.weight"}[gname]
        key = rest + " " + kind

        if kind == "F32":
            w = raw.view(np.float32).reshape(dims).copy()
            if rest in ("ssm_a", "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight"):
                w = grouped(w)
                if rest == "ssm_a":
                    w = np.log(-w)
            elif rest == "ssm_conv1d.weight":
                w[4096:] = grouped(w[4096:], 128)  # the value channels follow the query and key ones
                w = w.reshape(dims[0], 1, dims[1])
            elif rest.endswith("norm.weight") and rest != "ssm_norm.weight":
                w = w - 1  # llama.cpp stores 1 + w for every norm but the linear attention's
            out[name] = w.astype(np.float32)
            got = w
        else:
            rows, cols = dims
            if kind in ("Q4_0", "Q4_1"):
                q, s, b = (q4_0 if kind == "Q4_0" else q4_1)(raw)
                if rest == "token_embd.weight":  # a gather table: q8, whose codes hold the 4-bit ones exactly
                    assert kind == "Q4_0"
                    parts, fmt = pack_q8(q.astype(np.int16) - 8, s, rows, cols), "q8"
                else:
                    parts, fmt = pack_q4(q, s, b, rows, cols), "q4"
            elif kind == "Q8_0":
                codes, s = q8_0(raw)
                parts, fmt = pack_q8(codes, s, rows, cols), "q8"
            else:  # Q5_K, Q6_K: decoded, then rounded to q8
                w = (q5_k if kind == "Q5_K" else q6_k)(raw).reshape(rows, cols)
                if rest == "ssm_out.weight":  # its inputs are the value heads' outputs: columns, 128 a head
                    w = np.ascontiguousarray(grouped(w.T, 128).T)
                errors[key + " (decoded, before rounding again)"].append(
                    float(np.linalg.norm(w - hf(name)) / np.linalg.norm(hf(name))) if hf else 0.0)
                parts, fmt = round_q8(w), "q8"
            if rest == "attn_qkv.weight":  # rows: queries, keys, then values by head
                for a in parts.values():
                    a[4096:] = grouped(a[4096:], 128)
            elif rest == "attn_gate.weight":
                for a in parts.values():
                    a[:] = grouped(a, 128)
            emit(name, parts, fmt, dims)
            got = values(parts, fmt, rows, cols)
        if hf is not None:
            ref = hf(name).reshape(got.shape)
            errors[key].append(float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)))

    for key, e in sorted(errors.items()):
        print(f"  {key:52s} {len(e):4d} tensors, relative error against the checkpoint: median {np.median(e):.4f}, max {max(e):.4f}")

    os.makedirs(args.out, exist_ok=True)
    shards, current, size = [], {}, 0
    for name in sorted(out):
        if size and size + out[name].nbytes > (4 << 30):
            shards.append(current)
            current, size = {}, 0
        current[name] = out[name]
        size += out[name].nbytes
    shards.append(current)
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(args.out, fname), metadata={"format": "pt"})
        weight_map.update({name: fname for name in shard})
    total = sum(a.nbytes for a in out.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(args.out, "model.safetensors.index.json"), "w"))
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(os.path.join(args.side, f), os.path.join(args.out, f))
    config = json.load(open(os.path.join(args.out, "config.json")))
    config.pop("quantization_config", None)
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump({"pack_version": 1, "converter": "gguf_to_pack.py (experiment)",
               "source": {"name": os.path.basename(args.gguf), "model_type": config.get("model_type", "")},
               "quant": {"format": "q4", "group": GROUP, "method": "gguf", "min_elements": 0, "q8_names": [], "scope": "all"},
               "tensors": manifest}, open(os.path.join(args.out, "xpu_pack.json"), "w"), indent=1)
    kinds = collections.Counter(m["format"] for m in manifest.values())
    print(f"{len(out)} tensors ({dict(kinds)} quantized weights) in {len(shards)} shard(s), {total / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
