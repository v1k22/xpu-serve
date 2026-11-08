"""Model-aware quantization tools for the manual-quantization experiment (branch exp/manual-quant).

Packs written here are read by xpu-serve as any pack (src/io/safetensors.cpp, quant/quant.h layouts):
  q8: int8 codes [rows, K], f16 scale per group (value = code * scale)
  q4: uint8 [rows, K / 2], even column in the low nibble
  q3: 12 bytes per 32 columns: bytes 0-7 the low two bits (column j at byte j / 4, bit 2 (j % 4)), bytes 8-11 the
      high bit (column j at byte 8 + j / 8, bit j % 8)
  q2: 8 bytes per 32 columns (the q3 low-bit plane)
and q4 / q3 / q2 carry an f16 bias per group (value = code * scale + bias). An expert bank [E, R, K] is E * R rows.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PACK_VERSION = 1
SHARD_BYTES = 4 << 30


def storage_cols(bits, k):
    return k if bits == 8 else k * bits // 8


def pack_codes(codes, bits):
    """codes: uint8 [rows, K] in [0, 2^bits) (q8: int8) -> stored bytes [rows, storage_cols]."""
    rows, k = codes.shape
    if bits == 8:
        return codes.to(torch.int8)
    c = codes.to(torch.uint8)
    if bits == 4:
        return c[:, 0::2] | (c[:, 1::2] << 4)
    chunks = c.reshape(rows, k // 32, 32)
    lo = chunks & 3
    lo = lo.reshape(rows, k // 32, 8, 4)
    lo_bytes = lo[..., 0] | (lo[..., 1] << 2) | (lo[..., 2] << 4) | (lo[..., 3] << 6)  # [rows, k/32, 8]
    if bits == 2:
        return lo_bytes.reshape(rows, k // 4)
    hi = ((chunks >> 2) & 1).reshape(rows, k // 32, 4, 8)
    hi_bytes = torch.zeros(rows, k // 32, 4, dtype=torch.uint8, device=c.device)
    for i in range(8):
        hi_bytes |= hi[..., i] << i
    return torch.cat([lo_bytes, hi_bytes], dim=-1).reshape(rows, k // 32 * 12)


def unpack_codes(stored, bits, k):
    rows = stored.shape[0]
    if bits == 8:
        return stored.to(torch.int16)
    s = stored.to(torch.uint8)
    if bits == 4:
        out = torch.empty(rows, k, dtype=torch.uint8, device=s.device)
        out[:, 0::2] = s & 15
        out[:, 1::2] = s >> 4
        return out
    per = 12 if bits == 3 else 8
    chunks = s.reshape(rows, k // 32, per)
    lo = chunks[..., :8]
    vals = torch.stack([(lo >> (2 * i)) & 3 for i in range(4)], dim=-1).reshape(rows, k // 32, 32)
    if bits == 3:
        hi = chunks[..., 8:]
        hb = torch.stack([(hi >> i) & 1 for i in range(8)], dim=-1).reshape(rows, k // 32, 32)
        vals = vals | (hb << 2)
    return vals.reshape(rows, k)


def dequant(codes, scales, biases, bits, group):
    """codes [rows, K] (values) -> float32 [rows, K]."""
    rows, k = codes.shape
    c = codes.float().reshape(rows, k // group, group)
    out = c * scales.float().unsqueeze(-1)
    if bits != 8:
        out = out + biases.float().unsqueeze(-1)
    return out.reshape(rows, k)


CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def quantize_rtn(w, bits, group, col_weight=None, clips=CLIPS):
    """Round to nearest per group with a clip search (the range shrunk around its middle) that minimizes
    sum col_weight[c] * (deq - w)^2 per group. w: float [rows, K]; col_weight: [K] or [rows, K] (e.g. activation mean
    squares). Returns (codes uint8/int16 values [rows, K], scales f16, biases f16 or None)."""
    w = w.float()
    dev = w.device
    rows, k = w.shape
    g = k // group
    wg = w.reshape(rows, g, group)
    cw = torch.ones(1, 1, group, device=dev) if col_weight is None else col_weight.float().reshape(-1, g, group)
    if bits == 8:
        amax = wg.abs().amax(-1)
        s = (amax / 127).half()
        sf = s.float().clamp(min=1e-30)
        codes = torch.clamp(torch.round(wg / sf.unsqueeze(-1)), -127, 127).to(torch.int16)
        return codes.reshape(rows, k), s, None
    maxq = (1 << bits) - 1
    lo0, hi0 = wg.amin(-1), wg.amax(-1)
    mid = (lo0 + hi0) / 2
    best_err = torch.full((rows, g), float("inf"), device=dev)
    best_codes = torch.zeros(rows, g, group, dtype=torch.uint8, device=dev)
    best_s = torch.zeros(rows, g, dtype=torch.float16, device=dev)
    best_b = torch.zeros(rows, g, dtype=torch.float16, device=dev)
    for c in clips:
        lo = mid - (mid - lo0) * c
        hi = mid + (hi0 - mid) * c
        s = ((hi - lo) / maxq).half()
        b = lo.half()
        sf = s.float().clamp(min=1e-30).unsqueeze(-1)
        bf = b.float().unsqueeze(-1)
        q = torch.clamp(torch.round((wg - bf) / sf), 0, maxq)
        err = (((q * sf + bf) - wg) ** 2 * cw).sum(-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_codes = torch.where(better.unsqueeze(-1), q.to(torch.uint8), best_codes)
        best_s = torch.where(better, s, best_s)
        best_b = torch.where(better, b, best_b)
    return best_codes.reshape(rows, k), best_s, best_b


def gptq_prepare(w, h, damp=0.01):
    """The part of GPTQ every format shares: zeroes the columns no calibration input reaches and returns
    (w, upper Cholesky factor of the damped inverse Hessian, diag(H)). w: [B, rows, K]; h: [B, K, K]."""
    w = w.float().clone()
    h = h.float().clone()
    k = w.shape[-1]
    ar = torch.arange(k, device=w.device)
    diag = torch.diagonal(h, dim1=1, dim2=2).clone()
    dead = diag == 0
    diag_safe = torch.where(dead, torch.ones_like(diag), diag)
    h[:, ar, ar] = diag_safe
    w[dead.unsqueeze(1).expand_as(w)] = 0
    h[:, ar, ar] += damp * diag_safe.mean(-1, keepdim=True)
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return w, torch.linalg.cholesky(hinv, upper=True), diag


def gptq_run(w, hinv, diag, bits, group, block=128, clips=CLIPS):
    """GPTQ (Frantar et al.) with per-group asymmetric codes and the engine's f16 scale / bias, on gptq_prepare's
    output. Group parameters are set when a group starts, from the error-compensated weights, with a clip search
    weighted by diag(H). Returns (codes, scales, biases) shaped [B, rows, ...]."""
    w = w.clone()
    nb, rows, k = w.shape
    dev = w.device
    assert block % group == 0 and k % group == 0
    maxq = (1 << bits) - 1
    g = k // group
    codes = torch.zeros(nb, rows, k, dtype=torch.int16 if bits == 8 else torch.uint8, device=dev)
    scales = torch.zeros(nb, rows, g, dtype=torch.float16, device=dev)
    biases = None if bits == 8 else torch.zeros(nb, rows, g, dtype=torch.float16, device=dev)
    sf = bf = None
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        w1 = w[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        h1 = hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                gi = col // group
                seg = w1[:, :, i:i + group]  # block % group == 0: a group never straddles blocks
                cw = diag[:, col:col + group].unsqueeze(1)  # [nb, 1, group]
                c, s, b = quantize_rtn(seg.reshape(nb * rows, group), bits, group,
                                       col_weight=cw.expand(nb, rows, group).reshape(nb * rows, group), clips=clips)
                scales[:, :, gi] = s.reshape(nb, rows)
                if bits != 8:
                    biases[:, :, gi] = b.reshape(nb, rows)
                sf = s.float().reshape(nb, rows).clamp(min=1e-30)
                bf = None if bits == 8 else b.float().reshape(nb, rows)
            x = w1[:, :, i]
            if bits == 8:
                q = torch.clamp(torch.round(x / sf), -127, 127)
                deq = q * sf
            else:
                q = torch.clamp(torch.round((x - bf) / sf), 0, maxq)
                deq = q * sf + bf
            codes[:, :, col] = q.to(codes.dtype)
            d = h1[:, i, i].unsqueeze(1)
            e = (x - deq) / d
            w1[:, :, i:] -= e.unsqueeze(-1) * h1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = e
        w[:, :, i2:] -= torch.bmm(err1, hinv[:, i1:i2, i2:])
    return codes, scales, biases


def gptq(w, h, bits, group, damp=0.01, block=128, clips=CLIPS):
    """GPTQ on w: [rows, K] with h: [K, K] = sum x x^T over the calibration inputs (may be router-weighted), or
    batched over a leading dim (w [B, rows, K], h [B, K, K]). Returns (codes, scales, biases) like quantize_rtn's."""
    batched = w.dim() == 3
    if not batched:
        w, h = w.unsqueeze(0), h.unsqueeze(0)
    codes, scales, biases = gptq_run(*gptq_prepare(w, h, damp), bits, group, block, clips)
    if not batched:
        return codes[0], scales[0], None if biases is None else biases[0]
    return codes, scales, biases


def output_error(w, deq, h):
    """Per batch element: (tr(dW H dW^T), tr(W H W^T)), the squared error the quantization adds to the layer's output
    summed over the calibration tokens, and the output's own energy. w, deq: [B, rows, K]; h: [B, K, K]."""
    d = deq - w
    return (torch.bmm(d, h) * d).sum((1, 2)), (torch.bmm(w, h) * w).sum((1, 2))


class Checkpoint:
    """Tensors of a safetensors checkpoint directory, read one at a time."""

    def __init__(self, path):
        self.path = path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self._open = {}

    def names(self):
        return list(self.where)

    def get(self, name):
        f = self.where[name]
        if f not in self._open:
            self._open[f] = safe_open(os.path.join(self.path, f), framework="pt")
        return self._open[f].get_tensor(name)


class PackWriter:
    """Writes a pack (or an overlay: only the quantized entries) in shards of at most SHARD_BYTES."""

    def __init__(self, out, source_name="", model_type=""):
        os.makedirs(out, exist_ok=True)
        self.out = out
        self.manifest = {"pack_version": PACK_VERSION, "converter": "tools/manual_quant", "source":
                         {"name": source_name, "model_type": model_type}, "tensors": {}}
        self.pending = {}
        self.pending_bytes = 0
        self.weight_map = {}
        self.shards = []
        self.total = 0

    def _add(self, name, t):
        t = t.contiguous()
        self.pending[name] = t
        self.pending_bytes += t.numel() * t.element_size()
        self.total += t.numel() * t.element_size()
        if self.pending_bytes >= SHARD_BYTES:
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        file = f"shard-{len(self.shards):05d}.safetensors"
        save_file(self.pending, os.path.join(self.out, file), metadata={"format": "pt"})
        for n in self.pending:
            self.weight_map[n] = file
        self.shards.append(file)
        self.pending, self.pending_bytes = {}, 0

    def quantized(self, name, shape, bits, group, codes, scales, biases):
        """codes: values [rows, K] (rows = numel / K); stored with the leading dims of `shape`."""
        k = shape[-1]
        lead = list(shape[:-1])
        self._add(name, pack_codes(codes, bits).reshape(lead + [storage_cols(bits, k)]))
        self._add(name + ".scales", scales.reshape(lead + [k // group]).to(torch.float16))
        if bits != 8:
            self._add(name + ".biases", biases.reshape(lead + [k // group]).to(torch.float16))
        self.manifest["tensors"][name] = {"format": f"q{bits}", "group": group, "shape": list(shape)}

    def copy(self, name, t):
        self._add(name, t)

    def close(self):
        self._flush()
        json.dump({"metadata": {"total_size": self.total}, "weight_map": self.weight_map},
                  open(os.path.join(self.out, "model.safetensors.index.json"), "w"), indent=1)
        json.dump(self.manifest, open(os.path.join(self.out, "xpu_pack.json"), "w"), indent=1)
        return self.total
