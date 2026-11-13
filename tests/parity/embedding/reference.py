#!/usr/bin/env python3
"""Reference embeddings of a sentence-transformers checkpoint, from sentence-transformers in float32 on the CPU
(test-only).

    python -I tests/parity/embedding/reference.py --model-dir <checkpoint> --out <fixtures dir>
        [--texts tests/parity/embedding/texts.json] [--lengths N ...]

For each text of the file, then for each length (default: the model's max_seq_length) a passage of filler text that
sentence-transformers cuts to that many tokens: the token ids it embeds (its tokenizer, special tokens included), the
final hidden states (the Transformer module's token embeddings; <out>/case<i>.hidden.f32, float32 little-endian
[T, D]), the pooled vector before any Normalize module and the sentence embedding, in <out>/fixtures.json.
embedding_parity.cpp feeds the engine the same ids and compares. A length past the positions a nomic_bert model was
trained on (max_trained_positions) checks the rotation there.

nomic_bert runs its own modeling code (trust_remote_code); that code calls get_extended_attention_mask, which
transformers 5 removed, so it is put back as transformers 4 wrote it: for one unpadded sequence, a mask of zeros added
to the scores.
"""

import argparse
import json
import os

import torch


def restore_extended_attention_mask(model):
    encoder = model[0].auto_model
    if hasattr(encoder, "get_extended_attention_mask"):
        return

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        dtype = dtype or self.dtype
        mask = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    type(encoder).get_extended_attention_mask = get_extended_attention_mask


def run(model, text, max_length):
    """The ids, hidden states, pooled vector and embedding of `text`, cut to `max_length` tokens."""
    model.max_seq_length = max_length
    features = model.preprocess([text])
    with torch.inference_mode():
        out = model[0](dict(features))
        out = model[1](out)
        pooled = out["sentence_embedding"][0].clone()
        for module in list(model)[2:]:
            out = module(out)
    return (features["input_ids"][0].tolist(), out["token_embeddings"][0].float().numpy(), pooled.tolist(),
            out["sentence_embedding"][0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts.json"))
    ap.add_argument("--lengths", type=int, nargs="*", help="long passages of these token counts")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(os.path.abspath(args.model_dir), device="cpu", trust_remote_code=True)
    model.eval()
    restore_extended_attention_mask(model)
    encoder = model[0].auto_model
    for name, buffer in encoder.named_buffers():  # transformers 5 can leave a custom rotary buffer zeroed
        assert "inv_freq" not in name or bool(buffer.abs().sum() > 0), f"{name} is zero"
    names = [type(m).__name__ for m in model]
    assert names[:2] == ["Transformer", "Pooling"] and names[2:] in ([], ["Normalize"]), names
    limit = model.max_seq_length
    with open(args.texts) as f:
        corpus = json.load(f)
    cases = [(text, limit) for text in corpus["texts"]]
    for length in args.lengths or [limit]:
        cases.append((corpus["filler"] * (2 + length // 40), length))

    os.makedirs(args.out, exist_ok=True)
    fixtures = {"model": os.path.basename(os.path.abspath(args.model_dir)), "max_seq_length": limit,
                "pooling": model[1].pooling_mode, "normalize": len(model) > 2, "cases": []}
    for i, (text, length) in enumerate(cases):
        ids, hidden, pooled, embedding = run(model, text, length)
        assert len(ids) <= length
        hidden_file = f"case{i}.hidden.f32"
        hidden.astype("<f4").tofile(os.path.join(args.out, hidden_file))
        fixtures["cases"].append({"text": text, "cut": len(ids) == length, "ids": ids,
                                  "hidden": hidden_file, "pooled": pooled, "embedding": embedding})
        print(f"case {i}: {len(ids)} tokens")
    with open(os.path.join(args.out, "fixtures.json"), "w") as f:
        json.dump(fixtures, f, ensure_ascii=False)
    print(f"{fixtures['model']}: {len(cases)} cases, {fixtures['pooling']} pooling"
          f"{', normalized' if fixtures['normalize'] else ''} -> {args.out}")


if __name__ == "__main__":
    main()
