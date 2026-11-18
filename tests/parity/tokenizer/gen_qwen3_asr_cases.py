"""Writes tests/fixtures/tokenizer/qwen3_asr_cases.json: Hugging Face reference outputs for a tokenizer shipped as slow
files (vocab.json, merges.txt and tokenizer_config.json, no tokenizer.json): Qwen/Qwen3-ASR-1.7B's Qwen2Tokenizer,
which AutoTokenizer builds from them on load. The ids are also checked against the tokenizer transformers'
Qwen2Converter builds from the same files.

Run: python -I tests/parity/tokenizer/gen_qwen3_asr_cases.py <model_dir>   (needs only the files above)
"""

import json
import sys
import types
from pathlib import Path

from tokenizers import AddedToken
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import Qwen2Converter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import BYTE_CHARS, TEXTS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/qwen3_asr_cases.json"

ASR_TEXTS = {
    "asr_prompt": "<|im_start|>system\nTranscribe.<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>"
    "<|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n",
    "asr_output": "language English<asr_text>The quick brown fox, it's 3:45 p.m.!<|im_end|>",
    "asr_output_chinese": "language Chinese<asr_text>今天天气很好，我们去公园散步吧。<|im_end|>",
    "blank_and_tts": "<blank1><blank27> <non_speech>x<tts_text_eod><tts_text_bos_single>y<tts_pad>",
    "specials_partial": "<|audio_start <|audio_pad| <asr_text <blank28> <blank0> <|im_end| <tts_pad",
    "letters_and_marks": "क़िला नमस्ते é ñ ộ ̃x a̐b",
    "cantonese_japanese_korean": "佢哋今日去咗邊度？ 東京駅で会いましょう。 서울에서 만나요!",
}

PARSE_SPECIAL_OFF = ["chat_template", "special_adjacent", "tool_call", "config_added_tokens", "asr_prompt",
                     "every_added_token"]


def converter_tokenizer(model_dir: Path, hf):
    """The tokenizer convert_slow_tokenizer.Qwen2Converter builds from the files, with the added tokens in id order."""
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    merges = []
    for line in (model_dir / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line.startswith("#version"):
            merges.append(tuple(line.split(" ")))
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    original = types.SimpleNamespace(add_prefix_space=config.get("add_prefix_space") or False)
    raw = Qwen2Converter(original).converted(vocab=vocab, merges=merges)
    for token_id, token in sorted(hf.added_tokens_decoder.items()):
        added = AddedToken(token.content, single_word=token.single_word, lstrip=token.lstrip, rstrip=token.rstrip,
                           normalized=token.normalized, special=token.special)
        assert raw.add_tokens([added]) == 1 and raw.token_to_id(token.content) == token_id, token.content
    return raw


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = Path(sys.argv[1])
    assert not (model_dir / "tokenizer.json").exists(), "the model directory must hold the slow files only"
    hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)  # the tokenizer needs none of its code
    raw = converter_tokenizer(model_dir, hf)
    added = sorted(hf.added_tokens_decoder.items())
    contents = [token.content for _, token in added]
    texts = {**TEXTS, **ASR_TEXTS, "every_added_token": " ".join(contents) + "".join(contents) + "x"}

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        ids = hf(text, add_special_tokens=False, split_special_tokens=not parse_special)["input_ids"]
        raw.encode_special_tokens = not parse_special
        converted = raw.encode(text, add_special_tokens=False).ids
        assert ids == converted, f"transformers and Qwen2Converter disagree on {name}"
        return {
            "name": name,
            "text": text,
            "parse_special": parse_special,
            "ids": ids,
            "decoded": hf.decode(ids, skip_special_tokens=False),
            "decoded_skip_special": hf.decode(ids, skip_special_tokens=True),
        }

    cases = [make_case(name, text, True) for name, text in texts.items()]
    cases += [make_case(name + "_no_special", texts[name], False) for name in PARSE_SPECIAL_OFF]

    def byte_ids(data: bytes) -> list:
        ids = [hf.convert_tokens_to_ids(BYTE_CHARS[b]) for b in data]
        assert all(isinstance(i, int) and i < len(hf) for i in ids)
        return ids

    im_start = hf.convert_tokens_to_ids("<|im_start|>")
    decode_only = {
        "incomplete_cjk": byte_ids("你".encode()[:2]),
        "complete_cjk_bytes": byte_ids("你好".encode()),
        "incomplete_emoji_then_ascii": byte_ids("😀".encode()[:3] + b"a"),
        "lone_continuation": byte_ids(b"a\x80b"),
        "bytes_around_special": byte_ids("é".encode()[:1]) + [im_start] + byte_ids("é".encode()[1:]),
        "invalid_lead": byte_ids(b"\xff\xfe x \xc0\xaf \xed\xa0\x80"),
        "every_added_token": [token_id for token_id, _ in added] + byte_ids(b" ok"),
    }
    decode_cases = [
        {
            "name": name,
            "ids": ids,
            "decoded": hf.decode(ids, skip_special_tokens=False),
            "decoded_skip_special": hf.decode(ids, skip_special_tokens=True),
        }
        for name, ids in decode_only.items()
    ]

    payload = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "vocab_size": len(hf),
        "bos_id": hf.bos_token_id,
        "eos_id": hf.eos_token_id,
        "special_ids": sorted(hf.all_special_ids),
        "added_tokens": [[token_id, token.content, token.special] for token_id, token in added],
        "encode_cases": cases,
        "decode_cases": decode_cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} encode and {len(decode_cases)} decode cases to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
