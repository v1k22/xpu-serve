"""Writes tests/fixtures/tokenizer/lfm3_cases.json: Hugging Face reference outputs for LFM3's tokenizer (byte-level BPE
with the Llama 3 Split regex and ignore_merges).

Run: python -I tests/parity/tokenizer/gen_lfm3_cases.py <model_dir>   (a LiquidAI/LFM3-1.7B checkpoint directory)
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import BYTE_CHARS, TEXTS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/lfm3_cases.json"

LFM3_TEXTS = {
    "digit_groups": "1 12 123 1234 12345 1234567 3.14159 2026-11-07 0x1F4A9 ١٢٣٤٥ ²³ 1²3",
    "whole_words": "the The THE  the\nthe internationalization Internationalization antidisestablishmentarianism",
    "chat_turns": "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi there!"
    "<|im_end|>\n<|im_start|>assistant\n<think>Short thought.</think>Hello!<|im_end|>\n",
    "tool_call_xml": "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
    "</tool_call>",
    "tool_list": "<tools>\n{\"name\": \"get_weather\"}\n</tools><|tool_list_start|>[]<|tool_list_end|>"
    "<|tool_call_start|>x<|tool_call_end|>",
    "specials_adjacent": "<|im_end|><|im_start|><|endoftext|>x<|pad|><|fim_pre|>a<|fim_suf|>b<|fim_mid|>"
    "<|vision_start|><|image|><|vision_end|><image><|img_row_1_col_1|>",
    "specials_partial": "<|im_start <|im_end| <think </think <tool_call <|startoftext",
}

PARSE_SPECIAL_OFF = ["chat_turns", "tool_call_xml", "specials_adjacent"]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = sys.argv[1]
    hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)  # the tokenizer needs none of its code
    raw = RawTokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    texts = {**{k: v for k, v in TEXTS.items() if k != "config_added_tokens"}, **LFM3_TEXTS}

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        ids = hf(text, add_special_tokens=False, split_special_tokens=not parse_special)["input_ids"]
        raw.encode_special_tokens = not parse_special
        assert ids == raw.encode(text, add_special_tokens=False).ids, f"transformers and tokenizers disagree on {name}"
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
        "model": "LiquidAI/LFM3-1.7B",
        "vocab_size": len(hf),
        "bos_id": hf.bos_token_id,
        "eos_id": hf.eos_token_id,
        "special_ids": sorted(hf.all_special_ids),
        "encode_cases": cases,
        "decode_cases": decode_cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} encode and {len(decode_cases)} decode cases to {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
