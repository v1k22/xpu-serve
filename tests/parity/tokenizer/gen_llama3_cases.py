"""Writes tests/fixtures/tokenizer/llama3_cases.json: Hugging Face reference outputs for Llama 3's tokenizer (byte-level
BPE with the Llama 3 Split regex and ignore_merges; a post-processor Sequence that puts <|begin_of_text|> first).

Run: python -I tests/parity/tokenizer/gen_llama3_cases.py <model_dir>   (a Llama 3.x checkpoint directory, e.g.
unsloth/Llama-3.2-1B-Instruct: the Llama 3.1, 3.2 and 3.3 releases share the tokenizer)
"""

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import BYTE_CHARS, TEXTS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/llama3_cases.json"

LLAMA3_TEXTS = {
    "digit_groups": "1 12 123 1234 12345 1234567 3.14159 2026-11-07 0x1F4A9 ١٢٣٤٥ ²³ 1²3",
    "whole_words": "the The THE  the\nthe internationalization Internationalization antidisestablishmentarianism",
    "chat_turns": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi there!<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "tool_call_json": "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\", \"unit\": \"celsius\"}}",
    "specials_adjacent": "<|eot_id|><|eom_id|><|end_of_text|>x<|python_tag|>a<|finetune_right_pad_id|>b"
    "<|reserved_special_token_0|><|reserved_special_token_247|>",
    "specials_partial": "<|eot_id <|start_header_id| <|python_tag <|begin_of_text|",
}

PARSE_SPECIAL_OFF = ["chat_turns", "specials_adjacent"]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = sys.argv[1]
    hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    raw = RawTokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    texts = {**{k: v for k, v in TEXTS.items() if k != "config_added_tokens"}, **LLAMA3_TEXTS}

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        ids = hf(text, add_special_tokens=False, split_special_tokens=not parse_special)["input_ids"]
        raw.encode_special_tokens = not parse_special
        assert ids == raw.encode(text, add_special_tokens=False).ids, f"transformers and tokenizers disagree on {name}"
        with_special = raw.encode(text, add_special_tokens=True).ids
        return {
            "name": name,
            "text": text,
            "parse_special": parse_special,
            "ids": ids,
            "ids_with_special": with_special,
            "decoded": hf.decode(with_special, skip_special_tokens=False),
            "decoded_skip_special": hf.decode(with_special, skip_special_tokens=True),
        }

    cases = [make_case(name, text, True) for name, text in texts.items()]
    cases += [make_case(name + "_no_special", texts[name], False) for name in PARSE_SPECIAL_OFF]

    def byte_ids(data: bytes) -> list:
        ids = [hf.convert_tokens_to_ids(BYTE_CHARS[b]) for b in data]
        assert all(isinstance(i, int) and i < len(hf) for i in ids)
        return ids

    eot = hf.convert_tokens_to_ids("<|eot_id|>")
    decode_only = {
        "incomplete_cjk": byte_ids("你".encode()[:2]),
        "complete_cjk_bytes": byte_ids("你好".encode()),
        "incomplete_emoji_then_ascii": byte_ids("😀".encode()[:3] + b"a"),
        "lone_continuation": byte_ids(b"a\x80b"),
        "bytes_around_special": byte_ids("é".encode()[:1]) + [eot] + byte_ids("é".encode()[1:]),
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
        "model": "unsloth/Llama-3.2-1B-Instruct",
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
