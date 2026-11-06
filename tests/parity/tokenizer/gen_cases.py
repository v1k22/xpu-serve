"""Writes tests/fixtures/tokenizer/gemma4_cases.json: Hugging Face reference ids and decodes for the C++ tokenizer.

Run: /root/workspace/xs-ref/venv/bin/python tests/parity/tokenizer/gen_cases.py [model_dir]
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/gemma4_cases.json"
DEFAULT_MODEL = "/root/workspace/models/gemma-4-E2B-it"

TEXTS = {
    "empty": "",
    "ascii_prose": "The quick brown fox jumps over the lazy dog. It wasn't me, said the fox; \"really?\"",
    "numbers": "In 2024, 3.14159 * 2 = 6.28318; 1,000,000 and 0x1F and 1e-9.",
    "code_python": "def f(x):\n    if x > 0:\n        return x * 2\n\treturn -x  # tab\n",
    "code_crlf": "int main() {\r\n    printf(\"hi\\n\");\r\n    return 0;\r\n}\r\n",
    "whitespace_run": "a" + " " * 40 + "b" + "\t" * 17 + "c",
    "spaces_only": "     ",
    "newlines": "one\n\ntwo\n\n\n\nthree\n" + "\n" * 35 + "end",
    "leading_trailing": "   leading and trailing   ",
    "single_space": " ",
    "cjk": "我爱自然语言处理。東京は日本の首都です。한국어 문장입니다.",
    "hindi": "नमस्ते दुनिया, यह एक परीक्षण वाक्य है।",
    "arabic": "مرحبا بالعالم، هذه جملة اختبار.",
    "emoji": "I love it 😀👍🏽 family: 👨‍👩‍👧‍👦 flag: 🇯🇵 heart: ❤️",
    "emoji_zwj_skin": "👩🏿‍💻 🧑🏻‍🤝‍🧑🏾 🏳️‍🌈",
    "combining": "é ä ñ Z͓͔͑͒à́̂̃",
    "byte_fallback_rare": "͸  \U0010fffd \U00010000 \U0001d11e ⿿ \x00\x01\x7f",
    "mixed_scripts": "English, Ελληνικά, Русский, עברית, ไทย, Tiếng Việt",
    "chat_markers": "<|turn>user\nhi<turn|>\n<|turn>model\n",
    "bos_marker": "<bos>",
    "bos_in_text": "say <bos> and <eos> please",
    "partial_markers": "<|tur <turn <turn| |turn> <|channel <|\"| <|tool_call",
    "adjacent_markers": "<|turn><turn|><|channel><channel|><|\"|>x<|\"|>",
    "tool_call": "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
    "marker_like_text": "<pad> <mask> <unk> <unused0> <start_of_turn> <|image|> <|video|>",
    "spaces_around_markers": "  <|turn>  user  <turn|>  ",
    "html": "<div class=\"a\">&nbsp;<b>bold</b></div>",
    "url_json": "https://example.com/path?q=1&x=y {\"key\": [1, 2, {\"n\": null}]}",
    "math_symbols": "∀x∈ℝ: x² ≥ 0, ∑ᵢ aᵢ ≈ ∫ f(x) dx ≠ ∞",
    "fullwidth": "ＡＢＣ１２３！？ ｶﾀｶﾅ",
    "repeated_chars": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!!!!!!!!!!!!!!!!!!!!--------------",
    "nbsp_and_others": "a b c　d​e﻿",
}

PARSE_SPECIAL_OFF = ["chat_markers", "bos_marker", "bos_in_text", "adjacent_markers", "tool_call", "marker_like_text"]

DECODE_ONLY = {
    "invalid_byte_run": [238 + 0xE3, 238 + 0x81],
    "lone_continuation_byte": [236746, 238 + 0x80, 236746],
    "valid_then_invalid_run": [238 + 0xE3, 238 + 0x81, 238 + 0x82, 238 + 0xFF],
    "bytes_around_special": [238 + 0xE3, 105, 238 + 0x81, 238 + 0x82],
    "specials_only": [2, 105, 106, 1, 0],
}


def long_text(rng: random.Random, target_chars: int) -> str:
    pieces = [v for k, v in TEXTS.items() if k not in ("empty",)]
    out = []
    size = 0
    while size < target_chars:
        piece = rng.choice(pieces)
        sep = rng.choice([" ", "\n", "", "  ", "\n\n", "\t"])
        out.append(piece + sep)
        size += len(piece) + len(sep)
    return "".join(out)


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    hf = AutoTokenizer.from_pretrained(model_dir)
    raw = RawTokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    texts = dict(TEXTS)
    rng = random.Random(1234)
    texts["long_mixed"] = long_text(rng, 50_000)
    texts["long_mixed_markers"] = "<|turn>user\n" + long_text(rng, 4_000) + "<turn|>\n<|turn>model\n"

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        ids = hf(text, add_special_tokens=False, split_special_tokens=not parse_special)["input_ids"]
        raw.encode_special_tokens = not parse_special
        raw_ids = raw.encode(text, add_special_tokens=False).ids
        assert ids == raw_ids, f"transformers and tokenizers disagree on {name}"
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
    decode_cases = [
        {
            "name": name,
            "ids": ids,
            "decoded": hf.decode(ids, skip_special_tokens=False),
            "decoded_skip_special": hf.decode(ids, skip_special_tokens=True),
        }
        for name, ids in DECODE_ONLY.items()
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "google/gemma-4-E2B-it",
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
