"""Writes tests/fixtures/tokenizer/qwen35_cases.json: Hugging Face reference outputs for the byte-level BPE tokenizer.

Run: /root/workspace/xs-ref/venv/bin/python -I tests/parity/tokenizer/gen_qwen35_cases.py [model_dir]

Besides whole-tokenizer cases for Qwen/Qwen3.5-4B (encode ids, decodes), it records outputs that need no model: the
Split pre-tokenizer on the Qwen3.5, Qwen2 and Llama 3 regexes, and the NFC normalizer, on hand-picked and random text.
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Regex, normalizers, pre_tokenizers
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/qwen35_cases.json"
DEFAULT_MODEL = "/root/workspace/models/Qwen3.5-4B"

CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
SPACES = r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
PATTERNS = {
    "qwen35": CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+"
    r"|\s+(?!\S)|\s+",
    "qwen2": CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|" + SPACES,
    "llama3": CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|" + SPACES,
}

LONG_SAMPLE = """# Release notes

The tokenizer now handles byte-level BPE. It's been tested on English, code, and many scripts; we'd like more
feedback, so please don't hesitate to file issues. They'll be triaged weekly.

## Usage

```python
def encode(text: str) -> list[int]:
    \"\"\"Return token ids.\"\"\"
    if not text:
        return []
\tfor i in range(10):
\t\tprint(f"{i:>4} -> {i ** 2}")
    return tokenizer.encode(text)
```

Numbers like 3.14159, 1,000,000 and 2026-11-07T12:34:56Z split into single digits. Prices: $19.99, €5, ¥1200.

中文段落：自然语言处理是人工智能的一个重要方向。日本語の文章も含まれています。한국어 문장도 있습니다.
हिन्दी में एक वाक्य: यह परीक्षण के लिए है। العربية: هذه جملة للاختبار. Ελληνικά και Русский текст.

Emoji: 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 ❤️‍🔥 — and symbols ∑∫√∞ ≠ ≤ ≥.\r\n\r\nTrailing spaces here   \n\n\n   and tabs\t\t\n"""

TEXTS = {
    "empty": "",
    "ascii_prose": "The quick brown fox jumps over the lazy dog. It wasn't me, said the fox; \"really?\"",
    "contractions_mixed_case": "I'M sure they'Ve said it's fine; YOU'LL see, we'D go, she'S here, I'm, 'RE 'll 'T",
    "contraction_edges": "'s's 'sss ''s ' s 'S1 'Re 'LL 'lL x'ſ ſ 'veX",
    "numbers": "In 2024, 3.14159 * 2 = 6.28318; 1,000,000 and 0x1F and 1e-9. ١٢٣ ² ½ Ⅻ",
    "code_python": "def f(x):\n    if x > 0:\n        return x * 2\n\treturn -x  # tab\n\n\n    \n",
    "code_crlf": "int main() {\r\n    printf(\"hi\\n\");\r\n    return 0;\r\n}\r\n",
    "code_mixed_indent": "\t\tfoo();\n  \t  bar();\n\t \n    baz()   \n",
    "whitespace_run": "a" + " " * 40 + "b" + "\t" * 17 + "c" + " \r\n \r\n" + "  \n\n  d",
    "spaces_only": "     ",
    "newlines": "one\n\ntwo\r\n\r\nthree\n" + "\n" * 35 + "end\n",
    "trailing_spaces": "trailing   ",
    "leading_trailing": "   leading and trailing   ",
    "single_space": " ",
    "unicode_spaces": "a b　c d\u0085e f ​g  h",
    "cjk": "我爱自然语言处理。東京は日本の首都です。한국어 문장입니다.",
    "hindi": "नमस्ते दुनिया, यह एक परीक्षण वाक्य है। क़िला ज़रूर",
    "arabic": "مرحبا بالعالم، هذه جملة اختبار. ٱلْعَرَبِيَّة",
    "emoji": "I love it 😀👍🏽 family: 👨‍👩‍👧‍👦 flag: 🇯🇵 heart: ❤️",
    "emoji_zwj_skin": "👩🏿‍💻 🧑🏻‍🤝‍🧑🏾 🏳️‍🌈",
    "nfc_composed": "é ä ñ Å Ω",
    "nfc_decomposed": "é ä ñ Å Å Ω ộ ộ",
    "nfc_hangul_jamo": "한글 가 각",
    "nfc_marks_order": "ạ́̇ q̣̇ क़ख़ ཱི ́᷶",
    "mixed_scripts": "English, Ελληνικά, Русский, עברית, ไทย, Tiếng Việt",
    "chat_template": "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi there!<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n",
    "think_block": "<think>\nLet me think... 2+2=4.\n</think>\n\nThe answer is 4.",
    "tool_call": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "tool_response": "<tool_response>\n{\"temp\": 21}\n</tool_response>",
    "special_adjacent": "<|im_end|><|im_start|><|endoftext|>x<|vision_start|><|image_pad|><|vision_end|>",
    "special_partial": "<|im_start <|im_end| <think <|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "spaces_around_specials": "  <|im_start|>  user  <|im_end|>  ",
    "html": "<div class=\"a\">&nbsp;<b>bold</b></div>",
    "url_json": "https://example.com/path?q=1&x=y {\"key\": [1, 2, {\"n\": null}]}",
    "math_symbols": "∀x∈ℝ: x² ≥ 0, ∑ᵢ aᵢ ≈ ∫ f(x) dx ≠ ∞",
    "fullwidth": "ＡＢＣ１２３！？ ｶﾀｶﾅ",
    "repeated_chars": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!!!!!!!!!!!!!!!!!!!!--------------",
    "rare_chars": "\x00\x01\x7f \U0010fffd \U0001d11e ⿿ ﻿ �",
    "long_multi_paragraph": LONG_SAMPLE,
    "config_added_tokens": "a<tts_pad>b<|audio_start|>c<|audio_pad|><tts_text_bos_single>",
}

# Tokens that transformers takes from tokenizer_config.json added_tokens_decoder; tokenizer.json alone lacks them.
CONFIG_ONLY_TOKENS = {"config_added_tokens", "config_added_tokens_no_special"}

PARSE_SPECIAL_OFF = ["chat_template", "special_adjacent", "tool_call", "config_added_tokens"]

SPLIT_TEXTS = [
    "Hello world", "it's I'M they'Ve WE'LL", "x'ſ 'ſ", "  \n  x", "a  \n\n  b", "a \t b", "   ", "a   ", "\r\n\r\n",
    " !!\n\n", "!!\r\n x", "123456 ab12", "क़िला", "é́x", " ́x", "́", "\ńa", "😀👍🏽",
    "a b", "x y", "'ll'LL'lL'Ll", "''s", "foo.bar(baz)", " ?!", "\t\tfoo", "a 　b", "1²½Ⅻ",
]

SPLIT_POOL = (
    list("aZzé'sStTrReEvVlLmMdD1٣²Ⅻ !.-_😀中ß") + ["́", "ि", "क", "\t", "\n", "\r", " ", "　",
                                                  " ", "\u0085", "ſ", "‍", " "]
)

NFC_TEXTS = [
    "", "abc", "é", "Ẹ́", "ạ́̇", "각", "각", "क़",
    "ÅΩ", "̈́", "ཱི", "ୋ", "́᷶", "഻̖", "́abc", "x́ͅ",
    "ᾄ", "Ḉ", "q̣̇",
]

NFC_POOL = (
    list("eEaoAxu ") + ["́", "̀", "̈", "̣", "̧", "̛", "ͅ", "α", "ᄀ",
                        "ᅡ", "ᆨ", "가", "क", "़", "क़", "Å", "Ω", "̈́",
                        "ἀ", "ཱི", "େ", "ା", "ୗ", "᷶", "഻", "é", "ḋ",
                        "̇", "̖", "ְ", "֑"]
)


def random_texts(rng: random.Random, pool: list, count: int, max_len: int) -> list:
    return ["".join(rng.choice(pool) for _ in range(rng.randint(1, max_len))) for _ in range(count)]


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    hf = AutoTokenizer.from_pretrained(model_dir)
    raw = RawTokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    rng = random.Random(35)

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        ids = hf(text, add_special_tokens=False, split_special_tokens=not parse_special)["input_ids"]
        raw.encode_special_tokens = not parse_special
        if name not in CONFIG_ONLY_TOKENS:
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

    cases = [make_case(name, text, True) for name, text in TEXTS.items()]
    cases += [make_case(name + "_no_special", TEXTS[name], False) for name in PARSE_SPECIAL_OFF]

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
        "config_added_tokens": [87, 248070, 248072, 248076, 87],
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

    split_cases = {}
    for name, pattern in PATTERNS.items():
        split = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
        count = 200 if name == "qwen35" else 60
        texts = SPLIT_TEXTS + random_texts(rng, SPLIT_POOL, count, 16)
        split_cases[name] = {
            "pattern": pattern,
            "cases": [[t, [p for p, _ in split.pre_tokenize_str(t)]] for t in texts],
        }

    nfc = normalizers.NFC()
    nfc_cases = [[t, nfc.normalize_str(t)] for t in NFC_TEXTS + random_texts(rng, NFC_POOL, 200, 10)]

    payload = {
        "model": "Qwen/Qwen3.5-4B",
        "vocab_size": len(hf),
        "bos_id": hf.bos_token_id,
        "eos_id": hf.eos_token_id,
        "special_ids": sorted(hf.all_special_ids),
        "encode_cases": cases,
        "decode_cases": decode_cases,
        "split_cases": split_cases,
        "nfc_cases": nfc_cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} encode, {len(decode_cases)} decode, "
          f"{sum(len(v['cases']) for v in split_cases.values())} split and {len(nfc_cases)} NFC cases to {OUT} "
          f"({OUT.stat().st_size} bytes)")


def bytes_to_unicode() -> list:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = {b: chr(b) for b in bs}
    n = 0
    for b in range(256):
        if b not in chars:
            chars[b] = chr(256 + n)
            n += 1
    return [chars[b] for b in range(256)]


BYTE_CHARS = bytes_to_unicode()

if __name__ == "__main__":
    main()
