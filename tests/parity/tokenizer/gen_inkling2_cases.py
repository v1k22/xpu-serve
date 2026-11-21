"""Writes tests/fixtures/tokenizer/inkling2_cases.json: Hugging Face reference outputs for Inkling 2's tokenizer
(byte-level BPE with o200k_base's Split regex and ignore_merges), and that regex's pieces of hand-picked and random
text.

Run: python -I tests/parity/tokenizer/gen_inkling2_cases.py <model_dir>   (a thinkingmachines/Inkling-2-Mini
checkpoint directory; its tokenizer files are enough)
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Regex, pre_tokenizers
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import BYTE_CHARS, SPLIT_POOL, SPLIT_TEXTS, TEXTS, random_texts  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/inkling2_cases.json"

INKLING2_TEXTS = {
    "cased_words": "camelCase PascalCase XMLHttpRequest iPhone eBay HTTPServer getHTTPResponseCode ABCdefGHI aB Ab AB "
    "ab",
    "contractions": "it's IT'S It'S they'RE WE'll you'Ve I'm X'D don't DON'T 'tis rock'n'roll o'clock l'homme d'être",
    "digit_groups": "1 12 123 1234 12345 1234567 3.14159 2026-11-21 0x1F4A9 ١٢٣٤٥ ²³ 1²3 v2.1.0",
    "slashes": "https://example.com/a/b/c?x=1 //comment\n/* block */ a/b ./x ../y /// \\\\ n/a 1/2 :/\n/",
    "scripts": "Ελληνικά ǅungla ǆungla ǄUNGLA ʰmodifier 中文字 क़िला Ünïcödé ÀÉÎõü ß ẞ İstanbul ıi naïve",
    "chat_turns": "<|message_system|><|content_text|>You are helpful.<|end_message|><|message_user|><|content_text|>"
    "Hi there!<|end_message|><|message_model|><|content_thinking|>Short thought.<|end_message|><|message_model|>"
    "<|content_text|>Hello!<|end_message|><|content_model_end_sampling|>",
    "specials_adjacent": "<|endoftext|><|unused|>x<|message_tool|><|content_invoke_tool_text|>a<|unused_200054|>b",
    "specials_partial": "<|message_user <|end_message| <|content_text <content_thinking|> |end_message|>",
}

PARSE_SPECIAL_OFF = ["chat_turns", "specials_adjacent"]

# What the pattern tells apart beyond the Qwen and Llama 3 ones: letters by case, slashes after punctuation.
O200K_SPLIT_TEXTS = [
    "camelCase", "XMLHttp", "ABC", "ABc", "aBC", "abcDEF", "中A", "A中", "中a", "a中B", "ǅa", "Aǅ", "ʰA", "Aʰ", "aʰA",
    "É́x", "éÉ", " ́A", "́a", "A's", "a'S", "A'LL", "'s", "x 's", "a'sb", "A'", "a'x", "IT'S a", "'A", "-a", "-A",
    "_aB", "a_b", "1a", "a1", "A1b", "!!/", "!/\n/", " !//\n", "a/b", "/a", "//", " /", "/\n/", "a /b", ":/\r\n/x",
    "ÁB", "́", "́́a", "á́", "1́a", " ́", "\ńa",
]
O200K_POOL = SPLIT_POOL + list("ABCÉǅʰ中/") * 2 + list("abé") + ["'", "'", "/"]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_dir = sys.argv[1]
    hf = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    raw = RawTokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    texts = {**{k: v for k, v in TEXTS.items() if k != "config_added_tokens"}, **INKLING2_TEXTS}

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

    end_message = hf.convert_tokens_to_ids("<|end_message|>")
    decode_only = {
        "incomplete_cjk": byte_ids("你".encode()[:2]),
        "complete_cjk_bytes": byte_ids("你好".encode()),
        "incomplete_emoji_then_ascii": byte_ids("😀".encode()[:3] + b"a"),
        "lone_continuation": byte_ids(b"a\x80b"),
        "bytes_around_special": byte_ids("é".encode()[:1]) + [end_message] + byte_ids("é".encode()[1:]),
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

    pattern = json.loads(raw.to_str())["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
    split_texts = SPLIT_TEXTS + O200K_SPLIT_TEXTS + random_texts(random.Random(200), O200K_POOL, 600, 16)
    split_cases = {"o200k": {"pattern": pattern,
                             "cases": [[t, [p for p, _ in split.pre_tokenize_str(t)]] for t in split_texts]}}

    payload = {
        "model": "thinkingmachines/Inkling-2-Mini",
        "vocab_size": len(hf),
        "bos_id": hf.bos_token_id,
        "eos_id": hf.eos_token_id,
        "special_ids": sorted(hf.all_special_ids),
        "encode_cases": cases,
        "decode_cases": decode_cases,
        "split_cases": split_cases,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(cases)} encode, {len(decode_cases)} decode and {len(split_texts)} split cases to {OUT} "
          f"({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
