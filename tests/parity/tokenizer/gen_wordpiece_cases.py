"""Writes tests/fixtures/tokenizer/wordpiece_cases.json: Hugging Face reference outputs for WordPiece (BERT-family)
tokenizers, and for the BERT normalizer and pre-tokenizer over every code point.

Run: python -I tests/parity/tokenizer/gen_wordpiece_cases.py <model_dir>...
(directories of sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, nomic-ai/nomic-embed-text-v1.5; each
directory's name keys its cases, and the test finds a checkpoint's cases by the name of its directory)

The reference is the `tokenizers` library with tokenizer.json's truncation and padding turned off, as
sentence-transformers runs it; transformers is checked to agree. Per model: encode without and with special tokens
(`encode(text, add_special_tokens=True).ids`), and `decode` with and without the special tokens. Models whose outputs
are all equal to an earlier model's are written as {"same_as": <that model>}. Model-independent: FNV-1a 64 digests of
the BertNormalizer (several settings) and BertPreTokenizer outputs per block of 4096 code points, the text of a block
being its code points (surrogates left out) separated by spaces, for the normalizer, or each followed by "a", for the
pre-tokenizer, whose pieces are then joined with newlines.
"""

import json
import random
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer as RawTokenizer
from tokenizers import normalizers, pre_tokenizers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import TEXTS as QWEN35_TEXTS  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/wordpiece_cases.json"
BLOCK = 0x1000

CJK_EDGES = [0x4E00, 0x9FFF, 0x3400, 0x4DBF, 0x20000, 0x2A6DF, 0x2A700, 0x2B73F, 0x2B740, 0x2B81F, 0x2B920, 0x2CEAF,
             0xF900, 0xFAFF, 0x2F800, 0x2FA1F]
CJK_NEIGHBOURS = [0x4DFF, 0xA000, 0x33FF, 0x4DC0, 0x1FFFF, 0x2A6E0, 0x2A6FF, 0x2B820, 0x2B91F, 0x2CEB0, 0xF8FF,
                  0xFB00, 0x2F7FF, 0x2FA20, 0x30000, 0x31350, 0x3007, 0x2E80, 0x2F00, 0x3220, 0x3105]

TEXTS = {
    "empty": "",
    "whitespace_only": " \t\n   　 \r\n ",
    "plain_english": "The quick brown fox jumps over the lazy dog.",
    "sentence": "A man is eating food. Some people are playing the guitar in a park.",
    "mixed_case": "Hello WORLD, ThIs Is MiXeD cAsE. iPhone McDonald's NASA eBay",
    "punctuation_runs": "Wait... what?!? (yes) [no] {maybe} -- ok; \"quoted\" 'single' `tick` ~tilde~ $5 + 3 = 8 | a^b "
    "<tag> @user #hash 100% a_b a\\b ¡hola! ¿qué? «guillemets» „low“ — dash – en … ‰ †‡ ※ ¶§",
    "contractions": "I'm sure they've said it's fine; don't won't can't y'all o'clock rock'n'roll we're",
    "numbers": "In 2024, 3.14159 * 2 = 6.28318; 1,000,000 and 0x1F and 1e-9. ١٢٣ ² ½ Ⅻ 10th 1st ①②",
    "urls_emails": "https://www.example.com/path/to/page?query=1&x=y#frag, mailto:john.doe+tag@example.co.uk "
    "ftp://files.example.org/a_b-c.txt www.Test-Site.ORG/index.html",
    "accents_composed": "café naïve résumé ñandú Ångström Øre Æsir piñata façade crème brûlée",
    "accents_decomposed": "café naïve ñ Ångström ạ́̇ q̣̇ ę́",
    "german": "Straße STRASSE Maß Größe ẞ Fußball Äpfel ÖL über",
    "turkish": "İstanbul ISTANBUL ılık IĞDIR iı İi Diyarbakır",
    "greek": "ΟΔΥΣΣΕΥΣ Σίσυφος λόγος ΣΑΣ ς σ Σ ἀρχή Ἀθῆναι ΐ",
    "cyrillic": "Москва ЁЛКА Привет, мир! Ўзбек Їжак ѐ Ѝ",
    "arabic": "مرحبا بالعالم، هذه جملة اختبار. ٱلْعَرَبِيَّة؟",
    "hebrew": "שָׁלוֹם עוֹלָם",
    "hindi": "नमस्ते दुनिया, यह एक परीक्षण वाक्य है। क़िला",
    "thai": "ภาษาไทย สวัสดีครับ",
    "chinese": "我爱自然语言处理。北京是中国的首都，人口众多！",
    "japanese": "東京は日本の首都です。カタカナとひらがな、ｶﾀｶﾅ。",
    "korean": "한국어 문장입니다. 안녕하세요!",
    # The first and last code point of each block bert.rs pads, then neighbours outside them.
    "cjk_blocks": "".join(f"x{chr(cp)}" for cp in CJK_EDGES),
    "cjk_outside_blocks": "".join(f"x{chr(cp)}" for cp in CJK_NEIGHBOURS),
    "emoji": "I love it 😀👍🏽 family: 👨‍👩‍👧‍👦 flag: 🇯🇵 heart: ❤️ 🥰 🫠 ☃ ⚡",
    "zero_width": "zero​width‌join‍ner﻿bom soft­hyphen word⁠joiner ⁡ ᠎",
    "whitespace_kinds": "tab\there\nnew\r\nline nbsp em　ideographic ls ps\u0085nel"
    "\u000bvt\u000cff ogham nnbsp mmsp end",
    "control_chars": "a\x00b\x01c\x07d\x1be\x7ff\u0080g\u009fh‎i‪j⁦k\U000e0001lm\U000f0000n",
    "replacement_char": "bad�byte �� end�",
    "max_chars_boundary": "x" * 100 + " " + "y" * 101 + " " + "z" * 99,
    "long_word": "a" * 150 + " supercalifragilisticexpialidocious" * 4,
    "long_unicode_words": "é" * 101 + " " + "ü" * 99 + " " + "Å" * 100 + " " + "İ" * 51,
    "long_after_lowercase": "İ" * 60 + " " + "İ" * 40 + "x" * 30,
    "special_tokens_in_text": "[CLS] hello [SEP] world [MASK] [PAD] [UNK] [cls] [Sep] [ CLS ]",
    "special_adjacent": "[CLS][SEP]x[MASK]y[UNK][UNK]",
    "unknown_chars": "☃ ⚡ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ꙮ \U0001f9ff",
    "fullwidth": "ＡＢＣ１２３！？ ｈｅｌｌｏ",
    "math": "∀x∈ℝ: x² ≥ 0, ∑ᵢ aᵢ ≈ ∫ f(x) dx ≠ ∞",
    "code": "def f(x):\n    return x**2  # square\n\tprint(f'{x!r}')\n",
    "html_json": "<div class=\"a\">&nbsp;<b>bold</b></div> {\"key\": [1, 2, {\"n\": null}]}",
    "compatibility": "Kelvin K ohm Ω Å ﬁ ligature ǅ ẛ̣ Ⅰ µ ſ",
    "lowercase_special": "İ ẞ Ω K Ǆ ǅ ǈ ᾈ ᾼ Ⰰ Ꭰ ꭰ "
    "\U00010400 \U0001e900 ᲐᲑ",
    "newer_unicode": "Ᲊ Ⱟ Ꟁ Ɤ \U00010570 \U00010d50 \U0001e030 \U0001fae8 ࡰ "
    "\U00011f00 ⿼ ⹝ ⹄ ࢑ \U00016e40",
    "nfd_reordering": "ṩ̣ ệ ̈́ ཱི क़ 각 가",
}

PARSE_SPECIAL_OFF = ["special_tokens_in_text", "special_adjacent"]

WORDS = [
    "the", "running", "unbelievable", "tokenization", "embedding", "transformer", "playing", "happiness",
    "internationalization", "dog", "cats", "jumped", "quickly", "re-enter", "e-mail", "don't", "U.S.A.", "3.5",
    "naïve", "Zürich", "São", "Kraków", "東京", "москва", "λόγος", "##ing", "[UNK]", "[SEP]", "a", "I",
]

RANDOM_POOLS = (
    (30, lambda r: chr(r.randint(0x20, 0x7E))),
    (6, lambda r: r.choice(" \t\n\r 　 ​\u0085\x0b ﻿")),
    (5, lambda r: chr(r.choice([r.randint(0x00, 0x1F), r.randint(0x7F, 0x9F)]))),
    (10, lambda r: chr(r.randint(0xA0, 0x24F))),
    (6, lambda r: chr(r.randint(0x300, 0x36F))),
    (4, lambda r: chr(r.randint(0x370, 0x52F))),
    (12, lambda r: chr(r.choice([r.randint(0x530, 0xD7FF), r.randint(0xE000, 0xFFFD)]))),
    (8, lambda r: chr(r.randint(0x4E00, 0x9FFF))),
    (5, lambda r: chr(r.randint(0x10000, 0x2FFFF))),
    (4, lambda r: chr(r.randint(0x30000, 0x10FFFF))),
    (10, lambda r: r.choice("!?.,;:'\"()[]{}-_/@#$%^&*+=<>|~`¡¿«»—–…、。，！？")),
)

NORMALIZERS = {
    "bert_default": dict(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True),
    "lowercase_only": dict(clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=True),
    "strip_accents_only": dict(clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False),
    "clean_text_only": dict(clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=False),
    "chinese_only": dict(clean_text=False, handle_chinese_chars=True, strip_accents=False, lowercase=False),
}


def fnv1a64(text: str) -> str:
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def block_chars(block: int) -> list:
    return [chr(cp) for cp in range(block * BLOCK, (block + 1) * BLOCK) if not 0xD800 <= cp < 0xE000]


def component_digests() -> dict:
    blocks = range(0x110000 // BLOCK)
    normalizer_digests = {}
    for name, options in NORMALIZERS.items():
        normalizer = normalizers.BertNormalizer(**options)
        normalizer_digests[name] = {
            "config": {"type": "BertNormalizer", **options},
            "digests": [fnv1a64(normalizer.normalize_str(" ".join(block_chars(b)))) for b in blocks],
        }
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    pre_tokenizer_digests = []
    for b in blocks:
        pieces = pre_tokenizer.pre_tokenize_str("".join(c + "a" for c in block_chars(b)))
        pre_tokenizer_digests.append(fnv1a64("\n".join(piece for piece, _ in pieces)))
    return {"block": BLOCK, "normalizers": normalizer_digests, "pre_tokenizer": pre_tokenizer_digests}


def random_texts(rng: random.Random) -> dict:
    weights = [w for w, _ in RANDOM_POOLS]
    texts = {}
    for i in range(300):
        pools = rng.choices(RANDOM_POOLS, weights=weights, k=rng.randint(1, 24))
        texts[f"random_chars_{i}"] = "".join(draw(rng) for _, draw in pools)
    for i in range(100):
        words = []
        for _ in range(rng.randint(1, 10)):
            word = rng.choice(WORDS)
            word = rng.choice([word, word.upper(), word.title(), word + rng.choice(".,!?'")])
            words.append(word)
        texts[f"random_words_{i}"] = rng.choice([" ", "  ", "\t", "\n", ""]).join(words)
    return texts


def model_payload(model_dir: Path, texts: dict) -> dict:
    hf = AutoTokenizer.from_pretrained(str(model_dir))
    raw = RawTokenizer.from_file(str(model_dir / "tokenizer.json"))
    raw.no_truncation()
    raw.no_padding()

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        raw.encode_special_tokens = not parse_special
        ids = raw.encode(text, add_special_tokens=False).ids
        with_special = raw.encode(text, add_special_tokens=True).ids
        for add, expected in ((False, ids), (True, with_special)):
            got = hf(text, add_special_tokens=add, split_special_tokens=not parse_special)["input_ids"]
            assert got == expected, f"{model_dir.name}: transformers and tokenizers disagree on {name}"
        return {
            "name": name,
            "text": text,
            "parse_special": parse_special,
            "ids": ids,
            "ids_with_special": with_special,
            "decoded": raw.decode(with_special, skip_special_tokens=False),
            "decoded_skip_special": raw.decode(with_special, skip_special_tokens=True),
        }

    cases = [make_case(name, text, True) for name, text in texts.items()]
    cases += [make_case(name + "_no_special", texts[name], False) for name in PARSE_SPECIAL_OFF]
    assert any(len(c["ids_with_special"]) > 128 for c in cases), "no case is longer than MiniLM's truncation"

    def ids(*tokens: str) -> list:
        out = [raw.token_to_id(t) for t in tokens]
        assert None not in out, tokens
        return out

    decode_only = {
        "continuation_first": ids("##ing", "the", "##s", "dog"),
        "cleanup_per_token": ids("hello", ",", "world", ".", "isn", "'", "t", "it", "?", "i", "'", "m", "!", "do",
                                 "not", "'", "s", "'", "ve", "'", "re", "n", "'", "t"),
        "specials_only": ids("[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"),
        "specials_between": ids("[CLS]", "hello", "[MASK]", "##ing", "[SEP]", "[PAD]"),
        "prefix_alone": ids("#", "###", "##s", "###"),
    }
    decode_cases = [
        {
            "name": name,
            "ids": token_ids,
            "decoded": raw.decode(token_ids, skip_special_tokens=False),
            "decoded_skip_special": raw.decode(token_ids, skip_special_tokens=True),
        }
        for name, token_ids in decode_only.items()
    ]
    return {
        "vocab_size": raw.get_vocab_size(with_added_tokens=True),
        "bos_id": hf.bos_token_id,
        "eos_id": hf.eos_token_id,
        "special_ids": sorted(hf.all_special_ids),
        "encode_cases": cases,
        "decode_cases": decode_cases,
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    texts = {**TEXTS, **{f"qwen35_{k}": v for k, v in QWEN35_TEXTS.items()}, **random_texts(random.Random(30522))}
    models = {Path(model_dir).name: model_payload(Path(model_dir), texts) for model_dir in sys.argv[1:]}
    written = {}
    for name, payload in models.items():
        same = next((earlier for earlier in written if models[earlier] == payload), None)
        written[name] = {"same_as": same} if same else payload
        print(f"{name}: {len(payload['encode_cases'])} encode and {len(payload['decode_cases'])} decode cases"
              + (f", all the same as {same}'s" if same else ""))
    out = {"tokenizers": tokenizers.__version__, "components": component_digests(), "models": written}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
