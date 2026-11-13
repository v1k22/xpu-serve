"""Writes tests/fixtures/tokenizer/laya_julia_cases.json: Hugging Face reference outputs for the tokenizers of the Laya
and Julia-1 decision models, and for the ByteLevel (GPT-2 regex) and Metaspace pre-tokenizers on their own.

Run: python -I tests/parity/tokenizer/gen_laya_julia_cases.py <laya_dir> <julia_dir>
(checkpoint directories of convaiinnovations/laya and SupersonicLabs/Julia-1; only their tokenizer folders are read)

Laya's tokenizer/ and typed-decisions/tokenizer/ are ModernBERT's: NFC, ByteLevel with the GPT-2 regex, byte-level
BPE, normalized added tokens (runs of spaces), [MASK] with lstrip, [CLS] ... [SEP]. Its multilingual/tokenizer/ and
Julia-1's tokenizer/ are mmBERT's, Gemma's vocabulary: Replace(" " -> "▁"), Metaspace (prepend_scheme always), BPE
with byte fallback, <mask> with lstrip, <bos> ... <eos>. The test finds a tokenizer by its checkpoint's environment
variable and its folder (the keys below). Per tokenizer, as gen_wordpiece_cases.py writes them: encode without and
with special tokens, decode with and without them; a tokenizer whose outputs all equal an earlier one's is written as
{"same_as": <that key>}. Model-independent: the GPT-2 regex as a Split and inside ByteLevel (add_prefix_space on and
off), on hand-picked and random text and, by FNV-1a 64 digest per block of 4096, over every code point (each code
point followed by "a", then by "1 " and again: the pieces written as <UTF-8 length>:<piece>); Metaspace (prepend_scheme
always and never, split on and off, two replacement characters).
"""

import json
import random
import sys
from pathlib import Path

import tokenizers
from tokenizers import Regex, pre_tokenizers
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_qwen35_cases import BYTE_CHARS, SPLIT_POOL, SPLIT_TEXTS  # noqa: E402
from gen_wordpiece_cases import TEXTS as WORDPIECE_TEXTS  # noqa: E402
from gen_wordpiece_cases import block_chars, fnv1a64, random_texts  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tests/fixtures/tokenizer/laya_julia_cases.json"
BLOCK = 0x1000
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

FOLDERS = {"laya": ["tokenizer", "typed-decisions/tokenizer", "multilingual/tokenizer"], "julia": ["tokenizer"]}

TEXTS = {
    "mask_tokens": "[MASK] x [MASK]y  [MASK]\t[MASK] <mask> z <mask>, <mask><mask>  <mask>",
    "special_tokens_modernbert": "[CLS] a [SEP] b [PAD] [UNK] <|endoftext|> <|padding|> [unused0] [unused82] "
    "|||IP_ADDRESS||| x|||EMAIL_ADDRESS|||y |||PHONE_NUMBER|||",
    "special_tokens_gemma": "<bos>a<eos> <pad> <unk> <start_of_turn>user\nhi<end_of_turn>\n<2mass> [@BOS@] "
    "<unused0> <unused99> <table><tr><td>x</td></tr></table> <s>y</s> [toxicity=0]",
    "space_runs": "a  b   c    d" + " " * 30 + "e" + " " * 24 + "f" + " " * 25 + "g ",
    "newline_and_tab_runs": "a\n\nb\n\n\n\nc" + "\n" * 40 + "d\t\te" + "\t" * 33 + "f\r\n\r\n",
    "replacement_char_in_text": "▁ ▁▁x ▁▁▁ y▁z ▁",
    "leading_spaces": "   leading",
    "decision_head": "choice question: Which action should the agent take next?",
    "decision_option": " escalate to a human reviewer",
    "decision_state": json.dumps({"ticket": "Refund for order #4521", "amount": 129.99, "notes": ["late", "damaged"]},
                                 ensure_ascii=False),
    "gpt2_contractions": "it's I'M they'Ve WE'LL 's't're've'm'll'd 'S 'T x'ſ",
}

PARSE_SPECIAL_OFF = ["mask_tokens", "special_tokens_modernbert", "special_tokens_gemma"]

GPT2_TEXTS = SPLIT_TEXTS + [
    "it's", "IT'S", "'s'S", " 's", "  x", "x  ", " 1 22 333", "a1b2", " !?", "!! !!", "\t\n x", "a b",
    "ab12cd", " é ́", "😀 😀", "¿qué?", " 　x",
]

METASPACE_TEXTS = ["", "Hello world", " Hello", "  two  spaces ", "▁already", "a▁b ▁c", "▁", " ", "x", "日本 語"]


def metaspace_cases(rng: random.Random) -> list:
    texts = METASPACE_TEXTS + ["".join(rng.choice("ab ▁_ ") for _ in range(rng.randint(1, 10))) for _ in range(40)]
    groups = []
    for replacement in ("▁", "_"):
        for scheme in ("always", "never"):
            for split in (True, False):
                pre = pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=scheme, split=split)
                config = {"type": "Metaspace", "replacement": replacement, "prepend_scheme": scheme, "split": split}
                cases = [[t, [p for p, _ in pre.pre_tokenize_str(t)]] for t in texts]
                groups.append({"config": config, "cases": cases})
    return groups


def byte_level_cases(rng: random.Random) -> list:
    texts = GPT2_TEXTS + ["".join(rng.choice(SPLIT_POOL) for _ in range(rng.randint(1, 16))) for _ in range(200)]
    split = pre_tokenizers.Split(Regex(GPT2), "isolated")
    groups = [{
        "config": {"type": "Split", "pattern": {"Regex": GPT2}, "behavior": "Isolated", "invert": False},
        "byte_level": False,
        "cases": [[t, [p for p, _ in split.pre_tokenize_str(t)]] for t in texts],
    }]
    for add_prefix_space, use_regex in ((False, True), (True, True), (True, False)):
        pre = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space, use_regex=use_regex)
        groups.append({
            "config": {"type": "ByteLevel", "add_prefix_space": add_prefix_space, "trim_offsets": True,
                       "use_regex": use_regex},
            "byte_level": True,
            "cases": [[t, [p for p, _ in pre.pre_tokenize_str(t)]] for t in texts],
        })
    return groups


def gpt2_digests() -> list:
    split = pre_tokenizers.Split(Regex(GPT2), "isolated")
    digests = []
    for b in range(0x110000 // BLOCK):
        text = "".join(f"{c}a{c}1 {c}" for c in block_chars(b))
        pieces = [p for p, _ in split.pre_tokenize_str(text)]
        digests.append(fnv1a64("".join(f"{len(p.encode())}:{p}" for p in pieces)))
    return digests


def decode_only(raw: RawTokenizer, byte_level: bool) -> dict:
    def ids(*tokens: str) -> list:
        out = [raw.token_to_id(t) for t in tokens]
        assert None not in out, tokens
        return out

    if byte_level:
        byte = lambda data: ids(*(BYTE_CHARS[b] for b in data))  # noqa: E731
        return {
            "incomplete_cjk": byte("你".encode()[:2]),
            "incomplete_emoji_then_ascii": byte("😀".encode()[:3] + b"a"),
            "bytes_around_special": byte("é".encode()[:1]) + ids("[MASK]") + byte("é".encode()[1:]),
            "truncated_sequences": byte(b"\xe4\xbd x \xf0\x9f\x98 \x80y"),
            "space_run_tokens": ids("a", "   ", "b", "  ", "[CLS]", "|||IP_ADDRESS|||", "[unused3]", "[SEP]"),
        }
    byte = lambda data: ids(*(f"<0x{b:02X}>" for b in data))  # noqa: E731
    return {
        "incomplete_cjk": byte("你".encode()[:2]),
        "complete_cjk_bytes": byte("你好".encode()),
        "bytes_around_special": byte("é".encode()[:1]) + ids("<mask>") + byte("é".encode()[1:]),
        "invalid_lead": byte(b"\xff\xfe") + ids("▁x") + byte(b"\xc0\xaf"),
        "replacement_and_runs": ids("<bos>", "▁", "▁▁", "\n\n", "▁hello", "<unk>", "<eos>", "<pad>"),
    }


def tokenizer_payload(path: Path, texts: dict) -> dict:
    hf = AutoTokenizer.from_pretrained(str(path))
    raw = RawTokenizer.from_file(str(path / "tokenizer.json"))
    raw.no_truncation()
    raw.no_padding()

    def make_case(name: str, text: str, parse_special: bool) -> dict:
        raw.encode_special_tokens = not parse_special
        ids = raw.encode(text, add_special_tokens=False).ids
        with_special = raw.encode(text, add_special_tokens=True).ids
        for add, expected in ((False, ids), (True, with_special)):
            got = hf(text, add_special_tokens=add, split_special_tokens=not parse_special)["input_ids"]
            assert got == expected, f"{path}: transformers and tokenizers disagree on {name}"
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
    raw.encode_special_tokens = False
    byte_level = json.loads(raw.to_str())["pre_tokenizer"]["type"] == "ByteLevel"
    decode_cases = [
        {
            "name": name,
            "ids": token_ids,
            "decoded": raw.decode(token_ids, skip_special_tokens=False),
            "decoded_skip_special": raw.decode(token_ids, skip_special_tokens=True),
        }
        for name, token_ids in decode_only(raw, byte_level).items()
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
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    roots = {"laya": Path(sys.argv[1]), "julia": Path(sys.argv[2])}
    rng = random.Random(50368)
    texts = {**WORDPIECE_TEXTS, **TEXTS, **random_texts(rng)}
    written, payloads = {}, {}
    for model, folders in FOLDERS.items():
        for folder in folders:
            key = f"{model}/{folder}"
            payload = tokenizer_payload(roots[model] / folder, texts)
            same = next((earlier for earlier, p in payloads.items() if p == payload), None)
            payloads[key] = payload
            written[key] = {"same_as": same} if same else payload
            print(f"{key}: {len(payload['encode_cases'])} encode and {len(payload['decode_cases'])} decode cases"
                  + (f", all the same as {same}'s" if same else ""))
    components = {
        "block": BLOCK,
        "gpt2_digests": gpt2_digests(),
        "byte_level": byte_level_cases(rng),
        "metaspace": metaspace_cases(rng),
    }
    out = {"tokenizers": tokenizers.__version__, "components": components, "tokenizers_by_folder": written}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
