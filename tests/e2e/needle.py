"""A fact planted in a long text, asked for through a running server's /v1/chat/completions.

    needle.py <base url> <haystack.txt> [--tokens 8000,32000] [--depths 0.25,0.75] [--kwargs '{"enable_thinking": false}']
              [--json out.json]

For each length and depth: the first `tokens` tokens' worth of <haystack.txt> (any long text; about 4.3 characters a
token), one sentence naming a six-digit code planted at `depth` of it (0 the start, 1 the end), and a question for the
code. A probe passes when the answer holds the code. The codes differ per probe, so no probe is answered from another's
cache. What a setting that only matters at length changes (a KV cache format, a window, a context limit) shows as
probes that stop passing; a model that fails them at the default setting is too small for the length, not broken.

Prints one line per probe (prompt tokens from the server's usage, seconds, pass or the answer) and the pass count;
exits 1 when a probe fails.
"""
import argparse, json, random, sys, time, urllib.request

CHARS_PER_TOKEN = 4.3


def ask(url, messages, kwargs):
    body = {"messages": messages, "max_tokens": 48, "temperature": 0}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("haystack")
    ap.add_argument("--tokens", default="8000,32000")
    ap.add_argument("--depths", default="0.25,0.75")
    ap.add_argument("--kwargs", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    text = open(a.haystack, encoding="utf-8").read()
    kwargs = json.loads(a.kwargs) if a.kwargs else None
    rng = random.Random("needle")
    rows = []
    for tokens in (int(t) for t in a.tokens.split(",")):
        hay = text[: int(tokens * CHARS_PER_TOKEN)]
        if len(hay) < int(tokens * CHARS_PER_TOKEN):
            sys.exit(f"the haystack is shorter than {tokens} tokens")
        for depth in (float(d) for d in a.depths.split(",")):
            code = str(rng.randrange(100000, 1000000))
            name = rng.choice(["amber", "cobalt", "crimson", "ivory", "jade", "onyx", "saffron", "violet"])
            at = hay.rfind("\n", 0, int(len(hay) * depth)) + 1  # a line's start
            needle = f"The access code of the {name} vault is {code}.\n"
            prompt = (hay[:at] + needle + hay[at:] +
                      f"\n\nWhat is the access code of the {name} vault? Answer with the number only.")
            t0 = time.time()
            reply = ask(a.url, [{"role": "user", "content": prompt}], kwargs)
            answer = reply["choices"][0]["message"].get("content") or ""
            row = {"tokens": reply.get("usage", {}).get("prompt_tokens"), "depth": depth, "code": code,
                   "answer": answer.strip(), "pass": code in answer, "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{row['tokens']} tokens, depth {depth}: {row['seconds']} s, "
                  f"{'pass' if row['pass'] else 'FAIL: ' + repr(row['answer'][:80])}", flush=True)
    passed = sum(r["pass"] for r in rows)
    print(f"{passed} of {len(rows)} probes passed")
    if a.json:
        json.dump({"probes": rows, "passed": passed}, open(a.json, "w"), indent=1)
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
