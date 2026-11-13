#!/usr/bin/env python3
"""Sends a file of decisions to a System One server and scores the answers (standard library only).

    python3 tests/e2e/systemone_eval.py --url http://127.0.0.1:8080 [--questions tests/e2e/decisions25.json]
        [--reference <reference.py output for the same file, or another run's --save>] [--save <answers>]
        [--repeat 3] [--json <report>]

Checks, per request: the response follows the API (model, answers keyed by the question names, usage; each answer's
fields for its type, probabilities over exactly the request's options summing to 1, the choice being the most likely
option), the answer is the accepted one (a choice's option, a noul's side of 0.5, a score's most likely level), and,
with --reference, every probability against the model's own Python code. Latency is the client's wall time per
request: of the first send, which follows another request's state, and the median of the --repeat - 1 sends after it,
which ask about the state just read (a model that keeps a state's prefix answers those faster). The server's own
split comes from its Server-Timing header.
--save keeps every probability of the run in the reference's format, so that a later run (other weights, another
precision) can be compared with it through --reference: answers that changed, and how far probabilities moved.
A request the model refuses (400) counts its questions as not answered; with --reference, it must be one the
reference refused too (a case with "refused"), and the other way round.
Exit status 1 when a response breaks the API or (with --reference) a probability is off by more than --tolerance or a
refusal differs.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def post(url, body):
    """The response's body (a refusal: {"refused": its detail}), the wall time in ms and its Server-Timing."""
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/v1/systemone", data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            text = response.read()
            timing = response.headers.get("Server-Timing", "")
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        return {"refused": json.loads(e.read()).get("detail")}, (time.perf_counter() - start) * 1000, ""
    return json.loads(text), (time.perf_counter() - start) * 1000, timing


def server_ms(timing, name):
    for part in timing.split(","):
        key, _, value = part.strip().partition(";dur=")
        if key == name:
            return float(value)
    return float("nan")


def options(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def probabilities(question, answer):
    """The answer's probability per option, and what breaks the API in it."""
    problems = []
    if answer.get("type") != question["type"]:
        problems.append(f"type {answer.get('type')!r}")
    if question["type"] == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            return {}, problems + [f"noul {p!r}"]
        return {"true": p, "false": 1 - p}, problems
    p = answer.get("probabilities")
    if not isinstance(p, dict) or list(p) != options(question):
        return {}, problems + ["probabilities do not list the request's options in order"]
    if abs(sum(p.values()) - 1) > 1e-3:
        problems.append(f"probabilities sum to {sum(p.values()):.4f}")
    if not 0 <= answer.get("confidence", -1) <= 1:
        problems.append(f"confidence {answer.get('confidence')!r}")
    top = max(p, key=p.get)
    if question["type"] == "choice":
        if p.get(answer.get("choice"), -1) < p[top] - 1e-9:
            problems.append(f"choice {answer.get('choice')!r} is not the most likely option")
    else:
        expected = sum(i * v for i, v in enumerate(p.values()))
        if abs(answer.get("score", -1) - expected) > 1e-3:
            problems.append(f"score {answer.get('score')!r} is not the expected level {expected:.4f}")
        if answer.get("legend") != {str(i): level for i, level in enumerate(question["criteria"])}:
            problems.append("legend does not repeat the criteria")
    return p, problems


def given(question, p):
    """What the answer says, in the form the accepted answers use."""
    if question["type"] == "noul":
        return p["true"] >= 0.5
    top = max(p, key=p.get)
    return top if question["type"] == "choice" else int(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--questions", default="tests/e2e/decisions25.json")
    ap.add_argument("--reference")
    ap.add_argument("--tolerance", type=float, default=0.02, help="--reference: the largest probability difference accepted")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--save", help="write every answer's probabilities, in the format --reference reads")
    ap.add_argument("--quiet", action="store_true", help="print the summary only")
    ap.add_argument("--json")
    args = ap.parse_args()
    with open(args.questions) as f:
        data = json.load(f)
    reference = None
    if args.reference:
        with open(args.reference) as f:
            reference = json.load(f)["cases"]
        assert len(reference) == len(data["requests"]), "the reference is for another file"

    rows, broken, off, saved, refused = [], 0, 0, [], 0
    for index, request in enumerate(data["requests"]):
        saved.append({"probabilities": {}})
        sends = [post(args.url, request) for _ in range(args.repeat)]
        response, first, timing = sends[0]
        if "refused" in response:
            refused += 1
            saved[-1] = {"refused": response["refused"]}
            if not args.quiet:
                print(f"request {index}: refused: {response['refused']}")
            if reference is not None and "refused" not in reference[index]:
                broken += 1
                print(f"request {index}: refused, which the reference answers")
            continue
        if reference is not None and "refused" in reference[index]:
            broken += 1
            print(f"request {index}: answered, which the reference refuses ({reference[index]['refused']})")
        again = sends[1:] or sends
        wall = statistics.median(ms for _, ms, _ in again)
        problems = [k for k in ("model", "answers", "usage") if k not in response]
        if list(response.get("answers", {})) != list(request["questions"]):
            problems.append("answers do not follow the questions")
        for name, question in request["questions"].items():
            p, more = probabilities(question, response.get("answers", {}).get(name, {}))
            problems += more
            row = {"name": name, "type": question["type"], "tokens": response.get("usage", {}).get("input_tokens"),
                   "first_ms": first, "ms": wall,
                   "device_ms": statistics.median(server_ms(t, "device") for _, _, t in again),
                   "server_ms": statistics.median(server_ms(t, "total") for _, _, t in again), "problems": problems}
            if p:
                saved[-1]["probabilities"][name] = p
                row["answer"] = given(question, p)
                row["p"] = max(p.values()) if question["type"] != "noul" else p["true"]
                if "expected" in data:
                    row["expected"] = data["expected"][index][name]
                if reference is not None:
                    ref = reference[index]["probabilities"][name]
                    row["diff"] = max(abs(p[o] - ref[o]) for o in ref)
                    row["reference"] = given(question, ref)
                    off += row["diff"] > args.tolerance
            rows.append(row)
        broken += bool(problems)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"kind": "server", "cases": saved}, f)
    if not args.quiet:
        print(f"{'question':14s} {'type':7s} {'tokens':>6s} {'answer':>16s} {'p':>6s} {'ok':>3s} {'|dp| ref':>9s} "
              f"{'first ms':>9s} {'again ms':>9s} {'device':>8s}")
    for r in [] if args.quiet else rows:
        ok = "" if "expected" not in r else ("yes" if r.get("answer") == r["expected"] else "NO")
        diff = f"{r['diff']:.4f}" if "diff" in r else ""
        print(f"{r['name']:14s} {r['type']:7s} {r['tokens']!s:>6s} {str(r.get('answer')):>16s} {r.get('p', float('nan')):6.3f} {ok:>3s} "
              f"{diff:>9s} {r['first_ms']:9.1f} {r['ms']:9.1f} {r['device_ms']:8.1f}"
              + ("  " + "; ".join(r["problems"]) if r["problems"] else ""))

    def spread(values):
        values = sorted(values)
        return {"p50": statistics.median(values), "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
                "min": values[0], "max": values[-1]}

    summary = {"questions": len(rows), "refused_requests": refused, "api_problems": broken,
               "latency_ms": {"new_state": spread(r["first_ms"] for r in rows), "same_state": spread(r["ms"] for r in rows),
                              "server_p50": statistics.median(r["server_ms"] for r in rows),
                              "device_p50": statistics.median(r["device_ms"] for r in rows)},
               "tokens": {"min": min(r["tokens"] for r in rows), "max": max(r["tokens"] for r in rows)}}
    if "expected" in data:
        summary["correct"] = sum(r.get("answer") == r["expected"] for r in rows)
    if reference is not None:
        diffs = sorted(r.get("diff", 0) for r in rows)
        summary["max_probability_difference"] = diffs[-1]
        summary["mean_probability_difference"] = statistics.fmean(diffs)
        summary["p95_probability_difference"] = diffs[min(len(diffs) - 1, int(0.95 * len(diffs)))]
        summary["same_answer_as_reference"] = sum(r.get("answer") == r.get("reference") for r in rows)
        if "expected" in data:
            summary["reference_correct"] = sum(r.get("reference") == r["expected"] for r in rows)
        summary["off_by_more_than_tolerance"] = off
    print(json.dumps(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=1)
    sys.exit(1 if broken or off else 0)


if __name__ == "__main__":
    main()
