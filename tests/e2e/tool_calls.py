"""Tool-calling cases against a running server's /v1/chat/completions, with simulated tools.

    tool_calls.py <base url> <label> <out.json> [--kwargs '{"enable_thinking": false}']

23 cases, easy to medium: one tool or several offered, calls in parallel, no tool needed, none that fits, a system
prompt, thinking on, a streamed call, and six conversations of two or three user turns that need an earlier turn's
tool result. Each case is a script of user turns; after every user turn the model is called until it stops asking for
tools (at most 4 rounds), and its calls are answered by the simulated tools below. Everything received is written to
<out.json> for grading against tool_calls_rubric.md (by a person or a stronger model): the format rules there are what
a chat template and its output parser must get right, whatever the model's judgment.

Two runs of one model that differ in template or weight format (the original checkpoint and an import of it) can be
compared call by call.
"""
import argparse, json, re, sys, urllib.request

def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

S = {"type": "string"}
TOOLS = {
    "get_weather": fn("get_weather", "Get the current weather for a city.",
                      {"city": {"type": "string", "description": "City name, e.g. Paris"},
                       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit (default celsius)"}}, ["city"]),
    "get_stock_price": fn("get_stock_price", "Get the latest share price for a stock ticker symbol.",
                          {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}}, ["symbol"]),
    "convert_currency": fn("convert_currency", "Convert an amount of money between two currencies.",
                           {"amount": {"type": "number"}, "from": {"type": "string", "description": "ISO code, e.g. USD"},
                            "to": {"type": "string", "description": "ISO code, e.g. EUR"}}, ["amount", "from", "to"]),
    "get_order_status": fn("get_order_status", "Look up the shipping status of an order by its id.",
                           {"order_id": S}, ["order_id"]),
    "set_reminder": fn("set_reminder", "Create a reminder.",
                       {"text": {"type": "string", "description": "What to be reminded of"},
                        "time": {"type": "string", "description": "When, e.g. 'tomorrow 09:00'"}}, ["text", "time"]),
    "create_calendar_event": fn("create_calendar_event", "Create a calendar event.",
                                {"title": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                                 "start_time": {"type": "string", "description": "HH:MM, 24-hour"},
                                 "duration_minutes": {"type": "integer"},
                                 "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses"}},
                                ["title", "date", "start_time", "duration_minutes"]),
    "book_flight": fn("book_flight", "Book flight seats.",
                      {"origin": S, "destination": S, "date": {"type": "string", "description": "YYYY-MM-DD"},
                       "passengers": {"type": "integer"},
                       "cabin": {"type": "string", "enum": ["economy", "premium_economy", "business", "first"]}},
                      ["origin", "destination", "date", "passengers"]),
    "send_email": fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    "calculator": fn("calculator", "Evaluate an arithmetic expression and return the number.",
                     {"expression": {"type": "string", "description": "e.g. '0.2 * 150'"}}, ["expression"]),
    "search_contacts": fn("search_contacts", "Find a person in the address book by name.", {"name": S}, ["name"]),
}

WEATHER = {"paris": (18, "cloudy"), "tokyo": (22, "sunny"), "london": (11, "rain"), "madrid": (24, "sunny"),
           "berlin": (7, "overcast"), "sydney": (26, "sunny"), "lisbon": (21, "clear"), "rome": (19, "partly cloudy")}
STOCKS = {"AAPL": 231.40, "MSFT": 448.20}
RATES = {("USD", "EUR"): 0.92, ("USD", "JPY"): 152.3, ("EUR", "USD"): 1.087}
ORDERS = {"A1B2-9931": {"status": "shipped", "estimated_delivery": "2026-11-15"},
          "77-XYZ": {"status": "delayed", "reason": "held at customs", "estimated_delivery": "2026-11-21"}}

def simulate(name, args):
    try:
        if name == "get_weather":
            city = str(args.get("city", "")).split(",")[0].strip().lower()
            if city not in WEATHER: return {"error": f"unknown city '{args.get('city')}'"}
            c, cond = WEATHER[city]
            if str(args.get("unit", "celsius")).lower().startswith("f"):
                return {"city": args["city"], "temperature": round(c * 9 / 5 + 32), "unit": "fahrenheit", "condition": cond}
            return {"city": args["city"], "temperature": c, "unit": "celsius", "condition": cond}
        if name == "get_stock_price":
            sym = str(args.get("symbol", "")).upper()
            return {"symbol": sym, "price": STOCKS[sym], "currency": "USD"} if sym in STOCKS else {"error": f"unknown symbol '{sym}'"}
        if name == "convert_currency":
            key = (str(args.get("from", "")).upper(), str(args.get("to", "")).upper())
            if key not in RATES: return {"error": f"no rate for {key[0]}->{key[1]}"}
            return {"amount": round(float(args["amount"]) * RATES[key], 2), "currency": key[1], "rate": RATES[key]}
        if name == "get_order_status":
            return ORDERS.get(str(args.get("order_id", "")), {"error": "order not found"})
        if name == "calculator":
            expr = str(args.get("expression", ""))
            if not re.fullmatch(r"[0-9+\-*/().,% ]+", expr): return {"error": "unsupported expression"}
            return {"result": round(eval(expr.replace(",", "").replace("%", "/100"), {"__builtins__": {}}), 6)}
        if name == "search_contacts":
            if "alice" in str(args.get("name", "")).lower():
                return {"name": "Alice Chen", "city": "Lisbon", "email": "alice.chen@example.com"}
            return {"error": "no contact found"}
        if name in ("set_reminder", "create_calendar_event", "book_flight", "send_email"):
            return {"status": "ok", "id": name[:3] + "-1042"}
    except Exception as e:  # a malformed argument is reported back like a real tool would
        return {"error": f"bad arguments: {e}"}
    return {"error": f"unknown tool '{name}'"}

ALL = list(TOOLS)
CASES = [
    # --- easy, one turn
    dict(id="E1", level="easy", tools=["get_weather"], turns=["What's the weather in Paris right now?"]),
    dict(id="E2", level="easy", tools=["get_weather"], turns=["How hot is it in Tokyo? Give it to me in Fahrenheit."]),
    dict(id="E3", level="easy", tools=["get_stock_price", "get_weather"], turns=["What is Apple's stock price?"]),
    dict(id="E4", level="easy", tools=["convert_currency"], turns=["Convert 250 US dollars to euros."]),
    dict(id="E5", level="easy", tools=["get_weather", "get_stock_price"], turns=["What is the capital of France?"]),
    dict(id="E6", level="easy", tools=["get_order_status"], turns=["Where is my order A1B2-9931?"]),
    # --- medium, one turn
    dict(id="M1", level="medium", tools=ALL, turns=["Remind me to call the dentist tomorrow at 9am."]),
    dict(id="M2", level="medium", tools=ALL, turns=["Schedule a 45 minute meeting called 'Budget review' on 2026-12-03 at 2:30pm with alice@example.com and bob@example.com."]),
    dict(id="M3", level="medium", tools=["get_weather", "get_stock_price"], turns=["What's the weather in London and in Madrid?"]),
    dict(id="M4", level="medium", tools=ALL, turns=["I need two things: the weather in Berlin and Microsoft's current share price."]),
    dict(id="M5", level="medium", tools=ALL, turns=["Book 2 business class seats from Boston to Seattle on 2026-12-20."]),
    dict(id="M6", level="medium", tools=ALL, turns=["Email sam@example.com with the subject 'Lunch' and tell him I'll be 10 minutes late."]),
    dict(id="M7", level="medium", tools=["get_weather", "get_stock_price", "convert_currency"], turns=["Play some jazz music for me."]),
    dict(id="M8", level="medium", tools=["calculator", "get_weather"], turns=["Use the calculator: what is 18.5% of 2,400?"]),
    dict(id="M9", level="medium", tools=["get_order_status", "send_email"],
         system="You are a support agent for Acme. Always look up the order before you answer anything about it.",
         turns=["My order 77-XYZ still hasn't arrived. What's going on?"]),
    dict(id="M10", level="medium", tools=["get_weather"], thinking=True, turns=["Is it warm in Sydney today?"]),
    dict(id="M11", level="medium", tools=ALL, stream=True, turns=["Convert 80 euros to US dollars."]),
    # --- multi-turn
    dict(id="T1", level="multi-turn", tools=["get_weather"], turns=["What's the weather in Paris?", "Should I take an umbrella or sunglasses?"]),
    dict(id="T2", level="multi-turn", tools=["search_contacts", "get_weather"], turns=["What's the weather like where Alice Chen lives?"]),
    dict(id="T3", level="multi-turn", tools=["get_weather"], turns=["What's the temperature in Rome?", "And what is that in Fahrenheit? Check again with the tool."]),
    dict(id="T4", level="multi-turn", tools=["get_weather", "book_flight"],
         turns=["I'm thinking about a trip to Tokyo next week. Any quick tips?", "Good to know. Can you check the weather there right now?",
                "Great. Book me 1 economy seat from Lisbon to Tokyo on 2026-11-25."]),
    dict(id="T5", level="multi-turn", tools=["convert_currency", "calculator"],
         turns=["Convert 100 US dollars to Japanese yen.", "If I exchange that amount three times, how many yen do I have in total?"]),
    dict(id="T6", level="multi-turn", tools=["get_order_status", "send_email"],
         system="You are a support agent for Acme. Always look up the order before you answer anything about it.",
         turns=["Can you check order A1B2-9931?", "Thanks. Now email me the details at pat@example.com, subject 'Order update'."]),
]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read().decode()

def call(base, messages, tools, kwargs, stream):
    body = {"model": "m", "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 700,
            "chat_template_kwargs": kwargs}
    if not stream:
        r = json.loads(post(base + "/v1/chat/completions", body))
        ch = r["choices"][0]
        return ch["message"], ch.get("finish_reason"), r.get("usage")
    body["stream"] = True
    content, reasoning, calls, finish = "", "", {}, None
    for line in post(base + "/v1/chat/completions", body).splitlines():
        if not line.startswith("data: ") or line.strip() == "data: [DONE]": continue
        ch = json.loads(line[6:])["choices"]
        if not ch: continue
        d = ch[0].get("delta", {}); finish = ch[0].get("finish_reason") or finish
        content += d.get("content") or ""; reasoning += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}})
            c["id"] = c["id"] or tc.get("id")
            c["function"]["name"] += tc.get("function", {}).get("name") or ""
            c["function"]["arguments"] += tc.get("function", {}).get("arguments") or ""
    msg = {"role": "assistant", "content": content or None}
    if reasoning: msg["reasoning_content"] = reasoning
    if calls: msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, finish, {"streamed": True}

def run_case(base, case, default_kwargs):
    kwargs = dict(default_kwargs)
    if case.get("thinking"): kwargs["enable_thinking"] = True
    tools = [TOOLS[t] for t in case["tools"]]
    messages = [{"role": "system", "content": case["system"]}] if case.get("system") else []
    log = []
    for turn in case["turns"]:
        messages.append({"role": "user", "content": turn})
        log.append({"user": turn, "rounds": []})
        for _ in range(4):
            try:
                msg, finish, usage = call(base, messages, tools, kwargs, case.get("stream", False))
            except Exception as e:
                log[-1]["rounds"].append({"error": repr(e)[:300]}); break
            entry = {"finish_reason": finish, "content": msg.get("content"), "tool_calls": [], "usage": usage}
            if msg.get("reasoning_content"): entry["reasoning_chars"] = len(msg["reasoning_content"])
            keep = {"role": "assistant", "content": msg.get("content")}
            if msg.get("tool_calls"): keep["tool_calls"] = msg["tool_calls"]
            messages.append(keep)
            for tc in msg.get("tool_calls") or []:
                name, raw = tc["function"]["name"], tc["function"]["arguments"]
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw; valid = isinstance(args, dict)
                except Exception:
                    args, valid = None, False
                result = simulate(name, args) if valid else {"error": "arguments are not a JSON object"}
                entry["tool_calls"].append({"name": name, "arguments_raw": raw, "arguments_valid_json": valid,
                                            "known_tool": name in case["tools"], "result": result})
                messages.append({"role": "tool", "tool_call_id": tc.get("id") or "call_0", "content": json.dumps(result)})
            log[-1]["rounds"].append(entry)
            if not msg.get("tool_calls"): break
    return {"id": case["id"], "level": case["level"], "tools_offered": case["tools"], "system": case.get("system"),
            "thinking": bool(case.get("thinking")), "stream": bool(case.get("stream")), "turns": log}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("label"); ap.add_argument("out")
    ap.add_argument("--kwargs", default='{"enable_thinking": false}')
    a = ap.parse_args()
    out = {"label": a.label, "cases": [run_case(a.base, c, json.loads(a.kwargs)) for c in CASES]}
    json.dump(out, open(a.out, "w"), indent=1)
    n_calls = sum(len(r["tool_calls"]) for c in out["cases"] for t in c["turns"] for r in t["rounds"])
    errs = sum(1 for c in out["cases"] for t in c["turns"] for r in t["rounds"] if "error" in r)
    print(f"{a.label}: {len(out['cases'])} cases, {n_calls} tool calls, {errs} request errors")
