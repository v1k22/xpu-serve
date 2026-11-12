# Rubric: tool-calling cases

Each transcript file (written by `tool_calls.py`) holds one model's run of the same 23 cases. A case is a script of user turns; after each user
turn the model was called until it stopped asking for tools (at most 4 rounds); every tool call was answered by a
simulated tool whose result is recorded under `result`. `tools_offered` lists the only tools the model was given.

## What every case must satisfy (format)
- A tool call names a tool from `tools_offered` (`known_tool` true) and its arguments are a JSON object
  (`arguments_valid_json` true) with the tool's required fields.
- A round that calls a tool has `finish_reason` "tool_calls"; a round that answers has text in `content`.
- No markup leaks into `content`: no `<tool_call>`, `</tool_call>`, `<think>`, `</think>`, `<|im_start|>`,
  `<|im_end|>` or raw JSON of a call.
- In later turns the model behaves as if it saw the earlier turns and tool results.

## What each case expects (task)
Simulated data: weather Paris 18 C cloudy; Tokyo 22 C (72 F) sunny; London 11 C rain; Madrid 24 C sunny; Berlin 7 C
overcast; Sydney 26 C sunny; Lisbon 21 C clear; Rome 19 C (66 F) partly cloudy. AAPL 231.40, MSFT 448.20 USD.
USD->EUR 0.92, USD->JPY 152.3, EUR->USD 1.087. Order A1B2-9931: shipped, delivery 2026-11-15. Order 77-XYZ: delayed,
held at customs, delivery 2026-11-21. Contact Alice Chen lives in Lisbon.

| Case | Expected |
|---|---|
| E1 | get_weather(city Paris); the answer gives 18 C and cloudy |
| E2 | get_weather(city Tokyo, unit fahrenheit); the answer gives 72 F |
| E3 | get_stock_price(symbol AAPL), not get_weather; the answer gives 231.40 |
| E4 | convert_currency(amount 250, from USD, to EUR); the answer gives 230 euros |
| E5 | NO tool call; answers Paris |
| E6 | get_order_status(order_id A1B2-9931); the answer says shipped and 2026-11-15 |
| M1 | set_reminder only (text about calling the dentist, time tomorrow 9am in any clear form) |
| M2 | create_calendar_event(title Budget review, date 2026-12-03, start_time 14:30, duration_minutes 45, attendees both addresses) |
| M3 | two get_weather calls, London and Madrid (in one round or two); the answer gives both |
| M4 | get_weather(Berlin) and get_stock_price(MSFT); the answer gives both results |
| M5 | book_flight(origin Boston, destination Seattle, date 2026-12-20, passengers 2, cabin business) |
| M6 | send_email(to sam@example.com, subject Lunch, body saying the sender will be 10 minutes late) |
| M7 | NO tool call (no offered tool plays music); says it cannot do that |
| M8 | calculator with an expression worth 444; the answer gives 444 |
| M9 | get_order_status(77-XYZ) before any answer; the answer says delayed / held at customs (the date is a bonus) |
| M10 | thinking was ON: get_weather(Sydney); the answer says 26 C / warm; no `<think>` text inside `content` (`reasoning_chars` shows reasoning came back apart, if the model reasoned) |
| M11 | the reply was STREAMED: convert_currency(amount 80, from EUR, to USD) with valid JSON arguments; the answer gives 86.96 dollars |
| T1 | turn 1: get_weather(Paris), answer 18 C cloudy. Turn 2: advice consistent with cloudy weather (umbrella rather than sunglasses), from memory or by calling the tool again; must not invent other weather |
| T2 | search_contacts(Alice Chen), THEN get_weather(Lisbon) (the city must come from the contact); the answer gives Lisbon 21 C clear |
| T3 | turn 1: get_weather(Rome), answer 19 C. Turn 2: get_weather(Rome, unit fahrenheit) again, answer 66 F |
| T4 | turn 1: text tips, no booking (a weather call is acceptable). Turn 2: get_weather(Tokyo), answer 22 C sunny. Turn 3: book_flight(origin Lisbon, destination Tokyo, date 2026-11-25, passengers 1, cabin economy) |
| T5 | turn 1: convert_currency(100, USD, JPY), answer 15,230 yen. Turn 2: 45,690 yen, by the calculator or in its head; the number must be right |
| T6 | turn 1: get_order_status(A1B2-9931), answer shipped / 2026-11-15. Turn 2: send_email(to pat@example.com, subject Order update, body with the status and the delivery date) |

Minor variations are fine: city spelled "Paris, France", symbol case, an extra polite sentence, numbers rounded
sensibly, times written "9:00 AM tomorrow".
