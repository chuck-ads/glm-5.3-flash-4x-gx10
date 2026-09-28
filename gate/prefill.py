#!/usr/bin/env python3
"""Profile any OpenAI-compatible endpoint: TTFT, prefill and decode across
prompt sizes, plus how to make the model think.

Answers the two questions every newly booted model raises -- what is it worth at
each size, and which request shape turns reasoning on -- without assuming
anything about the engine. Run it on a node against a quiet box.

Three traps this avoids, each of which has produced a wrong number here:

Repeated prompts measure the prefix cache, not prefill. A 200k probe that took
198s cold came back in ~6s warm. Every prompt is gibberish drawn from a per-run
seed, so nothing shares a prefix with the cache or with another size.

Counting stream events instead of tokens undercounts decode by the acceptance
length. Speculative decoding packs several accepted tokens into one SSE event, so
events/second read 12.7 where the tokens were arriving at 63/s. Decode here comes
from `usage.completion_tokens`.

The reasoning field is spelled `reasoning_content` by SGLang and `reasoning` by
vLLM. Reading one and finding the other empty looks exactly like a parser
discarding the trace: a short answer, an empty reasoning field, and a token count
far larger than the answer. Both are checked, and the message keys are printed
when neither holds text.
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

VOWEL, CONS = "aeiou", "bcdfghjklmnprstvwz"
REASONING_KEYS = ("reasoning_content", "reasoning")

# Request shapes that ask for thinking. Engines disagree, and several shapes a
# client is likely to try are accepted and silently dropped as unknown fields,
# so the only way to know is to send each one and look.
THINKING_SHAPES = [
    ("(bare, engine default)", {}),
    ("chat_template_kwargs thinking", {"chat_template_kwargs": {"thinking": True}}),
    ("chat_template_kwargs enable_thinking",
     {"chat_template_kwargs": {"enable_thinking": True}}),
    ("reasoning_effort high", {"reasoning_effort": "high"}),
    ("reasoning {enabled:true}", {"reasoning": {"enabled": True}}),
    ("thinking {type:enabled}", {"thinking": {"type": "enabled"}}),
    ("enable_thinking (top level)", {"enable_thinking": True}),
]

EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]


def words(n, seed):
    """Gibberish with no shared prefix, so a prompt cannot be served from cache."""
    r = random.Random(seed)
    out = []
    for _ in range(n):
        k = r.randint(2, 4)
        out.append("".join(r.choice(CONS) + r.choice(VOWEL) for _ in range(k)))
    return " ".join(out)


def reasoning_of(message):
    """The reasoning text and the key it came under, or ("", None)."""
    for key in REASONING_KEYS:
        text = message.get(key)
        if text:
            return text, key
    return "", None


def chat(base, model, prompt, max_tokens, stream=True, extra=None, timeout=1800):
    """One turn. Returns (elapsed, ttft, usage, message, error).

    ttft is the first token of either channel: a thinking model emits reasoning
    first, and timing to the first `content` delta would report the whole trace
    as latency.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
        **(extra or {}),
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft, usage, message = None, None, {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                d = json.loads(r.read().decode())
                return (time.time() - t0, None, d.get("usage"),
                        d["choices"][0]["message"], None)
            parts, reasoning_parts, finish = [], [], None
            # Keep the key the server actually used: normalising it here would
            # hide the engine difference this column exists to show.
            wire_key = None
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                d = json.loads(chunk)
                if d.get("usage"):
                    usage = d["usage"]
                choices = d.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                finish = choices[0].get("finish_reason") or finish
                text = delta.get("content")
                think, key = reasoning_of(delta)
                if key:
                    wire_key = key
                if text:
                    parts.append(text)
                if think:
                    reasoning_parts.append(think)
                if ttft is None and (text or think):
                    ttft = time.time() - t0
            message = {"content": "".join(parts), "finish_reason": finish}
            if reasoning_parts:
                message[wire_key] = "".join(reasoning_parts)
    except urllib.error.HTTPError as e:
        return time.time() - t0, None, None, {}, f"HTTP {e.code} {e.read()[:120]!r}"
    except Exception as e:  # a timeout or a dropped stream is a result, not a crash
        return time.time() - t0, None, None, {}, f"{type(e).__name__}: {e}"
    return time.time() - t0, ttft, usage, message, None


def models(base):
    with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=60) as r:
        return json.loads(r.read())["data"]


def parse_size(text):
    text = text.strip().lower()
    mult = {"k": 1000, "m": 1000000}
    return int(float(text[:-1]) * mult[text[-1]]) if text[-1] in mult else int(text)


def size_sweep(base, model, sizes, seed, out_tokens, tpw, window):
    print(f"\nsize sweep (first touch, uncacheable, one at a time, "
          f"{out_tokens} output tokens)")
    print(f"{'prompt_tok':>11} {'ttft_s':>8} {'prefill_t/s':>12} "
          f"{'decode_t/s':>11} {'out_tok':>8} {'reasoning':>10}  finish")
    for i, target in enumerate(sizes):
        if window and target >= window:
            print(f"{target:>11} {'skipped':>8}  past the model's window ({window})")
            continue
        # Seeding by the word count too: two runs whose calibrations differ
        # would otherwise share every word up to the shorter length, and the
        # longer one then prefills partly from the prefix cache.
        n = int(target / tpw)
        prompt = words(n, seed * 7919 + i + n * 104729)
        elapsed, ttft, usage, message, error = chat(
            base, model, prompt, out_tokens)
        if error:
            print(f"{target:>11} {'-':>8}  {error}")
            continue
        got = usage["prompt_tokens"] if usage else 0
        completion = usage.get("completion_tokens", 0) if usage else 0
        # Decode excludes the first token, whose cost is the prefill.
        decode = (completion - 1) / (elapsed - ttft) if ttft and completion > 1 else 0
        prefill = got / ttft if ttft else 0
        think, _ = reasoning_of(message)
        print(f"{got:>11} {ttft:>8.2f} {prefill:>12.0f} {decode:>11.1f} "
              f"{completion:>8} {len(think):>10}  {message.get('finish_reason')}")


def thinking_probe(base, model):
    print("\nthinking shapes (a shape the engine does not know is dropped silently)")
    print(f"  {'request':<38} {'thinking':<9} {'field':<18} answer")
    working = []
    for label, extra in THINKING_SHAPES:
        _, _, usage, message, error = chat(
            base, model, "What is 17*23? Number only.", 600, extra=extra, timeout=300)
        if error:
            print(f"  {label:<38} {'-':<9} {'-':<18} {error}")
            continue
        think, key = reasoning_of(message)
        on = bool(think)
        if on:
            working.append((label, extra))
        answer = (message.get("content") or "").strip().replace("\n", " ")[:18]
        print(f"  {label:<38} {'ON' if on else 'off':<9} {key or '-':<18} {answer!r}")
        # Content empty with tokens spent is the shape that looks like a bug: the
        # trace never closed, so there is no answer yet.
        if not answer and usage and usage.get("completion_tokens"):
            print(f"  {'':<38} {'':<9} {'':<18} "
                  f"^ empty answer, {usage['completion_tokens']} tokens spent: "
                  f"the trace did not close inside max_tokens")
        if not on and not key:
            _, _, _, full, _ = chat(base, model, "hi", 16, extra=extra,
                                    stream=False, timeout=300)
            if full:
                print(f"  {'':<38} keys: {sorted(full)}")
    return working


def effort_sweep(base, model, shape):
    label, extra = shape
    print(f"\neffort levels, sent alongside {label}")
    print(f"  {'effort':<10} {'reasoning_tok':>14} {'out_tok':>8}  answer")
    for effort in EFFORTS:
        _, _, usage, message, error = chat(
            base, model, "What is 17*23? Number only.", 1200,
            extra={**extra, "reasoning_effort": effort}, timeout=300)
        if error:
            print(f"  {effort:<10} {'-':>14} {'-':>8}  {error}")
            continue
        think, _ = reasoning_of(message)
        completion = usage.get("completion_tokens", 0) if usage else 0
        answer = (message.get("content") or "").strip().replace("\n", " ")[:18]
        print(f"  {effort:<10} {len(think):>14} {completion:>8}  {answer!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8002/v1",
                    help="OpenAI base, e.g. http://127.0.0.1:8002/v1")
    ap.add_argument("--model", help="defaults to the first the endpoint lists")
    ap.add_argument("--sizes", default="1k,8k,32k,128k",
                    help="prompt sizes in tokens, comma separated")
    ap.add_argument("--out-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--calibrate-words", type=int, default=2000)
    ap.add_argument("--skip-sizes", action="store_true")
    ap.add_argument("--skip-thinking", action="store_true")
    a = ap.parse_args()

    listed = models(a.base)
    model = a.model or listed[0]["id"]
    window = next((m.get("max_model_len") for m in listed if m["id"] == model), None)
    print(f"model: {model}   window: {window or 'not reported'}   base: {a.base}")
    print(f"endpoint lists: {' '.join(m['id'] for m in listed)}")

    # Calibrate rather than guess: tokens per word depends on the tokenizer, and
    # this gibberish is not English.
    cal = words(a.calibrate_words, a.seed * 104729)
    _, _, usage, _, error = chat(a.base, model, cal, 1, stream=False, timeout=600)
    if error or not usage:
        sys.exit(f"calibration failed: {error}")
    tpw = usage["prompt_tokens"] / a.calibrate_words
    print(f"calibration: {a.calibrate_words} words -> {usage['prompt_tokens']} "
          f"tokens ({tpw:.2f} tok/word)")

    if not a.skip_sizes:
        size_sweep(a.base, model, [parse_size(s) for s in a.sizes.split(",")],
                   a.seed, a.out_tokens, tpw, window)
    if not a.skip_thinking:
        working = thinking_probe(a.base, model)
        if working:
            effort_sweep(a.base, model, working[0])
        else:
            print("\nno shape turned thinking on: either the model does not "
                  "reason or the server has it off")


if __name__ == "__main__":
    main()
