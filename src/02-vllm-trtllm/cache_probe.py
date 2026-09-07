#!/usr/bin/env python3
"""Prefix-cache probe: same prompt twice + control. Cached => repeat TTFT collapses."""
import json, sys, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
url = "http://localhost:%s/v1/completions" % port

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango ").split()

def prompt(salt):
    # ~16k tokens of varied words, differing by salt
    toks = []
    for i in range(16000):
        toks.append(WORDS[(i * 7 + salt * 13) % len(WORDS)])
    return "%d " % salt + " ".join(toks)

def ttft(p):
    body = json.dumps({"model": model, "prompt": p, "max_tokens": 1,
                       "temperature": 0}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
    return (time.time() - t0) * 1000

warm = ttft(prompt(99))  # throwaway: absorb any first-shape compile
a = ttft(prompt(1))      # fresh prompt P
b = ttft(prompt(1))      # SAME prompt P again
c = ttft(prompt(2))      # different prompt, same length
print("warmup: %.0f ms" % warm)
print("fresh P : %.0f ms" % a)
print("repeat P: %.0f ms   (cached if <<)" % b)
print("fresh Q : %.0f ms" % c)
verdict = "CACHING DETECTED" if b < 0.5 * min(a, c) else "no caching"
print("verdict: %s (repeat/fresh ratio %.2f)" % (verdict, b / min(a, c)))
