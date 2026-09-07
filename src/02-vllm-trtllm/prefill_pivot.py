#!/usr/bin/env python3
"""Prefill throughput pivot: total tok/s by context x concurrency, per config x engine."""
import json, glob, re

data = {}
tags = set()
for f in glob.glob("*/*/prefill_*.json"):
    tag, eng, fname = f.split("/")
    m = re.match(r".*_in(\d+)_out\d+_c(\d+)\.json", fname)
    if not m:
        continue
    d = json.load(open(f))
    if "skipped" in d:
        continue
    data[(tag, eng, int(m.group(1)), int(m.group(2)))] = d["total_token_throughput"]
    tags.add((tag, eng))

CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
CONC = [1, 8, 128]
for tag, eng in sorted(tags):
    print("%s / %s  (prefill total tok/s; c=128 n/a >=64k)" % (tag, eng))
    print("  ctx:    " + "".join("%9d" % i for i in CTX))
    for c in CONC:
        vals = []
        for i in CTX:
            v = data.get((tag, eng, i, c))
            vals.append("%8.0fk" % (v / 1000) if v else "        -")
        print("  c=%-4d" % c + "".join(vals))
    print()
