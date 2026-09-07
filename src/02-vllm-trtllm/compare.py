#!/usr/bin/env python3
"""Cross-engine comparison tables from engine-bench results. Run from results/."""
import json, glob, re

TAGS = ["4b-bf16", "4b-fp8", "4b-nvfp4", "9b-bf16", "9b-fp8", "9b-nvfp4",
        "27b-bf16", "27b-fp8", "27b-nvfp4"]

def load(sweep):
    out = {}
    for f in glob.glob("*/*/%s_*.json" % sweep):
        tag, eng, fname = f.split("/")
        m = re.match(r".*_in(\d+)_out(\d+)_c(\d+)\.json", fname)
        if not m:
            continue
        key = (tag, eng, int(m.group(1)), int(m.group(2)), int(m.group(3)))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "skipped" in d:
            continue
        out[key] = d
    return out

dec = load("decode")
pre = load("prefill")

print("=== DECODE (in=128, out=2048): output tok/s, vLLM vs TRT-LLM ===")
hdr = "config     " + " | ".join("c=%-3d  v      t   " % c for c in (1, 16, 128))
print(hdr)
for tag in TAGS:
    cells = []
    for c in (1, 16, 128):
        v = dec.get((tag, "vllm", 128, 2048, c))
        t = dec.get((tag, "trtllm", 128, 2048, c))
        vs = "%6.0f" % v["output_throughput"] if v else "     -"
        ts = "%6.0f" % t["output_throughput"] if t else "     -"
        cells.append("      %s %s" % (vs, ts))
    print("%-10s" % tag + " | ".join(cells))

print()
print("=== PREFILL single-request TTFT ms (ctx/TTFT = prefill tok/s) ===")
print("config     " + " | ".join("in=%-6d v      t   " % i for i in (8192, 32768, 131072)))
for tag in TAGS:
    cells = []
    for i in (8192, 32768, 131072):
        v = pre.get((tag, "vllm", i, 8, 1))
        t = pre.get((tag, "trtllm", i, 8, 1))
        vs = "%6.0f" % v["mean_ttft_ms"] if v else "     -"
        ts = "%6.0f" % t["mean_ttft_ms"] if t else "     -"
        cells.append("   %s %s" % (vs, ts))
    print("%-10s" % tag + " | ".join(cells))
