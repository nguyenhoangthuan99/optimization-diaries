#!/usr/bin/env python3
"""Summarize engine-bench results into a flat table / CSV."""
import json, glob, sys

sweep = sys.argv[1] if len(sys.argv) > 1 else "smoke"
csv = "--csv" in sys.argv

rows = []
for f in sorted(glob.glob(f"*/*/{sweep}_*.json")):
    tag, engine, fname = f.split("/")
    try:
        d = json.load(open(f))
    except Exception as e:
        rows.append({"tag": tag, "engine": engine, "file": fname, "err": str(e)[:40]})
        continue
    if "skipped" in d:
        rows.append({"tag": tag, "engine": engine, "file": fname, "err": "skipped:" + d["skipped"]})
        continue
    import re
    m = re.match(r".*_in(\d+)_out(\d+)_c(\d+)\.json", fname)
    fin, fout, fconc = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
    rows.append({
        "tag": tag, "engine": engine, "file": fname,
        "in": fin, "out": fout, "conc": fconc,
        "req_tput": d.get("request_throughput", 0),
        "out_tput": d.get("output_throughput", 0),
        "total_tput": d.get("total_token_throughput", 0),
        "ttft_mean": d.get("mean_ttft_ms", 0),
        "ttft_p99": d.get("p99_ttft_ms", 0),
        "tpot_mean": d.get("mean_tpot_ms", 0),
        "itl_p99": d.get("p99_itl_ms", 0),
        "e2el_mean": d.get("mean_e2el_ms", 0),
    })

if csv:
    keys = ["tag", "engine", "in", "out", "conc", "req_tput", "out_tput",
            "total_tput", "ttft_mean", "ttft_p99", "tpot_mean", "itl_p99", "e2el_mean"]
    print(",".join(keys))
    for r in rows:
        if "err" in r:
            continue
        print(",".join(str(r.get(k, "")) for k in keys))
else:
    hdr = "{:12} {:7} {:>7} {:>6} {:>5} {:>10} {:>10} {:>9} {:>9}"
    print(hdr.format("config", "engine", "in", "out", "conc", "out tok/s", "tot tok/s", "TTFT ms", "TPOT ms"))
    for r in rows:
        if "err" in r:
            print("{:12} {:7} {}  [{}]".format(r["tag"], r["engine"], r["file"], r["err"]))
            continue
        print(hdr.format(r["tag"], r["engine"], str(r["in"]), str(r["out"]), str(r["conc"]),
                         "{:.1f}".format(r["out_tput"]), "{:.1f}".format(r["total_tput"]),
                         "{:.1f}".format(r["ttft_mean"]), "{:.2f}".format(r["tpot_mean"])))
