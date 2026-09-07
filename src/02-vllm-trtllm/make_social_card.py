#!/usr/bin/env python3
"""Social card / thumbnail for post 02 (1200x630)."""
import json, glob, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "..", "..", "figures", "02-vllm-trtllm")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURF = "#fcfcfb"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"

SIZES = ["4b", "9b", "27b"]
QUANTS = ["bf16", "fp8"]
NVFP4_TAG = "q3next-80b-nvfp4-official"
CONCS = [1, 4, 8, 16, 32, 64, 128]

dec = {}
for f in glob.glob(os.path.join(RES, "*", "*", "decode_*out2048*.json")):
    parts = f.split(os.sep)
    tag, eng = parts[-3], parts[-2]
    if tag.endswith("spotcheck"):
        continue
    c = int(re.search(r"_c(\d+)", parts[-1]).group(1))
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if "skipped" not in j:
        dec[(tag, eng, c)] = j["output_throughput"]

tags = [s + "-" + q for s in SIZES for q in QUANTS] + [NVFP4_TAG]
lbls = [t.replace("-", " ").upper() for t in tags[:-1]] + ["80B NVFP4"]
M = np.full((len(tags), len(CONCS)), np.nan)
for r, tag in enumerate(tags):
    for ci, c in enumerate(CONCS):
        v, t = dec.get((tag, "vllm", c)), dec.get((tag, "trtllm", c))
        if v and t:
            M[r, ci] = t / v

fig = plt.figure(figsize=(12, 6.3), facecolor=SURF)

# left: text block
axt = fig.add_axes([0.045, 0.05, 0.52, 0.9]); axt.axis("off")
axt.text(0, 0.93, "vLLM  vs  TensorRT-LLM", fontsize=31, fontweight="bold",
         color=INK, va="top")
axt.text(0, 0.80, "on one RTX Pro 6000 Blackwell (SM120)", fontsize=17,
         color=INK2, va="top")
axt.text(0, 0.66, "4 models · BF16 / FP8 / NVFP4 · batch 1–128 · context 2K–128K",
         fontsize=13, color=MUTED, va="top")
axt.text(0, 0.52, "1,036", fontsize=44, fontweight="bold", color=BLUE, va="top")
axt.text(0.31, 0.485, "measured cells,\nno simple winner", fontsize=15,
         color=INK2, va="top")
axt.text(0, 0.27, "Qwen3-Next-80B NVFP4 (official, vLLM):", fontsize=13.5, color=MUTED, va="top")
axt.text(0, 0.20, "4,225 tok/s", fontsize=30, fontweight="bold", color=YELLOW,
         va="top")
axt.text(0.44, 0.185, "on a single GPU", fontsize=14, color=INK2, va="top")
axt.text(0, 0.045, "optimization diaries · 02", fontsize=11, color=MUTED, va="top")

# right: mini heatmap
axh = fig.add_axes([0.62, 0.14, 0.355, 0.74])
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#e34948"])
norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.7)
axh.imshow(M, cmap=cmap, norm=norm, aspect="auto")
axh.set_xticks(range(len(CONCS)))
axh.set_xticklabels(CONCS, fontsize=9, color=MUTED)
axh.set_yticks(range(len(tags)))
axh.set_yticklabels(lbls, fontsize=9, color=MUTED)
axh.set_xlabel("concurrency", fontsize=10, color=INK2)
axh.set_title("who wins decode  (red = TRT-LLM, blue = vLLM)",
              fontsize=11, color=INK2, pad=8)
for s in axh.spines.values():
    s.set_visible(False)

fig.savefig(os.path.join(OUT, "social_card_02.png"), dpi=100, facecolor=SURF)
print("wrote", os.path.join(OUT, "social_card_02.png"))
