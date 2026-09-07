#!/usr/bin/env python3
"""Blog figures for the vLLM vs TensorRT-LLM SM120 benchmark.

BF16/FP8: one figure per quant, 3 panels (model sizes) x 2 engine curves.
NVFP4: single-panel head-to-head on nvidia/Qwen3-32B-NVFP4 — the one official
NVFP4 checkpoint vanilla TRT-LLM can load on SM120 (community 4B/9B/27B
NVFP4 rows dropped per editorial decision; support matrix in the post).

Encoding contract (dataviz method):
  - color = engine (identity, fixed): vLLM blue #2a78d6, TRT-LLM aqua #1baf7a
  - linestyle/marker reinforce engine: vLLM solid/circle, TRT-LLM dashed/square
  - relief rule: both series direct-labeled in the last panel + legend present
  - one axis per chart; hairline grid; muted ink; log2 x for log2 sweeps
"""
import json, glob, re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "..", "..", "figures", "02-vllm-trtllm")
os.makedirs(OUT, exist_ok=True)

ECOLOR = {"vllm": "#2a78d6", "trtllm": "#1baf7a"}
ESTYLE = {"vllm": dict(ls="-", marker="o", label="vLLM"),
          "trtllm": dict(ls="--", marker="s", label="TRT-LLM")}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIZES = ["4b", "9b", "27b"]
SIZELBL = {"4b": "Qwen3.5-4B", "9b": "Qwen3.5-9B", "27b": "Qwen3.8-27B"}
QUANTS = ["bf16", "fp8"]
QLBL = {"bf16": "BF16", "fp8": "FP8"}
# NVFP4 head-to-head = Qwen3-Next-80B-A3B: matched cells only (c<=8, ctx<=64K).
# TRT-LLM on SM120 requires mbs<=8 + chunked prefill OFF for this model, so for
# fairness the vLLM PREFILL/TTFT cells come from an unchunked rerun (same
# config); decode uses the original tag (128-token prompts never chunk).
NVFP4_TAG = "q3next-80b-nvfp4-official"
NVFP4_PRE_VLLM_TAG = "q3next-80b-nvfp4-nochunk"
NVFP4_LBL = "Qwen3-Next-80B-A3B NVFP4 (official)"
NVFP4_CTXS = [2048, 4096, 8192, 16384, 32768]   # fair matched range (both unchunked)
NVFP4_CTXLBL = ["2K", "4K", "8K", "16K", "32K"]
NVFP4_CONCS = [1, 4, 8]
CONCS = [1, 4, 8, 16, 32, 64, 128]
CTXS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
CTXLBL = ["2K", "4K", "8K", "16K", "32K", "64K", "128K"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def load(sweep):
    d = {}
    for f in glob.glob(os.path.join(RES, "*", "*", sweep + "_*.json")):
        parts = f.split(os.sep)
        tag, eng = parts[-3], parts[-2]
        if tag.endswith("spotcheck"):
            continue
        m = re.match(r".*_in(\d+)_out(\d+)_c(\d+)\.json", parts[-1])
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if "skipped" in j:
            continue
        d[(tag, eng, int(m.group(1)), int(m.group(2)), int(m.group(3)))] = j
    return d


def style_ax(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def engine_legend(ax):
    handles = [plt.Line2D([], [], color=ECOLOR[e], ms=5,
                          **{k: v for k, v in ESTYLE[e].items() if k != "label"},
                          label=ESTYLE[e]["label"]) for e in ECOLOR]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")


def direct_label(ax, eng, x, y, dy=0):
    ax.annotate(ESTYLE[eng]["label"], xy=(x, y), xytext=(6, dy),
                textcoords="offset points", color=ECOLOR[eng],
                fontsize=9, fontweight="bold", va="center")


def direct_labels(ax, ends):
    # spread labels vertically when the two line ends nearly coincide
    if len(ends) == 2:
        (e1, (x1, y1)), (e2, (x2, y2)) = sorted(ends.items(), key=lambda kv: kv[1][1])
        close = abs(y2 - y1) < 0.08 * max(abs(y1), abs(y2), 1e-9)
        direct_label(ax, e1, x1, y1, dy=-7 if close else 0)
        direct_label(ax, e2, x2, y2, dy=7 if close else 0)
    else:
        for eng, (x, y) in ends.items():
            direct_label(ax, eng, x, y)


dec = load("decode")
pre = load("prefill")

# ------------- prefill vs context (c=8): one figure per quant -------------
for q in QUANTS:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, size in zip(axes, SIZES):
        ends = {}
        for eng in ECOLOR:
            xs, ys = [], []
            for i in CTXS:
                j = pre.get((size + "-" + q, eng, i, 8, 8))
                if j:
                    xs.append(i); ys.append(j["total_token_throughput"] / 1000)
            ax.plot(xs, ys, color=ECOLOR[eng], lw=1.8, ms=4.5, mec=SURF, mew=0.5,
                    ls=ESTYLE[eng]["ls"], marker=ESTYLE[eng]["marker"])
            if xs:
                ends[eng] = (xs[-1], ys[-1])
        if ax is axes[-1]:
            direct_labels(ax, ends)
        if not ends:
            ax.text(0.5, 0.5, "data pending", transform=ax.transAxes,
                    ha="center", color=MUTED, fontsize=10)
            ax.set_title(SIZELBL[size], fontsize=11, color=INK)
            style_ax(ax)
            continue
        ax.set_xscale("log", base=2)
        ax.set_xticks(CTXS); ax.set_xticklabels(CTXLBL)
        ax.set_title(SIZELBL[size], fontsize=11, color=INK)
        ax.set_xlabel("context length (tokens)")
        style_ax(ax)
    axes[0].set_ylabel("prefill throughput (K tokens/s)")
    handles = [plt.Line2D([], [], color=ECOLOR[e], ms=5,
                          **{k: v for k, v in ESTYLE[e].items() if k != "label"},
                          label=ESTYLE[e]["label"]) for e in ECOLOR]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=2,
               loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("%s prefill throughput vs context length (8 concurrent requests, cache-free) — 1× RTX Pro 6000"
                 % QLBL[q], fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prefill_%s.png" % q), dpi=160,
                bbox_inches="tight", facecolor=SURF)
    plt.close(fig)

# ------- NVFP4 prefill: single-panel official Qwen3-32B head-to-head -------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ends = {}
for eng in ECOLOR:
    src = NVFP4_PRE_VLLM_TAG if eng == "vllm" else NVFP4_TAG
    xs, ys = [], []
    for i in NVFP4_CTXS:
        j = pre.get((src, eng, i, 8, 8))
        if j:
            xs.append(i); ys.append(j["total_token_throughput"] / 1000)
    ax.plot(xs, ys, color=ECOLOR[eng], lw=1.8, ms=4.5, mec=SURF, mew=0.5,
            ls=ESTYLE[eng]["ls"], marker=ESTYLE[eng]["marker"])
    if xs:
        ends[eng] = (xs[-1], ys[-1])
direct_labels(ax, ends)
ax.set_xscale("log", base=2)
ax.set_xticks(NVFP4_CTXS); ax.set_xticklabels(NVFP4_CTXLBL)
ax.set_xlabel("context length (tokens)")
ax.set_ylabel("prefill throughput (K tokens/s)")
style_ax(ax)
ax.set_title("NVFP4 prefill — %s\n8 concurrent, cache-free, both engines unchunked — 1× RTX Pro 6000" % NVFP4_LBL,
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "prefill_nvfp4.png"), dpi=160,
            bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ------------- engine ratio heatmap (all head-to-head configs) -------------
tags = [s + "-" + q for s in SIZES for q in QUANTS] + [NVFP4_TAG]
TAGLBL = {s + "-" + q: "%s %s" % (SIZELBL[s].split("-")[-1], QLBL[q])
          for s in SIZES for q in QUANTS}
TAGLBL[NVFP4_TAG] = "80B MoE NVFP4*"
M = np.full((len(tags), len(CONCS)), np.nan)
for r, tag in enumerate(tags):
    for cidx, c in enumerate(CONCS):
        if tag == NVFP4_TAG and c not in NVFP4_CONCS:
            continue   # matched cells only: TRT-LLM serves this model at c<=8
        v = dec.get((tag, "vllm", 128, 2048, c))
        t = dec.get((tag, "trtllm", 128, 2048, c))
        if v and t:
            M[r, cidx] = t["output_throughput"] / v["output_throughput"]

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
fig, ax = plt.subplots(figsize=(8.2, 4.6))
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.7)
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(CONCS))); ax.set_xticklabels(CONCS)
ax.set_yticks(range(len(tags))); ax.set_yticklabels([TAGLBL[t] for t in tags])
ax.set_xlabel("concurrency")
for r in range(len(tags)):
    for cidx in range(len(CONCS)):
        if not np.isnan(M[r, cidx]):
            ax.text(cidx, r, "%.2f" % M[r, cidx], ha="center", va="center",
                    fontsize=8.5, color=INK)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("TRT-LLM ÷ vLLM output throughput", color=INK2)
cb.outline.set_visible(False)
ax.set_title("Decode: TRT-LLM ÷ vLLM throughput ratio (red = TRT-LLM faster, blue = vLLM faster)",
             fontsize=11, color=INK, pad=12)
fig.text(0.01, -0.02, "*nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 — matched cells only: TRT-LLM on SM120"
         " serves this model at batch <=8 with chunked prefill off, so c>8 has no head-to-head.",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ratio_heatmap.png"), dpi=160,
            bbox_inches="tight", facecolor=SURF)
plt.close(fig)

# ------------- TTFT ratio heatmap: vLLM / TRT-LLM at c=1 -------------
T = np.full((len(tags), len(CTXS)), np.nan)
for r, tag in enumerate(tags):
    vtag = NVFP4_PRE_VLLM_TAG if tag == NVFP4_TAG else tag
    for ci, i in enumerate(CTXS):
        v = pre.get((vtag, "vllm", i, 8, 1))
        t = pre.get((tag, "trtllm", i, 8, 1))
        if v and t:
            T[r, ci] = v["mean_ttft_ms"] / t["mean_ttft_ms"]

fig, ax = plt.subplots(figsize=(8.2, 4.6))
im = ax.imshow(T, cmap=cmap, norm=TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=1.7),
               aspect="auto")
ax.set_xticks(range(len(CTXS))); ax.set_xticklabels(CTXLBL)
ax.set_yticks(range(len(tags))); ax.set_yticklabels([TAGLBL[t] for t in tags])
ax.set_xlabel("context length")
for r in range(len(tags)):
    for ci in range(len(CTXS)):
        if not np.isnan(T[r, ci]):
            ax.text(ci, r, "%.2f" % T[r, ci], ha="center", va="center",
                    fontsize=8.5, color=INK)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("vLLM \u00f7 TRT-LLM TTFT (single request)", color=INK2)
cb.outline.set_visible(False)
ax.set_title("TTFT: vLLM \u00f7 TRT-LLM latency ratio (red = TRT-LLM faster, blue = vLLM faster)",
             fontsize=11, color=INK, pad=12)
fig.text(0.01, -0.02, "*80B NVFP4: fair range (both engines unchunked) ends at 32K;"
         " TRT-LLM itself tops out at 64K on SM120, vLLM (chunked) reaches 128K.",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ttft_ratio_heatmap.png"), dpi=160,
            bbox_inches="tight", facecolor=SURF)
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
