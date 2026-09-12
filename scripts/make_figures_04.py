#!/usr/bin/env python3
"""Figures for post 04 (W4A16 NVFP4 quantization fidelity, QAT vs QAD).
Palette matches the series style: light surface, categorical blue/aqua/yellow, muted gray."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figures", "04-w4a16-qad")
os.makedirs(FIGS, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#d1495b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
})

# ------------------------------------------------------------------
# Fig 1: precision-map sweep -- WikiText KL vs export size, PTQ only
# ------------------------------------------------------------------
configs = [
    ("Uniform\nW4A16", 0.07687, 2.639),
    ("Keep first/last\nblocks BF16", 0.05315, 2.909),
    ("Keep K/V\nBF16", 0.06990, 2.892),
    ("Keep Q/K/V +\nboundary BF16\n(selected)", 0.04272, 3.616),
]
labels = [c[0] for c in configs]
kls = [c[1] for c in configs]
sizes = [c[2] for c in configs]

fig, ax1 = plt.subplots(figsize=(8.6, 4.6), dpi=150)
xs = np.arange(len(configs))
colors = [MUTED, MUTED, MUTED, BLUE]
bars = ax1.bar(xs, kls, width=0.5, color=colors, zorder=3)
ax1.set_ylabel("WikiText KL(teacher \u2016 PTQ)", color=INK2)
ax1.set_xticks(xs)
ax1.set_xticklabels(labels, fontsize=9.5)
ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ("top", "right"):
    ax1.spines[spine].set_visible(False)

for x, v, s in zip(xs, kls, sizes):
    ax1.text(x, v + 0.0018, f"{v:.5f}", ha="center", fontsize=9.5, color=INK, weight="bold")
    ax1.text(x, -0.006, f"{s:.2f} GiB", ha="center", fontsize=8.5, color=MUTED)

ax1.set_ylim(-0.012, 0.088)
ax1.annotate("44.4% lower KL\nthan uniform W4A16",
             xy=(3, 0.04272), xytext=(2.15, 0.062),
             fontsize=9.5, color=INK, weight="bold",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))

ax1.set_title("Spending the BF16 budget: where precision goes matters more than how much",
              fontsize=12, color=INK, weight="bold", loc="left", pad=14)
fig.text(0.01, -0.02,
         "Real packed NVFP4 PTQ sweep, full-FP32 PyTorch KL evaluator, Jan-v3.5-4B, WikiText held-out.",
         fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_precision_map.png"), bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------
# Fig 2: PTQ vs QAD vs QAT -- 20-cell mean KL and 32K mean KL
# (QAD/QAT use different references; annotated clearly)
# ------------------------------------------------------------------
arms = ["W4A16 PTQ\n(QKV-compatible)", "W4A16 QAD\n(teacher \u2016 Q2)", "W4A16 QAT\n(Q1 \u2016 Q2 drift)"]
mean20 = [0.02625, 0.01982, 0.02318]
mean32k = [0.02328, 0.01801, 0.02105]

fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
xs = np.arange(len(arms))
w = 0.32
colors20 = [MUTED, BLUE, YELLOW]
b1 = ax.bar(xs - w/2, mean20, width=w, color=colors20, zorder=3, label="Mean KL, 20 domain/length cells")
b2 = ax.bar(xs + w/2, mean32k, width=w, color=colors20, alpha=0.55, zorder=3, label="Mean KL at 32K")

for x, v in zip(xs - w/2, mean20):
    ax.text(x, v + 0.0006, f"{v:.5f}", ha="center", fontsize=8.8, color=INK)
for x, v in zip(xs + w/2, mean32k):
    ax.text(x, v + 0.0006, f"{v:.5f}", ha="center", fontsize=8.8, color=INK2)

ax.set_xticks(xs)
ax.set_xticklabels(arms, fontsize=9.5)
ax.set_ylabel("KL (lower is better)", color=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_ylim(0, 0.030)

ax.annotate("-24.5% vs PTQ", xy=(1 - w/2, 0.01982), xytext=(0.55, 0.0255),
            fontsize=10, color=BLUE, weight="bold",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

ax.set_title("QAD vs matched PTQ: 24.5% lower teacher-relative KL",
              fontsize=12, color=INK, weight="bold", loc="left", pad=14)
fig.text(0.01, -0.04,
         "QAD metric = KL(pristine teacher \u2016 Q2). QAT metric = KL(Q1_QAT \u2016 Q2_QAT), a drift score vs its\n"
         "own fine-tuned checkpoint, not the same reference as QAD/PTQ \u2014 shown for the workflow comparison only.",
         fontsize=8.3, color=MUTED)
ax.legend(fontsize=8.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_ptq_qad_qat.png"), bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------
# Fig 3: domain x length matrix -- QAD vs QAT, all 20 cells, line plot
# ------------------------------------------------------------------
lengths = [4, 8, 16, 32]
qad = {
    "Code":     [0.01674, 0.01604, 0.01529, 0.01483],
    "Math":     [0.01515, 0.01418, 0.01301, 0.01227],
    "Alpaca":   [0.01844, 0.01542, 0.01353, 0.01243],
    "Dolly":    [0.02158, 0.02075, 0.02021, 0.01968],
    "OpenOrca": [0.03709, 0.03523, 0.03358, 0.03084],
}
qat = {
    "Code":     [0.01811, 0.01717, 0.01628, 0.01576],
    "Math":     [0.01554, 0.01455, 0.01346, 0.01263],
    "Alpaca":   [0.02184, 0.01905, 0.01698, 0.01549],
    "Dolly":    [0.02525, 0.02432, 0.02370, 0.02309],
    "OpenOrca": [0.04615, 0.04418, 0.04178, 0.03827],
}
domain_colors = {
    "Code": BLUE, "Math": AQUA, "Alpaca": YELLOW, "Dolly": "#8856a7", "OpenOrca": RED,
}

fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
for dom, color in domain_colors.items():
    ax.plot(lengths, qad[dom], color=color, marker="o", linewidth=2.2, zorder=4, label=f"{dom} (QAD)")
    ax.plot(lengths, qat[dom], color=color, marker="o", linewidth=1.3, linestyle="--",
            alpha=0.55, zorder=3, label=f"{dom} (QAT)")

ax.set_xticks(lengths)
ax.set_xticklabels(["4K", "8K", "16K", "32K"])
ax.set_xlabel("Evaluation context length", color=INK2)
ax.set_ylabel("KL (lower is better)", color=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

ax.set_title("QAD is lower in all 20/20 domain\u00d7length cells \u2014 and KL falls as context grows",
              fontsize=11.5, color=INK, weight="bold", loc="left", pad=14)
fig.text(0.01, -0.06,
         "Solid = QAD (teacher \u2016 Q2). Dashed = QAT (Q1 \u2016 Q2 drift, different reference).\n"
         "QAD recovery used only 1,024-token sequences; the decreasing trend at 4K\u201332K is evaluated, not trained, context.",
         fontsize=8.3, color=MUTED)
ax.legend(fontsize=7.6, frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_domain_length_matrix.png"), bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------------------
# Fig 4: side experiment -- size vs output throughput, BF16/FP8/NVFP4
# ------------------------------------------------------------------
reps = ["BF16", "FP8\nblock-128", "W4A16\nNVFP4"]
size_gib = [7.49, 4.806, 3.614]
tput = [1368, 1507, 1881]
rep_colors = [MUTED, YELLOW, BLUE]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150)

axa.bar(reps, size_gib, color=rep_colors, width=0.55, zorder=3)
for i, v in enumerate(size_gib):
    axa.text(i, v + 0.15, f"{v:.2f} GiB", ha="center", fontsize=9.5, color=INK, weight="bold")
axa.set_ylabel("Checkpoint size (GiB)", color=INK2)
axa.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
axa.set_axisbelow(True)
for spine in ("top", "right"):
    axa.spines[spine].set_visible(False)
axa.set_ylim(0, 8.6)
axa.set_title("Smallest checkpoint", fontsize=10.5, color=INK, weight="bold", loc="left")

axb.bar(reps, tput, color=rep_colors, width=0.55, zorder=3)
for i, v in enumerate(tput):
    axb.text(i, v + 30, f"{v:,} tok/s", ha="center", fontsize=9.5, color=INK, weight="bold")
axb.set_ylabel("Output throughput @ concurrency 32", color=INK2)
axb.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
axb.set_axisbelow(True)
for spine in ("top", "right"):
    axb.spines[spine].set_visible(False)
axb.set_ylim(0, 2150)
axb.set_title("Highest serving throughput", fontsize=10.5, color=INK, weight="bold", loc="left")

fig.suptitle("Side experiment: NVFP4 is the smallest and fastest-serving option in this vLLM setup",
             fontsize=11.5, color=INK, weight="bold", x=0.01, ha="left", y=1.03)
fig.text(0.01, -0.05,
         "vLLM, one RTX PRO 6000 Blackwell, explicit warmup, 2,048 input / 128 output tokens, concurrency 32.\n"
         "This is deployment context after the correctness result above, not the main finding of the post.",
         fontsize=8.3, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_size_speed.png"), bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print("Wrote figures to", FIGS)
