#!/usr/bin/env python3
"""LinkedIn/X card for post 01: hero number + mini ladder. 1200x675."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figures")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"

fig = plt.figure(figsize=(8, 4.5), dpi=150)
fig.patch.set_facecolor(SURFACE)

# ---- left: the hero stat ----
axl = fig.add_axes((0.0, 0.0, 0.46, 1.0))
axl.axis("off")
axl.text(0.09, 0.78, "One matmul,", fontsize=23, color=INK, weight="bold")
axl.text(0.09, 0.50, "×295", fontsize=68, color=BLUE, weight="bold")
axl.text(0.09, 0.36, "faster.", fontsize=23, color=INK, weight="bold")
axl.text(0.09, 0.24, "Same flops, zero removed.", fontsize=13, color=INK2)
axl.text(0.09, 0.17, "A measured walk down the\nCPU memory hierarchy.",
         fontsize=13, color=INK2, va="top")

# ---- right: the mini ladder ----
axr = fig.add_axes((0.52, 0.14, 0.44, 0.72))
steps = [
    ("naive ijk", 0.317),
    ("loop order", 20.7),
    ("tiling", 42.5),
    ("AVX-512\nmicrokernel", 93.4),
]
vals = [v for _, v in steps]
ys = range(len(steps))
axr.barh(ys, vals, height=0.6, color=BLUE, zorder=3)
for y, (name, v) in enumerate(steps):
    lbl = f"{v:g}" if v < 1 else f"{v:.0f}"
    if v > 50:
        axr.text(v * 0.94, y, lbl, va="center", ha="right", fontsize=13,
                 color="#ffffff", weight="bold", zorder=4)
    else:
        axr.text(v * 1.15, y, lbl, va="center", fontsize=13, color=INK,
                 weight="bold")
axr.set_yticks(ys, [s for s, _ in steps], fontsize=11.5, color=INK)
axr.invert_yaxis()
axr.set_xscale("log")
axr.set_xlim(0.25, 130)
axr.set_xticks([])
axr.set_xticks([], minor=True)
axr.set_title("GFLOP/s, one Zen 4 core, 2048³ sgemm", fontsize=11,
              color=INK2, loc="left", pad=8)
for s in axr.spines.values():
    s.set_visible(False)

fig.text(0.52, 0.045, "nguyenhoangthuan99.github.io/performance-measured",
         fontsize=9.5, color=MUTED)

fig.savefig(os.path.join(FIGS, "social_card.png"), facecolor=SURFACE)
print("social_card.png")
