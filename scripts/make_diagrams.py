#!/usr/bin/env python3
"""Explainer diagrams for the matmul ladder post: how each optimization
changes what the cache sees. Dataviz reference palette; text wears text
tokens, marks carry the identity."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
BLUE_PALE, YELLOW_PALE, AQUA_PALE = "#c4dcf5", "#fbe7b8", "#bfe9d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
})

G = 8          # grid cells per side
LINE = 4       # cells per illustrated "cache line"


def draw_matrix(ax, x0, y0, name, fills=None, lines=False, sub=None):
    """8x8 matrix at (x0,y0), size 8x8 units. fills: {(r,c): color}.
    lines=True draws cache-line boxes (1x4 groups, row-major)."""
    fills = fills or {}
    for r in range(G):
        for c in range(G):
            col = fills.get((r, c), SURFACE)
            ax.add_patch(Rectangle((x0 + c, y0 + G - 1 - r), 1, 1,
                                   facecolor=col, edgecolor=GRID, linewidth=0.6))
    if lines:
        for r in range(G):
            for c0 in range(0, G, LINE):
                ax.add_patch(Rectangle((x0 + c0, y0 + G - 1 - r), LINE, 1,
                                       fill=False, edgecolor=MUTED,
                                       linewidth=1.0))
    ax.add_patch(Rectangle((x0, y0), G, G, fill=False, edgecolor=INK2,
                           linewidth=1.2))
    ax.text(x0 + G / 2, y0 + G + 0.45, name, ha="center", fontsize=11,
            color=INK)
    if sub:
        ax.text(x0 + G / 2, y0 - 0.75, sub, ha="center", fontsize=8.0,
                color=INK2)


def arrow(ax, x, y, dx, dy, color=INK):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.05, head_width=0.32,
                            head_length=0.35, color=color,
                            length_includes_head=True))


# ------------------------------------------------------------------
# Diagram 1: ijk vs ikj — what the inner loop does to cache lines
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.6), dpi=150)
for ax in axes:
    ax.set_xlim(-1, 31)
    ax.set_ylim(-2.6, 10.4)
    ax.set_aspect("equal")
    ax.axis("off")

# --- top: ijk, inner loop k, fixed i=2, j=3 ---
ax = axes[0]
i_fix, j_fix = 2, 3
fa = {(i_fix, c): BLUE for c in range(G)}                    # A row i, streamed
fb = {(r, j_fix): YELLOW for r in range(G)}                  # B col j: bad
for r in range(G):                                           # wasted line bytes
    for c in range((j_fix // LINE) * LINE, (j_fix // LINE) * LINE + LINE):
        if c != j_fix:
            fb[(r, c)] = YELLOW_PALE
fc = {(i_fix, j_fix): AQUA}
draw_matrix(ax, 0, 0, "A", fa, sub="row i: unit stride ✓")
draw_matrix(ax, 11, 0, "B", fb, lines=True,
            sub="column j: stride N — new cache line every k ✗")
draw_matrix(ax, 22, 0, "C", fc, sub="one cell, accumulates")
arrow(ax, 0.5, G - 1 - i_fix + 0.5, 7, 0)                    # A row arrow
arrow(ax, 11 + j_fix + 0.5, 8.35, 0, -7.8, color=INK)        # B col arrow (down)
ax.text(15, -2.2,
        "ijk — inner loop over k:  every B access lands in a different 64-byte line;"
        " 1 float of 16 is used before eviction",
        ha="center", fontsize=9.5, color=INK)
ax.text(-0.6, 10.1, "rung 0 · ijk", fontsize=12, color=INK, weight="bold")

# --- bottom: ikj, inner loop j, fixed i=2, k=1 ---
ax = axes[1]
k_fix = 1
fa = {(i_fix, k_fix): BLUE}
fb = {(k_fix, c): BLUE for c in range(G)}
fc = {(i_fix, c): AQUA for c in range(G)}
draw_matrix(ax, 0, 0, "A", fa, sub="one value → register")
draw_matrix(ax, 11, 0, "B", fb, lines=True,
            sub="row k: unit stride — full lines consumed ✓")
draw_matrix(ax, 22, 0, "C", fc, sub="row i: unit stride ✓")
arrow(ax, 11.5, G - 1 - k_fix + 0.5, 7, 0)                   # B row arrow
arrow(ax, 22.5, G - 1 - i_fix + 0.5, 7, 0)                   # C row arrow
ax.text(15, -2.2,
        "ikj — inner loop over j:  A[i][k] is a scalar in a register; B and C stream"
        " line-by-line — prefetchable and vectorizable",
        ha="center", fontsize=9.5, color=INK)
ax.text(-0.6, 10.1, "rung 1 · ikj", fontsize=12, color=INK, weight="bold")

fig.suptitle("Same three loops, two spellings — thin boxes are 64-byte cache lines",
             fontsize=12.5, color=INK, x=0.06, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(FIGS, "diag_ijk_vs_ikj.png"), bbox_inches="tight",
            facecolor=SURFACE)
print("diag_ijk_vs_ikj.png")

# ------------------------------------------------------------------
# Diagram 2: tiling — the B tile stays hot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=150)
for ax in axes:
    ax.set_xlim(-1, 31)
    ax.set_ylim(-4.2, 10.6)
    ax.set_aspect("equal")
    ax.axis("off")

# untiled: whole B streamed per row of A
ax = axes[0]
fa = {(i_fix, c): BLUE for c in range(G)}
fb = {(r, c): YELLOW_PALE for r in range(G) for c in range(G)}
fc = {(i_fix, c): AQUA for c in range(G)}
draw_matrix(ax, 0, 0, "A", fa)
draw_matrix(ax, 11, 0, "B", fb, sub="16 MB — cannot stay cached")
draw_matrix(ax, 22, 0, "C", fc)
ax.text(15, -3.0, "ikj without tiles: every row of C re-streams ALL of B\n"
        "from DRAM — B is fetched N times over", ha="center", fontsize=9.5,
        color=INK)
ax.text(-0.6, 10.2, "rung 1 · no tiles", fontsize=11.5, color=INK, weight="bold")

# tiled
ax = axes[1]
ti, tj, tk = 4, 4, 4  # illustrated tile
fa = {(r, c): (BLUE if c < tk else BLUE_PALE) for r in range(ti) for c in range(G)}
fb = {}
for r in range(G):
    for c in range(G):
        fb[(r, c)] = YELLOW if (r < tk and c < tj) else SURFACE
fc = {(r, c): (AQUA if c < tj else AQUA_PALE) for r in range(ti) for c in range(G)}
draw_matrix(ax, 0, 0, "A", fa, sub="panel: TI rows")
draw_matrix(ax, 11, 0, "B", fb, sub="TK×TJ tile — fits in L2, stays hot")
draw_matrix(ax, 22, 0, "C", fc, sub="TI×TJ tile updated")
ax.text(15, -3.0, "tiled: the same B tile is reused by every one of the TI rows\n"
        "before moving on — B traffic drops by ~TI×", ha="center", fontsize=9.5,
        color=INK)
ax.text(-0.6, 10.2, "rung 2 · tiled", fontsize=11.5, color=INK, weight="bold")

fig.suptitle("Tiling: shrink the working set until the reuse actually happens in cache",
             fontsize=12.5, color=INK, x=0.05, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(FIGS, "diag_tiling.png"), bbox_inches="tight",
            facecolor=SURFACE)
print("diag_tiling.png")

# ------------------------------------------------------------------
# Diagram 3: the 6x16 register microkernel
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=150)
ax.set_xlim(-4.5, 22)
ax.set_ylim(-3.2, 11.5)
ax.set_aspect("equal")
ax.axis("off")

# B row: two vector registers on top
for v in range(2):
    x0 = 3 + v * 8.4
    ax.add_patch(Rectangle((x0, 9.2), 8, 1, facecolor=BLUE,
                           edgecolor=SURFACE, linewidth=1.5))
    ax.text(x0 + 4, 9.7, f"b{v} = B[k][j..j+{'7' if v == 0 else '15'}]",
            ha="center", va="center", fontsize=8.4, color="#ffffff")
ax.text(1.4, 9.7, "2 loads / k", ha="right", fontsize=9, color=INK2)

# A column: six broadcast scalars on the left
for r in range(6):
    y0 = 7.4 - r * 1.3
    ax.add_patch(Rectangle((-3.4, y0), 1, 1, facecolor=YELLOW,
                           edgecolor=SURFACE, linewidth=1.5))
    ax.text(-2.9, y0 + 0.5, "", ha="center", va="center")
ax.text(-2.9, 8.9, "A[i+r][k]\nbroadcast", ha="center", fontsize=8.4, color=INK2)

# C tile: 6 rows x 2 register columns
for r in range(6):
    y0 = 7.4 - r * 1.3
    for v in range(2):
        x0 = 3 + v * 8.4
        ax.add_patch(Rectangle((x0, y0), 8, 1, facecolor=AQUA,
                               edgecolor=SURFACE, linewidth=1.5))
    ax.text(19.9, y0 + 0.5, f"row {r}", va="center", fontsize=8, color=MUTED)
ax.text(11.2, -0.6,
        "C tile: 6 rows × 16 floats = 12 vector registers — loaded once,\n"
        "12 FMAs per k step, stored once after the whole K panel",
        ha="center", fontsize=9.5, color=INK)
for r in range(6):  # arrows from A scalars into C rows
    y0 = 7.4 - r * 1.3 + 0.5
    arrow(ax, -2.3, y0, 4.6, 0, color=MUTED)
arrow(ax, 7, 9.1, 0, -0.6, color=MUTED)
arrow(ax, 15.4, 9.1, 0, -0.6, color=MUTED)

ax.set_title("Rung 4 — the microkernel: keep the hottest data in registers, not cache",
             fontsize=12.5, color=INK, loc="left", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "diag_microkernel.png"), bbox_inches="tight",
            facecolor=SURFACE)
print("diag_microkernel.png")
