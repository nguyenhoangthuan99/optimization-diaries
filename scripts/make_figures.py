#!/usr/bin/env python3
"""Figures for the CPU matmul ladder post. Palette = dataviz reference instance
(light surface #fcfcfb; categorical blue/aqua/yellow; muted gray; text tokens)."""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def parse_results(path):
    """RESULT lines (key=value) -> list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "kernel=" not in line:
                continue
            row = dict(kv.split("=") for kv in line.replace("RESULT ", "").split())
            rows.append(row)
    return rows


def by_kernel(rows):
    return {r["kernel"]: float(r["gflops"]) for r in rows}


# ------------------------------------------------------------------
# Fig 1 (hero): the single-core ladder
# ------------------------------------------------------------------
lad = by_kernel(parse_results(os.path.join(DATA, "ladder.csv")))
steps = [
    ("ijk",      "0 · naive ijk"),
    ("ikj",      "1 · loop order (ikj)"),
    ("tiled",    "2 · cache blocking"),
    ("regblock", "3 · register blocking"),
    ("avx2",     "4a · AVX2 microkernel"),
    ("avx512",   "4b · AVX-512 microkernel"),
]
vals = [lad[k] for k, _ in steps]
blas1 = lad.get("blas")

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
ys = range(len(steps))
ax.barh(ys, vals, height=0.58, color=BLUE, zorder=3)
for y, v in zip(ys, vals):
    lbl = f"{v:.1f}  (×{v / vals[0]:.0f})" if v / vals[0] >= 2 else f"{v:.1f}"
    if v > 0.55 * max(vals):  # long bar: label inside, right-aligned
        ax.text(v * 0.97, y, lbl, va="center", ha="right", fontsize=9.5,
                color="#ffffff", zorder=4)
    else:
        ax.text(v * 1.08, y, lbl, va="center", fontsize=9.5, color=INK)
ax.set_yticks(ys, [lbl for _, lbl in steps], fontsize=10, color=INK)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("GFLOP/s - one core, 2048³ sgemm, same flops every step (log scale)",
              fontsize=9.5)
if blas1:
    ax.axvline(blas1, color=YELLOW, linewidth=1.6, zorder=2)
    ax.text(blas1 * 0.95, -0.62, f"OpenBLAS, 1 thread: {blas1:.0f} →",
            va="center", ha="right", fontsize=8.5, color=INK2)
ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title("Same flops, ×%d faster: the matmul ladder, one core" %
             round(vals[-1] / vals[0]),
             fontsize=12, color=INK, loc="left", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig_ladder.png"), bbox_inches="tight",
            facecolor=SURFACE)
print("fig_ladder.png")

# ------------------------------------------------------------------
# Fig 2: six loop orders - GFLOPS vs L1 miss rate (two panels, no dual axis)
# ------------------------------------------------------------------
ORDERS = ["ijk", "ikj", "jik", "jki", "kij", "kji"]


def parse_perf(path):
    """perf stat blocks -> {kernel: {event: value}}"""
    out, kern = {}, None
    with open(path) as f:
        for line in f:
            m = re.match(r"=== kernel=(\S+)", line)
            if m:
                kern = m.group(1)
                out[kern] = {}
                continue
            m = re.match(r"\s*([\d,]+)\s+(\S+)", line)
            if m and kern and not line.strip().startswith("#"):
                out[kern][m.group(2)] = int(m.group(1).replace(",", ""))
    return out


perf_path = os.path.join(DATA, "perf_1024.log")
if os.path.exists(perf_path):
    perf = parse_perf(perf_path)
    gf = {k: lad.get(k) for k in ORDERS}
    l1_per_kflop = {}
    for k in ORDERS:
        p = perf.get(k, {})
        miss = p.get("L1-dcache-load-misses")
        if miss:
            l1_per_kflop[k] = miss / (2 * 1024**3 / 1000)  # misses per kflop @1024^3

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=150)
    ys = range(len(ORDERS))
    a1.barh(ys, [gf[k] or 0 for k in ORDERS], height=0.55, color=BLUE, zorder=3)
    a1.set_yticks(ys, ORDERS, fontsize=10, color=INK)
    a1.invert_yaxis()
    a1.set_xlabel("GFLOP/s (2048³, one core)", fontsize=9)
    for y, k in zip(ys, ORDERS):
        if gf[k]:
            a1.text(gf[k] * 1.03, y, f"{gf[k]:.1f}", va="center", fontsize=9, color=INK)
    a2.barh(ys, [l1_per_kflop.get(k, 0) for k in ORDERS], height=0.55,
            color=YELLOW, zorder=3)
    a2.set_yticks(ys, ["" for _ in ORDERS])
    a2.invert_yaxis()
    a2.set_xlabel("L1d misses per 1000 flops (1024³)", fontsize=9)
    for y, k in zip(ys, ORDERS):
        v = l1_per_kflop.get(k)
        if v:
            a2.text(v * 1.03, y, f"{v:.0f}", va="center", fontsize=9, color=INK)
    for a in (a1, a2):
        a.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
        for s in ["top", "right", "left"]:
            a.spines[s].set_visible(False)
    a1.set_title("Six spellings of the same loop - speed …", fontsize=11,
                 color=INK, loc="left", pad=10)
    a2.set_title("… tracks cache misses, inversely", fontsize=11,
                 color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_loop_orders.png"), bbox_inches="tight",
                facecolor=SURFACE)
    print("fig_loop_orders.png")

# ------------------------------------------------------------------
# Fig 3: measured latency curve
# ------------------------------------------------------------------
mb = os.path.join(DATA, "membench.csv")
if os.path.exists(mb):
    ws, ns = [], []
    bw = []
    for line in open(mb):
        if line.startswith("LAT,"):
            _, b, v = line.strip().split(",")
            ws.append(int(b))
            ns.append(float(v))
        elif line.startswith("BW,"):
            _, t, v = line.strip().split(",")
            bw.append((int(t), float(v)))
    ws, ns = ws[1:], ns[1:]  # drop 4 KB point: first-measurement warmup artifact
    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=150)
    ax.plot(ws, ns, color=BLUE, linewidth=2, marker="o", markersize=4.5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [8192, 32768, 262144, 2 << 20, 16 << 20, 128 << 20, 1 << 30]
    ax.set_xticks(ticks, ["8K", "32K", "256K", "2M", "16M", "128M", "1G"], fontsize=9)
    ax.set_xlabel("working-set size (bytes)", fontsize=10)
    ax.set_ylabel("ns per dependent load", fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # cache-level regions (boundaries measured here = real Zen 4 sizes,
    # not the fictional lscpu ones)
    for x0, x1, name, lat in [
        (8192, 32768, "L1  32 KB", "1.3 ns"),
        (32768, 1 << 20, "L2  1 MB", "4 ns"),
        (1 << 20, 32 << 20, "L3  32 MB/CCX", "9-42 ns"),
        (32 << 20, 1 << 30, "DRAM", "~270 ns"),
    ]:
        mid = (x0 * x1) ** 0.5
        ax.text(mid, 400, name, ha="center", fontsize=8.6, color=INK2)
        ax.text(mid, 290, lat, ha="center", fontsize=8.2, color=MUTED)
        if x1 < (1 << 30):
            ax.axvline(x1, color=GRID, linewidth=1.0, linestyle=":", zorder=1)
    ax.set_ylim(1, 560)
    ax.set_title("Memory latency, measured: every cliff is a cache level",
                 fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_latency.png"), bbox_inches="tight",
                facecolor=SURFACE)
    print("fig_latency.png")

# ------------------------------------------------------------------
# Fig 4: thread scaling, pinned vs unpinned
# ------------------------------------------------------------------
sp = os.path.join(DATA, "scaling_pinned.log")
su = os.path.join(DATA, "scaling_unpinned.log")
if os.path.exists(sp) and os.path.exists(su):
    pin = {int(r["threads"]): float(r["gflops"]) for r in parse_results(sp)}
    unp = {int(r["threads"]): float(r["gflops"]) for r in parse_results(su)}
    ts = sorted(pin)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.plot(ts, [pin[t] for t in ts], color=BLUE, linewidth=2, marker="o",
            markersize=4.5, label="pinned (OMP_PROC_BIND=close)", zorder=3)
    ax.plot(sorted(unp), [unp[t] for t in sorted(unp)], color=AQUA, linewidth=2,
            marker="o", markersize=4.5, label="unpinned", zorder=3)
    ideal = [pin[1] * t for t in ts]
    ax.plot(ts, ideal, color=MUTED, linewidth=1.2, linestyle="--",
            label="ideal scaling", zorder=2)
    sb = os.path.join(DATA, "scaling_blas.log")
    if os.path.exists(sb):
        blas = {int(r["threads"]): float(r["gflops"]) for r in parse_results(sb)}
        bt = sorted(blas)
        ax.plot(bt, [blas[t] for t in bt], color=YELLOW, linewidth=0, marker="D",
                markersize=5, label="OpenBLAS", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    tick_ts = [1, 2, 4, 8, 16, 32, 48, 96, 144]
    ax.set_xticks(tick_ts, [str(t) for t in tick_ts], fontsize=9)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("threads", fontsize=10)
    ax.set_ylabel("GFLOP/s (4096³)", fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.annotate("beyond ~48 threads: run-to-run sd up to 40%\n(144 vCPUs on 96"
                " physical cores, shared KVM host)",
                xy=(80, 2500), fontsize=8.2, color=INK2, ha="center",
                xytext=(80, 900))
    ax.set_title("Thread scaling on 144 vCPUs - where the ideal line breaks",
                 fontsize=12, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_scaling.png"), bbox_inches="tight",
                facecolor=SURFACE)
    print("fig_scaling.png")

# ------------------------------------------------------------------
# Fig 5: tile-size sweep (small, for the tiling section)
# ------------------------------------------------------------------
tp = os.path.join(DATA, "tiles.csv")
if os.path.exists(tp):
    rows = parse_results(tp)
    xs = [int(r["TI"]) for r in rows]
    ys_ = [float(r["gflops"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.4), dpi=150)
    ax.plot(xs, ys_, color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs], fontsize=9)
    ax.set_xlabel("tile size TI = TK (TJ fixed at 512)", fontsize=10)
    ax.set_ylabel("GFLOP/s", fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Tile-size sweep: the optimum is where A-panel + B-panel fit in L2",
                 fontsize=11, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_tiles.png"), bbox_inches="tight",
                facecolor=SURFACE)
    print("fig_tiles.png")
