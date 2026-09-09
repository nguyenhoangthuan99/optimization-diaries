# Optimization Diaries

Technical posts on systems performance where every number comes from a
benchmark you can re-run. Read them at
**[nguyenhoangthuan99.github.io/optimization-diaries](https://nguyenhoangthuan99.github.io/optimization-diaries/)**.

| post | topic |
|---|---|
| [03 - An FP32 matmul from 8% to 83% of cuBLAS](posts/03-cuda-matmul-blackwell.md) | GPU memory hierarchy: shared-memory tiling, register tiling, cp.async double-buffering, and the register retune that made it compute-bound |
| [01 - One matmul, ×295 faster](posts/01-cpu-matmul-memory-hierarchy.md) | CPU memory hierarchy: the matmul optimization ladder |
| [02 - vLLM vs TensorRT-LLM on one RTX Pro 6000](posts/02-vllm-vs-trtllm-sm120.md) | 1,036-cell engine benchmark on Blackwell SM120: BF16/FP8 ladder + the official Qwen3-Next-80B NVFP4 head-to-head |

Repo layout: per-post sources in `src/<post>/`, figures in
`figures/<post>/`, post text in `posts/`.

---

## Post 02 - vLLM vs TensorRT-LLM on SM120

Everything lives in `src/02-vllm-trtllm/`: the benchmark harness, the
TRT-LLM weight-mapper patch, cache/agreement probes, figure scripts, and
every raw result JSON under `results/<tag>/<engine>/`.

`run_bench.sh` runs one (engine, model, sweep) lane: starts the server
in Docker with matched configs, warms both the batch and single-request
paths, then measures per-cell resumable JSONs (decode: concurrency 1-128 ×
output 512-8K; prefill: context 2K-128K, cache off, unique seeds per cell).

```sh
# one cell / one lane, by hand:
bash run_bench.sh vllm Qwen/Qwen3.5-4B 4b-bf16 0 decode

# the 80B NVFP4 on TRT-LLM — the only config that survives SM120
# (batch <=8, chunked prefill off, token budget >= longest prompt):
MAXLEN_OVERRIDE=65544 MBS_OVERRIDE=8 CONCCAP_OVERRIDE=8 \
CHUNKED_OVERRIDE=false MNT_OVERRIDE=65544 \
bash run_bench.sh trtllm nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 q3next-80b-nvfp4-official 0 prefill

# regenerate every figure + the social card from the raw JSONs:
python3 make_figures.py && python3 make_social_card.py
```

Overrides: `MAXLEN_OVERRIDE` (context window), `MBS_OVERRIDE` (max batch),
`CONCCAP_OVERRIDE` (skip cells above a concurrency), `CHUNKED_OVERRIDE`
(TRT-LLM chunked prefill), `MNT_OVERRIDE` (in-flight token budget),
`MOE_BACKEND` (TRT-LLM `moe_config.backend`). The measurement-bug story
(lazy compile, prefix-cache contamination, residual warmup, outlier
re-verification) is in the post's measurement notes.

## Post 01 - the CPU matmul ladder

A single-file matrix-multiplication "optimization ladder"
(`src/01-cpu-matmul/`) where every step does the exact same 2·M·N·K flops
and only the order the bytes move in changes:

| step | kernel | technique |
|---|---|---|
| 0 | `ijk` | naive triple loop |
| 1 | `ikj` (+4 others) | loop-order / access-pattern |
| 2 | `tiled` | cache blocking |
| 3 | `regblock` | register blocking (4×16 C tile in locals, plain C) |
| 4 | `avx2` / `avx512` | SIMD microkernel (6×16 / 6×32, C tile in registers) |
| 5 | `omp` | threads over row-blocks |
| 6 | *(run config)* | thread pinning: `OMP_PROC_BIND=close OMP_PLACES=cores` |
| - | `blas` | OpenBLAS `sgemm` ceiling |

Plus `membench`, which measures the memory hierarchy the post talks about:
a dependent pointer-chase latency curve (4 KB → 1 GB working set) and a
streaming-bandwidth thread sweep.

### Build & run

```sh
make                       # gcc, -O3 -march=native -fopenmp; links OpenBLAS if present
./matmul -l                # list kernels
./matmul -k avx512 -v -n 5 2048 2048 2048    # validate + benchmark one step
./membench                 # latency curve + bandwidth sweep (CSV to stdout)

scripts/run_ladder.sh 2048   # every step, single core, -> data/ladder.csv
scripts/run_tiles.sh 2048    # tile-size sweep for the tiled step
scripts/run_scaling.sh 4096  # thread sweep, pinned vs unpinned, + BLAS
scripts/run_perf.sh 1024     # perf stat cache counters per step
```

Every kernel is validated against a double-precision reference
(`-v`, error normalized by output RMS) - on odd, non-square shapes, because
that's where microkernel edge handling breaks.

Benchmarks report the **median of N runs** with standard deviation; single-core
runs are pinned with `taskset`. Numbers in the post: AMD EPYC 9454 (Zen 4,
KVM guest, 144 vCPUs), gcc 13.3, `-O3 -march=native` throughout - every
speedup on the ladder comes from code structure, not compiler flags.
