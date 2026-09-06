# Optimization Diaries

Technical posts on systems performance where every number comes from a
benchmark you can re-run. Read them at
**[nguyenhoangthuan99.github.io/optimization-diaries](https://nguyenhoangthuan99.github.io/optimization-diaries/)**.

| post | topic |
|---|---|
| [01 - One matmul, ×295 faster](posts/01-cpu-matmul-memory-hierarchy.md) | CPU memory hierarchy: the matmul optimization ladder |
| *(next)* | GPU kernel optimization |

---

## Post 01 - the CPU matmul ladder

A single-file matrix-multiplication "optimization ladder" where every step
does the exact same 2·M·N·K flops and only the order the bytes move in
changes:

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

## Build & run

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
