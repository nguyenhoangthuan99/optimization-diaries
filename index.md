---
layout: default
title: Performance, measured
---

Technical posts where every number comes from a benchmark you can re-run.
Code, raw logs, and figures live in this repo.

## Posts

- **[One matmul, ×295 faster: a walk down the CPU memory hierarchy](posts/01-cpu-matmul-memory-hierarchy)**
  - the same 17 GFLOP from 0.3 to 93 GFLOP/s on one Zen 4 core, climbing one
  optimization step at a time: loop order, tiling, register blocking,
  AVX-512 microkernels, threads, and pinning - with perf-counter evidence
  and cache-line diagrams at every step.

## Reproduce

```sh
git clone https://github.com/nguyenhoangthuan99/performance-measured
cd performance-measured
make && scripts/run_ladder.sh 2048
```
