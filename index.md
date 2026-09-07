---
layout: default
title: Optimization Diaries
---

Technical posts where every number comes from a benchmark you can re-run.
Code, raw logs, and figures live in this repo.

## Posts

- **[vLLM vs TensorRT-LLM on one RTX Pro 6000: 1,000+ benchmark cells, no simple winner](posts/02-vllm-vs-trtllm-sm120)**
  - Qwen3.5-4B/9B and Qwen3.8-27B in BF16/FP8, plus NVIDIA's official
  Qwen3-Next-80B NVFP4 checkpoint, swept across batch 1-128 and context
  2K-128K on Blackwell SM120: TRT-LLM owns saturated-batch decode on
  BF16/FP8, vLLM owns FP8 elsewhere and NVFP4 decode - and on 4-bit the real
  story is support: TRT-LLM serves the 80B only at batch ≤8 / 64K context
  with chunked prefill off, and two of seven official NVFP4 models not at
  all. With scenario-by-scenario recommendations and the four measurement
  bugs that almost shipped fake numbers.

- **[One matmul, ×295 faster: a walk down the CPU memory hierarchy](posts/01-cpu-matmul-memory-hierarchy)**
  - the same 17 GFLOP from 0.3 to 93 GFLOP/s on one Zen 4 core, climbing one
  optimization step at a time: loop order, tiling, register blocking,
  AVX-512 microkernels, threads, and pinning - with perf-counter evidence
  and cache-line diagrams at every step.

## Reproduce

```sh
git clone https://github.com/nguyenhoangthuan99/optimization-diaries
cd optimization-diaries
make && scripts/run_ladder.sh 2048
```
