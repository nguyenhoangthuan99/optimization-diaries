---
layout: default
title: Optimization Diaries
---

Technical posts where every number comes from a benchmark you can re-run.
Code, raw logs, and figures live in this repo.

## Posts

- **[vLLM vs TensorRT-LLM on one RTX Pro 6000: 1,300 benchmark cells, two bugs, no simple winner](posts/02-vllm-vs-trtllm-sm120)**
  - Qwen3.5-4B/9B and Qwen3.8-27B in BF16/FP8/NVFP4, swept across batch 1-128
  and context 2K-128K on Blackwell SM120: TRT-LLM owns saturated-batch decode,
  vLLM owns FP8, NVFP4 wins every cell - plus the TRT-LLM weight-loader patch
  and the two measurement bugs (lazy compile, prefix-cache contamination) that
  almost shipped fake numbers.

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
