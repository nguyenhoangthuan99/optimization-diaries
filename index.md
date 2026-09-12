---
layout: default
title: Optimization Diaries
---

Technical posts where every number comes from a benchmark you can re-run.
Code, raw logs, and figures live in this repo.

## Posts

- **[Before Serving Fast: QAT vs QAD for BF16-Faithful NVFP4](posts/04-w4a16-nvfp4-qad-vllm)**
  - Choose a vLLM-compatible W4A16 precision map, compare QAT during fine-tuning with post-training QAD, and then measure FP8/BF16/NVFP4 size and serving trade-offs.

- **[An FP32 matmul from 8% to 83% of cuBLAS: a walk down the GPU memory hierarchy](posts/03-cuda-matmul-blackwell)**
  - the same 2·M·N·K flops from 6.59 to 65 TFLOP/s on one RTX Pro 6000 Blackwell,
  climbing the GPU hierarchy one step at a time: naive → shared-memory tiling →
  register tiling → cp.async double-buffering → the register retune that made it
  compute-bound — with ncu profiler counters (bank conflicts 34M→3.6M, SM
  throughput 54.8%→73.2%) and a cross-check against the canonical siboehm
  kernel, both landing at ~83-84% of cuBLAS.

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
