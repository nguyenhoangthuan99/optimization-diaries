---
layout: default
title: "vLLM vs TensorRT-LLM on one RTX Pro 6000: 1,000+ benchmark cells, no simple winner"
description: "Qwen3.5-4B/9B and Qwen3.8-27B in BF16/FP8, plus NVIDIA's official Qwen3-Next-80B NVFP4 checkpoint, on Blackwell SM120 - decode and prefill swept across batch sizes 1-128 and contexts 2K-128K on one GPU."
---

**TL;DR** - 1,036 measured cells on one 96 GB RTX Pro 6000 Blackwell:

- **Decode at big batch (BF16/FP8): TensorRT-LLM wins, +20-59%.**
- **FP8 at small/mid batch: vLLM wins, +7-18% decode, +20-25% prefill.**
- **80B NVFP4 decode: vLLM wins every matched cell (+11-35%)** - and it's
  the only engine that runs the model at batch 128 or 128K context.
- **80B NVFP4 prefill: TRT-LLM wins +4-14%** - verified fair, both engines
  unchunked.
- **4-bit support is the real divider**: vLLM served all 7 official NVFP4
  checkpoints we tried at full config; TRT-LLM fully serves 4, runs the 80B
  only in a degraded config, and can't load 2 at all.

---

## The setup, in one paragraph

SM120 (the workstation Blackwell) is not the datacenter Blackwell: 99 KB of
shared memory per SM instead of 228 KB, no `tcgen05.mma` - so every
"Blackwell-optimized" kernel has to be re-ported, and engine performance
here is a genuinely open question. We swept **Qwen3.5-4B, Qwen3.5-9B,
Qwen3.8-27B** (hybrid linear-attention family) in BF16 and FP8, plus
NVIDIA's official **Qwen3-Next-80B-A3B NVFP4** (47 GB of 4-bit weights, 3B
active per token) - on vLLM (dev build) and TensorRT-LLM 1.3.0rc25 (newest
release; the stable release can't run these architectures). Matched
configs: batch 128, 8192-token budget with chunked prefill, 90% memory,
prefix caching off, same client (`vllm bench serve`, unique random prompts
per cell), warmed-up servers, decode swept over concurrency 1-128 × output
512-8K, prefill over context 2K-128K. Raw JSONs and the harness are in this
repo.

## Decode

![TRT-LLM / vLLM decode throughput ratio heatmap](../figures/02-vllm-trtllm/ratio_heatmap.png)

- **Saturated batch (c=128), BF16/FP8: TRT-LLM +20-59%**, growing with
  model size (27B BF16: 2,063 vs 1,301 tok/s). Kernels are near-tied at
  batch 1, so the gap is the batching runtime, not math.
- **FP8, batch ≤16: vLLM +7-18%** (197 vs 167 tok/s single-stream on 4B).
  Its FP8 GEMM path on SM120 is simply better.
- **80B NVFP4: vLLM wins every matched cell, +11-35%** (145 vs 129 tok/s
  single-stream; 819 vs 662 at c=8), widening with batch and output length.
  Past TRT-LLM's batch-8 ceiling, vLLM alone reaches **4,225 tok/s at
  c=128** - an 80B model on one workstation GPU.

## Prefill

![BF16 prefill throughput vs context](../figures/02-vllm-trtllm/prefill_bf16.png)
![FP8 prefill throughput vs context](../figures/02-vllm-trtllm/prefill_fp8.png)
![NVFP4 prefill: Qwen3-Next-80B head-to-head](../figures/02-vllm-trtllm/prefill_nvfp4.png)

- Ladder BF16: **parity** within a few percent, every context and size.
- Ladder FP8: **vLLM +20-25%** at mid contexts (4B/8K: 45K vs 37K tok/s).
- 80B NVFP4: **TRT-LLM +4-14%** at concurrency 4-8, parity single-stream,
  and TTFT 6-17% lower (1.41 vs 1.48 s at 32K). We suspected its forced
  unchunked prefill explained this, so we re-ran vLLM unchunked too - the
  gap barely moved. This one is real kernel speed.
- The boundary: TRT-LLM's config caps at **64K context**; vLLM prefills
  **128K at 11.6K tok/s** (TTFT 11.7 s).

![vLLM / TRT-LLM TTFT ratio heatmap](../figures/02-vllm-trtllm/ttft_ratio_heatmap.png)

## NVFP4 on SM120: support is the story

TRT-LLM's fused-MoE kernels are tiled for the datacenter chip's 228 KB of
shared memory; on SM120's 99 KB the autotuner can run out of tactics, and
whether a model survives depends on its expert shapes. Of NVIDIA's own
official NVFP4 checkpoints that fit 96 GB: **Qwen3-32B, Gemma-4-31B,
Gemma-4-26B-A4B, and Nemotron-3.5-Lightning-30B all serve fine; Qwen3.6-27B
and Qwen3.6-35B-A3B don't load at all** (a weight-mapper bug and the
shared-memory wall, upstream #17723 and #12706); and **Qwen3-Next-80B runs
only at batch ≤8, context ≤64K, chunked prefill off** - crossing any of
those kills the server. vLLM served every one of them at full config. On
this card today, choosing 4-bit largely chooses your engine.

## Which engine for which job

| Workload | Pick | Why |
|---|---|---|
| RL rollout generation (BF16, huge batch) | **TRT-LLM** | +20-59% decode at c=128 |
| Offline batch: synthetic data, distillation, evals | **TRT-LLM** | Same saturated-batch math |
| Interactive / personal deployment (FP8, small batch) | **vLLM** | +7-18% decode, +20-25% prefill |
| Big quantized models, high concurrency | **vLLM** | Only engine at batch 128 on the 80B |
| Long context (>64K) | **vLLM** | TRT-LLM structurally capped here |
| Prefill-heavy NVFP4 ≤64K, small batch | **TRT-LLM** | +4-14% prefill, lower TTFT |
| New/fast-moving quantized checkpoints | **vLLM** | Everything loaded, day one |

**RL training rollouts** are TRT-LLM's profile in its purest form: rollouts
run in BF16 (the weights must match the trainer's), at the largest batch
the card fits, and nobody is waiting on a single request - that's the
+20-59% column. The catch is ecosystem: today's RL frameworks (verl,
OpenRLHF) integrate vLLM first for weight sync and colocation, so TRT-LLM's
raw fit is ahead of its plumbing here.

**Offline batch jobs** - generating synthetic data, distillation corpora,
bulk evals - are the same shape without the framework constraint: point
TRT-LLM at the queue, batch 128, collect the +59%.

**Personal and small-team serving** inverts everything: you run quantized
(FP8/NVFP4) to fit the card, at whatever concurrency a handful of humans
generate - the regime where vLLM wins nearly every cell, loads every
checkpoint, and supports the model that came out last Tuesday.

**The 96 GB card's headline act** - big-model 4-bit serving - belongs to
vLLM outright: it runs the 80B at 4,225 tok/s saturated and 128K context,
configurations TRT-LLM cannot currently reach on this silicon.

## Caveats

- Default-ish matched configs, no hand-tuning; treat gaps under 10% as ties.
- The 80B head-to-head is limited to cells both engines serve (c≤8, ≤32K
  prefill / ≤64K probe), because TRT-LLM survives nothing wider.
- Dev build vs release candidate; both engines are moving targets, and the
  NVFP4 MoE gaps look like young code, not physics.
- MTP/speculative decoding off; 4B/9B FP8 are community quants (coherence
  checked, accuracy not benchmarked); one GPU, no TP.

## Measurement notes

Four ways this benchmark almost shipped fake numbers - condensed so you can
spot them in the next benchmark you read:

1. **Lazy compile**: vLLM's first prefill after boot costs ~34 s vs 128 ms
   warm. Every server is warmed before measuring.
2. **Prefix-cache contamination**: fixed-seed random prompts + default
   prefix caching made vLLM prefill read 2.8× faster than TRT-LLM - above
   the hardware roofline. Caching off, unique seed per cell, cache-freeness
   verified with a repeated-prompt probe, all affected cells re-measured.
3. **Residual warmup in single-sample cells**: a batch warmup doesn't flush
   the single-request path; c=1 cells recorded a fake 4× TTFT cliff.
4. **Re-verify outliers**: every anomaly was re-measured on a fresh server
   with doubled warmup and new seeds. Most reproduced exactly (that's how
   we know TRT-LLM's NVFP4 quirks are real); one cell came back 32% faster
   and was replaced. If an outlier drives your headline, measure it twice.

Decode additionally validated by re-running one full config: mean delta
0.85%, max 5%.

## Reproduce

Harness (per-cell resumable, with overrides for the 80B's survival config),
patches, probes, and every raw JSON: `src/02-vllm-trtllm/` in this repo.

```sh
bash run_bench.sh vllm Qwen/Qwen3.5-4B 4b-bf16 0 decode
# the 80B on TRT-LLM — the only config that survives SM120:
MAXLEN_OVERRIDE=65544 MBS_OVERRIDE=8 CONCCAP_OVERRIDE=8 \
CHUNKED_OVERRIDE=false MNT_OVERRIDE=65544 \
bash run_bench.sh trtllm nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 q3next-80b-nvfp4-official 0 prefill
```
