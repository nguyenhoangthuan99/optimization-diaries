---
layout: default
title: "vLLM vs TensorRT-LLM on one RTX Pro 6000: 1,000+ benchmark cells, no simple winner"
description: "Qwen3.5-4B/9B and Qwen3.8-27B in BF16/FP8, plus the official NVIDIA Qwen3-32B NVFP4 checkpoint, on Blackwell SM120 - decode and prefill swept across batch sizes 1-128 and contexts 2K-128K on one GPU."
---

*Three model sizes (4B, 9B, 27B) in BF16 and FP8, plus NVIDIA's official
NVFP4 checkpoint of Qwen3-32B, two engines (vLLM and TensorRT-LLM), one
96 GB RTX Pro 6000 Blackwell. 1,051 measured cells across decode and prefill
sweeps. The result is not "engine X wins" - it's a map of which engine wins
**where**. And for 4-bit quantization the first question isn't even speed:
it's which checkpoints TensorRT-LLM can load at all.*

---

The RTX Pro 6000 Blackwell is an interesting serving target: 96 GB of VRAM,
native NVFP4 tensor cores, workstation pricing - and an architecture (SM120)
that is *not* the datacenter Blackwell. SM100 kernels don't run on it: it has
99 KB of shared memory per SM instead of 228 KB and lacks `tcgen05.mma`, so
every "Blackwell-optimized" kernel an engine ships for B200 has to be
re-ported. That makes "which engine is faster" a genuinely open question on
this card, and for the model family we test here, nobody had published
numbers at all.

## The setup

**Models.** The size ladder is one family, three sizes, so the scaling story
is clean: Qwen3.5-4B, Qwen3.5-9B, and Qwen3.8-27B. All three are *hybrid
linear attention* models - 3 of every 4 layers use gated-DeltaNet-style
linear attention with constant-size state, only the remaining quarter are
ordinary full attention. This matters twice: the KV cache stays small at
long contexts, and the unusual architecture stresses each engine's
least-tested code paths.

| | BF16 | FP8 |
|---|---|---|
| 4B | Qwen/Qwen3.5-4B | lovedheart/Qwen3.5-4B-FP8 |
| 9B | Qwen/Qwen3.5-9B | lovedheart/Qwen3.5-9B-FP8 |
| 27B | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B-FP8 (official) |

For NVFP4 we use NVIDIA's own published checkpoint,
**nvidia/Qwen3-32B-NVFP4** - a classic full-attention dense transformer
(native 40K context), not a hybrid. Why that specific model is the NVFP4
head-to-head - and not one of the ladder models - is a support-matrix story
told below.

**Engines.** vLLM (recent dev build, `vllm/vllm-openai` image) and TensorRT-LLM
**1.3.0rc25**. The RC is not optional: stable TRT-LLM (1.2.1) does not know
the hybrid architecture at all - Qwen3.5/3.8 support landed only in the
1.3.0 release-candidate line. Two more things worth knowing if you last
touched TRT-LLM a year ago: since 1.2 the TensorRT engine backend is *gone* -
there is no `trtllm-build`, no engine files; the PyTorch backend is the only
backend and `trtllm-serve` loads HuggingFace checkpoints directly. And NVIDIA
still doesn't list this GPU in the officially-tested hardware matrix.

**Matched configs.** Both engines: max batch 128, in-flight token budget
8192 (vLLM `--max-num-batched-tokens` = TRT-LLM `max_num_tokens`, both with
chunked prefill), max sequence length 136K (40K for Qwen3-32B, its native
window), 90% GPU memory fraction, prefix caching **off** (see the
measurement notes at the end), no speculative decoding, no MTP. One GPU, one
model instance, same benchmark client (`vllm bench serve`, random dataset,
`--ignore-eos`) for both engines.

## A note on NVFP4 support: TensorRT-LLM is the bottleneck

vLLM loaded every NVFP4 checkpoint we pointed it at. Vanilla TensorRT-LLM
did not - of NVIDIA's own official NVFP4 exports in this size class, it
loads exactly one on this GPU:

| Official NVFP4 checkpoint | vLLM | TensorRT-LLM 1.3.0rc25 |
|---|---|---|
| nvidia/Qwen3.6-27B-NVFP4 (hybrid dense) | ✅ | ❌ weight-mapper shape mismatch (upstream issue #17723) |
| nvidia/Qwen3-32B-NVFP4 (classic dense) | ✅ | ✅ |
| nvidia/Qwen3.6-35B-A3B-NVFP4 (hybrid MoE) | ✅ | ❌ fused-MoE kernel wants more shared memory than SM120 has |

Community ModelOpt NVFP4 quants of the hybrid family (which, unlike
NVIDIA's conservative exports, also quantize the attention projections)
crash TRT-LLM's weight mapper in yet another way: it can't split the 0-dim
per-tensor scalars across the fused QKV projection. That one we fixed with a
~20-line mapper patch (in this repo under `src/engine-bench/patches/`), and
validated it properly: on 100 prompts, the first greedy token from patched
TRT-LLM matches vLLM serving the same checkpoint 87% of the time versus a
99% BF16 cross-engine baseline, with every disagreement a low-margin tie -
quantization noise through two kernel stacks, not mis-applied scales.

The practical takeaway is short: **on SM120 today, NVFP4 model support is a
vLLM strength and a TRT-LLM weakness.** If your quantized checkpoint of
choice loads in TRT-LLM, benchmark it; but don't assume it will load. So the
NVFP4 performance comparison below is the one official checkpoint both
engines serve: Qwen3-32B-NVFP4.

## Decode: TRT-LLM owns saturated batch — except in NVFP4

Each cell: fixed 128-token prompt, forced 2048-token generation, one wave of
N concurrent requests, N from 1 to 128. (Output lengths 512-8192 were also
swept; for the hybrid models TPOT is flat across them, as expected with a
constant-size linear-attention state.)

One heatmap summarizes all 49 head-to-head decode cells - the ratio of
TRT-LLM to vLLM output throughput (red: TRT-LLM ahead, blue: vLLM ahead):

![TRT-LLM / vLLM decode throughput ratio heatmap](../figures/02-vllm-trtllm/ratio_heatmap.png)

Three patterns:

1. **At saturated batch (c=128) on BF16/FP8, TRT-LLM wins everywhere: +20%
   to +59%.** The gap grows with model size (27B BF16: 2,063 vs 1,301
   tok/s). Since the kernels are near-tied at batch 1, this is runtime -
   scheduler and batching machinery - not math.
2. **In the low-to-mid range, vLLM wins wherever FP8 is involved** (up to
   +18% single-stream: 197 vs 167 tok/s on 4B). Its FP8 GEMM path on SM120
   is simply better. On BF16 the engines are within a few percent until
   c=32, TRT-LLM slightly ahead.
3. **The NVFP4 row breaks the pattern: vLLM wins at every concurrency.**
   Single-stream it's 61 vs 34 tok/s (+79%), and at c=4 TRT-LLM collapses
   to 0.27× - 63 tok/s, barely above its own single-stream rate, as if the
   batch path doesn't engage below 8 concurrent requests (the numbers are
   consistent across all five output lengths, so this is systematic, not
   noise). From c=8 up TRT-LLM recovers to 0.83-0.93× but never catches up,
   even at c=128 (3,707 vs 3,973 tok/s). TRT-LLM's saturated-batch
   superpower on this card evidently doesn't extend to its SM120 NVFP4
   path yet.

The headline single-GPU number: **Qwen3-32B NVFP4 decodes at 3,973 tok/s
at 128 concurrent requests, 61 tok/s single-stream** (vLLM; peak 5,139
tok/s at shorter outputs). A 32-billion-parameter dense model, one
workstation GPU.

## Prefill: mostly a tie

Each cell: N concurrent requests of a fixed context length (2K → 128K,
concurrency capped at 8 for 64K/128K), 8 output tokens, cache disabled,
unique prompts.

![BF16 prefill throughput vs context](../figures/02-vllm-trtllm/prefill_bf16.png)
![FP8 prefill throughput vs context](../figures/02-vllm-trtllm/prefill_fp8.png)
![NVFP4 prefill: Qwen3-32B head-to-head](../figures/02-vllm-trtllm/prefill_nvfp4.png)

The batched-prefill picture is mostly **parity** (measured cache-free - see
measurement note 2): BF16 matches within 1-2K tok/s at every context and
size. The exceptions are consistent and worth knowing:

- **FP8 prefill belongs to vLLM** - +20-25% at mid contexts (4B/8K: 45K vs
  37K; 9B/8K: 31K vs 25K). Same FP8 GEMM advantage as decode.
- **NVFP4 prefill (32B) crosses over**: vLLM is well ahead at 2K (14.0K vs
  9.8K tok/s), the engines tie at 8K, and TRT-LLM pulls ahead at 32K
  (6.9K vs 6.1K, +12%).
- On the hybrid models, both engines peak near 8-16K context and lose
  throughput toward 128K as the quadratic quarter of the attention layers
  takes over - the hybrid architecture softens the fall (at 128K you keep
  ~40-55% of peak). The full-attention 32B falls faster, as full attention
  does.

Time-to-first-token, single request, as the all-cells ratio view - vLLM
over TRT-LLM, so red again means TRT-LLM is ahead (lower latency):

![vLLM / TRT-LLM TTFT ratio heatmap](../figures/02-vllm-trtllm/ttft_ratio_heatmap.png)

Short contexts are a wash (both engines ~90-160 ms at 2K). At long context
TRT-LLM has a mild systematic edge on single-request latency - on the 32B
NVFP4 it grows from nothing at 2K to +15% at 32K (4.67 s vs 5.37 s), and
the FP8 hybrids tilt the other way, following the prefill-throughput story.

## When to use what

The map, condensed into scenarios:

| Your workload | Pick | Why |
|---|---|---|
| High-QPS serving, BF16 or FP8 (batch ≥64) | **TRT-LLM** | +20-59% decode at c=128, all sizes |
| Latency-sensitive FP8 (batch ≤16) | **vLLM** | +7-18% decode, +20-25% prefill |
| Anything NVFP4 | **vLLM** | Loads every checkpoint; on the 32B head-to-head, wins decode at every concurrency (up to 3.7× at c=4) |
| Long-context, prefill-dominated (≥16K), BF16/NVFP4 | **TRT-LLM** (slightly) | +10-15% prefill throughput and TTFT at 32K |
| Interactive single-stream, BF16 | **TRT-LLM** (slightly) | A few percent ahead at c=1 |
| Broad model/quant coverage, fast-moving checkpoints | **vLLM** | Served everything we tried; TRT-LLM needed an RC for the architecture and loads 1 of 3 official NVFP4 models |

And across precisions, from the ladder data: **FP8 is close to a free
upgrade over BF16 on this card** - on the 27B, single-stream decode goes
from 26 to 46 tok/s (+77%) and saturated batch from 2,063 to 2,664 tok/s
on TRT-LLM, with prefill up ~15-25% too. If your quality budget allows
4-bit weights, the 32B NVFP4 numbers show what the format can do on this
GPU's native tensor cores - but as of today that choice also largely picks
your engine for you (vLLM), because checkpoint support, not speed, is the
binding constraint.

Two caveats on reading the table: the NVFP4 row comes from one model that
is architecturally different from the ladder (classic full attention vs
hybrid), so it bundles "NVFP4 on SM120" with "this architecture"; and
everything here is *default-ish matched configs* - neither engine was
hand-tuned, so treat small gaps (<10%) as ties.

## Caveats, honestly

- Both engines ran **default-ish, matched** configs. Neither was hand-tuned;
  TRT-LLM's high-batch win and vLLM's FP8 win might each shrink with
  engine-specific tuning. Defaults are what most deployments run, so that's
  what we measured.
- vLLM is a dev build; TRT-LLM is a release candidate (the stable release
  can't run the hybrid models). Both engines are moving targets - the NVFP4
  gaps in particular look like young-code gaps, not physics.
- MTP/speculative decoding disabled on both - the hybrid models ship MTP
  heads, and enabling them is a different (interesting) benchmark.
- The 4B/9B FP8 checkpoints are community quants; we verified they load and
  generate coherently, not their benchmark accuracy.
- The NVFP4 comparison is one model (Qwen3-32B) because that's the one
  official NVFP4 checkpoint both engines serve on SM120 - n=1, different
  architecture from the ladder, native context capped at 40K.
- One GPU, one model instance. TP/multi-GPU behavior may reorder things.

## Measurement notes

Three bugs we caught during this benchmark produced plausible-looking but
wrong numbers. Condensed here so you can check for them in benchmarks you
read (or run):

1. **Lazy compile on first prefill.** vLLM's first real prefill after server
   start costs ~34 s (compile), the same request warm costs 128 ms. Every
   server is warmed up before anything is measured.
2. **Prefix-cache contamination.** Fixed-seed random prompts recur across
   cells (and share leading tokens across lengths); with vLLM's default
   prefix caching on, batched prefill read 2.8x faster than TRT-LLM - fake,
   and above the BF16 roofline. Caching is disabled on both engines, every
   cell gets a unique seed, and we verified cache-freeness by timing a
   repeated 16K prompt (repeat/fresh ratio: 1.04 vLLM, 0.99 TRT-LLM). All
   contaminated cells were re-measured.
3. **Single-sample cells eat residual warmup.** A concurrency-8 warmup does
   not flush the single-request path; c=1 cells (one request each) recorded
   the leftover ~250 ms one-time cost as a fake 4x TTFT cliff at 2K context.
   The harness warms both paths, and the 18 affected cells were re-measured
   (true values: 85-130 ms).

Decode cells were additionally validated by re-running one full config under
the final harness: mean delta 0.85%, max 5% on the shortest cell.

## Reproduce

The harness (server lifecycle + sweeps + per-cell resumable JSON results),
the TRT-LLM weight-mapper patch, the cache-probe script, the cross-engine
agreement checkers, and every raw result JSON are in this repo under
`src/engine-bench/`. Total: roughly 17 hours of GPU time across the sweeps
and re-runs.

```sh
# one cell, by hand:
bash run_bench.sh vllm Qwen/Qwen3.5-4B 4b-bf16 0 decode
# the official 32B NVFP4 head-to-head:
MAXLEN_OVERRIDE=40960 bash run_bench.sh trtllm nvidia/Qwen3-32B-NVFP4 q3-32b-nvfp4-official 0 decode
```
