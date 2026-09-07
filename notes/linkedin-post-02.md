# LinkedIn post draft — post 02 (vLLM vs TRT-LLM on SM120)

Attach: figures/02-vllm-trtllm/social_card_02.png
Link (first comment): https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120

---

vLLM or TensorRT-LLM? Wrong question. The right one: what's your workload?

I measured both engines on a single RTX Pro 6000 Blackwell — Qwen3.5-4B/9B, Qwen3.8-27B in BF16/FP8, plus NVIDIA's official Qwen3-Next-80B NVFP4. Batch 1→128, context 2K→128K, 1,036 cells. The map:

→ RL rollout generation / offline batch (synthetic data, distillation, evals): TensorRT-LLM. BF16 at saturated batch is its home turf — +20–59% decode at batch 128, every model size. (Caveat: verl and OpenRLHF integrate vLLM first, so the raw fit is ahead of the plumbing.)

→ Personal or small-team deployment: vLLM. You'll run quantized at small batch — exactly where it wins: FP8 +7–18% decode and +20–25% prefill, and the 80B NVFP4 at +11–35% over TRT-LLM in every matched decode cell.

→ Big-model 4-bit serving: vLLM, not close. It runs the 80B at 4,225 tok/s (batch 128) and 128K context on one workstation GPU. TensorRT-LLM serves the same checkpoint only at batch ≤8, ≤64K context, chunked prefill off — its fused-MoE kernels are tiled for the datacenter chip's 228 KB shared memory; this card has 99 KB.

→ One genuine TRT-LLM NVFP4 win: prefill, +4–14%. I suspected its forced unchunked prefill explained it, re-ran vLLM unchunked too — gap barely moved. Real kernel speed, credit where due.

And the finding that surprised me most: on this card, choosing 4-bit largely chooses your engine. Of 7 official NVIDIA NVFP4 checkpoints that fit 96 GB, vLLM served all 7 at full config. TensorRT-LLM: 4 fully, 1 degraded, 2 not at all.

Four measurement bugs almost shipped fake numbers along the way — lazy first-request compile (34 s vs 128 ms), prefix-cache contamination that manufactured a 2.8× gap out of thin air, warmup residue posing as a latency cliff, and an outlier that vanished (+32%) on a fresh server. All documented in the post so you can spot them in the next benchmark you read.

Full write-up, figures, raw JSONs, reproducible harness — link in first comment.

#LLM #inference #vLLM #TensorRT #GPU #benchmarking #quantization #NVFP4 #Blackwell #RLHF

---

First comment:
Full write-up: https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120
Harness + raw results: https://github.com/nguyenhoangthuan99/optimization-diaries
