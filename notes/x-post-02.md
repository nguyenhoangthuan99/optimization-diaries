# X (Twitter) post draft — post 02 (vLLM vs TRT-LLM on SM120)

Attach: figures/02-vllm-trtllm/social_card_02.png
Link goes in the last tweet (links in the first tweet get down-ranked).

---

## Option A — single post (fits in one tweet)

vLLM or TensorRT-LLM? Wrong question — what's your workload?

1,036 cells on one RTX Pro 6000:
• RL rollouts / batch jobs (BF16, batch 128): TRT-LLM +20–59%
• Personal quantized serving: vLLM
• 80B NVFP4: vLLM 4,225 tok/s — TRT-LLM can't go past batch 8

Write-up + raw data in reply 👇

---

## Option B — thread (more engagement, tells the support story)

**1/6**
vLLM or TensorRT-LLM on a workstation Blackwell (RTX Pro 6000)?

I measured 1,036 benchmark cells: 3 model sizes in BF16/FP8 + NVIDIA's official Qwen3-Next-80B NVFP4. Batch 1→128, context 2K→128K.

There's no winner. There's a map 🧵

**2/6**
Decode at saturated batch (BF16/FP8): TensorRT-LLM, +20–59% at batch 128, every model size.

Kernels are near-tied at batch 1 — the gap is pure batching runtime. If you do RL rollouts or offline batch generation, this is your column.

**3/6**
Small-batch quantized serving: vLLM.

FP8: +7–18% decode, +20–25% prefill.
80B NVFP4: +11–35% in every cell both engines can run — and past TRT-LLM's ceiling it hits 4,225 tok/s at batch 128. An 80B model. One workstation GPU.

**4/6**
The real 4-bit story is support, not speed.

7 official NVIDIA NVFP4 checkpoints fit in 96 GB.
vLLM served 7/7 at full config.
TRT-LLM: 4 fully, the 80B only at batch ≤8 / 64K ctx / chunked prefill off, and 2 not at all — its fused-MoE kernels want 228 KB of shared memory; this chip has 99 KB.

**5/6**
One genuine TRT-LLM NVFP4 win: prefill, +4–14%.

I suspected its forced unchunked prefill explained it, so I re-ran vLLM unchunked too. Gap barely moved. Real kernel speed — credit where due.

**6/6**
Also in the post: 4 measurement bugs that almost shipped fake numbers, including prefix-cache contamination that manufactured a 2.8× engine gap out of thin air (the numbers beat the hardware roofline — that's the tell).

Write-up, figures, raw JSONs, harness:
https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120

---

Reply for Option A:
Full write-up: https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120
Harness + raw results: https://github.com/nguyenhoangthuan99/optimization-diaries
