# LinkedIn post draft — post 02 (vLLM vs TRT-LLM on SM120)

Attach: figures/02-vllm-trtllm/social_card_02.png
Link (first comment): https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120

---

vLLM or TensorRT-LLM? I put both on a single RTX Pro 6000 Blackwell and measured — Qwen3.5-4B/9B and Qwen3.8-27B in BF16/FP8, plus NVIDIA's official Qwen3-32B NVFP4 checkpoint. Batch 1→128, context 2K→128K. 1,051 measured cells.

There's no simple winner. There's a map:

→ Saturated-batch decode on BF16/FP8: TensorRT-LLM, +20–59% at batch 128, every size. Near-tied kernels at batch 1 — the gap is the batching runtime.
→ FP8 at low/mid batch: vLLM, +7–18% decode and +20–25% prefill. Its FP8 GEMMs on SM120 are just better.
→ NVFP4: vLLM wins every single concurrency on the official Qwen3-32B checkpoint — +79% single-stream, and at batch 4 TRT-LLM drops to 0.27× (its batch path doesn't seem to engage below 8 concurrent). 3,973 tok/s from a 32B dense model on one workstation GPU.
→ Prefill everywhere else: parity, within a few percent.

But the NVFP4 story isn't really about speed. It's about support: of NVIDIA's own three official NVFP4 checkpoints in this class, vanilla TensorRT-LLM loads exactly one on this GPU (the hybrid dense model hits an upstream weight-mapper bug, the MoE wants more shared memory than SM120 has). vLLM loaded all three. Right now, choosing 4-bit on this card largely chooses your engine for you.

Also in the post: three measurement bugs that almost shipped fake numbers — lazy first-request compile (33.8 s vs 128 ms warm), prefix-cache contamination that manufactured a 2.8× engine gap out of thin air, and single-sample cells recording leftover warmup as a fake latency cliff. All caught, all re-measured, all documented so you can spot them in the next benchmark you read.

If a benchmark's prefill numbers beat the hardware roofline, ask what the prompt generator's seed policy was.

Full write-up with figures, scenario-by-scenario engine recommendations, raw JSONs, and a reproducible harness — link in the first comment.

#LLM #inference #vLLM #TensorRT #GPU #benchmarking #quantization #NVFP4 #Blackwell

---

First comment:
Full write-up: https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120
Harness + raw results: https://github.com/nguyenhoangthuan99/optimization-diaries
