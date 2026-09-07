# LinkedIn post draft — post 02 (vLLM vs TRT-LLM on SM120)

Attach: figures/02-vllm-trtllm/social_card_02.png
Link (first comment): https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120

---

vLLM or TensorRT-LLM? I put both on a single RTX Pro 6000 Blackwell and measured — 3 models (Qwen3.5-4B/9B, Qwen3.8-27B), 3 precisions (BF16, FP8, NVFP4), batch 1→128, context 2K→128K. ~1,300 cells.

There's no simple winner. There's a map:

→ Saturated-batch decode: TensorRT-LLM, +20–59% at batch 128, every size, every precision. Same kernels-speed at batch 1 — the gap is the batching runtime.
→ FP8 at low/mid batch: vLLM, +7–18%. Its FP8 GEMMs on SM120 are just better.
→ Prefill: parity, within a few percent almost everywhere.
→ NVFP4 wins literally every cell, on both engines. Qwen3.8-27B: 56 tok/s single-stream, 3,120 tok/s at batch 128 — on one workstation GPU.

Getting there wasn't plug-and-play:

• Stable TensorRT-LLM can't run this model family at all — hybrid linear-attention support only exists in the 1.3.0 RC line. And the old engine-build workflow is gone entirely: PyTorch backend only, loads HF checkpoints directly.
• TRT-LLM crashed loading every NVFP4 checkpoint (its weight mapper can't split the 0-dim scalar scales in ModelOpt checkpoints across fused QKV). A ~20-line patch fixed all nine configs — as far as I can tell, these are the first published TRT-LLM numbers for Qwen3.8-27B on this GPU.
• And three measurement bugs nearly shipped fake numbers — lazy first-request compile (33.8s vs 128ms warm), prefix-cache contamination that manufactured a 2.8× engine gap out of thin air, and single-sample cells recording leftover warmup as a fake latency cliff. All caught, all re-measured, all documented in the post so you can spot them in the next benchmark you read.

If a benchmark's prefill numbers beat the hardware roofline, ask what the prompt generator's seed policy was.

Full write-up with figures, the patch, raw JSONs, and a reproducible harness — link in the first comment.

#LLM #inference #vLLM #TensorRT #GPU #benchmarking #quantization #NVFP4 #Blackwell

---

First comment:
Full write-up: https://nguyenhoangthuan99.github.io/optimization-diaries/posts/02-vllm-vs-trtllm-sm120
Harness + raw results: https://github.com/nguyenhoangthuan99/optimization-diaries
