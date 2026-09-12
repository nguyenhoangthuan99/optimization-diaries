# LinkedIn draft — Optimization Diaries 04

Plain-text body below. Not posted.

---

Before making an LLM serve faster, I want to check what quantization changed.

This is the next step after my CUDA matmul study: moving from kernel performance to model fidelity.

I used Jan-v3.5-4B to explore a practical question: can we preserve a BF16 model's behavior in a vLLM-compatible NVFP4 checkpoint?

The work had two parts:

1. Choose where to spend precision.

Uniform W4A16 gave 0.07687 WikiText KL. Keeping Q/K/V and the first/last transformer blocks in BF16 brought that to 0.04272—44.4% lower, before recovery training.

The runtime mattered too: vLLM's fused-QKV path required Q, K, and V to share a precision. A good quantization map also has to load and run.

2. Test how to recover the remaining error.

I trained QAT and QAD independently from the same BF16 checkpoint: cross-entropy with quantizers active versus distillation from a frozen teacher.

For the matched W4A16 map, QAD reduced mean teacher-relative KL from 0.02625 to 0.01982, a 24.5% reduction across five domains and four context lengths.

One interesting observation: recovery used 1,024-token sequences, but QAD's average per-token KL decreased at every evaluated length from 4K to 32K in every domain. Encouraging transfer beyond the training window—not proof of better long-context task accuracy.

QAD also had lower reported KL than QAT in all 20 cells. An important caveat: QAT was measured against its own fine-tuned BF16 reference, while QAD was measured against the original teacher. Those numbers are not a same-reference head-to-head win.

The workflow question is whether to incorporate quantization during fine-tuning or keep BF16 training and add a later recovery stage. These results support QAD as a useful post-training recovery option; they don't settle that question universally.

I also compared size and serving speed with FP8 and BF16. NVFP4 was smaller and faster in the measured setup; FP8 stayed closer to the teacher. That was a side experiment, not the main result.

My takeaway: choose the precision map, measure fidelity on held-out data, test recovery, then optimize serving. Lower KL is useful evidence—not a replacement for downstream evaluation.

Full write-up:
https://nguyenhoangthuan99.github.io/optimization-diaries/posts/04-w4a16-nvfp4-qad-vllm

#LLM #Quantization #NVFP4 #Inference #MachineLearning
