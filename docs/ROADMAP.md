# Roadmap

The goal is a complete, self-contained stack: convert a model once, then execute
it with no Python and no framework in the token loop.

```
  HF checkpoint ──▶ whetstone convert ──▶ model.wstone ──▶ whetstone run
      (bf16)          quantize + pack                       native engine
                          ✅ done                              ✅ done
```

## The bar to clear

Measured on the reference GPU, same model, `llama-bench`:

| engine | format | bytes/token | decode (tg384) |
|---|---|---|---|
| HuggingFace | fp16 | 988 MB | 36.8 tok/s |
| **llama.cpp** | **Q4_K_M** | 392 MB | **282.95 ± 3.61** |
| **Whetstone** | **int4 `.wstone`** | **262 MB** | **431.8 tok/s** |

llama.cpp is the real competitor, not HuggingFace. Whetstone's structural edge
is **1.49× fewer bytes per token**, and the measured speed ratio is **1.53×** —
the roofline said the two should track each other, and they do. Every stage
below is judged against llama.cpp, not against the HuggingFace baseline.

The cost is quality: int4-g128 round-to-nearest is **+4.2 perplexity** against
fp16 on this model (18.0287 vs 13.8209). Closing that, not going faster, is now
the top of the list.

Status keys: **done** · *in progress* · planned

---

## Stage 1 — Measurement and ground truth  **done**

Nothing gets optimized before it can be measured, and nothing is trusted before
it is checked against an independent implementation.

- **done** `whetstone probe` — measures every arithmetic path on the actual GPU
  rather than trusting the spec sheet
- **done** Roofline model, unit-tested against real parameter counts
- **done** safetensors loader hardened against truncation, overlap and bad shapes
- **done** Independent fp64 numpy reference, written from `config.json` alone
- **done** Byte-level BPE tokenizer read straight from `tokenizer.json`
- **done** Baseline harness: tok/s, TTFT, achieved bandwidth, wikitext-2 perplexity

## Stage 2 — The weight pipeline  **done**

- **done** int4 group-128 asymmetric quantizer and packer
- **done** `.wstone` container: aligned, checksummed, self-describing
- **done** `whetstone convert` — 988 MB → 263.6 MB, mean relative error 0.110
- **done** `whetstone verify` — integrity and fidelity
- **done** int4 decode GEMV kernel, differential-tested against its dequantized
  reference, 1.50× fp16 wall-clock

## Stage 3 — The executor  **done**

- **done** RMSNorm, fp32 reduction, single pass when the vector fits one element
  per thread
- **done** RoPE, half-rotation layout, cos/sin tables precomputed in f64
- **done** Decode attention, GQA, online softmax, split across the sequence as
  well as the heads
- **done** SwiGLU, consuming a fused `gate|up` projection
- **done** On-device argmax; host-side temperature/top-p
- **done** Model graph and weight residency in `whetstone-core`
- **done** `whetstone run`, `whetstone ppl`, `whetstone logits`, `whetstone tune`
- **done** `--body fp16` lossless reference path, which is what separates an
  engine bug from quantization damage

**Acceptance, met:** the fp16 path gives wikitext-2 perplexity **13.8209**
against HuggingFace's **13.8182** over the same 40,940 predictions — 0.02%.

## Stage 4 — Removing the overhead  **done, and it taught us the model was wrong**

- **done** Static execution plan: every buffer allocated once, nothing per token
- **done** Fused `q|k|v` and `gate|up` projections, at load time so the file
  format is unchanged
- **done** CUDA graph capture of the whole decode step
- **done** Bias and the residual add folded into GEMV epilogues

The predicted win here was 1.5–3× from removing launch overhead. **Measured, the
CUDA graph was worth 0.5%** — 247 launches collapsed into 1 and the token rate
did not move, because the CPU was already running far ahead of the GPU and the
launches were entirely hidden. The reasoning that produced the prediction
(launch count × launch latency) treats a latency as an occupancy cost.

What *did* pay, and why, is in
`research/notes/2026-07-28-executor-built-and-tuned.md`. The short version: a
cheaper int4 dequant (an OR into an fp16 mantissa instead of a quarter-rate
`I2F`), fusing the projections that share an input, and splitting attention
across the sequence because 14 query heads cannot fill 30 SMs. 1.51× in total,
at bit-identical output.

## Stage 4b — Depth pruning  planned, cheap to evaluate

Removing layers cuts bytes/token proportionally, so unlike cheaper arithmetic it
attacks the actual bottleneck. Dropping 6 of 24 layers is ~22% fewer bytes.

Published methods (ShortGPT, LaCo, Gromov et al.) report 25–45% of layers
removable on **7B+** models with modest degradation, selected by angular distance
between a layer's input and output, then healed with a short LoRA finetune.

The caveat is the one that killed ternary here: those results are on large
models. Qwen2.5-0.5B has 24 layers and much less redundancy.

- planned Per-layer ablation using the existing fp64 reference — skip each layer,
  measure output KL, rank importance. **~1 hour, no GPU, no kernels.** Do this
  before committing to the idea.
- planned Drop the flattest layers, measure perplexity and top-1 agreement
- planned Healing finetune, if the un-healed degradation is close but not close enough

## Stage 5 — Beyond int4  *next, and now the top of the list*

Stages 3 and 4 are done and were lossless. This is where the accuracy budget
gets spent, and the measurements say it is already overspent:

| what is quantized | bits/wt | wikitext-2 ppl | Δ |
|---|---|---|---|
| nothing | 16.00 | 13.8209 | — |
| transformer blocks | 7.49 | 16.5696 | **+2.75** |
| blocks + `lm_head` | 4.25 | 18.0287 | **+4.21** |

int4-g128 round-to-nearest is not free, and the earlier "100% top-1 agreement"
finding was measured on three prompts — enough to miss a 2.75-perplexity
regression entirely, because the argmax stays stable long after the distribution
has moved. **Recovering that 4.2 is now worth more than another 1.5× of speed.**

- planned GPTQ with adequate calibration (~262k tokens; the earlier sweep used
  293, leaving every Hessian rank-deficient and the result inconclusive)
- **done** `--head int4` as an opt-in switch. Its cost is now measured: +1.46 ppl
  for 1.76× fewer bytes.
- planned fp16 top-k re-score so the argmax stays exact under a quantized head
- planned GPTQ with adequate calibration (~262k tokens; the current sweep used
  293, leaving every Hessian rank-deficient and the result inconclusive)
- planned AWQ activation-aware scaling
- planned Hadamard/rotation preprocessing, to test whether int3 becomes viable
- planned int8 KV cache
- planned Sensitivity-aware bit allocation (`v_proj` is consistently the worst
  tensor to quantize — measured across all 24 layers)

## Stage 6 — Speculative decoding  planned

Lossless: the output distribution is provably identical. It also converts the
memory-bound decode regime into a compute-bound verification regime, which is
the one regime where the measured 610 TOPS binary tensor-core path could matter.

- planned n-gram / prompt-lookup drafting (no draft model needed)
- planned Tree verification
- planned Dynamic acceptance-rate guard so the worst case is 1.0×, never a
  regression

## Stage 7 — Packaging  planned

- planned PyO3 bindings — `import whetstone`, with Python kept out of the token loop
- planned Prebuilt wheels
- planned OpenAI-compatible HTTP server
- planned Support for more architectures beyond Qwen2/3

---

## What this project will not do

Each of these was considered against measurements on the target hardware and
rejected for a stated reason. The full analysis is in the research notes.

| | why not |
|---|---|
| Ternary/binary weights as the core format | Round-to-nearest ternary destroys Qwen2.5-0.5B: KL 10.9 nats, 0% top-1 agreement. BitNet *trains* in ternary; these weights were not. |
| `bmma` as the main decode primitive | Real and fast (610 TOPS), but decode is bandwidth bound — cheaper arithmetic buys ~nothing at batch=1. |
| Anything requiring training or long QAT | 10–150B tokens is out of scope on one 6 GB card. |
| LUT / product-quantization matmul | GPUs read ~32 shared-memory words per clock against thousands of MACs. The CPU economics invert. |
| Marlin, FlashAttention-2, BitBLAS kernels | All architected around `cp.async`, which is sm_80+. Turing has no equivalent. |
| 2:4 structured sparsity | No hardware support on sm_75, and it saves no bytes at low bit widths. |
| Weight offloading | The model is 264 MB against 6 GB of VRAM. There is nothing to offload. |
| Fitting large-model knowledge into a small model | Language models store ~2 bits of knowledge per parameter ([Allen-Zhu & Li, ICLR 2025](https://arxiv.org/abs/2404.05405)). A 0.5B model caps near 1 Gbit; a 1T model holds ~2 Tbit. That ~2,000× gap is an information-capacity limit, not an efficiency one — no quantizer or kernel closes it. The achievable versions are retrieval, task-specific distillation, and speculative decoding (which is provably lossless). |
| Post-training sub-2-bit on this model | A *training* problem, not a kernel problem. The one credible route, BitDistill (arXiv 2510.13998), needs 10B tokens of continual pre-training plus distillation — infeasible on one 6 GB card. |

The recurring error these guard against is **quoting a speedup without its batch
axis**. Most published low-bit kernel results are measured at batch 16–256.
Whetstone runs at batch 1, and almost none of them transfer.
