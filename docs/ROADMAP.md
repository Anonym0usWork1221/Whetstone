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

| engine | format | bits/wt | bytes/token | decode (tg384) | Δ ppl vs fp16 |
|---|---|---|---|---|---|
| HuggingFace | fp16 | 16.00 | 988 MB | 40.3 | *(anchor)* |
| **llama.cpp** | **Q4_K_M** | 6.35 | 392 MB | **283.8** | **+0.3957** |
| Whetstone 0.3.0 | int4-g128 | 4.25 | 262 MB | 434.1 | +4.2078 |
| **Whetstone 0.4.0** | **int4-hier-g32** | 4.28 | **264 MB** | **415.2** | **+2.2011** |
| **+ GPTQ (opt-in)** | int4-hier-g32 | 4.28 | 264 MB | 414.0 | **+0.8174** |

llama.cpp is the real competitor, not HuggingFace:

- **Speed: 1.46×**, from reading 1.49× fewer bytes per token. The roofline said
  the ratios would track, and they do.
- **Quality: 2.1× its damage**, down from 10.6× in 0.3.0.

Every stage below is judged against llama.cpp on **both** axes.

*(Q4_K_M's row is llama.cpp's own weights dequantized with its own `gguf-py` and
measured in **this** harness. Quoting `llama-perplexity` instead is not valid: it
scores only the second half of each window, worth 1.57 perplexity on this corpus.
Two earlier versions of this file treated that offset as a tokenization
difference.)*

*(And Q4_K_M is not a 4-bit format on this model. `hidden = 896` is not a
multiple of `QK_K = 256`, so every projection except `down_proj` falls back to
`Q5_0` at 5.50 bpw and the tied head to `Q8_0` at 8.50 — body **5.53**, against
Whetstone's 4.25.)*

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
- **done** `whetstone convert` — 988 MB → 272.5 MB at 4.28 bits/weight
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

## Stage 5 — A quantizer that competes  **done**

This is where the accuracy budget gets spent, and in 0.3.0 it was overspent by
5×. At a fixed ~4.25 bits on the transformer body:

| quantizer | bits/wt | Δ ppl |
|---|---|---|
| round-to-nearest, group 128 *(0.3.0)* | 4.250 | +2.730 |
| llama.cpp's complete k-quant fitted scale/min search | 4.250 | +2.575 |
| **group 32, hierarchical scale metadata** *(0.4.0)* | **4.277** | **+1.575** |
| **+ GPTQ at 131k calibration tokens** | 4.277 | **+0.668** |

- **done** `Int4HierG32`: group 32 with two 4-bit indices per group against one
  `f16` pair per row. Group size is worth ~6× what the fitting algorithm is
  worth, and this is how to afford it — 0.036 bits/weight for a factor of four
  in granularity.
- **done** GPTQ at adequate calibration. The 0.3.0 sweep used **293** tokens
  against 896- and 4864-dimensional Hessians; at 131,072 it is the single
  largest lever in the project (−1.73 at group 128).
- **done** AWQ activation-aware scaling, with all four fusion points including
  the GQA-constrained `o_proj → v_proj` one. **Not shipped**: worth 0.88 alone
  but only 0.06 on top of GPTQ, since both read the same activation statistics.
- **done** `--head int4-hier` as an opt-in switch. Cost measured at **+0.52** ppl
  for 1.76× fewer bytes, against +1.10 in the 0.3.0 format.

**The caveat on the GPTQ row.** It is calibrated on held-out wikitext and
evaluated on wikitext. Recalibrated on 131k tokens of C/C++ source it reads
**+2.27 — worse than not running it at all.** The inverse Hessian is a claim
about which input directions matter, and code and Wikipedia disagree. `convert`
therefore ships the data-free format; GPTQ is an opt-in offline step.

## Stage 5b — What is left in the quantizer  *next*

- planned **fp16 top-k re-score under a quantized head.** Measured in the
  research harness: k=64 removes **82%** of the head's remaining cost for
  `64·896·2` = 114 KB/token — **0.17% more bandwidth** — plus 272 MB of VRAM,
  which a 6 GB card holding a 264 MB model is not short of. Needs a top-k
  reduction and a gathered 64×896 GEMV; neither touches the main decode path.
- planned Sensitivity-aware bit allocation. `v_proj` is the worst tensor in
  every one of the 24 layers and is 0.93% of parameters; llama.cpp bumps exactly
  this tensor and gates the rule on `n_gqa >= 4`, and this model is 7.
- planned Sequential GPTQ — propagate quantized activations layer by layer
  rather than taking every Hessian from the fp16 model.
- planned Calibration on a broad corpus (C4) rather than in-domain wikitext,
  which is what the literature does and what the domain-shift result above says
  is the honest configuration.
- planned int3 with hierarchical scales **and** GPTQ. int3-g128 round-to-nearest
  measured +29 and was written off; the two techniques are jointly worth 2.06 at
  4 bits and int3 has never been measured with them.
- planned Hadamard/rotation preprocessing
- planned int8 KV cache

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
| Optimising weight relative error | Measured twice, in both directions: a clip search that *lowers* it raises perplexity by 0.50, and GPTQ *raises* it while lowering perplexity by 1.73. Weight error and output error are different objectives. |
| Fitting large-model knowledge into a small model | Language models store ~2 bits of knowledge per parameter ([Allen-Zhu & Li, ICLR 2025](https://arxiv.org/abs/2404.05405)). A 0.5B model caps near 1 Gbit; a 1T model holds ~2 Tbit. That ~2,000× gap is an information-capacity limit, not an efficiency one — no quantizer or kernel closes it. The achievable versions are retrieval, task-specific distillation, and speculative decoding (which is provably lossless). |
| Post-training sub-2-bit on this model | A *training* problem, not a kernel problem. The one credible route, BitDistill (arXiv 2510.13998), needs 10B tokens of continual pre-training plus distillation — infeasible on one 6 GB card. |

The recurring error these guard against is **quoting a speedup without its batch
axis**. Most published low-bit kernel results are measured at batch 16–256.
Whetstone runs at batch 1, and almost none of them transfer.
