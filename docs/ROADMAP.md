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

- **done (0.6.0)** **fp16 top-k re-score under a quantized head**
  (`convert --head-rescore`). End to end on Qwen2.5-0.5B: **16.0220 → 15.4717**
  perplexity, against 15.3711 for a full fp16 head — **85% of the gap, at
  int4-head bandwidth**, for 272 MB of VRAM and ~4.5% of the decode step.

  It needed no top-k kernel. A *threshold* selects the same set for two
  reductions, chosen on device from a geometric ladder because logit spreads vary
  per token. Two things went wrong and are worth keeping: reducing in a **single
  block** cost 7.7% where the bandwidth model said 0.17% (one SM of thirty
  reading 608 KB — the model was right about bytes and silent about
  parallelism), and dropping rows past a buffer cap made the result depend on
  `atomicAdd` scheduling, so perplexity moved run to run. Fixed by rescoring
  every survivor and never letting the threshold overflow the buffer: **fewer
  rows deterministically beats more rows arbitrarily.**
- planned Sensitivity-aware bit allocation. `v_proj` is the worst tensor in
  every one of the 24 layers and is 0.93% of parameters; llama.cpp bumps exactly
  this tensor and gates the rule on `n_gqa >= 4`, and this model is 7.
- planned Sequential GPTQ — propagate quantized activations layer by layer
  rather than taking every Hessian from the fp16 model.
- planned Calibration on a broad corpus (C4) rather than in-domain wikitext,
  which is what the literature does and what the domain-shift result above says
  is the honest configuration.
- **killed** int3 with hierarchical scales. Hierarchical metadata does help —
  +29 (g128 RTN) becomes **+10.215** at 0.5 B — but nowhere near enough, and
  GPTQ's −0.907 does not close a ten-point gap. The ladder then killed it
  outright: **+10.215 at 0.5 B, +3.988 at 1.5 B, +5.688 at 3 B.** int4 improves
  monotonically with model size; **int3 does not.** An earlier note read the
  first two rungs, called it a trend, and estimated ~+1.5 at 7 B — the next rung
  falsified that. Two points looked like a line. int2-hier is +3 861.
- **killed** vector quantization below 4 bits, and the bound that explains it.
  A learned codebook at `d=2, b=8` does beat the shipped scalar format
  (+1.412 at 4.14 bpw against +1.575 at 4.277) — but Δ perplexity turns out to be
  a function of *effective levels per dimension*, `2^(b/d)`, and essentially not
  of `d`: 16 levels is fine at any `d`, 5.7 is +22.9, 4 is +562, 2 is +640 246.
  **Vectorising redistributes quality within a bit budget; it does not create
  one.** AQLM's 2-bit headline comes from *additive* quantization with many
  summed codebooks plus fine-tuning, not from vectorising — a single 256-entry
  codebook was never going to reach it.
- **killed** low-rank weight factorisation. The best possible rank-r
  approximation (truncated SVD, so no fitting error to blame) at half the
  parameters is **+184 294 perplexity at an effective 8 bits/weight**, while
  int4-hier reads *half* those bytes for +1.575. Transformer weight matrices are
  near full rank, and the arithmetic leaves almost no room anyway: a
  factorisation only saves when `r < out·in/(out+in)`, which for 4864×896 is
  1.18× of headroom before it costs more than it saves.

  **Four bits is this engine's floor**, and the three kills above are why.
- planned Hadamard/rotation preprocessing
- planned int8 KV cache

## Stage 5c — Architecture coverage  *in progress*

Whetstone executes **one block shape**: pre-norm RMSNorm, RoPE, grouped-query
attention, SwiGLU. Everything below is measured against whether a family fits
that shape, because the cost of the ones that do not is not a config change.

**Why this is a whitelist by name and not a structural probe.** A
mixture-of-experts `config.json` parses *perfectly* as a dense one — it simply
carries `num_experts` fields `ModelConfig` ignores. A permissive check would
load it, run it, and generate fluent text produced by one expert's worth of
weights, with no shape mismatch anywhere to catch it. The same is true of
QK-norm and of an unimplemented RoPE schedule. **Every architecture failure
available here is silent**, so the check refuses by family and each new one is a
deliberate addition.

### Tier 1 — fits the shape  **done**

`qwen2`, `qwen3`\*, `llama`, `mistral`, `smollm2`, and every DeepSeek-R1 distill
onto those skeletons. They differ only in widths, GQA ratio, whether q/k/v carry
biases, and the RoPE schedule.

- **done** `ModelConfig::architecture()` — records `qkv_bias` and `qk_norm`
  instead of hardcoding a family
- **done** Llama 3.1+ RoPE frequency schedule (`rope_scaling.rope_type = llama3`)
- **done** Unimplemented RoPE schedules (`yarn`, `dynamic`) refused rather than
  ignored — ignoring one does not fail, it degrades coherence past the trained
  context, which reads as the model being bad at long inputs
- **done** Sharded checkpoints (`model.safetensors.index.json`). Every model
  above ~2 B ships sharded, so without this the ladder stopped at 1.5 B

\* `qwen3` is refused pending Tier 2.

### Tier 2 — one kernel each  *next, in this order*

| | families | what is needed |
|---|---|---|
| **done (0.6.0)** | **Qwen3**, OLMo2, Gemma2 | **QK-RMSNorm** — folded into both RoPE kernels rather than given its own launch: the rotation already holds each head's vector in registers two elements per lane, which is exactly the layout a per-head RMS reduction wants, so it costs one block reduction and no extra pass over memory. Differential-tested against an f64 CPU reference with distinct q and k gains. **Not yet run on a real Qwen3 checkpoint** — the kernel is verified, the family is not. |
| planned | GLM-4 (dense), Phi-3 | **Partial rotary** — only `rotary_pct · head_dim` of each head is rotated. A bound on the existing RoPE kernel's loop, plus config plumbing. |
| planned | Gemma2 | Logit softcapping, sliding-window attention on alternate layers. |
| planned | Phi-3, some Llama forks | Fused `qkv_proj` in the checkpoint — a load-time split, since Whetstone fuses q/k/v itself and would otherwise fuse an already-fused tensor. |

Each is hours to a day, none touches the GEMV, and none changes the weight
format.

### Tier 3 — mixture of experts  *router done (0.6.0), execution planned*

**Status.** The routing half exists and is differential-tested; the execution
half does not. `wst_moe_router` does softmax over **all** experts then top-k —
HuggingFace's order, not the obvious one, because selecting first and softmaxing
the survivors gives different weights and a perfectly valid-looking distribution
while doing so. Ties break toward the lower index deterministically. Indices and
weights stay in device memory so a MoE step is still one graph launch rather than
`k` host round-trips per layer. Tested at the OLMoE (64-of-8), Qwen3-30B-A3B
(128-of-8) and Mixtral (8-of-2) geometries.

**What is missing:** the expert-gathered GEMV, the convert path for expert
tensors, the loader, engine dispatch, and expert residency. No MoE checkpoint
executes.

**And one correction worth carrying, because it inverts the economics below.**
The optimistic reading — "an 8×7 B model reads what a 13 B dense one does" — is
right about *bandwidth* and silent about *capacity*. Measured arithmetic for this
card (5 GB usable, PCIe Gen3 x8 at 5.77 GB/s against DRAM's 278):

| model | stored | read/token | file @4.28bpw | ceiling | **on a 6 GB card** |
|---|---|---|---|---|---|
| **OLMoE-1B-7B** | 6.9 B | 631 MB | **3.70 GB** | 441 | **~282** ✅ fits |
| Qwen1.5-MoE-A2.7B | 13.5 B | 828 MB | 7.21 GB | 336 | **23** ⚠ offload |
| Qwen3-30B-A3B | 30.5 B | 1627 MB | 16.33 GB | 171 | **5** ❌ offload |
| Mixtral-8x7B | 46.7 B | 6820 MB | 24.99 GB | 41 | **0.7** ❌ offload |

**Sparsity removes weights from the bandwidth bill, not from the machine.**
Qwen3-30B-A3B's 109 tok/s roofline becomes ~5 once 72% of its expert traffic
crosses PCIe. The MoE models worth executing on this card are the ones whose
*whole expert set* is resident, which today means the ~7 B-total class. (The
residency figure assumes uniform expert selection, the pessimistic case; real
routing is skewed and an LRU cache should beat it. By how much is an open
measurement, and it is the number that decides whether a 30 B model is usable
here at all.)

### Tier 3 details  planned, weeks

Mixtral, Qwen3-MoE, GLM-4.5, DeepSeek-V2/V3, Kimi.

A router GEMV per layer, top-k expert selection, and a **gather over expert
weights** in the MLP. Worth being precise about the economics, because they are
unusually favourable here: at batch 1 only the selected experts are read, so a
sparse model's *bytes per token* is set by its active parameters, not its total.
An 8×7 B model with 2 active experts reads roughly what a 13 B dense model does.
That is exactly the regime this engine is built for.

What it costs: a different execution graph. The MLP stops being three fixed
GEMVs and becomes a data-dependent gather, which means the CUDA graph capture has
to either be re-captured per routing decision or replaced by a kernel that takes
the expert indices from device memory — the latter, given that the position
cursor already works that way.

- planned Router GEMV and top-k selection on device
- planned Expert-indexed GEMV (the weight pointer becomes a device-side lookup)
- planned Re-derive the roofline for active-parameter traffic
- planned Shared-expert handling (DeepSeek, Qwen3-MoE keep one always-on expert)

### Tier 4 — Multi-head Latent Attention  planned, weeks

DeepSeek-V2/V3 and Kimi replace attention with MLA: K and V are compressed into a
shared low-rank latent that is what the cache actually stores, with per-head
up-projections applied at use. It is a genuinely different algorithm with a
different cache layout, not a variation on GQA — the existing `attn_decode`
kernel and `KvCache` do not generalise to it.

It is also the most interesting thing on this list for a bandwidth-bound engine,
because MLA exists precisely to shrink the KV cache, and KV traffic is what
eventually bounds long-context decode once the weights are 4 bits.

### What will not run here regardless

Kimi K2 is ~1 T parameters and DeepSeek-V3 is 671 B. At 4.28 bits that is 535 GB
and 359 GB of weights against 6 GB of VRAM. No quantizer closes a 60–90× gap;
those need a different machine, not a different kernel. Their runnable members on
this card are the distills — `DeepSeek-R1-Distill-Qwen-1.5B`,
`-Llama-8B` — and both are Tier 1, so both work today.

## Stage 6 — Speculative decoding  shipped in 0.5.0, partly

Lossless: a draft token is accepted only when it equals the model's own argmax,
so greedy output is reproduced token for token.

- **done** Multi-token verification pass (`Engine::forward_chunk`), which is the
  same kernel work as batched prefill and is what makes offload usable
- **done** n-gram / prompt-lookup drafting — no draft model needed
- planned Tree verification
- planned **Dynamic acceptance-rate guard.** Measured worst case today is
  **0.97×**, not 1.0×: on resident prose the draft almost never fires and the
  fallback still pays its bookkeeping. The guard should disable drafting after a
  run of empty rounds. Until it exists, `--spec` is off by default and documented
  as workload-dependent.
- planned A draft *model* path. The n-gram draft is free but only fires on
  self-quoting text; a 0.5B drafting a 3B costs 27% of a target token per draft
  step, which is too much at the measured chunk cost curve. It becomes viable
  once the chunk GEMM is nearer its bandwidth bound.

## Stage 6b — The chunk GEMM is compute bound  open

The multi-token GEMM moves weights at ~20 GB/s against the 145 GB/s the
single-token GEMV achieves, so a 16-token pass costs 4.93 single-token passes
where it should cost ~1.2. That single number caps resident speculation at 3.25×
and is now the largest remaining lever.

Measured, in `research/experiments/`:

- `probe_chunk_gemm.cu` — loading activations once per (group, token) and
  reusing them across `TILE` rows is **2× faster** at every shape. Shipped.
- `probe_chunk_wmma.cu` — an fp16 `wmma` version is **3× more accurate**
  (fp32 accumulate) and 1.3–2.4× faster on wide-output shapes, but *slower* on
  `down_proj` and `o_proj`, and the microbenchmark's run-to-run spread is ~25%.
  Not shipped: the honest end-to-end win did not clear the noise.
- Four independent accumulators to break the 16-deep dependent FMA chain: **no
  effect**. The kernel is not latency bound.
- A warp-cooperative dequantize to remove a 16-way shared-memory bank conflict:
  **worse**, because it makes the per-group scale arithmetic 32× redundant.

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
| ~~Weight offloading~~ | **Reversed in 0.5.0.** The original reason — "the model is 264 MB against 6 GB of VRAM" — was true only of the 0.5B reference model. At 3B and above the budget binds, and offload also buys context length: VRAM not spent on weights is KV cache. `--vram` ships. What has *not* changed is the arithmetic: host RAM is ~46× slower than VRAM here, so this buys the ability to run, not speed. |
| Optimising weight relative error | Measured twice, in both directions: a clip search that *lowers* it raises perplexity by 0.50, and GPTQ *raises* it while lowering perplexity by 1.73. Weight error and output error are different objectives. |
| Fitting large-model knowledge into a small model | Language models store ~2 bits of knowledge per parameter ([Allen-Zhu & Li, ICLR 2025](https://arxiv.org/abs/2404.05405)). A 0.5B model caps near 1 Gbit; a 1T model holds ~2 Tbit. That ~2,000× gap is an information-capacity limit, not an efficiency one — no quantizer or kernel closes it. The achievable versions are retrieval, task-specific distillation, and speculative decoding (which is provably lossless). |
| Post-training sub-2-bit on this model | A *training* problem, not a kernel problem. The one credible route, BitDistill (arXiv 2510.13998), needs 10B tokens of continual pre-training plus distillation — infeasible on one 6 GB card. |

The recurring error these guard against is **quoting a speedup without its batch
axis**. Most published low-bit kernel results are measured at batch 16–256.
Whetstone runs at batch 1, and almost none of them transfer.
