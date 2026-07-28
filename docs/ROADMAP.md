# Roadmap

The goal is a complete, self-contained stack: convert a model once, then execute
it with no Python and no framework in the token loop.

```
  HF checkpoint ──▶ whetstone convert ──▶ model.wstone ──▶ whetstone run
      (bf16)          quantize + pack                       native engine
```

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

## Stage 3 — The executor  *next, and the biggest gap*

This is what stands between the project and a runnable engine. Everything below
is required before `whetstone run` exists.

- planned RMSNorm kernel (fp32 reduction — `f16` accumulation over `h=896`
  loses precision badly)
- planned RoPE kernel, half-rotation layout, precomputed cos/sin tables
- planned Decode attention with a paged KV cache, GQA 7:1 expansion in-register
- planned SwiGLU, with `gate_proj` and `up_proj` fused (they share an input)
- planned Sampling: argmax, temperature, top-p, on-device
- planned Model graph and weight residency in `whetstone-core`
- planned `whetstone run` / `whetstone chat`
- planned Differential test of the whole stack against the fp64 reference

**Acceptance:** top-1 agreement ≥ 99% against fp16 on a fixed prompt set, and a
wikitext-2 perplexity delta that is reported, not hidden.

## Stage 4 — Removing the overhead  *the largest measured win*

The baseline reaches **11% of its own roofline**. Roughly 24 ms of every 27 ms
token is dispatch, launch and allocator cost — not memory traffic. This stage
costs **zero accuracy** and is worth more than any quantization step.

- planned Static execution plan: allocate once, no per-token allocation
- planned Kernel fusion — 7 GEMVs/layer × 24 layers = 168 launches per token,
  which alone caps decode near 1,200 tok/s at ~5 µs of launch overhead each
- planned CUDA graphs for the decode step
- planned Per-layer megakernel where fusion is not enough

## Stage 5 — Beyond int4  planned

Only after Stages 3 and 4, because they are lossless and this is not.

- planned Quantize `lm_head` by default — 27.6% of decode traffic in one matrix,
  with an fp16 top-k re-score so argmax stays exact
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

The recurring error these guard against is **quoting a speedup without its batch
axis**. Most published low-bit kernel results are measured at batch 16–256.
Whetstone runs at batch 1, and almost none of them transfer.
