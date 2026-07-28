# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/), with one project-specific rule:

> **The `.wstone` format version is independent of the crate version.** A change
> to the on-disk format bumps `format::VERSION`, and a reader refuses a file it
> was not built for rather than guessing. Format changes are called out
> explicitly below.

## [Unreleased]

### Planned
- GPTQ with adequate calibration (~262k tokens). int4-g128 round-to-nearest
  costs **+2.75 perplexity** on this model, which is now the largest open number
  in the project — larger than any remaining speed win.
- AWQ-style activation-aware scaling
- fp16 top-k re-score under a quantized `lm_head`, so `--head int4` keeps its
  1.76x bandwidth win without moving the argmax
- Split-K over the reduction for `down_proj` and `o_proj`, which are
  output-starved the way attention was before it was split over the sequence
- RMSNorm fused into the following GEMV's prologue
- Batched prefill (it currently runs the decode path one token at a time)

---

## [0.2.0] — 2026-07-28

**The executor works.** `whetstone run` executes a `.wstone` end to end with no
Python in the token loop, at **431.8 tok/s** against llama.cpp Q4_K_M's
**282.95** on the same RTX 2060, same model, same 384-token generation —
**1.53x**, at 1.49x fewer bytes per token. (431.8 is the median of five cold
process launches; run-to-run spans 418-443. A warm engine reaches 486.)

The correctness claim rests on the fp16 path: wikitext-2 perplexity **13.8209**
against HuggingFace's **13.8182** over the same 40,940 predictions, a 0.02%
difference. That single number validates the RoPE half-rotation layout, GQA
expansion, the online-softmax attention, SwiGLU, the fp32 residual stream and
the `.wstone` loader at once.

### Added

**The engine**
- Full CUDA forward pass: RMSNorm (fp32 reduction, single-pass when the vector
  fits one element per thread), RoPE with cos/sin tables precomputed in f64,
  GQA decode attention with an online softmax split across the sequence,
  SwiGLU, on-device argmax
- `whetstone run` — generation, tokens/second, p10/p50/p90 latency, achieved
  bandwidth and roofline attainment
- `whetstone ppl` — perplexity over a token stream, with the negative
  log-likelihood accumulated on the device so a 41,000-step evaluation never
  blocks on a scalar copy
- `whetstone logits` — raw f32 dumps for top-1 and KL comparison
- `whetstone bench` — int4 GEMV variant sweep across every shape the model issues
- `whetstone tune` — sweeps the per-shape kernel rule by real generation
  throughput
- `whetstone run --profile N` — per-stage attribution from stream-ordered CUDA
  events
- `whetstone convert --body fp16` — a lossless reference model. Keeping one
  runnable is what separates "the engine is wrong" from "the quantizer is
  lossy"; without it the perplexity figures below would be guesses.
- `bench/prepare_tokens.py` — materialises the evaluation token stream once, so
  two harnesses provably read the same tokens
- CUDA graph capture of the whole decode step (`--graph`)

### Changed

**Kernels — 1.51x end to end, at bit-identical output**
- int4 dequantization now ORs into an fp16 mantissa instead of using `I2F`.
  fp16 `1024.0` is `0x6400` and at that exponent the mantissa ULP is exactly 1,
  so `0x6400 | q` *is* `1024 + q` — an integer-to-float conversion done with a
  logic op, replacing an instruction that runs at quarter rate on Turing.
- `q|k|v` and `gate|up` are concatenated **at load time** into single matrices.
  The file format is unchanged and existing `.wstone` files keep working;
  concatenating int4-g128 along the output dimension is a plain row append.
- Attention is split across the sequence as well as the heads, because 14 query
  heads cannot fill 30 SMs. Two kernels, merged by the online-softmax
  recurrence, which is associative.
- Bias and the residual add moved into GEMV epilogues
- Per-shape kernel selection, tuned by whole-generation throughput

### Measured

| format | bits/wt | bytes/token | tok/s (tg384) | wikitext-2 ppl |
|---|---|---|---|---|
| fp16 | 16.00 | 987.9 MB | 185.6 | 13.8209 |
| int4 body, fp16 head | 7.49 | 462.4 MB | 305.9 | 16.5696 |
| int4 everywhere | 4.25 | 262.4 MB | 431.8 | 18.0287 |

**int4-g128 round-to-nearest costs 2.75 perplexity, and `--head int4` costs
another 1.46.** The 0.1.0 notes reported int4 at "100% top-1 agreement" — true,
and measured on three prompts, which is not enough to detect a regression of
this size. Top-1 agreement is not a quality gate; the argmax stays stable long
after the distribution has moved.

### Negative results

Recorded because they cost real time and the reasoning behind them is the kind
that repeats:

- **CUDA graphs were worth 0.5%, not the predicted 1.5–3x.** 247 launches
  collapsed into 1 and the token rate did not move: the CPU was already running
  far ahead of the GPU and the launches were entirely hidden. Multiplying launch
  count by launch *latency* treats a latency as an occupancy cost. Kept, because
  it enables a host-free generation loop.
- **A profiler that synchronises between stages reports a different ranking, not
  just a worse number.** It measured a 448-byte embedding gather at 0.486 ms per
  token; the event-based version measures 0.005 ms.
- **GEMV tiling is not about activation reuse** — `x` is L1-resident. It trades
  memory-level parallelism against block count, which is the same constraint
  that made fusing the projections and splitting attention pay.
- **Microbenchmark rankings do not transfer.** Sweeping one matrix keeps it in
  L2; the engine never does. Swept properly by generation throughput, the entire
  27-rule kernel-selection space spans 2.9%.

Full record in the project's research notes.

---

## [0.1.0] — 2026-07-28

First release. The **weight pipeline works end to end; the executor does not
yet.** Conversion, verification, the int4 decode GEMV and the whole measurement
stack are done and tested. The full forward pass is the next milestone.

### Added

**Weight pipeline**
- `.wstone` container format (format version **1**) — mmap-friendly, 256-byte
  aligned payloads, FNV-1a checksums per blob, embedded model config so no
  sidecar files are needed. Spec in [docs/FORMAT.md](docs/FORMAT.md).
- int4 group-128 asymmetric quantizer and bit packer
- `whetstone convert` — Qwen2.5-0.5B from 988.1 MB to 263.6 MB at 4.25
  bits/weight, mean relative weight error 0.110
- `whetstone verify` — blob integrity, plus fidelity against the source checkpoint

**Kernels**
- int4-g128 decode GEMV, differential-tested against its dequantized reference
- fp16 reference GEMV, same schedule, for separating kernel bugs from
  quantization loss
- `whetstone probe` — measures every arithmetic path on the actual device
  (fp16/int8/int4 WMMA, 1-bit BMMA, dp4a, popcount) and verifies the XNOR dot
  identity `dot = K − 2·popcount(a⊕b)` on-device
- Capability gating for the sm_75 boundaries that constrain kernel design:
  `bmma.xor.popc` yes, `bmma.and.popc` / `cp.async` / 2:4 sparsity / fp8 no

**Core**
- `config` with a roofline model, unit-tested against real parameter counts
- `safetensors` loader hardened against truncation, overlapping tensors and
  shape/range disagreement
- `whetstone inspect` — architecture, tensor inventory, roofline, KV cache sizes

**Harness**
- `bench/chat.py` — interactive chat with live token streaming, tok/s, TTFT and
  a roofline attainment bar; `--bench` for non-interactive throughput runs
- `bench/reference_numpy.py` — independent fp64 forward pass written from
  `config.json` alone, the ground truth for differential testing
- `bench/tokenizer.py` — byte-level BPE read straight from `tokenizer.json`
- `bench/baseline_hf.py` — HuggingFace baseline: tok/s, achieved bandwidth,
  wikitext-2 perplexity, reference logits

**Documentation**
- README states plainly what is and is not novel: the int4 pipeline is a
  reimplementation of GPTQ/GGUF-era work, and the uncommon parts are the
  abandoned-`sm_75` niche, Turing's unexploited INT4/INT1 tensor cores, and the
  published negative results
- `docs/RELEASES.md`, `docs/FORMAT.md`, `docs/ROADMAP.md`

**Packaging**
- `scripts/deploy.sh` / `scripts/deploy.ps1` — versioned, checksummed release
  packages for Linux and Windows
- `scripts/run.sh` / `scripts/run.bat` — one launcher for probe, inspect,
  convert, verify, chat, bench, setup and doctor
- Tag-triggered GitHub release workflow
- `whetstone --version` reports commit, build date, target and CUDA architecture

### Measured on the reference hardware

RTX 2060 (sm_75, 30 SMs, 6 GB, 336 GB/s peak / 278 GB/s achievable),
Qwen2.5-0.5B-Instruct:

| | |
|---|---|
| HF fp16 baseline | 36.8 tok/s, wikitext-2 perplexity 13.8182 |
| fp16 roofline | 340 tok/s — baseline attains **11%** |
| int4 GEMV vs fp16 GEMV | **1.5–1.9×** wall-clock, 3.75× fewer bytes read |
| `.wstone` int4 (body + head) | 263.6 MB, 4.25 bits/weight, ceiling 1059 tok/s |

### Known limitations

- **No executor.** `whetstone run` does not exist; the forward pass is not
  written. `bench/chat.py --engine whetstone` fails with an explanation rather
  than silently falling back to HuggingFace.
- **Single architecture per build.** Artifacts are named with their `sm_` target
  and will not run on older cards.
- Round-to-nearest only. GPTQ exists in the research tree but its sub-4-bit
  results are inconclusive — the calibration set was 293 tokens, leaving every
  Hessian rank-deficient.
- Quantization is applied uniformly. Sensitivity-aware bit allocation is not
  implemented, though `v_proj` is measurably the worst tensor to quantize.
- `deploy.ps1` and `run.bat` are exercised by CI but have not been run
  interactively on Windows by the author.

[Unreleased]: https://github.com/Anonym0usWork1221/Whetstone/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.2.0
[0.1.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.1.0
