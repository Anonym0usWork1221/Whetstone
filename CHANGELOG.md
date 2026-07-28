# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/), with one project-specific rule:

> **The `.wstone` format version is independent of the crate version.** A change
> to the on-disk format bumps `format::VERSION`, and a reader refuses a file it
> was not built for rather than guessing. Format changes are called out
> explicitly below.

## [Unreleased]

### Planned
- Full CUDA forward pass: RMSNorm, RoPE, decode attention with paged KV cache,
  SwiGLU, on-device sampling — this is what `whetstone run` is waiting on
- Kernel fusion and CUDA graphs (168 launches per token currently cap decode
  near 1,200 tok/s regardless of kernel quality)
- `lm_head` quantized by default, with an fp16 top-k re-score so argmax stays exact
- GPTQ with adequate calibration (~262k tokens)

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

[Unreleased]: https://github.com/Anonym0usWork1221/Whetstone/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.1.0
