# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/), with one project-specific rule:

> **The `.wstone` format version is independent of the crate version.** A change
> to the on-disk format bumps `format::VERSION`, and a reader refuses a file it
> was not built for rather than guessing. Format changes are called out
> explicitly below.

## [Unreleased]

Nothing yet.

---

## [0.4.0] — 2026-07-28

**The quantizer caught up with the engine.** 0.3.0 was 1.53x llama.cpp
Q4_K_M at 10.6x its quantization damage. 0.4.0 is 1.46x at 2.1x, from a new
weight format that costs 0.03 bits/weight.

### Added

- **`Int4HierG32` — a new weight format, and the new default for `convert`.**
  int4 with group 32, where each group stores a 4-bit scale index and a 4-bit
  minimum index against one fp16 `(d, dmin)` pair per row:
  `scale = d*ls`, `min = -dmin*lm`, `w = q*scale + min`. That is
  `4 + 8/32 + 32/in_features` bits per weight — **4.28 against the old format's
  4.25** — and it measured **1.15 perplexity better** on the transformer body.

  Group size turned out to be worth roughly six times what the fitting algorithm
  is worth (group 128 → 64 buys 0.96 perplexity; llama.cpp's complete k-quant
  alternating least-squares fit at fixed group size buys 0.16), and the reason
  the old format could not use it is that an fp16 scale plus an fp16 zero per
  group of 32 costs 1.0 bits/weight of metadata against group 128's 0.25.
  Small unsigned indices against a per-row pair is what makes it affordable.

  `--body int4` still selects the group-128 format so an A/B is one flag.

- **`quantize_int4_hier` / `dequantize_int4_hier`** in `whetstone-quant`, and
  `QuantLinearHier` plus `cuda/gemv_hier.cu` in `whetstone-kernels`. The GEMV
  computes the per-group activation sums *inside* the reduction rather than in a
  prologue kernel — the lane that owns a group has already loaded exactly the 32
  activations it needs to sum, and that sum is shared across every row the warp
  accumulates — so the format change added no kernel launches, no scratch buffer
  and no API change.

- `whetstone verify` reads both quantized formats, and rejects a rank-1 tensor
  declared as quantized instead of indexing out of bounds.

### Changed

- Measured end to end, 384 generated tokens, median of four interleaved rounds
  against llama.cpp on the same GPU in the same run. Perplexity is wikitext-2,
  20 × 2048-token windows; llama.cpp's row is **its own weights measured in this
  harness**, not a number quoted from `llama-perplexity`, because that tool
  scores only the second half of each window and the two are not comparable.

  | format | bytes/token | bits/wt | tok/s | ppl | Δ vs own fp16 |
  |---|---|---|---|---|---|
  | llama.cpp Q4_K_M | 392 MB | 6.35 | 283.8 | 14.2138 | +0.396 |
  | int4-g128 (0.3.0) | 262 MB | 4.25 | 434.1 | 18.0287 | +4.208 |
  | **int4-hier-g32** | 264 MB | 4.28 | 415.2 | 16.0220 | +2.201 |
  | **int4-hier-g32 + GPTQ** | 264 MB | 4.28 | 414.0 | 14.6383 | **+0.817** |

  **1.46× llama.cpp Q4_K_M at 2.06× its quantization damage**, against 0.3.0's
  1.53× at 10.6×. The new format costs 4.6% throughput for a 5.1× reduction in
  damage at the same width.

- The `.wstone` header checksum's multiplier is documented as **not** the FNV-1a
  prime — `0x1000_0000_01b3` is one hex digit longer than `0x100000001b3`. It is
  deliberately left alone (changing it would invalidate every existing file for
  no benefit, since its only job is detecting corruption) and now says so, so it
  does not get "fixed". An independent reimplementation of the container is what
  surfaced it.

### Notes

- **Weight relative error is not a quality measure.** A clip search that lowers
  mean weight error from 0.1102 to 0.1067 raises perplexity by 0.50; GPTQ raises
  weight error to 0.1416 and lowers perplexity by 1.73. `convert` still reports
  it, as a smoke test for a broken packer and nothing more.
- GPTQ is an offline step and does not change the weight format. The tooling
  lives outside this repo, in the research tree, and writes the same container.

---

## [0.3.0] — 2026-07-28

**Whetstone can now hold a conversation without Python.** `whetstone chat` is an
interactive REPL that reports throughput per turn, and the tokenizer that makes
it possible is written in Rust and verified token-for-token against the
reference implementation.

### Added

- **`whetstone chat`** — an interactive REPL with tokens/second reported per
  turn. The KV cache is kept across turns, so turn *n* prefills only its own
  message instead of re-sending the transcript: re-sending is quadratic in
  conversation length, and by turn ten most of the machine's time goes on
  recomputing what it already knows.
- **A byte-level BPE tokenizer in Rust**, read from `tokenizer.json`. Verified
  by producing **token-for-token identical output to the reference
  implementation across all 299,078 tokens** of wikitext-2. Whetstone's premise
  is that no Python sits in the token loop; a chat REPL that shells out to
  `transformers` for ids would have broken that at the first step.
  - The pre-tokenizer is hand-written rather than a `Regex`: the pattern
    `tokenizer.json` declares contains `\s+(?!\S)`, and Rust's `regex` crate
    excludes negative lookahead by construction.
  - `StreamDecoder` holds incomplete UTF-8 across tokens, so a multi-token
    emoji streams as one character instead of replacement marks.
- **The tokenizer is embedded in the `.wstone`** (7 MB), so `whetstone chat
  model.wstone` needs no sidecar files. The header gained an `extras` section;
  it is `#[serde(default)]` on both sides, so old files still load and old
  readers ignore it — no format version bump, because `format::VERSION` moves
  when a reader would *misinterpret* a file, not when the header grows.
- `run.sh chat` / `run.bat chat` now drive the native REPL; the HuggingFace
  harness moved to `hfchat`. `bench/compare.py` ships in the release archives.

### Fixed

- **A clippy failure that only a newer toolchain caught.**
  `unnecessary_sort_by` in the tokenizer passed under a local stable from March
  and failed CI's July stable. `scripts/deploy.sh` now runs
  `cargo clippy --release --all-targets --locked -- -D warnings` — the exact CI
  command — and refuses to package if it fails. A release preflight weaker than
  CI is not a preflight; it just moves the discovery to after the tag is public.
- **`run.bat` passed every subcommand twice.** `shift` does not affect `%*` in
  batch, so `run.bat probe --iters 100` invoked `whetstone probe probe --iters
  100`, and `run.bat convert <dir> <out>` repeated both positional arguments
  after the flags. Every label now collects its remaining arguments explicitly.
  Delayed expansion is also switched off, because it eats `!` — and prompts
  routinely contain one.
- **The chat REPL emitted ANSI escapes unconditionally**, which show up as
  literal `←[1m` on a Windows console that has not enabled virtual-terminal
  processing, and as noise in a redirected file. Styling is now gated on
  `IsTerminal`, `NO_COLOR`, and — on Windows — `WT_SESSION`. The per-turn stats
  line is plain ASCII, since a console on codepage 437 renders UTF-8 punctuation
  as mojibake.
- **Nucleus sampling sorted the entire 151936-entry vocabulary on every token**,
  costing ~8 ms — four times the forward pass it was sampling from — and capping
  chat at 111 tok/s against greedy's 467. It now partitions off the top 512
  candidates in O(n) with `select_nth_unstable_by` and sorts only those: **369
  tok/s**. (The "early exit" that was supposed to bound the work ran *after* the
  sort and its condition was always true, so it bounded nothing.)
- `StreamDecoder` could wedge permanently on a malformed byte sequence: it held
  bytes back whenever UTF-8 validation failed, including when the failure was
  final rather than "incomplete so far". One bad byte would silence the stream
  for the rest of the generation. It now distinguishes the two cases via
  `Utf8Error::error_len`.

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

**The executor works, and measuring it against llama.cpp properly produced a
result that is half good news.**

`whetstone run` executes a `.wstone` end to end with no Python in the token
loop, at **423.9 tok/s** against llama.cpp Q4_K_M's **281.9** on the same RTX
2060 and model — **1.50x**, at 1.49x fewer bytes per token.

**The quantizer is 13x worse than Q4_K_M.** int4-g128 round-to-nearest costs
**+4.21 perplexity** against fp16; Q4_K_M costs **+0.33** against its own. That
is not a bit-budget difference: k-quants keep the embedding and output wide, so
Q4_K_M is 6.35 bits/weight, and Whetstone's 7.49-bit variant still costs +2.75 --
worse at more bits. The engine is not the problem; the rounding is.

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
- `bench/compare.py` — one command that measures your `.wstone`, the original
  weights and llama.cpp in a single run, and anchors each format's perplexity to
  fp16 *in its own harness* (the two disagree by 1.57 on identical weights, so
  absolute figures cannot be read across)
- `baseline_hf.py --tokens` — the HuggingFace baseline can now read the
  materialised token stream instead of re-tokenizing the corpus itself
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

All rows from one run of the new `bench/compare.py`:

| engine / format | bits/wt | bytes/token | decode tok/s | ppl | Δ vs own fp16 |
|---|---|---|---|---|---|
| HuggingFace fp16 | 16.00 | 988 MB | 40.3 | 13.8182 | *(anchor)* |
| llama.cpp fp16 | 16.00 | 988 MB | 131.0 | 12.2484 | *(anchor)* |
| **llama.cpp Q4_K_M** | 6.35 | 392 MB | **281.9** | 12.5737 | **+0.3253** |
| Whetstone fp16 | 16.00 | 988 MB | 211.6 | 13.8209 | +0.0028 |
| Whetstone int4 body | 7.49 | 462 MB | 331.2 | 16.5712 | +2.7530 |
| **Whetstone int4** | 4.25 | 262 MB | **423.9** | 18.0287 | **+4.2106** |

Engines are **interleaved** — one sample of each, round-robin — because measuring
all of A then all of B compares A cold to B hot. An earlier run of this harness
did exactly that and read llama.cpp at 250.8 instead of 281.9, inflating the
speed ratio from 1.50× to 1.69×.

**Absolute perplexity is not comparable across the two harnesses** — the *same
fp16 weights* score 13.8182 here and 12.2484 under `llama-perplexity`, a
1.57-point offset from different tokenization and chunking. Only the last column
is comparable.

The 0.1.0 notes reported int4 at "100% top-1 agreement" — true, and measured on
three prompts, which is not enough to detect a regression of this size. Top-1
agreement is not a quality gate; the argmax stays stable long after the
distribution has moved.

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

[Unreleased]: https://github.com/Anonym0usWork1221/Whetstone/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.3.0
[0.2.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.2.0
[0.1.0]: https://github.com/Anonym0usWork1221/Whetstone/releases/tag/v0.1.0
