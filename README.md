# Whetstone

A from-scratch LLM inference engine that replaces the arithmetic inside existing
transformer layers with cheaper primitives — sub-byte integer formats, bit-packed
weights, and XOR/popcount dot products — to raise tokens/second on low-end GPUs.

Whetstone does not invent a new architecture. It keeps trained weights and layer
topology exactly as they are, and replaces **the math that evaluates them**.

- **Reference model:** Qwen2.5-0.5B-Instruct (494 M params, 988 MB bf16)
- **Reference GPU:** NVIDIA RTX 2060 — `sm_75` Turing, 30 SMs, 6 GB, 336 GB/s
- **Stack:** CUDA C++ kernels, Rust engine, Python for evaluation

> **Status: the weight pipeline works end to end; the executor does not yet.**
> Conversion, the `.wstone` format, verification, the int4 decode GEMV and the
> whole measurement stack are done and tested. The full forward pass
> (RMSNorm, RoPE, attention, SwiGLU) is the next milestone — see
> [docs/ROADMAP.md](docs/ROADMAP.md). Every number below is measured on the
> reference GPU, not projected.

---

## The one fact that shapes everything

At batch=1 autoregressive decode, every weight is read from memory once and used
for a single multiply-add. That is an arithmetic intensity of ~2 FLOP/byte,
against the ~120 FLOP/byte a Turing GPU needs to saturate its tensor cores.

**Decode is memory-bandwidth bound by roughly 60×.** So:

```
tok/s_ceiling  =  bandwidth / bytes_read_per_token
```

Compute throughput does not appear in that equation. An optimization that
reduces FLOPs but not bytes moved does approximately nothing for decode speed.

The denominator is **not** just the transformer blocks. Qwen2.5-0.5B ties its
embeddings, which makes it tempting to dismiss the `[vocab, hidden]` matrix as a
lookup table — but the *output* projection is a full GEMV over all 136.1 M of it
on every single token. That is **27.6% of decode traffic**:

| | params | share |
|---|---|---|
| transformer blocks | 357.8 M | 72.4% |
| `lm_head` (tied) | 136.1 M | **27.6%** |
| **read per token** | **494.0 M** | |

At the **measured** 278 GB/s:

| weight format | bits/weight | bytes/token | tok/s ceiling |
|---|---|---|---|
| fp16 | 16.00 | 987.9 MB | 281 |
| int8 per-channel | 8.00 | 494.0 MB | 563 |
| int4 g128 | 4.25 | 262.4 MB | 1060 |
| int3 g128 | 3.25 | 200.7 MB | 1385 |
| int2 g128 | 2.25 | 138.9 MB | 2001 |
| ternary g128 | 1.71 | 105.6 MB | 2633 |

Ceilings assume 100% bandwidth utilisation; a good kernel attains 60–80%.
Because the head is over a quarter of the traffic, **quantizing `lm_head` is
worth more than any further work on the transformer blocks.**

---

## The baseline, and where the opportunity actually is

HuggingFace `transformers` fp16, batch=1, on the reference GPU:

| metric | value |
|---|---|
| decode | **36.8 tok/s** (27.1 ms/token, p10 26.6, p90 28.0) |
| achieved bandwidth | 36 GB/s of 336 — **11% utilisation** |
| roofline attainment | **11%** of the 340 tok/s fp16 itself permits |
| wikitext-2 perplexity | **13.8182** |

The baseline is not limited by its number format. It spends ~24 ms of every
27 ms token on Python dispatch, kernel launches and allocator churn.

So the ranked opportunity is the opposite of where this project started:

1. **Remove framework overhead** — up to ~9×, and **costs zero accuracy**
2. **Quantize `lm_head`** — one matrix, 27.6% of all decode traffic
3. **Shrink the block weights** to int4 — the remaining 72.4%, at KL 0.19 and
   100% top-1 agreement
4. **Exotic arithmetic** — worth approximately nothing at batch=1 decode

The lossless win is both larger and safer than the lossy one.

> Measurement hygiene: an earlier version of this table read 17.5 tok/s with a
> p10–p90 of 35–69 ms. Those runs overlapped a CPU-saturating job, and HF decode
> at batch=1 is CPU-dispatch bound. The wide spread was the tell. Perplexity was
> unaffected, being GPU-bound.

## Measured hardware facts

`whetstone probe` measures every arithmetic path rather than trusting the spec
sheet. On the RTX 2060:

| path | throughput | vs fp16 |
|---|---|---|
| `wmma` fp16 (fp32 accumulate) | 15.0 TFLOPS | 1.00× |
| `wmma` int8 | 97.4 TOPS | 6.5× |
| `wmma` int4 | 143.4 TOPS | 9.5× |
| **`bmma` b1 (XOR+popcount)** | **609.7 TOPS** | **40.6×** |
| `__dp4a` (CUDA core) | 17.5 TOPS | 1.16× |
| `__popc` (CUDA core) | 281.8 TOPS | 18.8× |

Read these with two caveats, both of which we got wrong first time:

- These are **dependent accumulate chains**, so they sit between latency and
  issue rate — an ordering of the paths, not attainable GEMM throughput.
- The fp16 baseline **accumulates in fp32**, which on consumer Turing is
  half-rate versus fp16 accumulation. Every ratio above is therefore ~2×
  flattering to the alternative. fp32 accumulation is what a numerically sound
  GEMM needs, so it is the honest baseline here — but the choice must be stated.

`bmma` is ~2.2× a hand-written `__popc` loop, so bit arithmetic is fastest
through the tensor core. But CUDA-core popcount is itself ~19× fp16, and
`__dp4a` slightly beats fp16 — neither is the dead end that a per-warp/per-lane
units error initially made them look.

The identity underpinning the binary path is verified on-device:

```
for a, b ∈ {-1,+1}^K packed as bits {1,0}:
    dot(a, b) = K - 2 · popcount(a XOR b)
```

`sm_75` boundaries that constrain the kernels: `bmma.xor.popc` is available,
`bmma.and.popc` is sm_80+, and there is **no `cp.async`** — every modern low-bit
GEMM kernel is architected around it, so the software pipeline has to be
re-derived with register double-buffering.

---

## What the weights can actually survive

Published extreme-quantization results are overwhelmingly measured on 7B+
models. They do not transfer to 0.5B. Measured here, with round-to-nearest, by
quantizing every linear weight and comparing output distributions against the
unquantized model:

| format | bits/wt | weight rel. err | output KL (nats) | top-1 agreement |
|---|---|---|---|---|
| fp16 | 16.00 | 0.0000 | 0.00000 | 100% |
| int8 per-channel | 8.00 | 0.0100 | 0.01418 | 100% |
| int4 g128 | 4.25 | 0.1102 | 0.18679 | 100% |
| int3 g128 | 3.25 | 0.2348 | 1.96878 | 33% |
| int2 g128 | 2.25 | 0.5304 | 12.43208 | 0% |
| ternary g128 | 1.71 | 0.5414 | 10.93628 | 0% |
| binary g128 | 1.12 | 0.6274 | 13.07954 | 0% |

For scale: the model's own output entropy on these prompts is 1.3–3.1 nats. A KL
of 10+ means the distribution has no meaningful relationship to the original.

**Round-to-nearest ternary destroys this model.** That is not a contradiction of
the BitNet results — BitNet *trains* in ternary with a straight-through
estimator, so its weights are built to be representable on that grid. Weights
trained in bf16 were never constrained that way, and a 0.5B model has far less
redundancy to spend than a 7B one.

The open question is therefore not "how few bits" but "how much of that gap a
better quantizer recovers". A GPTQ pass improves int4 (KL 0.187 → 0.171, with
*higher* weight error — the expected signature of trading weight fidelity for
output fidelity), but the sub-4-bit runs are **not yet a fair test**: the
calibration set was 293 tokens, so `H = 2XᵀX` has rank ≤ 293 against dimensions
of 896 and 4864. Every Hessian was rank-deficient and dominated by damping. That
experiment needs re-running with ~262k calibration tokens before any conclusion
about sub-4-bit is drawn.

---

## Building

Requires the CUDA toolkit (`nvcc`) and a Rust toolchain.

```bash
cargo build --release
cargo test
```

`build.rs` drives `nvcc` directly and compiles for a single architecture
(`sm_75` by default). Override with:

```bash
WHETSTONE_CUDA_ARCH=86 cargo build --release
```

Single-architecture builds are deliberate: these kernels are written against
capabilities that differ by GPU generation, so a "portable" fat binary would be
a fiction.

## Using

```bash
# What can this GPU actually do?
whetstone probe

# Architecture, tensor inventory, and the roofline for a checkpoint
whetstone inspect /path/to/Qwen2.5-0.5B-Instruct

# Convert to Whetstone's own weight format
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone --head int4

# Check integrity, and fidelity against the source
whetstone verify qwen05b.wstone --source /path/to/Qwen2.5-0.5B-Instruct
```

### The `.wstone` format

A `.wstone` file stores weights **already arranged the way the kernels read
them** — nibbles packed in the order the GEMV's 128-bit load consumes them,
scales placed so one metadata load serves 32 weights, every blob 256-byte
aligned, and the model config embedded so no sidecar files are needed. Loading
is an `mmap` and a pointer walk, not a decode.

Measured on Qwen2.5-0.5B-Instruct:

| variant | size | bits/weight | mean rel. error | decode ceiling |
|---|---|---|---|---|
| source (bf16) | 988.1 MB | 16.00 | — | 281 tok/s |
| int4 body, fp16 head | 463.6 MB | 7.49 | 0.1102 | 601 tok/s |
| **int4 everywhere** | **263.6 MB** | **4.25** | 0.1095 | **1059 tok/s** |

Full specification in [docs/FORMAT.md](docs/FORMAT.md). The trade is
portability: no other runtime can execute a `.wstone`, and it is not meant to.
Use `safetensors` for interchange.

### Live chat and benchmarking

```bash
# interactive chat, tokens stream as they are produced
python bench/chat.py --model /path/to/model

# one prompt, with a speed and roofline readout
python bench/chat.py --model /path/to/model --prompt "explain GPU rooflines" --temperature 0

# non-interactive throughput run
python bench/chat.py --model /path/to/model --bench --out report.json
```

```
────────────────────────────────────────────────────────────────────
  speed              38.8 tok/s  (25.8 ms/token, p10 25.3, p90 26.5)
  first token        37 ms
  generated          495 tokens from a 41-token prompt

  bandwidth          38 GB/s (14% of peak)
  roofline           ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 14%
                     (39 of 281 tok/s possible)
────────────────────────────────────────────────────────────────────
```

`--engine whetstone` is wired but not yet runnable: the `.wstone` format,
quantizer and int4 GEMV are done, the full forward pass is not. It fails with a
clear message rather than silently falling back to the baseline — see
[docs/ROADMAP.md](docs/ROADMAP.md).

## Evaluation harness

```bash
# Independent fp64 reference forward pass, implemented from the config alone
python bench/reference_numpy.py --model /path/to/model

# Byte-level BPE tokenizer read straight from tokenizer.json
python bench/tokenizer.py /path/to/model

# HuggingFace fp16 baseline: tok/s, achieved bandwidth, wikitext-2 perplexity
python bench/baseline_hf.py --model /path/to/model
```

`bench/reference_numpy.py` is deliberately slow and obvious — no fusion, no
numerical shortcuts, float64 throughout. When a fast kernel disagrees with it,
the reference is what we trust.

## Layout

```
crates/
  whetstone-kernels/   CUDA sources + the only unsafe code in the project
  whetstone-core/      config, checkpoint loading, roofline model
  whetstone-quant/     quantizer and the .wstone container format
  whetstone-cli/       the `whetstone` binary
bench/                 evaluation, reference implementations, chat harness
docs/                  format spec, roadmap, release process
scripts/               deploy.sh / deploy.ps1, run.sh / run.bat
```

## Releases

Prebuilt packages are published on tag. Each contains the CLI, the Python
harness, the docs, and a `run.sh` / `run.bat` launcher:

```bash
tar xzf whetstone-*-linux-x86_64-sm75.tar.gz
cd whetstone-*-linux-x86_64-sm75
./run.sh doctor      # GPU, driver, binary, Python, model — all at once
./run.sh probe
```

Artifacts are named with the GPU architecture they were built for, because
Whetstone compiles for exactly one. Check yours with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`; for anything other than
`7.5`, build from source with `WHETSTONE_CUDA_ARCH=<cc>`.

Building a package locally:

```bash
./scripts/deploy.sh                 # Linux  -> dist/*.tar.gz + .sha256
.\scripts\deploy.ps1               # Windows -> dist\*.zip + .sha256
```

See [docs/RELEASES.md](docs/RELEASES.md) for the release process and versioning
policy, and [CHANGELOG.md](CHANGELOG.md) for what changed.

## Quality gates

No optimization is finished until it passes all of these. Speed without the gate
is not a result.

1. Kernel output matches an fp32 CPU reference within a stated tolerance
2. Top-1 agreement ≥ 99% against the fp16 reference on a fixed prompt set
3. wikitext-2 perplexity, reported as an absolute number and a delta
4. tok/s over ≥256 generated tokens after warmup, median with p10/p90

## License

Apache-2.0
