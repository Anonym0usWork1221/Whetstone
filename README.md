# Whetstone

**An experiment in making LLM inference much, much faster on cheap GPUs — by
replacing the arithmetic inside existing transformer layers with sub-byte
integer formats, bit-packed weights, and XOR/popcount dot products.**

Whetstone does not invent a new architecture. It keeps trained weights and layer
topology exactly as they are, and replaces **the math that evaluates them** —
then measures, at every step, whether that actually made anything faster.

- **Reference model:** Qwen2.5-0.5B-Instruct (494 M params, 988 MB bf16)
- **Reference GPU:** NVIDIA RTX 2060 — `sm_75` Turing, 30 SMs, 6 GB, 336 GB/s
- **Stack:** CUDA C++ kernels, Rust engine, Python for evaluation

> **Status (0.5.0): 1.46× llama.cpp Q4_K_M on the reference GPU, at 2.1× its
> quantization damage.** 414.0 tok/s against 283.8, and +0.82 perplexity against
> Q4_K_M's +0.40 — measured in one harness, on llama.cpp's own weights, so the
> two deltas are the same measurement. 0.3.0 was 1.53× at **10.6×** the damage;
> the difference is a new weight format that costs 0.03 bits/weight.
>
> **New in 0.5.0 — one pass over the weights, many tokens.** A multi-token
> ("chunk") kernel path makes **prefill 3.9× faster** (408.9 → 1288.5 tok/s,
> output byte-identical), and the same kernels are what make two new flags work:
> `--vram` runs a model **larger than VRAM** by leaving whole blocks in host RAM,
> and `--spec` verifies an n-gram draft in one chunk pass. Offloaded, those two
> together are worth **3.9×** on text that repeats itself. See
> [Offload and speculation](#offload-and-speculation).

> **This is a research project, not a production engine.** If you want to run a
> quantized model today, use [llama.cpp](https://github.com/ggml-org/llama.cpp)
> — it is mature, fast, and supports this hardware. Read
> [Prior art](#prior-art--what-is-actually-new-here) for an honest account of
> what here is genuinely new and what is a reimplementation of solved problems.

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
| **int4 hier-g32** *(current)* | **4.28** | **264.4 MB** | **1051** |
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
3. **Shrink the block weights** to int4 — the remaining 72.4%
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

> **That "100% top-1 agreement" for int4-g128 is exactly the trap this table
> sets.** It was measured on three prompts at one position each. Over 40,940
> predictions the same format costs **+2.73 perplexity** — a fifth of the
> model's quality. The argmax stays put long after the distribution has moved.
> Both columns above are kept because the *ordering* is informative and the
> ternary result is decisive, but neither is a quality gate. Use
> `whetstone ppl`.

**Round-to-nearest ternary destroys this model.** That is not a contradiction of
the BitNet results — BitNet *trains* in ternary with a straight-through
estimator, so its weights are built to be representable on that grid. Weights
trained in bf16 were never constrained that way, and a 0.5B model has far less
redundancy to spend than a 7B one.

The open question is therefore not "how few bits" but **how much of that gap a
better quantizer recovers — and the answer turned out to be most of it.** At a
fixed 4 bits, on the transformer body, perplexity delta against fp16:

| quantizer, all at ~4.25–4.28 bits/weight | Δ ppl |
|---|---|
| round-to-nearest, group 128 *(0.3.0)* | +2.730 |
| llama.cpp's complete k-quant fitted scale/min search | +2.575 |
| **group 32 with hierarchical scale metadata** *(0.4.0)* | **+1.575** |
| **the same, plus GPTQ at 131k calibration tokens** | **+0.668** |

Two things did that, and neither is the one you would guess:

**Group size, not the fitting algorithm.** Halving the group buys 0.96
perplexity; replacing round-to-nearest with the full k-quant alternating
least-squares fit buys 0.16. Group 32 was previously unaffordable because an
`f16` scale plus an `f16` zero per 32 weights is 1.0 bits/weight of metadata
against group 128's 0.25 — so 0.4.0 stores two 4-bit *indices* per group against
one `f16` pair per row instead, and gets group 32 for **0.03** bits/weight.

**Weight error is the wrong objective.** A one-parameter clip search that
provably *lowers* mean weight error from 0.1102 to 0.1067 *raises* perplexity by
0.50. GPTQ does the opposite — it *raises* weight error to 0.1416 and lowers
perplexity by 1.73. Anything that minimises `‖W − Ŵ‖` without reference to what
multiplies `W` is, at this model size, as likely to hurt as help.

The earlier GPTQ result here was recorded as inconclusive. It was never a test:
the calibration set was **293 tokens** against Hessians of dimension 896 and
4864, so every `H = 2XᵀX` was rank-deficient and dominated by its damping term.
At 131,072 tokens it is the single largest lever in the project.

> **GPTQ's gain is substantially in-domain.** Calibrated on held-out wikitext
> and evaluated on wikitext it is worth −0.91. Calibrated on 131k tokens of
> C/C++ source and evaluated on wikitext it reads **+2.27 — worse than not
> running it at all**. The inverse Hessian is a claim about which input
> directions matter, and code and Wikipedia disagree about that. So `convert`
> ships the data-free format and GPTQ stays an opt-in offline step.

---

## Prior art — what is actually new here

Being straight about this, because a repo that overstates its novelty wastes
everyone's time.

### What is not new

Essentially the whole current pipeline is a reimplementation of solved problems:

| in this repo | who did it first, and better |
|---|---|
| int4 group quantization with hierarchical scales | **GGUF `Q4_K`** does exactly this — 6-bit sub-scales against a super-block `f16` pair. Whetstone's variant differs in the block geometry (a whole row, 4-bit indices, group 32) because `hidden = 896` is not a multiple of `QK_K`, not because the idea is new. |
| GPTQ error compensation | **GPTQ** (2022). Reimplemented, not invented. |
| the `.wstone` container | **GGUF** — self-describing, mmap-able, aligned, embedded config, and an actual ecosystem. `.wstone` is a GGUF with fewer features and no users. |
| int4 decode GEMV kernel | **llama.cpp** `mul_mat_vec_q`, **exllamav2**, AWQ kernels — all faster and battle-tested. |
| HF → quantized converter | `convert_hf_to_gguf.py` + `llama-quantize`, AutoGPTQ, AutoAWQ, optimum. |
| chat CLI with a tok/s readout | `llama-cli`, ollama, LM Studio. |
| an inference engine in Rust | **candle**, **mistral.rs**, burn. |

**Converting a model to 4 bits is a commodity.** If that is what you need, use
llama.cpp.

### What is uncommon

Three things, honestly ranked.

**1. Turing (`sm_75`) has been abandoned by every serious engine.** Verified
against primary sources:

| engine | on sm_75 |
|---|---|
| Marlin | requires `cc >= 8.0` and `cp.async` — **vLLM refuses it on sm_75** |
| TensorRT-LLM | Turing **removed** from the support matrix |
| BitBLAS / Ladder | codegen targets sm_80+ |
| FlashAttention-2 / 3 | sm_80+ |

Only llama.cpp still supports this hardware properly. Every modern low-bit
kernel is architected around `cp.async`, which Turing does not have — so the
software pipeline has to be re-derived from scratch with register
double-buffering. That is a genuine gap, though it exists because the hardware
is old rather than because the problem is hard.

**2. Turing's INT4 and INT1 tensor cores are unexploited by anyone.** No
production engine uses them. llama.cpp's Turing path uses INT8 IMMA; even its
1-bit `Q1_0` format unpacks to `int8` and uses `dp4a` — no `popc`, no
`bmma_sync`. NVIDIA deprecated `s4`/`b1` after Turing.

The 610 TOPS binary path measured here is real and verified on-device. The
catch, established by this project's own measurements, is that it does **not**
help decode, because decode is bandwidth-bound. It can only pay off in prefill
or speculative verification, where the regime is compute-bound. That is the
most defensible direction available.

**3. The measurements, including the ones that failed.** Most projects publish
what worked. This one publishes the bits-versus-quality curve on real weights,
the negative results with numbers (round-to-nearest ternary destroys this model:
KL 10.9 nats, 0% top-1 agreement), and a log of four published figures that
turned out to be wrong — a roofline that omitted `lm_head`, a 32× units error on
`dp4a`/`popc`, an unstated half-rate fp16 baseline, and a CPU-starved benchmark.

If you are about to try "just make the weights 1-bit," the data here will save
you a week.

### Measured against llama.cpp on the same GPU

Not an estimate — llama.cpp built from source for `sm_75` and run on the same
RTX 2060 with the same checkpoint (`llama-bench`, 3 repetitions):

| engine / format | bits/wt | bytes/token | decode tok/s | ppl | Δ vs fp16 |
|---|---|---|---|---|---|
| HuggingFace fp16 | 16.00 | 988 MB | 40.3 | 13.8182 | *(anchor)* |
| **llama.cpp Q4_K_M** | 6.35 | 392 MB | **283.8** | 14.2138 | **+0.3957** |
| Whetstone fp16 | 16.00 | 988 MB | 211.6 | 13.8209 | +0.0028 |
| Whetstone int4-g128 *(0.3.0)* | 4.25 | 262 MB | **434.1** | 18.0287 | +4.2078 |
| Whetstone int4-hier-g32 | 4.28 | 264 MB | 415.2 | 16.0220 | +2.2011 |
| **Whetstone int4-hier-g32 + GPTQ** | 4.28 | 264 MB | **414.0** | 14.6383 | **+0.8174** |

Engines are **interleaved** — one sample of each, round-robin — because measuring
all of A then all of B compares A cold to B hot. An earlier run of this harness
did exactly that and read llama.cpp at 250.8 instead of 281.9, inflating the
speed ratio from 1.50× to 1.69×.

**Q4_K_M's perplexity is llama.cpp's own weights measured in this harness**, not
a number quoted from `llama-perplexity`, so every row above is the same
measurement. That matters more than it sounds: `llama-perplexity` scores only
the **second half** of each window (`perplexity.cpp:542`,
`const int first = n_ctx/2`), so every token it grades carries ≥1024 tokens of
context. Matching that rule here makes the same fp16 weights read 12.2462
against `llama-perplexity`'s 12.2484 — **the entire 1.57-point "harness offset"
that earlier versions of this README attributed to tokenization is the scoring
protocol.** `bench/gguf_ppl.py` in the research tree dequantizes a `.gguf` with
llama.cpp's own `gguf-py` and runs it here, which removes the argument entirely.

**Speed: 1.46× llama.cpp Q4_K_M**, from reading 1.49× fewer bytes per token. The
speed ratio tracks the byte ratio, which is what the roofline says should happen.

**Quality: 2.1× its damage**, down from 10.6× in 0.3.0.

And a correction to what this README used to claim. It said the gap was "the
rounding, not the budget", on the grounds that Whetstone's 7.49-bit variant lost
more than Q4_K_M at 6.35. That compared a total containing an *fp16* head
against a total containing an *8.5-bit* head. Read per tensor out of the file:
k-quants need the row length to be a multiple of `QK_K = 256`, and Qwen2.5-0.5B
has `hidden = 896` — so **896 mod 256 = 128**, every projection except
`down_proj` falls back to `Q5_0` at 5.50 bpw, and the tied head to `Q8_0` at
8.50. Q4_K_M's body is **5.53** bits/weight against Whetstone's 4.25. It was
substantially the budget.

It was not *only* the budget, and the remaining part is what 0.4.0 closes. See
[docs/ROADMAP.md](docs/ROADMAP.md).

Whetstone's fp16 path is the control: **13.8209 against HuggingFace's 13.8182**
on the same 40,940 predictions — a 0.02% difference. That is what says the
engine is right and the perplexity gap is the quantizer, not a bug.

*(An earlier version of this README projected "~445 tok/s, about 1.5×
llama.cpp" from the roofline. Measured: 434.1 and 1.53×, before 0.4.0 spent 4.6%
of it on a better weight format.)*

**Reproduce this yourself:**

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-bench llama-quantize
python convert_hf_to_gguf.py /path/to/Qwen2.5-0.5B-Instruct --outfile f16.gguf --outtype f16
./build/bin/llama-quantize f16.gguf q4km.gguf Q4_K_M
./build/bin/llama-bench -m q4km.gguf -m f16.gguf -p 512 -n 128 -ngl 99
```

### So what is this for

An experiment in **what actually governs inference speed on cheap hardware**,
with the measurements to back every claim. The headline result so far is not
about exotic arithmetic at all:

> The baseline reaches **11% of its own roofline**. Roughly 24 ms of every 27 ms
> token is framework overhead, not memory traffic. Removing that is worth ~9×
> and costs *zero* accuracy — more than quantization, and safer.

That finding is the opposite of where this project started, and it came from
measuring the baseline instead of assuming it was near-optimal. The comparison
table above shows the humbling half of it: llama.cpp has already banked that 9×.
The remaining edge is 1.49× fewer bytes per token, and as of 0.4.0 it is
realised — at a quality cost that is now within about 2× of the competition
rather than 10×.

The second finding, from 0.4.0, is smaller but more transferable: **three
different cheap proxies for quality have each been wrong here, in a way that
would have changed what got built.** Top-1 agreement on a few prompts missed a
2.73-perplexity regression. Weight relative error moves in the *opposite*
direction to quality under two different techniques. And a perplexity delta
borrowed from another engine's harness was off by 1.57 because that harness
scores only half of each window. Each was caught by measuring the thing itself
instead of the proxy, and each had already been used to make a decision.

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
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone --head int4-hier

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

| variant | size | bits/weight | Δ ppl | decode ceiling | measured |
|---|---|---|---|---|---|
| source (bf16) | 988.1 MB | 16.00 | — | 281 tok/s | 211.6 |
| int4-hier body, fp16 head | 471.8 MB | 7.51 | +1.550 | 599 tok/s | — |
| **int4-hier everywhere** | **272.5 MB** | **4.28** | **+2.201** | **1051 tok/s** | **415.2** |
| **+ GPTQ (offline, opt-in)** | 272.5 MB | 4.28 | **+0.817** | 1051 tok/s | 414.0 |

`--body int4` still selects the group-128 format from 0.3.0 (4.25 bpw, +4.208)
so an A/B against it is one flag.

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

### Running the engine

The binary is a normal cargo artifact — `./target/release/whetstone`, or
`cargo install --path crates/whetstone-cli` to put `whetstone` on your PATH.

```bash
# Convert once, then execute with no Python in the token loop.
# The tokenizer is embedded, so the .wstone needs no sidecar files.
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o model.wstone --head int4-hier

# Interactive chat, throughput reported per turn
whetstone chat model.wstone

# Or a single generation from token ids, for timing without a tokenizer in the loop
whetstone run model.wstone --ids 785,6722,315,9625,374 --max-new 256 --graph

# Quality gate: perplexity over a fixed token stream, comparable to any
# harness reading the same file
python bench/prepare_tokens.py --model /path/to/model --out wikitext2.u32
whetstone ppl model.wstone --tokens wikitext2.u32 --window 2048 --windows 20

# Where the time goes, and which kernel to use for each shape
whetstone run model.wstone --ids 785 --profile 64
whetstone tune model.wstone
```

### Offload and speculation

Two flags, on both `run` and `chat`. Neither changes what the model outputs.

```bash
# Run a model that does not fit: whole blocks past the budget stay in host RAM
# and the kernels read them over PCIe. Frees VRAM for a longer context.
whetstone chat model.wstone --ctx 32768 --vram 1200MB

# Speculative decoding: an n-gram draft, verified in one multi-token pass.
# A draft token is accepted only when it equals the model's own argmax, so the
# output is exactly what greedy decoding produces. Needs --temperature 0.
whetstone chat model.wstone --temperature 0 --spec 8

# The two together, which is where the win actually is
whetstone chat model.wstone --ctx 32768 --vram 1200MB --temperature 0 --spec 8
```

Qwen2.5-3B at 4.26 bpw (1644 MB), 96 generated tokens, greedy, median of three
interleaved runs — `research/experiments/bench_spec_offload.sh`:

| | prose | repetitive |
|---|---|---|
| resident | 96.8 tok/s | 90.7 tok/s |
| resident, `--spec 8` | **93.7** (0.97×) | 141.1 (1.56×) |
| `--vram 1200MB` (11 of 36 blocks off-card) | 11.9 | 11.7 |
| `--vram 1200MB --spec 8` | 13.4 (1.13×) | **45.8 (3.92×)** |

**Read the first row before the last one.** On open-ended prose with the whole
model resident, `--spec` is a small net *loss*: an n-gram draft only fires when
the text repeats itself, so nearly every round finds nothing, falls back to an
ordinary decode step, and pays the bookkeeping anyway. It is off by default and
it is workload-dependent, not free.

What the table does show is that **offload and speculation multiply each other.**
A 16-token pass costs 4.93 single-token passes with the weights in VRAM — so
speculation can never beat 3.25× there however good the draft — but only **1.07**
with them in host RAM, because one 6 GB/s PCIe read serves every token in the
chunk. The flag that is worth 1.56× resident is worth 3.9× offloaded.

**What offload cannot do.** Host RAM is ~46× slower than VRAM on this machine
(6.0 GB/s over a PCIe 3.0 x8 link against 278 GB/s measured on the card). Bits
per weight divide both, so no quantizer closes that gap, and a model half
off-card runs at roughly a tenth of resident speed at batch 1. Offload buys the
ability to run at all, and to spend VRAM on context instead of weights;
speculation is what buys back the speed, and only on text where the draft lands.
The full arithmetic, including why trillion-parameter models are a host-RAM
*capacity* wall long before they are a bandwidth one, is in
`research/notes/2026-07-29-offload-roofline.md`.

`whetstone chat` keeps the KV cache across turns, so turn twenty prefills only
its own message rather than the whole transcript, and each reply reports its own
tokens/second:

```text
> What is the capital of Japan?
The capital of Japan is Tokyo.
  [419.4 tok/s · 7 tokens in 0.02 s · prefill 26 in 62 ms · 110 GB/s · ctx 33]

> What is its population, roughly?
As of 2021, the estimated population of Tokyo is around 11 million.
  [448.4 tok/s · 20 tokens in 0.04 s · prefill 17 in 41 ms · 118 GB/s · ctx 70]
```

Note the second turn prefills 17 tokens, not 43 — the first turn is still in the
cache.

`--temperature 0` is the fastest path at **476 tok/s** on a warm cache: greedy
decode never
leaves the GPU, because the argmax writes into the device cursor that the next
step's embedding gather reads. Sampling runs at **369 tok/s** — it needs the
distribution on the host, which is a 608 KB copy plus an O(vocab) selection per
token.

`--body fp16` produces a lossless reference model. Keeping one runnable at all
times is what separates "the engine is wrong" from "the quantizer is lossy" —
two failures that look identical from a perplexity number alone.

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
2. **wikitext-2 perplexity**, absolute and as a delta against fp16 in the *same*
   harness — this is the gate, and the only one that has never misled us
3. tok/s over ≥256 generated tokens after warmup, median with p10/p90,
   **interleaved** with anything it is being compared against
4. For a numerics change, **bit-identical token ids** from a fixed prompt

Two things that look like quality gates and are not:

- **Top-1 agreement on a few prompts.** int4-g128 read "100% top-1" on three
  prompts and costs +2.73 perplexity over 40,940 predictions.
- **Weight relative error.** A clip search that lowers it raises perplexity by
  0.50; GPTQ raises it and lowers perplexity by 1.73. `convert` still prints it,
  as a smoke test for a broken packer and nothing more.

## License

Apache-2.0
