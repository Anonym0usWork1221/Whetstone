# Releasing

How a Whetstone release is cut, what each artifact contains, and how to use one.

---

## What a release contains

Each release publishes one archive per platform plus a combined `SHA256SUMS`.

| artifact | for |
|---|---|
| `whetstone-<ver>-linux-x86_64-sm75.tar.gz` | Linux, NVIDIA Turing |
| `whetstone-<ver>-windows-x86_64-sm75.zip` | Windows, NVIDIA Turing |

**The architecture is in the filename because it matters.** Whetstone compiles
for exactly one GPU family — its kernels use capabilities that differ by
architecture (`bmma.xor.popc` is sm_75+, `cp.async` is sm_80+) — so an `sm75`
build will not run on an older card. Shipping under a generic name would be
misleading.

Check your card:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

`compute_cap 7.5` → the `sm75` artifact. Anything else: build from source with
`WHETSTONE_CUDA_ARCH=<cc without the dot>`.

### Archive layout

```
whetstone-0.2.0-linux-x86_64-sm75/
├── bin/whetstone            the CLI (probe, inspect, convert, verify,
│                             run, ppl, logits, bench, tune)
├── bench/
│   ├── chat.py              live chat + throughput benchmark
│   ├── baseline_hf.py       HF baseline: tok/s, bandwidth, perplexity
│   ├── reference_numpy.py   independent fp64 forward pass
│   ├── tokenizer.py         byte-level BPE from tokenizer.json
│   ├── prepare_tokens.py    materialise an evaluation token stream
│   └── download_model.py    fetch the reference model
├── docs/{FORMAT,ROADMAP}.md
├── run.sh                   one launcher for everything
├── VERSION                  version, commit, build date, target, sm_, nvcc, rustc
├── REQUIREMENTS.txt         what the machine needs
├── README.md
├── CHANGELOG.md
└── LICENSE
```

`bin/whetstone` is self-contained apart from `libcudart`. The Python harness is
optional and needs its own environment — `./run.sh setup` builds one.

---

## Using a release

```bash
tar xzf whetstone-0.2.0-linux-x86_64-sm75.tar.gz
cd whetstone-0.2.0-linux-x86_64-sm75

./run.sh doctor      # GPU, driver, binary, Python, model — all at once
./run.sh probe       # measured throughput of every arithmetic path
```

Windows is the same with `.\run.bat`.

Full workflow:

```bash
./run.sh setup                                   # one-time Python env
./run.sh download                                # Qwen2.5-0.5B-Instruct
./run.sh inspect  models/Qwen2.5-0.5B-Instruct   # architecture + roofline
./run.sh convert  models/Qwen2.5-0.5B-Instruct model.wstone
./run.sh verify   model.wstone models/Qwen2.5-0.5B-Instruct
./run.sh chat                                    # live chat, tok/s readout
./run.sh bench                                   # throughput run
```

### Verifying a download

```bash
sha256sum -c SHA256SUMS
```

```powershell
(Get-FileHash -Algorithm SHA256 whetstone-0.2.0-windows-x86_64-sm75.zip).Hash
```

---

## Cutting a release

### 1. Verify on real hardware first

CI runners have **no GPU**. They prove the code compiles and that CPU-only tests
pass; they cannot prove a kernel is correct. So before tagging, on a real card:

```bash
cargo test --release                      # correctness, must pass
cargo test --release -- --ignored         # performance, recorded not enforced
./scripts/deploy.sh                       # full build + package
```

And a real end-to-end check:

```bash
./target/release/whetstone convert ../models/Qwen2.5-0.5B-Instruct -o /tmp/m.wstone
./target/release/whetstone verify /tmp/m.wstone --source ../models/Qwen2.5-0.5B-Instruct
```

### 2. Update the version and changelog

Both are checked by CI, and a mismatch fails the release before anything builds.

```bash
# Cargo.toml [workspace.package]
version = "0.3.0"

# CHANGELOG.md: move Unreleased items under a new ## [0.3.0] — YYYY-MM-DD
```

Record measured numbers in the changelog entry, not adjectives. "431.8 tok/s
against llama.cpp's 282.95" is checkable; "much faster" is not. Record the
**quality** number in the same entry — a release that is 1.5x faster and 2.75
perplexity worse is a trade the reader has to be able to see.

### 3. Tag and push

```bash
git add -A
git commit -m "Release 0.3.0"
git tag -a v0.3.0 -m "Whetstone 0.3.0"
git push origin main
git push origin v0.3.0
```

The tag push triggers `.github/workflows/release.yml`, which:

1. checks the tag matches `Cargo.toml` and that `CHANGELOG.md` has a section for it;
2. builds and packages on Linux and Windows;
3. publishes a release with notes assembled from the changelog section plus
   install and usage instructions.

Pre-release tags (`v0.2.0-rc1`) are marked as pre-releases automatically.

### 4. If something goes wrong

A tag can be moved before anyone depends on it:

```bash
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
```

Once a release is public, do not move the tag — cut a patch release instead.

---

## Versioning

Semantic versioning, with one project-specific rule.

**The `.wstone` format version is independent of the crate version.** It lives
in `format::VERSION` and is checked exactly on load: a reader refuses a file it
was not built for rather than guessing at a layout. A format change is a
breaking change for weight files even if the crate version is only a minor bump,
so it is always called out explicitly in the changelog.

| change | crate version | format version |
|---|---|---|
| new CLI flag, faster kernel | minor / patch | unchanged |
| new tensor `kind` added, old files still load | minor | unchanged |
| existing layout changes meaning | major | **bumped** |

Practical consequence: **re-run `whetstone convert` after upgrading across a
format bump.** `whetstone verify` will tell you if a file is from the wrong
version rather than producing garbage.

---

## Building for a different GPU

```bash
WHETSTONE_CUDA_ARCH=86 ./scripts/deploy.sh --arch 86     # Ampere
WHETSTONE_CUDA_ARCH=89 ./scripts/deploy.sh --arch 89     # Ada
```

The kernels are written against sm_75 and will *compile* for newer
architectures, but they will not use anything newer than Turing offers —
`cp.async`, `bmma.and.popc` and 2:4 sparsity are all left on the table. Expect
correct results and unremarkable performance until the kernels grow
architecture-specific paths.
