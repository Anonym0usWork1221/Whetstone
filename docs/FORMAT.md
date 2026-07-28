# The `.wstone` format

A `.wstone` file stores model weights **already arranged the way Whetstone's
kernels read them**. It is not a checkpoint that gets decoded at load time; the
conversion has already happened, and loading is an `mmap` plus a pointer walk.

## Why another format

`safetensors` and GGUF both solve "store tensors portably". That is a different
problem from "store an execution plan". Concretely, a `.wstone` fixes at
conversion time:

- **The bit layout.** int4 nibbles are packed eight to a `u32` in exactly the
  order the GEMV's `uint4` load consumes them. No shuffling at runtime.
- **The scale placement.** Per-group metadata is indexed by the same loop
  counter that walks the weights, so one metadata load serves a whole `uint4` of
  them and is uniform across the warp.
- **Alignment.** Every blob starts on a 256-byte boundary, so no vector load
  ever straddles a cache line and a future direct-I/O path stays possible.
- **The configuration.** The source `config.json` is embedded verbatim. A
  `.wstone` needs no sidecar files to be executed.

The cost of that specialisation is portability: no other runtime can execute a
`.wstone`, and it is not meant to. Use `safetensors` for interchange.

## Layout

```
offset  size        field
0       8           magic          "WHETSTON"
8       4           version        u32 LE, currently 1
12      4           flags          u32 LE, currently 0
16      8           header_len     u64 LE
24      8           header_hash    u64 LE, FNV-1a of the header bytes
32      header_len  header         UTF-8 JSON
        ...         padding        to a 256-byte boundary
        ...         payloads       each blob 256-byte aligned
```

Payloads are written first and the header is patched in afterwards, so
converting a model never requires holding it in RAM.

## Header

```json
{
  "format": "wstone",
  "version": 1,
  "producer": "whetstone-quant 0.1.0",
  "model_config": { "...": "the source config.json, verbatim" },
  "quant": {
    "scheme": "int4-hier-g32",
    "group": "32",
    "method": "kqx2-weighted-ls",
    "lm_head": "int4-hier-g32"
  },
  "tensors": [
    {
      "name": "model.layers.0.mlp.gate_proj.weight",
      "kind": "int4_g128",
      "shape": [4864, 896],
      "blobs": {
        "qw": { "offset": 4096,   "len": 2179072, "hash": 1234567890 },
        "sz": { "offset": 2183168, "len": 136192, "hash": 987654321 }
      }
    }
  ]
}
```

Tensor entries are sorted by name so the file is byte-reproducible.

## Tensor kinds

| kind | blobs | encoding |
|---|---|---|
| `fp16` | `data` | IEEE-754 binary16, row-major, dense |
| `fp32` | `data` | IEEE-754 binary32, row-major, dense |
| `int4_hier_g32` | `qw`, `si`, `sb` | int4, groups of 32, 4-bit scale/min indices against one `f16` pair per row — **the current default** |
| `int4_g128` | `qw`, `sz` | int4 asymmetric, groups of 128 along the input dimension — the 0.3.0 format |

### `int4_hier_g32`

The default since 0.4.0. For a `[out_features, in_features]` matrix:

- **`qw`** — `[out_features][in_features/8]` `u32`. Nibble `i` of a word holds
  the quantized value for column `8k+i`, in bits `4i..4i+3`. Identical to
  `int4_g128`.
- **`si`** — `[out_features][in_features/32]` `u8`. Low nibble is the scale
  index `ls`, high nibble is the minimum index `lm`.
- **`sb`** — `[out_features]` `u32`. Low 16 bits are an `f16` `d`, high 16 bits
  an `f16` `dmin`. One pair per row.

Reconstruction:

```
scale = d * ls          min = -dmin * lm          w = q * scale + min
```

Quantization, per row: fit `(scale, min)` for every group of 32 by llama.cpp's
`make_qkx2_quants` (sweep 21 candidate grids, refit by weighted least squares in
closed form for each, keep the best), then set `d = max_g scale_g / 15` and
`dmin = max_g (-min_g) / 15`, round both to `f16`, derive the indices, and
**re-assign the levels against the quantized parameters**. That last pass is not
optional — the kernel reconstructs with the stored `d*ls`, so choosing levels
against the unquantized fit bakes in an error the dequantizer cannot undo.

`ls` is clamped to **at least 1**. A zero scale index is representable — every
weight in the group would reconstruct to `min` — but it forces the quantizer to
special-case a division it otherwise does unconditionally, and a special case
that two implementations must agree on is a bug waiting for a rare tensor.

`in_features` must be a multiple of 32.

Effective cost:

```
bits/weight = 4 + 8/32 + 32/in_features
            = 4.286 at in=896,   4.257 at in=4864
```

**Why this replaced `int4_g128`.** Measured on Qwen2.5-0.5B over wikitext-2,
group size is worth about six times what the fitting algorithm is worth — group
128 → 64 buys 0.96 perplexity, the complete k-quant fit at fixed group size buys
0.16. Group 32 was unaffordable in the old layout because an `f16` scale plus an
`f16` zero per 32 weights costs 1.0 bits/weight of metadata against group 128's
0.25. Two 4-bit indices against a per-row `f16` pair costs **0.036**, and
measured **1.15 perplexity better**.

Note also that `w = q*scale + min` is not merely a reparameterisation of
`(q - z)*scale`. The old form needs `z = -min/scale` rounded to `f16` after
`scale` is, which rounds the same quantity twice — and the production kernel
subtracts `1024 + z` in `f16`, where the mantissa step at 1024 is exactly 1, so a
*fractional* zero point was being silently rounded to an integer whatever the
file said.

### `int4_g128`

For a `[out_features, in_features]` matrix:

- **`qw`** — `[out_features][in_features/8]` `u32`. Nibble `i` of a word holds
  the quantized value for column `8k+i`, in bits `4i..4i+3`.
- **`sz`** — `[out_features][in_features/128]` `u32`. The low 16 bits are an
  `f16` scale; the high 16 bits are an `f16` zero point.

Quantization, per group of 128:

```
s = (max - min) / 15
z = round(-min / s)
q = clamp(round(w/s) + z, 0, 15)
```

Reconstruction is `w = (q - z) * s`.

`s` and `z` are **rounded to f16 before** `q` is computed, so the values encoded
are exactly the ones the kernel reconstructs. Computing `q` against
full-precision scales and then storing rounded ones introduces an error the
dequantizer cannot undo — a subtle bug that costs accuracy for nothing.

`in_features` must be a multiple of 128. Tensors that are not are stored dense
rather than padded, because padding would change the arithmetic.

Effective cost is **4.25 bits/weight**: 4 for the value plus 32 bits of metadata
per 128 weights. Quoting it as "4-bit" understates bandwidth by 6%, and
bandwidth is what sets decode speed.

Kept in 0.4.0 so an A/B against the format it replaced is one flag
(`--body int4`), not because it is recommended.

## Integrity

Every blob carries a 64-bit FNV-1a-shaped hash, and the header carries one of
its own. `whetstone verify` checks all of them.

**The multiplier is not the FNV-1a prime.** `format.rs` uses
`0x1000_0000_01b3`, one hex digit longer than the real `0x100000001b3`. That was
a typo, and it survived because the only thing that ever checked the hash was the
implementation that produced it — an independent reimplementation of the
container is what surfaced it. It is deliberately **not** being corrected:
changing it would invalidate every existing file in exchange for nothing, since
any odd multiplier detects a flipped bit equally well. Anyone writing a
`.wstone` from another language must use `0x1000000001B3`.

This is deliberate. A truncated download or a flipped bit in a weight file does
not crash — it produces plausible-looking garbage tokens, and that failure mode
is expensive to diagnose. Turning it into an error at load time costs one linear
pass.

FNV-1a is not cryptographic and is not meant to be; the threat model is a bad
disk or an interrupted transfer, not an adversary.

## Validation on load

Header parsing treats the file as hostile. Before any offset is used to index
memory, the reader checks that:

- the magic and version match;
- the header hash matches;
- every blob's `offset + len` lies inside the file;
- every blob's offset is 256-byte aligned;
- each tensor carries the blobs its `kind` requires.

A malformed file must produce an error, never an out-of-bounds read.

## Compatibility

`version` is checked exactly. A reader refuses a file it was not built for
rather than guessing, and `flags` is reserved for additions that a reader can
safely ignore.

## Using it

```bash
# Build one. --head int4-hier also quantizes lm_head: the largest single
# bandwidth win available, and the largest quality risk. Measured, it costs
# +0.52 perplexity in this format against +1.10 in the 0.3.0 one.
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone --head int4-hier

# Check integrity, and fidelity against the source.
whetstone verify qwen05b.wstone --source /path/to/Qwen2.5-0.5B-Instruct
```

Measured on Qwen2.5-0.5B-Instruct:

| variant | size | bits/weight | Δ ppl | decode ceiling |
|---|---|---|---|---|
| source (bf16) | 988.1 MB | 16.00 | — | 281 tok/s |
| int4-hier body, fp16 head | 471.8 MB | 7.51 | +1.550 | 599 tok/s |
| int4-hier everywhere | 272.5 MB | 4.28 | +2.201 | **1051 tok/s** |
| int4-g128 everywhere (0.3.0) | 263.6 MB | 4.25 | +4.208 | 1059 tok/s |

Ceilings are at the measured 278 GB/s and count `lm_head`, which is 27.6% of
per-token traffic. **Weight error is not the objective** and `verify` says so —
a clip search that lowers it raises perplexity by 0.50, and GPTQ raises it while
lowering perplexity by 1.73. Run `whetstone ppl` before trusting a file.
