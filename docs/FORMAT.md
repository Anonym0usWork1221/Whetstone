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
- **The scale placement.** Per-group scale and zero live as two `f16` halves of
  a single `u32`, indexed by the same loop counter that walks the weights, so
  one metadata load serves 32 weights and is uniform across the warp.
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
    "scheme": "int4-g128-asymmetric",
    "group": "128",
    "method": "rtn",
    "lm_head": "fp16"
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
| `int4_g128` | `qw`, `sz` | int4 asymmetric, groups of 128 along the input dimension |

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

## Integrity

Every blob carries an FNV-1a-64 hash, and the header carries one of its own.
`whetstone verify` checks all of them.

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
# Build one. --head int4 also quantizes lm_head: the largest single
# bandwidth win available, and the largest quality risk.
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone
whetstone convert /path/to/Qwen2.5-0.5B-Instruct -o qwen05b.wstone --head int4

# Check integrity, and fidelity against the source.
whetstone verify qwen05b.wstone --source /path/to/Qwen2.5-0.5B-Instruct
```

Measured on Qwen2.5-0.5B-Instruct:

| variant | size | bits/weight | mean rel. error | decode ceiling |
|---|---|---|---|---|
| source (bf16) | 988.1 MB | 16.00 | — | 281 tok/s |
| int4 body, fp16 head | 463.6 MB | 7.49 | 0.1102 | 601 tok/s |
| int4 everywhere | 263.6 MB | 4.25 | 0.1095 | **1059 tok/s** |

Ceilings are at the measured 278 GB/s and count `lm_head`, which is 27.6% of
per-token traffic. Weight error is not the objective — run the quality gate
(top-1 agreement, wikitext-2 perplexity) before trusting a file.
