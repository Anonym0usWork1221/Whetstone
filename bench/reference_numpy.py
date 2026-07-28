#!/usr/bin/env python3
"""An independent fp64 reference implementation of the Qwen2 forward pass.

Whetstone's CUDA kernels need ground truth to be diffed against. Using
HuggingFace for that is circular in one important way: if we misread the
checkpoint layout, HF and Whetstone can disagree for reasons that have nothing
to do with the kernels. This file reads the safetensors directly and implements
the architecture from the config alone, in numpy, at float64.

It is deliberately slow and obvious. Every operation is written the textbook
way, with no fusion and no numerical shortcuts, so that when a fast kernel
disagrees with it, the reference is the thing we trust.

    python bench/reference_numpy.py --model ../models/Qwen2.5-0.5B-Instruct \
                                    --out ../research/experiments/reference_fp64.json

Cost: ~1-2 s per token on CPU for the 0.5B model. Fine for a handful of prompts.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------- checkpoint

class Checkpoint:
    """Minimal safetensors reader. Mirrors crates/whetstone-core/src/safetensors.rs."""

    _DT = {
        "F64": (np.float64, 8), "F32": (np.float32, 4),
        "F16": (np.float16, 2), "BF16": (None, 2),
        "I64": (np.int64, 8), "I32": (np.int32, 4),
        "I8": (np.int8, 1), "U8": (np.uint8, 1),
    }

    def __init__(self, path: Path):
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")
        n = struct.unpack("<Q", bytes(self.raw[:8]))[0]
        self.header = json.loads(bytes(self.raw[8:8 + n]))
        self.base = 8 + n

    def names(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name: str) -> np.ndarray:
        """Returns the tensor as float64. bf16 widens by a bit shift, exactly."""
        if name not in self.header:
            raise KeyError(f"{name} not in checkpoint")
        e = self.header[name]
        lo, hi = e["data_offsets"]
        buf = self.raw[self.base + lo:self.base + hi]

        if e["dtype"] == "BF16":
            # bf16 is the top 16 bits of an f32: widen, do not approximate.
            u16 = np.frombuffer(buf.tobytes(), dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32)
        else:
            np_dt, _ = self._DT[e["dtype"]]
            arr = np.frombuffer(buf.tobytes(), dtype=np_dt)

        return arr.astype(np.float64).reshape(e["shape"])


# ------------------------------------------------------------------- layers

def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * w.

    Note there is no mean subtraction and no bias -- that is what distinguishes
    it from LayerNorm, and it is why a fast reciprocal-sqrt is the only
    transcendental in the whole operation.
    """
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * w


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: x * sigmoid(x). Smooth, and never exactly zero for x != 0,
    which is why Qwen has no exploitable activation sparsity."""
    return x / (1.0 + np.exp(-x))


def rope(x: np.ndarray, pos: np.ndarray, theta: float) -> np.ndarray:
    """Rotary position embedding, applied to (seq, heads, head_dim).

    HuggingFace uses the "half rotation" layout: the vector is split in two
    halves and rotated pairwise across them, NOT as adjacent (even, odd) pairs.
    Getting this wrong produces a model that still generates fluent text but
    with subtly wrong long-range behaviour, so it is worth being explicit.
    """
    seq, heads, hd = x.shape
    half = hd // 2

    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
    ang = pos[:, None].astype(np.float64) * inv[None, :]     # (seq, half)
    cos, sin = np.cos(ang), np.sin(ang)

    cos = cos[:, None, :]
    sin = sin[:, None, :]

    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-subtracted softmax. The subtraction is what keeps exp() in range."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# -------------------------------------------------------------------- model

class ReferenceModel:
    """Qwen2 forward pass, float64, no shortcuts."""

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.ck = Checkpoint(model_dir / "model.safetensors")

        c = self.cfg
        self.L = c["num_hidden_layers"]
        self.H = c["hidden_size"]
        self.nq = c["num_attention_heads"]
        self.nkv = c.get("num_key_value_heads", self.nq)
        self.hd = c.get("head_dim", self.H // self.nq)
        self.eps = c["rms_norm_eps"]
        self.theta = float(c["rope_theta"])
        self.tied = c.get("tie_word_embeddings", False)
        self.gqa = self.nq // self.nkv

        self._cache: dict[str, np.ndarray] = {}

    def w(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = self.ck.get(name)
        return self._cache[name]

    def forward(self, ids: list[int]) -> np.ndarray:
        """Runs the full stack and returns final-position logits."""
        c = self.cfg
        pos = np.arange(len(ids))

        # Embedding lookup.
        x = self.w("model.embed_tokens.weight")[ids]           # (seq, H)

        # Causal mask, additive.
        seq = len(ids)
        mask = np.triu(np.full((seq, seq), -np.inf), k=1)

        for l in range(self.L):
            p = f"model.layers.{l}"

            # ---- attention ----
            h = rms_norm(x, self.w(f"{p}.input_layernorm.weight"), self.eps)

            q = h @ self.w(f"{p}.self_attn.q_proj.weight").T + self.w(f"{p}.self_attn.q_proj.bias")
            k = h @ self.w(f"{p}.self_attn.k_proj.weight").T + self.w(f"{p}.self_attn.k_proj.bias")
            v = h @ self.w(f"{p}.self_attn.v_proj.weight").T + self.w(f"{p}.self_attn.v_proj.bias")

            q = rope(q.reshape(seq, self.nq, self.hd), pos, self.theta)
            k = rope(k.reshape(seq, self.nkv, self.hd), pos, self.theta)
            v = v.reshape(seq, self.nkv, self.hd)

            # GQA: each KV head serves `gqa` query heads.
            k = np.repeat(k, self.gqa, axis=1)
            v = np.repeat(v, self.gqa, axis=1)

            # (heads, seq, seq)
            scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(self.hd)
            attn = softmax(scores + mask[None, :, :], axis=-1)
            ctx = np.einsum("hqk,khd->qhd", attn, v).reshape(seq, self.nq * self.hd)

            x = x + ctx @ self.w(f"{p}.self_attn.o_proj.weight").T

            # ---- MLP (SwiGLU) ----
            h = rms_norm(x, self.w(f"{p}.post_attention_layernorm.weight"), self.eps)
            gate = h @ self.w(f"{p}.mlp.gate_proj.weight").T
            up = h @ self.w(f"{p}.mlp.up_proj.weight").T
            x = x + (silu(gate) * up) @ self.w(f"{p}.mlp.down_proj.weight").T

        x = rms_norm(x, self.w("model.norm.weight"), self.eps)

        head = "model.embed_tokens.weight" if self.tied else "lm_head.weight"
        return x[-1] @ self.w(head).T

    def free(self) -> None:
        self._cache.clear()


# --------------------------------------------------------------------- main

# Pre-tokenized so this script needs no tokenizer dependency.
#
# These ids were resolved by direct lookup in the checkpoint's own vocab.json
# rather than written by hand, so the label and the ids are guaranteed to
# correspond. (Hand-written ids are a trap: an id sequence that decodes to
# something other than its label still produces a perfectly valid reference,
# so the mistake is invisible until someone reads the decoded tokens.)
PROMPT_IDS = {
    "The capital of France is": [785, 6722, 315, 9625, 374],
    "def Fibonacci(n):": [750, 79683, 1445, 1648],
    "import numpy as np": [474, 8591, 438, 2595],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ids", default=None,
                    help="comma-separated token ids to run instead of the built-ins")
    args = ap.parse_args()

    model_dir = Path(args.model)
    print(f"loading {model_dir} ...")
    m = ReferenceModel(model_dir)
    print(f"  {m.L} layers, hidden {m.H}, {m.nq}Q/{m.nkv}KV heads, "
          f"head_dim {m.hd}, GQA {m.gqa}:1, tied={m.tied}")

    cases = ({"custom": [int(x) for x in args.ids.split(",")]}
             if args.ids else PROMPT_IDS)

    out = []
    for text, ids in cases.items():
        t0 = time.perf_counter()
        logits = m.forward(ids)
        dt = time.perf_counter() - t0

        p = softmax(logits)
        order = np.argsort(-p)[:args.topk]

        rec = {
            "prompt": text,
            "ids": ids,
            "top1_id": int(order[0]),
            "topk_ids": [int(i) for i in order],
            "topk_probs": [float(p[i]) for i in order],
            "logits_mean": float(logits.mean()),
            "logits_std": float(logits.std()),
            "logits_max": float(logits.max()),
            "entropy": float(-(p * np.log(np.clip(p, 1e-300, None))).sum()),
            "seconds": dt,
        }
        out.append(rec)
        print(f"  {text!r:<40} -> id {rec['top1_id']:<7} "
              f"p={rec['topk_probs'][0]:.4f}  H={rec['entropy']:.3f}  ({dt:.1f}s)")

    if args.out:
        pth = Path(args.out)
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text(json.dumps(
            {"model": str(model_dir), "dtype": "float64", "impl": "numpy-reference",
             "results": out}, indent=2))
        print(f"\nwrote {pth}")

    print("\nThis is the ground truth. A Whetstone kernel that disagrees with it")
    print("is wrong, regardless of what any other implementation says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
