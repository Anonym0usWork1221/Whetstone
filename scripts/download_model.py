#!/usr/bin/env python3
"""Fetch the reference model and print its architecture facts.

Whetstone's reference model is Qwen2.5-0.5B-Instruct. Everything the engine does
is validated against this checkpoint first.

    python scripts/download_model.py --out ../models

The output directory is deliberately outside the repo: checkpoints are large and
must never be committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

# Everything needed for inference. We skip .bin duplicates and training junk.
ALLOW = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "merges.txt",
    "vocab.json",
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def describe(model_dir: Path) -> dict:
    """Read config.json and derive the numbers that govern our roofline."""
    cfg = json.loads((model_dir / "config.json").read_text())

    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_q)
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    head_dim = cfg.get("head_dim", h // n_q)
    tied = cfg.get("tie_word_embeddings", False)

    # Per-layer weight counts (Qwen2 uses biases on q/k/v but not o, and SwiGLU MLP).
    q = h * (n_q * head_dim)
    k = h * (n_kv * head_dim)
    v = h * (n_kv * head_dim)
    o = (n_q * head_dim) * h
    attn = q + k + v + o
    mlp = 3 * h * inter  # gate, up, down
    per_layer = attn + mlp

    embed = vocab * h
    non_embed = L * per_layer
    total = non_embed + embed + (0 if tied else embed)

    return {
        "config": cfg,
        "layers": L,
        "hidden": h,
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "intermediate": inter,
        "vocab": vocab,
        "tied_embeddings": tied,
        "params_attn_per_layer": attn,
        "params_mlp_per_layer": mlp,
        "params_per_layer": per_layer,
        "params_non_embed": non_embed,
        "params_embed": embed,
        "params_total": total,
    }


def roofline(d: dict, bandwidth_gbs: float = 336.0) -> None:
    """The table that governs every design decision in this project."""
    non_embed = d["params_non_embed"]

    print()
    print("  Roofline for batch=1 decode @ %.0f GB/s" % bandwidth_gbs)
    print("  (embeddings excluded: tied + only one row gathered per token)")
    print()
    print(f"  {'format':<16} {'bits/wt':>8} {'bytes/token':>14} {'tok/s ceiling':>14}")
    print("  " + "-" * 56)

    # bits/weight includes group scale/zero overhead where applicable.
    formats = [
        ("fp16", 16.0),
        ("int8 (per-ch)", 8.0),
        ("int4 g128", 4.0 + 16.0 / 128 * 2),   # 4b + fp16 scale&zero per 128
        ("int3 g128", 3.0 + 16.0 / 128 * 2),
        ("ternary g128", 1.58 + 16.0 / 128),   # 1.58b + fp16 scale per 128
        ("binary g128", 1.0 + 16.0 / 128),
    ]
    for name, bits in formats:
        byts = non_embed * bits / 8.0
        ceil = bandwidth_gbs * 1e9 / byts
        print(f"  {name:<16} {bits:>8.2f} {human(byts):>14} {ceil:>13.0f}")
    print()
    print("  NOTE: these are hard ceilings at 100% bandwidth utilisation.")
    print("        A good kernel attains 60-80%. Compute (TOPS) does not appear")
    print("        in this table at all -- that is the whole point.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace repo id")
    ap.add_argument("--out", default="../models", help="download root (keep outside the repo)")
    ap.add_argument("--bandwidth", type=float, default=336.0, help="GPU GB/s for the roofline table")
    ap.add_argument("--describe-only", action="store_true", help="skip download, just analyse")
    args = ap.parse_args()

    out_root = Path(args.out).expanduser().resolve()
    local_dir = out_root / args.repo.split("/")[-1]

    if not args.describe_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("error: huggingface_hub not installed", file=sys.stderr)
            return 1

        out_root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.repo} -> {local_dir}")
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(local_dir),
            allow_patterns=ALLOW,
            max_workers=8,
        )

    if not (local_dir / "config.json").exists():
        print(f"error: no config.json under {local_dir}", file=sys.stderr)
        return 1

    d = describe(local_dir)

    on_disk = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())

    print()
    print("=" * 60)
    print(f"  {args.repo}")
    print("=" * 60)
    print(f"  path              {local_dir}")
    print(f"  on disk           {human(on_disk)}")
    print(f"  layers            {d['layers']}")
    print(f"  hidden            {d['hidden']}")
    print(f"  heads             {d['n_q_heads']} Q / {d['n_kv_heads']} KV  (GQA {d['n_q_heads'] // d['n_kv_heads']}:1)")
    print(f"  head_dim          {d['head_dim']}")
    print(f"  intermediate      {d['intermediate']}")
    print(f"  vocab             {d['vocab']}")
    print(f"  tied embeddings   {d['tied_embeddings']}")
    print()
    print(f"  params/layer      {d['params_per_layer'] / 1e6:.2f} M"
          f"   (attn {d['params_attn_per_layer'] / 1e6:.2f} M,"
          f" mlp {d['params_mlp_per_layer'] / 1e6:.2f} M)")
    print(f"  params non-embed  {d['params_non_embed'] / 1e6:.2f} M   <- what we quantize")
    print(f"  params embed      {d['params_embed'] / 1e6:.2f} M")
    print(f"  params total      {d['params_total'] / 1e6:.2f} M")

    mlp_frac = d["params_mlp_per_layer"] / d["params_per_layer"]
    print()
    print(f"  MLP is {mlp_frac * 100:.0f}% of per-layer weights -- optimise it first.")

    roofline(d, args.bandwidth)

    (local_dir / "whetstone_arch.json").write_text(
        json.dumps({k: v for k, v in d.items() if k != "config"}, indent=2))
    print(f"  wrote {local_dir / 'whetstone_arch.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
