#!/usr/bin/env python3
"""The number Whetstone has to beat.

Measures HuggingFace `transformers` on the reference checkpoint and records:

  * batch=1 decode throughput (tok/s), median + p10/p90 over per-token latencies
  * prefill throughput (tok/s) at several context lengths
  * achieved memory bandwidth, as a fraction of the hardware roof
  * wikitext-2 perplexity
  * reference logits on a fixed prompt set, for differential testing later

Everything is written to a JSON report so later runs can be diffed against it.

    python bench/baseline_hf.py --model ../models/Qwen2.5-0.5B-Instruct \
                               --out ../research/experiments/baseline_fp16.json

Timing discipline (see CLAUDE.md §7): warmup first, `torch.cuda.synchronize()`
around every timed region, and never trust a single sample.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------- utils

def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Wall-clock around a GPU region, with synchronisation on both edges."""

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.dt = time.perf_counter() - self.t0
        return False


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[i]


# --------------------------------------------------------------------------- model

def load(model_dir: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, device_map=None, attn_implementation="sdpa",
    ).to("cuda").eval()
    return tok, model


def weight_bytes(model) -> tuple[int, int]:
    """(all params, bytes actually read per decode step).

    The second value is the roofline denominator, and it is NOT
    "everything except the embeddings". Two different things use the embedding
    matrix, and they cost very differently:

      - the INPUT embedding is a gather of one row: negligible;
      - the OUTPUT projection (lm_head) is a full GEMV over the whole
        [vocab, hidden] matrix, read in its entirety, every token.

    With tied weights those are the same tensor, which makes it easy to write
    the whole thing off as a lookup. For Qwen2.5-0.5B the head is 27.6% of
    per-token traffic; excluding it overstates the ceiling by 1.38x.
    """
    total = 0
    input_embed = 0
    for name, p in model.named_parameters():
        n = p.numel() * p.element_size()
        total += n
        if "embed_tokens" in name:
            input_embed = n

    # Tied:   total = blocks + one shared matrix = blocks + head. Read it all.
    # Untied: total = blocks + embed + head. Drop only the input-side gather.
    tied = getattr(model.config, "tie_word_embeddings", False)
    resident = total if tied else total - input_embed
    return total, resident


# --------------------------------------------------------------------------- decode

@torch.inference_mode()
def bench_decode(model, tok, n_tokens: int, prompt: str, warmup: int) -> dict:
    """Per-token latency for greedy autoregressive decode at batch=1.

    Deliberately hand-rolled rather than using .generate() so nothing is hidden
    behind the generation loop, and so each token is timed individually.
    """
    from transformers import DynamicCache

    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

    def one_run(n: int) -> list[float]:
        cache = DynamicCache()
        out = model(ids, use_cache=True, past_key_values=cache)
        nxt = out.logits[:, -1:].argmax(-1)
        lat = []
        for _ in range(n):
            with Timer() as t:
                out = model(nxt, use_cache=True, past_key_values=cache)
                nxt = out.logits[:, -1:].argmax(-1)
            lat.append(t.dt)
        return lat

    one_run(warmup)                      # warm caches, autotune, clocks
    lat = one_run(n_tokens)

    ms = [x * 1e3 for x in lat]
    return {
        "n_tokens": n_tokens,
        "prompt_tokens": int(ids.numel()),
        "tok_per_s_median": 1e3 / statistics.median(ms),
        "tok_per_s_mean": n_tokens / sum(lat),
        "ms_median": statistics.median(ms),
        "ms_p10": pct(ms, 10),
        "ms_p90": pct(ms, 90),
        "ms_min": min(ms),
    }


@torch.inference_mode()
def bench_prefill(model, tok, lengths: list[int], warmup: int) -> list[dict]:
    """Prompt-processing throughput -- the compute-bound regime."""
    res = []
    for L in lengths:
        ids = torch.randint(0, model.config.vocab_size, (1, L), device="cuda")
        for _ in range(warmup):
            model(ids, use_cache=False)
        runs = []
        for _ in range(5):
            with Timer() as t:
                model(ids, use_cache=False)
            runs.append(t.dt)
        dt = statistics.median(runs)
        res.append({"ctx": L, "seconds": dt, "tok_per_s": L / dt})
    return res


# --------------------------------------------------------------------------- quality

@torch.inference_mode()
def perplexity(model, tok, window: int, max_windows: int,
               token_file: str | None = None) -> dict:
    """Perplexity over non-overlapping windows.

    `token_file` points at a flat little-endian `uint32` stream produced by
    `bench/prepare_tokens.py`. **Prefer it.** Passing a corpus *name* to two
    harnesses does not guarantee they see the same tokens — tokenizer versions
    differ, and so does how documents get joined — and a perplexity comparison
    between implementations that read different tokens is not a comparison. When
    no file is given this falls back to loading wikitext-2 itself, which is
    convenient and only comparable to itself.
    """
    if token_file:
        import numpy as np

        ids_np = np.fromfile(token_file, dtype="<u4")
        ids = torch.from_numpy(ids_np.astype("int64")).unsqueeze(0).to("cuda")
        return _ppl_over(model, ids, window, max_windows, f"file:{token_file}")

    # `datasets` >= 5 requires a fully qualified namespace/name, so the bare
    # "wikitext" id that most published harnesses use no longer resolves.
    # Try the canonical id first and fall back for older installs.
    text = None
    errors = []
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(ds["text"])
            break
        except Exception as e:
            errors.append(f"{repo}: {e}")
    if text is None:
        return {"error": "could not load wikitext2 -- " + " | ".join(errors)}

    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    return _ppl_over(model, ids, window, max_windows, "wikitext-2-raw-v1/test")


@torch.inference_mode()
def _ppl_over(model, ids, window: int, max_windows: int, source: str) -> dict:
    """The window loop itself, shared by both token sources."""
    n = min(max_windows, ids.shape[1] // window)

    # Cross-entropy over a full window would materialise
    # window x 151936 floats in fp32 -- 1.2 GB at window=2048, which OOMs a 6 GB
    # card that is already holding the model. Accumulate over slices instead;
    # the result is identical, the peak allocation is not.
    CE_CHUNK = 256

    nll, count = 0.0, 0
    for i in range(n):
        chunk = ids[:, i * window:(i + 1) * window]
        logits = model(chunk, use_cache=False).logits[0]     # (window, vocab)

        pred = logits[:-1]
        target = chunk[0, 1:]

        for j in range(0, pred.shape[0], CE_CHUNK):
            part = pred[j:j + CE_CHUNK].float()
            tgt = target[j:j + CE_CHUNK]
            loss = torch.nn.functional.cross_entropy(part, tgt, reduction="sum")
            nll += loss.item()
            count += tgt.numel()
            del part, loss

        del logits, pred
        torch.cuda.empty_cache()

    return {
        "dataset": source,
        "window": window,
        "windows": n,
        "tokens": count,
        "nll": nll / count,
        "ppl": float(torch.exp(torch.tensor(nll / count))),
    }


@torch.inference_mode()
def reference_logits(model, tok, prompts: list[str], topk: int) -> list[dict]:
    """Golden outputs. Every future Whetstone kernel is diffed against these."""
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to("cuda")
        logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        v, i = probs.topk(topk)
        out.append({
            "prompt": p,
            "n_tokens": int(ids.numel()),
            "top1_id": int(i[0]),
            "top1_token": tok.decode([int(i[0])]),
            "topk_ids": [int(x) for x in i],
            "topk_probs": [float(x) for x in v],
            # cheap fingerprints that catch numeric drift without storing 152k floats
            "logits_mean": float(logits.mean()),
            "logits_std": float(logits.std()),
            "logits_max": float(logits.max()),
            "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum()),
        })
    return out


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a village at the edge of the forest,",
    "The three laws of thermodynamics state that",
    "import numpy as np\n\ndef softmax(x):",
    "Translate to French: 'The weather is beautiful today.'\n",
    "In 1969, humanity first",
]


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--ppl-window", type=int, default=2048)
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--tokens", default=None,
                    help="flat u32 token stream from bench/prepare_tokens.py. Use this "
                         "when comparing against another implementation: it is the only "
                         "way to guarantee both read the same tokens.")
    ap.add_argument("--gen", type=int, default=None,
                    help="alias for --decode-tokens, so comparison harnesses can use "
                         "one flag name across engines")
    ap.add_argument("--json", action="store_true",
                    help="print a compact JSON summary on stdout, for scripting")
    ap.add_argument("--bandwidth", type=float, default=336.0, help="hardware GB/s")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: no CUDA device")
        return 1

    dtype = getattr(torch, args.dtype)
    dev = torch.cuda.get_device_properties(0)

    print("=" * 66)
    print("  Whetstone baseline -- HuggingFace transformers")
    print("=" * 66)
    print(f"  gpu          {dev.name}  sm_{dev.major}{dev.minor}  "
          f"{dev.total_memory / 1e9:.1f} GB  {dev.multi_processor_count} SMs")
    print(f"  torch        {torch.__version__}   cuda {torch.version.cuda}")
    print(f"  dtype        {args.dtype}")
    print(f"  model        {args.model}")

    tok, model = load(args.model, dtype)
    total_b, nonembed_b = weight_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  params       {n_params / 1e6:.1f} M")
    print(f"  weights      {total_b / 1e6:.0f} MB total / "
          f"{nonembed_b / 1e6:.0f} MB read per token (blocks + lm_head)")
    print(f"  vram in use  {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    print()

    report: dict = {
        "meta": {
            "gpu": dev.name,
            "sm": f"{dev.major}{dev.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": args.dtype,
            "model": args.model,
            "host": platform.node(),
            "params": n_params,
            "weight_bytes_total": total_b,
            "weight_bytes_decode_resident": nonembed_b,
            "hw_bandwidth_gbs": args.bandwidth,
        }
    }

    # ---- decode -----------------------------------------------------------
    print("  [decode] batch=1 greedy, per-token timing ...")
    dec = bench_decode(model, tok, args.gen or args.decode_tokens,
                       "The history of computing began", args.warmup)
    report["decode"] = dec

    # Achieved bandwidth: one full pass over blocks + lm_head per token.
    achieved = nonembed_b * dec["tok_per_s_median"] / 1e9
    roof = args.bandwidth * 1e9 / nonembed_b
    dec["achieved_bandwidth_gbs"] = achieved
    dec["bandwidth_utilisation"] = achieved / args.bandwidth
    dec["roofline_tok_per_s"] = roof
    dec["roofline_attainment"] = dec["tok_per_s_median"] / roof

    print(f"           {dec['tok_per_s_median']:.1f} tok/s median "
          f"({dec['ms_median']:.2f} ms/tok, p10 {dec['ms_p10']:.2f}, p90 {dec['ms_p90']:.2f})")
    print(f"           achieved {achieved:.0f} GB/s of {args.bandwidth:.0f} GB/s "
          f"= {dec['bandwidth_utilisation'] * 100:.0f}% bandwidth utilisation")
    print(f"           roofline for this dtype is {roof:.0f} tok/s "
          f"-> attaining {dec['roofline_attainment'] * 100:.0f}%")
    print()

    # ---- prefill ----------------------------------------------------------
    print("  [prefill] compute-bound regime ...")
    pre = bench_prefill(model, tok, [128, 512, 2048], warmup=2)
    report["prefill"] = pre
    for r in pre:
        print(f"           ctx {r['ctx']:>5}   {r['tok_per_s']:>9.0f} tok/s")
    print()

    # ---- quality ----------------------------------------------------------
    print("  [reference] golden logits for differential testing ...")
    report["reference_logits"] = reference_logits(model, tok, PROMPTS, topk=10)
    for r in report["reference_logits"][:3]:
        print(f"           {r['prompt'][:38]!r:<42} -> {r['top1_token']!r} "
              f"p={r['topk_probs'][0]:.3f}")
    print()

    if not args.skip_ppl and args.ppl_windows > 0:
        src = Path(args.tokens).name if args.tokens else "wikitext-2"
        print(f"  [perplexity] {src}, {args.ppl_windows} x {args.ppl_window} tokens ...")
        ppl = perplexity(model, tok, args.ppl_window, args.ppl_windows, args.tokens)
        report["perplexity"] = ppl
        if "error" in ppl:
            print(f"           SKIPPED: {ppl['error']}")
        else:
            print(f"           ppl = {ppl['ppl']:.4f}   (nll {ppl['nll']:.4f}, "
                  f"{ppl['tokens']} tokens)")
        print()

    report["meta"]["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
        print(f"  report -> {p}")

    print()
    print("  " + "-" * 62)
    print(f"  TARGET TO BEAT: {dec['tok_per_s_median']:.1f} tok/s "
          f"at ppl {report.get('perplexity', {}).get('ppl', float('nan')):.4f}")
    print("  " + "-" * 62)

    if args.json:
        print(json.dumps({
            "decode": dec["tok_per_s_median"],
            "decode_p10_ms": dec["ms_p10"],
            "decode_p90_ms": dec["ms_p90"],
            "mb_per_token": nonembed_b / 1e6,
            "ppl": report.get("perplexity", {}).get("ppl"),
            "nll": report.get("perplexity", {}).get("nll"),
            "positions": report.get("perplexity", {}).get("tokens"),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
