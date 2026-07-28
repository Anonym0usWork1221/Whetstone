#!/usr/bin/env python3
"""Three-way comparison: your `.wstone`, the original weights, and llama.cpp.

    python bench/compare.py --model ../models/Qwen2.5-0.5B-Instruct \
                            --wstone ../models/qwen05b-int4-head.wstone \
                            --gguf   /tmp/qwen05b-q4km.gguf \
                            --tokens ../research/experiments/wikitext2.u32

Every row is measured here, on this machine, in one run — which is the only way
the numbers are comparable. Quoting a tok/s figure from one session against a
perplexity figure from another is how a regression ships.

# What "comparable" requires, and where it breaks down

**Speed** is comparable across all three because the engines are **interleaved**:
one sample of each, round-robin, repeated. Measuring all of engine A and then all
of engine B does not compare them, it compares A cold to B hot. That is not
hypothetical — an earlier version of this script ran llama.cpp last, after ten
minutes of continuous GPU load, and read it at 250.8 tok/s against the 284.6 it
sustains when interleaved. The resulting "1.69x" was 10% thermal drift. Round
robin, and the ratio holds at 1.51-1.55x across rounds while the GPU climbs from
71 to 75 degrees.

Everyone generates the same number of tokens (`--gen`, default 384, matching
`llama-bench`'s `tg384`) after a discarded warm-up, and the median of several
samples is reported — a desktop GPU shares the card with the compositor and
single samples are bimodal.

**Quality** is exactly comparable between Whetstone and HuggingFace: both read
the *same materialised token stream* (`bench/prepare_tokens.py`), the same window
length and the same window count, so the only thing that differs is the
arithmetic.

**llama.cpp's perplexity is on a different scale and must not be compared
across harnesses.** `llama-perplexity` tokenizes the corpus itself and applies
its own chunking, and the size of the resulting offset is not small: measured
here, *the same fp16 weights* score 12.2484 under llama.cpp and 13.8182 under
this harness. Putting 12.57 next to 18.03 and concluding anything would be
reading a 1.57-point harness offset as a quality difference.

So the script measures **fp16 in every harness** and reports each format's
degradation **against its own harness's fp16**. That difference is comparable;
the absolute numbers are not. Which is why `--model` and an fp16 `.gguf` are
worth the extra minutes — without a same-harness fp16 row there is no anchor and
the quantizer comparison cannot be made at all.

# Reading the result

The question a quantized format has to answer is not "is it faster" — dropping
bits is always faster. It is **what did that speed cost**, and the two have to be
read together. A format that is 1.7x faster and 4 perplexity worse has not beaten
one that is slower and 0.3 worse; it has made a different trade, and probably a
bad one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Prompt for the speed runs. Short on purpose: prefill is not what is being
# measured, and a long prompt would dilute the decode figure.
DEFAULT_IDS = "785,6722,315,9625,374"  # "The capital of France is"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# ------------------------------------------------------------------ whetstone

def whetstone_sample(binary: Path, wstone: Path, ids: str, gen: int, ctx: int) -> dict:
    """One generation. Interleaving is the caller's job — see the docstring."""
    r = run([str(binary), "run", str(wstone), "--ids", ids,
             "--max-new", str(gen), "--ctx", str(ctx), "--graph"])
    if r.returncode != 0:
        die(f"whetstone run failed:\n{r.stderr or r.stdout}")
    m = re.search(r"decode\s+([\d.]+) tok/s", r.stdout)
    b = re.search(r"weights\s+([\d.]+) MB/token\s+\(([\d.]+) bits", r.stdout)
    if not m:
        die(f"could not parse whetstone output:\n{r.stdout}")
    return {
        "decode": float(m.group(1)),
        "mb_per_token": float(b.group(1)) if b else None,
        "bits_per_weight": float(b.group(2)) if b else None,
    }


def whetstone_ppl(binary: Path, wstone: Path, tokens: Path, window: int,
                  windows: int, tmp: Path) -> dict:
    out = tmp / f"ppl_{wstone.stem}.json"
    r = run([str(binary), "ppl", str(wstone), "--tokens", str(tokens),
             "--window", str(window), "--windows", str(windows), "--out", str(out)])
    if r.returncode != 0:
        die(f"whetstone ppl failed:\n{r.stderr or r.stdout}")
    return json.loads(out.read_text())


# ---------------------------------------------------------------- huggingface

def hf_measure(python: Path, script: Path, model: Path, tokens: Path, gen: int,
               window: int, windows: int) -> dict:
    """Speed and perplexity for the original weights, on the same token stream."""
    r = run([str(python), str(script), "--model", str(model),
             "--tokens", str(tokens), "--gen", str(gen),
             "--ppl-window", str(window), "--ppl-windows", str(windows),
             "--json"])
    if r.returncode != 0:
        die(f"HuggingFace baseline failed:\n{r.stderr or r.stdout}")
    try:
        return json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    except ValueError:
        die(f"could not parse baseline JSON:\n{r.stdout}")


# ------------------------------------------------------------------ llama.cpp

def llama_sample(bench: Path, gguf: Path, gen: int) -> dict:
    """One llama-bench measurement. `-r 1`: repetition is the caller's, so that
    it can interleave engines rather than batching each one."""
    r = run([str(bench), "-m", str(gguf), "-p", "0", "-n", str(gen),
             "-ngl", "99", "-r", "1", "-o", "json"])
    if r.returncode != 0:
        die(f"llama-bench failed:\n{r.stderr or r.stdout}")
    try:
        rows = json.loads(r.stdout[r.stdout.index("["):r.stdout.rindex("]") + 1])
    except ValueError:
        die(f"could not parse llama-bench JSON:\n{r.stdout}")
    row = rows[-1]
    return {
        "decode": float(row["avg_ts"]),
        "size_mb": float(row.get("model_size", 0)) / 1e6,
    }


def llama_ppl(binary: Path, gguf: Path, text: Path, window: int,
              chunks: int) -> dict | None:
    """llama.cpp's own perplexity. Not exactly comparable — see the docstring."""
    if binary is None or not binary.exists():
        return None
    r = run([str(binary), "-m", str(gguf), "-f", str(text), "-c", str(window),
             "--chunks", str(chunks), "-ngl", "99"])
    blob = r.stdout + r.stderr
    m = re.findall(r"Final estimate: PPL = ([\d.]+)", blob)
    if not m:
        m = re.findall(r"\[\d+\]([\d.]+),?\s*$", blob.strip())
    if not m:
        return None
    return {"ppl": float(m[-1])}


# ---------------------------------------------------------------------- table

def table(rows: list[dict], params: float | None = None) -> str:
    """Each row's perplexity delta is against fp16 *in the same harness*.

    Comparing absolute perplexity across harnesses is the mistake this layout
    exists to prevent — see the module docstring for the 1.57-point offset that
    the same weights show between them.
    """
    w = [22, 12, 9, 13, 11, 15, 9]
    head = ["engine / format", "bytes/token", "bits/wt", "decode tok/s", "ppl",
            "Δ vs own fp16", "speedup"]
    out = ["  " + "".join(h.ljust(x) for h, x in zip(head, w)),
           "  " + "-" * sum(w)]
    base_speed = next((r["decode"] for r in rows if r.get("is_base")), None)

    for r in rows:
        ppl = r.get("ppl")
        anchor = r.get("anchor_ppl")
        if ppl is None:
            d = "—"
        elif anchor is None:
            d = "no fp16 anchor"
        elif abs(ppl - anchor) < 1e-9:
            d = "(the anchor)"
        else:
            d = f"{ppl - anchor:+.4f}"
        # bits/weight from the bytes actually streamed. It is the number that
        # makes a format comparison fair: Q4_K_M keeps the embedding and output
        # at higher precision, so its *average* width is well above the "4" in
        # its name -- 6.35 bits here, against int4-g128's 4.25.
        bits = r.get("bits")
        if bits is None and r.get("mb") and params:
            bits = r["mb"] * 1e6 * 8 / params
        cells = [
            r["name"],
            f"{r['mb']:.0f} MB" if r.get("mb") else "—",
            f"{bits:.2f}" if bits else "—",
            f"{r['decode']:.1f}",
            f"{ppl:.4f}" if ppl is not None else "—",
            d,
            f"{r['decode'] / base_speed:.2f}x" if base_speed else "—",
        ]
        out.append("  " + "".join(c.ljust(x) for c, x in zip(cells, w)))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wstone", action="append", default=[], required=True,
                    help="a .wstone to measure; repeatable")
    ap.add_argument("--model", help="HuggingFace checkpoint dir, for the fp16 baseline")
    ap.add_argument("--gguf", action="append", default=[],
                    help="a llama.cpp .gguf to measure; repeatable")
    ap.add_argument("--tokens", required=True,
                    help="token stream from bench/prepare_tokens.py")
    ap.add_argument("--text", help="the same corpus as UTF-8, for llama.cpp's perplexity")
    ap.add_argument("--binary", default=None, help="path to the whetstone binary")
    ap.add_argument("--llama-bench", default="../llama.cpp/build/bin/llama-bench")
    ap.add_argument("--llama-ppl", default="../llama.cpp/build/bin/llama-perplexity")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--gen", type=int, default=384, help="tokens generated per speed run")
    ap.add_argument("--reps", type=int, default=3, help="speed samples, excluding warm-up")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--params", type=float, default=494.03e6,
                    help="parameter count, for the bits/weight column. Defaults to "
                         "Qwen2.5-0.5B's 494.03 M.")
    ap.add_argument("--skip-ppl", action="store_true",
                    help="speed only; perplexity is the slow half")
    ap.add_argument("--out", help="write the whole comparison as JSON")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    binary = Path(args.binary) if args.binary else here.parent / "target/release/whetstone"
    if not binary.exists():
        die(f"whetstone binary not found at {binary}; build with `cargo build --release`")

    tokens = Path(args.tokens)
    if not tokens.exists():
        die(f"{tokens} not found — create it with bench/prepare_tokens.py")

    tmp = Path(".compare-tmp")
    tmp.mkdir(exist_ok=True)

    print("=" * 78)
    print("  Whetstone vs the original weights vs llama.cpp")
    print("=" * 78)
    print(f"  generating {args.gen} tokens per run, median of {args.reps} after a warm-up")
    if not args.skip_ppl:
        print(f"  perplexity over {args.windows} x {args.window} tokens from {tokens.name}")
    print("-" * 78)

    rows: list[dict] = []
    baseline_ppl: float | None = None
    t0 = time.time()

    # --- the original weights, which is what everything else is judged against
    if args.model:
        hf_script = here / "baseline_hf.py"
        if not hf_script.exists():
            die(f"{hf_script} not found")
        print("  [1/3] original weights (HuggingFace fp16) ...", flush=True)
        hf = hf_measure(Path(args.python), hf_script, Path(args.model), tokens,
                        args.gen, args.window, 0 if args.skip_ppl else args.windows)
        baseline_ppl = hf.get("ppl")
        rows.append({
            "name": "HuggingFace fp16", "mb": hf.get("mb_per_token"),
            "decode": hf["decode"], "ppl": baseline_ppl, "is_base": True,
        })

    # --- speed: every engine interleaved, so thermal drift hits them equally
    targets: list[dict] = []
    for w in args.wstone:
        wp = Path(w)
        if not wp.exists():
            die(f"{wp} not found")
        targets.append({"kind": "wstone", "path": wp,
                        "name": f"whetstone {wp.stem.split('-')[-1]}"})
    lb = Path(args.llama_bench)
    for g in args.gguf:
        gp = Path(g)
        if not gp.exists():
            die(f"{gp} not found")
        if not lb.exists():
            die(f"llama-bench not found at {lb}")
        targets.append({"kind": "gguf", "path": gp,
                        "name": f"llama.cpp {gp.stem.split('-')[-1]}"})

    print(f"  [2/3] speed, {len(targets)} targets x {args.reps} interleaved rounds "
          f"(+1 warm-up) ...", flush=True)
    for t in targets:
        t["samples"] = []

    for rnd in range(args.reps + 1):
        for t in targets:
            if t["kind"] == "wstone":
                sp = whetstone_sample(binary, t["path"], DEFAULT_IDS, args.gen, args.ctx)
                t.setdefault("mb", sp["mb_per_token"])
                t.setdefault("bits", sp["bits_per_weight"])
            else:
                sp = llama_sample(lb, t["path"], args.gen)
                t.setdefault("mb", sp["size_mb"])
            if rnd == 0:
                continue  # warm-up round: clocks still ramping
            t["samples"].append(sp["decode"])
        if rnd:
            print("        round %d: %s" % (
                rnd, "  ".join(f"{t['name'].split()[-1]} {t['samples'][-1]:.0f}"
                               for t in targets)), flush=True)

    for t in targets:
        row = {"name": t["name"], "mb": t.get("mb"), "bits": t.get("bits"),
               "decode": statistics.median(t["samples"]),
               "decode_range": (min(t["samples"]), max(t["samples"]))}
        rows.append(row)

    # --- quality, which is not timing-sensitive and so needs no interleaving
    if not args.skip_ppl:
        print("  [3/3] perplexity ...", flush=True)
        for t, row in zip(targets, [r for r in rows if not r.get("is_base")]):
            if t["kind"] == "wstone":
                row["ppl"] = whetstone_ppl(binary, t["path"], tokens, args.window,
                                           args.windows, tmp)["ppl"]
            elif args.text:
                q = llama_ppl(Path(args.llama_ppl), t["path"], Path(args.text),
                              args.window, args.windows)
                if q:
                    row["ppl"] = q["ppl"]
            print(f"        {t['name']}: "
                  f"{row.get('ppl', float('nan')):.4f}", flush=True)

    # Anchor every row to fp16 measured in its own harness. Absolute perplexity
    # is not comparable across harnesses; degradation from a same-harness fp16
    # baseline is.
    ws = [r for r in rows if r["name"].startswith("whetstone")]
    lc = [r for r in rows if r["name"].startswith("llama.cpp")]
    ws_anchor = baseline_ppl  # HuggingFace fp16 reads the identical token stream
    lc_anchor = next((r["ppl"] for r in lc if "f16" in r["name"] and r.get("ppl")), None)
    for r in rows:
        r["anchor_ppl"] = lc_anchor if r["name"].startswith("llama.cpp") else ws_anchor

    print("-" * 78)
    print()
    print(table(rows, params=args.params))
    print()

    if lc_anchor and baseline_ppl and abs(lc_anchor - baseline_ppl) > 0.05:
        print(f"  The two harnesses disagree by {abs(lc_anchor - baseline_ppl):.2f} "
              f"perplexity on identical fp16 weights")
        print(f"  ({baseline_ppl:.4f} here, {lc_anchor:.4f} under llama-perplexity), so "
              f"only the")
        print("  right-hand column is comparable between them. Do not read across.")
        print()

    if ws and lc:
        best_w = max(ws, key=lambda r: r["decode"])
        best_l = max((r for r in lc if "f16" not in r["name"]), key=lambda r: r["decode"],
                     default=None)
        if best_l:
            print(f"  SPEED    {best_w['name']} is "
                  f"{best_w['decode'] / best_l['decode']:.2f}x {best_l['name']}.")
            if best_w.get("mb") and best_l.get("mb"):
                print(f"           It reads {best_l['mb'] / best_w['mb']:.2f}x fewer "
                      f"bytes per token, which is where that comes from.")

            wd = (best_w["ppl"] - ws_anchor) if best_w.get("ppl") and ws_anchor else None
            ld = (best_l["ppl"] - lc_anchor) if best_l.get("ppl") and lc_anchor else None
            if wd is not None and ld is not None:
                print()
                print(f"  QUALITY  {best_w['name']} costs {wd:+.2f} perplexity against its "
                      f"own fp16.")
                print(f"           {best_l['name']} costs {ld:+.2f}.")
                if wd > ld:
                    print(f"           **The quantizer is {wd / max(ld, 1e-9):.0f}x worse.** "
                          f"The engine is not the problem;")
                    print("           round-to-nearest is. Speed bought at this price is "
                          "not a win.")
                    # Check the comparison is not being flattered by bit width.
                    wb = best_w.get("bits") or (best_w["mb"] * 1e6 * 8 / args.params)
                    lb_ = best_l["mb"] * 1e6 * 8 / args.params
                    if wb < lb_:
                        # The closest *quantized* Whetstone row at or above the
                        # competitor's width. An fp16 row trivially satisfies
                        # "more bits" and would make the comparison meaningless.
                        fair = sorted(
                            (r for r in ws if r.get("ppl") and r.get("bits")
                             and lb_ <= r["bits"] < 16.0),
                            key=lambda r: r["bits"])
                        print()
                        print(f"           In fairness, {best_w['name']} is "
                              f"{wb:.2f} bits/weight against {lb_:.2f} -- k-quants keep")
                        print("           the embedding and output wide, so Q4_K_M is not "
                              "a 4-bit format.")
                        if fair:
                            f = min(fair, key=lambda r: r["ppl"])
                            print(f"           But at {f['bits']:.2f} bits Whetstone still "
                                  f"costs {f['ppl'] - ws_anchor:+.2f} -- worse at *more*")
                            print("           bits, so the gap is the rounding, not the "
                                  "budget.")
                else:
                    print("           Whetstone loses no more quality, so the speed is a "
                          "clean win.")
    if lc and not any(r.get("ppl") for r in lc):
        print()
        print("  NOTE: no perplexity for llama.cpp — pass --text with the corpus and")
        print("  build llama-perplexity. Until then the speed comparison is not")
        print("  like-for-like on quality, and should not be presented as if it were.")

    print()
    print(f"  {time.time() - t0:.0f} s total")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"generated_tokens": args.gen, "window": args.window,
             "windows": args.windows, "tokens_file": str(tokens), "rows": rows},
            indent=2, default=str))
        print(f"  wrote {args.out}")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
