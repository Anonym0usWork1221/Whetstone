#!/usr/bin/env python3
"""Live terminal chat with streaming tokens and a running speed readout.

Two modes:

    # interactive chat, tokens stream as they are produced
    python bench/chat.py --model ../models/Qwen2.5-0.5B-Instruct

    # non-interactive throughput run over a fixed prompt set
    python bench/chat.py --model ../models/Qwen2.5-0.5B-Instruct --bench

The `--engine` flag selects the backend. `hf` runs HuggingFace `transformers`
and is the baseline Whetstone has to beat. `whetstone` will run the native
engine once its forward pass lands; until then it fails with a clear message
rather than silently falling back, because a benchmark that quietly measures
something other than what it claims is worse than no benchmark.

Every number printed is measured with `torch.cuda.synchronize()` on both edges
of the timed region, and the roofline it is compared against counts `lm_head` —
which tied embeddings make easy to forget, and which is 27.6% of the bytes read
per token on Qwen2.5-0.5B.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------- colours

class C:
    """ANSI codes, disabled when stdout is not a terminal."""
    on = sys.stdout.isatty()

    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"

    @classmethod
    def p(cls, code: str, text: str) -> str:
        return f"{code}{text}{cls.RESET}" if cls.on else text


def hr(ch: str = "─", n: int = 68) -> str:
    return C.p(C.GREY, ch * n)


# ------------------------------------------------------------------ roofline

def model_facts(model_dir: Path) -> dict:
    """Parameter counts including lm_head, which sets the real ceiling."""
    cfg = json.loads((model_dir / "config.json").read_text())
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    nq = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nq)
    hd = cfg.get("head_dim", h // nq)
    inter = cfg["intermediate_size"]
    V = cfg["vocab_size"]

    attn = h * nq * hd + 2 * h * nkv * hd + nq * hd * h
    mlp = 3 * h * inter
    body = L * (attn + mlp)
    head = V * h

    return {
        "config": cfg,
        "body_params": body,
        "head_params": head,
        # The input embedding is a one-row gather; the OUTPUT projection is a
        # full GEMV over the whole matrix, every token. Both use the same tensor
        # under tied weights, but only the second costs bandwidth.
        "resident_params": body + head,
        "head_fraction": head / (body + head),
    }


# -------------------------------------------------------------------- engines

class HFEngine:
    """HuggingFace transformers. The baseline."""

    name = "huggingface"

    def __init__(self, model_dir: Path, dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=getattr(torch, dtype), attn_implementation="sdpa",
        ).to("cuda").eval()

        self.bytes_per_weight = {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]
        self.dtype = dtype

    def encode_chat(self, messages: list[dict]) -> list[int]:
        """Applies the model's chat template and returns plain token ids.

        Normalising the return type matters: across transformers versions this
        call has returned a list, a BatchEncoding, and a tokenizers.Encoding,
        and only the first can be handed straight to torch.tensor().
        """
        try:
            out = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
        except Exception:
            out = self.tok(messages[-1]["content"]).input_ids

        return self._to_ids(out)

    @staticmethod
    def _to_ids(out) -> list[int]:
        for attr in ("ids", "input_ids"):
            if hasattr(out, attr):
                out = getattr(out, attr)
                break
        if isinstance(out, dict):
            out = out["input_ids"]
        # Unwrap a leading batch dimension if there is one.
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return [int(t) for t in out]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def eos_ids(self) -> set[int]:
        out = set()
        for attr in ("eos_token_id", "pad_token_id"):
            v = getattr(self.tok, attr, None)
            if isinstance(v, int):
                out.add(v)
        gen = getattr(self.model, "generation_config", None)
        if gen is not None and getattr(gen, "eos_token_id", None) is not None:
            e = gen.eos_token_id
            out.update(e if isinstance(e, list) else [e])
        return out

    def stream(self, ids: list[int], max_new: int, temperature: float, top_p: float):
        """Yields (token_id, seconds_for_this_token). First yield is prefill."""
        torch = self.torch
        from transformers import DynamicCache

        stops = self.eos_ids()
        x = torch.tensor([ids], device="cuda")
        cache = DynamicCache()

        with torch.inference_mode():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.model(x, use_cache=True, past_key_values=cache)
            torch.cuda.synchronize()
            prefill = time.perf_counter() - t0

            nxt = self._sample(out.logits[:, -1], temperature, top_p)
            yield int(nxt.item()), prefill, True

            for _ in range(max_new - 1):
                if int(nxt.item()) in stops:
                    return
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = self.model(nxt.view(1, 1), use_cache=True, past_key_values=cache)
                nxt = self._sample(out.logits[:, -1], temperature, top_p)
                torch.cuda.synchronize()
                yield int(nxt.item()), time.perf_counter() - t, False

    def _sample(self, logits, temperature: float, top_p: float):
        torch = self.torch
        if temperature <= 0:
            return logits.argmax(-1)

        probs = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            srt, idx = torch.sort(probs, descending=True, dim=-1)
            keep = (srt.cumsum(-1) - srt) < top_p
            srt = srt * keep
            srt = srt / srt.sum(-1, keepdim=True)
            pick = torch.multinomial(srt, 1)
            return idx.gather(-1, pick).squeeze(-1)
        return torch.multinomial(probs, 1).squeeze(-1)


class WhetstoneEngine:
    """The native engine. Not yet able to run a forward pass."""

    name = "whetstone"

    def __init__(self, wstone: Path, **_):
        raise SystemExit(
            C.p(C.YELLOW, "\n  whetstone engine: not available yet.\n\n")
            + "  The .wstone format, quantizer and int4 GEMV kernel are done and\n"
              "  tested, but the full forward pass (RMSNorm, RoPE, attention with\n"
              "  KV cache, SwiGLU, sampling) is not wired up, so there is nothing\n"
              "  honest to benchmark yet.\n\n"
            + "  What you can run today:\n"
              "    whetstone convert <model_dir> -o model.wstone   # build the weights\n"
              "    whetstone verify  model.wstone --source <dir>   # check them\n"
              "    whetstone probe                                 # what the GPU can do\n"
              "    python bench/chat.py --engine hf ...            # the baseline to beat\n"
        )


# ---------------------------------------------------------------- statistics

class Stats:
    """Per-token latencies and everything derived from them."""

    def __init__(self, facts: dict, bytes_per_weight: float, bandwidth: float):
        self.lat: list[float] = []
        self.prefill = 0.0
        self.prompt_tokens = 0
        self.facts = facts
        self.bytes_per_token = facts["resident_params"] * bytes_per_weight
        self.bandwidth = bandwidth

    def add(self, dt: float) -> None:
        self.lat.append(dt)

    @property
    def tok_s(self) -> float:
        return 1.0 / statistics.median(self.lat) if self.lat else 0.0

    def summary(self) -> dict:
        if not self.lat:
            return {}
        ms = sorted(x * 1e3 for x in self.lat)
        med = statistics.median(ms)
        gbs = self.bytes_per_token * (1e3 / med) / 1e9
        ceiling = self.bandwidth * 1e9 / self.bytes_per_token
        return {
            "generated": len(self.lat),
            "prompt_tokens": self.prompt_tokens,
            "ttft_ms": self.prefill * 1e3,
            "tok_s": 1e3 / med,
            "ms_median": med,
            "ms_p10": ms[int(0.10 * (len(ms) - 1))],
            "ms_p90": ms[int(0.90 * (len(ms) - 1))],
            "achieved_gbs": gbs,
            "utilisation": gbs / self.bandwidth,
            "ceiling_tok_s": ceiling,
            "attainment": (1e3 / med) / ceiling,
        }


def print_summary(s: dict, engine: str, facts: dict) -> None:
    if not s:
        return

    bar_w = 40
    filled = max(0, min(bar_w, int(round(s["attainment"] * bar_w))))
    bar = C.p(C.GREEN, "\u2588" * filled) + C.p(C.GREY, "\u2591" * (bar_w - filled))

    speed = f"{s['tok_s']:.1f} tok/s"
    timing = f"({s['ms_median']:.1f} ms/token, p10 {s['ms_p10']:.1f}, p90 {s['ms_p90']:.1f})"
    possible = f"({s['tok_s']:.0f} of {s['ceiling_tok_s']:.0f} tok/s possible)"
    read = (f"engine {engine} \u00b7 {facts['resident_params'] / 1e6:.0f} M weights read "
            f"per token (lm_head is {facts['head_fraction'] * 100:.0f}% of them)")

    print()
    print(hr())
    print(f"  {C.p(C.BOLD, 'speed'):<18} {C.p(C.CYAN, speed)}  {C.p(C.GREY, timing)}")
    print(f"  {C.p(C.BOLD, 'first token'):<18} {s['ttft_ms']:.0f} ms")
    print(f"  {C.p(C.BOLD, 'generated'):<18} {s['generated']} tokens "
          f"from a {s['prompt_tokens']}-token prompt")
    print()
    print(f"  {C.p(C.BOLD, 'bandwidth'):<18} {s['achieved_gbs']:.0f} GB/s "
          f"({s['utilisation'] * 100:.0f}% of peak)")
    print(f"  {C.p(C.BOLD, 'roofline'):<18} {bar} {s['attainment'] * 100:.0f}%")
    print(f"  {'':<18} {C.p(C.GREY, possible)}")
    print()
    print(C.p(C.GREY, "  " + read))
    print(hr())


# --------------------------------------------------------------------- modes

BENCH_PROMPTS = [
    "Explain how a CPU cache works.",
    "Write a Python function that reverses a linked list.",
    "What causes the seasons on Earth?",
    "Summarise the plot of Hamlet in three sentences.",
]


def run_stream(engine, stats: Stats, ids: list[int], max_new: int,
               temperature: float, top_p: float, show: bool) -> str:
    """Streams one completion, printing tokens as they arrive."""
    stats.prompt_tokens = len(ids)
    produced: list[int] = []
    last_render = 0

    for tid, dt, is_prefill in engine.stream(ids, max_new, temperature, top_p):
        if is_prefill:
            stats.prefill = dt
        else:
            stats.add(dt)

        produced.append(tid)
        if show:
            # Decode incrementally so multi-byte tokens render correctly.
            piece = engine.decode(produced)
            sys.stdout.write(piece[last_render:])
            sys.stdout.flush()
            last_render = len(piece)

    return engine.decode(produced)


def interactive(engine, facts: dict, args) -> int:
    print(hr("═"))
    print(f"  {C.p(C.BOLD, 'Whetstone chat')}  {C.p(C.GREY, '· engine: ' + engine.name)}")
    print(hr("═"))
    print(C.p(C.GREY, "  /reset clears history · /bench runs the throughput test · /quit exits"))
    print()

    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input(C.p(C.BLUE + C.BOLD, "you  ") + "› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            return 0
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print(C.p(C.GREY, "  history cleared"))
            continue
        if user == "/bench":
            benchmark(engine, facts, args)
            continue

        history.append({"role": "user", "content": user})
        ids = engine.encode_chat(history)

        stats = Stats(facts, engine.bytes_per_weight, args.bandwidth)
        print(C.p(C.MAGENTA + C.BOLD, "bot  ") + "› ", end="")
        sys.stdout.flush()

        text = run_stream(engine, stats, ids, args.max_new, args.temperature,
                          args.top_p, show=True)
        print()

        s = stats.summary()
        if s:
            print(C.p(C.GREY,
                      f"       {s['tok_s']:.1f} tok/s · {s['generated']} tokens · "
                      f"ttft {s['ttft_ms']:.0f} ms · "
                      f"{s['attainment']*100:.0f}% of roofline"))
        print()
        history.append({"role": "assistant", "content": text})


def benchmark(engine, facts: dict, args) -> dict:
    print()
    print(hr("═"))
    print(f"  {C.p(C.BOLD, 'throughput benchmark')}  {C.p(C.GREY, '· engine: ' + engine.name)}")
    print(hr("═"))

    # Warmup: first call pays for autotuning and clock ramp.
    warm = Stats(facts, engine.bytes_per_weight, args.bandwidth)
    run_stream(engine, warm, engine.encode_chat([{"role": "user", "content": "hi"}]),
               16, 0.0, 1.0, show=False)

    combined = Stats(facts, engine.bytes_per_weight, args.bandwidth)
    per_prompt = []

    for i, prompt in enumerate(BENCH_PROMPTS, 1):
        ids = engine.encode_chat([{"role": "user", "content": prompt}])
        st = Stats(facts, engine.bytes_per_weight, args.bandwidth)

        print(f"  {C.p(C.GREY, f'[{i}/{len(BENCH_PROMPTS)}]')} {prompt[:52]}")
        t0 = time.perf_counter()
        run_stream(engine, st, ids, args.max_new, 0.0, 1.0, show=False)
        wall = time.perf_counter() - t0

        s = st.summary()
        per_prompt.append(s)
        combined.lat.extend(st.lat)
        combined.prefill = max(combined.prefill, st.prefill)
        combined.prompt_tokens = st.prompt_tokens
        rate = f"{s['tok_s']:.1f} tok/s"
        ttft = f"ttft {s['ttft_ms']:.0f} ms"
        print(f"        {C.p(C.CYAN, rate)}  {s['generated']} tokens in "
              f"{wall:.2f}s  {C.p(C.GREY, ttft)}")

    print_summary(combined.summary(), engine.name, facts)

    if args.out:
        rec = {
            "engine": engine.name,
            "combined": combined.summary(),
            "per_prompt": per_prompt,
            "model_facts": {k: v for k, v in facts.items() if k != "config"},
        }
        Path(args.out).write_text(json.dumps(rec, indent=2))
        print(C.p(C.GREY, f"  report -> {args.out}"))
    return combined.summary()


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model directory")
    ap.add_argument("--engine", choices=["hf", "whetstone"], default="hf")
    ap.add_argument("--wstone", help=".wstone file, for --engine whetstone")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bench", action="store_true", help="non-interactive throughput run")
    ap.add_argument("--prompt", help="single prompt, then exit")
    ap.add_argument("--system", default=None, help="system message")
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--bandwidth", type=float, default=278.0,
                    help="achievable GB/s, for the roofline comparison")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--no-colour", action="store_true")
    args = ap.parse_args()

    if args.no_colour:
        C.on = False

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        print(f"error: no config.json in {model_dir}", file=sys.stderr)
        return 1

    facts = model_facts(model_dir)

    if args.engine == "whetstone":
        WhetstoneEngine(Path(args.wstone) if args.wstone else model_dir)
        return 1  # constructor always raises for now

    print(C.p(C.GREY, f"  loading {model_dir.name} ..."), end="\r")
    engine = HFEngine(model_dir, args.dtype)
    print(" " * 60, end="\r")

    if args.bench:
        benchmark(engine, facts, args)
        return 0

    if args.prompt:
        msgs = ([{"role": "system", "content": args.system}] if args.system else [])
        msgs.append({"role": "user", "content": args.prompt})

        # Warm up first: the very first forward pass pays for CUDA context
        # creation, kernel autotuning and clock ramp, which would otherwise be
        # charged to time-to-first-token and inflate it by more than a second.
        warm = Stats(facts, engine.bytes_per_weight, args.bandwidth)
        run_stream(engine, warm, engine.encode_chat([{"role": "user", "content": "hi"}]),
                   8, 0.0, 1.0, show=False)

        stats = Stats(facts, engine.bytes_per_weight, args.bandwidth)
        run_stream(engine, stats, engine.encode_chat(msgs), args.max_new,
                   args.temperature, args.top_p, show=True)
        print()
        print_summary(stats.summary(), engine.name, facts)
        return 0

    return interactive(engine, facts, args)


if __name__ == "__main__":
    raise SystemExit(main())
