#!/usr/bin/env python3
"""Tokenize an evaluation corpus into the flat `u32` file `whetstone ppl` reads.

Perplexity is only meaningful as a comparison, and a comparison is only valid if
both sides see the *same tokens in the same windows*. Passing a corpus name to
two different harnesses does not guarantee that: tokenizer versions differ, and
so do the joins between documents. So the token stream is materialised once,
here, and both HuggingFace and Whetstone read the same file.

    python bench/prepare_tokens.py --model ../models/Qwen2.5-0.5B-Instruct \
                                   --out /tmp/wikitext2.u32

Output is little-endian `uint32`, no header. `np.fromfile(path, dtype="<u4")`
reads it back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def wikitext2_test() -> str:
    """The wikitext-2 raw test split, joined the way perplexity harnesses do.

    `datasets` >= 5 requires a fully qualified namespace, so the bare "wikitext"
    id that most published harnesses still use no longer resolves on a current
    install. Try the canonical name first.
    """
    errors = []
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            from datasets import load_dataset

            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
            return "\n\n".join(ds["text"])
        except Exception as e:  # noqa: BLE001 - report every attempt, then give up
            errors.append(f"{repo}: {e}")
    raise SystemExit("could not load wikitext-2:\n  " + "\n  ".join(errors))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, help="checkpoint dir holding tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="wikitext2", choices=["wikitext2", "file"])
    ap.add_argument("--text", help="path to a UTF-8 file, when --corpus file")
    ap.add_argument("--limit", type=int, default=0, help="truncate to N tokens (0 = all)")
    ap.add_argument("--dump-text", default=None,
                    help="also write the raw corpus here. llama.cpp's perplexity tool "
                         "tokenizes text itself, so giving it this file is the closest "
                         "the two harnesses can get to reading the same thing.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import QwenTokenizer

    tok = QwenTokenizer(args.model)

    if args.corpus == "wikitext2":
        text = wikitext2_test()
    else:
        if not args.text:
            raise SystemExit("--corpus file needs --text")
        text = Path(args.text).read_text(encoding="utf-8")

    print(f"corpus: {len(text):,} characters")
    if args.dump_text:
        dt = Path(args.dump_text)
        dt.parent.mkdir(parents=True, exist_ok=True)
        dt.write_text(text, encoding="utf-8")
        print(f"wrote {dt}  ({len(text):,} characters)")
    ids = tok.encode(text)
    if args.limit:
        ids = ids[: args.limit]

    arr = np.asarray(ids, dtype="<u4")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out)

    print(f"wrote {out}  ({len(arr):,} tokens, {arr.nbytes / 1e6:.1f} MB)")
    print(f"  first 16: {arr[:16].tolist()}")
    print(f"  max id:   {int(arr.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
