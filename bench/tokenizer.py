"""Byte-level BPE tokenizer for Qwen2, implemented from the checkpoint alone.

Whetstone needs to tokenize calibration and evaluation text without pulling in
`transformers`. This reads `tokenizer.json` directly and implements the two
stages the config describes:

  1. a `Split` pre-tokenizer with the GPT-4-style regex, `Isolated` behaviour
     (the delimiters are kept as their own pieces),
  2. `ByteLevel` mapping, which makes every byte a printable codepoint so BPE
     never has to deal with invalid UTF-8,

then greedy BPE merges by rank.

Verified against the checkpoint's own vocabulary in `test_tokenizer.py`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex  # `regex`, not `re`: the pattern needs \p{L} / \p{N}


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte <-> printable-codepoint map.

    Printable ASCII, Latin-1 letters and a few symbol ranges map to themselves;
    everything else is shifted to U+0100 and up. The point is that every one of
    the 256 byte values becomes a character BPE can manipulate as text, with no
    byte sequence ever being unrepresentable.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class QwenTokenizer:
    """Byte-level BPE, loaded from a `tokenizer.json`."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        spec = json.loads((model_dir / "tokenizer.json").read_text())

        self.vocab: dict[str, int] = spec["model"]["vocab"]
        self.inv_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Merge rank: lower rank wins. The file stores them in priority order.
        merges = spec["model"]["merges"]
        if merges and isinstance(merges[0], str):
            pairs = (tuple(m.split(" ")) for m in merges)
        else:
            pairs = (tuple(m) for m in merges)
        self.ranks: dict[tuple[str, str], int] = {p: i for i, p in enumerate(pairs)}

        # Special tokens are matched before the regex so they stay atomic.
        self.special: dict[str, int] = {
            a["content"]: a["id"] for a in spec.get("added_tokens", [])
        }
        for tok, tid in self.special.items():
            self.inv_vocab.setdefault(tid, tok)

        pattern = None
        pre = spec.get("pre_tokenizer", {})
        for p in pre.get("pretokenizers", [pre]):
            if p.get("type") == "Split":
                pattern = p["pattern"]["Regex"]
        if pattern is None:
            raise ValueError("tokenizer.json has no Split pre-tokenizer pattern")
        self.pat = regex.compile(pattern)

        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        self._cache: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- BPE

    def _bpe(self, token: str) -> list[str]:
        """Greedy merge of the lowest-ranked adjacent pair, repeatedly."""
        if token in self._cache:
            return self._cache[token]

        word = list(token)
        while len(word) > 1:
            best, best_rank = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            word[best : best + 2] = [word[best] + word[best + 1]]

        self._cache[token] = word
        return word

    # ------------------------------------------------------------- encode

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """Text -> token ids."""
        if allow_special and self.special:
            return self._encode_with_special(text)
        return self._encode_ordinary(text)

    def _encode_with_special(self, text: str) -> list[int]:
        # Longest-first so <|im_start|> is not shadowed by a shorter special.
        specials = sorted(self.special, key=len, reverse=True)
        pattern = "|".join(regex.escape(s) for s in specials)

        ids: list[int] = []
        pos = 0
        for m in regex.finditer(pattern, text):
            if m.start() > pos:
                ids.extend(self._encode_ordinary(text[pos : m.start()]))
            ids.append(self.special[m.group()])
            pos = m.end()
        if pos < len(text):
            ids.extend(self._encode_ordinary(text[pos:]))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self.pat.findall(text):
            mapped = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for sym in self._bpe(mapped):
                tid = self.vocab.get(sym)
                if tid is None:
                    # Cannot happen with a well-formed vocab: every single
                    # byte-character is in it, so BPE always bottoms out.
                    raise KeyError(f"symbol {sym!r} missing from vocab")
                ids.append(tid)
        return ids

    # ------------------------------------------------------------- decode

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text."""
        out = bytearray()
        for i in ids:
            sym = self.inv_vocab.get(i)
            if sym is None:
                continue
            if sym in self.special:
                out.extend(sym.encode("utf-8"))
            else:
                out.extend(self.u2b[c] for c in sym)
        return out.decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    import sys

    tk = QwenTokenizer(sys.argv[1] if len(sys.argv) > 1
                       else "../models/Qwen2.5-0.5B-Instruct")
    print(f"vocab {len(tk)}, {len(tk.ranks)} merges, {len(tk.special)} special")

    cases = [
        "The capital of France is",
        "def Fibonacci(n):",
        "import numpy as np",
        "In 1969, humanity first walked on the Moon.",
        "  leading and trailing  ",
        "unicode: café, 日本語, emoji 🔥",
        "<|im_start|>user\nhi<|im_end|>",
    ]
    ok = True
    for c in cases:
        ids = tk.encode(c)
        back = tk.decode(ids)
        good = back == c
        ok &= good
        print(f"{'ok ' if good else 'FAIL'} {c!r}")
        print(f"      {len(ids)} ids {ids[:12]}{'...' if len(ids) > 12 else ''}")
        if not good:
            print(f"      decoded {back!r}")
    print("\nround-trip:", "all passed" if ok else "FAILURES")
