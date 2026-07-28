//! Byte-level BPE, read straight from a `tokenizer.json`.
//!
//! Whetstone's premise is that no Python sits in the token loop, and a chat REPL
//! that shells out to `transformers` to turn text into ids would break that at
//! the very first step. So this is the tokenizer, in Rust, implemented from the
//! checkpoint's own `tokenizer.json` and nothing else.
//!
//! # The three stages
//!
//! 1. **NFC normalisation**, because `tokenizer.json` declares it. Skipping it
//!    would tokenize a composed `é` differently from a decomposed one, and the
//!    two look identical on screen.
//! 2. **Pre-tokenization**, splitting on the GPT-4 pattern the file specifies.
//! 3. **Byte-level mapping then greedy BPE by merge rank.** Every byte becomes a
//!    printable codepoint first, so BPE never has to reason about invalid UTF-8
//!    and no byte sequence is unrepresentable.
//!
//! # Why the pre-tokenizer is hand-written
//!
//! The declared pattern is
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! and `\s+(?!\S)` is a negative lookahead, which Rust's `regex` crate does not
//! support — it guarantees linear time by excluding exactly this construct. The
//! options were a backtracking regex dependency or a scanner. The scanner is
//! seven branches of `match`, has no dependency, and is verified against the
//! reference implementation on the whole wikitext-2 corpus, which is a stronger
//! check than either would get on its own.
//!
//! The two alternatives that need care are the ones the lookahead exists for:
//!
//! - `\s*[\r\n]+` — greedy, so it consumes a whitespace run up to and including
//!   its **last** newline, leaving any trailing spaces behind.
//! - `\s+(?!\S)` — a whitespace run that is *not* followed by visible text, so
//!   at end of input it takes the whole run and otherwise it must leave the last
//!   character for the next piece. That single held-back space is what makes
//!   `" hello"` tokenize as one piece rather than two.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

use crate::error::{Error, Result};

/// A byte-level BPE tokenizer.
pub struct Tokenizer {
    /// Byte-level token text to id.
    vocab: HashMap<String, u32>,
    /// Id to byte-level token text.
    inv: Vec<String>,
    /// Merge priority; lower wins.
    ranks: HashMap<(String, String), u32>,
    /// Added tokens, longest first so `<|im_start|>` cannot be shadowed.
    specials: Vec<(String, u32)>,
    /// The 256 printable codepoints bytes map to.
    byte_to_char: [char; 256],
    /// The inverse.
    char_to_byte: HashMap<char, u8>,
}

/// GPT-2's reversible byte to printable-codepoint map.
///
/// Printable ASCII, Latin-1 letters and a few symbol ranges map to themselves;
/// every other byte is shifted to U+0100 and up. The point is that all 256 byte
/// values become characters BPE can manipulate as text, with no byte sequence
/// unrepresentable and no invalid UTF-8 ever constructed mid-merge.
fn byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1u8..=0xAC);
    bs.extend(0xAEu8..=0xFF);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut fwd = ['\0'; 256];
    let mut rev = HashMap::with_capacity(256);
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("byte map codepoints are all valid");
        fwd[b as usize] = ch;
        rev.insert(ch, b);
    }
    (fwd, rev)
}

impl Tokenizer {
    /// Parses a `tokenizer.json`.
    pub fn from_json(text: &str) -> Result<Self> {
        let spec: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Config(format!("tokenizer.json is not valid JSON: {e}")))?;

        let model = spec.get("model").ok_or_else(|| {
            Error::Config("tokenizer.json has no \"model\" section".into())
        })?;

        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Config("tokenizer.json has no vocab map".into()))?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut max_id = 0u32;
        for (k, v) in vocab_obj {
            let id = v
                .as_u64()
                .ok_or_else(|| Error::Config(format!("vocab entry {k:?} is not an integer")))?
                as u32;
            max_id = max_id.max(id);
            vocab.insert(k.clone(), id);
        }

        // Added tokens carry ids above the base vocabulary and must be matched
        // before any splitting, or `<|im_start|>` would be chopped into pieces.
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(added) = spec.get("added_tokens").and_then(|v| v.as_array()) {
            for t in added {
                let (Some(c), Some(id)) = (
                    t.get("content").and_then(|v| v.as_str()),
                    t.get("id").and_then(|v| v.as_u64()),
                ) else {
                    continue;
                };
                max_id = max_id.max(id as u32);
                specials.push((c.to_string(), id as u32));
                vocab.entry(c.to_string()).or_insert(id as u32);
            }
        }
        // Longest first: a shorter special must never win over a longer one that
        // starts at the same place.
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));

        let mut inv = vec![String::new(); max_id as usize + 1];
        for (tok, &id) in &vocab {
            inv[id as usize] = tok.clone();
        }

        // Merges are either "a b" strings or ["a", "b"] pairs depending on the
        // `tokenizers` version that wrote the file. Accept both.
        let merges = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Config("tokenizer.json has no merge list".into()))?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let pair = if let Some(s) = m.as_str() {
                let mut it = s.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                    _ => return Err(Error::Config(format!("malformed merge {s:?}"))),
                }
            } else if let Some(a) = m.as_array() {
                match (a.first().and_then(|v| v.as_str()), a.get(1).and_then(|v| v.as_str())) {
                    (Some(x), Some(y)) => (x.to_string(), y.to_string()),
                    _ => return Err(Error::Config(format!("malformed merge {m:?}"))),
                }
            } else {
                return Err(Error::Config(format!("malformed merge {m:?}")));
            };
            ranks.insert(pair, rank as u32);
        }

        let (byte_to_char, char_to_byte) = byte_maps();
        Ok(Self { vocab, inv, ranks, specials, byte_to_char, char_to_byte })
    }

    /// Reads `tokenizer.json` from a directory.
    pub fn from_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = dir.as_ref().join("tokenizer.json");
        let s = std::fs::read_to_string(&p)
            .map_err(|e| Error::Io(format!("could not read {}: {e}", p.display())))?;
        Self::from_json(&s)
    }

    /// Vocabulary size, counting added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inv.len()
    }

    /// Looks up a token by its literal text, e.g. `"<|im_end|>"`.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.vocab.get(text).copied()
    }

    /// Encodes text to token ids.
    ///
    /// Added tokens such as `<|im_start|>` are recognised in the input, which is
    /// what lets a chat template be built as a plain string.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized: String = text.nfc().collect();
        let mut out = Vec::new();
        self.encode_with_specials(&normalized, &mut out);
        out
    }

    fn encode_with_specials(&self, text: &str, out: &mut Vec<u32>) {
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // Earliest special token anywhere in what is left; ties broken by the
            // longest, since `specials` is sorted that way.
            let mut best: Option<(usize, &String, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(pos) = rest.find(s.as_str()) {
                    if best.map_or(true, |(p, _, _)| pos < p) {
                        best = Some((pos, s, *id));
                    }
                }
            }

            match best {
                Some((pos, s, id)) => {
                    if pos > 0 {
                        self.encode_plain(&rest[..pos], out);
                    }
                    out.push(id);
                    rest = &rest[pos + s.len()..];
                }
                None => {
                    self.encode_plain(rest, out);
                    break 'outer;
                }
            }
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pre_tokenize(text) {
            // Byte level first, so BPE only ever sees printable characters.
            let mapped: String =
                piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            self.bpe(&mapped, out);
        }
    }

    /// Greedy BPE: repeatedly merge the adjacent pair with the lowest rank.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.is_empty() {
            return;
        }

        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map_or(true, |(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }

        for p in parts {
            match self.vocab.get(&p) {
                Some(&id) => out.push(id),
                // Unreachable for a well-formed vocabulary: byte-level mapping
                // guarantees every single character is in it. Falling back to
                // per-character is still better than dropping text silently.
                None => {
                    for c in p.chars() {
                        if let Some(&id) = self.vocab.get(&c.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Decodes ids back to bytes.
    ///
    /// Bytes rather than a `String`, because a single token can end mid-UTF-8 —
    /// one emoji is several tokens — and a streaming caller must be able to hold
    /// the tail until it completes. See [`StreamDecoder`].
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let Some(tok) = self.inv.get(id as usize) else { continue };
            for c in tok.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => out.push(b),
                    // An added token like `<|im_end|>` is stored as literal text
                    // rather than byte-level, so it decodes as itself.
                    None => {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        out
    }

    /// Decodes ids to a string, replacing anything that is not valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// Emits text as tokens arrive, holding back incomplete UTF-8.
///
/// A multi-byte character is usually several tokens, so decoding each token
/// independently produces replacement characters mid-word. This buffers the tail
/// until it forms valid UTF-8 — which is the difference between a chat REPL that
/// streams cleanly and one that flickers `<?>` on every non-ASCII word.
#[derive(Default)]
pub struct StreamDecoder {
    buf: Vec<u8>,
}

impl StreamDecoder {
    /// Feeds one token and returns whatever text is now complete.
    ///
    /// The distinction that matters is between the two ways `from_utf8` can
    /// fail. `error_len() == None` means the tail is a *truncated but still
    /// valid* prefix — hold it, the next token completes it. `Some(n)` means
    /// those n bytes can never become valid whatever follows, so they must be
    /// dropped. Treating both as "hold" is the bug this replaced: one malformed
    /// byte would wedge the buffer and the stream would go silent for the rest
    /// of the generation.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.buf.extend_from_slice(&tok.decode_bytes(&[id]));

        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    if good > 0 {
                        out.push_str(&String::from_utf8_lossy(&self.buf[..good]));
                    }
                    match e.error_len() {
                        Some(n) => {
                            // Unsalvageable: report it and move past.
                            self.buf.drain(..good + n);
                            out.push(char::REPLACEMENT_CHARACTER);
                        }
                        None => {
                            self.buf.drain(..good);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// Flushes anything left, lossily. Call at the end of a generation.
    pub fn finish(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        out
    }
}

// ------------------------------------------------------------- pre-tokenizer

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Splits text on the GPT-4 pattern `tokenizer.json` declares.
///
/// Alternatives are tried in order, which is what a leftmost-first regex engine
/// does with `|`. See the module docs for why this is not a `Regex`.
fn pre_tokenize(text: &str) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;

    // Character at a byte offset, and the offset just past it.
    let at = |i: usize| -> Option<(char, usize)> {
        if i >= b.len() {
            return None;
        }
        let c = text[i..].chars().next()?;
        Some((c, i + c.len_utf8()))
    };

    while i < b.len() {
        let (c, next) = match at(i) {
            Some(v) => v,
            None => break,
        };
        let start = i;

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' {
            let tail = &text[next..];
            let lower = tail.chars().take(2).collect::<String>().to_lowercase();
            let n = if lower.starts_with("re") || lower.starts_with("ve")
                || lower.starts_with("ll")
            {
                2
            } else if lower.starts_with('s') || lower.starts_with('t')
                || lower.starts_with('m') || lower.starts_with('d')
            {
                1
            } else {
                0
            };
            if n > 0 {
                let end = next + tail.chars().take(n).map(char::len_utf8).sum::<usize>();
                out.push(&text[start..end]);
                i = end;
                continue;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            let mut j = i;
            if !is_nl(c) && !is_letter(c) && !is_number(c) {
                // The optional prefix only counts if a letter actually follows.
                if let Some((c2, _)) = at(next) {
                    if is_letter(c2) {
                        j = next;
                    }
                }
            }
            if let Some((cj, _)) = at(j) {
                if is_letter(cj) {
                    let mut k = j;
                    while let Some((ck, nk)) = at(k) {
                        if !is_letter(ck) {
                            break;
                        }
                        k = nk;
                    }
                    out.push(&text[start..k]);
                    i = k;
                    continue;
                }
            }
        }

        // 3. \p{N}  -- one digit at a time, which is what Qwen's pattern says
        if is_number(c) {
            out.push(&text[start..next]);
            i = next;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let mut j = i;
            if c == ' ' {
                match at(next) {
                    Some((c2, _))
                        if !c2.is_whitespace() && !is_letter(c2) && !is_number(c2) =>
                    {
                        j = next
                    }
                    _ => {}
                }
            }
            if let Some((cj, _)) = at(j) {
                if !cj.is_whitespace() && !is_letter(cj) && !is_number(cj) {
                    let mut k = j;
                    while let Some((ck, nk)) = at(k) {
                        if ck.is_whitespace() || is_letter(ck) || is_number(ck) {
                            break;
                        }
                        k = nk;
                    }
                    while let Some((ck, nk)) = at(k) {
                        if !is_nl(ck) {
                            break;
                        }
                        k = nk;
                    }
                    out.push(&text[start..k]);
                    i = k;
                    continue;
                }
            }
        }

        // 5/6/7 all begin with a whitespace run, so measure it once.
        if c.is_whitespace() {
            let mut end = i;
            let mut last_nl: Option<usize> = None;
            while let Some((ck, nk)) = at(end) {
                if !ck.is_whitespace() {
                    break;
                }
                if is_nl(ck) {
                    last_nl = Some(nk);
                }
                end = nk;
            }

            // 5. \s*[\r\n]+ -- greedy, so it stops after the run's last newline
            // and leaves any trailing spaces to the next piece.
            if let Some(nl_end) = last_nl {
                out.push(&text[start..nl_end]);
                i = nl_end;
                continue;
            }

            // 6. \s+(?!\S) -- the whole run at end of input, otherwise all but
            // the last character, so the next piece keeps its leading space.
            if end >= b.len() {
                out.push(&text[start..end]);
                i = end;
                continue;
            }
            let last_char_start = text[start..end].chars().next_back().map(|c| end - c.len_utf8());
            if let Some(lcs) = last_char_start {
                if lcs > start {
                    out.push(&text[start..lcs]);
                    i = lcs;
                    continue;
                }
            }

            // 7. \s+
            out.push(&text[start..end]);
            i = end;
            continue;
        }

        // Nothing matched: consume one character rather than loop forever.
        out.push(&text[start..next]);
        i = next;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_is_a_bijection() {
        let (fwd, rev) = byte_maps();
        assert_eq!(rev.len(), 256);
        for b in 0u8..=255 {
            assert_eq!(rev[&fwd[b as usize]], b, "byte {b} did not round trip");
        }
    }

    #[test]
    fn pre_tokenizer_holds_back_the_space_before_a_word() {
        // This is what `\s+(?!\S)` is for: the run of spaces before "hello"
        // must leave its last space to be absorbed into " hello".
        assert_eq!(pre_tokenize("a  hello"), vec!["a", " ", " hello"]);
        assert_eq!(pre_tokenize("hi there"), vec!["hi", " there"]);
        // Trailing whitespace has nothing after it, so it is kept whole.
        assert_eq!(pre_tokenize("hi  "), vec!["hi", "  "]);
    }

    #[test]
    fn pre_tokenizer_stops_a_newline_run_after_its_last_newline() {
        // `\s*[\r\n]+` is greedy, so "  \n\n  " splits after the newlines.
        assert_eq!(pre_tokenize("a  \n\n  b"), vec!["a", "  \n\n", " ", " b"]);
        assert_eq!(pre_tokenize("x\ny"), vec!["x", "\n", "y"]);
    }

    #[test]
    fn pre_tokenizer_splits_digits_singly_and_keeps_contractions() {
        assert_eq!(pre_tokenize("in 2024"), vec!["in", " ", "2", "0", "2", "4"]);
        assert_eq!(pre_tokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pre_tokenize("It's"), vec!["It", "'s"]);
        assert_eq!(pre_tokenize("we'll"), vec!["we", "'ll"]);
    }

    #[test]
    fn pre_tokenizer_terminates_on_every_input() {
        // The scanner advances at least one character per iteration; this pins
        // that, because the failure mode is a hang rather than a wrong answer.
        for s in ["", " ", "\u{0}", "🙂", "\u{200b}", "a\u{301}", "!!!", "\r\n\r\n"] {
            let parts = pre_tokenize(s);
            assert_eq!(parts.concat(), s, "pieces must reassemble {s:?}");
        }
    }

    #[test]
    fn pre_tokenizer_output_always_reassembles() {
        let corpus = "The quick brown fox — 42 apples, 3.14159!\n\n  Résumé naïve \
                      \u{4f60}\u{597d} 🙂🙃 tabs\there\r\n  trailing   ";
        assert_eq!(pre_tokenize(corpus).concat(), corpus);
    }

    #[test]
    fn stream_decoder_holds_partial_utf8() {
        // Nothing model-specific: a hand-built tokenizer whose two tokens are
        // the halves of one 3-byte character.
        let spec = serde_json::json!({
            "added_tokens": [],
            "model": {
                "type": "BPE",
                // 0xE4 0xBD 0xA0 is U+4F60. Byte-level chars for those bytes.
                "vocab": { "ä": 0, "½": 1, " ": 2, "A": 3 },
                "merges": []
            }
        });
        let t = Tokenizer::from_json(&spec.to_string()).unwrap();

        let mut d = StreamDecoder::default();
        assert_eq!(d.push(&t, 0), "", "an incomplete character must not be emitted");
        assert_eq!(d.push(&t, 1), "", "still incomplete");

        // Now break it. The two held bytes can never complete, so they become a
        // replacement character and the stream continues -- the property under
        // test is that it does not wedge. An earlier version returned "" here
        // and stayed silent for the rest of the generation.
        assert_eq!(d.push(&t, 3), "\u{fffd}A");
        assert_eq!(d.push(&t, 3), "A", "the decoder recovered");
        assert_eq!(d.finish(), "", "nothing left held");
    }

    #[test]
    fn stream_decoder_completes_a_split_character() {
        // The case that actually happens: a multi-byte character arriving as
        // several tokens, which must emit once and only once, when complete.
        let spec = serde_json::json!({
            "added_tokens": [],
            "model": {
                "type": "BPE",
                // U+4F60 is the bytes E4 BD A0. Under the byte-level map E4 and
                // BD fall in the identity ranges, but A0 does not — it is the
                // 34th unmapped byte and lands on U+0142. Writing "\u{a0}" here
                // would silently encode a different byte and the test would be
                // checking nothing.
                "vocab": { "ä": 0, "½": 1, "ł": 2 },
                "merges": []
            }
        });
        let t = Tokenizer::from_json(&spec.to_string()).unwrap();

        let mut d = StreamDecoder::default();
        assert_eq!(d.push(&t, 0), "");
        assert_eq!(d.push(&t, 1), "");
        assert_eq!(d.push(&t, 2), "\u{4f60}", "the character emits once complete");
        assert_eq!(d.finish(), "");
    }
}
