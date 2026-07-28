//! The tokenizer against the real checkpoint, when one is present.
//!
//! The unit tests in `tokenizer.rs` pin the pre-tokenizer's tricky branches on
//! synthetic input. This pins the whole thing against the actual vocabulary and
//! merge table, because the failure mode that matters — an off-by-one in the
//! byte-level map, a merge rank read in the wrong order — produces *plausible*
//! ids that only diverge from the reference on real text.
//!
//! Skipped when the checkpoint is absent, so CI on a machine without a 1 GB
//! model download stays green. Point `WHETSTONE_TEST_MODEL` at a model directory
//! to run it elsewhere.

use std::path::PathBuf;

use whetstone_core::{StreamDecoder, Tokenizer};

fn model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WHETSTONE_TEST_MODEL") {
        let p = PathBuf::from(p);
        return p.join("tokenizer.json").exists().then_some(p);
    }
    let guess = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/Qwen2.5-0.5B-Instruct");
    guess.join("tokenizer.json").exists().then_some(guess)
}

fn tokenizer() -> Option<Tokenizer> {
    Tokenizer::from_dir(model_dir()?).ok()
}

#[test]
fn text_survives_a_round_trip() {
    let Some(t) = tokenizer() else {
        eprintln!("skip: no checkpoint");
        return;
    };

    // Each of these has broken a byte-level BPE at some point: CJK and emoji
    // are multi-token single characters, combining marks exercise NFC, and the
    // whitespace cases are what the `(?!\S)` lookahead exists for.
    let cases = [
        "The capital of France is Paris.",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
        "  leading and trailing   ",
        "line one\n\nline two\r\nline three",
        "\u{4f60}\u{597d}\u{ff0c}\u{4e16}\u{754c}",
        "emoji: \u{1f642}\u{1f643}\u{1f680}",
        "caf\u{e9} vs cafe\u{301}",
        "1234567890 and 3.14159",
        "don't can't we'll they've I'm he'd It's",
        "<|im_start|>user\nhi<|im_end|>",
        "",
        " ",
        "\t\t",
    ];

    for c in cases {
        let ids = t.encode(c);
        let back = t.decode(&ids);
        // NFC is applied on the way in, so the composed form is what comes
        // back. Normalising here independently, rather than by round-tripping
        // again, keeps the assertion from being tautological.
        let want: String = unicode_normalization::UnicodeNormalization::nfc(c).collect();
        assert_eq!(back, want, "round trip failed for {c:?} (ids {ids:?})");
    }
}

#[test]
fn chat_markup_tokenizes_as_single_tokens() {
    let Some(t) = tokenizer() else {
        eprintln!("skip: no checkpoint");
        return;
    };
    // If these split into pieces the chat template silently degrades: the model
    // still answers, just worse, which is the hardest kind of bug to notice.
    for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        let ids = t.encode(s);
        assert_eq!(ids.len(), 1, "{s} tokenized as {ids:?}, expected one token");
        assert_eq!(t.token_id(s), Some(ids[0]));
    }

    let prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n";
    let ids = t.encode(prompt);
    let start = t.token_id("<|im_start|>").unwrap();
    let end = t.token_id("<|im_end|>").unwrap();
    assert_eq!(ids.iter().filter(|&&i| i == start).count(), 2);
    assert_eq!(ids.iter().filter(|&&i| i == end).count(), 2);
    assert_eq!(t.decode(&ids), prompt);
}

#[test]
fn known_prompt_matches_the_reference_ids() {
    let Some(t) = tokenizer() else {
        eprintln!("skip: no checkpoint");
        return;
    };
    // Resolved from the checkpoint's own vocabulary and used throughout the
    // benchmarks; if these drift, every recorded measurement stops being
    // reproducible.
    assert_eq!(t.encode("The capital of France is"), vec![785, 6722, 315, 9625, 374]);
    assert_eq!(t.encode("import numpy as np"), vec![474, 8591, 438, 2595]);
}

#[test]
fn streaming_reassembles_exactly_what_a_bulk_decode_gives() {
    let Some(t) = tokenizer() else {
        eprintln!("skip: no checkpoint");
        return;
    };
    // A multi-byte character spans several tokens, so a decoder that emits each
    // token independently produces replacement characters mid-word.
    let text = "\u{4f60}\u{597d} \u{1f680} caf\u{e9} \u{4e16}\u{754c}!";
    let ids = t.encode(text);

    let mut d = StreamDecoder::default();
    let mut streamed = String::new();
    for &id in &ids {
        streamed.push_str(&d.push(&t, id));
    }
    streamed.push_str(&d.finish());

    assert_eq!(streamed, t.decode(&ids));
    assert!(!streamed.contains('\u{fffd}'), "streaming emitted a replacement char");
}
