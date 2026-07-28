//! `whetstone chat` — an interactive REPL, with throughput per turn.
//!
//! # Why the KV cache is never rebuilt
//!
//! The obvious way to hold a conversation is to re-send the whole transcript
//! each turn, which is what most wrappers do because it is stateless and easy.
//! It also means turn *n* re-prefills everything turns 1..n-1 already computed —
//! quadratic work in the length of the conversation, and by turn ten most of the
//! machine's time is spent recomputing what it already knows.
//!
//! Whetstone keeps the cache. Each turn appends only the new tokens, so the
//! reported prefill figure covers the user's message and nothing else, and turn
//! twenty costs the same as turn two. What that buys is visible in the per-turn
//! numbers: prefill tokens stay small while the context grows.
//!
//! The cost is that the cache is the conversation. `/reset` drops it; running
//! out of context is unrecoverable without one, so it is reported rather than
//! silently truncated — dropping the oldest turns would change the model's
//! answers without saying so.
//!
//! # Sampling
//!
//! Greedy stays entirely on the device: the argmax writes the chosen id into the
//! device cursor the next step's embedding gather reads, so a whole generation
//! runs without a token crossing the bus. Temperature sampling cannot — it needs
//! the distribution on the host, which is a 608 KB transfer plus an O(vocab)
//! selection per token. Measured here: **369 tok/s sampling against 467 greedy**,
//! so roughly a fifth of the throughput.
//!
//! The default is Qwen's own recommended temperature anyway, because a REPL that
//! loops on repeated text demonstrates nothing; `--temperature 0` is there when
//! the fastest path is what matters. (An earlier version of the sampler sorted
//! the whole 151936-entry vocabulary per token and ran at 111 tok/s — four times
//! slower than the forward pass it was sampling from.)

use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use whetstone_core::{Engine, ModelWeights, Sampler, StreamDecoder, Tokenizer};

/// Whether to emit ANSI styling.
///
/// Three ways this goes wrong if you just always emit it:
///
/// - **piped output** picks up escape codes as literal text, which ruins
///   `whetstone chat --prompt ... > file`;
/// - **`NO_COLOR`** is a convention worth honouring;
/// - **Windows consoles** do not process virtual-terminal sequences unless the
///   process enables it, so on anything but Windows Terminal the escapes print
///   as `←[1m`. Enabling VT would mean a `windows-sys` dependency for
///   decoration, so the answer is simply not to style there.
fn use_ansi() -> bool {
    if !std::io::stdout().is_terminal() || std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if cfg!(windows) {
        // Windows Terminal sets this and does process VT sequences.
        return std::env::var_os("WT_SESSION").is_some();
    }
    true
}

/// Qwen2's chat markup.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

pub struct ChatArgs<'a> {
    pub model: &'a Path,
    /// Directory holding `tokenizer.json`, when the `.wstone` has none embedded.
    pub tokenizer: Option<&'a Path>,
    pub system: Option<String>,
    pub ctx: usize,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    /// Answer this and exit, instead of reading from the terminal.
    pub prompt: Option<String>,
}

pub fn run(args: ChatArgs<'_>) -> Result<()> {
    let t0 = Instant::now();
    let weights = ModelWeights::load(args.model)
        .with_context(|| format!("could not load {}", args.model.display()))?;

    // Prefer the tokenizer the converter embedded: a `.wstone` is meant to be
    // self-contained, and requiring the original checkpoint alongside it would
    // make that claim false.
    let tokenizer = match (&weights.tokenizer_json, args.tokenizer) {
        (Some(json), _) => Tokenizer::from_json(json)?,
        (None, Some(dir)) => Tokenizer::from_dir(dir)
            .with_context(|| format!("could not read tokenizer.json from {}", dir.display()))?,
        (None, None) => bail!(
            "{} has no embedded tokenizer and --tokenizer was not given.\n\
             Re-run `whetstone convert` (it embeds one automatically), or pass\n\
             --tokenizer <dir containing tokenizer.json>",
            args.model.display()
        ),
    };

    let im_end = tokenizer
        .token_id(IM_END)
        .context("tokenizer has no <|im_end|>; this does not look like a Qwen chat model")?;
    let eos = tokenizer.token_id("<|endoftext|>");

    let decode_bytes = weights.decode_bytes();
    let bpw = weights.bits_per_weight();
    let scheme = weights.quant_meta.get("scheme").cloned().unwrap_or_default();

    let mut engine = Engine::new(weights, args.ctx)?;
    let peak = engine.device().bandwidth_gbs();
    let launches = engine.capture_graph()?;
    let load_s = t0.elapsed().as_secs_f64();

    println!("{:=<72}", "");
    println!("  {}", engine.device());
    println!("  {}", args.model.display());
    println!("{:=<72}", "");
    println!(
        "  format             {scheme}, {bpw:.2} bits/weight, {:.0} MB/token",
        decode_bytes as f64 / 1e6
    );
    println!(
        "  roofline           {:.0} tok/s at {peak:.0} GB/s peak",
        peak * 1e9 / decode_bytes as f64
    );
    println!("  context            {} tokens", args.ctx);
    println!("  cuda graph         {launches} launches per token collapsed into 1");
    println!(
        "  sampling           {}",
        if args.temperature <= 0.0 {
            "greedy (stays on the device)".to_string()
        } else {
            format!(
                "temperature {:.2}, top-p {:.2} (costs a logit copy per token)",
                args.temperature, args.top_p
            )
        }
    );
    println!("  loaded in          {load_s:.2} s");
    println!("{:-<72}", "");

    let sampler = if args.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::TopP { temperature: args.temperature, top_p: args.top_p, seed: args.seed }
    };

    let ansi = use_ansi();
    let system = args.system.clone().unwrap_or_else(|| DEFAULT_SYSTEM.into());
    let mut turn = 0usize;
    let mut totals = (0usize, 0.0f64); // generated tokens, decode seconds

    // One-shot mode: answer and exit, so the command is scriptable.
    if let Some(p) = &args.prompt {
        let prompt = segment(&system, p, true);
        answer(&mut engine, &tokenizer, &prompt, sampler, args.max_new, im_end, eos,
               decode_bytes, ansi, &mut turn, &mut totals)?;
        return Ok(());
    }

    println!("  Type a message. /reset clears the conversation, /quit exits.");
    println!();

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("{}", if ansi { "\x1b[1m> \x1b[0m" } else { "> " });
        std::io::stdout().flush()?;

        let Some(line) = lines.next() else { break };
        let line = line?;
        let msg = line.trim();

        match msg {
            "" => continue,
            "/quit" | "/exit" | "/q" => break,
            "/reset" => {
                engine.reset()?;
                turn = 0;
                println!("  (conversation cleared, KV cache dropped)\n");
                continue;
            }
            "/stats" => {
                if totals.0 > 0 {
                    println!(
                        "  {} tokens over {:.2} s = {:.1} tok/s across the session\n",
                        totals.0,
                        totals.1,
                        totals.0 as f64 / totals.1
                    );
                } else {
                    println!("  nothing generated yet\n");
                }
                continue;
            }
            _ => {}
        }

        // Only the new turn goes in; everything before it is already in the KV
        // cache. The system prompt therefore appears exactly once.
        let prompt = segment(&system, msg, turn == 0);
        if let Err(e) = answer(&mut engine, &tokenizer, &prompt, sampler, args.max_new,
                               im_end, eos, decode_bytes, ansi, &mut turn, &mut totals) {
            println!("\n  error: {e}\n");
            if format!("{e}").contains("context is full") {
                println!("  The KV cache is full. /reset to start over, or restart with a");
                println!("  larger --ctx. Silently dropping old turns would change the");
                println!("  model's answers without telling you, so it is not done.\n");
            }
        }
    }

    if totals.0 > 0 {
        println!();
        println!(
            "  session: {} tokens over {:.2} s = {:.1} tok/s",
            totals.0,
            totals.1,
            totals.0 as f64 / totals.1
        );
    }
    Ok(())
}

/// The tokens for one user turn, in Qwen2's chat markup.
///
/// `with_system` only on the first turn: the cache already holds every earlier
/// turn, so repeating the system prompt would both waste context and give the
/// model two of them.
fn segment(system: &str, user: &str, with_system: bool) -> String {
    if with_system {
        format!(
            "{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"
        )
    } else {
        // The previous turn stopped *before* emitting <|im_end|>, so this closes
        // it. Getting that wrong shifts every subsequent turn's markup by one
        // token and degrades the model in a way that reads as it being dim.
        format!("{IM_END}\n{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n")
    }
}

#[allow(clippy::too_many_arguments)]
fn answer(
    engine: &mut Engine,
    tok: &Tokenizer,
    prompt: &str,
    sampler: Sampler,
    max_new: usize,
    im_end: u32,
    eos: Option<u32>,
    decode_bytes: usize,
    ansi: bool,
    turn: &mut usize,
    totals: &mut (usize, f64),
) -> Result<()> {
    let ids = tok.encode(prompt);
    let mut dec = StreamDecoder::default();
    let mut stdout = std::io::stdout();

    let t0 = Instant::now();
    let stats = engine.generate(&ids, max_new, sampler, |id| {
        if id == im_end || Some(id) == eos {
            return false;
        }
        let text = dec.push(tok, id);
        if !text.is_empty() {
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        }
        true
    })?;
    let tail = dec.finish();
    if !tail.is_empty() {
        let _ = stdout.write_all(tail.as_bytes());
    }
    let wall = t0.elapsed().as_secs_f64();

    *turn += 1;
    totals.0 += stats.generated;
    totals.1 += stats.decode_seconds;

    // ASCII only in the stats line. A Windows console on codepage 437 or 1252
    // renders UTF-8 punctuation as mojibake, and a throughput readout that is
    // hard to read on the platform it is reporting for is not much of a readout.
    let achieved = decode_bytes as f64 * stats.decode_tok_s() / 1e9;
    let (dim, off) = if ansi { ("\x1b[2m", "\x1b[0m") } else { ("", "") };

    println!();
    print!(
        "{dim}  [{:.1} tok/s | {} tokens in {:.2} s | prefill {} in {:.0} ms | {:.0} GB/s | ctx {}]{off}",
        stats.decode_tok_s(),
        stats.generated,
        stats.decode_seconds,
        stats.prompt_tokens,
        stats.prefill_seconds * 1e3,
        achieved,
        engine.position(),
    );
    // The wall clock includes tokenization and terminal writes; when it diverges
    // from the decode time the bottleneck is not the model, and saying so beats
    // letting someone tune kernels against a number the terminal set.
    if wall > stats.decode_seconds * 1.25 {
        print!("{dim}  (wall {wall:.2} s - output, not the engine){off}");
    }
    println!();
    println!();
    Ok(())
}
