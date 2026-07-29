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
use whetstone_core::{
    Engine, ModelWeights, Sampler, SamplingConfig, StreamDecoder, Tokenizer,
};

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
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    /// Answer this and exit, instead of reading from the terminal.
    pub prompt: Option<String>,
    /// VRAM budget for weights. `None` keeps everything on the device.
    pub vram: Option<String>,
    /// Draft width for speculative decoding. `0` disables.
    pub spec: usize,
}

impl ChatArgs<'_> {
    fn sampling(&self) -> SamplingConfig {
        SamplingConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed: self.seed,
        }
    }
}

/// One line describing what the sampler will do.
fn describe(cfg: &SamplingConfig) -> String {
    if cfg.temperature <= 0.0 {
        return "greedy (stays on the device)".into();
    }
    let mut parts = vec![format!("temp {:.2}", cfg.temperature)];
    if cfg.top_k > 0 {
        parts.push(format!("top-k {}", cfg.top_k));
    }
    if cfg.top_p < 1.0 {
        parts.push(format!("top-p {:.2}", cfg.top_p));
    }
    if cfg.min_p > 0.0 {
        parts.push(format!("min-p {:.2}", cfg.min_p));
    }
    if cfg.repeat_penalty != 1.0 {
        parts.push(format!("repeat {:.2}/{}", cfg.repeat_penalty, cfg.repeat_last_n));
    }
    format!("{} (costs a logit copy per token)", parts.join(", "))
}

pub fn run(args: ChatArgs<'_>) -> Result<()> {
    let budget = args.vram.as_deref().map(crate::run::parse_bytes).transpose()?;

    let t0 = Instant::now();
    let weights = ModelWeights::load_with(args.model, budget)
        .with_context(|| format!("could not load {}", args.model.display()))?;
    let residency = weights.residency;

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
    let kv = engine.weights().config.kv_cache_bytes(args.ctx, 2);
    println!(
        "  context            {} tokens  (KV cache {:.0} MB)",
        args.ctx,
        kv as f64 / 1e6
    );
    if !residency.fully_resident() {
        println!(
            "  offload            {} of {} blocks in host RAM ({:.0} MB over PCIe/token)",
            residency.host_layers,
            residency.host_layers + residency.device_layers,
            residency.host_bytes as f64 / 1e6
        );
    }
    if args.spec > 1 {
        println!(
            "  speculation        draft {} per verification pass (output is greedy-exact)",
            args.spec
        );
    }
    println!(
        "  vram               {:.0} MB weights + {:.0} MB cache of {:.1} GB",
        decode_bytes as f64 / 1e6,
        kv as f64 / 1e6,
        engine.device().mem_total() as f64 / 1e9
    );
    println!("  cuda graph         {launches} launches per token collapsed into 1");
    println!("  sampling           {}", describe(&args.sampling()));
    println!("  loaded in          {load_s:.2} s");
    println!("{:-<72}", "");

    // Mutable, because `/set` changes it between turns. Kept as a config rather
    // than a `Sampler` so `/set temperature 0` can switch to the greedy path,
    // which is a different code path and 20% faster.
    let mut cfg = args.sampling();
    let sampler_of = |c: &SamplingConfig| {
        if c.temperature <= 0.0 { Sampler::Greedy } else { Sampler::Sample(*c) }
    };

    let ansi = use_ansi();
    let system = args.system.clone().unwrap_or_else(|| DEFAULT_SYSTEM.into());
    let mut turn = 0usize;
    let mut totals = (0usize, 0.0f64); // generated tokens, decode seconds

    // One-shot mode: answer and exit, so the command is scriptable.
    if let Some(p) = &args.prompt {
        let prompt = segment(&system, p, true);
        answer(&mut engine, &tokenizer, &prompt, sampler_of(&cfg), args.max_new, im_end, eos,
               decode_bytes, ansi, &mut turn, &mut totals, args.spec)?;
        return Ok(());
    }

    println!("  Type a message. /help lists the commands, /quit exits.");
    println!();

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut system = system;

    loop {
        print!("{}", if ansi { "\x1b[1m> \x1b[0m" } else { "> " });
        std::io::stdout().flush()?;

        let Some(line) = lines.next() else { break };
        let line = line?;
        let msg = line.trim();

        let (cmd, rest) = match msg.split_once(char::is_whitespace) {
            Some((c, r)) => (c, r.trim()),
            None => (msg, ""),
        };

        match cmd {
            "" => continue,
            "/quit" | "/exit" | "/q" => break,
            "/help" | "/?" => {
                print_help();
                continue;
            }
            "/reset" => {
                engine.reset()?;
                turn = 0;
                println!("  (conversation cleared, KV cache dropped)\n");
                continue;
            }
            "/system" => {
                if rest.is_empty() {
                    println!("  system: {system}\n");
                } else {
                    system = rest.to_string();
                    engine.reset()?;
                    turn = 0;
                    println!("  (system prompt set; the cache held the old one, so it \
                              was dropped)\n");
                }
                continue;
            }
            "/params" => {
                println!("  {}\n  ctx {} of {}, max-new {}\n",
                         describe(&cfg), engine.position(), args.ctx, args.max_new);
                continue;
            }
            "/set" => {
                match set_param(&mut cfg, rest) {
                    Ok(()) => println!("  {}\n", describe(&cfg)),
                    Err(e) => println!("  {e}\n"),
                }
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
        if let Err(e) = answer(&mut engine, &tokenizer, &prompt, sampler_of(&cfg), args.max_new,
                               im_end, eos, decode_bytes, ansi, &mut turn, &mut totals, args.spec) {
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
    spec: usize,
) -> Result<()> {
    let ids = tok.encode(prompt);
    let mut dec = StreamDecoder::default();
    let mut stdout = std::io::stdout();

    let t0 = Instant::now();
    let mut emit = |id: u32| -> bool {
        if id == im_end || Some(id) == eos {
            return false;
        }
        let text = dec.push(tok, id);
        if !text.is_empty() {
            let _ = stdout.write_all(text.as_bytes());
            let _ = stdout.flush();
        }
        true
    };
    let (stats, spec_stats) = if spec > 1 && matches!(sampler, Sampler::Greedy) {
        let cfg = whetstone_core::SpecConfig { draft: spec, ..Default::default() };
        let (s, sp) = whetstone_core::speculate::generate(engine, &ids, max_new, cfg, &mut emit)?;
        (s, Some(sp))
    } else {
        (engine.generate(&ids, max_new, sampler, &mut emit)?, None)
    };
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
    if let Some(sp) = spec_stats {
        print!("{dim}  [spec {:.2} tok/pass, {:.0}% accepted]{off}", sp.tokens_per_round(), sp.acceptance() * 100.0);
    }
    if wall > stats.decode_seconds * 1.25 {
        print!("{dim}  (wall {wall:.2} s - output, not the engine){off}");
    }
    println!();
    println!();
    Ok(())
}

/// The REPL's commands, listed where someone will actually look for them.
fn print_help() {
    println!(
        "  /help                 this list
  /params               current sampling settings and context use
  /set <name> <value>   change one setting for the next turn
  /system [text]        show, or replace, the system prompt (clears the cache)
  /reset                clear the conversation and drop the KV cache
  /stats                throughput across the session
  /quit                 exit

  settable: temperature, top-p, top-k, min-p, repeat-penalty, repeat-last-n, seed
  temperature 0 switches to the greedy path, which is about 20% faster
  because it never copies the logits back to the host.
"
    );
}

/// Applies one `/set name value` pair.
///
/// Values are range-checked here rather than left to the sampler, because a
/// silently-clamped setting in an interactive session is worse than an error:
/// the user sees the output change and attributes it to the wrong knob.
fn set_param(cfg: &mut SamplingConfig, rest: &str) -> std::result::Result<(), String> {
    let mut it = rest.split_whitespace();
    let (Some(name), Some(value)) = (it.next(), it.next()) else {
        return Err("usage: /set <name> <value>   (/help lists the names)".into());
    };

    fn num<T: std::str::FromStr>(v: &str, name: &str) -> std::result::Result<T, String> {
        v.parse::<T>().map_err(|_| format!("{name}: {v:?} is not a number"))
    }

    match name {
        "temperature" | "temp" => {
            let v: f32 = num(value, name)?;
            if !(0.0..=5.0).contains(&v) {
                return Err(format!("temperature {v} is outside 0..=5"));
            }
            cfg.temperature = v;
        }
        "top-p" | "top_p" => {
            let v: f32 = num(value, name)?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("top-p {v} is outside 0..=1"));
            }
            cfg.top_p = v;
        }
        "top-k" | "top_k" => cfg.top_k = num(value, name)?,
        "min-p" | "min_p" => {
            let v: f32 = num(value, name)?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("min-p {v} is outside 0..=1"));
            }
            cfg.min_p = v;
        }
        "repeat-penalty" | "repeat_penalty" => {
            let v: f32 = num(value, name)?;
            if !(0.5..=2.0).contains(&v) {
                return Err(format!("repeat-penalty {v} is outside 0.5..=2"));
            }
            cfg.repeat_penalty = v;
        }
        "repeat-last-n" | "repeat_last_n" => cfg.repeat_last_n = num(value, name)?,
        "seed" => cfg.seed = num(value, name)?,
        _ => return Err(format!("unknown setting {name:?}; /help lists them")),
    }
    Ok(())
}
