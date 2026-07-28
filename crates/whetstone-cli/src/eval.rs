//! `whetstone eval` — the quality gates.
//!
//! Two things, both of which have to be reported every time a format or a kernel
//! changes:
//!
//! - **perplexity** on a fixed token stream, against a fixed windowing, so it is
//!   comparable to the HuggingFace fp16 number taken the same way;
//! - **logit fidelity**, dumped as raw f32 so the Python side can compute top-1
//!   agreement and KL against the reference without this binary needing to know
//!   what a KL divergence is.
//!
//! Speed without one of these is not a result. A quantizer that is 2x faster and
//! 0.3 perplexity worse is a trade to be argued about; a quantizer that is 2x
//! faster and silently 3 perplexity worse is a bug that ships.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use whetstone_core::{Engine, ModelWeights};

/// Reads a file of little-endian `u32` token ids.
fn read_tokens(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{}: length {} is not a whole number of u32 tokens", path.display(), bytes.len());
    }
    Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Perplexity over a token stream.
pub fn perplexity(
    model_path: &Path,
    tokens_path: &Path,
    window: usize,
    windows: usize,
    out: Option<&Path>,
) -> Result<()> {
    let tokens = read_tokens(tokens_path)?;
    let weights = ModelWeights::load(model_path)
        .with_context(|| format!("could not load {}", model_path.display()))?;

    let vocab = weights.config.vocab_size;
    if let Some(bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
        bail!("token id {bad} is outside the model's {vocab}-entry vocabulary");
    }

    let bpw = weights.bits_per_weight();
    let scheme = weights.quant_meta.get("scheme").cloned().unwrap_or_default();
    let head = weights.quant_meta.get("lm_head").cloned().unwrap_or_default();

    let mut engine = Engine::new(weights, window)?;

    let n = (tokens.len() / window).min(windows);
    println!("{:=<72}", "");
    println!("  perplexity  {}", model_path.display());
    println!("{:=<72}", "");
    println!("  tokens             {} from {}", tokens.len(), tokens_path.display());
    println!("  windows            {n} x {window}   ({} predictions)", n * (window - 1));
    println!("  format             {scheme}, lm_head {head}, {bpw:.3} bits/weight");
    println!("{:-<72}", "");

    let t0 = Instant::now();
    let (nll, count) = engine.cross_entropy(&tokens, window, windows, |i, nll, c| {
        let ppl = (nll / c as f64).exp();
        println!(
            "  window {:>3}/{n}   running ppl {ppl:.4}   ({c} positions)",
            i + 1
        );
        let _ = std::io::stdout().flush();
    })?;
    let secs = t0.elapsed().as_secs_f64();

    if count == 0 {
        bail!("no windows evaluated: {} tokens is fewer than one {window}-token window",
              tokens.len());
    }

    let mean = nll / count as f64;
    let ppl = mean.exp();

    println!("{:-<72}", "");
    println!("  nll                {mean:.4}");
    println!("  perplexity         {ppl:.4}");
    println!("  positions          {count}");
    println!("  elapsed            {secs:.1} s   ({:.0} positions/s)", count as f64 / secs);
    println!();

    if let Some(p) = out {
        let json = serde_json::json!({
            "impl": "whetstone",
            "model": model_path.display().to_string(),
            "tokens_file": tokens_path.display().to_string(),
            "scheme": scheme,
            "lm_head": head,
            "bits_per_weight": bpw,
            "window": window,
            "windows": n,
            "positions": count,
            "nll": mean,
            "ppl": ppl,
            "seconds": secs,
        });
        std::fs::write(p, serde_json::to_vec_pretty(&json)?)?;
        println!("  wrote {}", p.display());
    }

    Ok(())
}

/// Dumps the final-position logits for each prompt in a JSON array of id arrays.
///
/// Raw little-endian f32, `prompts * vocab` of them, so the comparison script
/// can `np.fromfile` it. JSON would be 20x the size and lose the last bits.
pub fn logits(model_path: &Path, prompts_path: &Path, out: &Path, ctx: usize) -> Result<()> {
    let text = std::fs::read_to_string(prompts_path)
        .with_context(|| format!("could not read {}", prompts_path.display()))?;
    let prompts: Vec<Vec<u32>> = serde_json::from_str(&text)
        .with_context(|| format!("{}: expected a JSON array of id arrays", prompts_path.display()))?;
    if prompts.is_empty() {
        bail!("{}: no prompts", prompts_path.display());
    }

    let weights = ModelWeights::load(model_path)
        .with_context(|| format!("could not load {}", model_path.display()))?;
    let vocab = weights.config.vocab_size;
    let mut engine = Engine::new(weights, ctx)?;

    let file = std::fs::File::create(out)
        .with_context(|| format!("could not create {}", out.display()))?;
    let mut w = BufWriter::new(file);

    for (i, ids) in prompts.iter().enumerate() {
        if ids.is_empty() {
            bail!("prompt {i} is empty");
        }
        engine.reset()?;
        engine.prefill(ids)?;
        let l = engine.logits()?;
        debug_assert_eq!(l.len(), vocab);
        for v in &l {
            w.write_all(&v.to_le_bytes())?;
        }
        println!("  prompt {:>3}/{}  {} tokens", i + 1, prompts.len(), ids.len());
    }
    w.flush()?;

    println!("  wrote {} ({} x {} f32)", out.display(), prompts.len(), vocab);
    Ok(())
}
