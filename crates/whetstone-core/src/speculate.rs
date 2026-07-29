//! Speculative decoding against a draft that costs nothing to run.
//!
//! # Why this is worth more here than the literature suggests
//!
//! Speculative decoding trades weight *reads* for arithmetic: propose `k` tokens
//! cheaply, then verify all `k` in one pass over the weights. Its ceiling is
//! therefore set by what a `k`-token pass costs relative to a single-token one.
//! Measured on this machine (Qwen2.5-0.5B, int4-g32-hier, 264 MB/token):
//!
//! | k | cost of a k-token pass | ceiling |
//! |---|---|---|
//! | 4 | 1.68x | 2.38x |
//! | 8 | 2.85x | 2.81x |
//! | 16 | 4.93x | 3.25x |
//!
//! Useful, not spectacular — the chunk pass is compute bound once the weights
//! are already in VRAM, so verifying eight tokens really does cost most of eight
//! forward passes' arithmetic.
//!
//! **Offloaded, the same table changes completely.** With 20 of 24 blocks in
//! host RAM the pass is bound by a 6 GB/s PCIe read instead, and that read is
//! shared by every token in the chunk. Measured at a 100 MB VRAM budget:
//! single-token decode 30.5 tok/s, sixteen-token passes 454 tok/s — a
//! **sixteen-token pass costs 1.07 single-token passes.** The ceiling is no
//! longer 3.25x, it is ~15x.
//!
//! That is the whole reason this module exists in the same session as offload:
//! the two compose, and the composition is what makes a model that does not fit
//! in VRAM usable rather than merely runnable.
//!
//! # The draft
//!
//! A second model is the textbook draft, and on this hardware it does not pay:
//! the smallest converted model runs at 27% of the 3B's cost per token, so
//! `k = 4` spends 1.08 target-passes on drafting alone before verification. What
//! does pay is an **n-gram draft** — look for the most recent occurrence of the
//! last few tokens in the context and propose whatever followed it. It costs a
//! backwards `memcmp`, needs no second model and no training, and on text that
//! quotes its own context (chat with retrieved documents, code editing,
//! summarisation, structured output) it is accepted often.
//!
//! On open-ended prose it is accepted rarely — and rarely accepted costs almost
//! nothing, because a round with no match falls back to an ordinary decode step.
//! The floor is 1.0x minus a memcmp; that asymmetry is why this draft is the
//! right one for a general-purpose engine.
//!
//! # Exactness
//!
//! Greedy verification accepts a drafted token only when it equals the target's
//! own argmax at that position, so **the output is what greedy decoding would
//! have produced**, token for token. There is no accuracy budget being spent
//! here and no quality gate to clear — only the arithmetic of the chunk pass
//! differs from the single-token path, and `tests/chunk_kernels.rs` pins that
//! against the single-token kernels.

use crate::error::Result;
use crate::engine::{Engine, RunStats};

/// How a speculative run is configured.
#[derive(Debug, Clone, Copy)]
pub struct SpecConfig {
    /// Tokens fed to one verification pass, including the known-good first one.
    /// The draft supplies `draft - 1` of them.
    pub draft: usize,
    /// Longest n-gram to match against the history. Longer matches are tried
    /// first because they are far likelier to be accepted.
    pub ngram_max: usize,
    /// Shortest n-gram to accept as a match. Below about 2 the proposals are
    /// noise and every round wastes its verification.
    pub ngram_min: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        // draft 4 is the measured knee of the cost curve when resident (1.68x
        // for four tokens); offloaded, wider is strictly better, which is why
        // this is a knob and not a constant.
        Self { draft: 4, ngram_max: 3, ngram_min: 2 }
    }
}

/// What a speculative run actually did.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecStats {
    /// Verification passes run.
    pub rounds: usize,
    /// Rounds where the n-gram draft found nothing and a plain step ran.
    pub empty_rounds: usize,
    /// Draft tokens proposed across all rounds.
    pub proposed: usize,
    /// Draft tokens the target confirmed.
    pub accepted: usize,
}

impl SpecStats {
    /// Fraction of proposed tokens the target confirmed.
    pub fn acceptance(&self) -> f64 {
        if self.proposed == 0 {
            0.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }

    /// Mean tokens emitted per verification pass. This is the speedup numerator:
    /// the round produced this many tokens for one pass over the weights.
    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            (self.rounds + self.accepted) as f64 / self.rounds as f64
        }
    }
}

/// Proposes up to `want` tokens by finding the most recent earlier occurrence of
/// the history's tail.
///
/// Longest n-gram first: a 3-token match is a much stronger signal than a
/// 2-token one, and proposing from a weak match burns the round's whole draft
/// budget on tokens that will be rejected at position one.
fn propose(history: &[u32], cfg: &SpecConfig, want: usize) -> Vec<u32> {
    if want == 0 {
        return Vec::new();
    }
    let n_max = cfg.ngram_max.min(history.len().saturating_sub(1));
    for n in (cfg.ngram_min..=n_max).rev() {
        if history.len() <= n {
            continue;
        }
        let tail = &history[history.len() - n..];
        // Search backwards so the *most recent* occurrence wins: in a chat
        // transcript the relevant continuation is nearly always the latest one.
        for start in (0..history.len() - n).rev() {
            if &history[start..start + n] != tail {
                continue;
            }
            let from = start + n;
            let take = want.min(history.len() - from);
            if take > 0 {
                return history[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

/// Greedy generation with n-gram speculation.
///
/// Emits exactly the token sequence greedy decoding would emit. `on_token`
/// returning `false` stops generation, as with [`Engine::generate`].
pub fn generate(
    engine: &mut Engine,
    prompt: &[u32],
    max_new: usize,
    cfg: SpecConfig,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<(RunStats, SpecStats)> {
    if prompt.is_empty() {
        return Err(crate::Error::Shape("cannot generate from an empty prompt".into()));
    }
    if !engine.supports_chunk() {
        return Err(crate::Error::Unsupported(
            "speculative decoding needs the multi-token kernel; reconvert with the default \
             hierarchical int4 format"
                .into(),
        ));
    }
    let width = engine.chunk_width().min(cfg.draft.max(1));

    let mut stats = RunStats { prompt_tokens: prompt.len(), ..Default::default() };
    let mut spec = SpecStats::default();

    engine.device().synchronize()?;
    let t0 = std::time::Instant::now();
    engine.prefill(prompt)?;
    engine.device().synchronize()?;
    stats.prefill_seconds = t0.elapsed().as_secs_f64();

    let mut history: Vec<u32> = prompt.to_vec();
    // Prefill left the argmax in the device cursor; that is the first token.
    let mut next = engine.greedy_pick()?;

    let t1 = std::time::Instant::now();
    'outer: while stats.generated < max_new {
        let tok_start = std::time::Instant::now();

        // `next` is confirmed: it is the target's own argmax at this position.
        if !on_token(next) {
            break;
        }
        stats.generated += 1;
        history.push(next);
        if stats.generated >= max_new || engine.position() >= engine.max_seq() {
            break;
        }

        let room = engine.max_seq() - engine.position();
        let drafts = propose(&history, &cfg, (width - 1).min(room.saturating_sub(1)));

        if drafts.is_empty() {
            // Nothing to speculate on. One ordinary step, which is the floor
            // this scheme degrades to rather than a penalty it pays.
            spec.rounds += 1;
            spec.empty_rounds += 1;
            engine.forward(next)?;
            next = engine.greedy_pick()?;
            stats.token_ms.push(tok_start.elapsed().as_secs_f64() * 1e3);
            continue;
        }

        let mut inputs = Vec::with_capacity(1 + drafts.len());
        inputs.push(next);
        inputs.extend_from_slice(&drafts);

        let pos0 = engine.position();
        engine.forward_chunk(&inputs)?;
        let picks = engine.chunk_picks(inputs.len())?;

        spec.rounds += 1;
        spec.proposed += drafts.len();

        // `picks[j]` is the target's greedy choice after consuming `inputs[..=j]`,
        // so it is the prediction that `inputs[j+1]` was guessing at. The first
        // disagreement ends the run; everything before it is exactly what greedy
        // decoding would have produced.
        let mut m = inputs.len() - 1;
        for j in 0..inputs.len() - 1 {
            if picks[j] != inputs[j + 1] {
                m = j;
                break;
            }
        }
        spec.accepted += m;

        // The cache must hold inputs[0..=m] and nothing after it. Rewinding is
        // free: attention reads only below the cursor, so the rejected tail is
        // simply overwritten by the next write.
        engine.rewind(pos0 + m + 1)?;
        engine.unnote(inputs.len() - (m + 1));

        for &t in &inputs[1..=m] {
            if !on_token(t) {
                break 'outer;
            }
            stats.generated += 1;
            history.push(t);
            if stats.generated >= max_new {
                break 'outer;
            }
        }

        next = picks[m];
        engine.set_pending(next)?;
        stats.token_ms.push(tok_start.elapsed().as_secs_f64() * 1e3);
    }
    engine.device().synchronize()?;
    stats.decode_seconds = t1.elapsed().as_secs_f64();

    Ok((stats, spec))
}
