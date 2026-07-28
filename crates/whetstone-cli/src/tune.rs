//! `whetstone tune` — pick the per-shape GEMV rule by generating tokens.
//!
//! # Why this is not a microbenchmark
//!
//! Two cheaper measurements were tried first and both misranked the kernels.
//!
//! A **microbenchmark** reruns one matrix a few hundred times, so anything under
//! this card's 3 MB L2 stays resident and reads far faster than it ever does in
//! a decode step, where 262 MB sweeps past exactly once. It ranked `gate|up`
//! best at TILE=8 and `lm_head` best at TILE=4. In the engine it is the other
//! way round for both.
//!
//! An **in-situ CUDA-event profile** fixed the cache problem but introduced its
//! own: recording an event at every stage boundary serialises the boundaries,
//! and the rule it selected measured *slower* end to end than the rule it
//! replaced. Its per-stage numbers are still the right tool for finding which
//! stage to work on — they are not the right tool for choosing between two
//! kernels that differ by a few percent.
//!
//! So this sweeps candidate rules by running an actual generation and taking the
//! best of several samples. It is slow and it is the only version that has not
//! been wrong yet.

use anyhow::{Context, Result};
use std::path::Path;
use whetstone_core::{Engine, ModelWeights, Sampler};
use whetstone_kernels::gemv::variant;

/// Candidate assignments for (wide reduction, huge output, everything else).
///
/// Only the tile counts that ever won a shape are candidates; sweeping all
/// eleven variants in all three slots is 1331 generations and the extra ones
/// lost by margins no sampling here could resolve.
const CANDIDATES: &[usize] = &[3, 4, 9]; // h2 t2, h2 t4, h2 t8

pub fn run(model_path: &Path, ids: &[u32], tokens: usize, samples: usize) -> Result<()> {
    let weights = ModelWeights::load(model_path)
        .with_context(|| format!("could not load {}", model_path.display()))?;
    let ctx = ids.len() + tokens + 1;
    let mut engine = Engine::new(weights, ctx.max(512))?;
    engine.capture_graph()?;

    println!("{:=<72}", "");
    println!("  tuning the per-shape GEMV rule on {}", model_path.display());
    println!("{:=<72}", "");
    println!("  {tokens} tokens per sample, best of {samples}");
    println!("  measuring whole-generation throughput -- a microbenchmark and an");
    println!("  event profile both misranked these kernels; see tune.rs");
    println!("{:-<72}", "");

    // Warm the clocks. The first generation after a load runs at a lower boost
    // state and would make whichever rule is tried first look worse.
    let _ = measure(&mut engine, ids, tokens)?;

    let mut best: Option<([usize; 3], f64)> = None;
    let mut rows: Vec<([usize; 3], f64)> = Vec::new();

    for &wide in CANDIDATES {
        for &huge in CANDIDATES {
            for &other in CANDIDATES {
                let rule = [wide, huge, other];
                variant::set_shape_rule(rule);

                let mut tok_s = 0.0f64;
                for _ in 0..samples {
                    tok_s = tok_s.max(measure(&mut engine, ids, tokens)?);
                }

                println!(
                    "  down={:<12} head={:<12} rest={:<12} {tok_s:>8.1} tok/s",
                    variant::name(wide),
                    variant::name(huge),
                    variant::name(other)
                );
                rows.push((rule, tok_s));
                if best.map_or(true, |(_, b)| tok_s > b) {
                    best = Some((rule, tok_s));
                }
            }
        }
    }

    println!("{:-<72}", "");
    if let Some((rule, tok_s)) = best {
        variant::set_shape_rule(rule);
        let worst = rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
        println!(
            "  best   down={} head={} rest={}   {tok_s:.1} tok/s",
            variant::name(rule[0]),
            variant::name(rule[1]),
            variant::name(rule[2])
        );
        println!("  spread {:.1}x between best and worst rule", tok_s / worst);
        println!();
        println!("  To make this the built-in default, edit kRule in");
        println!("  crates/whetstone-kernels/cuda/gemv_variants.cu to [{}, {}, {}]",
                 rule[0], rule[1], rule[2]);
    }
    println!();

    Ok(())
}

/// Tokens per second for one generation, from a clean KV cache.
fn measure(engine: &mut Engine, ids: &[u32], tokens: usize) -> Result<f64> {
    engine.reset()?;
    let stats = engine.generate(ids, tokens, Sampler::Greedy, |_| true)?;
    Ok(stats.decode_tok_s())
}
