//! `whetstone run` — execute a `.wstone` model.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use whetstone_core::{Engine, ModelWeights, Sampler};

/// What to do with the generated tokens.
pub struct RunArgs<'a> {
    /// The `.wstone` file.
    pub model: &'a Path,
    /// Prompt token ids.
    pub ids: Vec<u32>,
    /// Tokens to generate.
    pub max_new: usize,
    /// KV cache capacity.
    pub ctx: usize,
    /// Sampling temperature; zero means greedy.
    pub temperature: f32,
    /// Nucleus mass.
    pub top_p: f32,
    /// PRNG seed.
    pub seed: u64,
    /// Where to write the final-position logits, if anywhere.
    pub dump_logits: Option<&'a Path>,
    /// Emit only the numbers, for scripting.
    pub quiet: bool,
    /// Report a per-stage time breakdown instead of generating.
    pub profile: usize,
    /// Capture the decode step as a CUDA graph before generating.
    pub graph: bool,
    /// Force a specific int4 GEMV variant. `-1` selects the original kernel.
    ///
    /// The point of this flag is in-situ measurement. A microbenchmark reruns
    /// one matrix 200 times, so anything under ~3 MB stays L2-resident and reads
    /// far faster than it ever does in a real decode step, where 262 MB of
    /// weights sweep past exactly once. Selecting the arithmetic-free `mem`
    /// variant here measures the memory path under real cache pressure.
    pub gemv_variant: Option<i32>,
}

pub fn run(args: RunArgs<'_>) -> Result<()> {
    if args.ids.is_empty() {
        bail!("no prompt tokens; pass --ids or --prompt");
    }

    let t0 = Instant::now();
    let weights = ModelWeights::load(args.model)
        .with_context(|| format!("could not load {}", args.model.display()))?;
    let load_s = t0.elapsed().as_secs_f64();

    let cfg_desc = {
        let c = &weights.config;
        format!(
            "{} layers, hidden {}, {}Q/{}KV heads, head_dim {}, vocab {}",
            c.num_hidden_layers,
            c.hidden_size,
            c.num_attention_heads,
            c.n_kv_heads(),
            c.head_dim(),
            c.vocab_size
        )
    };

    let decode_bytes = weights.decode_bytes();
    let bpw = weights.bits_per_weight();
    let params = weights.config.decode_resident_params();

    let ctx = args.ctx.max(args.ids.len() + args.max_new + 1);
    let mut engine = Engine::new(weights, ctx)?;
    let peak = engine.device().bandwidth_gbs();

    if !args.quiet {
        println!("{:=<72}", "");
        println!("  {}", engine.device());
        println!("  {}", args.model.display());
        println!("{:=<72}", "");
        println!("  {cfg_desc}");
        println!(
            "  weights            {:.1} MB/token  ({bpw:.3} bits/weight over {:.1} M params)",
            decode_bytes as f64 / 1e6,
            params as f64 / 1e6
        );
        println!(
            "  kv cache + rope    {:.1} MB at ctx {ctx}",
            engine.state_bytes() as f64 / 1e6
        );
        println!(
            "  roofline           {:.0} tok/s at {peak:.0} GB/s peak",
            peak * 1e9 / decode_bytes as f64
        );
        println!("  load               {load_s:.2} s");
        println!("{:-<72}", "");
    }

    if let Some(v) = args.gemv_variant {
        whetstone_kernels::variant::select(if v < 0 { None } else { Some(v as usize) });
    }
    if !args.quiet {
        let sel = whetstone_kernels::variant::selected();
        println!(
            "  gemv kernel        {}",
            sel.map_or_else(|| "baseline (gemv_int4.cu)".to_string(), whetstone_kernels::variant::name)
        );
    }

    if args.graph {
        let n = engine.capture_graph()?;
        if !args.quiet {
            println!("  cuda graph         captured, {n} launches per token collapsed into 1");
        }
    }

    if args.profile > 0 {
        // Warm up first: the first launch of every kernel pays JIT and module
        // load, which would otherwise land entirely on whichever stage happens
        // to run first and read as that stage being slow.
        engine.prefill(&args.ids)?;
        let p = engine.profile(args.ids[0], args.profile)?;

        println!("  per-stage breakdown over {} steps", p.steps);
        println!("  (synchronised between stages, so the total is inflated --");
        println!("   read the attribution, not the absolute milliseconds)");
        println!("{:-<72}", "");
        for (name, ms, share) in p.breakdown() {
            let bar = "#".repeat((share * 40.0).round() as usize);
            println!("  {name:<14} {ms:>7.3} ms  {:>5.1}%  {bar}", share * 100.0);
        }
        println!("{:-<72}", "");
        println!("  profiled total {:>7.3} ms/token", p.total_ms());
        println!();
        return Ok(());
    }

    let sampler = if args.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::Sample(whetstone_core::SamplingConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            seed: args.seed,
            ..Default::default()
        })
    };

    let mut out_ids: Vec<u32> = Vec::with_capacity(args.max_new);
    let stats = engine.generate(&args.ids, args.max_new, sampler, |t| {
        out_ids.push(t);
        true
    })?;

    if let Some(path) = args.dump_logits {
        let logits = engine.logits()?;
        dump(path, &args.ids, &out_ids, &logits)?;
        if !args.quiet {
            println!("  wrote {} ({} logits)", path.display(), logits.len());
        }
    }

    if args.quiet {
        println!(
            "{:.3} {:.3} {:?}",
            stats.decode_tok_s(),
            stats.prefill_tok_s(),
            out_ids
        );
        return Ok(());
    }

    println!("  ids  {out_ids:?}");
    println!("{:-<72}", "");
    println!(
        "  prefill            {:.1} tok/s   ({} tokens in {:.3} s)",
        stats.prefill_tok_s(),
        stats.prompt_tokens,
        stats.prefill_seconds
    );
    println!(
        "  decode             {:.1} tok/s   ({} tokens in {:.3} s)",
        stats.decode_tok_s(),
        stats.generated,
        stats.decode_seconds
    );
    if let Some((p10, p50, p90)) = stats.latency_percentiles() {
        println!("  latency            p50 {p50:.2} ms   (p10 {p10:.2}, p90 {p90:.2})");
        // A spread this wide means the measurement is dominated by whatever else
        // is on the GPU, not by the engine. Saying so is cheaper than a wrong
        // number being quoted later.
        if p90 > p50 * 1.5 {
            println!("  NOTE: p90/p50 = {:.1}x -- the machine is contended; re-measure idle.",
                     p90 / p50);
        }
    }

    let achieved = decode_bytes as f64 * stats.decode_tok_s() / 1e9;
    println!(
        "  bandwidth          {achieved:.0} GB/s of {peak:.0} peak   ({:.0}% roofline attainment)",
        achieved / peak * 100.0
    );
    println!();

    Ok(())
}

fn dump(path: &Path, prompt: &[u32], generated: &[u32], logits: &[f32]) -> Result<()> {
    // Top-32 plus summary statistics, not 151936 floats: the quality gates
    // compare distributions, and a 3 MB JSON per prompt is unusable in a diff.
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        logits[b as usize].partial_cmp(&logits[a as usize]).unwrap()
    });

    let max = logits[order[0] as usize] as f64;
    let sum: f64 = logits.iter().map(|&l| ((l as f64) - max).exp()).sum();
    let probs: Vec<f64> =
        order.iter().take(32).map(|&i| ((logits[i as usize] as f64) - max).exp() / sum).collect();
    let entropy: f64 = -logits
        .iter()
        .map(|&l| {
            let p = ((l as f64) - max).exp() / sum;
            if p > 0.0 {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f64>();

    let mean = logits.iter().map(|&l| l as f64).sum::<f64>() / logits.len() as f64;
    let var = logits.iter().map(|&l| (l as f64 - mean).powi(2)).sum::<f64>() / logits.len() as f64;

    let json = serde_json::json!({
        "impl": "whetstone",
        "prompt_ids": prompt,
        "generated_ids": generated,
        "top1_id": order[0],
        "topk_ids": &order[..32],
        "topk_probs": probs,
        "logits_mean": mean,
        "logits_std": var.sqrt(),
        "logits_max": max,
        "entropy": entropy,
    });
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    std::fs::write(path, serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}
