//! `whetstone bench` — sweep GEMV kernel variants across the model's shapes.
//!
//! One kernel does not win everywhere. `k_proj` is 896x128 — 57 KB, too small to
//! fill 30 SMs at any blocking — while `lm_head` is 896x151936 and streams 68 MB
//! in one call. The blocking that amortises the activation read on the second
//! starves the first of parallelism. So the sweep is per shape, and the totals
//! are weighted by how often each shape actually runs in a decode step.
//!
//! The `mem` row is not a candidate. It loads the weights and skips the
//! arithmetic entirely, which makes it a lower bound on what the memory path
//! alone costs — the gap between it and the best real kernel is the arithmetic,
//! and therefore the size of the prize left.

use anyhow::Result;
use whetstone_kernels::{gemv, Device};

/// The shapes a Qwen2-family decode step actually issues, and how many times.
struct Shape {
    name: &'static str,
    in_f: usize,
    out_f: usize,
    /// Invocations per token, given the layer count.
    per_token: usize,
}

fn shapes(layers: usize, hidden: usize, kv_dim: usize, inter: usize, vocab: usize) -> Vec<Shape> {
    vec![
        Shape { name: "q/o proj", in_f: hidden, out_f: hidden, per_token: 2 * layers },
        Shape { name: "k/v proj", in_f: hidden, out_f: kv_dim, per_token: 2 * layers },
        Shape { name: "gate/up", in_f: hidden, out_f: inter, per_token: 2 * layers },
        Shape { name: "down", in_f: inter, out_f: hidden, per_token: layers },
        Shape { name: "lm_head", in_f: hidden, out_f: vocab, per_token: 1 },
    ]
}

pub fn run(reps: i32, repeats: usize) -> Result<()> {
    let device = Device::default_device()?;
    let peak = device.bandwidth_gbs();
    let n = gemv::variant::count();

    println!("{:=<98}", "");
    println!("  {device}");
    println!("  int4-g128 GEMV variant sweep, {n} variants, {reps} reps x {repeats} repeats");
    println!("{:=<98}", "");

    // Qwen2.5-0.5B-Instruct.
    let shapes = shapes(24, 896, 128, 4864, 151_936);

    // Best of N. Other work on the GPU inflates any single sample, and the
    // minimum is the least-contended observation -- the honest estimate of what
    // the kernel costs rather than of what the machine was doing.
    let best = |v: usize, s: &Shape| -> Result<gemv::GemvBench> {
        let mut b = gemv::variant::bench(v, s.in_f, s.out_f, reps)?;
        for _ in 1..repeats {
            let c = gemv::variant::bench(v, s.in_f, s.out_f, reps)?;
            if c.ms < b.ms {
                b = c;
            }
        }
        Ok(b)
    };

    print!("  {:<14}", "shape");
    for s in &shapes {
        print!("{:>15}", s.name);
    }
    println!("{:>17}", "token total");
    print!("  {:<14}", "");
    for s in &shapes {
        print!("{:>15}", format!("{}x{}", s.in_f, s.out_f));
    }
    println!();
    println!("{:-<98}", "");

    let mut totals: Vec<(usize, f64)> = Vec::new();

    for v in 0..n {
        let label = gemv::variant::name(v);
        print!("  {label:<14}");
        let mut token_ms = 0.0;
        for s in &shapes {
            let b = best(v, s)?;
            token_ms += b.ms * s.per_token as f64;
            print!("{:>15}", format!("{:.0} GB/s", b.gbs));
        }
        println!("{:>17}", format!("{token_ms:.3} ms"));
        totals.push((v, token_ms));
    }

    println!("{:-<98}", "");

    // The `mem` probe is a floor, not a candidate: it computes the wrong answer.
    let mut ranked: Vec<&(usize, f64)> = totals
        .iter()
        .filter(|(v, _)| !gemv::variant::name(*v).starts_with("mem"))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let floor = totals
        .iter()
        .find(|(v, _)| gemv::variant::name(*v).starts_with("mem"))
        .map(|(_, ms)| *ms);

    if let Some((v, ms)) = ranked.first() {
        println!(
            "  best            {} at {ms:.3} ms of GEMV per token",
            gemv::variant::name(*v)
        );
        let baseline = totals.iter().find(|(i, _)| *i == 0).map(|(_, m)| *m).unwrap_or(*ms);
        println!("  vs variant 0    {:.2}x", baseline / ms);
        if let Some(f) = floor {
            println!(
                "  memory floor    {f:.3} ms  -- arithmetic still costs {:.3} ms/token ({:.0}% of GEMV time)",
                ms - f,
                (ms - f) / ms * 100.0
            );
        }
        // 262.4 MB is what int4-g128 streams per token for this model.
        println!(
            "  implied         {:.0} tok/s from GEMV alone, {:.0}% of the {peak:.0} GB/s roofline",
            1e3 / ms,
            262.4e6 / (ms * 1e-3) / 1e9 / peak * 100.0
        );
    }
    println!();

    Ok(())
}
