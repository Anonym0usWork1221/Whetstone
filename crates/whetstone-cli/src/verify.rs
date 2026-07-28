//! `whetstone verify` — is this `.wstone` file intact, and how damaged are its
//! weights relative to the source?
//!
//! Two independent questions, and the command answers both:
//!
//! 1. **Integrity** — does every blob match its checksum? Cheap to check and it
//!    turns a corrupt download into an error instead of a bad generation.
//! 2. **Fidelity** — how far did quantization move each tensor from the
//!    original? Requires the source checkpoint, so it is optional.

use std::path::Path;

use anyhow::{bail, Context, Result};
use whetstone_core::SafeTensors;
use whetstone_quant::format::{self, TensorKind};

pub fn run(wstone: &Path, source: Option<&Path>, bandwidth: Option<f64>) -> Result<()> {
    let bytes = std::fs::read(wstone)
        .with_context(|| format!("could not read {}", wstone.display()))?;

    let header = format::read_header(&bytes, bytes.len() as u64)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("{:=<72}", "");
    println!("  {}", wstone.display());
    println!("{:=<72}", "");
    println!("  format             {} v{}", header.format, header.version);
    println!("  producer           {}", header.producer);
    for (k, v) in &header.quant {
        println!("  {:<18} {}", k, v);
    }
    println!("  tensors            {}", header.tensors.len());
    println!("  file size          {:.1} MB", bytes.len() as f64 / 1e6);

    let mut by_kind = std::collections::BTreeMap::new();
    for t in &header.tensors {
        let e = by_kind.entry(format!("{:?}", t.kind)).or_insert((0usize, 0u64, 0usize));
        e.0 += 1;
        e.1 += t.stored_bytes();
        e.2 += t.numel();
    }
    println!();
    println!("  {:<12} {:>7} {:>12} {:>14} {:>10}", "kind", "count", "params", "bytes", "bits/wt");
    for (kind, (n, b, p)) in &by_kind {
        println!(
            "  {:<12} {:>7} {:>11.1} M {:>12.1} MB {:>10.3}",
            kind,
            n,
            *p as f64 / 1e6,
            *b as f64 / 1e6,
            *b as f64 * 8.0 / *p as f64
        );
    }

    let resident = header.decode_resident_bytes();
    let total_params: usize = header.tensors.iter().map(|t| t.numel()).sum();
    let bw = bandwidth.unwrap_or(278.0);
    println!();
    println!(
        "  read per token     {:.1} MB  ->  {:.0} tok/s ceiling at {bw:.0} GB/s",
        resident as f64 / 1e6,
        bw * 1e9 / resident as f64
    );
    println!("  total params       {:.1} M", total_params as f64 / 1e6);

    // --- integrity ---------------------------------------------------------
    println!();
    print!("  checking {} blobs ... ", header.tensors.iter().map(|t| t.blobs.len()).sum::<usize>());
    match format::verify_payloads(&header, &bytes) {
        Ok(()) => println!("all checksums OK"),
        Err(e) => {
            println!("FAILED");
            bail!("{e}");
        }
    }

    // --- fidelity ----------------------------------------------------------
    let Some(src_dir) = source else {
        println!();
        println!("  Pass --source <model_dir> to also measure quantization error");
        println!("  against the original weights.");
        println!();
        return Ok(());
    };

    let st = SafeTensors::open(src_dir.join("model.safetensors"))
        .with_context(|| format!("could not open source checkpoint in {}", src_dir.display()))?;

    println!();
    println!("  comparing against {} ...", src_dir.display());

    let mut errors: Vec<(f64, String)> = Vec::new();
    let mut checked = 0usize;

    for t in &header.tensors {
        // Filter to the 2-D quantized kinds FIRST. `model.norm.weight` and the
        // q/k/v biases are rank-1, so reading `shape[1]` before this check is an
        // out-of-bounds panic on a perfectly valid file.
        if !matches!(t.kind, TensorKind::Int4G128 | TensorKind::Int4HierG32) {
            continue;
        }
        if t.shape.len() != 2 {
            return Err(anyhow::anyhow!(
                "{}: quantized tensors must be rank 2, found shape {:?}",
                t.name,
                t.shape
            ));
        }
        let Ok(src) = st.to_f32(&t.name) else { continue };
        let (in_features, out_features) = (t.shape[1], t.shape[0]);

        // Dequantizing through the real reader is the point: this measures the
        // file as the engine will read it, so it catches a packer and a loader
        // that disagree as well as a quantizer that is simply lossy.
        let deq = match t.kind {
            TensorKind::Int4G128 => {
                let qw_b = t.blob("qw").map_err(|e| anyhow::anyhow!("{e}"))?;
                let sz_b = t.blob("sz").map_err(|e| anyhow::anyhow!("{e}"))?;
                let packed = whetstone_quant::PackedInt4 {
                    qw: read_u32(&bytes, qw_b.offset, qw_b.len),
                    sz: read_u32(&bytes, sz_b.offset, sz_b.len),
                    in_features,
                    out_features,
                };
                whetstone_quant::dequantize_int4_g128(&packed)
            }
            TensorKind::Int4HierG32 => {
                let qw_b = t.blob("qw").map_err(|e| anyhow::anyhow!("{e}"))?;
                let si_b = t.blob("si").map_err(|e| anyhow::anyhow!("{e}"))?;
                let sb_b = t.blob("sb").map_err(|e| anyhow::anyhow!("{e}"))?;
                let lo = si_b.offset as usize;
                let packed = whetstone_quant::PackedInt4Hier {
                    qw: read_u32(&bytes, qw_b.offset, qw_b.len),
                    si: bytes[lo..lo + si_b.len as usize].to_vec(),
                    sb: read_u32(&bytes, sb_b.offset, sb_b.len),
                    in_features,
                    out_features,
                };
                whetstone_quant::dequantize_int4_hier(&packed)
            }
            _ => continue,
        };
        errors.push((whetstone_quant::relative_error(&src, &deq), t.name.clone()));
        checked += 1;
    }

    if errors.is_empty() {
        println!("  no quantized tensors to compare");
        return Ok(());
    }

    errors.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mean = errors.iter().map(|e| e.0).sum::<f64>() / errors.len() as f64;

    println!();
    println!("  quantization error over {checked} tensors");
    println!("    mean             {mean:.4}");
    println!("    median           {:.4}", errors[errors.len() / 2].0);
    println!("    worst 5:");
    for (e, name) in errors.iter().take(5) {
        println!("      {e:.4}  {name}");
    }

    // Calibrated against the measured curve: int4-g128 lands near 0.11 on real
    // weights and int3 near 0.23, so drifting past ~0.15 means something is
    // wrong with the packing, not just with the format.
    println!();
    if mean < 0.15 {
        println!("  Mean error is consistent with int4-g128 on real weights (~0.11).");
    } else {
        println!("  WARNING: mean error {mean:.4} is higher than int4-g128 should produce");
        println!("  (~0.11 measured). Suspect a packing or layout bug rather than the format.");
    }
    println!();
    println!("  Weight error is not the objective. Run the quality gate");
    println!("  (top-1 agreement and wikitext-2 perplexity) before trusting this file.");
    println!();

    Ok(())
}

fn read_u32(bytes: &[u8], offset: u64, len: u64) -> Vec<u32> {
    bytes[offset as usize..(offset + len) as usize]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
