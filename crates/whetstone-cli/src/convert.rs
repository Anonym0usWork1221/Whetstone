//! `whetstone convert` — HuggingFace checkpoint to `.wstone`.

use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use whetstone_core::{ModelConfig, SafeTensors};
use whetstone_quant::{format, quantize_int4_g128, relative_error, PackedInt4};

/// Precision for the transformer-block projections.
///
/// `Fp16` exists to keep a lossless reference path alive at all times. Without
/// one there is no way to separate "the engine is wrong" from "the quantizer is
/// lossy", and those two failures look identical from a perplexity number.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BodyPrecision {
    /// int4 group-128. The production format.
    Int4,
    /// Dense fp16. 988 MB for Qwen2.5-0.5B, and the differential-testing baseline.
    Fp16,
}

/// Precision for the output projection.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum HeadPrecision {
    /// Keep `lm_head` in fp16. Safe default.
    Fp16,
    /// Quantize `lm_head` to int4-g128. Largest single bandwidth win available,
    /// and the riskiest — it sets the output distribution directly.
    Int4,
}

/// Suffixes of the projections a decode step streams.
const LINEAR_SUFFIXES: &[&str] = &[
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

const EMBED: &str = "model.embed_tokens.weight";

pub fn run(
    model_dir: &Path,
    out_path: &Path,
    head: HeadPrecision,
    body: BodyPrecision,
    bandwidth: Option<f64>,
) -> Result<()> {
    let cfg = ModelConfig::from_dir(model_dir)
        .with_context(|| format!("could not load config from {}", model_dir.display()))?;

    let weights = model_dir.join("model.safetensors");
    if !weights.exists() {
        bail!("no model.safetensors in {}", model_dir.display());
    }
    let st = SafeTensors::open(&weights)?;

    let raw_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;

    println!("{:=<72}", "");
    println!("  converting {}", model_dir.display());
    println!("  to         {}", out_path.display());
    println!("{:=<72}", "");

    let file = std::fs::File::create(out_path)
        .with_context(|| format!("could not create {}", out_path.display()))?;

    // The directory holds ~300 entries with two blobs each; 1 MiB is generous
    // and costs nothing, and the writer errors rather than truncating if it is
    // ever too small.
    let mut w = format::Writer::new(BufWriter::new(file), raw_config, 1 << 20)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    w.set_quant_meta(
        "scheme",
        match body {
            BodyPrecision::Int4 => "int4-g128-asymmetric",
            BodyPrecision::Fp16 => "fp16",
        },
    );
    w.set_quant_meta("group", "128");
    w.set_quant_meta("method", "rtn");
    w.set_quant_meta("source", &weights.display().to_string());
    w.set_quant_meta(
        "lm_head",
        match head {
            HeadPrecision::Fp16 => "fp16",
            HeadPrecision::Int4 => "int4-g128",
        },
    );

    let t0 = Instant::now();
    let mut n_quant = 0usize;
    let mut n_dense = 0usize;
    let mut worst: (f64, String) = (0.0, String::new());
    let mut err_sum = 0.0f64;

    // --- transformer projections ------------------------------------------
    for layer in 0..cfg.num_hidden_layers {
        for suffix in LINEAR_SUFFIXES {
            let name = format!("model.layers.{layer}.{suffix}");
            let t = st.get(&name)?;
            let (out_f, in_f) = t.shape_2d()?;

            let w32 = st.to_f32(&name)?;

            if body == BodyPrecision::Fp16 {
                write_fp16(&mut w, &name, &w32, &[out_f, in_f])?;
                n_dense += 1;
                continue;
            }

            if in_f % whetstone_quant::GROUP != 0 {
                // Not representable in this format; keep it dense rather than
                // silently padding, which would change the maths.
                println!("  {name}: in_features {in_f} not a multiple of 128, keeping fp16");
                write_fp16(&mut w, &name, &w32, &[out_f, in_f])?;
                n_dense += 1;
                continue;
            }

            let packed = quantize_int4_g128(&w32, in_f, out_f).map_err(|e| anyhow::anyhow!("{e}"))?;
            let e = report_error(&w32, &packed);
            err_sum += e;
            if e > worst.0 {
                worst = (e, name.clone());
            }

            w.write_int4(&name, &packed).map_err(|e| anyhow::anyhow!("{e}"))?;
            n_quant += 1;
        }

        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let name = format!("model.layers.{layer}.{suffix}");
            copy_dense(&st, &mut w, &name)?;
            n_dense += 1;
        }
        for suffix in ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"] {
            let name = format!("model.layers.{layer}.{suffix}");
            if st.get(&name).is_ok() {
                copy_dense(&st, &mut w, &name)?;
                n_dense += 1;
            }
        }

        if layer % 6 == 0 || layer + 1 == cfg.num_hidden_layers {
            println!("  layer {:>2}/{} ...", layer + 1, cfg.num_hidden_layers);
        }
    }

    // --- final norm --------------------------------------------------------
    copy_dense(&st, &mut w, "model.norm.weight")?;
    n_dense += 1;

    // --- embeddings / output projection ------------------------------------
    //
    // With tied weights this one tensor serves two very different uses: a
    // single-row gather on input (free) and a full GEMV on output (27.6% of
    // decode traffic). Quantizing it is the largest single bandwidth win
    // available and also the one that most directly perturbs the output
    // distribution, so it is opt-in.
    let embed_t = st.get(EMBED)?;
    let (vocab, hidden) = embed_t.shape_2d()?;
    let embed32 = st.to_f32(EMBED)?;

    match head {
        HeadPrecision::Int4 if hidden % whetstone_quant::GROUP == 0 => {
            let packed = quantize_int4_g128(&embed32, hidden, vocab)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let e = report_error(&embed32, &packed);
            println!("  lm_head quantized to int4-g128, relative error {e:.4}");
            w.write_int4(EMBED, &packed).map_err(|e| anyhow::anyhow!("{e}"))?;
            n_quant += 1;
        }
        _ => {
            write_fp16(&mut w, EMBED, &embed32, &[vocab, hidden])?;
            n_dense += 1;
        }
    }

    if !cfg.tie_word_embeddings && st.get("lm_head.weight").is_ok() {
        copy_dense(&st, &mut w, "lm_head.weight")?;
        n_dense += 1;
    }

    // Embed the tokenizer so the .wstone is genuinely self-contained: `whetstone
    // chat model.wstone` should not need the original checkpoint on disk.
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists() {
        let bytes = std::fs::read(&tok_path)?;
        let n = bytes.len();
        w.write_extra("tokenizer.json", &bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("  embedded tokenizer.json ({:.1} MB)", n as f64 / 1e6);
    } else {
        println!("  no tokenizer.json in the source; `whetstone chat` will need --tokenizer");
    }

    let header = w.finish().map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = t0.elapsed();

    // --- report ------------------------------------------------------------
    let out_bytes = std::fs::metadata(out_path)?.len();
    let src_bytes = std::fs::metadata(&weights)?.len();
    let resident = header.decode_resident_bytes();
    let bw = bandwidth.unwrap_or(278.0);

    println!();
    println!("{:-<72}", "");
    println!("  wrote {} in {:.1}s", out_path.display(), elapsed.as_secs_f64());
    println!("{:-<72}", "");
    println!("  tensors            {} quantized, {} dense", n_quant, n_dense);
    println!(
        "  file size          {:.1} MB   (source {:.1} MB, {:.2}x smaller)",
        out_bytes as f64 / 1e6,
        src_bytes as f64 / 1e6,
        src_bytes as f64 / out_bytes as f64
    );
    println!(
        "  read per token     {:.1} MB   ({:.3} bits/weight over {:.1} M params)",
        resident as f64 / 1e6,
        resident as f64 * 8.0 / cfg.decode_resident_params() as f64,
        cfg.decode_resident_params() as f64 / 1e6
    );
    if n_quant > 0 {
        println!("  mean rel. error    {:.4}", err_sum / n_quant as f64);
        println!("  worst tensor       {:.4}  {}", worst.0, worst.1);
    }

    let fp16_bytes = cfg.decode_resident_params() as f64 * 2.0;
    println!();
    println!(
        "  decode ceiling at {bw:.0} GB/s:  {:.0} tok/s   (fp16 would be {:.0})",
        bw * 1e9 / resident as f64,
        bw * 1e9 / fp16_bytes
    );
    if head == HeadPrecision::Fp16 {
        println!();
        println!(
            "  lm_head is still fp16 and is {:.0}% of the bytes above.\n  \
             Re-run with --head int4 to cut that, then check quality with\n  \
             `whetstone verify` before trusting it.",
            cfg.lm_head_traffic_fraction() * 100.0
        );
    }
    println!();

    Ok(())
}

fn report_error(w32: &[f32], packed: &PackedInt4) -> f64 {
    let deq = whetstone_quant::dequantize_int4_g128(packed);
    relative_error(w32, &deq)
}

fn write_fp16<W: std::io::Write + std::io::Seek>(
    w: &mut format::Writer<W>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    let bits: Vec<u16> = data.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    w.write_fp16(name, &bits, shape).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

fn copy_dense<W: std::io::Write + std::io::Seek>(
    st: &SafeTensors,
    w: &mut format::Writer<W>,
    name: &str,
) -> Result<()> {
    let t = st.get(name)?;
    let shape = t.shape.clone();
    let data = st.to_f32(name)?;
    write_fp16(w, name, &data, &shape)
}
