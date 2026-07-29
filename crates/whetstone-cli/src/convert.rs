//! `whetstone convert` — HuggingFace checkpoint to `.wstone`.
//!
//! # The three stages, and why they are a pipeline
//!
//! Conversion is read → quantize → write, and each stage saturates a different
//! resource: the read is a page-fault storm against the checkpoint (on this
//! machine, a 5400 rpm disk), the quantize is 21 candidate grids per group of 32
//! across every core, and the write is sequential I/O. Run in sequence they
//! serialise, and a 7 B checkpoint takes ~40 minutes with eleven cores idle.
//!
//! So the loader runs on its own thread one tensor ahead of the packer, handing
//! over across a **rendezvous channel**. Capacity zero is deliberate: it gives
//! exactly one tensor of lookahead, which is all that is needed to hide the disk,
//! and it bounds peak memory at two widened tensors. A deeper queue would buy
//! nothing and would hold several 2 GB f32 buffers at 7 B.

use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use whetstone_core::{Checkpoint, ModelConfig};
use whetstone_quant::{format, quantize_int4_g128_measured, quantize_int4_hier_measured};

/// Precision for the transformer-block projections.
///
/// `Fp16` exists to keep a lossless reference path alive at all times. Without
/// one there is no way to separate "the engine is wrong" from "the quantizer is
/// lossy", and those two failures look identical from a perplexity number.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BodyPrecision {
    /// int4 group-32 with hierarchical scales. The production format.
    ///
    /// 4 + 8/32 + 32/in_features bits/weight — 0.036 more than `Int4` — for
    /// 1.15 less perplexity on Qwen2.5-0.5B. Group size turned out to be worth
    /// six times what the fitting algorithm is worth, and this is how to afford
    /// it: the per-group metadata is two 4-bit indices against one fp16 pair per
    /// row, instead of an fp16 scale and an fp16 zero per group.
    Int4Hier,
    /// int4 group-128 with an fp16 scale and zero per group. The 0.3.0 format,
    /// kept so an A/B against it is one flag.
    Int4,
    /// Dense fp16. 988 MB for Qwen2.5-0.5B, and the differential-testing baseline.
    Fp16,
}

/// Precision for the output projection.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum HeadPrecision {
    /// Keep `lm_head` in fp16. Safe default.
    Fp16,
    /// Quantize `lm_head` to int4 with hierarchical scales.
    Int4Hier,
    /// Quantize `lm_head` to int4-g128. Largest single bandwidth win available,
    /// and the riskiest — it sets the output distribution directly.
    Int4,
}

/// What a single tensor is packed as, once the precision knobs and the tensor's
/// own shape have both had their say.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pack {
    Fp16,
    Hier,
    G128,
}

/// Which precision knob governs a tensor.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A transformer projection: `--body`.
    Body,
    /// An embedding or output projection: `--head`.
    Head,
    /// Norms and biases. Always fp16 — they are kilobytes, and quantizing a
    /// gain vector that multiplies every activation is all risk and no bandwidth.
    Dense,
}

struct Item {
    name: String,
    role: Role,
}

/// A tensor read off the checkpoint and widened, on its way to the packer.
struct Loaded {
    item: Item,
    w32: Vec<f32>,
    shape: Vec<usize>,
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

/// Every tensor the converter will emit, in the order it emits them.
///
/// Built up front rather than discovered while writing, because the loader
/// thread has to know what to fetch next without racing the writer.
fn build_plan(cfg: &ModelConfig, st: &Checkpoint) -> Vec<Item> {
    let mut plan = Vec::new();
    let mut push = |name: String, role| plan.push(Item { name, role });

    for layer in 0..cfg.num_hidden_layers {
        for suffix in LINEAR_SUFFIXES {
            push(format!("model.layers.{layer}.{suffix}"), Role::Body);
        }
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            push(format!("model.layers.{layer}.{suffix}"), Role::Dense);
        }
        for suffix in ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"] {
            let name = format!("model.layers.{layer}.{suffix}");
            if st.get(&name).is_ok() {
                push(name, Role::Dense);
            }
        }
        // QK-RMSNorm gains (Qwen3, Qwen3-MoE, OLMo2, Gemma2): head_dim entries
        // each, applied to every head before RoPE. Dense fp16 -- they are half a
        // kilobyte per layer, and they multiply every query and key.
        for suffix in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            let name = format!("model.layers.{layer}.{suffix}");
            if st.get(&name).is_ok() {
                push(name, Role::Dense);
            }
        }
    }

    push("model.norm.weight".into(), Role::Dense);

    // With tied weights this one tensor serves two very different uses: a
    // single-row gather on input (free) and a full GEMV on output (27.6% of
    // decode traffic on the 0.5B). Quantizing it is the largest single
    // bandwidth win available and also the one that most directly perturbs the
    // output distribution, so it is opt-in.
    push(EMBED.into(), Role::Head);

    // Untied models carry a separate output projection, and **that** is the one
    // the decode GEMV reads. Earlier versions applied `--head` to
    // `embed_tokens` and copied `lm_head` as dense fp16, which is exactly
    // inverted. On Qwen2.5-7B that is a 1.09 GB fp16 matrix where 291 MB was
    // intended — the difference between fitting in 6 GB and not.
    if !cfg.tie_word_embeddings && st.get("lm_head.weight").is_ok() {
        push("lm_head.weight".into(), Role::Head);
    }

    plan
}

pub fn run(
    model_dir: &Path,
    out_path: &Path,
    head: HeadPrecision,
    body: BodyPrecision,
    bandwidth: Option<f64>,
    head_rescore: bool,
) -> Result<()> {
    let cfg = ModelConfig::from_dir(model_dir)
        .with_context(|| format!("could not load config from {}", model_dir.display()))?;

    let st = Checkpoint::open(model_dir)?;
    if st.shard_count() > 1 {
        println!("  reading {} shards", st.shard_count());
    }

    let raw_config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;

    println!("{:=<72}", "");
    println!("  converting {}", model_dir.display());
    println!("  to         {}", out_path.display());
    println!(
        "  {} threads, {} packer",
        rayon::current_num_threads(),
        whetstone_quant::cpu::detect().name()
    );
    println!("{:=<72}", "");

    let file = std::fs::File::create(out_path)
        .with_context(|| format!("could not create {}", out_path.display()))?;

    // The directory holds ~300 entries with two blobs each; 1 MiB is generous
    // and costs nothing, and the writer errors rather than truncating if it is
    // ever too small. MoE checkpoints have far more tensors, so it scales with
    // the plan rather than being a constant that silently stops being enough.
    let plan = build_plan(&cfg, &st);
    let header_reserve = (1 << 20).max(plan.len() as u64 * 512);

    let mut w = format::Writer::new(BufWriter::new(file), raw_config, header_reserve)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    w.set_quant_meta(
        "scheme",
        match body {
            BodyPrecision::Int4Hier => "int4-hier-g32",
            BodyPrecision::Int4 => "int4-g128-asymmetric",
            BodyPrecision::Fp16 => "fp16",
        },
    );
    w.set_quant_meta("group", match body {
        BodyPrecision::Int4Hier => "32",
        _ => "128",
    });
    w.set_quant_meta("method", match body {
        BodyPrecision::Int4Hier => "kqx2-weighted-ls",
        _ => "rtn",
    });
    w.set_quant_meta(
        "source",
        &st.files().first().map(|p| p.display().to_string()).unwrap_or_default(),
    );
    w.set_quant_meta("head_rescore", if head_rescore { "fp16-topk" } else { "none" });
    w.set_quant_meta(
        "lm_head",
        match head {
            HeadPrecision::Fp16 => "fp16",
            HeadPrecision::Int4 => "int4-g128",
            HeadPrecision::Int4Hier => "int4-hier-g32",
        },
    );

    let t0 = Instant::now();
    let mut n_quant = 0usize;
    let mut n_dense = 0usize;
    let mut worst: (f64, String) = (0.0, String::new());
    let mut err_sum = 0.0f64;
    let mut src_read = 0u64;
    let total = plan.len();

    // --- the pipeline ------------------------------------------------------
    //
    // Capacity 0 is a rendezvous: the loader fetches tensor N+1 while the packer
    // works on N, then blocks. One tensor of lookahead hides the disk; a deeper
    // queue would hold several 2 GB f32 buffers at 7 B and buy nothing.
    let (tx, rx) = sync_channel::<Result<Loaded>>(0);

    std::thread::scope(|scope| -> Result<()> {
        let st_ref = &st;
        scope.spawn(move || {
            for item in plan {
                let loaded = (|| -> Result<Loaded> {
                    let shape = st_ref.get(&item.name)?.shape.clone();
                    let w32 = st_ref.to_f32(&item.name)?;
                    Ok(Loaded { item, w32, shape })
                })();
                let failed = loaded.is_err();
                // A closed receiver means the packer already errored; stop
                // rather than reading the rest of a 15 GB checkpoint into a
                // channel nobody is draining.
                if tx.send(loaded).is_err() || failed {
                    return;
                }
            }
        });

        let mut done = 0usize;
        let mut next_report = Instant::now();
        for msg in rx {
            let Loaded { item, w32, shape } = msg?;
            src_read += (w32.len() * 2) as u64;

            let pack = match item.role {
                Role::Dense => Pack::Fp16,
                Role::Body => match body {
                    BodyPrecision::Fp16 => Pack::Fp16,
                    BodyPrecision::Int4Hier => Pack::Hier,
                    BodyPrecision::Int4 => Pack::G128,
                },
                Role::Head => match head {
                    HeadPrecision::Fp16 => Pack::Fp16,
                    HeadPrecision::Int4Hier => Pack::Hier,
                    HeadPrecision::Int4 => Pack::G128,
                },
            };

            // An fp16 copy of the output projection, for the top-k rescore. Not
            // read per token -- only the few rows that win -- so it is excluded
            // from `decode_resident_bytes` by its name suffix. Pointless when the
            // head is already fp16, and pointless on the *input* embedding, which
            // is a single-row gather either way.
            if head_rescore && matches!(item.role, Role::Head) && pack != Pack::Fp16 {
                let is_output = item.name != EMBED || cfg.tie_word_embeddings;
                if is_output {
                    let name = format!("{}{}", item.name, format::RESCORE_SUFFIX);
                    write_fp16(&mut w, &name, &w32, &shape)?;
                    n_dense += 1;
                }
            }

            match pack_one(&mut w, &item.name, &w32, &shape, pack)? {
                Some(e) => {
                    // A non-finite error means the packer produced something the
                    // kernel cannot reconstruct. It has to be named here: summed
                    // into the mean it shows up as a single `NaN` at the end of a
                    // seven-minute conversion with no indication of which of 339
                    // tensors caused it, which is how the f16 shared-scale
                    // underflow survived three model sizes.
                    if !e.is_finite() {
                        bail!(
                            "{}: relative error is {e}. The packed tensor cannot be \
                             reconstructed, so the conversion is aborted rather than \
                             written out. This is a quantizer bug -- please report it \
                             with the checkpoint name and this tensor.",
                            item.name
                        );
                    }
                    n_quant += 1;
                    err_sum += e;
                    if e > worst.0 {
                        worst = (e, item.name.clone());
                    }
                }
                None => n_dense += 1,
            }

            done += 1;
            if Instant::now() >= next_report || done == total {
                let secs = t0.elapsed().as_secs_f64();
                println!(
                    "  {done:>4}/{total} tensors   {:.2} GB read   {:.0} MB/s",
                    src_read as f64 / 1e9,
                    src_read as f64 / 1e6 / secs.max(1e-3),
                );
                next_report = Instant::now() + std::time::Duration::from_secs(5);
            }
        }
        Ok(())
    })?;

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
    let src_bytes = st.total_bytes();
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
        // The *byte* share, read out of the header that was just written.
        //
        // `cfg.lm_head_traffic_fraction()` is a **parameter** ratio and is only
        // equal to the byte ratio when every tensor has the same bits/weight --
        // which is exactly what this branch does not have. On the default
        // invocation (int4 body, fp16 head) it reported 28% where the true share
        // of the bytes printed above is 59%, understating the project's single
        // largest bandwidth decision by 2.1x.
        let head_bytes = header
            .tensor("lm_head.weight")
            .or_else(|_| header.tensor(EMBED))
            .map(|t| t.stored_bytes())
            .unwrap_or(0);
        println!();
        println!(
            "  lm_head is still fp16 and is {:.0}% of the bytes above.\n  \
             Re-run with --head int4 to cut that, then check quality with\n  \
             `whetstone verify` before trusting it.",
            head_bytes as f64 / resident.max(1) as f64 * 100.0
        );
    }
    println!();

    Ok(())
}

/// Packs one tensor and appends it. Returns the relative weight error if it was
/// quantized, `None` if it went out dense.
///
/// A shape the format cannot represent falls back to fp16 rather than being
/// padded, because padding would change the arithmetic the kernel performs and
/// the file would decode to something the converter never scored.
fn pack_one<W: std::io::Write + std::io::Seek>(
    w: &mut format::Writer<W>,
    name: &str,
    w32: &[f32],
    shape: &[usize],
    pack: Pack,
) -> Result<Option<f64>> {
    let (out_f, in_f) = match shape {
        [r, c] => (*r, *c),
        _ => {
            write_fp16(w, name, w32, shape)?;
            return Ok(None);
        }
    };

    match pack {
        Pack::Hier if in_f % whetstone_quant::HGROUP == 0 => {
            let (packed, e) =
                quantize_int4_hier_measured(w32, in_f, out_f).map_err(|e| anyhow::anyhow!("{e}"))?;
            w.write_int4_hier(name, &packed).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(Some(e))
        }
        Pack::G128 if in_f % whetstone_quant::GROUP == 0 => {
            let (packed, e) =
                quantize_int4_g128_measured(w32, in_f, out_f).map_err(|e| anyhow::anyhow!("{e}"))?;
            w.write_int4(name, &packed).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(Some(e))
        }
        Pack::Fp16 => {
            write_fp16(w, name, w32, shape)?;
            Ok(None)
        }
        // Divisibility fallback. Loud, because a model that silently converts at
        // fp16 is a model whose bytes/token is not what the report claims.
        _ => {
            let g = if pack == Pack::Hier { whetstone_quant::HGROUP } else { whetstone_quant::GROUP };
            println!("  {name}: in_features {in_f} not a multiple of {g}, keeping fp16");
            write_fp16(w, name, w32, shape)?;
            Ok(None)
        }
    }
}

fn write_fp16<W: std::io::Write + std::io::Seek>(
    w: &mut format::Writer<W>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<()> {
    // The head is 545 M elements on an untied 7 B model; a serial narrowing loop
    // over that is seconds, on the critical path, for arithmetic with no
    // dependencies at all. Parallel over strips, and each strip goes through the
    // ISA-dispatched converter so the F16C instruction gets used where it exists.
    const STRIP: usize = 1 << 16;
    let mut bits = vec![0u16; data.len()];
    bits.par_chunks_mut(STRIP)
        .zip(data.par_chunks(STRIP))
        .for_each(|(dst, src)| whetstone_quant::cpu::narrow_f16(src, dst));
    w.write_fp16(name, &bits, shape).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
