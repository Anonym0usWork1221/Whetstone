//! `whetstone inspect` — architecture, tensor inventory, and the roofline.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use whetstone_core::{Checkpoint, ModelConfig};
use whetstone_kernels::Device;

/// Candidate weight formats and their true cost in bits per weight, including
/// the per-group scale/zero metadata. Quoting a format as "4-bit" while ignoring
/// its scales understates bandwidth by 5-10%, which is exactly the quantity that
/// decides decode speed, so the overhead is counted here.
const FORMATS: &[(&str, f64)] = &[
    ("fp16", 16.0),
    ("int8 per-channel", 8.0),
    ("int4 g128 + fp16 s/z", 4.0 + 32.0 / 128.0),
    ("int3 g128 + fp16 s/z", 3.0 + 32.0 / 128.0),
    ("int2 g128 + fp16 s/z", 2.0 + 32.0 / 128.0),
    ("ternary g128 + fp16 s", 1.585 + 16.0 / 128.0),
    ("binary g128 + fp16 s", 1.0 + 16.0 / 128.0),
];

pub fn run(model_dir: &Path, bandwidth: Option<f64>, list_tensors: bool) -> Result<()> {
    let cfg = ModelConfig::from_dir(model_dir)
        .with_context(|| format!("could not load config from {}", model_dir.display()))?;

    let st = Checkpoint::open(model_dir)?;

    // Prefer the real GPU's bandwidth; fall back to a stated value so the tool
    // is still useful on a machine without a CUDA device.
    let bw = match bandwidth {
        Some(b) => b,
        None => Device::default_device().map(|d| d.bandwidth_gbs()).unwrap_or(336.0),
    };

    println!("{:=<72}", "");
    println!("  {}", model_dir.display());
    println!("{:=<72}", "");
    println!("  model_type        {}", cfg.model_type);
    println!("  layers            {}", cfg.num_hidden_layers);
    println!("  hidden            {}", cfg.hidden_size);
    println!(
        "  heads             {} Q / {} KV   (GQA {}:1)",
        cfg.num_attention_heads,
        cfg.n_kv_heads(),
        cfg.gqa_ratio()
    );
    println!("  head_dim          {}", cfg.head_dim());
    println!("  intermediate      {}", cfg.intermediate_size);
    println!("  vocab             {}", cfg.vocab_size);
    println!("  tied embeddings   {}", cfg.tie_word_embeddings);
    println!("  rms_norm_eps      {:e}", cfg.rms_norm_eps);
    println!("  rope_theta        {}", cfg.rope_theta);

    println!();
    println!("  parameters");
    println!(
        "    per layer       {:>12.2} M   (attn {:.2} M, mlp {:.2} M)",
        cfg.params_per_layer() as f64 / 1e6,
        cfg.attn_params_per_layer() as f64 / 1e6,
        cfg.mlp_params_per_layer() as f64 / 1e6
    );
    println!(
        "    transformer     {:>12.2} M",
        cfg.non_embedding_params() as f64 / 1e6
    );
    println!(
        "    lm_head         {:>12.2} M   <- {:.1}% of decode traffic{}",
        cfg.lm_head_params() as f64 / 1e6,
        cfg.lm_head_traffic_fraction() * 100.0,
        if cfg.tie_word_embeddings { ", tied to embeddings" } else { "" }
    );
    println!(
        "    read per token  {:>12.2} M   <- the roofline denominator",
        cfg.decode_resident_params() as f64 / 1e6
    );
    println!("    total           {:>12.2} M", cfg.total_params() as f64 / 1e6);
    println!(
        "    MLP share       {:>11.1}% of a layer, {:.1}% of decode traffic",
        cfg.mlp_weight_fraction() * 100.0,
        (cfg.num_hidden_layers * cfg.mlp_params_per_layer()) as f64
            / cfg.decode_resident_params() as f64
            * 100.0
    );

    // --- checkpoint contents ------------------------------------------------
    let stored: usize = st.iter().map(|t| t.numel()).sum();
    println!();
    println!("  checkpoint");
    println!("    tensors         {:>12}", st.len());
    println!("    data            {:>12.1} MB", st.data_bytes() as f64 / 1e6);
    println!("    elements        {:>12.2} M", stored as f64 / 1e6);

    let mut by_dtype: BTreeMap<String, usize> = BTreeMap::new();
    for t in st.iter() {
        *by_dtype.entry(format!("{:?}", t.dtype)).or_default() += t.numel();
    }
    for (d, n) in &by_dtype {
        println!("    {:<15} {:>12.2} M elements", d, *n as f64 / 1e6);
    }

    if stored != cfg.total_params() {
        println!(
            "    NOTE: checkpoint holds {:.2} M elements, config implies {:.2} M \
             (difference is norm weights and biases, which config does not count)",
            stored as f64 / 1e6,
            cfg.total_params() as f64 / 1e6
        );
    }

    // --- required tensors ---------------------------------------------------
    let mut missing = Vec::new();
    let mut check = |name: String| {
        if st.get(&name).is_err() {
            missing.push(name);
        }
    };
    check("model.embed_tokens.weight".into());
    check("model.norm.weight".into());
    for l in 0..cfg.num_hidden_layers {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            check(format!("model.layers.{l}.{suffix}"));
        }
    }

    println!();
    if missing.is_empty() {
        println!("  all {} required tensors present", 2 + cfg.num_hidden_layers * 9);
    } else {
        println!("  MISSING {} required tensors:", missing.len());
        for m in missing.iter().take(10) {
            println!("    {m}");
        }
        if missing.len() > 10 {
            println!("    ... and {} more", missing.len() - 10);
        }
    }

    if list_tensors {
        println!();
        println!("  {:<52} {:<6} {:>14}", "tensor", "dtype", "shape");
        for t in st.iter() {
            println!("  {:<52} {:<6} {:>14?}", t.name, format!("{:?}", t.dtype), t.shape);
        }
    }

    // --- roofline -----------------------------------------------------------
    let rl = cfg.roofline(bw);
    println!();
    println!("{:-<72}", "");
    println!("  Roofline for batch=1 decode at {bw:.0} GB/s");
    println!("{:-<72}", "");
    println!(
        "  {:<24} {:>9} {:>13} {:>14}",
        "weight format", "bits/wt", "bytes/token", "tok/s ceiling"
    );
    println!("  {:-<64}", "");
    for (name, bits) in FORMATS {
        println!(
            "  {:<24} {:>9.2} {:>11.1} MB {:>14.0}",
            name,
            bits,
            rl.bytes_per_token(*bits) / 1e6,
            rl.max_tokens_per_sec(*bits)
        );
    }
    println!();
    println!(
        "  Ceilings assume 100% bandwidth utilisation; a well-written kernel\n  \
         attains 60-80%. Compute throughput does not appear in this table,\n  \
         which is the point: at batch=1 the arithmetic is free and the memory\n  \
         traffic is everything."
    );
    if cfg.lm_head_traffic_fraction() > 0.1 {
        println!(
            "\n  Note: lm_head is {:.0}% of the bytes above. Tied embeddings make it\n  \
             look like a lookup table, but the OUTPUT projection is a full GEMV\n  \
             over the whole [vocab, hidden] matrix on every token. Quantizing it\n  \
             is worth more than any further work on the transformer blocks.",
            cfg.lm_head_traffic_fraction() * 100.0
        );
    }

    // --- KV cache -----------------------------------------------------------
    println!();
    println!("  KV cache (GQA {}:1 keeps this small)", cfg.gqa_ratio());
    for len in [2048usize, 8192, 32768] {
        println!(
            "    {:>6} tokens   {:>7.1} MB fp16   {:>7.1} MB int8",
            len,
            cfg.kv_cache_bytes(len, 2) as f64 / 1e6,
            cfg.kv_cache_bytes(len, 1) as f64 / 1e6
        );
    }
    println!();

    Ok(())
}
