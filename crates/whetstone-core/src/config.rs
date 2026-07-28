//! Model configuration, and the roofline arithmetic derived from it.
//!
//! The roofline numbers are not decoration. Whetstone's central design claim is
//! that batch=1 decode is bandwidth bound, so [`Roofline`] is the tool used to
//! decide whether an optimization is worth implementing *before* implementing
//! it. See `docs/design.md`.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A Qwen2-family model configuration, as parsed from `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model family identifier, e.g. `"qwen2"`.
    #[serde(default)]
    pub model_type: String,
    /// Residual stream width.
    pub hidden_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads. Fewer than query heads means GQA.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// SwiGLU inner width.
    pub intermediate_size: usize,
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Maximum trained context length.
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    /// Whether `lm_head` reuses the embedding matrix.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Activation function name. Whetstone supports `silu`.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Explicit head dimension, when the config overrides `hidden/heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn default_rms_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_max_pos() -> usize {
    32768
}
fn default_hidden_act() -> String {
    "silu".into()
}

impl ModelConfig {
    /// Parses a `config.json`.
    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(s)
            .map_err(|e| Error::Config(format!("could not parse config.json: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reads and parses `config.json` from a model directory.
    pub fn from_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = dir.as_ref().join("config.json");
        let s = std::fs::read_to_string(&p)
            .map_err(|e| Error::Config(format!("could not read {}: {e}", p.display())))?;
        Self::from_json(&s)
    }

    /// Rejects configurations Whetstone cannot execute correctly.
    ///
    /// Failing loudly at load time is deliberate: a silently mis-shaped model
    /// produces plausible-looking garbage that is expensive to debug later.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 || self.num_attention_heads == 0 {
            return Err(Error::Config("hidden_size and num_attention_heads must be non-zero".into()));
        }
        if self.head_dim.is_none() && self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::Config(format!(
                "hidden_size {} is not divisible by num_attention_heads {} and no head_dim is given",
                self.hidden_size, self.num_attention_heads
            )));
        }
        let kv = self.n_kv_heads();
        if kv == 0 || self.num_attention_heads % kv != 0 {
            return Err(Error::Config(format!(
                "num_attention_heads {} must be a multiple of num_key_value_heads {kv}",
                self.num_attention_heads
            )));
        }
        if self.hidden_act != "silu" && self.hidden_act != "swish" {
            return Err(Error::Config(format!(
                "unsupported activation {:?}; Whetstone implements SwiGLU with SiLU",
                self.hidden_act
            )));
        }
        if !self.model_type.is_empty() && self.model_type != "qwen2" && self.model_type != "qwen3" {
            return Err(Error::Config(format!(
                "unsupported model_type {:?}; Whetstone targets qwen2/qwen3 layer topology",
                self.model_type
            )));
        }
        Ok(())
    }

    /// Number of key/value heads, defaulting to MHA when unspecified.
    pub fn n_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head width.
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Query heads sharing each KV head. `1` means no grouping.
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.n_kv_heads()
    }

    /// Attention weight count in one block, excluding biases.
    pub fn attn_params_per_layer(&self) -> usize {
        let h = self.hidden_size;
        let d = self.head_dim();
        let q = h * self.num_attention_heads * d;
        let k = h * self.n_kv_heads() * d;
        let v = k;
        let o = self.num_attention_heads * d * h;
        q + k + v + o
    }

    /// MLP weight count in one block: gate, up and down projections.
    pub fn mlp_params_per_layer(&self) -> usize {
        3 * self.hidden_size * self.intermediate_size
    }

    /// Total weight count in one block.
    pub fn params_per_layer(&self) -> usize {
        self.attn_params_per_layer() + self.mlp_params_per_layer()
    }

    /// Transformer-block weights, excluding embeddings and `lm_head`.
    pub fn non_embedding_params(&self) -> usize {
        self.num_hidden_layers * self.params_per_layer()
    }

    /// Weights actually streamed from memory on every decode step.
    ///
    /// This is the roofline denominator, and it is **not** just the transformer
    /// blocks. Two different things use the embedding matrix:
    ///
    /// - the *input* embedding is a gather of a single row — negligible;
    /// - the *output* projection (`lm_head`) is a full GEMV against the entire
    ///   `[vocab, hidden]` matrix, read in its entirety, every token.
    ///
    /// Tied embeddings mean those are the same weights, which makes it easy to
    /// dismiss the whole matrix as "just a lookup". For Qwen2.5-0.5B the head is
    /// 136.1 M of 494.0 M parameters — **27.6% of all decode traffic**. Omitting
    /// it overstates the token-rate ceiling by 1.38x.
    pub fn decode_resident_params(&self) -> usize {
        self.non_embedding_params() + self.lm_head_params()
    }

    /// Parameters in the output projection.
    ///
    /// Identical to the embedding matrix when weights are tied, but it is a
    /// distinct *use* of them, and the only one that costs bandwidth.
    pub fn lm_head_params(&self) -> usize {
        self.vocab_size * self.hidden_size
    }

    /// Fraction of per-token weight traffic spent on the output projection.
    pub fn lm_head_traffic_fraction(&self) -> f64 {
        self.lm_head_params() as f64 / self.decode_resident_params() as f64
    }

    /// Embedding matrix element count.
    pub fn embedding_params(&self) -> usize {
        self.vocab_size * self.hidden_size
    }

    /// Every stored parameter, counting `lm_head` only when it is untied.
    pub fn total_params(&self) -> usize {
        let extra = if self.tie_word_embeddings { 0 } else { self.embedding_params() };
        self.non_embedding_params() + self.embedding_params() + extra
    }

    /// Fraction of per-block weights belonging to the MLP.
    ///
    /// For Qwen2.5-0.5B this is ~88%, which is why the MLP projections are the
    /// first thing Whetstone quantizes.
    pub fn mlp_weight_fraction(&self) -> f64 {
        self.mlp_params_per_layer() as f64 / self.params_per_layer() as f64
    }

    /// KV cache bytes for `seq_len` tokens at `bytes_per_element` per entry.
    pub fn kv_cache_bytes(&self, seq_len: usize, bytes_per_element: usize) -> usize {
        // 2 tensors (K and V) x layers x kv_heads x head_dim x tokens
        2 * self.num_hidden_layers * self.n_kv_heads() * self.head_dim() * seq_len
            * bytes_per_element
    }

    /// Builds the bandwidth model for this configuration.
    pub fn roofline(&self, bandwidth_gbs: f64) -> Roofline {
        Roofline { decode_resident_params: self.decode_resident_params(), bandwidth_gbs }
    }
}

/// The bandwidth model that governs batch=1 decode.
///
/// At batch=1 every weight is read once and used for a single multiply-add, so
/// arithmetic intensity is ~2 FLOP/byte against a tensor-core balance point of
/// ~120 FLOP/byte. Decode speed is therefore set by bytes moved, and a
/// technique that reduces FLOPs without reducing bytes buys nothing.
#[derive(Debug, Clone, Copy)]
pub struct Roofline {
    /// Weights streamed per token, including the output projection.
    pub decode_resident_params: usize,
    /// Device memory bandwidth in GB/s.
    pub bandwidth_gbs: f64,
}

impl Roofline {
    /// Bytes read per decode step at a given weight width.
    pub fn bytes_per_token(&self, bits_per_weight: f64) -> f64 {
        self.decode_resident_params as f64 * bits_per_weight / 8.0
    }

    /// Upper bound on tokens/second at a given weight width.
    ///
    /// Assumes perfect bandwidth utilisation; a good kernel attains 60-80%.
    pub fn max_tokens_per_sec(&self, bits_per_weight: f64) -> f64 {
        self.bandwidth_gbs * 1e9 / self.bytes_per_token(bits_per_weight)
    }

    /// Fraction of the ceiling an observed rate achieves.
    pub fn attainment(&self, bits_per_weight: f64, observed_tok_s: f64) -> f64 {
        observed_tok_s / self.max_tokens_per_sec(bits_per_weight)
    }

    /// Bandwidth actually consumed by an observed token rate, in GB/s.
    pub fn achieved_bandwidth_gbs(&self, bits_per_weight: f64, observed_tok_s: f64) -> f64 {
        self.bytes_per_token(bits_per_weight) * observed_tok_s / 1e9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact config.json shipped with Qwen2.5-0.5B-Instruct.
    const QWEN_05B: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": 896,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": true,
        "vocab_size": 151936
    }"#;

    fn qwen() -> ModelConfig {
        ModelConfig::from_json(QWEN_05B).unwrap()
    }

    #[test]
    fn parses_reference_model() {
        let c = qwen();
        assert_eq!(c.hidden_size, 896);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.n_kv_heads(), 2);
        assert_eq!(c.gqa_ratio(), 7);
        assert!(c.tie_word_embeddings);
    }

    #[test]
    fn parameter_counts_match_the_checkpoint() {
        let c = qwen();
        // attn: q 896*896 + k 896*128 + v 896*128 + o 896*896
        assert_eq!(c.attn_params_per_layer(), 802_816 + 114_688 + 114_688 + 802_816);
        // mlp: 3 * 896 * 4864
        assert_eq!(c.mlp_params_per_layer(), 13_074_432);
        assert_eq!(c.params_per_layer(), 14_909_440);
        assert_eq!(c.non_embedding_params(), 357_826_560);
        assert_eq!(c.embedding_params(), 136_134_656);
        // Tied embeddings: lm_head is not stored separately.
        assert_eq!(c.total_params(), 493_961_216);
    }

    #[test]
    fn mlp_dominates_the_weight_budget() {
        // 88% of per-layer weights are MLP, so that is where quantization pays.
        let f = qwen().mlp_weight_fraction();
        assert!((0.87..0.89).contains(&f), "mlp fraction was {f}");
    }

    #[test]
    fn roofline_ranks_formats_by_bytes_not_flops() {
        let r = qwen().roofline(336.0);

        let fp16 = r.max_tokens_per_sec(16.0);
        let int8 = r.max_tokens_per_sec(8.0);
        let int4 = r.max_tokens_per_sec(4.125);
        let tern = r.max_tokens_per_sec(1.705);

        // ~340 tok/s at fp16 on 336 GB/s. An earlier version of this test
        // asserted ~470 because the roofline omitted lm_head; the head is 27.6%
        // of decode traffic, so that was optimistic by 1.38x.
        assert!((320.0..360.0).contains(&fp16), "fp16 ceiling was {fp16}");
        // Halving the width doubles the ceiling. Exactly.
        assert!((int8 / fp16 - 2.0).abs() < 1e-9);
        assert!(int4 > 1200.0 && int4 < 1400.0, "int4 ceiling was {int4}");
        assert!(tern > 2800.0 && tern < 3400.0, "ternary ceiling was {tern}");
    }

    #[test]
    fn attainment_round_trips_against_the_ceiling() {
        let r = qwen().roofline(336.0);
        let ceiling = r.max_tokens_per_sec(16.0);
        assert!((r.attainment(16.0, ceiling) - 1.0).abs() < 1e-9);
        assert!((r.achieved_bandwidth_gbs(16.0, ceiling) - 336.0).abs() < 1e-6);
    }

    #[test]
    fn roofline_counts_the_output_projection() {
        let c = qwen();

        // Tied embeddings make it tempting to treat the embedding matrix as a
        // pure lookup. The output projection is a full GEMV against all of it,
        // every token, and it is over a quarter of decode traffic.
        assert_eq!(c.lm_head_params(), 136_134_656);
        assert_eq!(c.decode_resident_params(), 357_826_560 + 136_134_656);
        assert!(
            (c.lm_head_traffic_fraction() - 0.2756).abs() < 1e-3,
            "head fraction was {}",
            c.lm_head_traffic_fraction()
        );

        // Omitting it overstates the ceiling by exactly that ratio.
        let full = c.roofline(336.0).max_tokens_per_sec(16.0);
        let body_only = 336e9 / (c.non_embedding_params() as f64 * 2.0);
        assert!(
            (body_only / full - 1.38).abs() < 0.01,
            "omitting lm_head inflates the ceiling by {:.3}x",
            body_only / full
        );
    }

    #[test]
    fn gqa_shrinks_the_kv_cache_sevenfold() {
        let c = qwen();
        // 2 KV heads instead of 14: the cache is 7x smaller than MHA would be.
        let gqa = c.kv_cache_bytes(4096, 2);
        let mha = 2 * c.num_hidden_layers * c.num_attention_heads * c.head_dim() * 4096 * 2;
        assert_eq!(mha / gqa, 7);
        assert_eq!(gqa, 2 * 24 * 2 * 64 * 4096 * 2);
    }

    #[test]
    fn malformed_configs_are_rejected() {
        let bad_heads = QWEN_05B.replace("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 5");
        assert!(ModelConfig::from_json(&bad_heads).is_err(), "14 is not a multiple of 5");

        let bad_act = QWEN_05B.replace("\"silu\"", "\"gelu\"");
        assert!(ModelConfig::from_json(&bad_act).is_err(), "gelu is not implemented");

        assert!(ModelConfig::from_json("{ not json }").is_err());
    }
}
