//! Loading a `.wstone` file onto the device.
//!
//! A `.wstone` is not a checkpoint that needs converting — the packing already
//! happened at `whetstone convert` time. Loading is an `mmap`, a pointer walk,
//! and one `cudaMemcpy` per blob. Nothing is transposed, repacked, or
//! dequantized on the way in; if it were, the format would not be doing its job.
//!
//! # The tied-embedding problem
//!
//! Qwen2.5-0.5B ties its input embedding to its output projection, so one
//! `[151936, 896]` matrix serves two uses that could not be more different:
//!
//! - **input**: gather one row. 1.8 KB. Free.
//! - **output**: a full GEMV against all 136.1 M parameters, every single token.
//!   **27.6% of decode traffic**, in one matrix.
//!
//! It is stored once and used both ways, which is why [`Embedding`] owns the
//! buffers and exposes two methods rather than there being separate `embed` and
//! `lm_head` tensors. It is also why quantizing "just the head" is not a free
//! action: it quantizes the input embedding at the same time.

use std::path::Path;

use whetstone_kernels::{gemv, DeviceBuffer, DeviceCursor, QuantLinear};
use whetstone_quant::format::{self, Header, TensorEntry, TensorKind};

use crate::error::{Error, Result};
use crate::ModelConfig;

/// A linear layer resident on the device, in whatever precision it was stored.
pub enum DeviceLinear {
    /// int4 group-128, the format Whetstone targets.
    Int4(QuantLinear),
    /// Dense fp16. Used for anything whose input width is not a multiple of 128
    /// and for `lm_head` when the converter was told to keep it exact.
    Fp16 {
        /// Row-major `[out_features][in_features]` f16 bit patterns.
        w: DeviceBuffer<u16>,
        /// Input width.
        in_features: usize,
        /// Output width.
        out_features: usize,
    },
}

impl DeviceLinear {
    /// Input width.
    pub fn in_features(&self) -> usize {
        match self {
            Self::Int4(q) => q.in_features(),
            Self::Fp16 { in_features, .. } => *in_features,
        }
    }

    /// Output width.
    pub fn out_features(&self) -> usize {
        match self {
            Self::Int4(q) => q.out_features(),
            Self::Fp16 { out_features, .. } => *out_features,
        }
    }

    /// Weight bytes streamed per invocation. The roofline numerator.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Int4(q) => q.bytes(),
            Self::Fp16 { w, .. } => w.bytes(),
        }
    }

    /// `y = Wx + b`, or `y += Wx + b` when `accumulate`.
    ///
    /// The accumulating form is how the residual stream is updated: `o_proj` and
    /// `down_proj` add straight into it instead of writing a temporary that a
    /// separate kernel would then have to add.
    pub fn forward(
        &self,
        x: &DeviceBuffer<u16>,
        bias: Option<&DeviceBuffer<u16>>,
        y: &mut DeviceBuffer<f32>,
        accumulate: bool,
    ) -> Result<()> {
        match self {
            Self::Int4(q) => q.gemv_ex(x, bias, y, accumulate)?,
            Self::Fp16 { w, in_features, out_features } => {
                gemv::gemv_fp16_ex(w, x, bias, y, *in_features, *out_features, accumulate)?
            }
        }
        Ok(())
    }
}

/// The embedding matrix, serving both the input gather and the output
/// projection.
pub enum Embedding {
    /// Dense fp16 table.
    Fp16 {
        /// `[vocab][hidden]` f16 bit patterns.
        w: DeviceBuffer<u16>,
        /// Vocabulary size.
        vocab: usize,
        /// Residual width.
        hidden: usize,
    },
    /// int4-g128 table. Quantizing the head quantizes the input embedding too.
    Int4(QuantLinear),
}

impl Embedding {
    /// Vocabulary size.
    pub fn vocab(&self) -> usize {
        match self {
            Self::Fp16 { vocab, .. } => *vocab,
            Self::Int4(q) => q.out_features(),
        }
    }

    /// Residual width.
    pub fn hidden(&self) -> usize {
        match self {
            Self::Fp16 { hidden, .. } => *hidden,
            Self::Int4(q) => q.in_features(),
        }
    }

    /// Bytes read when this matrix is used as the output projection.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Fp16 { w, .. } => w.bytes(),
            Self::Int4(q) => q.bytes(),
        }
    }

    /// Gathers the row named by a device-resident token id.
    ///
    /// The id is on the device rather than a plain `u32` so the gather is
    /// identical on every token and can be captured into the decode graph.
    pub fn gather(&self, token: &DeviceCursor, out: &mut DeviceBuffer<f32>) -> Result<()> {
        match self {
            Self::Fp16 { w, .. } => whetstone_kernels::embed_fp16(w, token, out)?,
            Self::Int4(q) => q.gather_row(token, out)?,
        }
        Ok(())
    }

    /// `logits = E x`, the output projection.
    pub fn project(&self, x: &DeviceBuffer<u16>, logits: &mut DeviceBuffer<f32>) -> Result<()> {
        match self {
            Self::Fp16 { w, vocab, hidden } => {
                gemv::gemv_fp16_ex(w, x, None, logits, *hidden, *vocab, false)?
            }
            Self::Int4(q) => q.gemv_ex(x, None, logits, false)?,
        }
        Ok(())
    }
}

/// One transformer block's weights.
///
/// The q/k/v and gate/up projections are **concatenated at load time** into
/// single matrices. They share an input, so nothing about the arithmetic
/// changes — what changes is how many warps the GEMV can create, and that is
/// what sets how many loads it can keep in flight.
///
/// Measured on the RTX 2060, the three separate attention projections ran at
/// 92, 19 and 19 GB/s: a 128-row matrix cannot fill 30 SMs however it is
/// blocked, and `lm_head` reaches 254 GB/s on the same kernel purely because it
/// has 151936 rows to spread across the machine. Fusing gives the attention
/// block one 1152-row matrix and the MLP one 9728-row matrix instead.
///
/// Fusion happens on the host during load rather than in the file format, so
/// existing `.wstone` files keep working and nothing needs re-converting.
/// Concatenating int4-g128 along the output dimension is just appending rows:
/// the packed layout is row-major with a fixed stride, and the scales are
/// per-row.
pub struct LayerWeights {
    /// Pre-attention RMSNorm gain.
    pub input_norm: DeviceBuffer<u16>,
    /// Pre-MLP RMSNorm gain.
    pub post_attn_norm: DeviceBuffer<u16>,

    /// Query, key and value projections as one matrix of `n_q*hd + 2*n_kv*hd`
    /// rows.
    pub qkv_proj: DeviceLinear,
    /// The three q/k/v biases concatenated, when the architecture has them.
    pub qkv_bias: Option<DeviceBuffer<u16>>,
    /// Output projection. Writes into the residual stream.
    pub o_proj: DeviceLinear,

    /// SwiGLU gate and up projections as one matrix of `2 * intermediate` rows.
    pub gate_up_proj: DeviceLinear,
    /// SwiGLU down projection. Writes into the residual stream.
    pub down_proj: DeviceLinear,
}

impl LayerWeights {
    /// Weight bytes this block streams per token.
    pub fn bytes(&self) -> usize {
        self.qkv_proj.bytes() + self.o_proj.bytes() + self.gate_up_proj.bytes()
            + self.down_proj.bytes()
    }
}

/// A whole model, resident on the device.
pub struct ModelWeights {
    /// Architecture, read from the `.wstone` header rather than a sidecar.
    pub config: ModelConfig,
    /// Transformer blocks in order.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain.
    pub final_norm: DeviceBuffer<u16>,
    /// The embedding matrix. Also the output projection when weights are tied.
    pub embed: Embedding,
    /// A separate `lm_head`, present only when embeddings are untied.
    pub lm_head: Option<DeviceLinear>,
    /// Provenance from the header: quantization scheme, source path, and so on.
    pub quant_meta: std::collections::BTreeMap<String, String>,
    /// The source `tokenizer.json`, when the converter embedded one.
    ///
    /// Carried so a `.wstone` needs no sidecar: text in, text out, from one file.
    pub tokenizer_json: Option<String>,
}

impl ModelWeights {
    /// Reads a `.wstone` file and uploads it.
    ///
    /// Every tensor the architecture calls for must be present; a missing one is
    /// an error here rather than a silently wrong forward pass later.
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::Io(format!("could not open {}: {e}", path.display())))?;
        // SAFETY: the mapping is read-only and lives no longer than this
        // function. A concurrent writer truncating the file underneath us would
        // be UB, which is the standard and unavoidable caveat of mmap; the
        // format's checksums are the defence against a corrupt file, not a
        // racing one.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Io(format!("could not mmap {}: {e}", path.display())))?;

        let bytes: &[u8] = &map;
        let header = format::read_header(bytes, bytes.len() as u64)
            .map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;

        let config: ModelConfig = serde_json::from_value(header.model_config.clone())
            .map_err(|e| Error::Config(format!("{}: embedded config is unusable: {e}", path.display())))?;
        config.validate()?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(LayerWeights {
                input_norm: load_fp16(&header, bytes, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: load_fp16(
                    &header,
                    bytes,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                qkv_proj: fuse_linears(
                    &header,
                    bytes,
                    &[
                        format!("{p}.self_attn.q_proj.weight"),
                        format!("{p}.self_attn.k_proj.weight"),
                        format!("{p}.self_attn.v_proj.weight"),
                    ],
                )?,
                qkv_bias: fuse_biases(
                    &header,
                    bytes,
                    &[
                        format!("{p}.self_attn.q_proj.bias"),
                        format!("{p}.self_attn.k_proj.bias"),
                        format!("{p}.self_attn.v_proj.bias"),
                    ],
                    &[
                        config.num_attention_heads * config.head_dim(),
                        config.n_kv_heads() * config.head_dim(),
                        config.n_kv_heads() * config.head_dim(),
                    ],
                )?,
                o_proj: load_linear(&header, bytes, &format!("{p}.self_attn.o_proj.weight"))?,
                gate_up_proj: fuse_linears(
                    &header,
                    bytes,
                    &[
                        format!("{p}.mlp.gate_proj.weight"),
                        format!("{p}.mlp.up_proj.weight"),
                    ],
                )?,
                down_proj: load_linear(&header, bytes, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }

        let final_norm = load_fp16(&header, bytes, "model.norm.weight")?;

        let embed_entry = tensor(&header, "model.embed_tokens.weight")?;
        let embed = match embed_entry.kind {
            TensorKind::Int4G128 => Embedding::Int4(load_int4(bytes, embed_entry)?),
            _ => {
                let (vocab, hidden) = shape2(embed_entry)?;
                Embedding::Fp16 {
                    w: load_fp16(&header, bytes, "model.embed_tokens.weight")?,
                    vocab,
                    hidden,
                }
            }
        };

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load_linear(&header, bytes, "lm_head.weight")?)
        };

        // Cross-check the header against the architecture before anything runs.
        // A mismatch here means the file and its embedded config disagree, which
        // produces plausible garbage rather than a crash if it goes unchecked.
        if embed.hidden() != config.hidden_size {
            return Err(Error::Shape(format!(
                "embedding width {} does not match hidden_size {}",
                embed.hidden(),
                config.hidden_size
            )));
        }
        if embed.vocab() != config.vocab_size {
            return Err(Error::Shape(format!(
                "embedding rows {} do not match vocab_size {}",
                embed.vocab(),
                config.vocab_size
            )));
        }

        let tokenizer_json = match header.extras.get("tokenizer.json") {
            Some(b) => {
                let lo = b.offset as usize;
                let hi = lo + b.len as usize;
                let raw = bytes.get(lo..hi).ok_or_else(|| {
                    Error::Format("embedded tokenizer.json is out of range".into())
                })?;
                Some(String::from_utf8(raw.to_vec()).map_err(|e| {
                    Error::Format(format!("embedded tokenizer.json is not UTF-8: {e}"))
                })?)
            }
            None => None,
        };

        Ok(Self {
            config,
            layers,
            final_norm,
            embed,
            lm_head,
            quant_meta: header.quant,
            tokenizer_json,
        })
    }

    /// Bytes streamed per decode step, including the output projection.
    ///
    /// This is the roofline denominator. It counts the embedding matrix because
    /// the *output* use reads all of it every token — omitting it overstates the
    /// ceiling by 1.38x on Qwen2.5-0.5B.
    pub fn decode_bytes(&self) -> usize {
        let body: usize = self.layers.iter().map(LayerWeights::bytes).sum();
        let head = self.lm_head.as_ref().map_or_else(|| self.embed.bytes(), DeviceLinear::bytes);
        body + head
    }

    /// Effective bits per weight across everything a decode step reads.
    pub fn bits_per_weight(&self) -> f64 {
        self.decode_bytes() as f64 * 8.0 / self.config.decode_resident_params() as f64
    }
}

// ------------------------------------------------------------------ loading

fn tensor<'h>(h: &'h Header, name: &str) -> Result<&'h TensorEntry> {
    h.tensor(name).map_err(|_| Error::MissingTensor(name.into()))
}

fn shape2(t: &TensorEntry) -> Result<(usize, usize)> {
    match t.shape.as_slice() {
        [a, b] => Ok((*a, *b)),
        other => Err(Error::Shape(format!("{}: expected a matrix, got {other:?}", t.name))),
    }
}

fn blob_bytes<'b>(bytes: &'b [u8], t: &TensorEntry, which: &str) -> Result<&'b [u8]> {
    let b = t.blob(which).map_err(|e| Error::Format(format!("{e}")))?;
    let lo = b.offset as usize;
    let hi = lo
        .checked_add(b.len as usize)
        .ok_or_else(|| Error::Format(format!("{}/{which}: range overflows", t.name)))?;
    bytes
        .get(lo..hi)
        .ok_or_else(|| Error::Format(format!("{}/{which}: range past end of file", t.name)))
}

/// Reinterprets a byte range as `u32`s, copying only if the mapping is not
/// aligned.
///
/// `.wstone` payloads are 256-byte aligned and `mmap` returns a page-aligned
/// base, so the borrow path is what actually runs. The copy exists so a future
/// non-mmap reader cannot turn into undefined behaviour.
fn as_u32(bytes: &[u8]) -> std::borrow::Cow<'_, [u32]> {
    // SAFETY: `u32` has no invalid bit patterns and no padding, and `align_to`
    // itself guarantees the middle slice is correctly aligned and in bounds. We
    // only use it when nothing was split off the front, i.e. the mapping was
    // already 4-byte aligned.
    let (head, mid, _) = unsafe { bytes.align_to::<u32>() };
    if head.is_empty() {
        std::borrow::Cow::Borrowed(mid)
    } else {
        std::borrow::Cow::Owned(
            bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
        )
    }
}

fn as_u16(bytes: &[u8]) -> std::borrow::Cow<'_, [u16]> {
    // SAFETY: as above, for `u16`.
    let (head, mid, _) = unsafe { bytes.align_to::<u16>() };
    if head.is_empty() {
        std::borrow::Cow::Borrowed(mid)
    } else {
        std::borrow::Cow::Owned(
            bytes.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect(),
        )
    }
}

fn load_int4(bytes: &[u8], t: &TensorEntry) -> Result<QuantLinear> {
    let (out_f, in_f) = shape2(t)?;
    let qw = as_u32(blob_bytes(bytes, t, "qw")?);
    let sz = as_u32(blob_bytes(bytes, t, "sz")?);
    Ok(QuantLinear::from_packed(&qw, &sz, in_f, out_f)?)
}

fn load_linear(h: &Header, bytes: &[u8], name: &str) -> Result<DeviceLinear> {
    let t = tensor(h, name)?;
    let (out_f, in_f) = shape2(t)?;
    match t.kind {
        TensorKind::Int4G128 => Ok(DeviceLinear::Int4(load_int4(bytes, t)?)),
        TensorKind::Fp16 => {
            let w = as_u16(blob_bytes(bytes, t, "data")?);
            if w.len() != in_f * out_f {
                return Err(Error::Shape(format!(
                    "{name}: blob holds {} elements, shape says {}",
                    w.len(),
                    in_f * out_f
                )));
            }
            Ok(DeviceLinear::Fp16 {
                w: DeviceBuffer::from_slice(&w)?,
                in_features: in_f,
                out_features: out_f,
            })
        }
        TensorKind::Fp32 => Err(Error::Format(format!(
            "{name}: fp32 tensors are not an execution format; re-run convert"
        ))),
    }
}

fn load_fp16(h: &Header, bytes: &[u8], name: &str) -> Result<DeviceBuffer<u16>> {
    let t = tensor(h, name)?;
    if t.kind != TensorKind::Fp16 {
        return Err(Error::Format(format!("{name}: expected an fp16 tensor, found {:?}", t.kind)));
    }
    let data = as_u16(blob_bytes(bytes, t, "data")?);
    if data.len() != t.numel() {
        return Err(Error::Shape(format!(
            "{name}: blob holds {} elements, shape {:?} says {}",
            data.len(),
            t.shape,
            t.numel()
        )));
    }
    Ok(DeviceBuffer::from_slice(&data)?)
}

fn load_fp16_opt(h: &Header, bytes: &[u8], name: &str) -> Result<Option<DeviceBuffer<u16>>> {
    if h.tensor(name).is_err() {
        return Ok(None);
    }
    load_fp16(h, bytes, name).map(Some)
}

/// Concatenates several matrices along the output dimension and uploads one.
///
/// All of them share the input width by construction — they are projections of
/// the same residual stream — so this is a plain row append. For int4-g128 that
/// means appending the packed nibble rows and the per-row scale rows; nothing is
/// requantized and the result is bit-identical to running them separately.
fn fuse_linears(h: &Header, bytes: &[u8], names: &[String]) -> Result<DeviceLinear> {
    let entries: Vec<&TensorEntry> =
        names.iter().map(|n| tensor(h, n)).collect::<Result<Vec<_>>>()?;

    let (_, in_f) = shape2(entries[0])?;
    let kind = entries[0].kind;
    let mut total_out = 0usize;

    for (e, name) in entries.iter().zip(names) {
        let (o, i) = shape2(e)?;
        if i != in_f {
            return Err(Error::Shape(format!(
                "{name}: input width {i} does not match {in_f}; these cannot be fused"
            )));
        }
        if e.kind != kind {
            return Err(Error::Format(format!(
                "{name}: stored as {:?} but its siblings are {kind:?}; \
                 re-convert so the fused projections share a precision",
                e.kind
            )));
        }
        total_out += o;
    }

    match kind {
        TensorKind::Int4G128 => {
            let mut qw = Vec::with_capacity(total_out * in_f / 8);
            let mut sz = Vec::with_capacity(total_out * in_f / whetstone_kernels::GROUP);
            for e in &entries {
                qw.extend_from_slice(&as_u32(blob_bytes(bytes, e, "qw")?));
                sz.extend_from_slice(&as_u32(blob_bytes(bytes, e, "sz")?));
            }
            Ok(DeviceLinear::Int4(QuantLinear::from_packed(&qw, &sz, in_f, total_out)?))
        }
        TensorKind::Fp16 => {
            let mut w = Vec::with_capacity(total_out * in_f);
            for e in &entries {
                w.extend_from_slice(&as_u16(blob_bytes(bytes, e, "data")?));
            }
            if w.len() != total_out * in_f {
                return Err(Error::Shape(format!(
                    "fused matrix holds {} elements, shape says {}",
                    w.len(),
                    total_out * in_f
                )));
            }
            Ok(DeviceLinear::Fp16 {
                w: DeviceBuffer::from_slice(&w)?,
                in_features: in_f,
                out_features: total_out,
            })
        }
        TensorKind::Fp32 => Err(Error::Format(
            "fp32 tensors are not an execution format; re-run convert".into(),
        )),
    }
}

/// Concatenates biases to match a fused projection, zero-filling any that the
/// architecture omits.
///
/// Qwen2 has biases on q/k/v; Qwen3 has none. A partially-present set would
/// otherwise silently shift the fused vector, so the missing ones are written as
/// explicit zeros at their correct offsets.
fn fuse_biases(
    h: &Header,
    bytes: &[u8],
    names: &[String],
    widths: &[usize],
) -> Result<Option<DeviceBuffer<u16>>> {
    if names.iter().all(|n| h.tensor(n).is_err()) {
        return Ok(None);
    }
    let mut out: Vec<u16> = Vec::with_capacity(widths.iter().sum());
    for (name, &w) in names.iter().zip(widths) {
        match load_fp16_opt(h, bytes, name)? {
            Some(b) => {
                if b.len() != w {
                    return Err(Error::Shape(format!(
                        "{name}: {} elements, expected {w}",
                        b.len()
                    )));
                }
                out.extend_from_slice(&b.to_vec()?);
            }
            None => out.extend(std::iter::repeat(0u16).take(w)), // f16 zero
        }
    }
    Ok(Some(DeviceBuffer::from_slice(&out)?))
}
