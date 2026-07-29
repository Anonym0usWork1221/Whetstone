//! The multi-token pass: one weight read serving `n` tokens.
//!
//! Every buffer here is **token-major** `[n][dim]`, so a token's activations are
//! contiguous and each kernel indexes by a plain stride. The cache position of
//! the chunk's first token is `pos0`, passed by value rather than through a
//! [`DeviceCursor`](crate::decode::DeviceCursor): a chunk pass changes width
//! every round under speculative decoding, so there is nothing for a CUDA graph
//! to capture and the simpler signature wins.
//!
//! See `cuda/chunk_gemm.cu` for why this path exists at all — at batch 1 the
//! GEMV is bandwidth bound and cheaper arithmetic buys nothing, so using a weight
//! more than once is the only lever that moves the roofline.

use crate::decode::{KvCache, RopeTable};
use crate::ffi;
use crate::{check, DeviceBuffer, Error, Result};

/// The largest token count one weight pass covers. Wider chunks are sliced by
/// the kernel and re-read the weights once per slice.
pub fn max_tokens() -> usize {
    // SAFETY: reads a compile-time constant out of the CUDA module.
    (unsafe { ffi::wst_chunk_max_tokens() }).max(1) as usize
}

/// Validates that `x`/`y` hold *at least* `n` rows of the stated widths.
///
/// At least, not exactly: the engine allocates chunk scratch once at the maximum
/// width and then runs passes narrower than that — a speculative round accepts a
/// variable number of tokens, and the last chunk of a prompt is whatever is left
/// over. Demanding an exact match would mean reallocating per pass.
fn check_rows(
    what: &str,
    x_len: usize,
    y_len: usize,
    in_f: usize,
    out_f: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Err(Error::Shape(format!("{what}: n must be positive")));
    }
    if x_len < n * in_f {
        return Err(Error::Shape(format!(
            "{what}: x has {x_len} elements, need {n}*{in_f}"
        )));
    }
    if y_len < n * out_f {
        return Err(Error::Shape(format!(
            "{what}: y has {y_len} elements, need {n}*{out_f}"
        )));
    }
    Ok(())
}

/// `y[j] = W x[j] + b` for every `j < n`, or `+=` when `accumulate`.
#[allow(clippy::too_many_arguments)] // mirrors the C ABI shape one for one
pub fn gemm_fp16(
    w: &DeviceBuffer<u16>,
    x: &DeviceBuffer<u16>,
    bias: Option<&DeviceBuffer<u16>>,
    y: &mut DeviceBuffer<f32>,
    in_features: usize,
    out_features: usize,
    n: usize,
    accumulate: bool,
) -> Result<()> {
    check_rows("gemm_fp16", x.len(), y.len(), in_features, out_features, n)?;
    if w.len() != in_features * out_features {
        return Err(Error::Shape(format!(
            "gemm_fp16: weights have {} elements, expected {in_features}*{out_features}",
            w.len()
        )));
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(Error::Shape(format!(
                "gemm_fp16: bias has {} elements, expected {out_features}",
                b.len()
            )));
        }
    }
    let bias_ptr = bias.map_or(std::ptr::null(), DeviceBuffer::as_ptr);
    // SAFETY: all three shapes are checked above against the buffers' real
    // lengths, and the bias pointer is null exactly when no bias was supplied.
    check(unsafe {
        ffi::wst_gemm_fp16(
            w.as_ptr(),
            x.as_ptr(),
            bias_ptr,
            y.as_mut_ptr(),
            in_features as i32,
            out_features as i32,
            n as i32,
            i32::from(accumulate),
        )
    })
}

/// RMSNorm over each of `n` rows of `[n][dim]`, narrowing to f16.
pub fn rmsnorm(
    x: &DeviceBuffer<f32>,
    w: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    dim: usize,
    n: usize,
) -> Result<()> {
    rmsnorm_eps(x, w, out, dim, n, 1e-6)
}

/// RMSNorm with an explicit epsilon.
pub fn rmsnorm_eps(
    x: &DeviceBuffer<f32>,
    w: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    dim: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_rows("rmsnorm_chunk", x.len(), out.len(), dim, dim, n)?;
    if w.len() != dim {
        return Err(Error::Shape(format!(
            "rmsnorm_chunk: weight has {} elements, expected {dim}",
            w.len()
        )));
    }
    // SAFETY: shapes validated above; the kernel writes only out[0..n*dim].
    check(unsafe {
        ffi::wst_rmsnorm_chunk(
            x.as_ptr(),
            w.as_ptr(),
            out.as_mut_ptr(),
            dim as i32,
            n as i32,
            eps,
        )
    })
}

/// Rotary embedding and KV append for `n` consecutive positions from `pos0`.
pub fn rope_cache(
    qkv: &mut DeviceBuffer<f32>,
    cache: &mut KvCache,
    table: &RopeTable,
    n_q: usize,
    pos0: usize,
    n: usize,
    qk_norm: Option<crate::decode::QkNorm<'_>>,
) -> Result<()> {
    let hd = cache.head_dim;
    let stride = (n_q + 2 * cache.n_kv) * hd;
    if qkv.len() < n * stride {
        return Err(Error::Shape(format!(
            "rope_cache_chunk: qkv[{}] needs {n} * {stride}",
            qkv.len()
        )));
    }
    if table.half_dim != hd / 2 || table.max_seq < cache.max_seq {
        return Err(Error::Shape(
            "rope_cache_chunk: rotary table does not match the cache geometry".into(),
        ));
    }
    if pos0 + n > cache.max_seq {
        return Err(Error::Shape(format!(
            "rope_cache_chunk: chunk [{pos0}, {}) runs past the {} token cache",
            pos0 + n,
            cache.max_seq
        )));
    }
    if let Some(qn) = &qk_norm {
        if qn.q.len() != hd || qn.k.len() != hd {
            return Err(Error::Shape(format!(
                "rope_cache_chunk: gains are q[{}], k[{}]; both must be head_dim {hd}",
                qn.q.len(),
                qn.k.len()
            )));
        }
    }
    let (qw, kw, eps) = crate::decode::norm_ptrs(&qk_norm);

    // SAFETY: geometry, cache capacity, the chunk's position range and the two
    // gain vectors are all validated above against the dimensions the kernel
    // indexes. Null gain pointers select the no-norm path.
    check(unsafe {
        ffi::wst_rope_cache_chunk(
            qkv.as_mut_ptr(),
            cache.k.as_mut_ptr(),
            cache.v.as_mut_ptr(),
            table.cos.as_ptr(),
            table.sin.as_ptr(),
            n_q as i32,
            cache.n_kv as i32,
            hd as i32,
            pos0 as i32,
            n as i32,
            cache.max_seq as i32,
            qw,
            kw,
            eps,
        )
    })
}

/// Causal GQA attention for `n` queries at positions `pos0 .. pos0+n`.
///
/// Query `j` attends to cache entries `0 ..= pos0+j`, which is the causal mask
/// expressed as a loop bound rather than as arithmetic on scores.
pub fn attn(
    qkv: &DeviceBuffer<f32>,
    cache: &KvCache,
    out: &mut DeviceBuffer<u16>,
    n_q: usize,
    pos0: usize,
    n: usize,
) -> Result<()> {
    let hd = cache.head_dim;
    let stride = (n_q + 2 * cache.n_kv) * hd;
    if qkv.len() < n * stride {
        return Err(Error::Shape(format!(
            "attn_chunk: qkv[{}] needs {n} * {stride}",
            qkv.len()
        )));
    }
    if out.len() < n * n_q * hd {
        return Err(Error::Shape(format!(
            "attn_chunk: out[{}] needs {n} * {n_q} * {hd}",
            out.len()
        )));
    }
    if pos0 + n > cache.max_seq {
        return Err(Error::Shape(format!(
            "attn_chunk: chunk [{pos0}, {}) runs past the {} token cache",
            pos0 + n,
            cache.max_seq
        )));
    }
    let scale = 1.0f32 / (hd as f32).sqrt();
    // SAFETY: shapes and the position range are validated above; the kernel
    // reads only cache entries below pos0+n, which rope_cache has just written.
    check(unsafe {
        ffi::wst_attn_chunk(
            qkv.as_ptr(),
            cache.k.as_ptr(),
            cache.v.as_ptr(),
            out.as_mut_ptr(),
            n_q as i32,
            cache.n_kv as i32,
            hd as i32,
            pos0 as i32,
            n as i32,
            cache.max_seq as i32,
            scale,
        )
    })
}

/// `out[j] = f16( silu(gate[j]) * up[j] )` for `n` rows.
pub fn swiglu(
    gate_up: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<u16>,
    inter: usize,
    n: usize,
) -> Result<()> {
    if gate_up.len() < 2 * n * inter {
        return Err(Error::Shape(format!(
            "swiglu_chunk: gate_up[{}] needs 2 * {n} * {inter}",
            gate_up.len()
        )));
    }
    if out.len() < n * inter {
        return Err(Error::Shape(format!(
            "swiglu_chunk: out[{}] needs {n} * {inter}",
            out.len()
        )));
    }
    // SAFETY: both shapes validated above against what the kernel indexes.
    check(unsafe {
        ffi::wst_swiglu_chunk(gate_up.as_ptr(), out.as_mut_ptr(), inter as i32, n as i32)
    })
}

/// Validates an embedding gather's shapes. Shared by the three storage formats.
fn check_gather(
    what: &str,
    tokens: usize,
    out: usize,
    hidden: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Err(Error::Shape(format!("{what}: n must be positive")));
    }
    if tokens < n {
        return Err(Error::Shape(format!(
            "{what}: token buffer holds {tokens}, need {n}"
        )));
    }
    if out < n * hidden {
        return Err(Error::Shape(format!(
            "{what}: out[{out}] needs {n} * {hidden}"
        )));
    }
    Ok(())
}

/// Gathers `n` rows from a dense f16 embedding table.
pub fn embed_fp16(
    table: &DeviceBuffer<u16>,
    tokens: &DeviceBuffer<i32>,
    out: &mut DeviceBuffer<f32>,
    hidden: usize,
    vocab: usize,
    n: usize,
) -> Result<()> {
    check_gather("embed_fp16_chunk", tokens.len(), out.len(), hidden, n)?;
    // SAFETY: shapes validated above. Token ids are clamped into [0, vocab) by
    // the kernel rather than trusted, because they may come from a draft model.
    check(unsafe {
        ffi::wst_embed_fp16_chunk(
            table.as_ptr(),
            tokens.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            vocab as i32,
            n as i32,
        )
    })
}

/// Gathers `n` rows from an int4 group-128 table.
pub fn embed_int4_g128(
    qw: &DeviceBuffer<u32>,
    sz: &DeviceBuffer<u32>,
    tokens: &DeviceBuffer<i32>,
    out: &mut DeviceBuffer<f32>,
    hidden: usize,
    vocab: usize,
    n: usize,
) -> Result<()> {
    check_gather("embed_int4_g128_chunk", tokens.len(), out.len(), hidden, n)?;
    // SAFETY: as `embed_fp16`; the kernel clamps the row index.
    check(unsafe {
        ffi::wst_embed_int4_g128_chunk(
            qw.as_ptr(),
            sz.as_ptr(),
            tokens.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            vocab as i32,
            n as i32,
        )
    })
}

/// Gathers `n` rows from an int4 hierarchical-scale table.
#[allow(clippy::too_many_arguments)] // three packed arrays plus the shape
pub fn embed_int4_hier(
    qw: &DeviceBuffer<u32>,
    si: &DeviceBuffer<u8>,
    sb: &DeviceBuffer<u32>,
    tokens: &DeviceBuffer<i32>,
    out: &mut DeviceBuffer<f32>,
    hidden: usize,
    vocab: usize,
    n: usize,
) -> Result<()> {
    check_gather("embed_int4_hier_chunk", tokens.len(), out.len(), hidden, n)?;
    // SAFETY: as `embed_fp16`; the kernel clamps the row index.
    check(unsafe {
        ffi::wst_embed_int4_hier_chunk(
            qw.as_ptr(),
            si.as_ptr(),
            sb.as_ptr(),
            tokens.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            vocab as i32,
            n as i32,
        )
    })
}

/// Greedy choice for each of `n` rows of `[n][vocab]` logits.
///
/// Reduced on the device: the alternative is `n * vocab` floats over PCIe, which
/// at n=8 and a 151936 vocabulary is 4.9 MB — 0.85 ms on this machine's Gen3 x8
/// link, or a third of a token.
pub fn argmax(
    logits: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<i32>,
    vocab: usize,
    n: usize,
) -> Result<()> {
    if logits.len() < n * vocab {
        return Err(Error::Shape(format!(
            "argmax_chunk: logits[{}] needs {n} * {vocab}",
            logits.len()
        )));
    }
    if out.len() < n {
        return Err(Error::Shape(format!(
            "argmax_chunk: out holds {}, need {n}",
            out.len()
        )));
    }
    // SAFETY: shapes validated above; the kernel writes out[0..n] only.
    check(unsafe {
        ffi::wst_argmax_chunk(logits.as_ptr(), out.as_mut_ptr(), vocab as i32, n as i32)
    })
}
