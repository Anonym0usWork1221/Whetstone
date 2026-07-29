//! Weight quantization and bit packing.
//!
//! # Which format, and how that was decided
//!
//! [`hier`] holds the production format: int4 with group 32 and hierarchical
//! scale metadata. This module keeps the group-128 format it replaced, because
//! an A/B against the thing you are improving on should be one flag.
//!
//! Measured on Qwen2.5-0.5B, wikitext-2, 20x2048 windows, the 168 transformer
//! projections quantized and the head left in fp16 — perplexity delta against
//! fp16, which reads 13.8182 on this stream:
//!
//! | format | bits/weight | Δ perplexity |
//! |---|---|---|
//! | int4 g128, fp16 scale + fp16 zero | 4.250 | +2.730 |
//! | the same, with llama.cpp's complete k-quant fit | 4.250 | +2.575 |
//! | int4 g64, fp16 scale + fp16 zero | 4.500 | +1.771 |
//! | **int4 g32, hierarchical (see [`hier`])** | **4.277** | **+1.575** |
//!
//! **Group size is worth about six times what the fitting algorithm is worth**,
//! and the only reason the group-128 format could not have it is that an fp16
//! scale plus an fp16 zero per group of 32 costs 1.0 bits/weight of metadata
//! against group 128's 0.25.
//!
//! # Two things this module reports that are not quality measures
//!
//! [`relative_error`] is a smoke test for a broken packer, not a quality gate.
//! Measured on this model: a clip search that *lowers* mean weight error from
//! 0.1102 to 0.1067 *raises* perplexity by 0.50, and GPTQ *raises* weight error
//! to 0.1416 while *lowering* perplexity by 1.73. Weight error and output error
//! are different objectives and they do not reliably move together.
//!
//! Top-1 agreement on a handful of prompts is worse still. An earlier version of
//! this file reported int4-g128 at "100% top-1 agreement, costs nothing"; over
//! 40,940 predictions the same format costs +2.73 perplexity. The argmax is
//! stable long after the distribution has moved.
//!
//! Use `whetstone ppl`.

#![deny(missing_docs)]

pub mod cpu;
pub mod format;
pub mod hier;

use half::f16;
use rayon::prelude::*;

pub use format::{Header, TensorEntry, TensorKind, Writer};
pub use hier::{
    dequantize_int4_hier, quantize_int4_hier, quantize_int4_hier_measured, PackedInt4Hier, HGROUP,
};

/// Running `‖w − ŵ‖²` and `‖w‖²`, accumulated while a matrix is packed.
///
/// Exists so weight error costs nothing. The obvious implementation —
/// dequantize the packed matrix and call [`relative_error`] on the pair — needs
/// a full f32 reconstruction of every tensor and two extra passes over it. Over
/// a 7 B checkpoint that is roughly 15 GB of transient allocation spent
/// recomputing a number the packer already held: at the instant `q` is chosen,
/// `q·scale + min` *is* the reconstruction.
///
/// Splittable and mergeable, so a parallel pack reduces over it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ErrorAccum {
    num: f64,
    den: f64,
}

impl ErrorAccum {
    /// Accumulates one weight and the value the kernel will reconstruct for it.
    #[inline]
    pub fn push(&mut self, original: f32, reconstructed: f32) {
        let d = (original - reconstructed) as f64;
        self.num += d * d;
        self.den += (original as f64) * (original as f64);
    }

    /// Combines two partial accumulations. Associative, so the reduction order a
    /// work-stealing pool happens to pick does not change the result beyond f64
    /// rounding.
    pub fn merge(a: Self, b: Self) -> Self {
        Self { num: a.num + b.num, den: a.den + b.den }
    }

    /// Relative Frobenius error `‖w − ŵ‖ / ‖w‖`, or 0 for an all-zero matrix.
    pub fn relative(&self) -> f64 {
        if self.den == 0.0 {
            0.0
        } else {
            (self.num / self.den).sqrt()
        }
    }
}

/// Errors from quantization and the `.wstone` container.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A weight matrix had a shape the packer cannot represent.
    #[error("shape: {0}")]
    Shape(String),
    /// A `.wstone` file was malformed, truncated, or failed a checksum.
    #[error("format: {0}")]
    Format(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Weights sharing one scale/zero pair.
///
/// 128 is the standard choice: small enough to track local weight statistics,
/// large enough that the fp16 scale and zero cost only 0.25 bits/weight.
pub const GROUP: usize = 128;

/// Quantization levels for 4-bit.
const QMAX: f32 = 15.0;

/// An int4 group-quantized weight matrix, packed for the decode GEMV.
///
/// Layout matches `whetstone-kernels/cuda/gemv_int4.cu`:
/// - `qw`: `[out_features][in_features/8]` `u32`, nibble `i` in bits `4i..4i+3`
/// - `sz`: `[out_features][in_features/GROUP]` `u32`, low half `scale`, high half `zero`
#[derive(Debug, Clone)]
pub struct PackedInt4 {
    /// Packed nibbles, eight per word.
    pub qw: Vec<u32>,
    /// Per-group scale and zero, as two f16s packed into a u32.
    pub sz: Vec<u32>,
    /// Input width.
    pub in_features: usize,
    /// Output width.
    pub out_features: usize,
}

impl PackedInt4 {
    /// Bytes the GEMV reads per invocation, including scale metadata.
    pub fn bytes(&self) -> usize {
        self.qw.len() * 4 + self.sz.len() * 4
    }

    /// Effective bits per weight, counting scale/zero overhead.
    ///
    /// Reports ~4.25, not 4.0. Quoting the nominal width while ignoring
    /// per-group metadata understates bandwidth by 5-10%, and bandwidth is
    /// exactly what sets decode speed.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes() as f64 * 8.0 / (self.in_features * self.out_features) as f64
    }
}

/// Quantizes a row-major `[out_features][in_features]` matrix to int4-g128.
///
/// Asymmetric affine quantization, per group of [`GROUP`] along the input
/// dimension (the dimension a GEMV reduces over, so one scale applies across a
/// contiguous run of the accumulation):
///
/// ```text
/// s = (max - min) / 15
/// z = round(-min / s)
/// q = clamp(round(w/s) + z, 0, 15)
/// w' = (q - z) * s
/// ```
///
/// `s` and `z` are rounded to f16 *before* quantizing, so the values encoded
/// here are exactly the ones the kernel reconstructs. Computing `q` against
/// full-precision scales and then storing rounded ones introduces an error the
/// dequantizer cannot undo.
pub fn quantize_int4_g128(w: &[f32], in_features: usize, out_features: usize) -> Result<PackedInt4> {
    quantize_int4_g128_measured(w, in_features, out_features).map(|(p, _)| p)
}

/// [`quantize_int4_g128`], plus the relative Frobenius weight error accumulated
/// during packing. See [`ErrorAccum`] for why it is not measured afterwards.
pub fn quantize_int4_g128_measured(
    w: &[f32],
    in_features: usize,
    out_features: usize,
) -> Result<(PackedInt4, f64)> {
    if in_features % GROUP != 0 {
        return Err(Error::Shape(format!(
            "in_features {in_features} must be a multiple of {GROUP}"
        )));
    }
    if w.len() != in_features * out_features {
        return Err(Error::Shape(format!(
            "weight slice has {} elements, expected {}",
            w.len(),
            in_features * out_features
        )));
    }

    let groups = in_features / GROUP;
    let words = in_features / 8;
    let mut qw = vec![0u32; out_features * words];
    let mut sz = vec![0u32; out_features * groups];

    // Rows own disjoint output slices, so the fan-out is synchronisation-free.
    let acc = qw
        .par_chunks_mut(words)
        .zip(sz.par_chunks_mut(groups))
        .zip(w.par_chunks(in_features))
        .map(|((qw_row, sz_row), row)| pack_row_g128(row, groups, qw_row, sz_row))
        .reduce(ErrorAccum::default, ErrorAccum::merge);

    Ok((PackedInt4 { qw, sz, in_features, out_features }, acc.relative()))
}

#[inline(always)]
fn pack_row_g128_body(row: &[f32], groups: usize, qw_row: &mut [u32], sz_row: &mut [u32]) -> ErrorAccum {
    let mut acc = ErrorAccum::default();
    for (g, sz) in sz_row.iter_mut().enumerate().take(groups) {
        let slice = &row[g * GROUP..(g + 1) * GROUP];

        let lo = slice.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut scale = (hi - lo) / QMAX;
        if !scale.is_finite() || scale == 0.0 {
            scale = 1.0; // constant group: every value maps to the zero point
        }
        let zero = (-lo / scale).round();

        // Round the metadata first: these are the values the kernel sees.
        let sh = f16::from_f32(scale);
        let zh = f16::from_f32(zero);
        *sz = (sh.to_bits() as u32) | ((zh.to_bits() as u32) << 16);

        let scale_r = sh.to_f32();
        let zero_r = zh.to_f32();

        for (i, &v) in slice.iter().enumerate() {
            let q = ((v / scale_r).round() + zero_r).clamp(0.0, QMAX);
            let col = g * GROUP + i;
            qw_row[col / 8] |= (q as u32) << (4 * (col % 8));
            acc.push(v, (q - zero_r) * scale_r);
        }
    }
    acc
}

crate::isa_dispatch! {
    body  = pack_row_g128_body,
    avx2  = pack_row_g128_avx2,
    sse41 = pack_row_g128_sse41;
    /// [`pack_row_g128_body`], compiled per instruction set and selected at run
    /// time. See [`cpu`] for why the baseline is worth escaping.
    fn pack_row_g128(row: &[f32], groups: usize, qw_row: &mut [u32], sz_row: &mut [u32]) -> ErrorAccum;
}

/// Reconstructs the weights a packed matrix represents.
///
/// This is what the GEMV computes with, so it is the correct reference for
/// differential-testing a kernel: a disagreement against *this* is a kernel bug,
/// whereas a disagreement against the original weights is quantization error.
pub fn dequantize_int4_g128(p: &PackedInt4) -> Vec<f32> {
    let groups = p.in_features / GROUP;
    let words = p.in_features / 8;
    let mut out = vec![0f32; p.in_features * p.out_features];

    out.par_chunks_mut(p.in_features)
        .zip(p.qw.par_chunks(words))
        .zip(p.sz.par_chunks(groups))
        .for_each(|((dst, qw_row), sz_row)| {
            for (g, &packed) in sz_row.iter().enumerate().take(groups) {
                let scale = f16::from_bits(packed as u16).to_f32();
                let zero = f16::from_bits((packed >> 16) as u16).to_f32();
                for i in 0..GROUP {
                    let col = g * GROUP + i;
                    let q = ((qw_row[col / 8] >> (4 * (col % 8))) & 0xF) as f32;
                    dst[col] = (q - zero) * scale;
                }
            }
        });
    out
}

/// Relative Frobenius error `||a - b|| / ||a||`.
///
/// The standard summary of quantization damage to a weight matrix. Note it
/// measures *weight* error, which is not the objective that matters: two
/// quantizers with equal weight error can differ substantially in output error.
pub fn relative_error(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*x as f64) * (*x as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(n: usize) -> Vec<f32> {
        // Deterministic, roughly zero-mean, with the heavy-ish tails real
        // weight matrices have.
        (0..n)
            .map(|i| {
                let a = ((i * 2_654_435_761usize) % 10_000) as f32 / 10_000.0 - 0.5;
                let b = ((i * 40_503usize) % 977) as f32 / 977.0 - 0.5;
                a * 0.15 + b * b * b * 0.5
            })
            .collect()
    }

    #[test]
    fn round_trip_is_within_a_quantization_step() {
        let (in_f, out_f) = (256usize, 16usize);
        let w = weights(in_f * out_f);
        let p = quantize_int4_g128(&w, in_f, out_f).unwrap();
        let d = dequantize_int4_g128(&p);

        // Every reconstructed value must be within half a step of the original,
        // plus f16 slack on the scale itself.
        for g in 0..(in_f / GROUP) {
            for r in 0..out_f {
                let base = r * in_f + g * GROUP;
                let slice = &w[base..base + GROUP];
                let lo = slice.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step = (hi - lo) / QMAX;

                for i in 0..GROUP {
                    let err = (w[base + i] - d[base + i]).abs();
                    assert!(
                        err <= step * 0.55 + 1e-4,
                        "element {} off by {err} (step {step})",
                        base + i
                    );
                }
            }
        }
    }

    #[test]
    fn packing_layout_is_eight_nibbles_per_word() {
        let (in_f, out_f) = (128usize, 2usize);
        let w = vec![0.0f32; in_f * out_f];
        let p = quantize_int4_g128(&w, in_f, out_f).unwrap();

        assert_eq!(p.qw.len(), out_f * in_f / 8);
        assert_eq!(p.sz.len(), out_f * in_f / GROUP);
        assert_eq!(p.bytes(), p.qw.len() * 4 + p.sz.len() * 4);

        // 4 bits per weight plus one fp16 scale and one fp16 zero per 128.
        let expected = 4.0 + 32.0 / GROUP as f64;
        assert!((p.bits_per_weight() - expected).abs() < 1e-9);
    }

    #[test]
    fn error_is_far_below_the_next_format_down() {
        // The measured curve: int4-g128 lands near 0.11 relative error on real
        // weights, int3 near 0.23. This guards the packer against a regression
        // that would silently move it toward int3 territory.
        let (in_f, out_f) = (896usize, 64usize);
        let w = weights(in_f * out_f);
        let p = quantize_int4_g128(&w, in_f, out_f).unwrap();
        let e = relative_error(&w, &dequantize_int4_g128(&p));
        assert!(e < 0.15, "int4 relative error {e} is higher than expected");
    }

    #[test]
    fn constant_group_does_not_divide_by_zero() {
        let (in_f, out_f) = (128usize, 1usize);
        let w = vec![0.25f32; in_f * out_f];
        let p = quantize_int4_g128(&w, in_f, out_f).unwrap();
        let d = dequantize_int4_g128(&p);
        assert!(d.iter().all(|v| v.is_finite()), "constant group produced non-finite output");
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(quantize_int4_g128(&[0.0; 100], 100, 1).is_err(), "100 is not a multiple of 128");
        assert!(quantize_int4_g128(&[0.0; 10], 128, 1).is_err(), "slice too short for shape");
    }

    /// The fused metric replaced a reconstruct-then-compare pass. If the two
    /// ever disagree, the packer and the dequantizer have drifted apart — which
    /// is exactly the class of bug that would otherwise surface as unexplained
    /// perplexity, because the file would decode to something the converter
    /// never scored.
    #[test]
    fn fused_error_equals_reconstruct_then_measure() {
        let (in_f, out_f) = (896usize, 97usize);
        let w = weights(in_f * out_f);

        let (p, fused) = quantize_int4_g128_measured(&w, in_f, out_f).unwrap();
        let post = relative_error(&w, &dequantize_int4_g128(&p));
        assert!(
            (fused - post).abs() < 1e-9,
            "g128 fused {fused} vs reconstructed {post}"
        );

        let (h, fused_h) = quantize_int4_hier_measured(&w, in_f, out_f).unwrap();
        let post_h = relative_error(&w, &dequantize_int4_hier(&h));
        assert!(
            (fused_h - post_h).abs() < 1e-9,
            "hier fused {fused_h} vs reconstructed {post_h}"
        );
    }

    /// Rows are packed by a work-stealing pool, so the bytes must not depend on
    /// how the work happened to be split. A shape with a prime row count and a
    /// non-power-of-two group count is the awkward case.
    #[test]
    fn parallel_pack_is_deterministic() {
        let (in_f, out_f) = (1216usize, 53usize);
        let w = weights(in_f * out_f);
        let a = quantize_int4_hier(&w, in_f, out_f).unwrap();
        let b = quantize_int4_hier(&w, in_f, out_f).unwrap();
        assert_eq!(a.qw, b.qw);
        assert_eq!(a.si, b.si);
        assert_eq!(a.sb, b.sb);
    }
}
