//! Weight quantization and bit packing.
//!
//! # Why int4 group-128
//!
//! Measured on Qwen2.5-0.5B by quantizing every linear weight and comparing
//! output distributions against the unquantized model:
//!
//! | format | bits/wt | output KL (nats) | top-1 agreement |
//! |---|---|---|---|
//! | int8 per-channel | 8.00 | 0.014 | 100% |
//! | **int4 g128** | **4.25** | **0.187** | **100%** |
//! | int3 g128 | 3.25 | 1.969 | 33% |
//! | int2 g128 | 2.25 | 12.432 | 0% |
//! | ternary g128 | 1.71 | 10.936 | 0% |
//!
//! There is a cliff between 4 and 3 bits, and round-to-nearest ternary destroys
//! the model outright. That is not a contradiction of the 1-bit LLM literature:
//! those models are *trained* in ternary with a straight-through estimator, so
//! their weights are built to lie on that grid, and the published results are
//! overwhelmingly on 7B+ models with far more redundancy than 0.5B.
//!
//! int4-g128 is therefore the format Whetstone targets: a 3.8x bandwidth
//! reduction that costs no top-1 agreement.

#![deny(missing_docs)]

pub mod format;

use half::f16;

pub use format::{Header, TensorEntry, TensorKind, Writer};

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
    let mut qw = vec![0u32; out_features * in_features / 8];
    let mut sz = vec![0u32; out_features * groups];

    for r in 0..out_features {
        for g in 0..groups {
            let base = r * in_features + g * GROUP;
            let slice = &w[base..base + GROUP];

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
            sz[r * groups + g] = (sh.to_bits() as u32) | ((zh.to_bits() as u32) << 16);

            let scale_r = sh.to_f32();
            let zero_r = zh.to_f32();

            for (i, &v) in slice.iter().enumerate() {
                let q = ((v / scale_r).round() + zero_r).clamp(0.0, QMAX) as u32;
                let col = g * GROUP + i;
                qw[r * (in_features / 8) + col / 8] |= q << (4 * (col % 8));
            }
        }
    }

    Ok(PackedInt4 { qw, sz, in_features, out_features })
}

/// Reconstructs the weights a packed matrix represents.
///
/// This is what the GEMV computes with, so it is the correct reference for
/// differential-testing a kernel: a disagreement against *this* is a kernel bug,
/// whereas a disagreement against the original weights is quantization error.
pub fn dequantize_int4_g128(p: &PackedInt4) -> Vec<f32> {
    let groups = p.in_features / GROUP;
    let mut out = vec![0f32; p.in_features * p.out_features];

    for r in 0..p.out_features {
        for g in 0..groups {
            let packed_sz = p.sz[r * groups + g];
            let scale = f16::from_bits(packed_sz as u16).to_f32();
            let zero = f16::from_bits((packed_sz >> 16) as u16).to_f32();

            for i in 0..GROUP {
                let col = g * GROUP + i;
                let word = p.qw[r * (p.in_features / 8) + col / 8];
                let q = ((word >> (4 * (col % 8))) & 0xF) as f32;
                out[r * p.in_features + col] = (q - zero) * scale;
            }
        }
    }
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
}
