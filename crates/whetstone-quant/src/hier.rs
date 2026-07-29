//! int4 with hierarchical scale metadata — group 32 at the price of group 128.
//!
//! # What this replaces and why
//!
//! Measured on Qwen2.5-0.5B, wikitext-2, 20×2048 windows, the 168 transformer
//! projections quantized and the head left in fp16. Perplexity delta against the
//! fp16 model, which reads 13.8182 on this stream:
//!
//! | format | bits/weight | Δ perplexity |
//! |---|---|---|
//! | int4 g128, fp16 scale + fp16 zero (shipped in 0.3.0) | 4.250 | +2.730 |
//! | the same, but with llama.cpp's full k-quant fit | 4.250 | +2.575 |
//! | int4 g64, fp16 scale + fp16 zero | 4.500 | +1.771 |
//! | int4 g32, fp16 scale + fp16 zero | 5.000 | +1.696 |
//! | **this format** | **4.277** | **+1.575** |
//!
//! Two things are worth reading off that table. **Granularity is worth six times
//! what the fitting algorithm is worth** — going from group 128 to group 64 buys
//! 0.96 perplexity, while replacing round-to-nearest with the complete k-quant
//! alternating least-squares fit buys 0.16. And granularity is exactly what the
//! previous format could not afford: an fp16 scale plus an fp16 zero per group of
//! 32 costs 1.0 bits/weight of metadata against group 128's 0.25, which is
//! spending the bandwidth the engine exists to save.
//!
//! # The layout
//!
//! ```text
//! per row     half2 (d, dmin)              32 bits
//! per group   uint8 (ls | lm<<4)            8 bits per 32 weights
//! per weight  nibble q                      4 bits
//!
//! scale = d * ls        min = -dmin * lm        w = q*scale + min
//!
//! bits/weight = 4 + 8/32 + 32/in_features
//!             = 4.286 at in=896,   4.257 at in=4864
//! ```
//!
//! # Two decisions that look arbitrary and are not
//!
//! **`w = q*scale + min`, not `(q - zero)*scale`.** The old form needs
//! `zero = -min/scale`, and rounding that to fp16 after rounding `scale` to fp16
//! rounds the same quantity twice. Storing what was actually fitted removes a
//! conversion. It also removes a constraint nobody had noticed: the production
//! `h2` kernel subtracts `1024 + zero` in fp16, where the mantissa step at 1024
//! is exactly 1 — so a fractional zero point was being **silently rounded to an
//! integer** by the kernel, whatever the file said.
//!
//! **`ls` is clamped to at least 1.** A zero scale index is a representable
//! encoding — every weight in the group reconstructs to `min` — but it forces the
//! quantizer to special-case a division it would otherwise do unconditionally,
//! and a special case that two implementations have to agree on is a bug waiting
//! for a rare tensor. Clamping costs a group whose fitted scale is below `d/2` an
//! over-wide step and buys a kernel with no branch.

use half::f16;
use rayon::prelude::*;

use crate::{Error, ErrorAccum, Result};

/// Weights sharing one `(ls, lm)` pair. 32 is not tunable: it is exactly one
/// `uint4` load, so a lane handles precisely one group per step.
pub const HGROUP: usize = 32;

/// Quantization levels for the weights and for the scale/min indices.
const QMAX: f32 = 15.0;
const SMAX: f32 = 15.0;

/// An int4 hierarchical-scale weight matrix, packed for the decode GEMV.
///
/// Layout matches `whetstone-kernels/cuda/gemv_hier.cu`:
/// - `qw`: `[out_features][in_features/8]` `u32`, nibble `i` in bits `4i..4i+3`
/// - `si`: `[out_features][in_features/32]` `u8`, `ls` low, `lm` high
/// - `sb`: `[out_features]` `u32`, an fp16 `d` low and an fp16 `dmin` high
#[derive(Debug, Clone)]
pub struct PackedInt4Hier {
    /// Packed nibbles, eight per word.
    pub qw: Vec<u32>,
    /// Per-group scale and min indices, four bits each.
    pub si: Vec<u8>,
    /// Per-row `(d, dmin)`, two f16s in a u32.
    pub sb: Vec<u32>,
    /// Input width.
    pub in_features: usize,
    /// Output width.
    pub out_features: usize,
}

impl PackedInt4Hier {
    /// Bytes the GEMV reads per invocation, including all scale metadata.
    pub fn bytes(&self) -> usize {
        self.qw.len() * 4 + self.si.len() + self.sb.len() * 4
    }

    /// Effective bits per weight. Depends on `in_features`, because one fp16
    /// pair is amortised over a whole row.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes() as f64 * 8.0 / (self.in_features * self.out_features) as f64
    }
}

/// Fits `(scale, min)` for one group by llama.cpp's `make_qkx2_quants`.
///
/// Two ideas stacked: sweep `nstep+1` candidate grids that clip slightly harder
/// and slightly softer than the exact range, and for each — having fixed the
/// integer levels — refit `(scale, min)` by **weighted least squares in closed
/// form**. Round-to-nearest has no analogue of the second step: it picks a scale
/// and accepts whatever error the levels then have, where this picks the levels
/// and then the best possible scale for them.
///
/// The importance weight `sqrt(mean(x²)) + |x|` is llama.cpp's. The offset
/// matters — pure `|x|` weighting chases the outliers and abandons the bulk.
#[inline(always)]
fn fit_group(x: &[f32]) -> (f32, f32) {
    const NSTEP: usize = 20;
    const RMIN: f32 = -1.0;
    const RDELTA: f32 = 0.1;

    let n = x.len() as f32;
    let av = (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt();

    let lo = x.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
    let hi = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let rng = hi - lo;
    if !rng.is_finite() || rng <= 0.0 {
        return (0.0, lo.min(0.0));
    }

    // Stack, not heap. This is called once per group of 32 -- 11 M times over a
    // 0.5 B model, 238 M times over a 7 B one -- and a `collect()` here is one
    // malloc/free pair per call for 128 bytes that never outlive the frame.
    let mut wbuf = [0f32; 64];
    for (o, v) in wbuf.iter_mut().zip(x) {
        *o = av + v.abs();
    }
    let w = &wbuf[..x.len()];
    let sum_w: f32 = w.iter().sum();
    let sum_x: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();

    let mut best_scale = rng / QMAX;
    let mut best_min = lo;
    let mut best_err = {
        let s = best_scale;
        x.iter()
            .zip(w)
            .map(|(v, wi)| {
                let q = (((v - lo) / s).round()).clamp(0.0, QMAX);
                let d = q * s + lo - v;
                wi * d * d
            })
            .sum::<f32>()
    };

    for step in 0..=NSTEP {
        let iscale = (QMAX + RMIN + RDELTA * step as f32) / rng;
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        // The levels are needed twice (for the normal equations and for the
        // error), and 32 floats is nothing to keep.
        let mut lvl = [0.0f32; 64];
        for (i, (&v, &wi)) in x.iter().zip(w).enumerate() {
            let l = (iscale * (v - lo)).round().clamp(0.0, QMAX);
            lvl[i] = l;
            sum_l += wi * l;
            sum_l2 += wi * l * l;
            sum_xl += wi * l * v;
        }

        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det <= 0.0 {
            continue;
        }
        let mut scale = (sum_w * sum_xl - sum_x * sum_l) / det;
        let mut mn = (sum_l2 * sum_x - sum_l * sum_xl) / det;
        if mn > 0.0 {
            // llama.cpp refuses a positive minimum and refits without one.
            mn = 0.0;
            scale = if sum_l2 > 0.0 { sum_xl / sum_l2 } else { scale };
        }

        let err: f32 = x
            .iter()
            .zip(w)
            .enumerate()
            .map(|(i, (v, wi))| {
                let d = scale * lvl[i] + mn - v;
                wi * d * d
            })
            .sum();
        if err < best_err {
            best_err = err;
            best_scale = scale;
            best_min = mn;
        }
    }
    (best_scale, best_min)
}

/// Rounds a row's shared scale to f16 while keeping it strictly positive and
/// finite.
///
/// `d = max_scale / 15` is the one number every weight in the row is
/// reconstructed against, and f16 has a narrow range on both ends:
///
/// - a row spanning less than about 1.3e-5 rounds `d` **to zero**, so
///   `s = d * ls` is zero, `q = (w - m) / s` is an infinity or a NaN, and the
///   whole row packs as garbage;
/// - a row spanning more than about 1.5e7 rounds `d` to **infinity**, so every
///   reconstruction is `inf * q + m`.
///
/// The first case is not hypothetical: it is what made `convert` report
/// `mean rel. error NaN` on Qwen2.5-7B. It was invisible on the three smaller
/// models, and invisible on 7 B too until the error metric moved inside the
/// packer — the old reconstruct-afterwards path computed `q * 0 + m`, which is
/// finite and wrong rather than NaN.
///
/// Clamping to the smallest positive subnormal rather than erroring is right
/// because such a row *is* numerically zero next to a typical weight of ~0.02.
/// The point is that it reconstructs as approximately zero instead of as NaN.
#[inline(always)]
fn shared_scale(v: f32) -> f16 {
    if v.is_nan() || v <= 0.0 {
        return f16::from_f32(1.0); // zero, negative, or NaN: nothing to scale
    }
    let h = f16::from_f32(v);
    if h.to_f32() > 0.0 {
        if h.is_finite() {
            h
        } else {
            f16::MAX
        }
    } else {
        f16::from_bits(1) // smallest positive subnormal, 5.96e-8
    }
}

/// Packs one row. Writes into caller-owned, row-local slices so that rows carry
/// no shared state at all and the outer loop can be a `rayon` fan-out.
///
/// `scratch_s` / `scratch_m` are per-thread buffers of length `groups`, reused
/// across rows: at 7B there are ~1.9 M rows and a pair of allocations each would
/// be pure churn.
///
/// Returns the row's contribution to `‖w − ŵ‖²` and `‖w‖²`. Accumulating it here
/// rather than dequantizing afterwards is the difference between one pass and
/// three: the old `report_error` path allocated a full f32 reconstruction of
/// every tensor (15 GB of transient allocation over a 7B convert) purely to
/// subtract it from the input again. The reconstruction is already in hand at
/// the moment `q` is chosen.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn pack_row_hier_body(
    row: &[f32],
    groups: usize,
    scratch_s: &mut [f32],
    scratch_m: &mut [f32],
    qw_row: &mut [u32],
    si_row: &mut [u8],
    sb_row: &mut u32,
) -> ErrorAccum {
    let mut max_scale = 0f32;
    let mut max_min = 0f32;
    for g in 0..groups {
        let (s, m) = fit_group(&row[g * HGROUP..(g + 1) * HGROUP]);
        scratch_s[g] = s;
        scratch_m[g] = m;
        max_scale = max_scale.max(s);
        max_min = max_min.max(-m);
    }

    // The fp16 rounding happens here, before the indices are derived, so the
    // indices are chosen against the value the kernel will actually read.
    let d = shared_scale(if max_scale > 0.0 { max_scale / SMAX } else { 1.0 });
    let dm_pos = max_min > 0.0;
    let dm = shared_scale(if dm_pos { max_min / SMAX } else { 1.0 });
    *sb_row = (d.to_bits() as u32) | ((dm.to_bits() as u32) << 16);

    let df = d.to_f32();
    let dmf = dm.to_f32();

    let mut acc = ErrorAccum::default();
    for (g, si) in si_row.iter_mut().enumerate().take(groups) {
        // ls >= 1: see the module docs. A zero scale index would make every
        // weight in the group reconstruct to `min` and force a special case
        // into both the quantizer and the kernel.
        let ls = (scratch_s[g] / df).round().clamp(1.0, SMAX) as u8;
        let lm = if dm_pos {
            ((-scratch_m[g]).max(0.0) / dmf).round().clamp(0.0, SMAX) as u8
        } else {
            0
        };
        *si = ls | (lm << 4);

        let s = df * ls as f32;
        let m = -dmf * lm as f32;
        for i in 0..HGROUP {
            let col = g * HGROUP + i;
            let v = row[col];
            let q = ((v - m) / s).round().clamp(0.0, QMAX);
            qw_row[col / 8] |= (q as u32) << (4 * (col % 8));
            acc.push(v, q * s + m);
        }
    }
    acc
}

crate::isa_dispatch! {
    body  = pack_row_hier_body,
    avx2  = pack_row_hier_avx2,
    sse41 = pack_row_hier_sse41;
    /// [`pack_row_hier_body`], compiled per instruction set and selected at run
    /// time. The dispatch is per row — thousands of cycles of work — so the
    /// branch is free, and it keeps the packer a single source of truth.
    #[allow(clippy::too_many_arguments)]
    fn pack_row_hier(
        row: &[f32],
        groups: usize,
        scratch_s: &mut [f32],
        scratch_m: &mut [f32],
        qw_row: &mut [u32],
        si_row: &mut [u8],
        sb_row: &mut u32,
    ) -> ErrorAccum;
}

/// Quantizes a row-major `[out_features][in_features]` matrix to int4 with
/// hierarchical scales.
///
/// Per row: fit every group of 32, normalise the fitted scales and minima
/// against one fp16 pair, then **re-assign the levels against the quantized
/// parameters**. That last pass is not optional — the kernel reconstructs with
/// the stored `d*ls`, so choosing levels against the unquantized fit bakes in an
/// error the dequantizer cannot undo. `quantize_row_q4_K_ref` does the same
/// second pass for the same reason.
///
/// Rows are independent and are packed in parallel. See
/// [`quantize_int4_hier_measured`] if the weight error is wanted — it comes
/// free with the pack and costs a second full pass to recover afterwards.
pub fn quantize_int4_hier(
    w: &[f32],
    in_features: usize,
    out_features: usize,
) -> Result<PackedInt4Hier> {
    quantize_int4_hier_measured(w, in_features, out_features).map(|(p, _)| p)
}

/// [`quantize_int4_hier`], plus the relative Frobenius weight error, accumulated
/// during packing rather than by reconstructing the matrix afterwards.
///
/// The error is a **smoke test for a broken packer, not a quality gate** — see
/// the crate docs. A clip search that lowers it by 0.0035 raises perplexity by
/// 0.50.
pub fn quantize_int4_hier_measured(
    w: &[f32],
    in_features: usize,
    out_features: usize,
) -> Result<(PackedInt4Hier, f64)> {
    if in_features % HGROUP != 0 {
        return Err(Error::Shape(format!(
            "in_features {in_features} must be a multiple of {HGROUP}"
        )));
    }
    if w.len() != in_features * out_features {
        return Err(Error::Shape(format!(
            "weight slice has {} elements, expected {}",
            w.len(),
            in_features * out_features
        )));
    }

    let groups = in_features / HGROUP;
    let words = in_features / 8;
    let mut qw = vec![0u32; out_features * words];
    let mut si = vec![0u8; out_features * groups];
    let mut sb = vec![0u32; out_features];

    // Every row owns a disjoint slice of all three outputs, so the fan-out needs
    // no synchronisation and no atomics. The reduction is over the error only.
    let acc = qw
        .par_chunks_mut(words)
        .zip(si.par_chunks_mut(groups))
        .zip(sb.par_iter_mut())
        .zip(w.par_chunks(in_features))
        .map_init(
            || (vec![0f32; groups], vec![0f32; groups]),
            |(ss, sm), (((qw_row, si_row), sb_row), row)| {
                pack_row_hier(row, groups, ss, sm, qw_row, si_row, sb_row)
            },
        )
        .reduce(ErrorAccum::default, ErrorAccum::merge);

    Ok((PackedInt4Hier { qw, si, sb, in_features, out_features }, acc.relative()))
}

/// Reconstructs the weights a packed matrix represents.
///
/// This is what the GEMV computes with, so it is the correct reference for
/// differential-testing the kernel: a disagreement against *this* is a kernel
/// bug, a disagreement against the original weights is quantization error.
pub fn dequantize_int4_hier(p: &PackedInt4Hier) -> Vec<f32> {
    let groups = p.in_features / HGROUP;
    let words = p.in_features / 8;
    let mut out = vec![0f32; p.in_features * p.out_features];

    out.par_chunks_mut(p.in_features)
        .zip(p.qw.par_chunks(words))
        .zip(p.si.par_chunks(groups))
        .zip(p.sb.par_iter())
        .for_each(|(((dst, qw_row), si_row), &sbw)| {
            let d = f16::from_bits(sbw as u16).to_f32();
            let dm = f16::from_bits((sbw >> 16) as u16).to_f32();
            for (g, &idx) in si_row.iter().enumerate().take(groups) {
                let s = d * (idx & 0xF) as f32;
                let m = -dm * (idx >> 4) as f32;
                for i in 0..HGROUP {
                    let col = g * HGROUP + i;
                    let q = ((qw_row[col / 8] >> (4 * (col % 8))) & 0xF) as f32;
                    dst[col] = q * s + m;
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relative_error;

    fn weights(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let a = ((i * 2_654_435_761usize) % 10_000) as f32 / 10_000.0 - 0.5;
                let b = ((i * 40_503usize) % 977) as f32 / 977.0 - 0.5;
                a * 0.15 + b * b * b * 0.5
            })
            .collect()
    }

    #[test]
    fn beats_group_128_at_almost_the_same_width() {
        // The whole justification for this format: finer groups, no bit budget.
        let (in_f, out_f) = (896usize, 64usize);
        let w = weights(in_f * out_f);

        let flat = crate::quantize_int4_g128(&w, in_f, out_f).unwrap();
        let hier = quantize_int4_hier(&w, in_f, out_f).unwrap();

        let e_flat = relative_error(&w, &crate::dequantize_int4_g128(&flat));
        let e_hier = relative_error(&w, &dequantize_int4_hier(&hier));

        assert!(
            e_hier < e_flat,
            "hierarchical error {e_hier} should beat g128's {e_flat}"
        );
        // 4 + 8/32 + 32/896 = 4.2857
        let bpw = hier.bits_per_weight();
        assert!(
            (bpw - 4.2857).abs() < 1e-3,
            "expected 4.286 bits/weight at in=896, got {bpw}"
        );
        assert!(
            bpw < flat.bits_per_weight() + 0.04,
            "the point is that it costs almost nothing: {bpw} vs {}",
            flat.bits_per_weight()
        );
    }

    #[test]
    fn scale_index_is_never_zero() {
        // A zero index would make the quantizer divide by zero and the kernel
        // reconstruct a whole group as `min`. The clamp is load-bearing.
        let (in_f, out_f) = (256usize, 8usize);
        let mut w = weights(in_f * out_f);
        // One group with a huge range next to groups with almost none, which is
        // what drives a scale index toward zero.
        for (i, v) in w.iter_mut().enumerate().take(HGROUP) {
            *v = if i == 0 { 40.0 } else { -40.0 };
        }
        for v in w.iter_mut().skip(HGROUP).take(in_f - HGROUP) {
            *v = 1e-6;
        }
        let p = quantize_int4_hier(&w, in_f, out_f).unwrap();
        for (i, b) in p.si.iter().enumerate() {
            assert!(b & 0xF >= 1, "group {i} got scale index 0");
        }
        assert!(dequantize_int4_hier(&p).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn constant_and_zero_rows_survive() {
        let (in_f, out_f) = (128usize, 3usize);
        let mut w = vec![0.0f32; in_f * out_f];
        for v in w.iter_mut().skip(in_f).take(in_f) {
            *v = 0.25;
        }
        for v in w.iter_mut().skip(2 * in_f) {
            *v = -0.5;
        }
        let p = quantize_int4_hier(&w, in_f, out_f).unwrap();
        let d = dequantize_int4_hier(&p);
        assert!(d.iter().all(|v| v.is_finite()), "non-finite output");
        for (a, b) in w.iter().zip(&d) {
            assert!((a - b).abs() < 1e-3, "constant row not reproduced: {a} vs {b}");
        }
    }

    #[test]
    fn packing_widths_are_what_the_roofline_assumes() {
        let (in_f, out_f) = (4864usize, 4usize);
        let w = weights(in_f * out_f);
        let p = quantize_int4_hier(&w, in_f, out_f).unwrap();
        assert_eq!(p.qw.len(), out_f * in_f / 8);
        assert_eq!(p.si.len(), out_f * in_f / HGROUP);
        assert_eq!(p.sb.len(), out_f);
        // 4 + 8/32 + 32/4864
        assert!((p.bits_per_weight() - 4.2566).abs() < 1e-3);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(quantize_int4_hier(&[0.0; 100], 100, 1).is_err());
        assert!(quantize_int4_hier(&[0.0; 10], 32, 1).is_err());
    }

    /// A row whose weights are tiny but non-zero underflows the shared fp16 `d`.
    ///
    /// `d = max_scale / 15`, and f16's smallest positive subnormal is 5.96e-8,
    /// so a row spanning less than about 1.3e-5 rounds `d` to **zero**. Then
    /// `s = d * ls` is zero, `q = (w - m) / s` is an infinity or a NaN, and the
    /// row reconstructs to garbage.
    ///
    /// Found on Qwen2.5-7B, where `convert` reported `mean rel. error NaN`. It
    /// was invisible on the three smaller models and invisible before the error
    /// metric was fused into the packer, because the old reconstruct-afterwards
    /// path computed `q * 0 + m`, which is finite and wrong rather than NaN.
    #[test]
    fn tiny_rows_do_not_underflow_the_shared_scale() {
        let (in_f, out_f) = (256usize, 4usize);
        let mut w = vec![0f32; in_f * out_f];
        // Row 0: ordinary magnitudes. Row 1: a hundred-millionth of that.
        for i in 0..in_f {
            w[i] = ((i % 17) as f32 - 8.0) * 0.01;
            w[in_f + i] = ((i % 17) as f32 - 8.0) * 1e-10;
        }
        let (p, e) = quantize_int4_hier_measured(&w, in_f, out_f).unwrap();
        assert!(e.is_finite(), "relative error is {e}");

        let d = dequantize_int4_hier(&p);
        assert!(d.iter().all(|v| v.is_finite()), "dequantized row is not finite");

        // Every shared scale must be a usable positive number, not zero and not
        // an infinity: the kernel divides nothing by it but it multiplies every
        // weight in the row.
        for (r, &sb) in p.sb.iter().enumerate() {
            let dv = half::f16::from_bits(sb as u16).to_f32();
            assert!(dv > 0.0 && dv.is_finite(), "row {r} shared scale is {dv}");
        }
    }
}
