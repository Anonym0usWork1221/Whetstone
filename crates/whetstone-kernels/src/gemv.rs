//! Batch=1 decode GEMV.
//!
//! At batch=1 a linear layer is a matrix-vector product: every weight is read
//! once and used for a single multiply-add. That is ~2 FLOP/byte against a
//! machine balance near 120, so these kernels are judged purely on achieved
//! memory bandwidth. A GEMV that hits 80% of peak bandwidth is essentially
//! optimal no matter what its FLOP count looks like.

use crate::ffi;
use crate::{check, DeviceBuffer, Error, Result};

/// Weights per scale/zero pair in the int4 format.
pub const GROUP: usize = 128;

/// A quantized linear layer, resident on the device.
///
/// Layout, matching `cuda/gemv_int4.cu`:
/// - `qw`: `[out_features][in_features/8]` `u32`, eight nibbles per word
/// - `sz`: `[out_features][in_features/128]` packed `half2`, scale then zero
pub struct QuantLinear {
    qw: DeviceBuffer<u32>,
    sz: DeviceBuffer<u32>, // half2 pairs, kept as u32 so the host never needs f16
    in_features: usize,
    out_features: usize,
}

impl QuantLinear {
    /// Uploads a pre-packed quantized weight matrix.
    pub fn from_packed(
        qw: &[u32],
        sz: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if in_features % GROUP != 0 {
            return Err(Error::Shape(format!(
                "in_features {in_features} must be a multiple of {GROUP}"
            )));
        }
        let want_qw = out_features * in_features / 8;
        let want_sz = out_features * in_features / GROUP;
        if qw.len() != want_qw {
            return Err(Error::Shape(format!(
                "packed weights have {} words, expected {want_qw}",
                qw.len()
            )));
        }
        if sz.len() != want_sz {
            return Err(Error::Shape(format!(
                "scale/zero array has {} entries, expected {want_sz}",
                sz.len()
            )));
        }

        Ok(Self {
            qw: DeviceBuffer::from_slice(qw)?,
            sz: DeviceBuffer::from_slice(sz)?,
            in_features,
            out_features,
        })
    }

    /// Input width.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output width.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Bytes of weight data read per invocation, including scale metadata.
    ///
    /// This is the numerator of the roofline: `tok/s <= bandwidth / bytes`.
    pub fn bytes(&self) -> usize {
        self.qw.bytes() + self.sz.bytes()
    }

    /// Effective bits per weight, counting the scale/zero overhead.
    ///
    /// Quoting "4-bit" while ignoring per-group scales understates bandwidth by
    /// 5-10%, and bandwidth is exactly what sets decode speed.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes() as f64 * 8.0 / (self.in_features * self.out_features) as f64
    }

    /// `y = W x`, with `x` in fp16 and `y` in fp32.
    pub fn gemv(&self, x: &DeviceBuffer<u16>, y: &mut DeviceBuffer<f32>) -> Result<()> {
        if x.len() != self.in_features {
            return Err(Error::Shape(format!(
                "x has {} elements, expected {}",
                x.len(),
                self.in_features
            )));
        }
        if y.len() != self.out_features {
            return Err(Error::Shape(format!(
                "y has {} elements, expected {}",
                y.len(),
                self.out_features
            )));
        }

        // SAFETY: shapes are validated above; all four buffers are live device
        // allocations of the sizes the kernel indexes, and the kernel writes
        // only y[0..out_features].
        check(unsafe {
            ffi::wst_gemv_int4_g128(
                self.qw.as_ptr(),
                self.sz.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                self.in_features as i32,
                self.out_features as i32,
            )
        })
    }
}

/// `y = W x` with dense fp16 weights.
///
/// The reference path. It separates "the kernel is wrong" from "the
/// quantization is lossy" during differential testing, and gives the honest
/// same-schedule baseline the quantized kernels must beat.
pub fn gemv_fp16(
    w: &DeviceBuffer<u16>,
    x: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<f32>,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if w.len() != in_features * out_features {
        return Err(Error::Shape(format!(
            "weights have {} elements, expected {}",
            w.len(),
            in_features * out_features
        )));
    }
    if x.len() != in_features || y.len() != out_features {
        return Err(Error::Shape(format!(
            "expected x[{in_features}] and y[{out_features}], got x[{}] y[{}]",
            x.len(),
            y.len()
        )));
    }
    // SAFETY: all three shapes are validated above against the dimensions the
    // kernel indexes; the kernel writes only y[0..out_features].
    check(unsafe {
        ffi::wst_gemv_fp16(
            w.as_ptr(),
            x.as_ptr(),
            y.as_mut_ptr(),
            in_features as i32,
            out_features as i32,
        )
    })
}

/// Measured throughput of a single GEMV.
#[derive(Debug, Clone, Copy)]
pub struct GemvBench {
    /// Achieved bandwidth in GB/s.
    pub gbs: f64,
    /// Milliseconds per invocation.
    pub ms: f64,
}

impl GemvBench {
    /// Fraction of the device's peak bandwidth this GEMV attained.
    ///
    /// The figure of merit. Above ~0.7 the kernel is near-optimal and further
    /// effort should go into shrinking the format, not tuning the code.
    pub fn utilisation(&self, peak_gbs: f64) -> f64 {
        self.gbs / peak_gbs
    }
}

/// Benchmarks a GEMV of the given shape with synthetic weights.
pub fn bench_gemv(in_f: usize, out_f: usize, reps: i32, int4: bool) -> Result<GemvBench> {
    let mut gbs = 0.0f64;
    let mut ms = 0.0f64;
    // SAFETY: both out-params are owned, initialised f64s; the callee validates
    // the scalar arguments and allocates its own device buffers.
    check(unsafe {
        ffi::wst_bench_gemv(
            in_f as i32,
            out_f as i32,
            reps,
            i32::from(int4),
            &mut gbs,
            &mut ms,
        )
    })?;
    Ok(GemvBench { gbs, ms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn gpu() -> Option<Device> {
        Device::default_device().ok()
    }

    #[test]
    fn gemv_matches_the_dequantized_reference() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // Qwen2.5-0.5B's attention projection shape.
        let (in_f, out_f) = (896usize, 128usize);

        // Deterministic pseudo-random weights and activations.
        let w: Vec<f32> = (0..in_f * out_f)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..in_f)
            .map(|i| ((i * 40503) % 200) as f32 / 100.0 - 1.0)
            .collect();

        // The real quantizer, not a test-local copy: this exercises the exact
        // packing the engine ships.
        let packed = whetstone_quant::quantize_int4_g128(&w, in_f, out_f).unwrap();
        let dequant = whetstone_quant::dequantize_int4_g128(&packed);

        let xh: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let x_dev = DeviceBuffer::from_slice(&xh).unwrap();
        let mut y_dev = DeviceBuffer::<f32>::zeros(out_f).unwrap();

        let layer = QuantLinear::from_packed(&packed.qw, &packed.sz, in_f, out_f).unwrap();
        layer.gemv(&x_dev, &mut y_dev).unwrap();
        let got = y_dev.to_vec().unwrap();

        // Reference uses the SAME dequantized weights, so any disagreement is a
        // kernel bug rather than quantization error. The activations round-trip
        // through f16, so the tolerance covers f16 input precision only.
        for r in 0..out_f {
            let want: f32 = (0..in_f)
                .map(|c| dequant[r * in_f + c] * half::f16::from_f32(x[c]).to_f32())
                .sum();
            let tol = 2e-2 * want.abs().max(1.0);
            assert!(
                (got[r] - want).abs() < tol,
                "row {r}: kernel {} vs reference {want} (tol {tol})",
                got[r]
            );
        }
    }

    #[test]
    fn rejects_shape_violations() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // in_features must be a multiple of the group size.
        assert!(matches!(
            QuantLinear::from_packed(&[0u32; 100], &[0u32; 1], 100, 8),
            Err(Error::Shape(_))
        ));
        // Packed array sizes must match the declared shape.
        assert!(matches!(
            QuantLinear::from_packed(&[0u32; 8], &[0u32; 1], 128, 1),
            Err(Error::Shape(_))
        ));
    }

    /// Performance, not correctness — so it is opt-in.
    ///
    /// Timing assertions are load-sensitive: run immediately after a build, or
    /// alongside other GPU tests, this measures the machine's contention as
    /// much as the kernel. A flaky gate on a release build is worse than no
    /// gate, so `deploy.sh` runs this separately and reports it rather than
    /// failing on it.
    ///
    ///     cargo test --release -- --ignored --nocapture
    #[test]
    #[ignore = "timing-sensitive; run explicitly with --ignored"]
    fn int4_moves_a_quarter_of_the_bytes_and_gains_accordingly() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // Qwen2.5-0.5B MLP shape: this is 87.7% of the model's weights.
        let (in_f, out_f) = (896usize, 4864usize);

        // Best of N. cargo runs tests in parallel by default, so several GPU
        // tests contend for the same device and a single timing sample is
        // unreliable. The minimum is the least-contended observation, which is
        // the honest estimate of what the kernel costs.
        let best = |int4: bool| {
            (0..3)
                .map(|_| bench_gemv(in_f, out_f, 200, int4).unwrap())
                .min_by(|a, b| a.ms.partial_cmp(&b.ms).unwrap())
                .unwrap()
        };
        let f16 = best(false);
        let i4 = best(true);

        let peak = d.bandwidth_gbs();
        println!(
            "fp16: {:.0} GB/s ({:.0}% peak) {:.3} ms | int4: {:.0} GB/s ({:.0}% peak) {:.3} ms | speedup {:.2}x",
            f16.gbs,
            f16.utilisation(peak) * 100.0,
            f16.ms,
            i4.gbs,
            i4.utilisation(peak) * 100.0,
            i4.ms,
            f16.ms / i4.ms
        );

        // The property under test is wall-clock: int4 reads ~4.25 bits/weight
        // against fp16's 16, so it must finish sooner.
        assert!(
            i4.ms < f16.ms * 0.85,
            "int4 GEMV ({:.3} ms) should beat fp16 ({:.3} ms)",
            i4.ms,
            f16.ms
        );

        // Loose sanity floor only. int4 currently reaches ~25% of peak
        // bandwidth while fp16 reaches ~60%: at these shapes the int4 path is
        // latency-bound, not bandwidth-bound, because 2.3 MB of weights is
        // small enough that launch and reduction overhead are a real fraction
        // of a ~30 us kernel. That gap is a known optimization target, so this
        // assertion is deliberately not set to where we want it to be -- a test
        // that encodes an aspiration just goes flaky.
        assert!(
            i4.gbs > 0.12 * peak,
            "int4 GEMV bandwidth {:.0} GB/s is below the sanity floor",
            i4.gbs
        );
    }
}
