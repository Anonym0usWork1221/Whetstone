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

/// Alternative int4 GEMV implementations, selected by measurement.
///
/// The engine calls [`select`] once at startup and every [`QuantLinear::gemv`]
/// afterwards routes through the chosen kernel. Keeping the choice in one
/// process-global integer rather than threading it through every call site is
/// deliberate: it makes an A/B a single argument, which is the difference
/// between a sweep that gets run and one that does not.
pub mod variant {
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicI32, Ordering};

    use crate::ffi;
    use crate::{check, GemvBench, Result};

    /// `i32::MIN` means "not chosen yet", which resolves to the swept default
    /// on first use. `-1` means the original hand-written kernel in
    /// `gemv_int4.cu`, kept reachable so an A/B against it is one flag.
    static SELECTED: AtomicI32 = AtomicI32::new(i32::MIN);

    /// The variant the sweep selected for this architecture.
    ///
    /// Defined in `gemv_variants.cu` alongside the measurement table that
    /// justifies it, so the number and its evidence cannot drift apart.
    pub fn default_index() -> usize {
        // SAFETY: no arguments, returns a compile-time constant.
        (unsafe { ffi::wst_gemv_default_variant() }).max(0) as usize
    }

    /// The variant the sweep selected for a particular matrix shape.
    ///
    /// One blocking does not win everywhere: the model's shapes run from
    /// 896x128 to 896x151936, and what changes between them is how many warps
    /// the shape can create and therefore how many loads it can keep in flight.
    pub fn for_shape(in_f: usize, out_f: usize) -> usize {
        // SAFETY: two scalars in, a table index out.
        (unsafe { ffi::wst_gemv_variant_for_shape(in_f as i32, out_f as i32) }).max(0) as usize
    }

    /// Number of swept variants.
    pub fn count() -> usize {
        // SAFETY: no arguments, returns a compile-time constant.
        (unsafe { ffi::wst_gemv_variant_count() }).max(0) as usize
    }

    /// Human-readable description of a variant.
    pub fn name(v: usize) -> String {
        // SAFETY: the callee returns a pointer to a static string literal for
        // any input, including out-of-range ones (it yields "?").
        unsafe {
            let p = ffi::wst_gemv_variant_name(v as i32);
            if p.is_null() {
                "?".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    /// The active variant, or `None` for the baseline kernel.
    pub fn selected() -> Option<usize> {
        let v = SELECTED.load(Ordering::Relaxed);
        if v == i32::MIN {
            return Some(default_index());
        }
        if v < 0 {
            None
        } else {
            Some(v as usize)
        }
    }

    /// Chooses the variant every subsequent GEMV will use.
    pub fn select(v: Option<usize>) {
        SELECTED.store(v.map_or(-1, |x| x as i32), Ordering::Relaxed);
    }

    /// The per-shape rule: which variant serves a wide reduction, a huge
    /// output, and everything else.
    pub fn shape_rule() -> [usize; 3] {
        let mut r = [0i32; 3];
        // SAFETY: `r` is an owned, initialised array of exactly the three i32s
        // the callee writes.
        unsafe { ffi::wst_gemv_get_shape_rule(r.as_mut_ptr()) };
        [r[0].max(0) as usize, r[1].max(0) as usize, r[2].max(0) as usize]
    }

    /// Overrides the per-shape rule. Out-of-range entries are ignored.
    pub fn set_shape_rule(rule: [usize; 3]) {
        // SAFETY: three plain scalars; the callee range-checks each one.
        unsafe { ffi::wst_gemv_set_shape_rule(rule[0] as i32, rule[1] as i32, rule[2] as i32) };
    }

    /// True when no variant has been forced, so each shape picks its own.
    pub fn is_auto() -> bool {
        SELECTED.load(Ordering::Relaxed) == i32::MIN
    }

    /// Times one variant at one shape with synthetic weights.
    pub fn bench(v: usize, in_f: usize, out_f: usize, reps: i32) -> Result<GemvBench> {
        let mut gbs = 0.0f64;
        let mut ms = 0.0f64;
        // SAFETY: both out-params are owned, initialised f64s; the callee
        // validates the variant index and shape and allocates its own buffers.
        check(unsafe {
            ffi::wst_bench_gemv_variant(v as i32, in_f as i32, out_f as i32, reps, &mut gbs, &mut ms)
        })?;
        Ok(GemvBench { gbs, ms })
    }
}

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
        self.gemv_ex(x, None, y, false)
    }

    /// Dequantizes one row into a float vector.
    ///
    /// This exists because a tied embedding matrix has two uses that look
    /// nothing alike: a single-row gather on the way in (free) and a full GEMV
    /// on the way out (27.6% of decode traffic on Qwen2.5-0.5B). Quantizing the
    /// matrix quantizes both, so the gather has to dequantize.
    pub fn gather_row(
        &self,
        row: &crate::decode::DeviceCursor,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if out.len() != self.in_features {
            return Err(Error::Shape(format!(
                "gather_row: output has {} elements, row width is {}",
                out.len(),
                self.in_features
            )));
        }
        crate::decode::embed_int4(&self.qw, &self.sz, row, out)
    }

    /// Dequantizes `n` rows named by a device-resident id array.
    pub fn gather_rows(
        &self,
        tokens: &DeviceBuffer<i32>,
        out: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        crate::chunk::embed_int4_g128(
            &self.qw,
            &self.sz,
            tokens,
            out,
            self.in_features,
            self.out_features,
            n,
        )
    }

    /// `y = W x + b` or `y += W x + b`.
    ///
    /// Bias and accumulation are epilogue flags rather than separate kernels.
    /// Each is one instruction on a row that has just streamed hundreds of
    /// bytes, but as launches they would cost four dispatches per transformer
    /// block — 96 per token on a 24-layer model, which at batch=1 is a
    /// measurable fraction of the token.
    pub fn gemv_ex(
        &self,
        x: &DeviceBuffer<u16>,
        bias: Option<&DeviceBuffer<u16>>,
        y: &mut DeviceBuffer<f32>,
        accumulate: bool,
    ) -> Result<()> {
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
        if let Some(b) = bias {
            if b.len() != self.out_features {
                return Err(Error::Shape(format!(
                    "bias has {} elements, expected {}",
                    b.len(),
                    self.out_features
                )));
            }
        }

        let bias_ptr = bias.map_or(std::ptr::null(), DeviceBuffer::as_ptr);

        // SAFETY (both arms): shapes are validated above; all buffers are live
        // device allocations of the sizes the kernel indexes, the bias pointer
        // is null exactly when no bias was supplied, and the kernel writes only
        // y[0..out_features]. The variant index comes from `variant::select`,
        // which the callee range-checks anyway.
        // Nothing forced: let the shape pick. Something forced: honour it, so
        // an A/B is one flag and the sweep stays reproducible.
        let choice = if variant::is_auto() {
            Some(variant::for_shape(self.in_features, self.out_features))
        } else {
            variant::selected()
        };

        match choice {
            None => check(unsafe {
                ffi::wst_gemv_int4_g128_ex(
                    self.qw.as_ptr(),
                    self.sz.as_ptr(),
                    x.as_ptr(),
                    bias_ptr,
                    y.as_mut_ptr(),
                    self.in_features as i32,
                    self.out_features as i32,
                    i32::from(accumulate),
                )
            }),
            Some(v) => check(unsafe {
                ffi::wst_gemv_int4_variant(
                    v as i32,
                    self.qw.as_ptr(),
                    self.sz.as_ptr(),
                    x.as_ptr(),
                    bias_ptr,
                    y.as_mut_ptr(),
                    self.in_features as i32,
                    self.out_features as i32,
                    i32::from(accumulate),
                )
            }),
        }
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
    gemv_fp16_ex(w, x, None, y, in_features, out_features, false)
}

/// `y = W x + b` or `y += W x + b`, with dense fp16 weights.
pub fn gemv_fp16_ex(
    w: &DeviceBuffer<u16>,
    x: &DeviceBuffer<u16>,
    bias: Option<&DeviceBuffer<u16>>,
    y: &mut DeviceBuffer<f32>,
    in_features: usize,
    out_features: usize,
    accumulate: bool,
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
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err(Error::Shape(format!(
                "bias has {} elements, expected {out_features}",
                b.len()
            )));
        }
    }
    // SAFETY: all shapes are validated above against the dimensions the kernel
    // indexes, the bias pointer is null exactly when no bias was supplied, and
    // the kernel writes only y[0..out_features].
    check(unsafe {
        ffi::wst_gemv_fp16_ex(
            w.as_ptr(),
            x.as_ptr(),
            bias.map_or(std::ptr::null(), DeviceBuffer::as_ptr),
            y.as_mut_ptr(),
            in_features as i32,
            out_features as i32,
            i32::from(accumulate),
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

// ---------------------------------------------------------------- hierarchical

/// Weights per `(ls, lm)` index pair in the hierarchical int4 format.
///
/// Not tunable: 32 nibbles is exactly one `uint4`, so a lane handles precisely
/// one group per load and the metadata read is one byte per lane.
pub const HGROUP: usize = 32;

/// An int4 linear layer with hierarchical scale metadata.
///
/// Layout, matching `cuda/gemv_hier.cu`:
/// - `qw`: `[out_features][in_features/8]` `u32`, eight nibbles per word
/// - `si`: `[out_features][in_features/32]` `u8`, scale index low, min index high
/// - `sb`: `[out_features]` `u32`, an fp16 `d` low and an fp16 `dmin` high
///
/// Reconstruction is `w = q*(d*ls) - dmin*lm`. See `whetstone-quant::hier` for
/// why this format replaces the flat group-128 one: it buys group-32 granularity
/// for 0.036 bits/weight, and granularity measured six times more valuable than
/// the fitting algorithm.
pub struct QuantLinearHier {
    qw: DeviceBuffer<u32>,
    si: DeviceBuffer<u8>,
    sb: DeviceBuffer<u32>,
    in_features: usize,
    out_features: usize,
}

impl QuantLinearHier {
    /// Uploads a pre-packed matrix, validating every array against the shape.
    pub fn from_packed(
        qw: &[u32],
        si: &[u8],
        sb: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if in_features % HGROUP != 0 {
            return Err(Error::Shape(format!(
                "in_features {in_features} must be a multiple of {HGROUP}"
            )));
        }
        let (want_qw, want_si, want_sb) = (
            out_features * in_features / 8,
            out_features * in_features / HGROUP,
            out_features,
        );
        if qw.len() != want_qw || si.len() != want_si || sb.len() != want_sb {
            return Err(Error::Shape(format!(
                "hierarchical int4 arrays are {}/{}/{}, expected {want_qw}/{want_si}/{want_sb}",
                qw.len(),
                si.len(),
                sb.len()
            )));
        }
        Ok(Self {
            qw: DeviceBuffer::from_slice(qw)?,
            si: DeviceBuffer::from_slice(si)?,
            sb: DeviceBuffer::from_slice(sb)?,
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

    /// Weight bytes streamed per invocation, including all scale metadata.
    pub fn bytes(&self) -> usize {
        self.qw.bytes() + self.si.bytes() + self.sb.bytes()
    }

    /// Effective bits per weight.
    pub fn bits_per_weight(&self) -> f64 {
        self.bytes() as f64 * 8.0 / (self.in_features * self.out_features) as f64
    }

    /// Dequantizes one row — the input gather when the tied matrix lives here.
    pub fn gather_row(
        &self,
        row: &crate::decode::DeviceCursor,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if out.len() != self.in_features {
            return Err(Error::Shape(format!(
                "gather_row: output has {} elements, row width is {}",
                out.len(),
                self.in_features
            )));
        }
        crate::decode::embed_int4_hier(&self.qw, &self.si, &self.sb, row, out)
    }

    /// Dequantizes `n` rows named by a device-resident id array.
    pub fn gather_rows(
        &self,
        tokens: &DeviceBuffer<i32>,
        out: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        crate::chunk::embed_int4_hier(
            &self.qw,
            &self.si,
            &self.sb,
            tokens,
            out,
            self.in_features,
            self.out_features,
            n,
        )
    }

    /// `y = W x + b`, or `y += W x + b` when `accumulate`.
    pub fn gemv_ex(
        &self,
        x: &DeviceBuffer<u16>,
        bias: Option<&DeviceBuffer<u16>>,
        y: &mut DeviceBuffer<f32>,
        accumulate: bool,
    ) -> Result<()> {
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
        if let Some(b) = bias {
            if b.len() != self.out_features {
                return Err(Error::Shape(format!(
                    "bias has {} elements, expected {}",
                    b.len(),
                    self.out_features
                )));
            }
        }
        let bias_ptr = bias.map_or(std::ptr::null(), DeviceBuffer::as_ptr);
        // SAFETY: shapes are validated above against the buffers' real lengths,
        // the bias pointer is null exactly when no bias was supplied, and the
        // kernel writes only y[0..out_features].
        check(unsafe {
            ffi::wst_gemv_int4_hier_ex(
                self.qw.as_ptr(),
                self.si.as_ptr(),
                self.sb.as_ptr(),
                x.as_ptr(),
                bias_ptr,
                y.as_mut_ptr(),
                self.in_features as i32,
                self.out_features as i32,
                i32::from(accumulate),
            )
        })
    }

    /// `y = W x`.
    pub fn gemv(&self, x: &DeviceBuffer<u16>, y: &mut DeviceBuffer<f32>) -> Result<()> {
        self.gemv_ex(x, None, y, false)
    }

    /// The multi-token form: `y[j] = W x[j] + b` for every `j < n`.
    ///
    /// `x` is `[n][in_features]` and `y` is `[n][out_features]`, token-major. The
    /// weights are read **once** for all `n` tokens, which is the entire reason
    /// the chunk path exists — see `cuda/chunk_gemm.cu`.
    pub fn gemm_ex(
        &self,
        x: &DeviceBuffer<u16>,
        bias: Option<&DeviceBuffer<u16>>,
        y: &mut DeviceBuffer<f32>,
        n: usize,
        accumulate: bool,
    ) -> Result<()> {
        if n == 0 {
            return Err(Error::Shape("gemm_ex: n must be positive".into()));
        }
        // At least, not exactly: chunk scratch is allocated once at the maximum
        // width and used at whatever width the current pass needs.
        if x.len() < n * self.in_features {
            return Err(Error::Shape(format!(
                "gemm_ex: x has {} elements, need {n}*{}",
                x.len(),
                self.in_features
            )));
        }
        if y.len() < n * self.out_features {
            return Err(Error::Shape(format!(
                "gemm_ex: y has {} elements, need {n}*{}",
                y.len(),
                self.out_features
            )));
        }
        if let Some(b) = bias {
            if b.len() != self.out_features {
                return Err(Error::Shape(format!(
                    "gemm_ex: bias has {} elements, expected {}",
                    b.len(),
                    self.out_features
                )));
            }
        }
        let bias_ptr = bias.map_or(std::ptr::null(), DeviceBuffer::as_ptr);
        // SAFETY: shapes are validated above against the buffers' real lengths,
        // the bias pointer is null exactly when no bias was supplied, and the
        // kernel writes only y[0..n*out_features].
        check(unsafe {
            ffi::wst_gemm_int4_hier(
                self.qw.as_ptr(),
                self.si.as_ptr(),
                self.sb.as_ptr(),
                x.as_ptr(),
                bias_ptr,
                y.as_mut_ptr(),
                self.in_features as i32,
                self.out_features as i32,
                n as i32,
                i32::from(accumulate),
            )
        })
    }
}

/// The rows-per-warp rule for the hierarchical kernel, as three tile indices
/// (0 → 1 row, 1 → 2 rows, 2 → 4 rows) for wide-reduction, huge-output and
/// everything-else shapes.
///
/// Exposed for sweeping, because the trade this rule makes — in-flight bytes
/// against warp-level parallelism — depends on the shape, and the only
/// measurement that has ever ranked these correctly is whole-generation
/// throughput. A microbenchmark exaggerates the spread by more than an order of
/// magnitude and the per-stage event profiler reorders it.
pub fn hier_set_rule(wide: i32, huge: i32, other: i32) {
    // SAFETY: a plain store into three process-global ints in the CUDA module.
    unsafe { ffi::wst_gemv_hier_set_rule(wide, huge, other) }
}

/// Applies `WHETSTONE_HIER_RULE=wide,huge,other` if it is set.
///
/// Sweeping this rule needs the *whole generation* to be re-run per candidate --
/// a microbenchmark exaggerates the spread by more than an order of magnitude
/// and the per-stage event profiler reorders it, both measured on the g128
/// kernel. An environment variable is the cheapest thing that makes "run the
/// engine 64 times with different rules" a shell loop instead of 64 rebuilds.
pub fn hier_rule_from_env() {
    let Ok(v) = std::env::var("WHETSTONE_HIER_RULE") else { return };
    let parts: Vec<i32> = v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if parts.len() == 3 {
        hier_set_rule(parts[0], parts[1], parts[2]);
    } else {
        eprintln!("WHETSTONE_HIER_RULE={v:?} ignored: expected three integers 0..3");
    }
}

/// The rule currently in force.
pub fn hier_get_rule() -> [i32; 3] {
    let mut out = [0i32; 3];
    // SAFETY: writes exactly three ints into a live stack array.
    unsafe { ffi::wst_gemv_hier_get_rule(out.as_mut_ptr()) }
    out
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

    /// Every variant must agree with the dequantized reference, not just the
    /// one currently selected.
    ///
    /// The `h2` variants replace an integer-to-float conversion with an OR into
    /// an fp16 mantissa and accumulate the group's dot product on the fp16 pipe.
    /// Both are exactly the kind of change that is fast and subtly wrong, and
    /// both would still produce plausible text. The activations here are scaled
    /// to the magnitudes an RMSNorm actually emits, because the fp16 accumulator
    /// is the part with a range limit.
    #[test]
    fn every_variant_agrees_with_the_dequantized_reference() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // down_proj's shape: the longest reduction in the model, so the fp16
        // group accumulator sees the most terms here.
        let (in_f, out_f) = (4864usize, 64usize);

        let w: Vec<f32> = (0..in_f * out_f)
            .map(|i| ((i * 2_654_435_761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let x: Vec<f32> =
            (0..in_f).map(|i| (((i * 40503) % 200) as f32 / 100.0 - 1.0) * 4.0).collect();

        let packed = whetstone_quant::quantize_int4_g128(&w, in_f, out_f).unwrap();
        let dequant = whetstone_quant::dequantize_int4_g128(&packed);

        let xh: Vec<u16> = x.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let x_dev = DeviceBuffer::from_slice(&xh).unwrap();
        let layer = QuantLinear::from_packed(&packed.qw, &packed.sz, in_f, out_f).unwrap();

        let want: Vec<f32> = (0..out_f)
            .map(|r| {
                (0..in_f)
                    .map(|c| {
                        dequant[r * in_f + c] as f64 * half::f16::from_f32(x[c]).to_f32() as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect();

        let n = variant::count();
        assert!(n > 0, "no GEMV variants compiled in");

        // Relative L2 error against the reference, per variant. Printing it is
        // the point: the `h2` variants are an explicit precision-for-speed
        // trade, and a number is the only way to argue about the size of it.
        for v in 0..n {
            let label = variant::name(v);
            // The memory-path probe deliberately computes the wrong thing --
            // its job is to measure the floor, not to be correct.
            if label.starts_with("mem") {
                continue;
            }

            variant::select(Some(v));
            let mut y = DeviceBuffer::<f32>::zeros(out_f).unwrap();
            layer.gemv(&x_dev, &mut y).unwrap();
            let got = y.to_vec().unwrap();

            let (mut num, mut den) = (0f64, 0f64);
            for r in 0..out_f {
                let d = (got[r] - want[r]) as f64;
                num += d * d;
                den += (want[r] as f64) * (want[r] as f64);
            }
            let rel = (num / den).sqrt();
            println!("  variant {v:>2} {label:<14} relative error {rel:.5}");

            if label.starts_with("f32") {
                // fp32 accumulation must reproduce the reference to within f16
                // input rounding and nothing more.
                assert!(rel < 5e-3, "variant {v} ({label}) drifted: relative error {rel}");
            } else {
                // The fp16 accumulator carries 11 mantissa bits into a reduction
                // whose terms largely cancel, so it loses precision in proportion
                // to how much cancellation there is -- worst at down_proj's 4864
                // terms, which is the shape used here. This bound only catches a
                // *bug*; whether the remaining error is affordable is a question
                // for perplexity, not for a tolerance invented here.
                assert!(rel < 0.10, "variant {v} ({label}) is broken, not just imprecise: {rel}");
            }
        }
        variant::select(None);
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
