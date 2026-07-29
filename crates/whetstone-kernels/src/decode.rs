//! Safe wrappers over the decode-step kernels.
//!
//! Everything here operates on a single token's activations. The shapes are
//! tiny — a few kilobytes against a quarter of a gigabyte of weights — so none
//! of these kernels is on the bandwidth critical path. They are on the *launch*
//! critical path, which is why several of them do more than their name suggests
//! (see [`rope_cache`]).
//!
//! Dtype convention across the whole decode step:
//!
//! | buffer | type | why |
//! |---|---|---|
//! | residual stream | `f32` | 24 layers of accumulation; Turing has no bf16 |
//! | GEMV input | `f16` | halves activation traffic, feeds `half2` loads |
//! | GEMV output | `f32` | reductions and residual adds stay exact |
//! | KV cache | `f16` | it is read every token, so its width is bandwidth |

use crate::ffi;
use crate::{check, DeviceBuffer, Error, Result};

/// `out = f16( x * rsqrt(mean(x^2) + eps) * w )`.
///
/// The reduction is fp32 regardless of the storage type.
pub fn rmsnorm(
    x: &DeviceBuffer<f32>,
    w: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    eps: f32,
) -> Result<()> {
    let n = x.len();
    if w.len() != n || out.len() != n {
        return Err(Error::Shape(format!(
            "rmsnorm: x[{n}], w[{}], out[{}] must agree",
            w.len(),
            out.len()
        )));
    }
    // SAFETY: lengths are checked equal above; the kernel reads and writes only
    // `n` elements of each live device allocation.
    check(unsafe { ffi::wst_rmsnorm(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), n as i32, eps) })
}

/// Precomputed rotary cos/sin, `[max_seq][head_dim/2]` each.
///
/// Built on the host in `f64` and uploaded once. Evaluating `sin`/`cos` in the
/// kernel would mean either the fast intrinsics — whose argument reduction
/// degrades past a few thousand radians, which is exactly where long contexts
/// live — or a double-precision `sincos` at 1/32 rate. The table is 1 MB at 4k
/// context and turns the whole thing into two loads.
pub struct RopeTable {
    cos: DeviceBuffer<f32>,
    sin: DeviceBuffer<f32>,
    /// Positions the table covers.
    pub max_seq: usize,
    /// Rotated pair count, `head_dim / 2`.
    pub half_dim: usize,
}

/// How the inverse frequencies are stretched for long context.
///
/// The rotation itself is identical across every architecture Whetstone
/// supports; only the frequency schedule differs, which is why this is a table
/// parameter rather than a kernel variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// No stretching. Qwen2/2.5, Mistral, Llama 2.
    None,
    /// Llama 3.1+ piecewise schedule.
    ///
    /// Wavelengths shorter than the original context are left alone, those
    /// longer than it are divided by `factor`, and the band between is blended.
    /// Getting this wrong does not crash — it degrades coherence past the
    /// original context length, which reads as the model being bad at long
    /// inputs rather than as a bug.
    Llama3 {
        /// Context multiplier.
        factor: f64,
        /// Wavelength below which frequencies are untouched, as a divisor of
        /// the original context.
        low_freq_factor: f64,
        /// Wavelength above which frequencies are fully divided.
        high_freq_factor: f64,
        /// The context length the model was trained at.
        original_max_position: usize,
    },
}

impl RopeTable {
    /// Builds the table for `max_seq` positions with no frequency scaling.
    pub fn new(max_seq: usize, head_dim: usize, theta: f64) -> Result<Self> {
        Self::with_scaling(max_seq, head_dim, theta, RopeScaling::None)
    }

    /// Builds the table for `max_seq` positions.
    pub fn with_scaling(
        max_seq: usize,
        head_dim: usize,
        theta: f64,
        scaling: RopeScaling,
    ) -> Result<Self> {
        if head_dim % 2 != 0 || head_dim == 0 {
            return Err(Error::Shape(format!("rope: head_dim {head_dim} must be even")));
        }
        let half = head_dim / 2;
        let mut cos = vec![0f32; max_seq * half];
        let mut sin = vec![0f32; max_seq * half];

        for j in 0..half {
            let mut invf = theta.powf(-(j as f64) / half as f64);
            if let RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            } = scaling
            {
                let orig = original_max_position as f64;
                let low_wl = orig / low_freq_factor;
                let high_wl = orig / high_freq_factor;
                let wavelen = 2.0 * std::f64::consts::PI / invf;
                if wavelen > low_wl {
                    // Longer than the low-frequency wavelength: divide fully.
                    invf /= factor;
                } else if wavelen > high_wl {
                    // The blend band. `smooth` runs 1 -> 0 across it.
                    let smooth = (orig / wavelen - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    invf = (1.0 - smooth) * (invf / factor) + smooth * invf;
                }
            }
            for p in 0..max_seq {
                // f64 throughout: the angle reaches ~3e4 radians at long
                // context, where an f32 argument has already lost the low bits
                // that decide the rotation.
                let a = p as f64 * invf;
                cos[p * half + j] = a.cos() as f32;
                sin[p * half + j] = a.sin() as f32;
            }
        }

        Ok(Self {
            cos: DeviceBuffer::from_slice(&cos)?,
            sin: DeviceBuffer::from_slice(&sin)?,
            max_seq,
            half_dim: half,
        })
    }

    /// Bytes resident on the device.
    pub fn bytes(&self) -> usize {
        self.cos.bytes() + self.sin.bytes()
    }
}

/// A key/value cache laid out `[kv_head][pos][head_dim]`, `f16`.
///
/// The head-major, position-contiguous order is what makes decode attention a
/// linear sweep: a warp reading one position issues a single request, and
/// consecutive positions are adjacent. With GQA the query heads sharing a KV
/// head hit the same lines, so the 7:1 grouping saves L2 traffic as well as
/// capacity.
pub struct KvCache {
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
    /// Per-slice partial softmaxes, for the sequence-split attention.
    partials: DeviceBuffer<f32>,
    /// Key/value head count.
    pub n_kv: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// Capacity in tokens.
    pub max_seq: usize,
}

impl KvCache {
    /// Allocates a cache for `layers` is handled by the caller; this is one layer.
    pub fn new(n_kv: usize, n_q: usize, head_dim: usize, max_seq: usize) -> Result<Self> {
        let n = n_kv * max_seq * head_dim;
        // SAFETY: three scalars in, a count out; no pointers involved.
        let floats =
            unsafe { ffi::wst_attn_partial_floats(n_q as i32, head_dim as i32, max_seq as i32) };
        Ok(Self {
            k: DeviceBuffer::zeros(n)?,
            v: DeviceBuffer::zeros(n)?,
            partials: DeviceBuffer::zeros(floats.max(1) as usize)?,
            n_kv,
            head_dim,
            max_seq,
        })
    }

    /// Bytes resident on the device for this layer.
    pub fn bytes(&self) -> usize {
        self.k.bytes() + self.v.bytes() + self.partials.bytes()
    }
}

/// A device-resident int32, used for the values that change every token.
///
/// The decode position and the current token id are both read by kernels rather
/// than passed as arguments, because a CUDA graph bakes its kernel arguments in
/// at instantiation. Putting them in device memory is what makes a whole
/// generation runnable as repeated launches of one graph, with the host doing
/// nothing between tokens.
pub struct DeviceCursor {
    buf: DeviceBuffer<i32>,
}

impl DeviceCursor {
    /// Allocates a cursor holding `value`.
    pub fn new(value: i32) -> Result<Self> {
        Ok(Self { buf: DeviceBuffer::from_slice(&[value])? })
    }

    /// Overwrites the cursor. Blocking, so not for use inside a graph.
    pub fn set(&self, value: i32) -> Result<()> {
        self.buf.copy_from_host(&[value])
    }

    /// Reads the cursor back. Blocking.
    pub fn get(&self) -> Result<i32> {
        let mut v = [0i32; 1];
        self.buf.copy_to_host(&mut v)?;
        Ok(v[0])
    }

    /// The underlying buffer, for kernels that write it (the argmax).
    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer<i32> {
        &mut self.buf
    }

    /// Queues `value += 1`, saturating at `max_seq`, as a kernel.
    ///
    /// A kernel rather than a host-side increment so that it can be a node of
    /// the captured graph.
    pub fn advance(&mut self, max_seq: usize) -> Result<()> {
        // SAFETY: the buffer holds exactly the one i32 the kernel updates.
        check(unsafe { ffi::wst_advance_pos(self.buf.as_mut_ptr(), max_seq as i32) })
    }
}

/// Applies rotary embedding to `q` in place and to `k` on the way into the
/// cache, and copies `v` into the cache alongside it.
///
/// Three operations in one launch because they all touch the same freshly
/// projected vectors and there is no arithmetic left to amortise — only
/// dispatch. `v` is not rotated: position enters only through the scores.
///
/// The position comes from a [`DeviceCursor`], so this call is identical on
/// every token and can be captured once into a graph.
pub fn rope_cache(
    qkv: &mut DeviceBuffer<f32>,
    cache: &mut KvCache,
    table: &RopeTable,
    n_q: usize,
    pos: &DeviceCursor,
) -> Result<()> {
    let hd = cache.head_dim;
    let want = (n_q + 2 * cache.n_kv) * hd;
    if qkv.len() != want {
        return Err(Error::Shape(format!(
            "rope_cache: qkv[{}] should be ({n_q} + 2*{}) * {hd} = {want}",
            qkv.len(),
            cache.n_kv
        )));
    }
    if table.half_dim != hd / 2 || table.max_seq < cache.max_seq {
        return Err(Error::Shape(
            "rope_cache: rotary table does not match the cache geometry".into(),
        ));
    }

    // SAFETY: every shape is validated above against the dimensions the kernel
    // indexes. The position is a live one-element device buffer, and the kernel
    // clamps it into the cache rather than trusting it -- it cannot be checked
    // here because inside a graph the host never sees the value.
    check(unsafe {
        ffi::wst_rope_cache(
            qkv.as_mut_ptr(),
            cache.k.as_mut_ptr(),
            cache.v.as_mut_ptr(),
            table.cos.as_ptr(),
            table.sin.as_ptr(),
            n_q as i32,
            cache.n_kv as i32,
            hd as i32,
            pos.buf.as_ptr(),
            cache.max_seq as i32,
        )
    })
}

/// Batch=1 GQA attention over cache entries `0..=pos`.
pub fn attn_decode(
    qkv: &DeviceBuffer<f32>,
    cache: &mut KvCache,
    out: &mut DeviceBuffer<u16>,
    n_q: usize,
    pos: &DeviceCursor,
) -> Result<()> {
    let hd = cache.head_dim;
    // The queries are the leading `n_q * hd` of the fused projection output, so
    // no offset is needed -- but the buffer is longer than the queries, and the
    // check has to say so rather than demanding an exact match.
    if qkv.len() < n_q * hd || out.len() != n_q * hd {
        return Err(Error::Shape(format!(
            "attn_decode: qkv[{}] must hold at least {n_q}*{hd} queries and out[{}] exactly that",
            qkv.len(),
            out.len()
        )));
    }
    let scale = 1.0f32 / (hd as f32).sqrt();

    // SAFETY: shapes validated above; the sequence length is derived on-device
    // from the cursor and clamped to the cache capacity by the kernel.
    check(unsafe {
        ffi::wst_attn_decode(
            qkv.as_ptr(),
            cache.k.as_ptr(),
            cache.v.as_ptr(),
            cache.partials.as_mut_ptr(),
            out.as_mut_ptr(),
            n_q as i32,
            cache.n_kv as i32,
            hd as i32,
            pos.buf.as_ptr(),
            cache.max_seq as i32,
            scale,
        )
    })
}

/// `out = f16( silu(gate) * up )`, where `gate_up` holds both halves.
pub fn swiglu(gate_up: &DeviceBuffer<f32>, out: &mut DeviceBuffer<u16>) -> Result<()> {
    let n = out.len();
    if gate_up.len() != 2 * n {
        return Err(Error::Shape(format!(
            "swiglu: gate_up[{}] should be twice out[{n}]",
            gate_up.len()
        )));
    }
    // SAFETY: the fused buffer is checked to hold both halves; the kernel reads
    // `2n` and writes `n`.
    check(unsafe { ffi::wst_swiglu(gate_up.as_ptr(), out.as_mut_ptr(), n as i32) })
}

/// Gathers row `token` of a dense fp16 embedding table into `out`.
pub fn embed_fp16(
    table: &DeviceBuffer<u16>,
    token: &DeviceCursor,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let hidden = out.len();
    if hidden == 0 || table.len() % hidden != 0 {
        return Err(Error::Shape(format!(
            "embed: a {}-element table is not a whole number of {hidden}-wide rows",
            table.len()
        )));
    }
    let rows = table.len() / hidden;
    // SAFETY: the row count is derived from the buffer's real length and passed
    // to the kernel, which clamps the device-resident token into it -- the host
    // cannot check the value because inside a graph it never sees it.
    check(unsafe {
        ffi::wst_embed_fp16(
            table.as_ptr(),
            token.buf.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            rows as i32,
        )
    })
}

/// Gathers and dequantizes row `token` of an int4-g128 embedding table.
pub fn embed_int4(
    qw: &DeviceBuffer<u32>,
    sz: &DeviceBuffer<u32>,
    token: &DeviceCursor,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let hidden = out.len();
    if hidden == 0 || hidden % crate::gemv::GROUP != 0 {
        return Err(Error::Shape(format!("embed: hidden {hidden} must be a multiple of 128")));
    }
    let rows = qw.len() / (hidden / 8);
    if sz.len() != rows * hidden / crate::gemv::GROUP {
        return Err(Error::Shape("embed: scale array does not match the table".into()));
    }
    // SAFETY: the row count is derived from the packed buffer's real length and
    // cross-checked against the scale array; the kernel clamps the
    // device-resident token into that range.
    check(unsafe {
        ffi::wst_embed_int4_g128(
            qw.as_ptr(),
            sz.as_ptr(),
            token.buf.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            rows as i32,
        )
    })
}

/// Gathers and dequantizes row `token` of an int4 hierarchical-scale table.
pub fn embed_int4_hier(
    qw: &DeviceBuffer<u32>,
    si: &DeviceBuffer<u8>,
    sb: &DeviceBuffer<u32>,
    token: &DeviceCursor,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let hidden = out.len();
    if hidden == 0 || hidden % whetstone_hgroup() != 0 {
        return Err(Error::Shape(format!("embed: hidden {hidden} must be a multiple of 32")));
    }
    let rows = qw.len() / (hidden / 8);
    if si.len() != rows * hidden / whetstone_hgroup() || sb.len() != rows {
        return Err(Error::Shape("embed: scale metadata does not match the table".into()));
    }
    // SAFETY: the row count is derived from the packed buffer's real length and
    // cross-checked against both metadata arrays; the kernel clamps the
    // device-resident token into that range, which the host cannot do because
    // inside a captured graph it never sees the value.
    check(unsafe {
        ffi::wst_embed_int4_hier(
            qw.as_ptr(),
            si.as_ptr(),
            sb.as_ptr(),
            token.buf.as_ptr(),
            out.as_mut_ptr(),
            hidden as i32,
            rows as i32,
        )
    })
}

const fn whetstone_hgroup() -> usize {
    crate::gemv::HGROUP
}

/// Writes the index of the largest logit into a one-element device buffer.
///
/// Stays on the device: the logit vector is 608 KB for Qwen2.5, and copying it
/// to the host to run `max` would cost more than the reduction.
pub fn argmax(logits: &DeviceBuffer<f32>, out: &mut DeviceBuffer<i32>) -> Result<()> {
    if out.len() != 1 {
        return Err(Error::Shape("argmax: output buffer must hold exactly one index".into()));
    }
    if logits.is_empty() {
        return Err(Error::Shape("argmax: empty logit vector".into()));
    }
    // SAFETY: the output holds exactly the one i32 the kernel writes, and the
    // input length is passed as the bound the kernel scans.
    check(unsafe { ffi::wst_argmax(logits.as_ptr(), out.as_mut_ptr(), logits.len() as i32) })
}

/// A captured, instantiated decode step.
///
/// Capture collapses ~250 kernel launches into one. What it costs is that every
/// per-token value has to live in device memory rather than in a kernel
/// argument — see [`DeviceCursor`] — because a graph fixes its arguments at
/// instantiation.
///
/// Nothing may allocate, copy synchronously, or synchronise between
/// [`Graph::capture`]'s two halves. Whetstone's argmax and NLL kernels lazily
/// allocate a few bytes of reduction scratch on first use, so the caller must
/// have run at least one ordinary decode step before capturing.
pub struct Graph {
    exec: *mut std::ffi::c_void,
    /// Kernels captured, for reporting.
    pub launches: usize,
}

// SAFETY: a `cudaGraphExec_t` is an opaque handle with no thread affinity, and
// this type is not `Clone`, so ownership stays unique.
unsafe impl Send for Graph {}

impl Graph {
    /// Captures everything `body` queues, and instantiates it.
    ///
    /// `body` must not synchronise, allocate, or perform a blocking copy. It
    /// reports how many launches it issued, purely so the saving is visible.
    pub fn capture(body: impl FnOnce() -> Result<usize>) -> Result<Self> {
        // SAFETY: no arguments; begins capture on this thread's default stream.
        check(unsafe { ffi::wst_graph_capture_begin() })?;

        let launches = match body() {
            Ok(n) => n,
            Err(e) => {
                // Capture must be ended even on failure, or the stream stays in
                // capture mode and every later launch fails with a confusing
                // error a long way from here.
                let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
                // SAFETY: valid out-pointer; the result is discarded because the
                // original error is the one worth reporting.
                let _ = unsafe { ffi::wst_graph_capture_end(&mut exec) };
                if !exec.is_null() {
                    // SAFETY: `exec` was just produced by capture_end.
                    let _ = unsafe { ffi::wst_graph_destroy(exec) };
                }
                return Err(e);
            }
        };

        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `exec` is a valid out-pointer we exclusively own; on failure
        // the callee leaves it null.
        check(unsafe { ffi::wst_graph_capture_end(&mut exec) })?;
        if exec.is_null() {
            return Err(Error::Cuda("graph capture produced nothing".into()));
        }
        Ok(Self { exec, launches })
    }

    /// Queues the whole captured step. Asynchronous.
    pub fn launch(&self) -> Result<()> {
        // SAFETY: `self.exec` is a live instantiated graph, non-null by
        // construction and destroyed only in Drop.
        check(unsafe { ffi::wst_graph_launch(self.exec) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // SAFETY: destroyed exactly once (Drop runs once and the type is
            // not Clone). Teardown errors are not actionable.
            let _ = unsafe { ffi::wst_graph_destroy(self.exec) };
            self.exec = std::ptr::null_mut();
        }
    }
}

/// A stream-ordered timestamp.
///
/// Recording one costs a marker in the stream, not a host stall — which is the
/// whole point. Profiling by synchronising between stages measures the
/// synchronisation: it broke the pipeline here badly enough to make a 448-byte
/// embedding gather look like half a millisecond of work.
pub struct Event {
    ev: *mut std::ffi::c_void,
}

// SAFETY: a `cudaEvent_t` is an opaque handle with no thread affinity, and this
// type is not `Clone`, so ownership stays unique.
unsafe impl Send for Event {}

impl Event {
    /// Creates a timing event.
    pub fn new() -> Result<Self> {
        let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: valid out-pointer we exclusively own; null on failure.
        check(unsafe { ffi::wst_event_create(&mut ev) })?;
        Ok(Self { ev })
    }

    /// Queues a timestamp. Asynchronous.
    pub fn record(&self) -> Result<()> {
        // SAFETY: `self.ev` is a live event, non-null by construction.
        check(unsafe { ffi::wst_event_record(self.ev) })
    }

    /// Milliseconds from `self` to `later`. Both must have completed.
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: both events are live; `ms` is an owned, initialised f32.
        check(unsafe { ffi::wst_event_elapsed_ms(self.ev, later.ev, &mut ms) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.ev.is_null() {
            // SAFETY: destroyed exactly once; teardown errors are not actionable.
            let _ = unsafe { ffi::wst_event_destroy(self.ev) };
            self.ev = std::ptr::null_mut();
        }
    }
}

/// Blocks until everything queued on the graph stream has finished.
pub fn stream_sync() -> Result<()> {
    // SAFETY: no arguments; synchronises this thread's default stream.
    check(unsafe { ffi::wst_stream_sync() })
}

/// Accumulates `-log p(target)` into `acc[0]` and the position count into
/// `acc[1]`, without leaving the device.
///
/// A wikitext-2 run is ~41,000 forward passes. Copying a scalar back after each
/// would put a synchronising 4-byte transfer inside a loop that otherwise never
/// blocks, and a synchronising transfer costs far more than the reduction it is
/// retrieving.
pub fn nll(logits: &DeviceBuffer<f32>, target: u32, acc: &mut DeviceBuffer<f32>) -> Result<()> {
    if acc.len() < 2 {
        return Err(Error::Shape("nll: accumulator needs two floats (sum, count)".into()));
    }
    if target as usize >= logits.len() {
        return Err(Error::Shape(format!(
            "nll: target {target} outside a vocabulary of {}",
            logits.len()
        )));
    }
    // SAFETY: the target is bounds-checked against the logit count, and the
    // accumulator is verified to hold the two floats the kernel writes.
    check(unsafe {
        ffi::wst_nll(logits.as_ptr(), target as i32, acc.as_mut_ptr(), logits.len() as i32)
    })
}
