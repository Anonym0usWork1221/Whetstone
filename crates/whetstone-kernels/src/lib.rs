//! Safe Rust bindings over Whetstone's CUDA kernels.
//!
//! This crate is the only place in Whetstone permitted to contain `unsafe`
//! outside a `// SAFETY:` justification. Everything above it works with owned,
//! typed device buffers that free themselves.
//!
//! # Design
//!
//! The C ABI in `cuda/whetstone.h` returns status codes rather than aborting,
//! so a CUDA failure surfaces as a Rust [`Error`] instead of killing the
//! process. Device allocations are owned by [`DeviceBuffer<T>`], which is
//! typed, length-checked, and released on drop.

#![deny(missing_docs)]

use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::marker::PhantomData;

pub mod decode;
mod ffi;
pub mod gemv;

pub use decode::{
    argmax, attn_decode, embed_fp16, embed_int4, nll, rmsnorm, rope_cache, stream_sync, swiglu,
    DeviceCursor, Event, Graph, KvCache, RopeTable,
};
pub use ffi::{DeviceInfo, ProbeResult};
pub use gemv::{bench_gemv, gemv_fp16, gemv_fp16_ex, variant, GemvBench, QuantLinear, GROUP};

/// Errors crossing the CUDA boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A CUDA API call failed. Carries the driver's message.
    #[error("cuda error: {0}")]
    Cuda(String),
    /// A pointer or scalar argument failed validation before dispatch.
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    /// The device lacks a capability this kernel requires.
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    /// Device allocation failed.
    #[error("out of device memory: {0}")]
    Oom(String),
    /// Tensor shapes are incompatible.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// No CUDA device is present.
    #[error("no CUDA device available")]
    NoDevice,
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Reads the thread-local error message set by the CUDA layer.
fn last_error() -> String {
    // SAFETY: wst_last_error returns a pointer to a thread-local, NUL-terminated
    // buffer that is never null and outlives this borrow (it is `thread_local`
    // storage with static duration, not a heap allocation we could race on).
    unsafe {
        let p: *const c_char = ffi::wst_last_error();
        if p.is_null() {
            return "unknown error".into();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Maps an ABI status code to a Rust result, attaching the driver message.
fn check(status: i32) -> Result<()> {
    match status {
        ffi::WST_OK => Ok(()),
        ffi::WST_ERR_CUDA => Err(Error::Cuda(last_error())),
        ffi::WST_ERR_INVALID_ARG => Err(Error::InvalidArg(last_error())),
        ffi::WST_ERR_UNSUPPORTED_ARCH => Err(Error::UnsupportedArch(last_error())),
        ffi::WST_ERR_OOM => Err(Error::Oom(last_error())),
        ffi::WST_ERR_SHAPE => Err(Error::Shape(last_error())),
        other => Err(Error::Cuda(format!("unknown status {other}: {}", last_error()))),
    }
}

// ---------------------------------------------------------------- device

/// A CUDA device, and the capability facts Whetstone dispatches on.
#[derive(Debug, Clone)]
pub struct Device {
    ordinal: i32,
    info: DeviceInfo,
}

impl Device {
    /// Number of CUDA devices visible to the process.
    pub fn count() -> Result<i32> {
        let mut n = 0i32;
        // SAFETY: `n` is a valid, aligned, initialised i32 we exclusively own.
        check(unsafe { ffi::wst_device_count(&mut n) })?;
        Ok(n)
    }

    /// Selects a device and captures its properties.
    pub fn new(ordinal: i32) -> Result<Self> {
        if Self::count()? == 0 {
            return Err(Error::NoDevice);
        }
        // SAFETY: FFI call with a plain scalar; validated by the callee.
        check(unsafe { ffi::wst_device_set(ordinal) })?;

        let mut info = DeviceInfo::default();
        // SAFETY: `info` is a fully-initialised, exclusively-owned repr(C) struct
        // matching wst_device_info_t; the callee only writes within it.
        check(unsafe { ffi::wst_device_info(ordinal, &mut info) })?;

        Ok(Self { ordinal, info })
    }

    /// Selects device 0.
    pub fn default_device() -> Result<Self> {
        Self::new(0)
    }

    /// Device ordinal.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Captured device properties.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Compute capability as a two-digit integer (`75` for Turing).
    pub fn compute_capability(&self) -> i32 {
        self.info.major * 10 + self.info.minor
    }

    /// Human-readable device name.
    pub fn name(&self) -> String {
        let bytes: Vec<u8> = self
            .info
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Theoretical peak memory bandwidth in GB/s.
    ///
    /// For batch=1 decode this, not any TOPS figure, sets the speed ceiling:
    /// `tok/s <= bandwidth / bytes_read_per_token`.
    pub fn bandwidth_gbs(&self) -> f64 {
        self.info.bandwidth_gbs
    }

    /// Blocks until all previously issued work completes.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: no arguments; the callee only synchronises the current context.
        check(unsafe { ffi::wst_device_synchronize() })
    }

    /// Measures achieved streaming-read bandwidth.
    ///
    /// Whetstone tunes against the measured number rather than the spec sheet,
    /// because attainable bandwidth is typically 75-90% of peak.
    pub fn measure_bandwidth(&self, bytes: usize, reps: i32) -> Result<f64> {
        let mut gbs = 0.0f64;
        // SAFETY: `gbs` is an owned, initialised f64; scalars are validated by the callee.
        check(unsafe { ffi::wst_bench_bandwidth(bytes, reps, &mut gbs) })?;
        Ok(gbs)
    }

    /// Microbenchmarks every arithmetic path the device supports.
    ///
    /// Unsupported paths report a negative value.
    pub fn probe(&self, iters: i32) -> Result<ProbeResult> {
        let mut r = ProbeResult::default();
        // SAFETY: `r` is an owned, initialised repr(C) struct matching wst_probe_t.
        check(unsafe { ffi::wst_probe(&mut r, iters) })?;
        Ok(r)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let i = &self.info;
        write!(
            f,
            "{} (sm_{}{}, {} SMs, {:.1} GB, {:.0} GB/s)",
            self.name(),
            i.major,
            i.minor,
            i.sm_count,
            i.mem_total as f64 / 1e9,
            i.bandwidth_gbs
        )
    }
}

// ---------------------------------------------------------------- memory

/// An owned, typed device allocation. Freed on drop.
pub struct DeviceBuffer<T: Copy> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: a device pointer is just an address; it carries no thread affinity.
// Ownership is unique (no Clone), so moving it across threads cannot alias.
unsafe impl<T: Copy + Send> Send for DeviceBuffer<T> {}
// SAFETY: `&DeviceBuffer` exposes only reads, which are safe to share.
unsafe impl<T: Copy + Sync> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocates `len` uninitialised elements on the device.
    pub fn new(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArg("allocation size overflowed usize".into()))?;

        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out-parameter we exclusively own; on failure
        // the callee leaves it null and we propagate the error without using it.
        check(unsafe { ffi::wst_malloc(&mut ptr, bytes) })?;

        Ok(Self { ptr, len, _marker: PhantomData })
    }

    /// Allocates `len` elements and zeroes them.
    pub fn zeros(len: usize) -> Result<Self> {
        let b = Self::new(len)?;
        b.fill_bytes(0)?;
        Ok(b)
    }

    /// Allocates and uploads `data`.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let b = Self::new(data.len())?;
        b.copy_from_host(data)?;
        Ok(b)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Raw device pointer, for passing to kernels.
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    /// Raw mutable device pointer, for passing to kernels.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    /// Sets every byte of the allocation to `value`.
    pub fn fill_bytes(&self, value: i32) -> Result<()> {
        // SAFETY: `self.ptr` owns `self.bytes()` valid device bytes.
        check(unsafe { ffi::wst_memset(self.ptr, value, self.bytes()) })
    }

    /// Uploads `src`, which must exactly match this buffer's length.
    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(Error::Shape(format!(
                "host slice has {} elements, device buffer has {}",
                src.len(),
                self.len
            )));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: lengths are checked equal above, so `self.bytes()` bytes are
        // readable from `src` and writable at `self.ptr`.
        check(unsafe {
            ffi::wst_memcpy_h2d(self.ptr, src.as_ptr() as *const c_void, self.bytes())
        })
    }

    /// Downloads into `dst`, which must exactly match this buffer's length.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(Error::Shape(format!(
                "host slice has {} elements, device buffer has {}",
                dst.len(),
                self.len
            )));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: lengths are checked equal above, so `self.bytes()` bytes are
        // writable at `dst` and readable from `self.ptr`.
        check(unsafe {
            ffi::wst_memcpy_d2h(dst.as_mut_ptr() as *mut c_void, self.ptr, self.bytes())
        })
    }

    /// Copies from another device buffer of the same length, without a host
    /// round trip.
    pub fn copy_from_device(&self, src: &DeviceBuffer<T>) -> Result<()> {
        if src.len != self.len {
            return Err(Error::Shape(format!(
                "source has {} elements, destination has {}",
                src.len, self.len
            )));
        }
        if self.len == 0 {
            return Ok(());
        }
        // SAFETY: lengths are checked equal above, so `self.bytes()` bytes are
        // readable from `src.ptr` and writable at `self.ptr`; both are live
        // device allocations owned by their respective buffers.
        check(unsafe { ffi::wst_memcpy_d2d(self.ptr, src.ptr, self.bytes()) })
    }

    /// Downloads the whole buffer into a fresh `Vec`.
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Default,
    {
        let mut out = vec![T::default(); self.len];
        self.copy_to_host(&mut out)?;
        Ok(out)
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` came from wst_malloc, is freed exactly once
            // (Drop runs once, and the type is not Clone/Copy), and is not used
            // afterwards. Errors during teardown are not actionable.
            let _ = unsafe { ffi::wst_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl<T: Copy> fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("len", &self.len)
            .field("bytes", &self.bytes())
            .field("elem", &std::any::type_name::<T>())
            .finish()
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests are skipped rather than failed when no GPU is present, so CI on a
    /// CPU-only runner stays green.
    fn gpu() -> Option<Device> {
        Device::default_device().ok()
    }

    #[test]
    fn device_reports_plausible_properties() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        println!("device: {d}");
        assert!(d.info().sm_count > 0);
        assert!(d.bandwidth_gbs() > 1.0);
        assert!(d.compute_capability() >= 50);
    }

    #[test]
    fn capability_flags_match_compute_capability() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let cc = d.compute_capability();
        let i = d.info();

        // The sm_75 boundaries Whetstone's kernel selection depends on.
        assert_eq!(i.has_bmma_xor != 0, cc >= 75, "bmma .xor.popc is sm_75+");
        assert_eq!(i.has_bmma_and != 0, cc >= 80, "bmma .and.popc is sm_80+");
        assert_eq!(i.has_cp_async != 0, cc >= 80, "cp.async is sm_80+");
        assert_eq!(i.has_sparse_tc != 0, cc >= 80, "2:4 sparsity is sm_80+");
    }

    #[test]
    fn roundtrip_through_device_memory() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let src: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5).collect();
        let buf = DeviceBuffer::from_slice(&src).unwrap();
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.bytes(), 4096 * 4);
        assert_eq!(buf.to_vec().unwrap(), src);
    }

    #[test]
    fn length_mismatch_is_rejected_not_ub() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let buf = DeviceBuffer::<f32>::zeros(64).unwrap();
        assert!(matches!(buf.copy_from_host(&[1.0f32; 32]), Err(Error::Shape(_))));
        let mut small = [0.0f32; 16];
        assert!(matches!(buf.copy_to_host(&mut small), Err(Error::Shape(_))));
    }

    #[test]
    fn zeros_are_actually_zero() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let buf = DeviceBuffer::<f32>::zeros(1024).unwrap();
        assert!(buf.to_vec().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn oversized_allocation_errors_cleanly() {
        if gpu().is_none() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // Far beyond any consumer GPU: must be a clean Err, never a panic/abort.
        let r = DeviceBuffer::<f32>::new(1 << 42);
        assert!(r.is_err(), "expected OOM error, got a buffer");
    }

    #[test]
    fn xnor_identity_holds_on_device() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let p = d.probe(2000).unwrap();
        // The whole binary path rests on dot = K - 2*popcount(a^b).
        assert_eq!(p.xnor_identity_ok, 1, "XNOR dot identity failed on device");
    }

    #[test]
    fn probe_orders_arithmetic_paths_as_expected() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let p = d.probe(20000).unwrap();
        println!(
            "fp16 {:.1} TFLOPS | int8 {:.1} | int4 {:.1} | b1 {:.1} | dp4a {:.1} | popc {:.1} TOPS",
            p.fp16_wmma_tflops, p.int8_wmma_tops, p.int4_wmma_tops,
            p.bin_bmma_tops, p.dp4a_tops, p.popc_tops
        );
        assert!(p.fp16_wmma_tflops > 0.0);

        if d.info().has_bmma_xor != 0 {
            assert!(p.bin_bmma_tops > p.fp16_wmma_tflops,
                    "binary tensor core should beat fp16");
            // bmma beats a hand-rolled __popc loop, but only by ~4x once popc
            // is counted correctly (32 lanes each doing 32 binary MACs, not 32
            // per warp). An earlier version of this probe undercounted popc and
            // dp4a by 32x, which made the gap look like 69x and supported the
            // much stronger claim that CUDA-core bit arithmetic loses to fp16.
            // It does not: corrected popc is several times fp16's rate.
            assert!(p.bin_bmma_tops > p.popc_tops,
                    "bmma should still beat scalar __popc");
            assert!(p.popc_tops > p.fp16_wmma_tflops,
                    "corrected popc should exceed fp16 throughput");
        }
    }

    #[test]
    fn measured_bandwidth_is_within_reason_of_peak() {
        let Some(d) = gpu() else {
            eprintln!("skip: no CUDA device");
            return;
        };
        let gbs = d.measure_bandwidth(256 << 20, 20).unwrap();
        let peak = d.bandwidth_gbs();
        println!("measured {gbs:.0} GB/s of {peak:.0} GB/s peak ({:.0}%)", gbs / peak * 100.0);
        assert!(gbs > 0.3 * peak, "achieved bandwidth implausibly low: {gbs:.0} GB/s");
        assert!(gbs < 1.15 * peak, "achieved bandwidth exceeds peak: {gbs:.0} GB/s");
    }
}
