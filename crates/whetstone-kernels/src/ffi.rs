//! Raw declarations mirroring `cuda/whetstone.h`.
//!
//! Every struct here is `#[repr(C)]` and must stay field-for-field identical to
//! its C counterpart. `layout_matches_c_abi` in the tests below guards the
//! sizes; if you add a field to the header, add it here in the same position.

use std::ffi::{c_char, c_void};

pub(crate) const WST_OK: i32 = 0;
pub(crate) const WST_ERR_CUDA: i32 = 1;
pub(crate) const WST_ERR_INVALID_ARG: i32 = 2;
pub(crate) const WST_ERR_UNSUPPORTED_ARCH: i32 = 3;
pub(crate) const WST_ERR_OOM: i32 = 4;
pub(crate) const WST_ERR_SHAPE: i32 = 5;

/// Device properties and capability flags. Mirrors `wst_device_info_t`.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// NUL-terminated device name.
    pub name: [c_char; 256],
    /// Compute capability major version.
    pub major: i32,
    /// Compute capability minor version.
    pub minor: i32,
    /// Streaming multiprocessor count.
    pub sm_count: i32,
    /// Core clock in kHz.
    pub clock_khz: i32,
    /// Memory clock in kHz.
    pub mem_clock_khz: i32,
    /// Memory bus width in bits.
    pub mem_bus_width: i32,
    /// Maximum threads per block.
    pub max_threads_per_block: i32,
    /// Maximum static shared memory per block, in bytes.
    pub max_smem_per_block: i32,
    /// Threads per warp.
    pub warp_size: i32,
    /// L2 cache size in bytes.
    pub l2_bytes: i32,
    /// Total device memory in bytes.
    pub mem_total: u64,
    /// Free device memory in bytes.
    pub mem_free: u64,
    /// Theoretical peak bandwidth in GB/s.
    pub bandwidth_gbs: f64,

    /// `sm_70+`: fp16 WMMA tensor cores.
    pub has_tensor_cores: i32,
    /// `sm_72+`: int8/int4 tensor cores.
    pub has_imma: i32,
    /// `sm_75+`: `bmma` with `.xor.popc` — Whetstone's binary path.
    pub has_bmma_xor: i32,
    /// `sm_80+`: `bmma` with `.and.popc`.
    pub has_bmma_and: i32,
    /// `sm_80+`: `cp.async` software pipelining.
    pub has_cp_async: i32,
    /// `sm_80+`: 2:4 structured-sparsity tensor cores.
    pub has_sparse_tc: i32,
    /// `sm_89+`: fp8.
    pub has_fp8: i32,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: [0; 256],
            major: 0,
            minor: 0,
            sm_count: 0,
            clock_khz: 0,
            mem_clock_khz: 0,
            mem_bus_width: 0,
            max_threads_per_block: 0,
            max_smem_per_block: 0,
            warp_size: 0,
            l2_bytes: 0,
            mem_total: 0,
            mem_free: 0,
            bandwidth_gbs: 0.0,
            has_tensor_cores: 0,
            has_imma: 0,
            has_bmma_xor: 0,
            has_bmma_and: 0,
            has_cp_async: 0,
            has_sparse_tc: 0,
            has_fp8: 0,
        }
    }
}

/// Measured throughput per arithmetic path. Mirrors `wst_probe_t`.
///
/// Negative values mean the path is unsupported on this device. These are
/// issue-rate upper bounds, not achievable GEMM rates — their value is the
/// ordering they establish between paths.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeResult {
    /// fp16 WMMA, in TFLOPS.
    pub fp16_wmma_tflops: f64,
    /// int8 WMMA, in TOPS.
    pub int8_wmma_tops: f64,
    /// int4 WMMA, in TOPS.
    pub int4_wmma_tops: f64,
    /// 1-bit BMMA (XOR + popcount), in TOPS.
    pub bin_bmma_tops: f64,
    /// `__dp4a` on CUDA cores, in TOPS.
    pub dp4a_tops: f64,
    /// Scalar `__popc` on CUDA cores, in TOPS.
    pub popc_tops: f64,
    /// 1 when `dot = K - 2*popcount(a^b)` was verified on-device.
    pub xnor_identity_ok: i32,
}

/// Struct sizes as the C compiler actually laid them out. Mirrors `wst_abi_layout_t`.
///
/// Only used by the ABI-agreement test; it exists to catch header/binding drift.
#[cfg_attr(not(test), allow(dead_code))]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AbiLayout {
    /// `sizeof(wst_device_info_t)`.
    pub device_info_size: u32,
    /// `alignof(wst_device_info_t)`.
    pub device_info_align: u32,
    /// `sizeof(wst_probe_t)`.
    pub probe_size: u32,
    /// `alignof(wst_probe_t)`.
    pub probe_align: u32,
}

extern "C" {
    pub(crate) fn wst_last_error() -> *const c_char;
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn wst_abi_layout(out: *mut AbiLayout);

    pub(crate) fn wst_device_count(out_count: *mut i32) -> i32;
    pub(crate) fn wst_device_set(ordinal: i32) -> i32;
    pub(crate) fn wst_device_info(ordinal: i32, out: *mut DeviceInfo) -> i32;
    pub(crate) fn wst_device_synchronize() -> i32;

    pub(crate) fn wst_malloc(out_ptr: *mut *mut c_void, bytes: usize) -> i32;
    pub(crate) fn wst_free(ptr: *mut c_void) -> i32;
    pub(crate) fn wst_memset(dst: *mut c_void, value: i32, bytes: usize) -> i32;
    pub(crate) fn wst_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub(crate) fn wst_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;

    pub(crate) fn wst_memcpy_d2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;

    pub(crate) fn wst_probe(out: *mut ProbeResult, iters: i32) -> i32;
    pub(crate) fn wst_bench_bandwidth(bytes: usize, reps: i32, out_gbs: *mut f64) -> i32;

    pub(crate) fn wst_gemv_int4_g128(
        qw: *const c_void,
        sz: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        in_f: i32,
        out_f: i32,
    ) -> i32;
    pub(crate) fn wst_gemv_fp16(
        w: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        in_f: i32,
        out_f: i32,
    ) -> i32;
    pub(crate) fn wst_bench_gemv(
        in_f: i32,
        out_f: i32,
        reps: i32,
        use_int4: i32,
        out_gbs: *mut f64,
        out_ms: *mut f64,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asks the C side for its own struct layout and compares. This catches the
    /// real failure mode -- a field added to the header but not to `ffi.rs`, or
    /// added in a different position -- which a hand-computed byte count would
    /// only catch by luck.
    #[test]
    fn layout_matches_c_abi() {
        let mut c = AbiLayout::default();
        // SAFETY: `c` is an owned, initialised repr(C) struct matching
        // wst_abi_layout_t; the callee only writes its four u32 fields.
        unsafe { wst_abi_layout(&mut c) };

        assert_eq!(
            std::mem::size_of::<DeviceInfo>(),
            c.device_info_size as usize,
            "DeviceInfo size differs from wst_device_info_t -- header and ffi.rs have drifted"
        );
        assert_eq!(std::mem::align_of::<DeviceInfo>(), c.device_info_align as usize);

        assert_eq!(
            std::mem::size_of::<ProbeResult>(),
            c.probe_size as usize,
            "ProbeResult size differs from wst_probe_t -- header and ffi.rs have drifted"
        );
        assert_eq!(std::mem::align_of::<ProbeResult>(), c.probe_align as usize);
    }
}
