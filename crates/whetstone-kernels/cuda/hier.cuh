/* Shared decode arithmetic for the int4 hierarchical-scale format.
 *
 * Extracted from gemv_hier.cu when the chunked (multi-token) GEMM needed the
 * identical bit tricks. Two copies of a nibble-unpacking routine that has to
 * agree exactly with a quantizer is precisely the kind of duplication this
 * project has already been burned by, so it lives in one place.
 *
 * Format recap -- see gemv_hier.cu's header for why it is shaped this way:
 *
 *     per row     : half2 (d, dmin)
 *     per group   : uint8  (ls | lm<<4), two 4-bit indices, 32 weights
 *     weight      : w = q*(d*ls) - dmin*lm
 */

#ifndef WHETSTONE_HIER_CUH
#define WHETSTONE_HIER_CUH

#include <cuda_fp16.h>
#include <cstdint>

/* Weights per (ls, lm) pair -- one uint4 exactly. */
#define HGROUP 32

/* fp16 1024.0 in both halves. At this exponent the mantissa step is exactly 1,
 * so OR-ing a value in [0,15] into the low bits adds it exactly. 1032 = 1024 + 8
 * re-centres the levels in the same operation, which keeps the half2 partial
 * sums centred instead of spending mantissa on an offset that cancels. */
#define HST_MAGIC_H2 0x64006400u
#define HST_CENTRE_F 1032.0f

/* Nibbles 2i and 2i+1 of `packed`, as a half2 holding (1024+q0, 1024+q1). */
__device__ __forceinline__ half2 hpair(uint32_t packed, int i) {
  uint32_t lo, hi;
  switch (i) {
    case 0:
      lo = packed & 0xFu;
      hi = (packed << 12) & 0x000F0000u;
      break;
    case 1:
      lo = (packed >> 8) & 0xFu;
      hi = (packed << 4) & 0x000F0000u;
      break;
    case 2:
      lo = (packed >> 16) & 0xFu;
      hi = (packed >> 4) & 0x000F0000u;
      break;
    default:
      lo = (packed >> 24) & 0xFu;
      hi = (packed >> 12) & 0x000F0000u;
      break;
  }
  const uint32_t bits = HST_MAGIC_H2 | lo | hi;
  return *(const half2 *)&bits;
}

/* Sum of a half2 pair as fp32. */
__device__ __forceinline__ float h2_sum(half2 v) {
  return __half2float(__low2half(v)) + __half2float(__high2half(v));
}

#endif /* WHETSTONE_HIER_CUH */
