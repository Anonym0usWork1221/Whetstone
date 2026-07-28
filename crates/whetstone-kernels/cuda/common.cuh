/* Internal device-side helpers. Not part of the C ABI. */

#ifndef WHETSTONE_COMMON_CUH
#define WHETSTONE_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#include "whetstone.h"

/* ------------------------------------------------------- error propagation */

/* Per-thread error string, so a failure can be reported to Rust without the
 * process aborting. */
extern thread_local char wst_err_buf[512];

inline void wst_set_error(const char *where, cudaError_t e) {
  snprintf(wst_err_buf, sizeof(wst_err_buf), "%s: %s (%s)", where,
           cudaGetErrorString(e), cudaGetErrorName(e));
}

inline void wst_set_error_msg(const char *msg) {
  snprintf(wst_err_buf, sizeof(wst_err_buf), "%s", msg);
}

/* Bail out of a C-ABI function on CUDA failure. Never throws. */
#define WST_TRY(expr)                                                          \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      wst_set_error(#expr, _e);                                                \
      return WST_ERR_CUDA;                                                     \
    }                                                                          \
  } while (0)

/* Check for an async kernel fault. Costs a sync; use at ABI boundaries only. */
#define WST_TRY_KERNEL(name)                                                   \
  do {                                                                         \
    cudaError_t _e = cudaGetLastError();                                       \
    if (_e != cudaSuccess) {                                                   \
      wst_set_error(name " (launch)", _e);                                     \
      return WST_ERR_CUDA;                                                     \
    }                                                                          \
  } while (0)

#define WST_REQUIRE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      wst_set_error_msg(msg);                                                  \
      return WST_ERR_INVALID_ARG;                                              \
    }                                                                          \
  } while (0)

/* ------------------------------------------------------------ arch gating */

/* Whetstone compiles for exactly one architecture (build.rs). These constants
 * let host code reason about it without re-deriving the boundaries. */
#ifndef WHETSTONE_ARCH
#define WHETSTONE_ARCH 75
#endif

#define WST_ARCH_HAS_BMMA_XOR (WHETSTONE_ARCH >= 75)
#define WST_ARCH_HAS_BMMA_AND (WHETSTONE_ARCH >= 80)
#define WST_ARCH_HAS_CP_ASYNC (WHETSTONE_ARCH >= 80)
#define WST_ARCH_HAS_IMMA     (WHETSTONE_ARCH >= 72)

/* ------------------------------------------------------------ device utils */

#define WST_WARP 32
#define WST_FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = WST_WARP / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(WST_FULL_MASK, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = WST_WARP / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(WST_FULL_MASK, v, off));
  return v;
}

__device__ __forceinline__ int warp_reduce_sum_i32(int v) {
#pragma unroll
  for (int off = WST_WARP / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(WST_FULL_MASK, v, off);
  return v;
}

/* Block-wide sum via shared memory. `smem` must hold blockDim.x/32 floats. */
__device__ __forceinline__ float block_reduce_sum(float v, float *smem) {
  const int lane = threadIdx.x & (WST_WARP - 1);
  const int wid = threadIdx.x / WST_WARP;
  const int nwarps = (blockDim.x + WST_WARP - 1) / WST_WARP;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  v = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
  if (wid == 0) v = warp_reduce_sum(v);
  return v;
}

/* ------------------------------------------------------ the XNOR identity */

/* For a, b in {-1,+1}^K packed as bits {1,0}:
 *
 *     dot(a, b) = K - 2 * popcount(a XOR b)
 *
 * Derivation: a_i * b_i = +1 when bits agree, -1 when they differ. If d is the
 * number of differing bits then dot = (K - d) - d = K - 2d, and d = popcount of
 * the XOR. This is the whole basis of the binary tensor-core path.
 *
 * Verified on-device in probe.cu. */
__device__ __forceinline__ int xnor_dot_from_popc(int popc, int K) {
  return K - 2 * popc;
}

/* Same identity applied to a bmma accumulator, which returns the raw popcount
 * of the XOR over the K dimension. */
__device__ __forceinline__ float bmma_acc_to_dot(int acc, int K, float scale) {
  return scale * (float)(K - 2 * acc);
}

/* --------------------------------------------------------- timing helper */

struct WstTimer {
  cudaEvent_t start, stop;
  bool ok;

  WstTimer() : ok(true) {
    if (cudaEventCreate(&start) != cudaSuccess) ok = false;
    if (cudaEventCreate(&stop) != cudaSuccess) ok = false;
  }
  ~WstTimer() {
    if (ok) { cudaEventDestroy(start); cudaEventDestroy(stop); }
  }
  void tic() { cudaEventRecord(start); }
  float toc_ms() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};

#endif /* WHETSTONE_COMMON_CUH */
