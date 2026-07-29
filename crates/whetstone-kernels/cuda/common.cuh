/* Internal device-side helpers. Not part of the C ABI. */

#ifndef WHETSTONE_COMMON_CUH
#define WHETSTONE_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

/* Whetstone builds a **fat binary**: build.rs emits one device-code image per
 * supported architecture plus a PTX tail, so one archive runs on Pascal through
 * Hopper and JITs onto anything newer.
 *
 * That makes "which architecture is this" two different questions, and
 * conflating them is how a fat binary fails:
 *
 *   - **device code** is compiled once per architecture, so the answer is a
 *     property of the *compilation pass*. It must key off `__CUDA_ARCH__`, which
 *     nvcc redefines for each pass, and which is **undefined in the host pass**.
 *   - **host code** decides at run time, against whatever card is installed. A
 *     `#if` cannot answer it at all; it needs `wst_device_cc()`.
 *
 * Gating device code on a build-time -D is what the single-arch build did, and
 * it would silently compile Turing-only instructions into a Pascal image. */
#ifndef WHETSTONE_ARCH
#define WHETSTONE_ARCH 75 /* the lowest arch in the fat binary; reporting only */
#endif

/* Device-side predicates. Zero in the host pass, deliberately: a `#if
 * WST_DEV_HAS_*` around host code is always a bug, and evaluating to zero there
 * makes it a loud one rather than an architecture-dependent one. */
#if defined(__CUDA_ARCH__)
#define WST_DEV_HAS_DP4A     (__CUDA_ARCH__ >= 610)
#define WST_DEV_HAS_WMMA     (__CUDA_ARCH__ >= 700)
#define WST_DEV_HAS_IMMA     (__CUDA_ARCH__ >= 720)
/* Sub-byte (`s4`) and single-bit (`b1`) WMMA are a Turing-through-Hopper
 * family: introduced at sm_75, deprecated after, and **not** part of the
 * Blackwell tensor-core ISA. The gate therefore needs a ceiling as well as a
 * floor, or the sm_100/sm_120 passes of an `all` build on CUDA 12.8+ fail in
 * ptxas. Unverified here -- this toolkit is 12.0 and tops out at compute_90 --
 * so the bound is set conservatively to the range that is known to compile. */
#define WST_DEV_HAS_BMMA_XOR (__CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 1000)
#define WST_DEV_HAS_BMMA_AND (__CUDA_ARCH__ >= 800)
#define WST_DEV_HAS_CP_ASYNC (__CUDA_ARCH__ >= 800)
#else
#define WST_DEV_HAS_DP4A     0
#define WST_DEV_HAS_WMMA     0
#define WST_DEV_HAS_IMMA     0
#define WST_DEV_HAS_BMMA_XOR 0
#define WST_DEV_HAS_BMMA_AND 0
#define WST_DEV_HAS_CP_ASYNC 0
#endif

/* The installed device's compute capability as major*10 + minor, or 0 if there
 * is no usable device. Host-side; this is the runtime half of the pair above. */
inline int wst_device_cc() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
    return 0;
  if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
    return 0;
  return major * 10 + minor;
}

#define WST_HOST_HAS_DP4A     (wst_device_cc() >= 61)
#define WST_HOST_HAS_WMMA     (wst_device_cc() >= 70)
#define WST_HOST_HAS_IMMA     (wst_device_cc() >= 72)
#define WST_HOST_HAS_BMMA_XOR (wst_device_cc() >= 75 && wst_device_cc() < 100)
#define WST_HOST_HAS_BMMA_AND (wst_device_cc() >= 80)
#define WST_HOST_HAS_CP_ASYNC (wst_device_cc() >= 80)

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

/* ------------------------------------------------------------- QK-RMSNorm */

/* Per-head RMSNorm on the query and key vectors, before RoPE.
 *
 * Qwen3, OLMo2 and Gemma2 normalise each *head's* vector rather than the
 * residual stream, with a learned gain of `head_dim` entries shared by every
 * head. It sits between the projection and the rotation, so it has to run
 * there or not at all.
 *
 * Folded into the RoPE kernels rather than given its own launch, because the
 * rotation already holds the whole head vector in registers two elements per
 * thread -- lane j owns element j and element j+halfd. The norm therefore costs
 * one block reduction and no extra pass over memory. A separate kernel would
 * re-read q and k, and at decode the launch itself would cost more than the
 * arithmetic.
 *
 * Every thread in the block must call this: it contains barriers. Pass
 * `active = false` for the padding lanes that exist only to complete a warp --
 * they contribute nothing and their x1/x2 come back unchanged.
 *
 * The reduction is over `hd` elements but a lane holds two, so the caller's
 * block must cover exactly halfd lanes plus warp padding. */
__device__ __forceinline__ void wst_qk_head_norm(float &x1, float &x2,
                                                 const half *__restrict__ w, int j,
                                                 int halfd, int hd, float eps,
                                                 bool active) {
  __shared__ float warp_sums[1024 / WST_WARP];
  __shared__ float inv_rms;

  float acc = active ? fmaf(x1, x1, x2 * x2) : 0.0f;
  acc = block_reduce_sum(acc, warp_sums);

  if (threadIdx.x == 0) inv_rms = rsqrtf(acc / (float)hd + eps);
  __syncthreads();

  if (!active) return;
  const float s = inv_rms;
  x1 = x1 * s * __half2float(w[j]);
  x2 = x2 * s * __half2float(w[j + halfd]);
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
