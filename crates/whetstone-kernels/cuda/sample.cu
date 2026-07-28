/* Token selection.
 *
 * The logit vector is 151936 floats -- 608 KB -- which is small next to the
 * ~260 MB a decode step streams, but not small next to a 3 ms token: copying it
 * to the host to pick an argmax costs ~100 us over PCIe, or 3% of the budget,
 * for a reduction the GPU finishes in twenty microseconds. So greedy decode
 * stays on the device and only the chosen id crosses the bus.
 *
 * The reduction packs value and index into one 64-bit key so a single atomicMax
 * does the whole cross-block combine -- no scratch array, no second pass, and a
 * deterministic result. Deterministic matters more than it looks: a tie broken
 * differently between two runs turns a reproducible generation into a flaky one,
 * and ties are common once a distribution saturates.
 */

#include "common.cuh"
#include <cuda_fp16.h>

#define SAMPLE_THREADS 256
#define SAMPLE_BLOCKS 256

/* Order-preserving float -> uint32.
 *
 * IEEE-754 floats compare correctly as sign-magnitude integers, not as two's
 * complement. Flipping the sign bit for positives and inverting everything for
 * negatives yields a key whose unsigned order matches the float order, for every
 * value including the infinities. */
__device__ __forceinline__ uint32_t float_key(float v) {
  const uint32_t b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/* Value in the high half, complemented index in the low half, so that on a tie
 * the *smallest* index wins the max. */
__device__ __forceinline__ unsigned long long pack(float v, int idx) {
  return ((unsigned long long)float_key(v) << 32) | (unsigned long long)(~(uint32_t)idx);
}

__global__ __launch_bounds__(SAMPLE_THREADS) void argmax_kernel(
    const float *__restrict__ logits, unsigned long long *__restrict__ acc, int n) {

  __shared__ unsigned long long warp_best[SAMPLE_THREADS / WST_WARP];

  unsigned long long best = 0ull;
  const int stride = gridDim.x * SAMPLE_THREADS;
  for (int i = blockIdx.x * SAMPLE_THREADS + threadIdx.x; i < n; i += stride) {
    const unsigned long long c = pack(logits[i], i);
    if (c > best) best = c;
  }

  const int lane = threadIdx.x % WST_WARP;
  const int warp = threadIdx.x / WST_WARP;

#pragma unroll
  for (int off = WST_WARP / 2; off > 0; off >>= 1) {
    const unsigned long long o = __shfl_xor_sync(WST_FULL_MASK, best, off);
    if (o > best) best = o;
  }
  if (lane == 0) warp_best[warp] = best;
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned long long b = warp_best[0];
    for (int w = 1; w < SAMPLE_THREADS / WST_WARP; ++w)
      if (warp_best[w] > b) b = warp_best[w];
    atomicMax(acc, b);
  }
}

__global__ void argmax_extract_kernel(const unsigned long long *__restrict__ acc,
                                      int32_t *__restrict__ out) {
  *out = (int32_t)(~(uint32_t)(*acc & 0xFFFFFFFFull));
}

/* One permanent 8-byte device allocation, created on first use.
 *
 * The alternative -- allocating and freeing per call -- puts two synchronising
 * driver calls inside the token loop, which is the same class of mistake as the
 * cudaGetDeviceProperties-per-launch that once cost this project 100x on the
 * GEMV. Whetstone drives one device from one thread, so a static is sufficient;
 * a multi-device build would key this by ordinal. */
static unsigned long long *argmax_scratch() {
  static unsigned long long *p = nullptr;
  if (p == nullptr) {
    if (cudaMalloc((void **)&p, sizeof(unsigned long long)) != cudaSuccess) p = nullptr;
  }
  return p;
}

/* ------------------------------------------------------------- perplexity */

/* Negative log-likelihood of one target token: logsumexp(logits) - logits[t].
 *
 * The logsumexp uses the same online recurrence as the attention softmax, so the
 * whole vocabulary is read once rather than twice (max pass, then exp pass).
 * That halves the traffic of the obvious implementation, and at 40,940
 * evaluation positions the difference is 25 GB.
 *
 * The result accumulates into a device float. A perplexity run is ~41,000
 * forward passes; copying a scalar back after each one would put a
 * synchronising 4-byte transfer in a loop that otherwise never blocks. */
__global__ __launch_bounds__(SAMPLE_THREADS) void nll_partial_kernel(
    const float *__restrict__ logits, float2 *__restrict__ partial, int n) {

  __shared__ float sm[SAMPLE_THREADS / WST_WARP];
  __shared__ float sl[SAMPLE_THREADS / WST_WARP];

  float m = -INFINITY, l = 0.0f;
  const int stride = gridDim.x * SAMPLE_THREADS;
  for (int i = blockIdx.x * SAMPLE_THREADS + threadIdx.x; i < n; i += stride) {
    const float v = logits[i];
    const float m_new = fmaxf(m, v);
    l = l * __expf(m - m_new) + __expf(v - m_new);
    m = m_new;
  }

  const int lane = threadIdx.x % WST_WARP;
  const int warp = threadIdx.x / WST_WARP;

#pragma unroll
  for (int off = WST_WARP / 2; off > 0; off >>= 1) {
    const float om = __shfl_xor_sync(WST_FULL_MASK, m, off);
    const float ol = __shfl_xor_sync(WST_FULL_MASK, l, off);
    const float mn = fmaxf(m, om);
    l = l * __expf(m - mn) + ol * __expf(om - mn);
    m = mn;
  }
  if (lane == 0) { sm[warp] = m; sl[warp] = l; }
  __syncthreads();

  if (threadIdx.x == 0) {
    float M = sm[0], L = sl[0];
    for (int w = 1; w < SAMPLE_THREADS / WST_WARP; ++w) {
      const float mn = fmaxf(M, sm[w]);
      L = L * __expf(M - mn) + sl[w] * __expf(sm[w] - mn);
      M = mn;
    }
    partial[blockIdx.x] = make_float2(M, L);
  }
}

__global__ void nll_combine_kernel(const float2 *__restrict__ partial, int nblocks,
                                   const float *__restrict__ logits, int target,
                                   float *__restrict__ acc) {
  float M = partial[0].x, L = partial[0].y;
  for (int b = 1; b < nblocks; ++b) {
    const float mn = fmaxf(M, partial[b].x);
    L = L * __expf(M - mn) + partial[b].y * __expf(partial[b].x - mn);
    M = mn;
  }
  acc[0] += logf(L) + M - logits[target];
  acc[1] += 1.0f;
}

static float2 *nll_scratch() {
  static float2 *p = nullptr;
  if (p == nullptr) {
    if (cudaMalloc((void **)&p, SAMPLE_BLOCKS * sizeof(float2)) != cudaSuccess) p = nullptr;
  }
  return p;
}

extern "C" wst_status_t wst_nll(const void *logits, int32_t target, void *acc, int32_t n) {
  WST_REQUIRE(logits && acc, "wst_nll: null pointer");
  WST_REQUIRE(n > 0, "wst_nll: n must be positive");
  WST_REQUIRE(target >= 0 && target < n, "wst_nll: target outside the vocabulary");

  float2 *partial = nll_scratch();
  if (partial == nullptr) {
    wst_set_error_msg("wst_nll: could not allocate reduction scratch");
    return WST_ERR_OOM;
  }

  const int blocks = min(SAMPLE_BLOCKS, (n + SAMPLE_THREADS - 1) / SAMPLE_THREADS);
  nll_partial_kernel<<<blocks, SAMPLE_THREADS>>>((const float *)logits, partial, n);
  nll_combine_kernel<<<1, 1>>>(partial, blocks, (const float *)logits, target, (float *)acc);

  WST_TRY_KERNEL("wst_nll");
  return WST_OK;
}

extern "C" wst_status_t wst_argmax(const void *logits, void *out_idx, int32_t n) {
  WST_REQUIRE(logits && out_idx, "wst_argmax: null pointer");
  WST_REQUIRE(n > 0, "wst_argmax: n must be positive");

  unsigned long long *acc = argmax_scratch();
  if (acc == nullptr) {
    wst_set_error_msg("wst_argmax: could not allocate reduction scratch");
    return WST_ERR_OOM;
  }

  WST_TRY(cudaMemsetAsync(acc, 0, sizeof(unsigned long long)));

  const int blocks = min(SAMPLE_BLOCKS, (n + SAMPLE_THREADS - 1) / SAMPLE_THREADS);
  argmax_kernel<<<blocks, SAMPLE_THREADS>>>((const float *)logits, acc, n);
  argmax_extract_kernel<<<1, 1>>>(acc, (int32_t *)out_idx);

  WST_TRY_KERNEL("wst_argmax");
  return WST_OK;
}
