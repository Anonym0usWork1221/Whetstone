/* Batch=1 GQA attention against a contiguous KV cache, split over the sequence.
 *
 * At batch=1 this is not a matmul, it is a GEMV against the cache: every cached
 * key and value is read exactly once and used for one multiply-add. So it obeys
 * the same law as the weight GEMVs -- bytes moved set the time -- and the cache
 * layout is the only real design decision.
 *
 * Layout: [kv_head][pos][head_dim], half.
 *   - the head_dim run is contiguous, so a warp reading lanes 0..31 of one
 *     position issues a single 64-byte request (128 B for head_dim 128);
 *   - consecutive positions are adjacent, so a warp sweeping the sequence walks
 *     memory linearly;
 *   - GQA means the seven query heads sharing a KV head read the *same* cache
 *     lines, which the L2 serves rather than DRAM. The 7:1 grouping is worth 7x
 *     on cache traffic, not just 7x on capacity.
 *
 * # Why it is split over the sequence
 *
 * One block per query head is the obvious decomposition and it leaves half the
 * GPU idle: Qwen2.5-0.5B has 14 query heads and this card has 30 SMs. Measured,
 * attention cost 0.32 ms of a 2.6 ms token -- more than the entire output
 * projection over a 68 MB matrix -- because 112 warps cannot hide DRAM latency
 * whatever they are doing.
 *
 * So the sequence is split too: block `(head, s)` sweeps its own slice and emits
 * a partial softmax, and a second kernel merges the slices. This is
 * flash-decoding's decomposition, and it works because the online softmax
 * recurrence is associative -- two partial `(max, denominator, numerator)`
 * triples combine into one exactly, with no re-reading of the cache.
 *
 * The split count is fixed at launch rather than derived from the sequence
 * length, because the sequence length lives in device memory (so that the whole
 * decode step is one CUDA graph) and a graph's grid dimensions are fixed at
 * instantiation. Slices past the end of the sequence exit immediately.
 */

#include "common.cuh"
#include <cuda_fp16.h>

#define ATTN_THREADS 256
#define ATTN_WARPS (ATTN_THREADS / WST_WARP)

/* head_dim <= 32 * ATTN_MAX_EPL. 8 covers every head width in current use
 * (64 for Qwen2.5, 128 for Qwen3 and Llama). */
#define ATTN_MAX_EPL 8

/* Upper bound on sequence slices per head. */
#define ATTN_MAX_SPLITS 16

/* Per-slice partial: the running max, the denominator, then head_dim
 * numerators. Laid out [head][split][2 + head_dim]. */
__device__ __forceinline__ float *partial_at(float *base, int head, int split, int splits,
                                             int hd) {
  return base + ((size_t)head * splits + split) * (size_t)(hd + 2);
}

__global__ __launch_bounds__(ATTN_THREADS) void attn_split_kernel(
    const float *__restrict__ q, const half *__restrict__ k_cache,
    const half *__restrict__ v_cache, float *__restrict__ partials, int gqa, int hd,
    const int32_t *__restrict__ pos_dev, int max_seq, int splits, float scale) {

  /* seq counts the cache entries that are valid, including the token just
   * appended -- so it is pos+1. Read from device memory, and clamped, because a
   * captured graph carries no per-token host arguments. */
  const int seq = min(max(*pos_dev, 0) + 1, max_seq);

  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int kvh = head / gqa;
  const int epl = hd / WST_WARP;

  /* Even slices. The last one absorbs the remainder. */
  const int per = (seq + splits - 1) / splits;
  const int t0 = split * per;
  const int t1 = min(t0 + per, seq);

  /* qs[hd] | m[warps] | l[warps] | acc[warps][hd] */
  extern __shared__ float smem[];
  float *qs = smem;
  float *m_s = qs + hd;
  float *l_s = m_s + ATTN_WARPS;
  float *acc_s = l_s + ATTN_WARPS;

  float *out = partial_at(partials, head, split, splits, hd);

  if (t0 >= t1) {
    /* This slice is past the end of the sequence. Publish an identity partial:
     * m = -inf makes exp(m - M) zero in the merge, so it contributes nothing
     * without the merge needing a special case. */
    if (threadIdx.x == 0) {
      out[0] = -INFINITY;
      out[1] = 0.0f;
    }
    for (int i = threadIdx.x; i < hd; i += ATTN_THREADS) out[2 + i] = 0.0f;
    return;
  }

  /* Fold the 1/sqrt(head_dim) into q once rather than into every score. */
  for (int i = threadIdx.x; i < hd; i += ATTN_THREADS)
    qs[i] = q[(size_t)head * hd + i] * scale;
  __syncthreads();

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const half *kb = k_cache + (size_t)kvh * max_seq * hd;
  const half *vb = v_cache + (size_t)kvh * max_seq * hd;

  /* A warp that gets no positions keeps m = -inf, and exp(-inf - M) = 0 makes
   * it contribute nothing to the combine below without a special case. */
  float m = -INFINITY, l = 0.0f;
  float acc[ATTN_MAX_EPL];
#pragma unroll
  for (int i = 0; i < ATTN_MAX_EPL; ++i) acc[i] = 0.0f;

  for (int t = t0 + warp; t < t1; t += ATTN_WARPS) {
    const half *kt = kb + (size_t)t * hd;

    float part = 0.0f;
    for (int i = 0; i < epl; ++i) {
      const int d = lane + i * WST_WARP;
      part = fmaf(qs[d], __half2float(kt[d]), part);
    }
    const float s = warp_reduce_sum(part); /* every lane holds the score */

    const float m_new = fmaxf(m, s);
    const float corr = __expf(m - m_new);
    const float p = __expf(s - m_new);

    const half *vt = vb + (size_t)t * hd;
    for (int i = 0; i < epl; ++i) {
      const int d = lane + i * WST_WARP;
      acc[i] = fmaf(p, __half2float(vt[d]), acc[i] * corr);
    }
    l = fmaf(l, corr, p);
    m = m_new;
  }

  if (lane == 0) {
    m_s[warp] = m;
    l_s[warp] = l;
  }
  for (int i = 0; i < epl; ++i) acc_s[warp * hd + lane + i * WST_WARP] = acc[i];
  __syncthreads();

  /* Merge this block's warps into one partial. */
  if (threadIdx.x < hd) {
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < ATTN_WARPS; ++w) M = fmaxf(M, m_s[w]);

    float num = 0.0f;
#pragma unroll
    for (int w = 0; w < ATTN_WARPS; ++w)
      num = fmaf(acc_s[w * hd + threadIdx.x], __expf(m_s[w] - M), num);
    out[2 + threadIdx.x] = num;

    if (threadIdx.x == 0) {
      float den = 0.0f;
#pragma unroll
      for (int w = 0; w < ATTN_WARPS; ++w) den = fmaf(l_s[w], __expf(m_s[w] - M), den);
      out[0] = M;
      out[1] = den;
    }
  }
}

/* Merges the sequence slices. One block per query head, one thread per output
 * element; `splits` is at most 16, so the loop is short and serial. */
__global__ void attn_merge_kernel(const float *__restrict__ partials,
                                  half *__restrict__ out, int hd, int splits) {
  const int head = blockIdx.x;
  const int d = threadIdx.x;
  if (d >= hd) return;

  const float *base = partials + (size_t)head * splits * (hd + 2);

  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, base[(size_t)s * (hd + 2)]);

  /* Every thread needs the denominator. Recomputing it per thread costs
   * `splits` FMAs off values already in L1 and saves a shared-memory round trip
   * and a barrier, which at splits <= 16 is the cheaper trade. */
  float num = 0.0f, den = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float *p = base + (size_t)s * (hd + 2);
    const float f = __expf(p[0] - M);
    num = fmaf(p[2 + d], f, num);
    den = fmaf(p[1], f, den);
  }

  out[(size_t)head * hd + d] = __float2half(num / den);
}

/* Slices per head such that the grid fills the machine.
 *
 * Two blocks per SM is the target: enough to overlap one block's memory latency
 * with another's arithmetic, few enough that each still has real work. The SM
 * count is cached because `cudaGetDeviceProperties` is a surprisingly expensive
 * host call against a kernel that runs in microseconds. */
static int pick_splits(int n_q, int max_seq) {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceProp p;
    sms = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.multiProcessorCount : 30;
  }

  int splits = (2 * sms + n_q - 1) / n_q;
  if (splits < 1) splits = 1;
  if (splits > ATTN_MAX_SPLITS) splits = ATTN_MAX_SPLITS;

  /* A slice thinner than one warp-sweep is pure overhead, so never cut the
   * cache into pieces smaller than the block can consume in one pass. */
  const int min_per_split = ATTN_WARPS;
  while (splits > 1 && max_seq / splits < min_per_split) splits--;
  return splits;
}

extern "C" wst_status_t wst_attn_decode(const void *q, const void *k_cache,
                                        const void *v_cache, void *partials, void *out,
                                        int32_t n_q, int32_t n_kv, int32_t head_dim,
                                        const void *pos, int32_t max_seq, float scale) {
  WST_REQUIRE(q && k_cache && v_cache && out && pos && partials,
              "wst_attn_decode: null pointer");
  WST_REQUIRE(n_q > 0 && n_kv > 0 && head_dim > 0, "wst_attn_decode: non-positive shape");
  WST_REQUIRE(n_q % n_kv == 0,
              "wst_attn_decode: query heads must be a multiple of key/value heads");
  WST_REQUIRE(head_dim % WST_WARP == 0,
              "wst_attn_decode: head_dim must be a multiple of the warp size");
  WST_REQUIRE(head_dim <= WST_WARP * ATTN_MAX_EPL, "wst_attn_decode: head_dim too large");
  WST_REQUIRE(max_seq > 0, "wst_attn_decode: empty cache");

  const size_t smem =
      (size_t)(head_dim + 2 * ATTN_WARPS + ATTN_WARPS * head_dim) * sizeof(float);
  WST_REQUIRE(smem <= 48u * 1024u, "wst_attn_decode: head_dim too large for shared memory");

  const int splits = pick_splits(n_q, max_seq);

  attn_split_kernel<<<dim3(n_q, splits), ATTN_THREADS, smem>>>(
      (const float *)q, (const half *)k_cache, (const half *)v_cache, (float *)partials,
      n_q / n_kv, head_dim, (const int32_t *)pos, max_seq, splits, scale);
  attn_merge_kernel<<<n_q, head_dim>>>((const float *)partials, (half *)out, head_dim,
                                       splits);

  WST_TRY_KERNEL("wst_attn_decode");
  return WST_OK;
}

/* Floats of scratch the split needs, so the caller can size the buffer without
 * duplicating `pick_splits`. */
extern "C" int32_t wst_attn_partial_floats(int32_t n_q, int32_t head_dim, int32_t max_seq) {
  return n_q * pick_splits(n_q, max_seq) * (head_dim + 2);
}
