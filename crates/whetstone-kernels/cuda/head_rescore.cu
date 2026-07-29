/* Recompute the few logits that decide the token, exactly.
 *
 * # The trade
 *
 * `lm_head` is one matrix read in full every token -- 27.6% of decode traffic on
 * Qwen2.5-0.5B -- so quantizing it is the largest single bandwidth win in the
 * engine and also the one that most directly perturbs the output distribution.
 * Measured (`research/experiments/head_lab.py`), on int4-hier-g32:
 *
 *   no rescore  +0.5186 ppl
 *   k=16        +0.1595
 *   k=64        +0.0957     <- 82% of the damage removed
 *   k=256       +0.0634
 *
 * The insight is that only the *largest* logits matter. Sampling looks at the
 * top of the distribution, and `logsumexp` is dominated by it: the top 64 of
 * 151,936 carry nearly all the denominator. So compute every logit from the
 * quantized head as usual, then recompute just the leaders from an fp16 copy.
 *
 * At k=64 on the 0.5B that is 64 x 896 x 2 = 114 KB against a 264 MB token --
 * **0.17% more bandwidth** -- plus 272 MB of VRAM for the fp16 copy on a card
 * with ~4 GB spare. It spends the resource that is not binding to buy back the
 * one that is.
 *
 * # Why there is no top-k kernel here
 *
 * A real top-k over 151,936 logits is a radix select: days of work for a 0.1
 * perplexity item. It is not needed. A *threshold* selects the same set, and a
 * threshold is two reductions.
 *
 * The threshold cannot be a constant -- logit spreads vary per token and per
 * model -- so it is chosen on device from a geometric ladder of candidates, by
 * counting how many logits each admits and taking the tightest one that still
 * admits at least `k`. Two passes over 608 KB, about 4 us.
 *
 * # Why the grid is fixed
 *
 * A CUDA graph bakes its launch shape in at instantiation, so a data-dependent
 * count cannot size the grid. It can, however, be *read from device memory* by
 * blocks that then decide to do nothing -- the same trick the position cursor
 * and the MoE router already use. Every launch here is a fixed shape, so the
 * whole rescore lives inside the single captured decode graph.
 */

#include "common.cuh"

#define WST_RESCORE_CANDIDATES 12

/* ------------------------------------------------- pass 1: pick a threshold */

/* Split across the grid, not run in one block.
 *
 * The first version did the max and the counting from a single block. That is
 * one SM of thirty reading 608 KB of logits, and it measured **7.7% of the
 * decode step** on Qwen2.5-0.5B against a bandwidth model that predicted 0.17%.
 * The model was not wrong about bytes; the kernel was wrong about parallelism.
 *
 * So both O(vocab) passes are grid-wide and the two reductions between them are
 * trivial kernels over a few hundred floats. Four launches instead of one, all
 * fixed-shape, and the whole thing drops to a couple of microseconds.
 */
__global__ void head_max_partial_kernel(const float *__restrict__ logits, int n,
                                        float *__restrict__ partial) {
  __shared__ float warp_max[256 / WST_WARP];
  float m = -INFINITY;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    m = fmaxf(m, logits[i]);
  }
  m = warp_reduce_max(m);
  if ((threadIdx.x & (WST_WARP - 1)) == 0) warp_max[threadIdx.x / WST_WARP] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = -INFINITY;
    const int nw = (blockDim.x + WST_WARP - 1) / WST_WARP;
    for (int i = 0; i < nw; ++i) best = fmaxf(best, warp_max[i]);
    partial[blockIdx.x] = best;
  }
}

/* Reduces the per-block maxima and clears the candidate counters. One block over
 * a few hundred floats. */
__global__ void head_max_finalize_kernel(const float *__restrict__ partial, int nblocks,
                                         float *__restrict__ out_max,
                                         int32_t *__restrict__ counts) {
  __shared__ float warp_max[256 / WST_WARP];
  float m = -INFINITY;
  for (int i = threadIdx.x; i < nblocks; i += blockDim.x) m = fmaxf(m, partial[i]);
  m = warp_reduce_max(m);
  if ((threadIdx.x & (WST_WARP - 1)) == 0) warp_max[threadIdx.x / WST_WARP] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    float best = -INFINITY;
    const int nw = (blockDim.x + WST_WARP - 1) / WST_WARP;
    for (int i = 0; i < nw; ++i) best = fmaxf(best, warp_max[i]);
    *out_max = best;
  }
  if (threadIdx.x < WST_RESCORE_CANDIDATES) counts[threadIdx.x] = 0;
}

/* How many logits each candidate margin admits, grid-wide.
 *
 * The ladder is geometric, 1/64 to 32 nats, because logit spreads are: a
 * confident token has its top 64 within a fraction of a nat, an uncertain one
 * spreads them over tens. The bottom rungs are what make a near-uniform
 * distribution selectable at all -- a ladder starting at 0.25 admits the entire
 * vocabulary there. */
__global__ void head_counts_kernel(const float *__restrict__ logits, int n,
                                   const float *__restrict__ maxv,
                                   int32_t *__restrict__ counts) {
  __shared__ int local[WST_RESCORE_CANDIDATES];
  if (threadIdx.x < WST_RESCORE_CANDIDATES) local[threadIdx.x] = 0;
  __syncthreads();

  const float smax = *maxv;
  int mine[WST_RESCORE_CANDIDATES];
#pragma unroll
  for (int c = 0; c < WST_RESCORE_CANDIDATES; ++c) mine[c] = 0;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const float d = smax - logits[i];
#pragma unroll
    for (int c = 0; c < WST_RESCORE_CANDIDATES; ++c) {
      if (d <= (0.015625f * (float)(1 << c))) mine[c]++;
    }
  }
#pragma unroll
  for (int c = 0; c < WST_RESCORE_CANDIDATES; ++c) {
    const int sum = warp_reduce_sum_i32(mine[c]);
    if ((threadIdx.x & (WST_WARP - 1)) == 0) atomicAdd(&local[c], sum);
  }
  __syncthreads();
  if (threadIdx.x < WST_RESCORE_CANDIDATES && local[threadIdx.x] != 0) {
    atomicAdd(&counts[threadIdx.x], local[threadIdx.x]);
  }
}

/* Tightest margin that still admits k, and clear the compaction counter.
 *
 * The last rung is the fallback: if even a 32-nat window holds fewer than k
 * logits then the distribution is extraordinarily peaked and rescoring what
 * there is, is correct.
 *
 * When even the tightest rung admits more than `cap`, the compaction keeps an
 * arbitrary `cap` of them rather than the largest. That is a real degradation of
 * the "rescore the leaders" guarantee, accepted deliberately: it needs a
 * distribution so flat that the survivors differ by less than 1/64 of a nat,
 * where which of them get the exact treatment cannot matter. The alternative --
 * re-running the search until the count fits -- is a data-dependent loop, and a
 * data-dependent loop cannot be captured into the decode graph. */
__global__ void head_pick_kernel(const float *__restrict__ maxv,
                                 const int32_t *__restrict__ counts, int k, int cap,
                                 float *__restrict__ out_thresh,
                                 int32_t *__restrict__ out_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  int pick = WST_RESCORE_CANDIDATES - 1;
  for (int c = 0; c < WST_RESCORE_CANDIDATES; ++c) {
    if (counts[c] >= k) { pick = c; break; }
  }

  /* Never choose a rung that overflows `cap`.
   *
   * This is a determinism requirement, not an optimisation. Every survivor is
   * rescored, so the compaction's arbitrary order is invisible -- unless the
   * count exceeds `cap`, at which point *which* rows survive is decided by
   * `atomicAdd` scheduling and perplexity stops being reproducible (measured:
   * 17.3402 / 17.3400 / 17.3345 on identical inputs).
   *
   * The ladder doubles, so a single rung can multiply the count by far more than
   * the headroom `cap` provides; stepping back to a tighter rung trades "more
   * rows than asked for" against "a different answer each run", and
   * reproducibility wins. Rescoring 40 rows deterministically beats rescoring an
   * arbitrary 1024. */
  while (pick > 0 && counts[pick] > cap) --pick;

  *out_thresh = *maxv - 0.015625f * (float)(1 << pick);
  *out_count = 0;
}

extern "C" wst_status_t wst_head_threshold(const void *logits, int32_t n, int32_t k,
                                           int32_t cap, void *scratch, int32_t nblocks,
                                           void *out_thresh, void *out_count) {
  WST_REQUIRE(logits && scratch && out_thresh && out_count,
              "wst_head_threshold: null pointer");
  WST_REQUIRE(n > 0 && k > 0 && cap >= k && nblocks > 0,
              "wst_head_threshold: bad shape");

  /* scratch layout: [nblocks] partial maxima, then one max, then the counters. */
  float *partial = (float *)scratch;
  float *maxv = partial + nblocks;
  int32_t *counts = (int32_t *)(maxv + 1);

  head_max_partial_kernel<<<nblocks, 256>>>((const float *)logits, n, partial);
  head_max_finalize_kernel<<<1, 256>>>(partial, nblocks, maxv, counts);
  head_counts_kernel<<<nblocks, 256>>>((const float *)logits, n, maxv, counts);
  head_pick_kernel<<<1, 32>>>(maxv, counts, k, cap, (float *)out_thresh,
                              (int32_t *)out_count);
  WST_TRY_KERNEL("wst_head_threshold");
  return WST_OK;
}

/* ---------------------------------------------------- pass 2: compact ids */

/* Every logit above the threshold, appended to `out_idx` under an atomic
 * counter, capped at `cap`.
 *
 * Order is arbitrary -- that is what an atomic append gives -- and that is fine:
 * every survivor is within one margin of the maximum, so which of them get the
 * exact treatment when there are more than `cap` matters far less than that the
 * launch shape stays fixed. The threshold search above keeps the count near `k`
 * precisely so this cap is rarely reached.
 */
__global__ void head_compact_kernel(const float *__restrict__ logits, int n,
                                    const float *__restrict__ thresh,
                                    int32_t *__restrict__ count,
                                    int32_t *__restrict__ out_idx, int cap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (logits[i] < *thresh) return;
  const int slot = atomicAdd(count, 1);
  if (slot < cap) out_idx[slot] = i;
}

extern "C" wst_status_t wst_head_compact(const void *logits, int32_t n,
                                         const void *thresh, void *count,
                                         void *out_idx, int32_t cap) {
  WST_REQUIRE(logits && thresh && count && out_idx, "wst_head_compact: null pointer");
  WST_REQUIRE(n > 0 && cap > 0, "wst_head_compact: n and cap must be positive");
  const int threads = 256;
  head_compact_kernel<<<(n + threads - 1) / threads, threads>>>(
      (const float *)logits, n, (const float *)thresh, (int32_t *)count,
      (int32_t *)out_idx, cap);
  WST_TRY_KERNEL("wst_head_compact");
  return WST_OK;
}

/* ------------------------------------------------- pass 3: exact rescoring */

/* Block `j` recomputes logit `idx[j]` from the fp16 head and overwrites it.
 *
 * The grid is `cap` blocks whatever the count turns out to be; blocks past the
 * count return immediately. That is what keeps this capturable: the graph is
 * instantiated once with `cap` blocks and the *data* decides how many do work.
 *
 * `x` is the same fp16 activation the quantized head GEMV consumed, so the only
 * difference between this logit and that one is the weight precision -- which is
 * the entire point.
 */
__global__ __launch_bounds__(256) void head_rescore_kernel(
    const half *__restrict__ head, const half *__restrict__ x,
    const int32_t *__restrict__ idx, const int32_t *__restrict__ count,
    float *__restrict__ logits, int hidden, int cap, int grid) {

  const int n = min(*count, cap);
  __shared__ float red[256 / WST_WARP];

  /* Every selected row, strided across a **fixed** grid.
   *
   * The first version used one block per row and returned when `blockIdx.x`
   * exceeded the count, which meant any row past `cap` was simply dropped -- and
   * *which* rows survived was decided by `atomicAdd` ordering in the compaction.
   * That made perplexity vary run to run (17.3402 / 17.3400 / 17.3345 on
   * identical inputs), ending the bit-exact reproducibility that is this
   * project's primary correctness check.
   *
   * Striding fixes it at the root: every survivor is rescored, so the *order*
   * the compaction happened to produce cannot matter. The launch shape is still
   * constant, so the graph capture is unaffected -- only the number of
   * iterations varies, and that is read from device memory. */
  for (int j = blockIdx.x; j < n; j += grid) {
    const int row = idx[j];
    const half *w = head + (size_t)row * hidden;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      acc = fmaf(__half2float(w[i]), __half2float(x[i]), acc);
    }
    acc = block_reduce_sum(acc, red);
    if (threadIdx.x == 0) logits[row] = acc;
    __syncthreads();  /* `red` is reused by the next iteration */
  }
}

extern "C" wst_status_t wst_head_rescore(const void *head, const void *x,
                                         const void *idx, const void *count,
                                         void *logits, int32_t hidden, int32_t cap,
                                         int32_t grid) {
  WST_REQUIRE(head && x && idx && count && logits, "wst_head_rescore: null pointer");
  WST_REQUIRE(hidden > 0 && cap > 0 && grid > 0, "wst_head_rescore: bad shape");
  head_rescore_kernel<<<grid, 256>>>((const half *)head, (const half *)x,
                                     (const int32_t *)idx, (const int32_t *)count,
                                     (float *)logits, hidden, cap, grid);
  WST_TRY_KERNEL("wst_head_rescore");
  return WST_OK;
}
