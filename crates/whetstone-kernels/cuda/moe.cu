/* Mixture-of-experts routing.
 *
 * # Why this is the interesting case for a bandwidth-bound engine
 *
 * A dense block reads all three MLP matrices every token. A MoE block stores
 * `n_experts` copies of them and reads `k`. Qwen3-30B-A3B stores 30.5 B
 * parameters and reads 3.0 B, so the roofline denominator is a sixth of what the
 * model's name suggests -- and this engine's entire thesis is that the
 * denominator is the only thing that matters at batch 1.
 *
 * The catch, measured and recorded in `research/01-V6-PLAN.md` §0.1: sparsity
 * removes weights from the *bandwidth* bill, not from the machine. Experts that
 * do not fit in VRAM are read over PCIe at 5.77 GB/s against DRAM's 278, which
 * turns a 109 tok/s roofline into 5 tok/s. Sparsity pays only for models whose
 * whole expert set is resident.
 *
 * # Everything here stays on the device
 *
 * The expert indices are data-dependent, which is exactly the property that
 * makes CUDA graph capture hard: a graph bakes its kernel arguments in at
 * instantiation, so anything varying per token must be *read from memory* by the
 * kernel rather than passed by the host. So the router writes indices and
 * weights into device buffers, and the expert GEMV reads its row offset from
 * one. It is the same trick the position cursor already uses, and it is what
 * keeps a MoE decode step to a single graph launch instead of `k` host
 * round-trips per layer.
 */

#include "common.cuh"

/* --------------------------------------------------------------- top-k */

/* Softmax over every expert logit, then the k largest.
 *
 * The order matters and is not the obvious one. HuggingFace computes
 * `softmax(logits)` over **all** experts and *then* takes the top k, so the
 * weights carry the full denominator. Taking the top k first and softmaxing
 * those would give different weights whenever the discarded experts hold
 * meaningful mass -- and would be indistinguishable from correct on any test
 * short of a logit comparison, because both produce a valid-looking
 * distribution.
 *
 * `norm_topk` then renormalises the k survivors to sum to 1. Qwen3-MoE and
 * Mixtral do; OLMoE does not. It is a config flag, not a constant, because
 * getting it wrong scales every expert's contribution by the same factor and so
 * reads as a slightly mis-tuned model rather than as a bug.
 *
 * One block, one expert per thread, `k` rounds of masked block-argmax. With
 * n_experts <= 1024 and k <= 32 that is a few microseconds and needs no sort.
 */
__global__ void moe_router_kernel(const float *__restrict__ logits, int n_experts, int k,
                                  int norm_topk, int32_t *__restrict__ out_idx,
                                  float *__restrict__ out_w) {
  extern __shared__ float smem[];
  float *vals = smem;                    /* n_experts logits, masked as taken */
  float *red = smem + n_experts;         /* blockDim/32 for the reductions */

  const int t = threadIdx.x;

  /* --- full softmax denominator ------------------------------------------ */
  float mine = (t < n_experts) ? logits[t] : -INFINITY;
  float m = warp_reduce_max(mine);
  __shared__ float smax, ssum;
  if ((t & (WST_WARP - 1)) == 0) red[t / WST_WARP] = m;
  __syncthreads();
  if (t == 0) {
    float best = -INFINITY;
    const int nw = (blockDim.x + WST_WARP - 1) / WST_WARP;
    for (int i = 0; i < nw; ++i) best = fmaxf(best, red[i]);
    smax = best;
  }
  __syncthreads();

  const float e = (t < n_experts) ? __expf(mine - smax) : 0.0f;
  if (t < n_experts) vals[t] = mine;     /* keep the raw logit for selection */
  float tot = block_reduce_sum(e, red);
  if (t == 0) ssum = tot;
  __syncthreads();
  const float denom = ssum;

  /* --- k rounds of masked argmax ----------------------------------------- */
  //
  // Ties are broken toward the lower index, deterministically. Two experts with
  // bit-identical logits is not a hypothetical -- it is what an untrained or
  // saturated router does -- and a nondeterministic choice there would end the
  // bit-exact reproducibility that every differential test in this project
  // depends on.
  for (int r = 0; r < k; ++r) {
    float best = -INFINITY;
    int arg = 0;
    for (int i = t; i < n_experts; i += blockDim.x) {
      const float v = vals[i];
      if (v > best) {
        best = v;
        arg = i;
      }
    }
    /* Reduce (value, index) pairs across the block, lower index wins a tie. */
    for (int off = WST_WARP / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(WST_FULL_MASK, best, off);
      const int oi = __shfl_xor_sync(WST_FULL_MASK, arg, off);
      if (ov > best || (ov == best && oi < arg)) {
        best = ov;
        arg = oi;
      }
    }
    __shared__ float wbest[1024 / WST_WARP];
    __shared__ int warg[1024 / WST_WARP];
    if ((t & (WST_WARP - 1)) == 0) {
      wbest[t / WST_WARP] = best;
      warg[t / WST_WARP] = arg;
    }
    __syncthreads();
    if (t == 0) {
      const int nw = (blockDim.x + WST_WARP - 1) / WST_WARP;
      float bv = wbest[0];
      int bi = warg[0];
      for (int i = 1; i < nw; ++i) {
        if (wbest[i] > bv || (wbest[i] == bv && warg[i] < bi)) {
          bv = wbest[i];
          bi = warg[i];
        }
      }
      out_idx[r] = bi;
      out_w[r] = __expf(bv - smax) / denom;
      vals[bi] = -INFINITY; /* mask it out of the next round */
    }
    __syncthreads();
  }

  /* --- optional renormalisation ------------------------------------------ */
  if (norm_topk && t == 0) {
    float s = 0.0f;
    for (int r = 0; r < k; ++r) s += out_w[r];
    /* A zero sum needs every expert's softmax weight to underflow, which the
     * max-subtraction above makes impossible for the argmax. Guarded anyway:
     * dividing by it would put NaNs into the residual stream, and a NaN there
     * is the one failure this engine cannot produce fluent text through. */
    const float inv = (s > 0.0f) ? 1.0f / s : 1.0f;
    for (int r = 0; r < k; ++r) out_w[r] *= inv;
  }
}

extern "C" wst_status_t wst_moe_router(const void *logits, int32_t n_experts, int32_t k,
                                       int32_t norm_topk, void *out_idx, void *out_w) {
  WST_REQUIRE(logits && out_idx && out_w, "wst_moe_router: null pointer");
  WST_REQUIRE(n_experts > 0 && n_experts <= 1024,
              "wst_moe_router: n_experts must be in 1..=1024");
  WST_REQUIRE(k > 0 && k <= n_experts, "wst_moe_router: k must be in 1..=n_experts");

  /* Whole warps: the reductions shuffle with a full mask. */
  const int threads = ((n_experts + WST_WARP - 1) / WST_WARP) * WST_WARP;
  const size_t smem = (size_t)(n_experts + threads / WST_WARP + WST_WARP) * sizeof(float);

  moe_router_kernel<<<1, threads, smem>>>((const float *)logits, n_experts, k, norm_topk,
                                          (int32_t *)out_idx, (float *)out_w);
  WST_TRY_KERNEL("wst_moe_router");
  return WST_OK;
}

/* ------------------------------------------------- weighted accumulate */

/* `dst += weight[slot] * src`, with the scalar read from device memory.
 *
 * The expert's routing weight is only known on the device, so it cannot be a
 * kernel argument without breaking graph capture -- hence the pointer and the
 * slot index. `src` is one expert's `down_proj` output and `dst` is the residual
 * stream, so this is what turns `k` independent expert outputs into their convex
 * combination.
 */
__global__ void moe_accumulate_kernel(float *__restrict__ dst,
                                      const float *__restrict__ src,
                                      const float *__restrict__ weights, int slot, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = fmaf(weights[slot], src[i], dst[i]);
}

extern "C" wst_status_t wst_moe_accumulate(void *dst, const void *src, const void *weights,
                                           int32_t slot, int32_t n) {
  WST_REQUIRE(dst && src && weights, "wst_moe_accumulate: null pointer");
  WST_REQUIRE(n > 0, "wst_moe_accumulate: n must be positive");
  WST_REQUIRE(slot >= 0, "wst_moe_accumulate: negative slot");

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  moe_accumulate_kernel<<<blocks, threads>>>((float *)dst, (const float *)src,
                                             (const float *)weights, slot, n);
  WST_TRY_KERNEL("wst_moe_accumulate");
  return WST_OK;
}
