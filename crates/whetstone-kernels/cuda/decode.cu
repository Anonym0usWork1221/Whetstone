/* The non-GEMV half of a decode step: normalisation, rotary embedding, the KV
 * cache append, SwiGLU and the embedding gather.
 *
 * None of these move much memory -- a decode step's whole activation footprint
 * is a few kilobytes against ~260 MB of weights -- so they are not where the
 * time goes. What they *do* cost is launches. Seven GEMVs and five elementwise
 * kernels per block, times 24 blocks, is ~290 launches per token; at a couple of
 * microseconds each that is a millisecond of pure dispatch. So the rule here is
 * to fuse whatever shares an input, even when the arithmetic saved is nil:
 *
 *   - RoPE, the KV append and the f16 narrowing are one kernel, because they all
 *     touch the same freshly projected q/k/v.
 *   - Bias and the residual add live in the GEMV epilogue (see gemv_int4.cu),
 *     not here.
 *   - SwiGLU consumes gate and up together and emits the down projection's
 *     input directly.
 *
 * Precision: every reduction and every activation runs in fp32 and narrows to
 * f16 only when handing a vector to the next GEMV. The residual stream itself
 * stays fp32 for the whole 24 layers. Turing has no bf16, and fp16 accumulation
 * over 896 terms is visibly lossy, so this is not a free choice.
 */

#include "common.cuh"
#include <cuda_fp16.h>

#define WST_NORM_THREADS 256
#define WST_ELEM_THREADS 256

/* ------------------------------------------------------------------ RMSNorm */

/* Single block: hidden is at most a few thousand, so a grid-wide reduction would
 * need a second kernel to combine partials, and the whole operation moves five
 * kilobytes.
 *
 * That does not make it free. There are 49 of these per token -- two per block
 * plus the final one -- and at ~7 us each they cost more than the entire output
 * projection over a 68 MB matrix. The cost is not bandwidth, it is two
 * dependent round trips to memory: read x to compute the sum, then read x again
 * to scale it.
 *
 * `WIDE` removes the second one. When the vector fits in one element per thread,
 * each thread keeps its value in a register across the reduction, so the kernel
 * touches x exactly once. Qwen2.5-0.5B's hidden size is 896, so this is the path
 * that runs; the loop below stays for anything wider. */
template <bool WIDE>
__global__ __launch_bounds__(1024) void rmsnorm_kernel(
    const float *__restrict__ x, const half *__restrict__ w,
    half *__restrict__ out, int n, float eps) {

  __shared__ float warp_sums[1024 / WST_WARP];
  __shared__ float inv_rms;

  float mine = 0.0f;
  float acc = 0.0f;

  if (WIDE) {
    if (threadIdx.x < n) {
      mine = x[threadIdx.x];
      acc = mine * mine;
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      const float v = x[i];
      acc = fmaf(v, v, acc);
    }
  }

  acc = block_reduce_sum(acc, warp_sums);

  if (threadIdx.x == 0) inv_rms = rsqrtf(acc / (float)n + eps);
  __syncthreads();

  const float s = inv_rms;
  if (WIDE) {
    if (threadIdx.x < n)
      out[threadIdx.x] = __float2half(mine * s * __half2float(w[threadIdx.x]));
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x)
      out[i] = __float2half(x[i] * s * __half2float(w[i]));
  }
}

extern "C" wst_status_t wst_rmsnorm(const void *x, const void *w, void *out,
                                    int32_t n, float eps) {
  WST_REQUIRE(x && w && out, "wst_rmsnorm: null pointer");
  WST_REQUIRE(n > 0, "wst_rmsnorm: n must be positive");

  if (n <= 1024) {
    /* Round up to a whole warp so the reduction has no partial warp to mask. */
    const int threads = ((n + WST_WARP - 1) / WST_WARP) * WST_WARP;
    rmsnorm_kernel<true><<<1, threads>>>((const float *)x, (const half *)w, (half *)out,
                                         n, eps);
  } else {
    rmsnorm_kernel<false><<<1, WST_NORM_THREADS>>>((const float *)x, (const half *)w,
                                                   (half *)out, n, eps);
  }
  WST_TRY_KERNEL("wst_rmsnorm");
  return WST_OK;
}

/* -------------------------------------------------- RoPE + KV cache append */

/* One block per head across q and k, plus one per v head for the plain copy.
 *
 * cos/sin come from a table the host built in f64. Computing them here would
 * mean either the fast intrinsics (whose argument reduction degrades past a few
 * thousand radians, exactly the regime long contexts live in) or a double-
 * precision sincos at 1/32 rate. The table is max_seq * head_dim/2 * 2 floats --
 * 1 MB at 4k context -- and turns the whole thing into two loads.
 */
__global__ void rope_cache_kernel(
    float *__restrict__ qkv, half *__restrict__ k_cache, half *__restrict__ v_cache,
    const float *__restrict__ cos_tab, const float *__restrict__ sin_tab,
    int n_q, int n_kv, int hd, const int32_t *__restrict__ pos_dev, int max_seq,
    const half *__restrict__ q_norm_w, const half *__restrict__ k_norm_w, float eps) {

  const int head = blockIdx.x;
  const int halfd = hd >> 1;
  const int j = threadIdx.x;

  /* The block is padded up to a whole warp when QK-norm is on, because the
   * block reduction masks a full warp. Padding lanes must reach the barriers,
   * so this is a predicate rather than an early return. */
  const bool active = (j < halfd);

  /* The position lives on the device so the whole decode step is capturable as
   * a CUDA graph -- a graph bakes its kernel arguments in at instantiation, so
   * anything that changes per token has to be read from memory rather than
   * passed. Clamped rather than trusted: a graph cannot be range-checked by the
   * host before it launches. */
  const int pos = min(max(*pos_dev, 0), max_seq - 1);

  const float c = active ? cos_tab[(size_t)pos * halfd + j] : 0.0f;
  const float s = active ? sin_tab[(size_t)pos * halfd + j] : 0.0f;

  /* q, k and v arrive as one contiguous vector because they come out of one
   * fused projection -- three separate GEMVs of 896, 128 and 128 rows cannot
   * create enough warps to keep the memory system busy, and the 128-row ones
   * measured 19 GB/s against lm_head's 254. */
  const float *k = qkv + (size_t)n_q * hd;
  const float *v = k + (size_t)n_kv * hd;

  /* Both branches below are uniform across the block -- they test blockIdx --
   * so the barriers inside `wst_qk_head_norm` are reached by every thread. */
  if (head < n_q) {
    /* Query heads: rotate in place; nothing is cached. */
    float *qh = qkv + (size_t)head * hd;
    float x1 = active ? qh[j] : 0.0f;
    float x2 = active ? qh[j + halfd] : 0.0f;
    if (q_norm_w) wst_qk_head_norm(x1, x2, q_norm_w, j, halfd, hd, eps, active);
    if (!active) return;
    qh[j] = x1 * c - x2 * s;
    qh[j + halfd] = x2 * c + x1 * s;
    return;
  }

  const int kvh = head - n_q;
  if (kvh >= n_kv) return;

  const size_t slot = ((size_t)kvh * max_seq + pos) * hd;

  const float *kh = k + (size_t)kvh * hd;
  float k1 = active ? kh[j] : 0.0f;
  float k2 = active ? kh[j + halfd] : 0.0f;
  if (k_norm_w) wst_qk_head_norm(k1, k2, k_norm_w, j, halfd, hd, eps, active);
  if (!active) return;
  k_cache[slot + j] = __float2half(k1 * c - k2 * s);
  k_cache[slot + j + halfd] = __float2half(k2 * c + k1 * s);

  /* v is not rotated -- position information enters only through the scores --
   * and it is never QK-normed either: the norm exists to stabilise the *scores*,
   * and v does not enter them. It is cached from the same block so the copy
   * costs no extra launch. */
  const float *vh = v + (size_t)kvh * hd;
  v_cache[slot + j] = __float2half(vh[j]);
  v_cache[slot + j + halfd] = __float2half(vh[j + halfd]);
}

extern "C" wst_status_t wst_rope_cache(void *qkv, void *k_cache, void *v_cache,
                                       const void *cos_tab, const void *sin_tab,
                                       int32_t n_q, int32_t n_kv, int32_t head_dim,
                                       const void *pos, int32_t max_seq,
                                       const void *q_norm_w, const void *k_norm_w,
                                       float eps) {
  WST_REQUIRE(qkv && k_cache && v_cache && cos_tab && sin_tab && pos,
              "wst_rope_cache: null pointer");
  WST_REQUIRE(n_q > 0 && n_kv > 0 && head_dim > 0, "wst_rope_cache: non-positive shape");
  WST_REQUIRE(head_dim % 2 == 0, "wst_rope_cache: head_dim must be even");
  WST_REQUIRE(max_seq > 0, "wst_rope_cache: empty cache");

  const int halfd = head_dim / 2;
  WST_REQUIRE(halfd <= 1024, "wst_rope_cache: head_dim/2 exceeds a block");

  /* Round up to a whole warp: the block reduction shuffles with a full mask,
   * which is undefined if the warp is not fully populated. Without QK-norm
   * there is no reduction and the exact width is kept. */
  const bool norm = (q_norm_w != nullptr) || (k_norm_w != nullptr);
  const int threads = norm ? ((halfd + WST_WARP - 1) / WST_WARP) * WST_WARP : halfd;

  rope_cache_kernel<<<n_q + n_kv, threads>>>(
      (float *)qkv, (half *)k_cache, (half *)v_cache,
      (const float *)cos_tab, (const float *)sin_tab, n_q, n_kv, head_dim,
      (const int32_t *)pos, max_seq, (const half *)q_norm_w, (const half *)k_norm_w,
      eps);
  WST_TRY_KERNEL("wst_rope_cache");
  return WST_OK;
}

/* pos += 1, saturating at the cache capacity.
 *
 * One thread. It exists so the position advance is a graph node rather than a
 * host-side increment, which is what lets a whole generation run as repeated
 * launches of one graph with no host involvement at all. */
__global__ void advance_pos_kernel(int32_t *pos, int max_seq) {
  const int p = *pos + 1;
  *pos = p < max_seq ? p : max_seq;
}

extern "C" wst_status_t wst_advance_pos(void *pos, int32_t max_seq) {
  WST_REQUIRE(pos, "wst_advance_pos: null pointer");
  WST_REQUIRE(max_seq > 0, "wst_advance_pos: empty cache");
  advance_pos_kernel<<<1, 1>>>((int32_t *)pos, max_seq);
  WST_TRY_KERNEL("wst_advance_pos");
  return WST_OK;
}

/* ------------------------------------------------------------------ SwiGLU */

/* out = silu(gate) * up, narrowing to f16 for the down projection.
 *
 * SiLU is never exactly zero for a nonzero input, which is why Qwen has no
 * exploitable activation sparsity -- a tempting-looking optimisation that the
 * activation function itself rules out. */
__global__ __launch_bounds__(WST_ELEM_THREADS) void swiglu_kernel(
    const float *__restrict__ gate_up, half *__restrict__ out, int n) {
  const int i = blockIdx.x * WST_ELEM_THREADS + threadIdx.x;
  if (i >= n) return;
  /* gate and up are the two halves of one fused projection's output. */
  const float g = gate_up[i];
  out[i] = __float2half(g * __frcp_rn(1.0f + __expf(-g)) * gate_up[i + n]);
}

extern "C" wst_status_t wst_swiglu(const void *gate_up, void *out, int32_t n) {
  WST_REQUIRE(gate_up && out, "wst_swiglu: null pointer");
  WST_REQUIRE(n > 0, "wst_swiglu: n must be positive");

  const int blocks = (n + WST_ELEM_THREADS - 1) / WST_ELEM_THREADS;
  swiglu_kernel<<<blocks, WST_ELEM_THREADS>>>((const float *)gate_up, (half *)out, n);
  WST_TRY_KERNEL("wst_swiglu");
  return WST_OK;
}

/* --------------------------------------------------------------- embedding */

/* The input embedding is a single-row gather: ~1.8 KB against the ~260 MB a
 * decode step streams. It is free, and it is the *only* free use of the
 * embedding matrix -- the output projection reads all 136 M of it every token. */
__global__ void embed_fp16_kernel(const half *__restrict__ table,
                                  const int32_t *__restrict__ token_dev,
                                  float *__restrict__ out, int hidden, int rows) {
  const int i = blockIdx.x * WST_ELEM_THREADS + threadIdx.x;
  if (i >= hidden) return;
  /* Clamped, not trusted: inside a CUDA graph the host never sees this value
   * before the launch, so the bounds check has to be here. */
  const int token = min(max(*token_dev, 0), rows - 1);
  out[i] = __half2float(table[(size_t)token * hidden + i]);
}

/* Same gather against an int4-g128 table. When lm_head is quantized the tied
 * input embedding is quantized with it -- one matrix, two uses -- so the gather
 * has to dequantize rather than read f16. */
__global__ void embed_int4_kernel(const uint32_t *__restrict__ qw,
                                  const uint32_t *__restrict__ sz,
                                  const int32_t *__restrict__ token_dev,
                                  float *__restrict__ out, int hidden, int rows) {
  const int i = blockIdx.x * WST_ELEM_THREADS + threadIdx.x;
  if (i >= hidden) return;
  const int token = min(max(*token_dev, 0), rows - 1);

  const size_t row_words = (size_t)hidden / 8;
  const size_t row_groups = (size_t)hidden / 128;

  const uint32_t word = qw[(size_t)token * row_words + i / 8];
  const uint32_t q = (word >> (4 * (i % 8))) & 0xFu;

  const uint32_t packed = sz[(size_t)token * row_groups + i / 128];
  const float scale = __half2float(__ushort_as_half((unsigned short)(packed & 0xFFFFu)));
  const float zero = __half2float(__ushort_as_half((unsigned short)(packed >> 16)));

  out[i] = ((float)q - zero) * scale;
}

extern "C" wst_status_t wst_embed_fp16(const void *table, const void *token, void *out,
                                       int32_t hidden, int32_t rows) {
  WST_REQUIRE(table && out && token, "wst_embed_fp16: null pointer");
  WST_REQUIRE(hidden > 0 && rows > 0, "wst_embed_fp16: bad argument");

  const int blocks = (hidden + WST_ELEM_THREADS - 1) / WST_ELEM_THREADS;
  embed_fp16_kernel<<<blocks, WST_ELEM_THREADS>>>(
      (const half *)table, (const int32_t *)token, (float *)out, hidden, rows);
  WST_TRY_KERNEL("wst_embed_fp16");
  return WST_OK;
}

extern "C" wst_status_t wst_embed_int4_g128(const void *qw, const void *sz,
                                            const void *token, void *out, int32_t hidden,
                                            int32_t rows) {
  WST_REQUIRE(qw && sz && out && token, "wst_embed_int4_g128: null pointer");
  WST_REQUIRE(hidden > 0 && rows > 0, "wst_embed_int4_g128: bad argument");
  WST_REQUIRE(hidden % 128 == 0, "wst_embed_int4_g128: hidden must be a multiple of 128");

  const int blocks = (hidden + WST_ELEM_THREADS - 1) / WST_ELEM_THREADS;
  embed_int4_kernel<<<blocks, WST_ELEM_THREADS>>>(
      (const uint32_t *)qw, (const uint32_t *)sz, (const int32_t *)token, (float *)out,
      hidden, rows);
  WST_TRY_KERNEL("wst_embed_int4_g128");
  return WST_OK;
}
