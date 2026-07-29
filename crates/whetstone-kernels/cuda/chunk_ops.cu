/* The non-GEMM half of a multi-token pass: normalisation, rotary embedding over
 * N consecutive positions, causal chunk attention, SwiGLU, the embedding gather
 * and a per-row argmax.
 *
 * These mirror decode.cu and attention.cu one for one, with two differences that
 * run through all of them:
 *
 *   - **Token-major activations.** Every buffer is [n][dim]; a token's slice is
 *     contiguous so the GEMM's `x + j*in_f` indexing is a plain stride.
 *   - **The position is a kernel argument, not a device cursor.** The
 *     single-token path reads `pos` from device memory so the whole step can be
 *     one CUDA graph. A chunk pass cannot be graph-captured usefully anyway --
 *     speculative decoding accepts a variable number of tokens per round, so the
 *     shape changes every iteration -- so `pos0` is passed by value and the
 *     kernels are simpler for it.
 *
 * # Chunk attention does not need the sequence split
 *
 * attention.cu splits the sequence across blocks because 14 query heads cannot
 * fill 30 SMs. Here the grid is (heads, tokens): at n=16 that is 224 blocks
 * before any splitting, so the chunk dimension supplies the parallelism the
 * split was invented to create. One block per (head, token) sweeping the whole
 * prefix is both simpler and better occupied, and it makes the causal mask
 * trivial -- token j attends to cache positions [0, pos0+j], which is just the
 * loop bound.
 */

#include "common.cuh"
#include "hier.cuh"
#include <cuda_fp16.h>

#define CH_THREADS 256
#define CH_WARPS (CH_THREADS / WST_WARP)
#define CH_MAX_EPL 8 /* head_dim <= 32 * 8 */

/* ------------------------------------------------------------------ RMSNorm */

/* One block per token. `x` is [n][dim] fp32 residual, `out` is [n][dim] f16 for
 * the next GEMM. Same two-path structure as decode.cu: when the row fits in one
 * element per thread the value stays in a register across the reduction and the
 * kernel touches x once instead of twice. */
template <bool WIDE>
__global__ __launch_bounds__(1024) void rmsnorm_chunk_kernel(
    const float *__restrict__ x, const half *__restrict__ w, half *__restrict__ out,
    int dim, float eps) {

  __shared__ float warp_sums[1024 / WST_WARP];
  __shared__ float inv_rms;

  const float *xr = x + (size_t)blockIdx.x * dim;
  half *orow = out + (size_t)blockIdx.x * dim;

  float mine = 0.0f, acc = 0.0f;
  if (WIDE) {
    if (threadIdx.x < dim) {
      mine = xr[threadIdx.x];
      acc = mine * mine;
    }
  } else {
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
      const float v = xr[i];
      acc = fmaf(v, v, acc);
    }
  }

  acc = block_reduce_sum(acc, warp_sums);
  if (threadIdx.x == 0) inv_rms = rsqrtf(acc / (float)dim + eps);
  __syncthreads();

  const float s = inv_rms;
  if (WIDE) {
    if (threadIdx.x < dim)
      orow[threadIdx.x] = __float2half(mine * s * __half2float(w[threadIdx.x]));
  } else {
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
      orow[i] = __float2half(xr[i] * s * __half2float(w[i]));
  }
}

extern "C" wst_status_t wst_rmsnorm_chunk(const void *x, const void *w, void *out,
                                          int32_t dim, int32_t n, float eps) {
  WST_REQUIRE(x && w && out, "wst_rmsnorm_chunk: null pointer");
  WST_REQUIRE(dim > 0 && n > 0, "wst_rmsnorm_chunk: non-positive shape");

  if (dim <= 1024) {
    const int threads = ((dim + WST_WARP - 1) / WST_WARP) * WST_WARP;
    rmsnorm_chunk_kernel<true><<<n, threads>>>((const float *)x, (const half *)w,
                                               (half *)out, dim, eps);
  } else {
    rmsnorm_chunk_kernel<false><<<n, CH_THREADS>>>((const float *)x, (const half *)w,
                                                   (half *)out, dim, eps);
  }
  WST_TRY_KERNEL("wst_rmsnorm_chunk");
  return WST_OK;
}

/* -------------------------------------------------- RoPE + KV cache append */

/* Grid (n_q + n_kv, n): one block per head per token. Query heads rotate in
 * place; key/value heads rotate and write to cache slot pos0+j. */
__global__ void rope_cache_chunk_kernel(
    float *__restrict__ qkv, int qkv_stride, half *__restrict__ k_cache,
    half *__restrict__ v_cache, const float *__restrict__ cos_tab,
    const float *__restrict__ sin_tab, int n_q, int n_kv, int hd, int pos0, int n,
    int max_seq, const half *__restrict__ q_norm_w, const half *__restrict__ k_norm_w,
    float eps) {

  const int head = blockIdx.x;
  const int j = blockIdx.y;
  if (j >= n) return;

  const int halfd = hd >> 1;
  const int t = threadIdx.x;
  /* Padded to a whole warp when QK-norm is on -- see the decode kernel. */
  const bool active = (t < halfd);

  const int pos = min(pos0 + j, max_seq - 1);
  const float c = active ? cos_tab[(size_t)pos * halfd + t] : 0.0f;
  const float s = active ? sin_tab[(size_t)pos * halfd + t] : 0.0f;

  float *row = qkv + (size_t)j * qkv_stride;
  const float *k = row + (size_t)n_q * hd;
  const float *v = k + (size_t)n_kv * hd;

  if (head < n_q) {
    float *qh = row + (size_t)head * hd;
    float x1 = active ? qh[t] : 0.0f;
    float x2 = active ? qh[t + halfd] : 0.0f;
    if (q_norm_w) wst_qk_head_norm(x1, x2, q_norm_w, t, halfd, hd, eps, active);
    if (!active) return;
    qh[t] = x1 * c - x2 * s;
    qh[t + halfd] = x2 * c + x1 * s;
    return;
  }

  const int kvh = head - n_q;
  if (kvh >= n_kv) return;

  const size_t slot = ((size_t)kvh * max_seq + pos) * hd;

  const float *kh = k + (size_t)kvh * hd;
  float k1 = active ? kh[t] : 0.0f;
  float k2 = active ? kh[t + halfd] : 0.0f;
  if (k_norm_w) wst_qk_head_norm(k1, k2, k_norm_w, t, halfd, hd, eps, active);
  if (!active) return;
  k_cache[slot + t] = __float2half(k1 * c - k2 * s);
  k_cache[slot + t + halfd] = __float2half(k2 * c + k1 * s);

  const float *vh = v + (size_t)kvh * hd;
  v_cache[slot + t] = __float2half(vh[t]);
  v_cache[slot + t + halfd] = __float2half(vh[t + halfd]);
}

extern "C" wst_status_t wst_rope_cache_chunk(void *qkv, void *k_cache, void *v_cache,
                                             const void *cos_tab, const void *sin_tab,
                                             int32_t n_q, int32_t n_kv, int32_t head_dim,
                                             int32_t pos0, int32_t n, int32_t max_seq,
                                             const void *q_norm_w, const void *k_norm_w,
                                             float eps) {
  WST_REQUIRE(qkv && k_cache && v_cache && cos_tab && sin_tab,
              "wst_rope_cache_chunk: null pointer");
  WST_REQUIRE(n_q > 0 && n_kv > 0 && head_dim > 0 && n > 0,
              "wst_rope_cache_chunk: non-positive shape");
  WST_REQUIRE(head_dim % 2 == 0, "wst_rope_cache_chunk: head_dim must be even");
  WST_REQUIRE(pos0 >= 0 && pos0 + n <= max_seq,
              "wst_rope_cache_chunk: chunk runs past the cache");

  const int halfd = head_dim / 2;
  WST_REQUIRE(halfd <= 1024, "wst_rope_cache_chunk: head_dim/2 exceeds a block");

  const int qkv_stride = (n_q + 2 * n_kv) * head_dim;
  const bool norm = (q_norm_w != nullptr) || (k_norm_w != nullptr);
  const int threads = norm ? ((halfd + WST_WARP - 1) / WST_WARP) * WST_WARP : halfd;

  rope_cache_chunk_kernel<<<dim3(n_q + n_kv, n), threads>>>(
      (float *)qkv, qkv_stride, (half *)k_cache, (half *)v_cache,
      (const float *)cos_tab, (const float *)sin_tab, n_q, n_kv, head_dim, pos0, n,
      max_seq, (const half *)q_norm_w, (const half *)k_norm_w, eps);
  WST_TRY_KERNEL("wst_rope_cache_chunk");
  return WST_OK;
}

/* --------------------------------------------------------- chunk attention */

/* Block (head, j) attends query j to cache positions [0, pos0+j]. The upper
 * bound *is* the causal mask -- no masking arithmetic, no -inf fill. */
__global__ __launch_bounds__(CH_THREADS) void attn_chunk_kernel(
    const float *__restrict__ qkv, int qkv_stride, const half *__restrict__ k_cache,
    const half *__restrict__ v_cache, half *__restrict__ out, int gqa, int hd, int n_q,
    int pos0, int n, int max_seq, float scale) {

  const int head = blockIdx.x;
  const int j = blockIdx.y;
  if (j >= n) return;

  const int seq = min(pos0 + j + 1, max_seq);
  const int kvh = head / gqa;
  const int epl = hd / WST_WARP;

  extern __shared__ float smem[];
  float *qs = smem;
  float *m_s = qs + hd;
  float *l_s = m_s + CH_WARPS;
  float *acc_s = l_s + CH_WARPS;

  const float *qrow = qkv + (size_t)j * qkv_stride + (size_t)head * hd;
  for (int i = threadIdx.x; i < hd; i += CH_THREADS) qs[i] = qrow[i] * scale;
  __syncthreads();

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const half *kb = k_cache + (size_t)kvh * max_seq * hd;
  const half *vb = v_cache + (size_t)kvh * max_seq * hd;

  float m = -INFINITY, l = 0.0f;
  float acc[CH_MAX_EPL];
#pragma unroll
  for (int i = 0; i < CH_MAX_EPL; ++i) acc[i] = 0.0f;

  for (int t = warp; t < seq; t += CH_WARPS) {
    const half *kt = kb + (size_t)t * hd;
    float part = 0.0f;
    for (int i = 0; i < epl; ++i) {
      const int d = lane + i * WST_WARP;
      part = fmaf(qs[d], __half2float(kt[d]), part);
    }
    const float s = warp_reduce_sum(part);

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

  if (threadIdx.x < hd) {
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < CH_WARPS; ++w) M = fmaxf(M, m_s[w]);

    float num = 0.0f, den = 0.0f;
#pragma unroll
    for (int w = 0; w < CH_WARPS; ++w) {
      const float f = __expf(m_s[w] - M);
      num = fmaf(acc_s[w * hd + threadIdx.x], f, num);
      den = fmaf(l_s[w], f, den);
    }
    out[(size_t)j * n_q * hd + (size_t)head * hd + threadIdx.x] =
        __float2half(num / den);
  }
}

extern "C" wst_status_t wst_attn_chunk(const void *qkv, const void *k_cache,
                                       const void *v_cache, void *out, int32_t n_q,
                                       int32_t n_kv, int32_t head_dim, int32_t pos0,
                                       int32_t n, int32_t max_seq, float scale) {
  WST_REQUIRE(qkv && k_cache && v_cache && out, "wst_attn_chunk: null pointer");
  WST_REQUIRE(n_q > 0 && n_kv > 0 && head_dim > 0 && n > 0,
              "wst_attn_chunk: non-positive shape");
  WST_REQUIRE(n_q % n_kv == 0,
              "wst_attn_chunk: query heads must be a multiple of key/value heads");
  WST_REQUIRE(head_dim % WST_WARP == 0,
              "wst_attn_chunk: head_dim must be a multiple of the warp size");
  WST_REQUIRE(head_dim <= WST_WARP * CH_MAX_EPL, "wst_attn_chunk: head_dim too large");
  WST_REQUIRE(pos0 >= 0 && pos0 + n <= max_seq, "wst_attn_chunk: chunk runs past cache");

  const size_t smem =
      (size_t)(head_dim + 2 * CH_WARPS + CH_WARPS * head_dim) * sizeof(float);
  WST_REQUIRE(smem <= 48u * 1024u, "wst_attn_chunk: head_dim too large for shared memory");

  const int qkv_stride = (n_q + 2 * n_kv) * head_dim;
  attn_chunk_kernel<<<dim3(n_q, n), CH_THREADS, smem>>>(
      (const float *)qkv, qkv_stride, (const half *)k_cache, (const half *)v_cache,
      (half *)out, n_q / n_kv, head_dim, n_q, pos0, n, max_seq, scale);
  WST_TRY_KERNEL("wst_attn_chunk");
  return WST_OK;
}

/* ------------------------------------------------------------------ SwiGLU */

/* gate_up is [n][2*inter] fp32 from one fused projection; out is [n][inter] f16. */
__global__ __launch_bounds__(CH_THREADS) void swiglu_chunk_kernel(
    const float *__restrict__ gate_up, half *__restrict__ out, int inter) {
  const int i = blockIdx.x * CH_THREADS + threadIdx.x;
  if (i >= inter) return;
  const float *row = gate_up + (size_t)blockIdx.y * 2 * inter;
  const float g = row[i];
  out[(size_t)blockIdx.y * inter + i] =
      __float2half(g * __frcp_rn(1.0f + __expf(-g)) * row[i + inter]);
}

extern "C" wst_status_t wst_swiglu_chunk(const void *gate_up, void *out, int32_t inter,
                                         int32_t n) {
  WST_REQUIRE(gate_up && out, "wst_swiglu_chunk: null pointer");
  WST_REQUIRE(inter > 0 && n > 0, "wst_swiglu_chunk: non-positive shape");
  const int blocks = (inter + CH_THREADS - 1) / CH_THREADS;
  swiglu_chunk_kernel<<<dim3(blocks, n), CH_THREADS>>>((const float *)gate_up,
                                                       (half *)out, inter);
  WST_TRY_KERNEL("wst_swiglu_chunk");
  return WST_OK;
}

/* --------------------------------------------------------------- embedding */

__global__ void embed_fp16_chunk_kernel(const half *__restrict__ table,
                                        const int32_t *__restrict__ tokens,
                                        float *__restrict__ out, int hidden, int rows) {
  const int i = blockIdx.x * CH_THREADS + threadIdx.x;
  if (i >= hidden) return;
  const int token = min(max(tokens[blockIdx.y], 0), rows - 1);
  out[(size_t)blockIdx.y * hidden + i] = __half2float(table[(size_t)token * hidden + i]);
}

__global__ void embed_int4_chunk_kernel(const uint32_t *__restrict__ qw,
                                        const uint32_t *__restrict__ sz,
                                        const int32_t *__restrict__ tokens,
                                        float *__restrict__ out, int hidden, int rows) {
  const int i = blockIdx.x * CH_THREADS + threadIdx.x;
  if (i >= hidden) return;
  const int token = min(max(tokens[blockIdx.y], 0), rows - 1);

  const size_t row_words = (size_t)hidden / 8;
  const size_t row_groups = (size_t)hidden / 128;

  const uint32_t word = qw[(size_t)token * row_words + i / 8];
  const uint32_t q = (word >> (4 * (i % 8))) & 0xFu;
  const uint32_t packed = sz[(size_t)token * row_groups + i / 128];
  const float scale = __half2float(__ushort_as_half((unsigned short)(packed & 0xFFFFu)));
  const float zero = __half2float(__ushort_as_half((unsigned short)(packed >> 16)));

  out[(size_t)blockIdx.y * hidden + i] = ((float)q - zero) * scale;
}

__global__ void embed_hier_chunk_kernel(const uint32_t *__restrict__ qw,
                                        const uint8_t *__restrict__ si,
                                        const half2 *__restrict__ sb,
                                        const int32_t *__restrict__ tokens,
                                        float *__restrict__ out, int hidden, int rows) {
  const int i = blockIdx.x * CH_THREADS + threadIdx.x;
  if (i >= hidden) return;
  const int token = min(max(tokens[blockIdx.y], 0), rows - 1);

  const int words_per_row = hidden / 8;
  const int groups_per_row = hidden / HGROUP;

  const half2 p = sb[token];
  const float d = __half2float(__low2half(p));
  const float dm = __half2float(__high2half(p));

  const uint32_t word = qw[(size_t)token * words_per_row + i / 8];
  const float q = (float)((word >> (4 * (i % 8))) & 0xFu);
  const uint8_t idx = si[(size_t)token * groups_per_row + i / HGROUP];

  out[(size_t)blockIdx.y * hidden + i] =
      q * (d * (float)(idx & 0xF)) - dm * (float)(idx >> 4);
}

extern "C" wst_status_t wst_embed_fp16_chunk(const void *table, const void *tokens,
                                             void *out, int32_t hidden, int32_t rows,
                                             int32_t n) {
  WST_REQUIRE(table && tokens && out, "wst_embed_fp16_chunk: null pointer");
  WST_REQUIRE(hidden > 0 && rows > 0 && n > 0, "wst_embed_fp16_chunk: bad argument");
  const int blocks = (hidden + CH_THREADS - 1) / CH_THREADS;
  embed_fp16_chunk_kernel<<<dim3(blocks, n), CH_THREADS>>>(
      (const half *)table, (const int32_t *)tokens, (float *)out, hidden, rows);
  WST_TRY_KERNEL("wst_embed_fp16_chunk");
  return WST_OK;
}

extern "C" wst_status_t wst_embed_int4_g128_chunk(const void *qw, const void *sz,
                                                  const void *tokens, void *out,
                                                  int32_t hidden, int32_t rows,
                                                  int32_t n) {
  WST_REQUIRE(qw && sz && tokens && out, "wst_embed_int4_g128_chunk: null pointer");
  WST_REQUIRE(hidden > 0 && rows > 0 && n > 0, "wst_embed_int4_g128_chunk: bad argument");
  WST_REQUIRE(hidden % 128 == 0,
              "wst_embed_int4_g128_chunk: hidden must be a multiple of 128");
  const int blocks = (hidden + CH_THREADS - 1) / CH_THREADS;
  embed_int4_chunk_kernel<<<dim3(blocks, n), CH_THREADS>>>(
      (const uint32_t *)qw, (const uint32_t *)sz, (const int32_t *)tokens, (float *)out,
      hidden, rows);
  WST_TRY_KERNEL("wst_embed_int4_g128_chunk");
  return WST_OK;
}

extern "C" wst_status_t wst_embed_int4_hier_chunk(const void *qw, const void *si,
                                                  const void *sb, const void *tokens,
                                                  void *out, int32_t hidden, int32_t rows,
                                                  int32_t n) {
  WST_REQUIRE(qw && si && sb && tokens && out, "wst_embed_int4_hier_chunk: null pointer");
  WST_REQUIRE(hidden > 0 && rows > 0 && n > 0, "wst_embed_int4_hier_chunk: bad argument");
  WST_REQUIRE(hidden % HGROUP == 0,
              "wst_embed_int4_hier_chunk: hidden must be a multiple of 32");
  const int blocks = (hidden + CH_THREADS - 1) / CH_THREADS;
  embed_hier_chunk_kernel<<<dim3(blocks, n), CH_THREADS>>>(
      (const uint32_t *)qw, (const uint8_t *)si, (const half2 *)sb,
      (const int32_t *)tokens, (float *)out, hidden, rows);
  WST_TRY_KERNEL("wst_embed_int4_hier_chunk");
  return WST_OK;
}

/* ------------------------------------------------------------- row argmax */

/* One block per row of [n][vocab]. Verification needs the target model's greedy
 * choice at every chunk position, and copying n x 151936 floats to the host to
 * find them would cost 4.9 MB over a 5.8 GB/s link -- 0.85 ms, a third of a
 * token. Reducing on the device returns n integers instead.
 *
 * Ties go to the lower index, matching sample.cu so greedy chunk decode and
 * greedy single-token decode cannot diverge on a tie. */
__global__ __launch_bounds__(CH_THREADS) void argmax_chunk_kernel(
    const float *__restrict__ logits, int32_t *__restrict__ out, int vocab) {

  __shared__ float s_val[CH_THREADS];
  __shared__ int s_idx[CH_THREADS];

  const float *row = logits + (size_t)blockIdx.x * vocab;

  float best = -INFINITY;
  int best_i = 0;
  for (int i = threadIdx.x; i < vocab; i += CH_THREADS) {
    const float v = row[i];
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  s_val[threadIdx.x] = best;
  s_idx[threadIdx.x] = best_i;
  __syncthreads();

  for (int stride = CH_THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float o = s_val[threadIdx.x + stride];
      const int oi = s_idx[threadIdx.x + stride];
      if (o > s_val[threadIdx.x] || (o == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
        s_val[threadIdx.x] = o;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) out[blockIdx.x] = s_idx[0];
}

extern "C" wst_status_t wst_argmax_chunk(const void *logits, void *out, int32_t vocab,
                                         int32_t n) {
  WST_REQUIRE(logits && out, "wst_argmax_chunk: null pointer");
  WST_REQUIRE(vocab > 0 && n > 0, "wst_argmax_chunk: non-positive shape");
  argmax_chunk_kernel<<<n, CH_THREADS>>>((const float *)logits, (int32_t *)out, vocab);
  WST_TRY_KERNEL("wst_argmax_chunk");
  return WST_OK;
}
