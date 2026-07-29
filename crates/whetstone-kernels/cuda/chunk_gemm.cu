/* Multi-token GEMM: the same weights, N activation vectors, one pass.
 *
 * # Why this kernel is the whole point
 *
 * Decode at batch 1 reads every weight once and does one multiply-add with it.
 * Arithmetic intensity ~2 FLOP/byte against a machine that wants ~120, so the
 * GEMV is bandwidth bound and *no amount of cheaper arithmetic helps* (CLAUDE.md
 * §2). The only way to change that is to use each weight more than once, and the
 * only way to do that at batch 1 is to have more than one token in flight.
 *
 * Three things need exactly that, and they are the same kernel:
 *
 *   1. **Prefill.** Currently `prompt_len` separate decode steps -- it re-reads
 *      the entire model once per prompt token.
 *   2. **Speculative decoding.** A draft model proposes N tokens; the target
 *      model verifies all N in one pass. Lossless, and the decode win is the
 *      mean accepted length.
 *   3. **Offload.** `notes/2026-07-29-offload-roofline.md` §6: streaming weights
 *      from host RAM is hideable only behind compute, and at N=1 there is none.
 *      N=8 makes the pass compute bound and buys ~9% of the model off-card free.
 *
 * # Structure
 *
 * `x` is [n][in_f] half, `y` is [n][out_f] float -- token-major, so each token's
 * activations stay contiguous and the next stage's per-token kernels index by a
 * plain stride.
 *
 * A warp owns TILE output rows and all N tokens. The loop nest is deliberately
 * **weights outermost**:
 *
 *     for group v:                  # the coalesced weight load
 *       for t in TILE:              # rows
 *         load + dequantize once -> q2[16] in registers
 *         for n in N:               # tokens
 *           reload x[n] (L1) and FMA
 *
 * The weight nibbles are unpacked **once per (row, group)** and reused across all
 * N tokens; `x` is re-read per token instead. That is the right way round: the
 * weight load is DRAM and the whole cost model, while `x` is a couple of
 * kilobytes per token that is L1-resident after first touch. Unpacking inside
 * the token loop instead would cost N x 16 `hpair` calls per group -- measured
 * as the difference between compute bound and ALU bound.
 *
 * # Register budget, which is what caps N
 *
 * Live across the inner loop: q2[16] + xw[16] + acc[TILE][NMAX] + sx[NMAX].
 * At TILE=2, NMAX=16 that is ~99 registers and occupancy drops to 2 blocks/SM,
 * so the dispatch drops to TILE=1 once NMAX >= 8 -- the batch dimension already
 * supplies the reuse that TILE existed to provide, so nothing is lost.
 *
 * Batches wider than CHUNK_NMAX re-read the weights once per slice of 16. For
 * prefill at n=512 that is 32 weight passes rather than 512, and the pass is
 * compute bound at that width anyway.
 */

#include "common.cuh"
#include "hier.cuh"
#include <cuda_fp16.h>

#define CHUNK_THREADS 256
#define CHUNK_NMAX 16

/* ------------------------------------------------- int4 hierarchical scales */

template <int THREADS, int TILE, int NMAX>
__global__ __launch_bounds__(THREADS) void gemm_int4_hier_kernel(
    const uint4 *__restrict__ qw,   /* [out][in/32] uint4, 32 nibbles each */
    const uint8_t *__restrict__ si, /* [out][in/32] uint8, ls | lm<<4      */
    const half2 *__restrict__ sb,   /* [out]        half2 (d, dmin)        */
    const half *__restrict__ x,     /* [n][in_f]                            */
    const half *__restrict__ bias, float *__restrict__ y, /* [n][out_f]     */
    int in_f, int out_f, int n, int rows_per_block, int accum) {

  constexpr int WARPS = THREADS / WST_WARP;

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const int vec_per_row = in_f / HGROUP;
  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);

  const half2 centre = __float2half2_rn(HST_CENTRE_F);

  for (int row = row0 + warp * TILE; row < row1; row += WARPS * TILE) {
    const int ntile = min(TILE, row1 - row);

    float d[TILE], dm[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      const half2 p = (t < ntile) ? sb[row + t] : __float2half2_rn(0.0f);
      d[t] = __half2float(__low2half(p));
      dm[t] = __half2float(__high2half(p));
    }

    float acc[TILE][NMAX];
#pragma unroll
    for (int t = 0; t < TILE; ++t)
#pragma unroll
      for (int j = 0; j < NMAX; ++j) acc[t][j] = 0.0f;

    for (int v = lane; v < vec_per_row; v += WST_WARP) {
      /* Every row of the tile, loaded and **unpacked once**, held in registers
       * across the whole token loop.
       *
       * This ordering is the entire performance story of the kernel. The obvious
       * arrangement -- rows outside, tokens inside, x re-read per row -- issues
       * `TILE * NMAX` vector loads of x per group against a single 17-byte weight
       * read, and Turing's L1 gives out long before the FP16x2 pipe does. Loading
       * x once per (group, token) and reusing it across all TILE rows measured
       * **2x faster at every shape and every N** (research/experiments/
       * probe_chunk_gemm.cu). The cost is 16*TILE registers for the unpacked
       * half2s, which is what caps TILE at 4. */
      half2 q2[TILE][16];
      float s[TILE], bias_term[TILE];
#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (t >= ntile) break;
        const uint4 packed = qw[(size_t)(row + t) * vec_per_row + v];
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
        const uint8_t idx = si[(size_t)(row + t) * vec_per_row + v];
        s[t] = d[t] * (float)(idx & 0xF);
        bias_term[t] = fmaf(8.0f, s[t], -dm[t] * (float)(idx >> 4));
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            q2[t][w * 4 + i] = __hsub2(hpair(words[w], i), centre);
      }

#pragma unroll
      for (int j = 0; j < NMAX; ++j) {
        if (j >= n) break;
        const uint4 *xv = (const uint4 *)(x + (size_t)j * in_f) + (size_t)v * 4;
        const uint4 a0 = __ldg(xv + 0), a1 = __ldg(xv + 1);
        const uint4 a2 = __ldg(xv + 2), a3 = __ldg(xv + 3);
        const uint32_t xw[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                                 a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};

        /* The group sum `w = q*s + m` needs, folded into the same pass over x
         * rather than a second one. */
        half2 s2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int i = 0; i < 16; ++i) s2 = __hadd2(s2, *(const half2 *)&xw[i]);
        const float sx = h2_sum(s2);

#pragma unroll
        for (int t = 0; t < TILE; ++t) {
          if (t >= ntile) break;
          half2 dot = __float2half2_rn(0.0f);
#pragma unroll
          for (int i = 0; i < 16; ++i)
            dot = __hfma2(q2[t][i], *(const half2 *)&xw[i], dot);
          acc[t][j] = fmaf(s[t], h2_sum(dot), acc[t][j]);
          acc[t][j] = fmaf(bias_term[t], sx, acc[t][j]);
        }
      }
    }

#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      if (t >= ntile) break;
#pragma unroll
      for (int j = 0; j < NMAX; ++j) {
        if (j >= n) break;
        const float r = warp_reduce_sum(acc[t][j]);
        if (lane == 0) {
          float o = r;
          if (bias) o += __half2float(bias[row + t]);
          float *dst = y + (size_t)j * out_f + row + t;
          *dst = accum ? *dst + o : o;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------- fp16 */

/* The lossless reference path, and whatever `lm_head` the converter kept exact.
 * Same loop order and the same reason for it: the weight row is the DRAM cost,
 * `x` is L1. */
template <int THREADS, int TILE, int NMAX>
__global__ __launch_bounds__(THREADS) void gemm_fp16_kernel(
    const half *__restrict__ w, const half *__restrict__ x,
    const half *__restrict__ bias, float *__restrict__ y, int in_f, int out_f, int n,
    int rows_per_block, int accum) {

  constexpr int WARPS = THREADS / WST_WARP;

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);
  const int nvec = in_f / 8; /* uint4 == 8 halves */

  for (int row = row0 + warp * TILE; row < row1; row += WARPS * TILE) {
    const int ntile = min(TILE, row1 - row);

    float acc[TILE][NMAX];
#pragma unroll
    for (int t = 0; t < TILE; ++t)
#pragma unroll
      for (int j = 0; j < NMAX; ++j) acc[t][j] = 0.0f;

    for (int v = lane; v < nvec; v += WST_WARP) {
#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (t >= ntile) break;
        const uint4 wv = __ldg((const uint4 *)(w + (size_t)(row + t) * in_f) + v);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};

#pragma unroll
        for (int j = 0; j < NMAX; ++j) {
          if (j >= n) break;
          const uint4 xv = __ldg((const uint4 *)(x + (size_t)j * in_f) + v);
          const uint32_t xx[4] = {xv.x, xv.y, xv.z, xv.w};
          half2 dot = __float2half2_rn(0.0f);
#pragma unroll
          for (int i = 0; i < 4; ++i)
            dot = __hfma2(*(const half2 *)&ww[i], *(const half2 *)&xx[i], dot);
          acc[t][j] += h2_sum(dot);
        }
      }
    }

    /* Tail for widths that are not a multiple of 8. */
    for (int c = nvec * 8 + lane; c < in_f; c += WST_WARP) {
#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (t >= ntile) break;
        const float wv = __half2float(__ldg(w + (size_t)(row + t) * in_f + c));
#pragma unroll
        for (int j = 0; j < NMAX; ++j) {
          if (j >= n) break;
          acc[t][j] = fmaf(wv, __half2float(__ldg(x + (size_t)j * in_f + c)), acc[t][j]);
        }
      }
    }

#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      if (t >= ntile) break;
#pragma unroll
      for (int j = 0; j < NMAX; ++j) {
        if (j >= n) break;
        const float r = warp_reduce_sum(acc[t][j]);
        if (lane == 0) {
          float o = r;
          if (bias) o += __half2float(bias[row + t]);
          float *dst = y + (size_t)j * out_f + row + t;
          *dst = accum ? *dst + o : o;
        }
      }
    }
  }
}

/* ------------------------------------------------------------- dispatch */

static int chunk_sm_count() {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceProp p;
    sms = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.multiProcessorCount : 30;
  }
  return sms;
}

template <int THREADS, int TILE>
static int chunk_rows_for(int out_f) {
  const int quant = (THREADS / WST_WARP) * TILE;
  int rows = (out_f + chunk_sm_count() * 4 - 1) / (chunk_sm_count() * 4);
  rows = ((rows + quant - 1) / quant) * quant;
  if (rows < quant) rows = quant;
  if (rows > 512) rows = 512;
  return rows;
}

/* The slice width actually used for a request of `n` tokens: the smallest
 * supported NMAX that covers it, capped at CHUNK_NMAX. Rounding up rather than
 * down means a chunk of 5 runs as one pass of NMAX=8 with three lanes idle,
 * which is cheaper than two passes that each re-read every weight. */
static int chunk_slice_width(int n) {
  if (n <= 1) return 1;
  if (n <= 2) return 2;
  if (n <= 4) return 4;
  if (n <= 8) return 8;
  return CHUNK_NMAX;
}

#define HIER_LAUNCH(NM, TL)                                                        \
  do {                                                                             \
    const int rows = chunk_rows_for<CHUNK_THREADS, TL>(out_f);                     \
    const int blocks = (out_f + rows - 1) / rows;                                  \
    gemm_int4_hier_kernel<CHUNK_THREADS, TL, NM><<<blocks, CHUNK_THREADS>>>(        \
        (const uint4 *)qw, (const uint8_t *)si, (const half2 *)sb,                 \
        (const half *)xs, (const half *)bias, ys, in_f, out_f, take, rows, accum); \
  } while (0)

extern "C" wst_status_t wst_gemm_int4_hier(const void *qw, const void *si,
                                           const void *sb, const void *x,
                                           const void *bias, void *y, int32_t in_f,
                                           int32_t out_f, int32_t n, int32_t accum) {
  WST_REQUIRE(qw && si && sb && x && y, "wst_gemm_int4_hier: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemm_int4_hier: non-positive dimension");
  WST_REQUIRE(n > 0, "wst_gemm_int4_hier: n must be positive");
  WST_REQUIRE(in_f % HGROUP == 0,
              "wst_gemm_int4_hier: in_features must be a multiple of 32");

  for (int base = 0; base < n; base += CHUNK_NMAX) {
    const int take = min(CHUNK_NMAX, n - base);
    const half *xs = (const half *)x + (size_t)base * in_f;
    float *ys = (float *)y + (size_t)base * out_f;

    /* TILE rises with N rather than falling. Each x load is amortised over TILE
     * rows, so a wider token batch wants *more* rows per warp, not fewer -- the
     * opposite of what the single-token kernel wants and the opposite of this
     * dispatch's first version. Swept in probe_chunk_gemm.cu across all five
     * model shapes; TILE=4 won at N>=4 everywhere, by 1.2-1.9x over TILE=1. */
    switch (chunk_slice_width(take)) {
      case 1: HIER_LAUNCH(1, 2); break;
      case 2: HIER_LAUNCH(2, 2); break;
      case 4: HIER_LAUNCH(4, 4); break;
      case 8: HIER_LAUNCH(8, 4); break;
      default: HIER_LAUNCH(CHUNK_NMAX, 4); break;
    }
    WST_TRY_KERNEL("wst_gemm_int4_hier");
  }
  return WST_OK;
}

#define FP16_LAUNCH(NM, TL)                                                        \
  do {                                                                             \
    const int rows = chunk_rows_for<CHUNK_THREADS, TL>(out_f);                     \
    const int blocks = (out_f + rows - 1) / rows;                                  \
    gemm_fp16_kernel<CHUNK_THREADS, TL, NM><<<blocks, CHUNK_THREADS>>>(            \
        (const half *)w, (const half *)xs, (const half *)bias, ys, in_f, out_f,    \
        take, rows, accum);                                                        \
  } while (0)

extern "C" wst_status_t wst_gemm_fp16(const void *w, const void *x, const void *bias,
                                      void *y, int32_t in_f, int32_t out_f, int32_t n,
                                      int32_t accum) {
  WST_REQUIRE(w && x && y, "wst_gemm_fp16: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemm_fp16: non-positive dimension");
  WST_REQUIRE(n > 0, "wst_gemm_fp16: n must be positive");

  for (int base = 0; base < n; base += CHUNK_NMAX) {
    const int take = min(CHUNK_NMAX, n - base);
    const half *xs = (const half *)x + (size_t)base * in_f;
    float *ys = (float *)y + (size_t)base * out_f;

    switch (chunk_slice_width(take)) {
      case 1: FP16_LAUNCH(1, 2); break;
      case 2: FP16_LAUNCH(2, 2); break;
      case 4: FP16_LAUNCH(4, 2); break;
      case 8: FP16_LAUNCH(8, 1); break;
      default: FP16_LAUNCH(CHUNK_NMAX, 1); break;
    }
    WST_TRY_KERNEL("wst_gemm_fp16");
  }
  return WST_OK;
}

extern "C" int32_t wst_chunk_max_tokens(void) { return CHUNK_NMAX; }
