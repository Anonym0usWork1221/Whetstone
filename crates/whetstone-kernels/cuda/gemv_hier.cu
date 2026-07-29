/* int4 with hierarchical scale metadata: group 32 at the cost of group 128.
 *
 * # Why this format exists
 *
 * Measured on Qwen2.5-0.5B (wikitext-2, 20x2048 windows, body tensors only,
 * perplexity delta against fp16):
 *
 *     int4 g128, fp16 scale + fp16 zero   4.250 bpw   +2.730
 *     the full k-quant alternating fit    4.250 bpw   +2.575
 *     int4 g64,  fp16 scale + fp16 zero   4.500 bpw   +1.771
 *     int4 g32,  fp16 scale + fp16 zero   5.000 bpw   +1.696
 *     THIS: int4 g32, 4-bit indices       4.277 bpw   +1.575
 *
 * Granularity is worth six times what the fitting algorithm is worth, and it is
 * the one thing the shipping format could not buy, because an fp16 scale and an
 * fp16 zero per group of 32 is 1.0 bits/weight of metadata against g128's 0.25 --
 * spending exactly the bandwidth the engine exists to save.
 *
 * The way out is k-quant's: make the per-group metadata SMALL and express it
 * against one fp16 pair per row.
 *
 *     stored per row     : half2 (d, dmin)
 *     stored per group   : uint8  (ls | lm<<4), two 4-bit indices
 *     reconstruction     : scale = d*ls,  min = -dmin*lm,  w = q*scale + min
 *
 *     bits/weight = 4 + 8/32 + 32/in_features
 *                 = 4.286 at in=896,  4.257 at in=4864
 *
 * against 4.250 for the format it replaces. **0.03 bits for a 1.15 perplexity
 * improvement**, and the arithmetic below is the same instruction mix.
 *
 * # The two things that make it as cheap as it is
 *
 * **1. A group is exactly one vector load.** `uint4` is 32 nibbles, and the group
 * is 32 weights, so every lane handles precisely one group per step and the
 * metadata read is one byte per lane -- a warp pulls 32 contiguous bytes.
 *
 * **2. The activation group-sums are computed here, not in a prologue kernel.**
 * `w = q*s + m` makes the dot product `s*sum(q_i x_i) + m*sum(x_i)`, so the
 * kernel needs a per-group sum of x. The obvious implementation is a small
 * kernel writing `in_features/32` floats before every GEMV -- 97 extra launches
 * per token, a scratch buffer, and a change to every call site. But the lane that
 * owns group g has already loaded exactly the 32 activations it needs to sum
 * them, and that sum is shared across all TILE rows the warp is accumulating. So
 * it costs 16 half2 adds per 32 weights, amortised over TILE rows, and the
 * format change needs no new kernel, no scratch, and no API change.
 *
 * The levels are re-centred on 8 before the dot product:
 *
 *     sum_i w_i x_i = s * sum_i (q_i - 8) x_i  +  (8s + m) * sum_i x_i
 *
 * which costs nothing (the magic constant becomes 1024+8 = 1032, still exactly
 * representable) and keeps the fp16 partial sums centred. Accumulating raw
 * q in [0,15] instead would push the half2 accumulator to ~15x the activation
 * magnitude and spend mantissa on an offset that cancels anyway.
 */

#include "common.cuh"
#include "hier.cuh"
#include <cuda_fp16.h>

/* One warp accumulates TILE output rows; one block sweeps `rows_per_block`.
 *
 * `x` is read through __ldg rather than staged in shared memory. Coalescing the
 * weight loads forces lane n to own columns [32n, 32n+32), so its reads of x land
 * 64 B apart -- two of thirty-two banks, a 16-way conflict that measured slower
 * than fp16 despite moving 3.75x fewer bytes. x is a couple of kilobytes and
 * every block reads all of it, so it is L1-resident after the first touch.
 */
template <int THREADS, int TILE>
__global__ __launch_bounds__(THREADS) void gemv_int4_hier_kernel(
    const uint4 *__restrict__ qw,   /* [out][in/32] uint4, 32 nibbles each   */
    const uint8_t *__restrict__ si, /* [out][in/32] uint8, ls | lm<<4        */
    const half2 *__restrict__ sb,   /* [out]        half2 (d, dmin)          */
    const half *__restrict__ x, const half *__restrict__ bias,
    float *__restrict__ y, int in_f, int out_f, int rows_per_block, int accum) {

  constexpr int WARPS = THREADS / WST_WARP;

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const int vec_per_row = in_f / HGROUP; /* == groups per row, by construction */

  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);

  const uint4 *xv = (const uint4 *)x;
  const half2 centre = __float2half2_rn(HST_CENTRE_F);

  for (int row = row0 + warp * TILE; row < row1; row += WARPS * TILE) {
    const int ntile = min(TILE, row1 - row);

    /* One fp16 pair per row, read once and held in registers for the whole
     * reduction. This is the entire cost of the hierarchy on the load side. */
    float d[TILE], dm[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      const half2 p = (t < ntile) ? sb[row + t] : __float2half2_rn(0.0f);
      d[t] = __half2float(__low2half(p));
      dm[t] = __half2float(__high2half(p));
    }

    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;

    for (int v = lane; v < vec_per_row; v += WST_WARP) {
      const int xbase = v * 4; /* 32 halves = 64 B = 4 uint4 */
      const uint4 xa = __ldg(&xv[xbase + 0]);
      const uint4 xb = __ldg(&xv[xbase + 1]);
      const uint4 xc = __ldg(&xv[xbase + 2]);
      const uint4 xd = __ldg(&xv[xbase + 3]);
      const uint32_t xw[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                               xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};

      /* Group sum of the activations, computed from the loads already in
       * registers and shared across all TILE rows. See the header comment for
       * why this is not a separate kernel. */
      half2 xs2 = __float2half2_rn(0.0f);
#pragma unroll
      for (int i = 0; i < 16; ++i) xs2 = __hadd2(xs2, *(const half2 *)&xw[i]);
      const float sx = __half2float(__low2half(xs2)) + __half2float(__high2half(xs2));

#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (t >= ntile) break;

        const uint4 packed = qw[(size_t)(row + t) * vec_per_row + v];
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

        const uint8_t idx = si[(size_t)(row + t) * vec_per_row + v];
        const float s = d[t] * (float)(idx & 0xF);
        const float m = -dm[t] * (float)(idx >> 4);

        half2 dot = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const half2 q2 = __hsub2(hpair(words[w], i), centre);
            dot = __hfma2(q2, *(const half2 *)&xw[w * 4 + i], dot);
          }
        }
        const float dotf =
            __half2float(__low2half(dot)) + __half2float(__high2half(dot));
        /* sum w_i x_i = s * sum (q_i - 8) x_i + (8s + m) * sum x_i */
        acc[t] = fmaf(s, dotf, acc[t]);
        acc[t] = fmaf(fmaf(8.0f, s, m), sx, acc[t]);
      }
    }

#pragma unroll
    for (int t = 0; t < TILE; ++t) {
      if (t >= ntile) break;
      float r = warp_reduce_sum(acc[t]);
      if (lane == 0) {
        if (bias) r += __half2float(bias[row + t]);
        y[row + t] = accum ? y[row + t] + r : r;
      }
    }
  }
}

/* Dequantize one row -- the input-embedding gather when the tied matrix is
 * stored in this format. Cheap and rare (1.8 KB against 68 MB of projection),
 * so it is written for obviousness rather than speed. */
__global__ void embed_int4_hier_kernel(const uint32_t *__restrict__ qw,
                                       const uint8_t *__restrict__ si,
                                       const half2 *__restrict__ sb,
                                       const int32_t *__restrict__ row_ptr,
                                       float *__restrict__ out, int in_f,
                                       int vocab) {
  const int row = min(max(*row_ptr, 0), vocab - 1);
  const int words_per_row = in_f / 8;
  const int groups_per_row = in_f / HGROUP;

  const half2 p = sb[row];
  const float d = __half2float(__low2half(p));
  const float dm = __half2float(__high2half(p));

  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < in_f;
       c += gridDim.x * blockDim.x) {
    const uint32_t word = qw[(size_t)row * words_per_row + c / 8];
    const float q = (float)((word >> (4 * (c % 8))) & 0xFu);
    const uint8_t idx = si[(size_t)row * groups_per_row + c / HGROUP];
    out[c] = q * (d * (float)(idx & 0xF)) - dm * (float)(idx >> 4);
  }
}

/* ---------------------------------------------------------------- dispatch */

static int hier_sm_count() {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceProp p;
    sms = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.multiProcessorCount : 30;
  }
  return sms;
}

template <int THREADS, int TILE>
static int hier_rows_for(int out_f) {
  const int quant = (THREADS / WST_WARP) * TILE;
  int rows = (out_f + hier_sm_count() * 4 - 1) / (hier_sm_count() * 4);
  rows = ((rows + quant - 1) / quant) * quant;
  if (rows < quant) rows = quant;
  if (rows > 512) rows = 512;
  return rows;
}

template <int THREADS, int TILE>
static void hier_launch(const void *qw, const void *si, const void *sb, const void *x,
                        const void *bias, void *y, int in_f, int out_f, int accum) {
  const int rows = hier_rows_for<THREADS, TILE>(out_f);
  const int blocks = (out_f + rows - 1) / rows;
  gemv_int4_hier_kernel<THREADS, TILE><<<blocks, THREADS>>>(
      (const uint4 *)qw, (const uint8_t *)si, (const half2 *)sb, (const half *)x,
      (const half *)bias, (float *)y, in_f, out_f, rows, accum);
}

/* TILE is swept the same way the g128 kernel's was, because the trade is the
 * same one: more rows per warp buys in-flight bytes and costs warp-level
 * parallelism, and which side wins depends on how many rows the shape has.
 * `whetstone tune` settles it end to end; the per-stage profiler misranks
 * choices this close together, and a microbenchmark exaggerates the spread by
 * more than an order of magnitude because a matrix rerun 200 times stays in a
 * 3 MB L2 that a decode step never gets to use. */
/* Tile index 0..3 selects 1, 2, 4 or 8 rows per warp (see the switch in
 * `wst_gemv_int4_hier_ex`).
 *
 * TILE=2 everywhere, and that is NOT what the g128 kernel wants -- its swept
 * winner is t8 for the q/k/v, o and gate/up shapes. Measured end to end at 384
 * generated tokens, median of three:
 *
 *     rule (wide, huge, other)      tok/s
 *     1,1,1   t2 everywhere         423.3   <- default
 *     1,3,1                         417.3
 *     1,2,1                         416.4
 *     1,1,2                         405.4
 *     1,1,3   (the g128 winner)     386.6
 *     0,0,1                         364.4
 *
 * A 16% spread, against 2.9% for the entire 27-rule space on the g128 kernel.
 * The mechanism is register pressure: this kernel holds `d[TILE]` and `dm[TILE]`
 * live across the whole reduction, so TILE=8 costs sixteen more registers per
 * thread than the g128 kernel does at the same tile, and the occupancy that buys
 * back more than the extra in-flight bytes are worth.
 *
 * Worth noting for the next format change: "the tile rule barely matters" was a
 * true statement about a different kernel, and it stopped being true the moment
 * the kernel started carrying per-row state. */
static int kHierRule[3] = {1, 1, 1}; /* wide reduction, huge output, everything else */

extern "C" void wst_gemv_hier_set_rule(int32_t wide, int32_t huge, int32_t other) {
  int *r = kHierRule;
  if (wide >= 0 && wide < 4) r[0] = wide;
  if (huge >= 0 && huge < 4) r[1] = huge;
  if (other >= 0 && other < 4) r[2] = other;
}

extern "C" void wst_gemv_hier_get_rule(int32_t *out) {
  out[0] = kHierRule[0];
  out[1] = kHierRule[1];
  out[2] = kHierRule[2];
}

static int hier_tile_for(int in_f, int out_f) {
  if (in_f >= 2048) return kHierRule[0];
  if (out_f >= 65536) return kHierRule[1];
  return kHierRule[2];
}

extern "C" wst_status_t wst_gemv_int4_hier_ex(const void *qw, const void *si,
                                              const void *sb, const void *x,
                                              const void *bias, void *y, int32_t in_f,
                                              int32_t out_f, int32_t accum) {
  WST_REQUIRE(qw && si && sb && x && y, "wst_gemv_int4_hier: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemv_int4_hier: non-positive dimension");
  WST_REQUIRE(in_f % HGROUP == 0,
              "wst_gemv_int4_hier: in_features must be a multiple of 32");

  switch (hier_tile_for(in_f, out_f)) {
    case 0: hier_launch<256, 1>(qw, si, sb, x, bias, y, in_f, out_f, accum); break;
    case 2: hier_launch<256, 4>(qw, si, sb, x, bias, y, in_f, out_f, accum); break;
    case 3: hier_launch<256, 8>(qw, si, sb, x, bias, y, in_f, out_f, accum); break;
    default: hier_launch<256, 2>(qw, si, sb, x, bias, y, in_f, out_f, accum); break;
  }
  WST_TRY_KERNEL("wst_gemv_int4_hier");
  return WST_OK;
}

extern "C" wst_status_t wst_embed_int4_hier(const void *qw, const void *si,
                                            const void *sb, const void *row,
                                            void *out, int32_t in_f, int32_t vocab) {
  WST_REQUIRE(qw && si && sb && row && out, "wst_embed_int4_hier: null pointer");
  WST_REQUIRE(in_f > 0 && vocab > 0, "wst_embed_int4_hier: non-positive dimension");
  WST_REQUIRE(in_f % HGROUP == 0, "wst_embed_int4_hier: in_features must be a multiple of 32");

  const int threads = 256;
  const int blocks = (in_f + threads - 1) / threads;
  embed_int4_hier_kernel<<<blocks, threads>>>(
      (const uint32_t *)qw, (const uint8_t *)si, (const half2 *)sb,
      (const int32_t *)row, (float *)out, in_f, vocab);
  WST_TRY_KERNEL("wst_embed_int4_hier");
  return WST_OK;
}
