/* Batch=1 decode GEMV against int4 group-quantized weights.
 *
 * This is the kernel that decides Whetstone's decode speed, so it is written
 * against one constraint above all others: MOVE THE FEWEST BYTES, PERFECTLY
 * COALESCED. At batch=1 the arithmetic is free -- roughly 2 FLOP per byte
 * against a ~120 FLOP/byte machine balance -- so the only thing that matters is
 * that every weight byte arrives at full bandwidth and is touched exactly once.
 *
 * Consequences of that constraint, all of which are visible below:
 *   - weights are read as uint4 (128-bit) loads, the widest the ISA offers
 *   - dequantization happens in registers; the dequantized weight is never
 *     written anywhere
 *   - each block sweeps many rows, so per-block setup amortises
 *   - each warp accumulates WST_TILE rows at once, which reuses the activation
 *     load and gives the FMA pipeline independent chains to overlap
 *   - reductions are warp shuffles and never touch memory
 *   - the activation vector is read through the read-only cache rather than
 *     staged in shared memory -- see the comment in the kernel for why the
 *     obvious shared-memory version is 2x slower here
 *
 * Layout (see whetstone-quant for the writer):
 *   qw : [out_features][in_features/8] uint32, 8 nibbles each, lane i in bits 4i..4i+3
 *   sz : [out_features][in_features/GROUP] half2, .x = scale, .y = zero
 *   x  : [in_features] half
 *   y  : [out_features] float
 *
 * Dequantization is the standard asymmetric affine map:
 *     w = (q - zero) * scale,  q in [0, 15]
 */

#include "common.cuh"
#include <cuda_fp16.h>

#define WST_GROUP 128    /* weights per scale/zero pair */
#define WST_THREADS 256  /* 8 warps per block */
#define WST_WARPS (WST_THREADS / WST_WARP)

/* Rows each block owns, chosen so that staging x into shared memory amortises.
 *
 * This number is not cosmetic. A block reads the whole activation vector into
 * shared memory once, then reuses it for every row it owns. With too few rows
 * per block, x is re-read from global memory by every block and its traffic
 * rivals the weight traffic itself -- measured at 4 rows/block, x accounted for
 * ~2.2 MB against 2.2 MB of int4 weights, i.e. half the bandwidth was spent on
 * an 1.8 KB vector. Amortising over many rows drops that to noise.
 *
 * The host picks the value; these bound it so the grid still fills the GPU. */
/* Output rows each warp accumulates simultaneously. */
#define WST_TILE 4

#define WST_MIN_ROWS_PER_BLOCK (WST_WARPS * WST_TILE)
#define WST_MAX_ROWS_PER_BLOCK 512

/* Unpack eight nibbles into eight floats.
 *
 * The naive form is eight shift-and-mask pairs. Instead we mask the even and
 * odd nibbles with two ops, which the compiler lowers to LOP3 -- Turing's
 * three-input logic instruction -- and then shift. It matters less than the
 * memory traffic, but the unpack sits directly in the dependency chain of every
 * loaded byte, so keeping it short keeps the loads issuing.
 */
__device__ __forceinline__ void unpack8_nibbles(uint32_t packed, float out[8]) {
  const uint32_t lo = packed & 0x0F0F0F0Fu;         /* nibbles 0,2,4,6 */
  const uint32_t hi = (packed >> 4) & 0x0F0F0F0Fu;  /* nibbles 1,3,5,7 */

#pragma unroll
  for (int b = 0; b < 4; ++b) {
    out[2 * b + 0] = (float)((lo >> (8 * b)) & 0xFu);
    out[2 * b + 1] = (float)((hi >> (8 * b)) & 0xFu);
  }
}

/* One warp per output row; each block sweeps many rows so that the shared-memory
 * copy of x is paid for once and reused. Each lane walks a row in 128-bit
 * strides, so a warp issues one fully coalesced 512-byte request per step. */
__global__ __launch_bounds__(WST_THREADS) void gemv_int4_g128_kernel(
    const uint4 *__restrict__ qw,
    const half2 *__restrict__ sz,
    const half *__restrict__ x,
    float *__restrict__ y,
    int in_f, int out_f, int rows_per_block) {

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  /* uint4 holds 4 uint32 = 32 nibbles = 32 weights. */
  const int vec_per_row = in_f / 32;
  const int groups_per_row = in_f / WST_GROUP;

  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);

  /* x is read straight from global rather than staged in shared memory.
   *
   * Shared staging looks obviously right here and is a trap. Coalescing the
   * weight loads forces lane n to own columns [32n, 32n+32), so its reads of x
   * land 64 B apart. Shared memory has 32 four-byte banks, so those addresses
   * collapse onto 2 banks -- a 16-way conflict that serialises every access.
   * Measured, that put int4 at 39 GB/s, slower in wall-clock than fp16 despite
   * moving 3.75x fewer bytes.
   *
   * x is at most a few KB and every block reads all of it, so it is L1-resident
   * after the first touch. Going through the read-only path costs an L1 hit
   * instead of a conflicted shared access, and drops __syncthreads and the
   * shared-memory ceiling on in_features as well. */
  const uint4 *xv = (const uint4 *)x;

  /* Each warp works on WST_TILE rows at once.
   *
   * Row tiling is what makes this kernel bandwidth-bound rather than
   * latency-bound. With one row per warp and in_features=896 the whole row is a
   * single loop iteration -- one 128-bit load, then a 5-step shuffle reduction
   * whose latency nothing hides. Tiling gives:
   *   - one x load feeding WST_TILE rows instead of one,
   *   - WST_TILE independent FMA chains, so the FMA latency overlaps,
   *   - the same reduction cost spread over WST_TILE outputs.
   * Weight traffic is unchanged; only the overhead per useful byte falls. */
  for (int row = row0 + warp * WST_TILE; row < row1; row += WST_WARPS * WST_TILE) {
    const int ntile = min(WST_TILE, row1 - row);

    float acc[WST_TILE];
#pragma unroll
    for (int t = 0; t < WST_TILE; ++t) acc[t] = 0.0f;

    for (int v = lane; v < vec_per_row; v += WST_WARP) {
      /* 32 activations = 64 B = 4 uint4. Consecutive lanes read consecutive
       * 64 B regions, so the warp still requests one contiguous 2 KB span. */
      const int col0 = v * 32;
      const int xbase = col0 / 8;
      const uint4 xa = __ldg(&xv[xbase + 0]);
      const uint4 xb = __ldg(&xv[xbase + 1]);
      const uint4 xc = __ldg(&xv[xbase + 2]);
      const uint4 xd = __ldg(&xv[xbase + 3]);

      const uint32_t xw[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                               xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};

      float2 xf[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) xf[i] = __half22float2(*(const half2 *)&xw[i]);

#pragma unroll
      for (int t = 0; t < WST_TILE; ++t) {
        if (t >= ntile) break;

        const uint4 packed = qw[(size_t)(row + t) * vec_per_row + v];
        /* GROUP is a multiple of 32, so all 32 weights share one scale/zero. */
        const half2 s = sz[(size_t)(row + t) * groups_per_row + col0 / WST_GROUP];
        const float scale = __half2float(__low2half(s));
        const float zero = __half2float(__high2half(s));

        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

#pragma unroll
        for (int w = 0; w < 4; ++w) {
          float q[8];
          unpack8_nibbles(words[w], q);

#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float2 xp = xf[w * 4 + i];
            /* (q - zero) * scale * x, refactored so the compiler emits FFMAs
             * and never materialises the dequantized weight in memory. */
            acc[t] = fmaf((q[2 * i + 0] - zero) * scale, xp.x, acc[t]);
            acc[t] = fmaf((q[2 * i + 1] - zero) * scale, xp.y, acc[t]);
          }
        }
      }
    }

#pragma unroll
    for (int t = 0; t < WST_TILE; ++t) {
      if (t >= ntile) break;
      const float r = warp_reduce_sum(acc[t]);
      if (lane == 0) y[row + t] = r;
    }
  }
}

/* fp16 GEMV, same blocking. Separates "the kernel is wrong" from "the
 * quantization is lossy" when a result looks off, and gives the honest
 * same-schedule baseline the int4 path must beat.
 *
 * Loads go through half2 so a warp moves 256 B per request instead of 64 B;
 * scalar half loads waste three quarters of every transaction. */
__global__ __launch_bounds__(WST_THREADS) void gemv_fp16_kernel(
    const half *__restrict__ w, const half *__restrict__ x,
    float *__restrict__ y, int in_f, int out_f, int rows_per_block) {

  extern __shared__ half sx[];
  for (int i = threadIdx.x; i < in_f; i += blockDim.x) sx[i] = x[i];
  __syncthreads();

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);
  const int half2_per_row = in_f / 2;

  const half2 *sx2 = (const half2 *)sx;

  for (int row = row0 + warp; row < row1; row += WST_WARPS) {
    const half2 *row_w = (const half2 *)(w + (size_t)row * in_f);

    float acc = 0.0f;
    for (int c = lane; c < half2_per_row; c += WST_WARP) {
      const float2 wv = __half22float2(row_w[c]);
      const float2 xv = __half22float2(sx2[c]);
      acc = fmaf(wv.x, xv.x, acc);
      acc = fmaf(wv.y, xv.y, acc);
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
  }
}

/* Rows per block such that the grid still fills the GPU while x staging stays
 * amortised. Targets ~4 blocks per SM: enough for latency hiding and tail
 * tolerance, few enough that each block does substantial work. */
static int pick_rows_per_block(int out_f) {
  /* Cached: cudaGetDeviceProperties is a surprisingly expensive host call, and
   * a GEMV at these sizes runs in tens of microseconds. Querying it per launch
   * measured 2-4 ms per call -- entirely swamping the kernel it was sizing. */
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceProp p;
    sms = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.multiProcessorCount : 30;
  }

  const int target_blocks = sms * 4;
  int rows = (out_f + target_blocks - 1) / target_blocks;

  /* Round up to a whole number of warp-sized row groups. */
  const int quant = WST_WARPS * WST_TILE;
  rows = ((rows + quant - 1) / quant) * quant;

  if (rows < WST_MIN_ROWS_PER_BLOCK) rows = WST_MIN_ROWS_PER_BLOCK;
  if (rows > WST_MAX_ROWS_PER_BLOCK) rows = WST_MAX_ROWS_PER_BLOCK;
  return rows;
}

/* ------------------------------------------------------------------ ABI */

extern "C" wst_status_t wst_gemv_int4_g128(const void *qw, const void *sz,
                                           const void *x, void *y,
                                           int32_t in_f, int32_t out_f) {
  WST_REQUIRE(qw && sz && x && y, "wst_gemv_int4_g128: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemv_int4_g128: non-positive dimension");
  WST_REQUIRE(in_f % WST_GROUP == 0,
              "wst_gemv_int4_g128: in_features must be a multiple of 128");

  const int rows = pick_rows_per_block(out_f);
  const int blocks = (out_f + rows - 1) / rows;
  gemv_int4_g128_kernel<<<blocks, WST_THREADS>>>(
      (const uint4 *)qw, (const half2 *)sz, (const half *)x, (float *)y,
      in_f, out_f, rows);

  WST_TRY_KERNEL("wst_gemv_int4_g128");
  return WST_OK;
}

extern "C" wst_status_t wst_gemv_fp16(const void *w, const void *x, void *y,
                                      int32_t in_f, int32_t out_f) {
  WST_REQUIRE(w && x && y, "wst_gemv_fp16: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemv_fp16: non-positive dimension");
  WST_REQUIRE(in_f % 2 == 0, "wst_gemv_fp16: in_features must be even for half2 loads");

  const size_t smem = (size_t)in_f * sizeof(half);
  WST_REQUIRE(smem <= 48u * 1024u, "wst_gemv_fp16: in_features too large for shared memory");

  const int rows = pick_rows_per_block(out_f);
  const int blocks = (out_f + rows - 1) / rows;
  gemv_fp16_kernel<<<blocks, WST_THREADS, smem>>>(
      (const half *)w, (const half *)x, (float *)y, in_f, out_f, rows);

  WST_TRY_KERNEL("wst_gemv_fp16");
  return WST_OK;
}

/* Times a GEMV and reports achieved bandwidth, which is the only figure of
 * merit that matters here: the kernel is good exactly insofar as it saturates
 * the memory system. */
extern "C" wst_status_t wst_bench_gemv(int32_t in_f, int32_t out_f, int32_t reps,
                                       int32_t use_int4, double *out_gbs,
                                       double *out_ms) {
  WST_REQUIRE(out_gbs && out_ms, "wst_bench_gemv: null out pointer");
  WST_REQUIRE(reps > 0, "wst_bench_gemv: reps must be positive");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_bench_gemv: non-positive dimension");
  WST_REQUIRE(in_f % WST_GROUP == 0, "wst_bench_gemv: in_features must be a multiple of 128");

  const size_t n = (size_t)in_f * out_f;
  const size_t w_bytes = use_int4 ? n / 2 : n * sizeof(half);
  const size_t sz_bytes = (size_t)out_f * (in_f / WST_GROUP) * sizeof(half2);

  void *w = nullptr, *sz = nullptr, *x = nullptr, *y = nullptr;
  wst_status_t st;
  if ((st = wst_malloc(&w, w_bytes)) != WST_OK) return st;
  if ((st = wst_malloc(&sz, sz_bytes)) != WST_OK) { wst_free(w); return st; }
  if ((st = wst_malloc(&x, (size_t)in_f * sizeof(half))) != WST_OK) {
    wst_free(w); wst_free(sz); return st;
  }
  if ((st = wst_malloc(&y, (size_t)out_f * sizeof(float))) != WST_OK) {
    wst_free(w); wst_free(sz); wst_free(x); return st;
  }

  cudaMemset(w, 0x11, w_bytes);
  cudaMemset(sz, 0x3C, sz_bytes);   /* 0x3C00 == 1.0h */
  cudaMemset(x, 0x3C, (size_t)in_f * sizeof(half));

  auto launch = [&]() {
    return use_int4 ? wst_gemv_int4_g128(w, sz, x, y, in_f, out_f)
                    : wst_gemv_fp16(w, x, y, in_f, out_f);
  };

  st = launch();
  if (st != WST_OK) { wst_free(w); wst_free(sz); wst_free(x); wst_free(y); return st; }
  cudaDeviceSynchronize();

  WstTimer t;
  if (!t.ok) {
    wst_free(w); wst_free(sz); wst_free(x); wst_free(y);
    wst_set_error_msg("wst_bench_gemv: event creation failed");
    return WST_ERR_CUDA;
  }

  t.tic();
  for (int i = 0; i < reps; ++i) launch();
  const float ms = t.toc_ms();

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    wst_free(w); wst_free(sz); wst_free(x); wst_free(y);
    wst_set_error("wst_bench_gemv", e);
    return WST_ERR_CUDA;
  }

  /* Weights dominate; scales are counted because they are real traffic. */
  const double bytes = (double)(w_bytes + (use_int4 ? sz_bytes : 0)) * reps;
  *out_ms = ms / reps;
  *out_gbs = bytes / (ms * 1.0e-3) / 1.0e9;

  wst_free(w); wst_free(sz); wst_free(x); wst_free(y);
  return WST_OK;
}
