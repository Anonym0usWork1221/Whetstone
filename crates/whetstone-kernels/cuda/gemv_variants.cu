/* A sweep of int4 decode-GEMV implementations, so the choice is measured.
 *
 * # Why this file exists
 *
 * The first int4 GEMV reached 81 GB/s at the MLP shape while the fp16 kernel on
 * the same schedule reached 202 GB/s. Reading 3.75x fewer bytes and going only
 * 1.5x faster means the kernel is not bandwidth bound, and a kernel that is not
 * bandwidth bound at batch=1 has something wrong with it. The register report
 * ruled out spilling (64 registers, zero spill, 100% occupancy), which leaves
 * the dequantization arithmetic.
 *
 * Counting it: the straightforward unpack is, per weight, a shift, a mask, an
 * `I2F`, a subtract, a multiply and an FMA. `I2F` is quarter rate on Turing --
 * it goes down the conversion pipe at 16/SM/cycle against FFMA's 64 -- so six
 * instructions per weight is really more like nine FFMA-equivalents. Against
 * half a byte of traffic per weight, that is an arithmetic intensity of ~18
 * ops/byte on a kernel whose whole premise is that arithmetic is free.
 *
 * # The fix, and why it is a bit trick
 *
 * fp16 1024.0 is `0x6400`, and at that exponent the mantissa ULP is exactly 1.
 * So for `q` in [0,15], `0x6400 | q` *is* the fp16 number `1024 + q` -- an
 * integer-to-float conversion done with an OR. Subtracting a precomputed
 * `1024 + zero` recovers `q - zero` exactly, and the group scale factors out of
 * the inner loop entirely because all 128 weights share it.
 *
 * That turns six ops per weight into about two and a half, all of them on the
 * full-rate fp16 pipe. (The trick is standard in fast int4 kernels -- AWQ,
 * Marlin and FasterTransformer all use a version of it -- but every one of those
 * is built on `cp.async`, which is sm_80+, so the surrounding pipeline here had
 * to be derived from scratch.)
 *
 * # What is swept
 *
 * Rows per warp (`TILE`) and block width, because they trade activation reuse
 * against parallelism and the right point depends on the matrix shape -- and
 * this model has shapes from 896x128 to 896x151936. Plus a deliberately wrong
 * variant that loads the weights and does no arithmetic at all, which measures
 * the memory path's floor and so says how much of the gap is left to close.
 */

#include "common.cuh"
#include <cuda_fp16.h>

#define VGROUP 128

/* fp16 1024.0. At this exponent the mantissa step is 1, so OR-ing a value in
 * [0,15] into the low bits adds it exactly. */
#define WST_MAGIC_H 0x6400u
#define WST_MAGIC_H2 0x64006400u
#define WST_MAGIC_F 1024.0f

/* Dequantization strategy. */
enum : int {
  MODE_F32 = 0,  /* shift, mask, I2F, affine -- the original */
  MODE_H2 = 1,   /* OR-into-mantissa, fp16 pipe */
  MODE_MEM = 2,  /* load the weights, skip the maths: the memory-path floor */
};

/* Unpack eight nibbles to eight floats via the conversion pipe. */
__device__ __forceinline__ void unpack8_f32(uint32_t packed, float out[8]) {
  const uint32_t lo = packed & 0x0F0F0F0Fu;
  const uint32_t hi = (packed >> 4) & 0x0F0F0F0Fu;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    out[2 * b + 0] = (float)((lo >> (8 * b)) & 0xFu);
    out[2 * b + 1] = (float)((hi >> (8 * b)) & 0xFu);
  }
}

/* Nibbles 2i and 2i+1 of `packed`, as a half2 holding (1024+q0, 1024+q1).
 *
 * Each case is two shifts, two masks and an OR against the magic constant,
 * which ptxas folds into a pair of LOP3s. */
__device__ __forceinline__ half2 pair_h2(uint32_t packed, int i) {
  uint32_t lo, hi;
  switch (i) {
    case 0:
      lo = packed & 0xFu;
      hi = (packed << 12) & 0x000F0000u;
      break;
    case 1:
      lo = (packed >> 8) & 0xFu;
      hi = (packed << 4) & 0x000F0000u;
      break;
    case 2:
      lo = (packed >> 16) & 0xFu;
      hi = (packed >> 4) & 0x000F0000u;
      break;
    default:
      lo = (packed >> 24) & 0xFu;
      hi = (packed >> 12) & 0x000F0000u;
      break;
  }
  const uint32_t bits = WST_MAGIC_H2 | lo | hi;
  return *(const half2 *)&bits;
}

/* One warp accumulates TILE output rows at a time; one block sweeps
 * `rows_per_block` of them.
 *
 * The activation vector is read through `__ldg` rather than staged in shared
 * memory. Coalescing the weight loads forces lane n to own columns
 * [32n, 32n+32), so its reads of x land 64 B apart -- two of thirty-two banks,
 * a 16-way conflict that measured *slower than fp16* despite moving 3.75x fewer
 * bytes. x is a couple of kilobytes and every block reads all of it, so it is
 * L1-resident after the first touch and the read-only path costs an L1 hit
 * instead. */
template <int THREADS, int TILE, int MODE>
__global__ __launch_bounds__(THREADS) void gemv_int4_var_kernel(
    const uint4 *__restrict__ qw, const half2 *__restrict__ sz,
    const half *__restrict__ x, const half *__restrict__ bias,
    float *__restrict__ y, int in_f, int out_f, int rows_per_block, int accum) {

  constexpr int WARPS = THREADS / WST_WARP;

  const int warp = threadIdx.x / WST_WARP;
  const int lane = threadIdx.x % WST_WARP;

  const int vec_per_row = in_f / 32;  /* uint4 = 4 words = 32 nibbles */
  const int groups_per_row = in_f / VGROUP;

  const int row0 = blockIdx.x * rows_per_block;
  const int row1 = min(row0 + rows_per_block, out_f);

  const uint4 *xv = (const uint4 *)x;

  for (int row = row0 + warp * TILE; row < row1; row += WARPS * TILE) {
    const int ntile = min(TILE, row1 - row);

    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; ++t) acc[t] = 0.0f;

    for (int v = lane; v < vec_per_row; v += WST_WARP) {
      const int col0 = v * 32;
      const int xbase = col0 / 8;
      const int grp = col0 / VGROUP;

      /* 32 activations = 64 B = four uint4. Consecutive lanes read consecutive
       * 64 B regions, so the warp still requests one contiguous span. */
      const uint4 xa = __ldg(&xv[xbase + 0]);
      const uint4 xb = __ldg(&xv[xbase + 1]);
      const uint4 xc = __ldg(&xv[xbase + 2]);
      const uint4 xd = __ldg(&xv[xbase + 3]);
      const uint32_t xw[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                               xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};

      float2 xf[16];
      if (MODE == MODE_F32) {
#pragma unroll
        for (int i = 0; i < 16; ++i) xf[i] = __half22float2(*(const half2 *)&xw[i]);
      }

#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        if (t >= ntile) break;

        const uint4 packed = qw[(size_t)(row + t) * vec_per_row + v];
        const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

        if (MODE == MODE_MEM) {
          /* Deliberately not a GEMV: consume the loaded words with the cheapest
           * dependency possible so nothing is optimised away. What this measures
           * is the floor the memory path alone imposes. */
          acc[t] += (float)(words[0] ^ words[1] ^ words[2] ^ words[3]);
          continue;
        }

        const half2 s = sz[(size_t)(row + t) * groups_per_row + grp];
        const float scale = __half2float(__low2half(s));
        const float zero = __half2float(__high2half(s));

        if (MODE == MODE_F32) {
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            float q[8];
            unpack8_f32(words[w], q);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const float2 xp = xf[w * 4 + i];
              acc[t] = fmaf((q[2 * i + 0] - zero) * scale, xp.x, acc[t]);
              acc[t] = fmaf((q[2 * i + 1] - zero) * scale, xp.y, acc[t]);
            }
          }
        } else {
          /* The scale is constant across all 128 weights of a group, so it
           * leaves the inner loop entirely and applies once at the end. What
           * remains per pair is one subtract and one FMA, both on the fp16
           * pipe, which Turing runs at twice fp32 rate. */
          const half zh = __float2half(WST_MAGIC_F + zero);
          const half2 magic = __halves2half2(zh, zh);

          half2 dot = __floats2half2_rn(0.0f, 0.0f);
#pragma unroll
          for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const half2 q2 = __hsub2(pair_h2(words[w], i), magic);
              dot = __hfma2(q2, *(const half2 *)&xw[w * 4 + i], dot);
            }
          }
          acc[t] = fmaf(scale, __half2float(__low2half(dot)) + __half2float(__high2half(dot)),
                        acc[t]);
        }
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

/* ---------------------------------------------------------------- dispatch */

/* Rows per block such that the grid still fills the GPU while activation reuse
 * stays amortised. Targets ~4 blocks per SM.
 *
 * The SM count is cached: `cudaGetDeviceProperties` is a surprisingly expensive
 * host call and these kernels run in tens of microseconds. Querying it per
 * launch once measured 2-4 ms, entirely swamping the kernel it was sizing. */
static int sm_count() {
  static int sms = 0;
  if (sms == 0) {
    cudaDeviceProp p;
    sms = (cudaGetDeviceProperties(&p, 0) == cudaSuccess) ? p.multiProcessorCount : 30;
  }
  return sms;
}

template <int THREADS, int TILE>
static int rows_for(int out_f) {
  const int quant = (THREADS / WST_WARP) * TILE;
  int rows = (out_f + sm_count() * 4 - 1) / (sm_count() * 4);
  rows = ((rows + quant - 1) / quant) * quant;
  if (rows < quant) rows = quant;
  if (rows > 512) rows = 512;
  return rows;
}

struct VariantDesc {
  const char *name;
  int threads;
  int tile;
  int mode;
  wst_status_t (*launch)(const void *, const void *, const void *, const void *, void *, int,
                         int, int);
};

template <int THREADS, int TILE, int MODE>
static wst_status_t launch_variant(const void *qw, const void *sz, const void *x,
                                   const void *bias, void *y, int in_f, int out_f,
                                   int accum) {
  const int rows = rows_for<THREADS, TILE>(out_f);
  const int blocks = (out_f + rows - 1) / rows;
  gemv_int4_var_kernel<THREADS, TILE, MODE><<<blocks, THREADS>>>(
      (const uint4 *)qw, (const half2 *)sz, (const half *)x, (const half *)bias, (float *)y,
      in_f, out_f, rows, accum);
  return WST_OK;
}

/* Keep this table small. Every entry is a separate kernel instantiation, and
 * build time is a real cost when the loop is edit-measure-edit. */
static const VariantDesc kVariants[] = {
    {"f32 t4 x256", 256, 4, MODE_F32, launch_variant<256, 4, MODE_F32>},
    {"f32 t1 x256", 256, 1, MODE_F32, launch_variant<256, 1, MODE_F32>},
    {"h2  t1 x256", 256, 1, MODE_H2, launch_variant<256, 1, MODE_H2>},
    {"h2  t2 x256", 256, 2, MODE_H2, launch_variant<256, 2, MODE_H2>},
    {"h2  t4 x256", 256, 4, MODE_H2, launch_variant<256, 4, MODE_H2>},
    {"h2  t1 x128", 128, 1, MODE_H2, launch_variant<128, 1, MODE_H2>},
    {"h2  t2 x128", 128, 2, MODE_H2, launch_variant<128, 2, MODE_H2>},
    {"h2  t1 x512", 512, 1, MODE_H2, launch_variant<512, 1, MODE_H2>},
    {"h2  t2 x512", 512, 2, MODE_H2, launch_variant<512, 2, MODE_H2>},
    {"h2  t8 x256", 256, 8, MODE_H2, launch_variant<256, 8, MODE_H2>},
    {"mem t2 x256", 256, 2, MODE_MEM, launch_variant<256, 2, MODE_MEM>},
};

static const int kVariantCount = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

/* The production choice, measured on an RTX 2060 (sm_75) over every shape
 * Qwen2.5-0.5B issues, weighted by how often each runs in a decode step:
 *
 *   variant        q/o      k/v    gate/up     down    lm_head    per token
 *   f32 t4 x256   58 GB/s   9      114        100      190        2.586 ms
 *   h2  t2 x256   92        19     174        187      216        1.643 ms  <-
 *   h2  t4 x256   85        13     168        169      254        1.737 ms
 *   h2  t1 x256   73        20     111        144      136        2.349 ms
 *   mem t2 x256  133        23     364        431      293        0.961 ms  (floor)
 *
 * 1.57x over the original, from replacing eight quarter-rate `I2F` per word with
 * an OR into an fp16 mantissa.
 *
 * TILE=2 winning is the part worth explaining, because it contradicts the
 * reasoning that produced TILE=4. More rows per warp reuses the activation load
 * across more output rows -- but x is a couple of kilobytes and L1-resident, so
 * that reuse was never the constraint. What tiling actually costs is
 * parallelism: at TILE=8 a 128-row matrix has too few warp-tasks to fill 30 SMs,
 * and even the 4864-row ones lose more to the tail than they gain. The
 * activation-reuse argument was right about the mechanism and wrong about which
 * side of the trade dominates.
 *
 * The `mem` row loads the weights and skips the arithmetic. It is L2-optimistic
 * for anything under ~3 MB (only lm_head is genuinely DRAM-bound at this size),
 * so read it as a bound on the *ranking*, not as an attainable target. */
static const int kDefaultVariant = 3; /* h2 t2 x256 */

extern "C" int32_t wst_gemv_default_variant(void) { return kDefaultVariant; }

/* Per-shape kernel selection, measured **in situ**.
 *
 * No single blocking wins everywhere, and the reason is memory-level
 * parallelism rather than anything about the arithmetic. Sustaining ~250 GB/s at
 * Turing's ~400 ns DRAM latency needs on the order of 100 KB of loads in flight
 * (Little's law). Each warp holds `TILE` 16-byte loads per reduction step, and
 * the number of warps a shape can create is `out_features / TILE` -- so a shape
 * with few rows cannot buy in-flight bytes by adding warps and has to buy them
 * with TILE instead, while a shape with many rows would rather have the warps.
 *
 * The numbers below come from running the *engine* with each variant forced and
 * reading the per-stage CUDA-event breakdown, in ms/token:
 *
 *   shape                    t2      t4      t8
 *   896x9728  (gate|up)     0.827   0.548   0.631
 *   4864x896  (down)        0.423   0.439   0.687
 *   896x151936 (lm_head)    0.369   0.362   0.257
 *   896x1152  (q|k|v)       0.210   0.194   0.242
 *   896x896   (o)           0.200   0.190   0.301
 *
 * **The microbenchmark disagrees with this table on both of the big shapes.**
 * Isolated, `gate/up` looks best at t8 and `lm_head` at t4; in the engine it is
 * exactly the other way round. The cause is cache: a microbenchmark reruns one
 * matrix 200 times, so anything under this card's 3 MB L2 stays resident and
 * reads far faster than it ever does in a decode step, where 262 MB sweeps past
 * once. Only `lm_head` at 68 MB is genuinely DRAM-bound in isolation.
 *
 * **And the in-situ profile misranks them too.** The rule its column selects
 * measured *slower* end to end than the one it replaced. Sweeping all 27
 * assignments by whole-generation throughput (`whetstone tune`) settles it, and
 * the answer is deflating: the entire space spans **472-486 tok/s, a 2.9%
 * spread**. Differences that look like 92 GB/s against 57 GB/s in isolation are
 * worth almost nothing once the kernels are pipelined back to back against a
 * cache that never holds their weights.
 *
 * The values below are the sweep's winner. Treat the margin as noise: what this
 * rule buys is a couple of percent, not the 1.6x the microbenchmark implied.
 *
 * Reading the table with the parallelism model: `lm_head` has 151936 rows, so
 * even at t8 it fills the machine several waves deep and takes the extra
 * in-flight bytes for free. `gate|up` has 9728 rows -- at t8 that is 1216 warps
 * against ~960 resident, so the second wave is a quarter full and the tail costs
 * more than the extra loads win. `down` has 896 rows but a 4864-wide reduction,
 * which supplies its own instruction-level parallelism through the loop, so it
 * wants TILE low.
 */
/* The three buckets, overridable so the rule can be swept end-to-end rather
 * than inferred. Even the in-situ *profiler* misranks these -- its events
 * serialise stage boundaries -- so the only ranking that settles the question is
 * whole-generation tokens per second. See `whetstone tune`. */
static int kRule[3] = {3, 3, 9}; /* wide reduction, huge output, everything else */

static int variant_for_shape(int in_f, int out_f) {
  if (in_f >= 2048) return kRule[0];    /* down_proj: the reduction supplies ILP */
  if (out_f >= 65536) return kRule[1];  /* lm_head: rows enough for many waves */
  return kRule[2];                      /* gate|up, q|k|v, o */
}

extern "C" void wst_gemv_set_shape_rule(int32_t wide, int32_t huge, int32_t other) {
  if (wide >= 0 && wide < kVariantCount) kRule[0] = wide;
  if (huge >= 0 && huge < kVariantCount) kRule[1] = huge;
  if (other >= 0 && other < kVariantCount) kRule[2] = other;
}

extern "C" void wst_gemv_get_shape_rule(int32_t *out) {
  out[0] = kRule[0];
  out[1] = kRule[1];
  out[2] = kRule[2];
}

extern "C" int32_t wst_gemv_variant_for_shape(int32_t in_f, int32_t out_f) {
  return variant_for_shape(in_f, out_f);
}

extern "C" int32_t wst_gemv_variant_count(void) { return kVariantCount; }

extern "C" const char *wst_gemv_variant_name(int32_t v) {
  if (v < 0 || v >= kVariantCount) return "?";
  return kVariants[v].name;
}

extern "C" wst_status_t wst_gemv_int4_variant(int32_t variant, const void *qw, const void *sz,
                                              const void *x, const void *bias, void *y,
                                              int32_t in_f, int32_t out_f, int32_t accum) {
  WST_REQUIRE(variant >= 0 && variant < kVariantCount, "wst_gemv_int4_variant: bad variant");
  WST_REQUIRE(qw && sz && x && y, "wst_gemv_int4_variant: null pointer");
  WST_REQUIRE(in_f > 0 && out_f > 0, "wst_gemv_int4_variant: non-positive dimension");
  WST_REQUIRE(in_f % VGROUP == 0,
              "wst_gemv_int4_variant: in_features must be a multiple of 128");

  kVariants[variant].launch(qw, sz, x, bias, y, in_f, out_f, accum);
  WST_TRY_KERNEL("wst_gemv_int4_variant");
  return WST_OK;
}

/* Times one variant at one shape.
 *
 * Reports achieved bandwidth, which for a batch=1 GEMV is the only figure of
 * merit -- the kernel is good exactly insofar as it saturates the memory
 * system. The weights are filled with a fixed pattern rather than left
 * uninitialised so that denormals or NaNs cannot change the timing. */
extern "C" wst_status_t wst_bench_gemv_variant(int32_t variant, int32_t in_f, int32_t out_f,
                                               int32_t reps, double *out_gbs, double *out_ms) {
  WST_REQUIRE(out_gbs && out_ms, "wst_bench_gemv_variant: null out pointer");
  WST_REQUIRE(reps > 0, "wst_bench_gemv_variant: reps must be positive");
  WST_REQUIRE(variant >= 0 && variant < kVariantCount, "wst_bench_gemv_variant: bad variant");
  WST_REQUIRE(in_f > 0 && out_f > 0 && in_f % VGROUP == 0,
              "wst_bench_gemv_variant: bad shape");

  const size_t n = (size_t)in_f * out_f;
  const size_t w_bytes = n / 2;
  const size_t sz_bytes = (size_t)out_f * (in_f / VGROUP) * sizeof(half2);

  void *w = nullptr, *szb = nullptr, *x = nullptr, *y = nullptr;
  wst_status_t st;
  if ((st = wst_malloc(&w, w_bytes)) != WST_OK) return st;
  if ((st = wst_malloc(&szb, sz_bytes)) != WST_OK) { wst_free(w); return st; }
  if ((st = wst_malloc(&x, (size_t)in_f * sizeof(half))) != WST_OK) {
    wst_free(w); wst_free(szb); return st;
  }
  if ((st = wst_malloc(&y, (size_t)out_f * sizeof(float))) != WST_OK) {
    wst_free(w); wst_free(szb); wst_free(x); return st;
  }

  cudaMemset(w, 0x11, w_bytes);
  cudaMemset(szb, 0x3C, sz_bytes); /* 0x3C00 == 1.0h in both halves */
  cudaMemset(x, 0x3C, (size_t)in_f * sizeof(half));

  auto go = [&]() {
    return wst_gemv_int4_variant(variant, w, szb, x, nullptr, y, in_f, out_f, 0);
  };

  st = go();
  if (st != WST_OK) { wst_free(w); wst_free(szb); wst_free(x); wst_free(y); return st; }
  cudaDeviceSynchronize();

  WstTimer t;
  if (!t.ok) {
    wst_free(w); wst_free(szb); wst_free(x); wst_free(y);
    wst_set_error_msg("wst_bench_gemv_variant: event creation failed");
    return WST_ERR_CUDA;
  }

  t.tic();
  for (int i = 0; i < reps; ++i) go();
  const float ms = t.toc_ms();

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    wst_free(w); wst_free(szb); wst_free(x); wst_free(y);
    wst_set_error("wst_bench_gemv_variant", e);
    return WST_ERR_CUDA;
  }

  /* Scales are counted: they are real traffic, and quoting "4 bits" while
   * ignoring them understates bandwidth by 6%. */
  *out_ms = ms / reps;
  *out_gbs = (double)(w_bytes + sz_bytes) * reps / (ms * 1.0e-3) / 1.0e9;

  wst_free(w); wst_free(szb); wst_free(x); wst_free(y);
  return WST_OK;
}
