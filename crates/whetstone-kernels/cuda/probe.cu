/* Measures the relative cost of every arithmetic path this GPU offers.
 *
 * Whetstone's design hinges on picking the right primitive, and the spec sheet
 * is not a reliable guide, so we measure. Read the numbers with two caveats:
 *
 *  1. These are DEPENDENT accumulate chains -- each iteration consumes the
 *     previous result. With only 8 warps per SM resident, the measurement sits
 *     between latency and issue rate. Treat it as an ordering of the paths, not
 *     as an attainable GEMM throughput.
 *
 *  2. The fp16 baseline accumulates in fp32. On consumer Turing that runs at
 *     HALF the rate of fp16 accumulation, so every ratio quoted against it is
 *     roughly 2x flattering to the alternative. fp32 accumulation is what a
 *     numerically sound GEMM actually needs, so it is the honest baseline for
 *     our purposes -- but the choice must be stated, not hidden.
 *
 * See kMacs* below for a units trap that previously made dp4a and popc look
 * 32x worse than they are.
 */

#include "common.cuh"
#include <mma.h>
#include <cuda_fp16.h>

/* `mma.h` only declares `nvcuda` when the pass can actually use tensor cores --
 * its own guard is exactly this condition. Importing it unconditionally breaks
 * the sm_60 and sm_61 passes of the fat binary with "name must be a namespace
 * name", which is a confusing way to be told the architecture has no wmma. */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
using namespace nvcuda;
#endif

/* ------------------------------------------------------------ fp16 baseline */

__global__ void probe_hmma(const half *__restrict__ A, const half *__restrict__ B,
                           float *__restrict__ C, int iters) {
#if WST_DEV_HAS_WMMA
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::fill_fragment(c, 0.0f);
  wmma::load_matrix_sync(a, A, 16);
  wmma::load_matrix_sync(b, B, 16);
  for (int i = 0; i < iters; ++i) wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
#else
  (void)A; (void)B; (void)C; (void)iters;
#endif
}

/* ------------------------------------------------------------------- int8 */

/* The symbol exists on every architecture; the body does not. A kernel that is
 * `#if`-ed away entirely cannot be launched from host code that is compiled
 * once for all of them. */
__global__ void probe_imma8(const int8_t *__restrict__ A, const int8_t *__restrict__ B,
                            int32_t *__restrict__ C, int iters) {
#if WST_DEV_HAS_IMMA
  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c;
  wmma::fill_fragment(c, 0);
  wmma::load_matrix_sync(a, A, 16);
  wmma::load_matrix_sync(b, B, 16);
  for (int i = 0; i < iters; ++i) wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
#else
  (void)A; (void)B; (void)C; (void)iters;
#endif
}

/* ------------------------------------------------------------------- int4 */

__global__ void probe_imma4(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B,
                            int32_t *__restrict__ C, int iters) {
#if WST_DEV_HAS_BMMA_XOR
  using s4 = wmma::experimental::precision::s4;
  wmma::fragment<wmma::matrix_a, 8, 8, 32, s4, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 8, 8, 32, s4, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 8, 8, 32, int32_t> c;
  wmma::fill_fragment(c, 0);
  wmma::load_matrix_sync(a, (const s4 *)A, 32);
  wmma::load_matrix_sync(b, (const s4 *)B, 32);
  for (int i = 0; i < iters; ++i) wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(C, c, 8, wmma::mem_row_major);
#else
  (void)A; (void)B; (void)C; (void)iters;
#endif
}

/* --------------------------------------------------------- binary (1 bit) */

/* bmma.sync.m8n8k128.row.col.s32.b1.b1.s32.xor.popc
 *
 * 128 MACs per lane per issue. On Turing this is the single fastest arithmetic
 * op available, by a wide margin. Note .xor.popc is sm_75; .and.popc needs
 * sm_80, which is why Whetstone's binary encoding is built around XOR. */
__global__ void probe_bmma(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B,
                           int32_t *__restrict__ C, int iters) {
#if WST_DEV_HAS_BMMA_XOR
  using b1 = wmma::experimental::precision::b1;
  wmma::fragment<wmma::matrix_a, 8, 8, 128, b1, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 8, 8, 128, b1, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 8, 8, 128, int32_t> c;
  wmma::fill_fragment(c, 0);
  wmma::load_matrix_sync(a, A, 128);
  wmma::load_matrix_sync(b, B, 128);
  for (int i = 0; i < iters; ++i)
    wmma::bmma_sync(c, a, b, c, wmma::experimental::bmmaBitOpXOR,
                    wmma::experimental::bmmaAccumulateOpPOPC);
  wmma::store_matrix_sync(C, c, 8, wmma::mem_row_major);
#else
  (void)A; (void)B; (void)C; (void)iters;
#endif
}

/* ----------------------------------------------------- CUDA-core fallbacks */

/* `__dp4a` is sm_61+. P100 (sm_60) is the one architecture in the fat binary
 * without it. */
__global__ void probe_dp4a(const int32_t *__restrict__ A, const int32_t *__restrict__ B,
                           int32_t *__restrict__ C, int iters) {
#if WST_DEV_HAS_DP4A
  int acc = 0;
  int a = A[threadIdx.x & 31], b = B[threadIdx.x & 31];
  for (int i = 0; i < iters; ++i) acc = __dp4a(a, b, acc);
  if (threadIdx.x == 0) C[blockIdx.x] = acc;
#else
  (void)A; (void)B; (void)C; (void)iters;
#endif
}

__global__ void probe_popc(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B,
                           int32_t *__restrict__ C, int iters) {
  int acc = 0;
  uint32_t a = A[threadIdx.x & 31], b = B[threadIdx.x & 31];
  for (int i = 0; i < iters; ++i) acc += __popc(a ^ b);
  if (threadIdx.x == 0) C[blockIdx.x] = acc;
}

/* ------------------------------------------- what the loaded IMAGE contains */

/* The device's compute capability and the *image* the driver loaded for it are
 * two different facts, and the fat binary made them diverge.
 *
 * `WST_HOST_HAS_*` asks the driver what the installed card can do.
 * `WST_DEV_HAS_*` was resolved at compile time, once per architecture, and
 * decides whether a kernel body exists at all. They agree only when the archive
 * happens to hold an exact image for the running device.
 *
 * When it does not -- an sm_72 card running the sm_70 image, or any card
 * running a `WHETSTONE_CUDA_ARCH=70` build -- the host sees "this device has
 * IMMA", launches a kernel whose body was `#if`-ed away to nothing, gets a
 * clean `cudaGetLastError()`, and `measure_tops` divides real work-units by
 * pure launch overhead. The result is a **fabricated throughput number**, which
 * is precisely the failure mode this project exists to not have.
 *
 * So the image reports itself. A capability is usable only when the device
 * supports it AND the loaded image contains it. */
__global__ void probe_image_caps(int32_t *__restrict__ out) {
  if (threadIdx.x != 0) return;
  out[0] = WST_DEV_HAS_WMMA ? 1 : 0;
  out[1] = WST_DEV_HAS_IMMA ? 1 : 0;
  out[2] = WST_DEV_HAS_BMMA_XOR ? 1 : 0;
  out[3] = WST_DEV_HAS_DP4A ? 1 : 0;
}

/* --------------------------------------------------- XNOR identity check */

/* Confirms dot = K - 2*popcount(a^b) against a scalar reference computed the
 * long way, so the identity underpinning the binary path is validated on the
 * actual silicon rather than assumed. */
__global__ void probe_xnor_identity(const uint32_t *A, const uint32_t *B, int K,
                                    int32_t *out) {
  int popc = 0;
  for (int i = 0; i < K / 32; ++i) popc += __popc(A[i] ^ B[i]);
  const int via_identity = xnor_dot_from_popc(popc, K);

  /* Reference: expand bits to +-1 and accumulate. */
  int reference = 0;
  for (int i = 0; i < K; ++i) {
    const int abit = (A[i >> 5] >> (i & 31)) & 1;
    const int bbit = (B[i >> 5] >> (i & 31)) & 1;
    reference += (abit ? 1 : -1) * (bbit ? 1 : -1);
  }
  *out = (via_identity == reference) ? 1 : 0;
}

/* ------------------------------------------------------------------ driver */

namespace {

struct ProbeCtx {
  int blocks, threads, iters;
  double warp_issues;   /* total warp-level op issues across the grid */
};

/* MACs performed per warp-issue.
 *
 * The distinction below is easy to get wrong and silently costs a factor of 32:
 *
 *   - wmma/bmma are WARP-WIDE instructions. One issue computes the entire
 *     M x N x K tile cooperatively, so the tile size IS the per-issue MAC count.
 *
 *   - dp4a and popc are PER-LANE instructions. When a warp issues one, all 32
 *     lanes each perform their own, so the per-issue count is 32x the per-lane
 *     work.
 *
 * Counting dp4a at 4 MACs instead of 4*32 reported it at 0.6 TOPS and produced
 * the conclusion "dp4a is 147x slower than the int8 tensor core". The corrected
 * figure is ~13-19 TOPS -- slower than IMMA, but a perfectly reasonable decode
 * primitive rather than a trap. Same error made popc look 32x worse than it is.
 */
constexpr double kMacsWmma16 = 16.0 * 16.0 * 16.0;
constexpr double kMacsImma4 = 8.0 * 8.0 * 32.0;
constexpr double kMacsBmma = 8.0 * 8.0 * 128.0;
constexpr double kMacsDp4a = 4.0 * WST_WARP;    /* 4 int8 MACs, per lane */
constexpr double kMacsPopc = 32.0 * WST_WARP;   /* 32 binary MACs, per lane */

template <typename Launch>
double measure_tops(const ProbeCtx &ctx, double macs_per_issue, Launch launch,
                    bool *supported) {
  cudaGetLastError();               /* clear stale */
  launch();                         /* warmup */
  if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
    *supported = false;
    return 0.0;
  }

  WstTimer t;
  if (!t.ok) { *supported = false; return 0.0; }

  const int reps = 20;
  t.tic();
  for (int i = 0; i < reps; ++i) launch();
  const float ms = t.toc_ms();

  if (cudaGetLastError() != cudaSuccess) { *supported = false; return 0.0; }

  *supported = true;
  const double ops = ctx.warp_issues * 2.0 * macs_per_issue * reps;
  return ops / (ms * 1.0e-3) / 1.0e12;
}

}  // namespace

extern "C" wst_status_t wst_probe(wst_probe_t *out, int32_t iters) {
  WST_REQUIRE(out != nullptr, "wst_probe: null out");
  WST_REQUIRE(iters > 0, "wst_probe: iters must be positive");

  memset(out, 0, sizeof(*out));

  cudaDeviceProp p;
  WST_TRY(cudaGetDeviceProperties(&p, 0));

  const size_t SZ = 1u << 16;
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  wst_status_t st;
  if ((st = wst_malloc(&dA, SZ)) != WST_OK) return st;
  if ((st = wst_malloc(&dB, SZ)) != WST_OK) { wst_free(dA); return st; }
  if ((st = wst_malloc(&dC, SZ)) != WST_OK) { wst_free(dA); wst_free(dB); return st; }

  cudaMemset(dA, 0x5A, SZ);
  cudaMemset(dB, 0x3C, SZ);
  cudaMemset(dC, 0, SZ);

  /* --- validate the identity the binary path depends on --- */
  {
    int32_t *flag = nullptr;
    if (wst_malloc((void **)&flag, sizeof(int32_t)) == WST_OK) {
      probe_xnor_identity<<<1, 1>>>((const uint32_t *)dA, (const uint32_t *)dB, 256, flag);
      if (cudaDeviceSynchronize() == cudaSuccess) {
        int32_t h = 0;
        cudaMemcpy(&h, flag, sizeof(h), cudaMemcpyDeviceToHost);
        out->xnor_identity_ok = h;
      }
      wst_free(flag);
    }
  }

  ProbeCtx ctx;
  ctx.blocks = p.multiProcessorCount * 8;
  ctx.threads = 32;
  ctx.iters = iters;
  ctx.warp_issues = (double)ctx.blocks * (ctx.threads / 32) * (double)iters;

  bool sup = false;

  /* What the *image* holds, as opposed to what the device could run. */
  int32_t img[4] = {0, 0, 0, 0};
  {
    int32_t *d_caps = nullptr;
    if (wst_malloc((void **)&d_caps, sizeof(img)) == WST_OK) {
      probe_image_caps<<<1, 1>>>(d_caps);
      if (cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(img, d_caps, sizeof(img), cudaMemcpyDeviceToHost);
      }
      wst_free(d_caps);
    }
  }
  const bool has_wmma = WST_HOST_HAS_WMMA && img[0];
  const bool has_imma = WST_HOST_HAS_IMMA && img[1];
  const bool has_bmma = WST_HOST_HAS_BMMA_XOR && img[2];
  const bool has_dp4a = WST_HOST_HAS_DP4A && img[3];

  if (has_wmma) {
    out->fp16_wmma_tflops = measure_tops(ctx, kMacsWmma16, [&] {
      probe_hmma<<<ctx.blocks, ctx.threads>>>((const half *)dA, (const half *)dB,
                                              (float *)dC, ctx.iters);
    }, &sup);
    if (!sup) out->fp16_wmma_tflops = -1.0;
  } else {
    out->fp16_wmma_tflops = -1.0;
  }

  /* Runtime, not `#if`: this host code is compiled once and runs against
   * whatever card is installed, which the build did not know. -1 means "this
   * device cannot do it", and is reported as such rather than as a zero. */
  if (has_imma) {
    out->int8_wmma_tops = measure_tops(ctx, kMacsWmma16, [&] {
      probe_imma8<<<ctx.blocks, ctx.threads>>>((const int8_t *)dA, (const int8_t *)dB,
                                               (int32_t *)dC, ctx.iters);
    }, &sup);
    if (!sup) out->int8_wmma_tops = -1.0;
  } else {
    out->int8_wmma_tops = -1.0;
  }

  if (has_bmma) {
    out->int4_wmma_tops = measure_tops(ctx, kMacsImma4, [&] {
      probe_imma4<<<ctx.blocks, ctx.threads>>>((const uint32_t *)dA, (const uint32_t *)dB,
                                               (int32_t *)dC, ctx.iters);
    }, &sup);
    if (!sup) out->int4_wmma_tops = -1.0;

    out->bin_bmma_tops = measure_tops(ctx, kMacsBmma, [&] {
      probe_bmma<<<ctx.blocks, ctx.threads>>>((const uint32_t *)dA, (const uint32_t *)dB,
                                              (int32_t *)dC, ctx.iters);
    }, &sup);
    if (!sup) out->bin_bmma_tops = -1.0;
  } else {
    out->int4_wmma_tops = -1.0;
    out->bin_bmma_tops = -1.0;
  }

  if (has_dp4a) {
    out->dp4a_tops = measure_tops(ctx, kMacsDp4a, [&] {
      probe_dp4a<<<ctx.blocks, ctx.threads>>>((const int32_t *)dA, (const int32_t *)dB,
                                              (int32_t *)dC, ctx.iters);
    }, &sup);
    if (!sup) out->dp4a_tops = -1.0;
  } else {
    out->dp4a_tops = -1.0;
  }

  out->popc_tops = measure_tops(ctx, kMacsPopc, [&] {
    probe_popc<<<ctx.blocks, ctx.threads>>>((const uint32_t *)dA, (const uint32_t *)dB,
                                            (int32_t *)dC, ctx.iters);
  }, &sup);
  if (!sup) out->popc_tops = -1.0;

  wst_free(dA);
  wst_free(dB);
  wst_free(dC);
  return WST_OK;
}
