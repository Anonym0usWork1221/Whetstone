/* Whetstone CUDA C ABI.
 *
 * This is the entire surface Rust is allowed to touch. Rules for anything added
 * here:
 *   - plain C types only, no C++ in the signature
 *   - every entry point returns wst_status_t; results go through out-parameters
 *   - no function may abort, longjmp, or throw across this boundary
 *   - pointers are device pointers unless the name says _host
 */

#ifndef WHETSTONE_H
#define WHETSTONE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  WST_OK = 0,
  WST_ERR_CUDA = 1,          /* underlying CUDA call failed; see wst_last_error */
  WST_ERR_INVALID_ARG = 2,
  WST_ERR_UNSUPPORTED_ARCH = 3,
  WST_ERR_OOM = 4,
  WST_ERR_SHAPE = 5,
} wst_status_t;

/* ------------------------------------------------------------------ device */

typedef struct {
  char name[256];
  int32_t major;             /* compute capability */
  int32_t minor;
  int32_t sm_count;
  int32_t clock_khz;
  int32_t mem_clock_khz;
  int32_t mem_bus_width;     /* bits */
  int32_t max_threads_per_block;
  int32_t max_smem_per_block;
  int32_t warp_size;
  int32_t l2_bytes;
  uint64_t mem_total;
  uint64_t mem_free;
  double bandwidth_gbs;      /* 2 * mem_clock * bus_width / 8 */

  /* Capability flags, resolved from compute capability. These gate kernel
   * selection at runtime -- see docs/hardware.md for why each boundary is
   * where it is. */
  int32_t has_tensor_cores;  /* sm_70+ : wmma fp16 */
  int32_t has_imma;          /* sm_72+ : int8/int4 tensor cores */
  int32_t has_bmma_xor;      /* sm_75+ : bmma .xor.popc  <- Whetstone's binary path */
  int32_t has_bmma_and;      /* sm_80+ : bmma .and.popc */
  int32_t has_cp_async;      /* sm_80+ : LDGSTS software pipelining */
  int32_t has_sparse_tc;     /* sm_80+ : 2:4 structured sparsity */
  int32_t has_fp8;           /* sm_89+ */
} wst_device_info_t;

/* Reports sizeof/alignof for every struct crossing the ABI, so the Rust side
 * can assert agreement instead of hardcoding offsets it computed by hand.
 * Any field added to a struct here must keep this in sync automatically. */
typedef struct {
  uint32_t device_info_size;
  uint32_t device_info_align;
  uint32_t probe_size;
  uint32_t probe_align;
} wst_abi_layout_t;

void wst_abi_layout(wst_abi_layout_t *out);

wst_status_t wst_device_count(int32_t *out_count);
wst_status_t wst_device_set(int32_t ordinal);
wst_status_t wst_device_info(int32_t ordinal, wst_device_info_t *out_info);
wst_status_t wst_device_synchronize(void);

/* Last CUDA error string for the calling thread. Never NULL. */
const char *wst_last_error(void);

/* ------------------------------------------------------------------ memory */

wst_status_t wst_malloc(void **out_ptr, size_t bytes);
wst_status_t wst_free(void *ptr);
wst_status_t wst_memset(void *dst, int32_t value, size_t bytes);
wst_status_t wst_memcpy_h2d(void *dst, const void *src_host, size_t bytes);
wst_status_t wst_memcpy_d2h(void *dst_host, const void *src, size_t bytes);
wst_status_t wst_memcpy_d2d(void *dst, const void *src, size_t bytes);

/* ------------------------------------------------------- capability probe */

/* Measured throughput of each low-precision path, in TOPS (TFLOPS for fp16).
 * A value <= 0 means the op is unsupported on this device.
 *
 * These are *issue-rate* microbenchmarks with register-resident fragments: an
 * upper bound, not an achievable GEMM rate. Their purpose is to establish the
 * relative ordering of the arithmetic paths on whatever GPU we land on. */
typedef struct {
  double fp16_wmma_tflops;
  double int8_wmma_tops;
  double int4_wmma_tops;
  double bin_bmma_tops;
  double dp4a_tops;
  double popc_tops;
  int32_t xnor_identity_ok;  /* 1 if dot = K - 2*popcount(a^b) verified on device */
} wst_probe_t;

wst_status_t wst_probe(wst_probe_t *out, int32_t iters);

/* --------------------------------------------------------------- bandwidth */

/* Achieved device-memory read bandwidth in GB/s. This is the number that
 * actually governs batch=1 decode speed, so we measure it rather than trusting
 * the spec sheet. */
wst_status_t wst_bench_bandwidth(size_t bytes, int32_t reps, double *out_gbs);

/* --------------------------------------------------------------- decode GEMV */

/* y[out_f] = dequant(qw) * x[in_f], for batch=1 decode.
 *
 *   qw : [out_f][in_f/8] uint32, 8 nibbles per word
 *   sz : [out_f][in_f/128] half2, .x = scale, .y = zero
 *   x  : [in_f] half
 *   y  : [out_f] float
 *
 * in_f must be a multiple of 128 and small enough to stage in shared memory. */
wst_status_t wst_gemv_int4_g128(const void *qw, const void *sz, const void *x,
                                void *y, int32_t in_f, int32_t out_f);

/* Same decomposition at fp16. Separates kernel bugs from quantization loss. */
wst_status_t wst_gemv_fp16(const void *w, const void *x, void *y,
                           int32_t in_f, int32_t out_f);

/* Times a GEMV and reports achieved bandwidth. For a batch=1 GEMV this is the
 * only figure of merit: the kernel is good exactly insofar as it saturates the
 * memory system. */
wst_status_t wst_bench_gemv(int32_t in_f, int32_t out_f, int32_t reps,
                            int32_t use_int4, double *out_gbs, double *out_ms);

#ifdef __cplusplus
}
#endif

#endif /* WHETSTONE_H */
