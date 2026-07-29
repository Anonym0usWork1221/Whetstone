/* Device management and memory. The boring half of the ABI. */

#include "common.cuh"

thread_local char wst_err_buf[512] = "no error";

extern "C" const char *wst_last_error(void) { return wst_err_buf; }

extern "C" void wst_abi_layout(wst_abi_layout_t *out) {
  if (out == nullptr) return;
  out->device_info_size = (uint32_t)sizeof(wst_device_info_t);
  out->device_info_align = (uint32_t)alignof(wst_device_info_t);
  out->probe_size = (uint32_t)sizeof(wst_probe_t);
  out->probe_align = (uint32_t)alignof(wst_probe_t);
}

/* ------------------------------------------------------------------ device */

extern "C" wst_status_t wst_device_count(int32_t *out_count) {
  WST_REQUIRE(out_count != nullptr, "wst_device_count: null out_count");
  int n = 0;
  WST_TRY(cudaGetDeviceCount(&n));
  *out_count = n;
  return WST_OK;
}

extern "C" wst_status_t wst_device_set(int32_t ordinal) {
  WST_TRY(cudaSetDevice(ordinal));
  return WST_OK;
}

extern "C" wst_status_t wst_device_synchronize(void) {
  WST_TRY(cudaDeviceSynchronize());
  return WST_OK;
}

extern "C" wst_status_t wst_device_info(int32_t ordinal, wst_device_info_t *out) {
  WST_REQUIRE(out != nullptr, "wst_device_info: null out");

  cudaDeviceProp p;
  WST_TRY(cudaGetDeviceProperties(&p, ordinal));

  memset(out, 0, sizeof(*out));
  snprintf(out->name, sizeof(out->name), "%s", p.name);
  out->major = p.major;
  out->minor = p.minor;
  out->sm_count = p.multiProcessorCount;
  out->clock_khz = p.clockRate;
  out->mem_clock_khz = p.memoryClockRate;
  out->mem_bus_width = p.memoryBusWidth;
  out->max_threads_per_block = p.maxThreadsPerBlock;
  out->max_smem_per_block = (int32_t)p.sharedMemPerBlock;
  out->warp_size = p.warpSize;
  out->l2_bytes = p.l2CacheSize;
  out->mem_total = (uint64_t)p.totalGlobalMem;

  /* DDR: two transfers per clock. */
  out->bandwidth_gbs = 2.0 * (double)p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6;

  size_t freeb = 0, totalb = 0;
  if (cudaMemGetInfo(&freeb, &totalb) == cudaSuccess) {
    out->mem_free = (uint64_t)freeb;
    out->mem_total = (uint64_t)totalb;
  }

  const int cc = p.major * 10 + p.minor;
  out->has_tensor_cores = cc >= 70;
  out->has_imma         = cc >= 72;
  out->has_bmma_xor     = cc >= 75;  /* Turing: .xor.popc only */
  out->has_bmma_and     = cc >= 80;  /* Ampere added .and.popc */
  out->has_cp_async     = cc >= 80;
  out->has_sparse_tc    = cc >= 80;
  out->has_fp8          = cc >= 89;

  return WST_OK;
}

/* ------------------------------------------------------------------ memory */

extern "C" wst_status_t wst_malloc(void **out_ptr, size_t bytes) {
  WST_REQUIRE(out_ptr != nullptr, "wst_malloc: null out_ptr");
  *out_ptr = nullptr;
  if (bytes == 0) return WST_OK;

  void *p = nullptr;
  cudaError_t e = cudaMalloc(&p, bytes);
  if (e == cudaErrorMemoryAllocation) {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    snprintf(wst_err_buf, sizeof(wst_err_buf),
             "wst_malloc: out of memory requesting %.1f MB (%.1f MB free of %.1f MB)",
             bytes / 1e6, freeb / 1e6, totalb / 1e6);
    return WST_ERR_OOM;
  }
  if (e != cudaSuccess) {
    wst_set_error("wst_malloc", e);
    return WST_ERR_CUDA;
  }
  *out_ptr = p;
  return WST_OK;
}

/* An allocation that stays in host RAM and is read by kernels over PCIe.
 *
 * # Why managed memory with an advice pair, and not the obvious alternatives
 *
 * Measured on this machine (research/experiments/probe_offload.cu, Gen3 x8):
 *
 *     cudaMallocManaged, fault-driven migration        0.47 GB/s   <- the trap
 *     cudaHostAlloc(Mapped), kernel reads host         5.13 GB/s
 *     cudaMemcpy pinned H2D, then read from VRAM       5.77 GB/s
 *     cudaMallocManaged + PreferredLocation(CPU)
 *                        + AccessedBy(GPU)             6.46 GB/s   <- this
 *
 * Plain managed memory *works* -- it allocates, the kernel runs, the output is
 * correct -- and it is **thirteen times slower** than the same allocation with
 * two advice calls, because every page migrates in and back out again on a
 * working set that cannot fit. That failure is invisible: no error, no warning,
 * just a model that reads as if the quantizer were broken.
 *
 * `SetPreferredLocation(cudaCpuDeviceId)` stops the migration and
 * `SetAccessedBy(device)` establishes the mapping, leaving plain PCIe reads --
 * which is the honest ceiling for a weight that does not fit in VRAM.
 *
 * Freed with `wst_free`: `cudaFree` accepts managed pointers, so the caller does
 * not have to remember which allocator produced a buffer.
 */
extern "C" wst_status_t wst_malloc_host(void **out_ptr, size_t bytes) {
  WST_REQUIRE(out_ptr != nullptr, "wst_malloc_host: null out_ptr");
  *out_ptr = nullptr;
  if (bytes == 0) return WST_OK;

  int device = 0;
  WST_TRY(cudaGetDevice(&device));

  void *p = nullptr;
  cudaError_t e = cudaMallocManaged(&p, bytes);
  if (e != cudaSuccess) {
    snprintf(wst_err_buf, sizeof(wst_err_buf),
             "wst_malloc_host: could not allocate %.1f MB of host-resident memory (%s)",
             bytes / 1e6, cudaGetErrorString(e));
    return e == cudaErrorMemoryAllocation ? WST_ERR_OOM : WST_ERR_CUDA;
  }

  /* Not optional. Without both of these the allocation runs at 0.47 GB/s. */
  cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, device);
  return (*out_ptr = p), WST_OK;
}

/* Whether the driver reports host-resident allocation as usable at all. */
extern "C" int32_t wst_host_alloc_supported(void) {
  int managed = 0;
  if (cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, 0) != cudaSuccess) return 0;
  return managed;
}

extern "C" wst_status_t wst_free(void *ptr) {
  if (ptr == nullptr) return WST_OK;
  WST_TRY(cudaFree(ptr));
  return WST_OK;
}

extern "C" wst_status_t wst_memset(void *dst, int32_t value, size_t bytes) {
  if (bytes == 0) return WST_OK;
  WST_REQUIRE(dst != nullptr, "wst_memset: null dst");
  WST_TRY(cudaMemset(dst, value, bytes));
  return WST_OK;
}

extern "C" wst_status_t wst_memcpy_h2d(void *dst, const void *src, size_t bytes) {
  if (bytes == 0) return WST_OK;
  WST_REQUIRE(dst && src, "wst_memcpy_h2d: null pointer");
  WST_TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
  return WST_OK;
}

extern "C" wst_status_t wst_memcpy_d2h(void *dst, const void *src, size_t bytes) {
  if (bytes == 0) return WST_OK;
  WST_REQUIRE(dst && src, "wst_memcpy_d2h: null pointer");
  WST_TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
  return WST_OK;
}

extern "C" wst_status_t wst_memcpy_d2d(void *dst, const void *src, size_t bytes) {
  if (bytes == 0) return WST_OK;
  WST_REQUIRE(dst && src, "wst_memcpy_d2d: null pointer");
  WST_TRY(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  return WST_OK;
}

/* --------------------------------------------------------------- bandwidth */

/* Pure streaming read. 128-bit loads, fully coalesced, grid sized to saturate.
 * The sum is written out so nothing is optimised away. */
__global__ void bandwidth_read_kernel(const float4 *__restrict__ src, size_t n4,
                                      float *__restrict__ sink) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  s = warp_reduce_sum(s);
  if ((threadIdx.x & 31) == 0 && s == 1.2345e-30f) sink[0] = s;  /* never true */
}

extern "C" wst_status_t wst_bench_bandwidth(size_t bytes, int32_t reps, double *out_gbs) {
  WST_REQUIRE(out_gbs != nullptr, "wst_bench_bandwidth: null out");
  WST_REQUIRE(reps > 0, "wst_bench_bandwidth: reps must be positive");
  WST_REQUIRE(bytes >= 4096, "wst_bench_bandwidth: buffer too small to measure");

  const size_t n4 = bytes / sizeof(float4);
  WST_REQUIRE(n4 > 0, "wst_bench_bandwidth: buffer smaller than one float4");

  void *buf = nullptr, *sink = nullptr;
  wst_status_t st = wst_malloc(&buf, n4 * sizeof(float4));
  if (st != WST_OK) return st;
  st = wst_malloc(&sink, sizeof(float) * 4);
  if (st != WST_OK) { wst_free(buf); return st; }

  cudaMemset(buf, 1, n4 * sizeof(float4));

  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    wst_free(buf); wst_free(sink);
    wst_set_error_msg("wst_bench_bandwidth: cannot query device");
    return WST_ERR_CUDA;
  }

  const int threads = 256;
  const int blocks = p.multiProcessorCount * 16;

  bandwidth_read_kernel<<<blocks, threads>>>((const float4 *)buf, n4, (float *)sink);
  cudaDeviceSynchronize();

  WstTimer t;
  if (!t.ok) { wst_free(buf); wst_free(sink); wst_set_error_msg("event create failed"); return WST_ERR_CUDA; }

  t.tic();
  for (int i = 0; i < reps; ++i)
    bandwidth_read_kernel<<<blocks, threads>>>((const float4 *)buf, n4, (float *)sink);
  float ms = t.toc_ms();

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    wst_free(buf); wst_free(sink);
    wst_set_error("wst_bench_bandwidth", e);
    return WST_ERR_CUDA;
  }

  const double total_bytes = (double)n4 * sizeof(float4) * (double)reps;
  *out_gbs = total_bytes / (ms * 1.0e-3) / 1.0e9;

  wst_free(buf);
  wst_free(sink);
  return WST_OK;
}
