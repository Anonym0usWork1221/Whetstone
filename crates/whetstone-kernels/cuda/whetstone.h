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

/* Allocate in host RAM, readable by kernels over PCIe. For weights that do not
 * fit in VRAM. Freed with wst_free. See runtime.cu for why this is managed
 * memory plus two cudaMemAdvise calls and not any of the obvious alternatives --
 * the naive form is 13x slower and fails silently. */
wst_status_t wst_malloc_host(void **out_ptr, size_t bytes);
int32_t wst_host_alloc_supported(void);
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

/* GEMV with a fused bias add and an optional accumulate-into-y epilogue.
 *
 * These two extras remove four kernel launches per transformer block: the q/k/v
 * projections carry biases in Qwen2, and o_proj/down_proj write straight into
 * the residual stream. At 24 blocks that is ~96 launches per token, which at
 * batch=1 is a real fraction of the token budget -- see docs/design.md.
 *
 *   bias  : [out_f] half, or NULL
 *   accum : 0 -> y = Wx + b,  nonzero -> y += Wx + b
 */
wst_status_t wst_gemv_int4_g128_ex(const void *qw, const void *sz, const void *x,
                                   const void *bias, void *y, int32_t in_f,
                                   int32_t out_f, int32_t accum);

wst_status_t wst_gemv_fp16_ex(const void *w, const void *x, const void *bias,
                              void *y, int32_t in_f, int32_t out_f, int32_t accum);

/* ------------------------------------------------------- GEMV variant sweep */

/* Alternative int4 GEMV implementations, so the choice between them is a
 * measurement rather than an argument. See gemv_variants.cu for what varies and
 * why. The engine picks one at startup; the sweep is what decides which. */
int32_t wst_gemv_variant_count(void);

/* Index of the variant the sweep selected on this architecture. */
int32_t wst_gemv_default_variant(void);

/* Index of the variant the sweep selected for a particular matrix shape.
 * No single blocking wins everywhere -- see gemv_variants.cu for the table. */
int32_t wst_gemv_variant_for_shape(int32_t in_f, int32_t out_f);

/* Override the per-shape rule: (wide reduction, huge output, everything else).
 * Exists so the rule can be swept by whole-generation tok/s, which is the only
 * measurement that has not misranked these kernels. */
void wst_gemv_set_shape_rule(int32_t wide, int32_t huge, int32_t other);
void wst_gemv_get_shape_rule(int32_t *out);
const char *wst_gemv_variant_name(int32_t variant);

wst_status_t wst_gemv_int4_variant(int32_t variant, const void *qw, const void *sz,
                                   const void *x, const void *bias, void *y, int32_t in_f,
                                   int32_t out_f, int32_t accum);

wst_status_t wst_bench_gemv_variant(int32_t variant, int32_t in_f, int32_t out_f,
                                    int32_t reps, double *out_gbs, double *out_ms);

/* ------------------------------------------------------------ decode layers */

/* out[i] = f16( x[i] * rsqrt(mean(x^2) + eps) * w[i] )
 *
 * The reduction runs in fp32 regardless of the input type. Accumulating 896
 * squares in fp16 loses roughly three decimal digits, and Turing has no bf16 to
 * fall back on.
 *
 *   x   : [n] float   (the residual stream)
 *   w   : [n] half
 *   out : [n] half    (the next projection's input)
 */
wst_status_t wst_rmsnorm(const void *x, const void *w, void *out, int32_t n, float eps);

/* Rotary embedding on q and k, plus the append of k/v into the KV cache.
 *
 * Fused because all three touch the same freshly projected vectors, and because
 * RoPE's cos/sin come from a precomputed table -- there is no arithmetic left to
 * amortise, only launches.
 *
 * HuggingFace's *half rotation* layout: the head vector splits into halves and
 * rotates across them, NOT as adjacent (even, odd) pairs. Getting this wrong
 * yields fluent text with subtly wrong long-range behaviour.
 *
 *   qkv      : [n_q + 2*n_kv][head_dim] float -- one fused projection's output;
 *              q is rotated in place, k is rotated into the cache, v is copied
 *   k_cache  : [n_kv][max_seq][head_dim] half, written at `pos`
 *   v_cache  : same
 *   cos, sin : [max_seq][head_dim/2] float, precomputed in f64 on the host
 */
wst_status_t wst_rope_cache(void *qkv, void *k_cache, void *v_cache, const void *cos_tab,
                            const void *sin_tab, int32_t n_q, int32_t n_kv,
                            int32_t head_dim, const void *pos, int32_t max_seq);

/* Batch=1 GQA attention against the KV cache.
 *
 * One block per query head, online (flash-style) softmax so nothing proportional
 * to the sequence length is ever materialised.
 *
 *   q        : [n_q][head_dim] float
 *   k/v cache: [n_kv][max_seq][head_dim] half
 *   partials : scratch, wst_attn_partial_floats() floats
 *   out      : [n_q][head_dim] half  (o_proj's input)
 *
 * The sequence is split across blocks as well as the heads -- 14 query heads
 * cannot fill 30 SMs -- so this issues two kernels: slices, then a merge. */
wst_status_t wst_attn_decode(const void *q, const void *k_cache, const void *v_cache,
                             void *partials, void *out, int32_t n_q, int32_t n_kv,
                             int32_t head_dim, const void *pos, int32_t max_seq,
                             float scale);

/* Scratch floats the sequence split needs for one layer. */
int32_t wst_attn_partial_floats(int32_t n_q, int32_t head_dim, int32_t max_seq);

/* out[i] = f16( silu(gu[i]) * gu[i+n] ),  silu(x) = x * sigmoid(x).
 * `gu` is the 2n-wide output of one fused gate|up projection. */
wst_status_t wst_swiglu(const void *gate_up, void *out, int32_t n);

/* Gathers row `token` of a dense fp16 table into a float vector. */
wst_status_t wst_embed_fp16(const void *table, const void *token, void *out,
                            int32_t hidden, int32_t rows);

/* Same, dequantizing on the way out of an int4-g128 table. */
wst_status_t wst_embed_int4_g128(const void *qw, const void *sz, const void *token,
                                 void *out, int32_t hidden, int32_t rows);

/* ------------------------------------------------------------ CUDA graphs */

/* Capture the decode step once, launch it per token.
 *
 * ~250 kernels become one launch. Everything that changes per token -- the
 * position and the token id -- lives in device int32s, because a graph bakes its
 * kernel arguments in at instantiation. Nothing may allocate, copy
 * synchronously, or synchronise between begin and end. See cuda/graph.cu. */
wst_status_t wst_graph_capture_begin(void);
wst_status_t wst_graph_capture_end(void **out_exec);
wst_status_t wst_graph_launch(void *exec);
wst_status_t wst_graph_destroy(void *exec);
wst_status_t wst_stream_sync(void);

/* Stream-ordered timers. Recorded into the stream, so the host never blocks and
 * the pipeline is never broken -- unlike a synchronise-between-stages profile,
 * which measures its own interference as much as the work. */
wst_status_t wst_event_create(void **out);
wst_status_t wst_event_record(void *ev);
wst_status_t wst_event_elapsed_ms(void *a, void *b, float *out);
wst_status_t wst_event_destroy(void *ev);

/* pos += 1, saturating at max_seq, as a graph node rather than a host update. */
wst_status_t wst_advance_pos(void *pos, int32_t max_seq);

/* ---------------------------------------------------------------- sampling */

/* Greedy decode: index of the largest logit, written to a device int32. */
wst_status_t wst_argmax(const void *logits, void *out_idx, int32_t n);

/* Accumulates -log p(target) into acc[0] and bumps the count in acc[1].
 *
 * Both stay on the device: a perplexity run is tens of thousands of forward
 * passes, and copying a scalar back after each one would put a synchronising
 * transfer inside a loop that otherwise never blocks. */
wst_status_t wst_nll(const void *logits, int32_t target, void *acc, int32_t n);

/* ------------------------------------------------------- multi-token chunk */

/* The batched forms of the decode step. Every activation buffer is token-major
 * [n][dim]; `pos0` is the cache position of the chunk's first token, passed by
 * value because a chunk pass is not graph-captured (its width changes with the
 * speculative acceptance length).
 *
 * These exist so a weight can be read once and used for n tokens. See
 * cuda/chunk_gemm.cu for why that is the only lever that changes the batch-1
 * roofline. */

/* Largest token count a single weight pass covers; wider requests are sliced. */
int32_t wst_chunk_max_tokens(void);

wst_status_t wst_gemm_int4_hier(const void *qw, const void *si, const void *sb,
                                const void *x, const void *bias, void *y, int32_t in_f,
                                int32_t out_f, int32_t n, int32_t accum);
wst_status_t wst_gemm_fp16(const void *w, const void *x, const void *bias, void *y,
                           int32_t in_f, int32_t out_f, int32_t n, int32_t accum);

wst_status_t wst_rmsnorm_chunk(const void *x, const void *w, void *out, int32_t dim,
                               int32_t n, float eps);
wst_status_t wst_rope_cache_chunk(void *qkv, void *k_cache, void *v_cache,
                                  const void *cos_tab, const void *sin_tab, int32_t n_q,
                                  int32_t n_kv, int32_t head_dim, int32_t pos0, int32_t n,
                                  int32_t max_seq);
wst_status_t wst_attn_chunk(const void *qkv, const void *k_cache, const void *v_cache,
                            void *out, int32_t n_q, int32_t n_kv, int32_t head_dim,
                            int32_t pos0, int32_t n, int32_t max_seq, float scale);
wst_status_t wst_swiglu_chunk(const void *gate_up, void *out, int32_t inter, int32_t n);

wst_status_t wst_embed_fp16_chunk(const void *table, const void *tokens, void *out,
                                  int32_t hidden, int32_t rows, int32_t n);
wst_status_t wst_embed_int4_g128_chunk(const void *qw, const void *sz, const void *tokens,
                                       void *out, int32_t hidden, int32_t rows,
                                       int32_t n);
wst_status_t wst_embed_int4_hier_chunk(const void *qw, const void *si, const void *sb,
                                       const void *tokens, void *out, int32_t hidden,
                                       int32_t rows, int32_t n);

/* Greedy choice for every row of [n][vocab], reduced on the device. */
wst_status_t wst_argmax_chunk(const void *logits, void *out, int32_t vocab, int32_t n);

#ifdef __cplusplus
}
#endif

#endif /* WHETSTONE_H */
