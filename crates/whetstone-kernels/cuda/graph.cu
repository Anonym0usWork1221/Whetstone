/* CUDA graph capture for the decode step.
 *
 * # The problem this solves
 *
 * A decode step for Qwen2.5-0.5B issues around 250 kernels: seven GEMVs and
 * five small kernels per block, times 24 blocks, plus the head and the argmax.
 * Every one of them is a driver call on the way out and a scheduler round trip
 * on the way in. Measured per-stage, the small kernels -- RMSNorm, RoPE, SwiGLU
 * -- account for about a third of the token while moving a few kilobytes each.
 * That is not bandwidth and it is not arithmetic; it is dispatch.
 *
 * A graph collapses all of it into one launch. The work is identical; what
 * disappears is the per-kernel host cost and most of the gap between kernels.
 *
 * # What it costs to be capturable
 *
 * A graph bakes its kernel arguments in at instantiation. Anything that changes
 * per token therefore cannot be an argument -- it has to be read from device
 * memory by the kernel itself. Two things change per token:
 *
 *   - the **position**, which RoPE indexes the cos/sin table with and which
 *     attention uses as its sequence length;
 *   - the **token id**, which the embedding gathers.
 *
 * Both now live in device int32s, and `advance_pos_kernel` increments the
 * position as the last node of the graph. The consequence is that neither can be
 * range-checked by the host before launch, so both kernels clamp instead.
 *
 * # Why the per-thread default stream
 *
 * The legacy default stream cannot be captured. Compiling with
 * `--default-stream per-thread` makes every `<<<>>>` with no explicit stream go
 * to `cudaStreamPerThread`, which can -- so the entire existing kernel surface
 * became capturable without a stream parameter being threaded through all of it.
 *
 * # What must not happen during capture
 *
 * No allocation, no blocking copy, no synchronise. Two of Whetstone's kernels
 * lazily `cudaMalloc` a few bytes of reduction scratch on first use, and
 * `pick_rows_per_block` caches `cudaGetDeviceProperties`. All of that has to
 * have already run, which is why the engine executes one ordinary decode step
 * before it captures.
 */

#include "common.cuh"

extern "C" wst_status_t wst_graph_capture_begin(void) {
  /* ThreadLocal mode: capture affects only this thread's stream, so an
   * unrelated CUDA user in the process is not caught up in it. */
  WST_TRY(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
  return WST_OK;
}

extern "C" wst_status_t wst_graph_capture_end(void **out_exec) {
  WST_REQUIRE(out_exec, "wst_graph_capture_end: null out pointer");
  *out_exec = nullptr;

  cudaGraph_t graph = nullptr;
  WST_TRY(cudaStreamEndCapture(cudaStreamPerThread, &graph));
  if (graph == nullptr) {
    wst_set_error_msg("wst_graph_capture_end: capture produced no graph");
    return WST_ERR_CUDA;
  }

  cudaGraphExec_t exec = nullptr;
  const cudaError_t e = cudaGraphInstantiateWithFlags(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (e != cudaSuccess) {
    wst_set_error("cudaGraphInstantiate", e);
    return WST_ERR_CUDA;
  }

  *out_exec = (void *)exec;
  return WST_OK;
}

extern "C" wst_status_t wst_graph_node_count(void *exec_or_null, size_t *out) {
  /* Instantiated graphs do not expose a node count, so this reports on nothing
   * once capture has ended. Kept as a stub so the Rust side has a stable place
   * to ask; the engine reports the count it captured instead. */
  (void)exec_or_null;
  WST_REQUIRE(out, "wst_graph_node_count: null out pointer");
  *out = 0;
  return WST_OK;
}

extern "C" wst_status_t wst_graph_launch(void *exec) {
  WST_REQUIRE(exec, "wst_graph_launch: null graph");
  WST_TRY(cudaGraphLaunch((cudaGraphExec_t)exec, cudaStreamPerThread));
  return WST_OK;
}

extern "C" wst_status_t wst_graph_destroy(void *exec) {
  if (exec == nullptr) return WST_OK;
  WST_TRY(cudaGraphExecDestroy((cudaGraphExec_t)exec));
  return WST_OK;
}

/* Synchronises the stream graphs are launched on.
 *
 * `cudaDeviceSynchronize` would also do it, but naming the stream keeps the
 * intent explicit: what the caller is waiting for is its own queued work, not
 * everything the device happens to be doing. */
extern "C" wst_status_t wst_stream_sync(void) {
  WST_TRY(cudaStreamSynchronize(cudaStreamPerThread));
  return WST_OK;
}

/* ------------------------------------------------------------------ events */

/* Stream-ordered timers, for attributing a decode step to its stages.
 *
 * The obvious way to profile -- `cudaDeviceSynchronize` between stages -- is the
 * way that lies. It serialises work the driver would otherwise overlap and adds
 * a driver round trip per stage, and it produced a breakdown here in which a
 * 448-byte embedding gather appeared to cost half a millisecond. Events are
 * recorded *into the stream*: the host never blocks, the pipeline is not broken,
 * and the timestamps come from the GPU's own clock. One synchronise at the end
 * of the run reads them all.
 */
extern "C" wst_status_t wst_event_create(void **out) {
  WST_REQUIRE(out, "wst_event_create: null out pointer");
  cudaEvent_t e = nullptr;
  WST_TRY(cudaEventCreate(&e));
  *out = (void *)e;
  return WST_OK;
}

extern "C" wst_status_t wst_event_record(void *ev) {
  WST_REQUIRE(ev, "wst_event_record: null event");
  WST_TRY(cudaEventRecord((cudaEvent_t)ev, cudaStreamPerThread));
  return WST_OK;
}

extern "C" wst_status_t wst_event_elapsed_ms(void *a, void *b, float *out) {
  WST_REQUIRE(a && b && out, "wst_event_elapsed_ms: null pointer");
  WST_TRY(cudaEventElapsedTime(out, (cudaEvent_t)a, (cudaEvent_t)b));
  return WST_OK;
}

extern "C" wst_status_t wst_event_destroy(void *ev) {
  if (ev == nullptr) return WST_OK;
  WST_TRY(cudaEventDestroy((cudaEvent_t)ev));
  return WST_OK;
}
