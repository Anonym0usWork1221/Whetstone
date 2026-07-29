//! The decode loop: one transformer forward pass per token, on the device.
//!
//! # Shape of the problem
//!
//! At batch=1 every projection is a matrix-*vector* product. Nothing here is a
//! GEMM, nothing tiles, and no amount of arithmetic cleverness helps: each
//! weight is read once and used for one multiply-add, so the token rate is
//! `bandwidth / bytes_read_per_token` and nothing else. The engine's job is
//! therefore narrow — stream the weights at full bandwidth and get out of the
//! way.
//!
//! # Where the time actually goes
//!
//! Two costs, in this order:
//!
//! 1. **Weight traffic.** ~262 MB per token at int4-g128, which at the ~278 GB/s
//!    this card actually achieves is a 0.94 ms floor.
//! 2. **Launch overhead.** Ten kernels per block times 24 blocks, plus the head:
//!    ~250 dispatches per token. At a couple of microseconds each that is a
//!    third of the budget, and it is *entirely* fixed cost — it does not shrink
//!    when the format does.
//!
//! Everything else — normalisation, rotary embedding, softmax, SwiGLU — moves a
//! few kilobytes and costs nothing measurable. That asymmetry is why bias and
//! the residual add live in GEMV epilogues and why RoPE, the KV append and the
//! f16 narrowing are one kernel: the goal is not to save arithmetic, it is to
//! save launches.
//!
//! # Precision
//!
//! The residual stream is fp32 for all 24 blocks. Activations narrow to f16 only
//! when handed to a GEMV, and the KV cache is f16 because it is re-read every
//! token and its width is bandwidth. Turing has no bf16, and fp16 accumulation
//! over 896 terms is visibly lossy, so the fp32 residual is not optional.

use std::time::Instant;

use whetstone_kernels::{decode, DeviceBuffer, Device};

use crate::error::Result;
use crate::model::ModelWeights;

mod profile;
pub use profile::{Profile, Stage};

/// Per-token activation buffers, allocated once and reused.
///
/// Allocating inside the token loop would put a synchronising driver call in the
/// hot path; the whole working set is under 700 KB, so it simply lives for the
/// duration of the session.
struct Activations {
    /// The residual stream, fp32.
    x: DeviceBuffer<f32>,
    /// Normalised activations, f16, feeding the next projection.
    h: DeviceBuffer<u16>,
    /// The fused q/k/v projection's output: queries first, then keys, then
    /// values. RoPE rotates the queries in place and the keys into the cache.
    qkv: DeviceBuffer<f32>,
    /// Attention output, f16, feeding `o_proj`.
    attn: DeviceBuffer<u16>,
    /// The fused gate/up projection's output: gate first, then up.
    gate_up: DeviceBuffer<f32>,
    /// SwiGLU product, f16, feeding `down_proj`.
    act: DeviceBuffer<u16>,
    /// Output logits.
    logits: DeviceBuffer<f32>,
    /// The current token id, on the device.
    ///
    /// The argmax writes it and the embedding gather reads it, so a whole
    /// generation can run without the id ever crossing the bus — which is what
    /// makes the step capturable as a graph.
    token: decode::DeviceCursor,
    /// The decode position, on the device, for the same reason.
    pos_dev: decode::DeviceCursor,
}

/// How the next token is chosen.
#[derive(Debug, Clone, Copy)]
pub enum Sampler {
    /// Highest logit. Stays entirely on the device.
    Greedy,
    /// Stochastic sampling, on the host.
    ///
    /// Costs a 608 KB device-to-host copy of the logits plus an O(vocab)
    /// selection, measured at 369 tok/s against greedy's 467 — about 20%. That
    /// is the price of a distribution the GPU cannot hand back cheaply, and it
    /// is why `--temperature 0` exists.
    Sample(SamplingConfig),
}

/// The knobs a stochastic sampler exposes.
///
/// Applied in this order, which is the order llama.cpp and the transformers
/// `LogitsProcessor` stack use, and the order matters: a repetition penalty
/// applied *after* truncation cannot push a token out of the candidate set,
/// which is most of what it is for.
///
/// 1. repetition penalty over the recent history
/// 2. temperature
/// 3. top-k
/// 4. min-p
/// 5. top-p (nucleus)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// Softmax temperature. `<= 0` means greedy.
    pub temperature: f32,
    /// Nucleus mass. `>= 1` disables.
    pub top_p: f32,
    /// Keep only the `k` highest logits. `0` disables.
    pub top_k: usize,
    /// Drop candidates below `min_p * p_max`. `0` disables.
    ///
    /// Scales the cut with how confident the model is, which top-p does not: at
    /// a sharp distribution it keeps almost nothing, at a flat one it keeps a
    /// lot. That is usually what you want and it is one parameter instead of
    /// two.
    pub min_p: f32,
    /// Divide the logits of recently emitted tokens by this. `1.0` disables.
    pub repeat_penalty: f32,
    /// How far back the repetition penalty looks.
    pub repeat_last_n: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        // Qwen's own recommended settings, so the default REPL behaves the way
        // the model was tuned to.
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            repeat_penalty: 1.05,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

/// Timing for one generation run.
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    /// Prompt tokens processed.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub generated: usize,
    /// Seconds spent on the prompt.
    pub prefill_seconds: f64,
    /// Seconds spent generating.
    pub decode_seconds: f64,
    /// Per-token decode latency in milliseconds, in order.
    pub token_ms: Vec<f64>,
}

impl RunStats {
    /// Decode tokens per second.
    pub fn decode_tok_s(&self) -> f64 {
        if self.decode_seconds <= 0.0 {
            0.0
        } else {
            self.generated as f64 / self.decode_seconds
        }
    }

    /// Prefill tokens per second.
    pub fn prefill_tok_s(&self) -> f64 {
        if self.prefill_seconds <= 0.0 {
            0.0
        } else {
            self.prompt_tokens as f64 / self.prefill_seconds
        }
    }

    /// Median and 10th/90th percentile of per-token latency, in milliseconds.
    ///
    /// A mean hides the thing worth knowing. Decode latency on a desktop GPU is
    /// bimodal — the compositor preempts — and a p10/p90 spread that is wide
    /// relative to the median means the number is measuring machine contention,
    /// not the engine.
    pub fn latency_percentiles(&self) -> Option<(f64, f64, f64)> {
        if self.token_ms.is_empty() {
            return None;
        }
        let mut v = self.token_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
        Some((at(0.10), at(0.50), at(0.90)))
    }
}

/// A loaded model plus everything a decode step needs.
pub struct Engine {
    weights: ModelWeights,
    acts: Activations,
    caches: Vec<decode::KvCache>,
    rope: decode::RopeTable,
    /// Tokens currently in the KV cache. Mirrors the device cursor, and exists
    /// so context-full is an error the host can report rather than a clamp.
    pos: usize,
    max_seq: usize,
    device: Device,
    graph: Option<decode::Graph>,
    /// Reused by nucleus sampling so the token loop never allocates.
    sample_order: Vec<u32>,
    /// Recently seen token ids, for the repetition penalty.
    ///
    /// Kept by the engine rather than the caller because it must span the whole
    /// conversation, not one `generate` call: in a chat REPL the tokens worth
    /// penalising are mostly from the turn before. Bounded so a long session
    /// does not grow it without limit.
    recent: std::collections::VecDeque<u32>,
}

impl Engine {
    /// Builds an engine around loaded weights, sized for `max_seq` tokens.
    pub fn new(weights: ModelWeights, max_seq: usize) -> Result<Self> {
        let device = Device::default_device()?;
        let c = &weights.config;

        let hidden = c.hidden_size;
        let hd = c.head_dim();
        let n_q = c.num_attention_heads;
        let n_kv = c.n_kv_heads();
        let inter = c.intermediate_size;

        let acts = Activations {
            x: DeviceBuffer::zeros(hidden)?,
            h: DeviceBuffer::zeros(hidden)?,
            qkv: DeviceBuffer::zeros((n_q + 2 * n_kv) * hd)?,
            attn: DeviceBuffer::zeros(n_q * hd)?,
            gate_up: DeviceBuffer::zeros(2 * inter)?,
            act: DeviceBuffer::zeros(inter)?,
            logits: DeviceBuffer::zeros(c.vocab_size)?,
            token: decode::DeviceCursor::new(0)?,
            pos_dev: decode::DeviceCursor::new(0)?,
        };

        let mut caches = Vec::with_capacity(c.num_hidden_layers);
        for _ in 0..c.num_hidden_layers {
            caches.push(decode::KvCache::new(n_kv, n_q, hd, max_seq)?);
        }

        // The rotation is identical across every family here; only the frequency
        // schedule differs, so this is a table parameter and not a kernel variant.
        let scaling = match &c.rope_scaling {
            Some(rs) if rs.rope_type == "llama3" => decode::RopeScaling::Llama3 {
                factor: rs.factor.unwrap_or(8.0),
                low_freq_factor: rs.low_freq_factor.unwrap_or(1.0),
                high_freq_factor: rs.high_freq_factor.unwrap_or(4.0),
                original_max_position: rs.original_max_position_embeddings.unwrap_or(8192),
            },
            _ => decode::RopeScaling::None,
        };
        let rope =
            decode::RopeTable::with_scaling(max_seq, hd, c.rope_theta as f64, scaling)?;

        Ok(Self {
            weights,
            acts,
            caches,
            rope,
            pos: 0,
            max_seq,
            device,
            graph: None,
            sample_order: Vec::new(),
            recent: std::collections::VecDeque::with_capacity(RECENT_CAP + 1),
        })
    }

    /// Captures the whole decode step as a CUDA graph.
    ///
    /// After this, [`Engine::forward`] issues one launch instead of ~250. The
    /// work is identical; what disappears is the per-kernel driver call and most
    /// of the gap between kernels.
    ///
    /// One ordinary step runs first, deliberately. Capture forbids allocation,
    /// and two of the kernels lazily `cudaMalloc` a few bytes of reduction
    /// scratch on first use while `pick_rows_per_block` caches a device query.
    /// All of that has to have already happened, or capture fails with an error
    /// a long way from its cause.
    pub fn capture_graph(&mut self) -> Result<usize> {
        if self.graph.is_some() {
            return Ok(self.graph.as_ref().map_or(0, |g| g.launches));
        }

        // Warm up every lazily-initialised path, then rewind.
        let saved = self.pos;
        self.step_eager()?;
        self.device.synchronize()?;
        self.pos = saved;
        self.acts.pos_dev.set(saved as i32)?;

        // SAFETY of the capture contract, not of memory: `step_eager` issues
        // only kernel launches and async memsets. It must not synchronise,
        // allocate, or copy back -- see cuda/graph.cu.
        let engine: *mut Self = self;
        let graph = decode::Graph::capture(|| {
            // SAFETY: the closure runs to completion inside `capture` before it
            // returns, and nothing else holds a borrow of `self` meanwhile. The
            // raw pointer exists only because `Graph::capture` cannot take a
            // `&mut` that also outlives the borrow of `self.graph`.
            // The two crates have separate error types and capture is a
            // kernels-level operation, so the message is what survives. A
            // failure here is a capture-contract violation (an allocation or a
            // sync inside the region), which the text names precisely.
            unsafe { (*engine).step_eager() }
                .map_err(|e| whetstone_kernels::Error::Cuda(e.to_string()))
        })?;

        let n = graph.launches;
        self.graph = Some(graph);

        // The warm-up step advanced the device cursor; put it back.
        self.pos = saved;
        self.acts.pos_dev.set(saved as i32)?;
        Ok(n)
    }

    /// True when a captured graph is driving the decode step.
    pub fn graph_enabled(&self) -> bool {
        self.graph.is_some()
    }

    /// The loaded weights.
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// The device the engine runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tokens currently in the KV cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Cache capacity in tokens.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Device bytes held by the KV cache and rotary table.
    pub fn state_bytes(&self) -> usize {
        self.caches.iter().map(decode::KvCache::bytes).sum::<usize>() + self.rope.bytes()
    }

    /// Discards the KV cache and returns to position zero.
    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        self.acts.pos_dev.set(0)?;
        self.recent.clear();
        Ok(())
    }

    /// Runs one token through the whole stack, leaving logits on the device.
    ///
    /// The caller owns the position bookkeeping: this appends to the cache at
    /// the current position and advances it.
    pub fn forward(&mut self, token: u32) -> Result<()> {
        if self.pos >= self.max_seq {
            return Err(crate::Error::Shape(format!(
                "context is full at {} tokens; raise --ctx or reset",
                self.max_seq
            )));
        }
        self.acts.token.set(token as i32)?;
        self.note(token);
        self.step()
    }

    /// Records a token in the repetition-penalty window.
    fn note(&mut self, token: u32) {
        if self.recent.len() == RECENT_CAP {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
    }

    /// Runs one step against whatever token id is already on the device.
    ///
    /// This is the form generation uses: the argmax leaves its choice in the
    /// device cursor, so the next step needs no host round trip at all.
    pub fn step(&mut self) -> Result<()> {
        if self.pos >= self.max_seq {
            return Err(crate::Error::Shape(format!(
                "context is full at {} tokens; raise --ctx or reset",
                self.max_seq
            )));
        }
        match &self.graph {
            Some(g) => {
                g.launch()?;
                self.pos += 1;
                Ok(())
            }
            None => {
                self.step_eager()?;
                self.pos += 1;
                Ok(())
            }
        }
    }

    /// The decode step as individual launches.
    ///
    /// Always ends with the argmax and the position advance, even when the
    /// caller intends to sample differently or not at all. Two reasons: greedy
    /// decode needs the result and it is only a few microseconds against a 3 ms
    /// token, and a graph has to contain *every* per-token operation or the
    /// generation loop stops being host-free. One code path is worth more than
    /// the 3 µs a perplexity run wastes on an argmax it ignores.
    ///
    /// Returns the number of launches issued, which is the figure the graph
    /// collapses to one.
    fn step_eager(&mut self) -> Result<usize> {
        let c = &self.weights.config;
        let eps = c.rms_norm_eps;
        let n_q = c.num_attention_heads;
        let a = &mut self.acts;
        let mut launches = 0usize;

        self.weights.embed.gather(&a.token, &mut a.x)?;
        launches += 1;

        for (l, layer) in self.weights.layers.iter().enumerate() {
            // ---- attention ----
            decode::rmsnorm(&a.x, &layer.input_norm, &mut a.h, eps)?;

            layer.qkv_proj.forward(&a.h, layer.qkv_bias.as_ref(), &mut a.qkv, false)?;

            decode::rope_cache(&mut a.qkv, &mut self.caches[l], &self.rope, n_q, &a.pos_dev)?;
            decode::attn_decode(&a.qkv, &mut self.caches[l], &mut a.attn, n_q, &a.pos_dev)?;

            // Accumulating GEMV: the projection adds straight into the residual
            // stream instead of writing a temporary for another kernel to add.
            layer.o_proj.forward(&a.attn, None, &mut a.x, true)?;

            // ---- MLP ----
            decode::rmsnorm(&a.x, &layer.post_attn_norm, &mut a.h, eps)?;
            layer.gate_up_proj.forward(&a.h, None, &mut a.gate_up, false)?;
            decode::swiglu(&a.gate_up, &mut a.act)?;
            layer.down_proj.forward(&a.act, None, &mut a.x, true)?;

            launches += 9; // the attention split emits a merge kernel of its own
        }

        decode::rmsnorm(&a.x, &self.weights.final_norm, &mut a.h, eps)?;
        match &self.weights.lm_head {
            Some(head) => head.forward(&a.h, None, &mut a.logits, false)?,
            None => self.weights.embed.project(&a.h, &mut a.logits)?,
        }
        launches += 2;

        decode::argmax(&a.logits, a.token.buffer_mut())?;
        a.pos_dev.advance(self.max_seq)?;
        launches += 4; // memset, reduce, extract, advance

        Ok(launches)
    }

    /// Cross-entropy over a token stream, in non-overlapping windows.
    ///
    /// Each window starts from an empty KV cache and predicts positions
    /// `1..window`, which is exactly what a full-window causal forward pass
    /// computes — so this is directly comparable to a HuggingFace perplexity
    /// number taken the same way, and that comparability is the whole point.
    /// The accumulation stays on the device; only two floats come back.
    ///
    /// `on_window` is called after each window with `(index, running_nll,
    /// positions)` so a long run can report progress without this function
    /// knowing how.
    pub fn cross_entropy(
        &mut self,
        tokens: &[u32],
        window: usize,
        max_windows: usize,
        mut on_window: impl FnMut(usize, f64, usize),
    ) -> Result<(f64, usize)> {
        if window < 2 {
            return Err(crate::Error::Shape("window must be at least 2 tokens".into()));
        }
        if window > self.max_seq {
            return Err(crate::Error::Shape(format!(
                "window {window} exceeds the {} token cache; raise --ctx",
                self.max_seq
            )));
        }

        let mut acc = DeviceBuffer::<f32>::zeros(2)?;
        let n = (tokens.len() / window).min(max_windows);

        for w in 0..n {
            self.reset()?;
            let chunk = &tokens[w * window..(w + 1) * window];
            for i in 0..window - 1 {
                self.forward(chunk[i])?;
                decode::nll(&self.acts.logits, chunk[i + 1], &mut acc)?;
            }
            let got = acc.to_vec()?;
            on_window(w, got[0] as f64, got[1] as usize);
        }

        let got = acc.to_vec()?;
        Ok((got[0] as f64, got[1] as usize))
    }

    /// Copies the current logits to the host.
    ///
    /// 608 KB across PCIe, so this is for evaluation, not the token loop.
    pub fn logits(&self) -> Result<Vec<f32>> {
        Ok(self.acts.logits.to_vec()?)
    }

    /// Picks the next token from the logits already on the device.
    pub fn sample(&mut self, sampler: Sampler, step: u64) -> Result<u32> {
        match sampler {
            // The step already ran the argmax into the device cursor, so greedy
            // is a 4-byte read rather than a reduction.
            Sampler::Greedy => {
                let t = self.acts.token.get()? as u32;
                self.note(t);
                Ok(t)
            }
            Sampler::Sample(cfg) => {
                let mut logits = self.logits()?;
                let t = sample_with(
                    &mut logits,
                    &cfg,
                    step,
                    self.recent.iter().copied(),
                    &mut self.sample_order,
                );
                // Put the choice where the next step's embedding gather looks.
                self.acts.token.set(t as i32)?;
                self.note(t);
                Ok(t)
            }
        }
    }

    /// Feeds a prompt, returning the logits position for the last token.
    ///
    /// Prefill runs the same single-token path in a loop. That is the honest
    /// starting point and it is not what a fast engine does — prefill is
    /// compute bound, so it wants a batched GEMM path. Until that exists this
    /// is `prompt_len` decode steps, and the reported prefill rate says so.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        for &t in tokens {
            self.forward(t)?;
        }
        Ok(())
    }

    /// Generates up to `max_new` tokens, invoking `on_token` for each.
    ///
    /// Returning `false` from the callback stops generation, which is how a stop
    /// token or a user interrupt is handled without this function knowing about
    /// either.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        sampler: Sampler,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<RunStats> {
        let mut stats = RunStats { prompt_tokens: prompt.len(), ..Default::default() };
        if prompt.is_empty() {
            return Err(crate::Error::Shape("cannot generate from an empty prompt".into()));
        }

        // Every timed region is bracketed by a device synchronise. Without it
        // the CPU races ahead of the queue and measures dispatch, not execution.
        self.device.synchronize()?;
        let t0 = Instant::now();
        self.prefill(prompt)?;
        self.device.synchronize()?;
        stats.prefill_seconds = t0.elapsed().as_secs_f64();

        let mut next = self.sample(sampler, 0)?;

        let t1 = Instant::now();
        for step in 0..max_new {
            let tok_start = Instant::now();

            if !on_token(next) {
                break;
            }
            stats.generated += 1;

            if self.pos >= self.max_seq {
                break;
            }
            // Greedy leaves its choice in the device cursor, so the next step
            // needs no token argument -- and therefore no host round trip.
            self.step()?;
            next = self.sample(sampler, step as u64 + 1)?;

            // `sample` blocks: greedy reads back the chosen id, top-p reads the
            // whole logit vector. Either way the time below covers the token.
            stats.token_ms.push(tok_start.elapsed().as_secs_f64() * 1e3);
        }
        self.device.synchronize()?;
        stats.decode_seconds = t1.elapsed().as_secs_f64();

        Ok(stats)
    }
}

/// Candidates considered for nucleus sampling.
///
/// Nucleus sampling needs the distribution in descending order, and sorting a
/// 151936-entry vocabulary costs about 8 ms — **four times the entire forward
/// pass**. `select_nth_unstable_by` partitions off the top `K` in O(n) and only
/// those get sorted, which is O(n + K log K).
///
/// 512 is far more than a nucleus of p ≤ 0.95 reaches on this vocabulary; when
/// the mass genuinely is that flat the effect is a top-k=512 filter, which is
/// what every other engine applies anyway (llama.cpp defaults to top-k 40).
const NUCLEUS_POOL: usize = 512;

/// How many recent tokens the repetition penalty can see.
///
/// The window a caller asks for is clamped to this. 2048 is far past any useful
/// `repeat_last_n` — llama.cpp defaults to 64 — and bounds the memory a very
/// long chat session holds for a feature that only ever looks at the tail.
const RECENT_CAP: usize = 2048;

/// Stochastic sampling on the host: penalty, temperature, top-k, min-p, top-p.
///
/// `logits` is mutated in place — the caller already owns a fresh copy from the
/// device, and the repetition penalty has to touch the full vector rather than
/// the truncated pool. Penalising only the survivors of a top-k cut cannot push
/// a repeated token *out* of the candidate set, which is most of what the
/// penalty is for.
///
/// `order` is caller-owned scratch so the token loop never allocates.
fn sample_with(
    logits: &mut [f32],
    cfg: &SamplingConfig,
    step: u64,
    recent: impl Iterator<Item = u32>,
    order: &mut Vec<u32>,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // --- 1. repetition penalty ------------------------------------------
    //
    // The CTRL formulation llama.cpp uses: divide a positive logit, multiply a
    // negative one. Both move it toward zero, which is the point; a plain
    // subtraction would flip the sign of a strongly negative logit and make a
    // penalised token *more* likely.
    if cfg.repeat_penalty != 1.0 && cfg.repeat_last_n > 0 {
        let window: Vec<u32> = {
            let all: Vec<u32> = recent.collect();
            let n = cfg.repeat_last_n.min(all.len());
            all[all.len() - n..].to_vec()
        };
        for &t in &window {
            let i = t as usize;
            if i < logits.len() {
                logits[i] = if logits[i] > 0.0 {
                    logits[i] / cfg.repeat_penalty
                } else {
                    logits[i] * cfg.repeat_penalty
                };
            }
        }
    }

    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }

    // Descending by logit. `unwrap_or(Equal)` rather than `unwrap`: a NaN logit
    // means a broken model, and panicking inside the sampler turns that into a
    // crash a long way from its cause.
    let by_logit = |a: &u32, b: &u32| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    order.clear();
    order.extend(0..logits.len() as u32);

    // --- 2. narrow to a pool, then sort only that ------------------------
    let pool = if cfg.top_k > 0 { cfg.top_k.min(NUCLEUS_POOL) } else { NUCLEUS_POOL };
    let k = pool.min(order.len());
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, by_logit);
        order.truncate(k);
    }
    order.sort_unstable_by(by_logit);

    // --- 3. temperature --------------------------------------------------
    let max = logits[order[0] as usize] as f64;
    let mut probs: Vec<f64> = Vec::with_capacity(k);
    let mut total = 0f64;
    for &i in order.iter() {
        let p = (((logits[i as usize] as f64) - max) / cfg.temperature as f64).exp();
        total += p;
        probs.push(p);
    }

    // --- 4. min-p ---------------------------------------------------------
    //
    // Relative to the top candidate, so the cut tightens automatically when the
    // model is confident and loosens when it is not. `probs[0]` is the largest
    // by construction, and the values are unnormalised, so the threshold is a
    // fraction of it directly.
    let mut cut = probs.len();
    if cfg.min_p > 0.0 {
        let floor = probs[0] * cfg.min_p as f64;
        cut = probs.iter().position(|&p| p < floor).unwrap_or(probs.len()).max(1);
    }

    // --- 5. top-p ---------------------------------------------------------
    if cfg.top_p < 1.0 {
        let mut acc = 0f64;
        for (i, &p) in probs.iter().enumerate().take(cut) {
            acc += p / total;
            if acc >= cfg.top_p as f64 {
                cut = i + 1;
                break;
            }
        }
    }

    // --- 6. draw ----------------------------------------------------------
    let mass: f64 = probs[..cut].iter().sum();
    let u = splitmix64(cfg.seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15)) as f64
        / u64::MAX as f64
        * mass;

    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate().take(cut) {
        acc += p;
        if acc >= u {
            return order[i];
        }
    }
    order[cut.saturating_sub(1)]
}

/// Index of the largest logit, NaN-tolerant.
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

/// SplitMix64. Deterministic, seeded, and one line — a full PRNG dependency for
/// picking a token would be more surface than the job needs.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut r = z;
    r = (r ^ (r >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    r = (r ^ (r >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    r ^ (r >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_report_the_spread_not_the_mean() {
        // Nearest-rank, so p90 is the 18th of 20 sorted samples. Three stalls in
        // twenty is enough to reach it; two would not be, which is the point of
        // a percentile rather than a max.
        let mut token_ms = vec![10.0; 17];
        token_ms.extend([60.0, 60.0, 60.0]);
        let s = RunStats { token_ms, ..Default::default() };

        let (p10, p50, p90) = s.latency_percentiles().unwrap();
        assert_eq!(p10, 10.0);
        assert_eq!(p50, 10.0);
        // The stall has to show up. A mean of 17.5 ms would report neither the
        // 10 ms the engine actually runs at nor the 60 ms stall.
        assert_eq!(p90, 60.0);
    }

    /// A config that isolates one knob: nothing else filters.
    fn plain(temperature: f32, top_p: f32, seed: u64) -> SamplingConfig {
        SamplingConfig {
            temperature,
            top_p,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed,
        }
    }

    fn draw(l: &[f32], cfg: &SamplingConfig, step: u64) -> u32 {
        let mut v = l.to_vec();
        sample_with(&mut v, cfg, step, std::iter::empty(), &mut Vec::new())
    }

    #[test]
    fn greedy_is_what_zero_temperature_means() {
        let mut l = vec![0.0f32; 64];
        l[37] = 5.0;
        assert_eq!(draw(&l, &plain(0.0, 0.9, 1), 0), 37);
    }

    #[test]
    fn nucleus_sampling_never_leaves_the_nucleus() {
        // One token holds essentially all the mass, so top_p = 0.9 must select
        // it every time regardless of seed.
        let mut l = vec![-20.0f32; 512];
        l[100] = 10.0;
        for seed in 0..32u64 {
            assert_eq!(draw(&l, &plain(1.0, 0.9, seed), seed), 100);
        }
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let l: Vec<f32> = (0..1024).map(|i| ((i * 37 % 101) as f32) / 20.0).collect();
        let cfg = plain(0.8, 0.95, 7);
        let a: Vec<u32> = (0..16).map(|s| draw(&l, &cfg, s)).collect();
        let b: Vec<u32> = (0..16).map(|s| draw(&l, &cfg, s)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_bounds_what_can_be_drawn() {
        // A flat distribution: without a cut, any of 512 tokens is reachable.
        // With top-k 3 only the three highest can come out, whatever the seed.
        let l: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let cfg = SamplingConfig { top_k: 3, ..plain(2.0, 1.0, 0) };
        for step in 0..64u64 {
            let t = draw(&l, &cfg, step);
            assert!(t >= 509, "top-k 3 produced {t}, which is not in the top three");
        }
    }

    #[test]
    fn min_p_tightens_when_the_model_is_confident() {
        // The point of min-p over top-p: the cut is relative to the leader, so a
        // peaked distribution collapses to it and a flat one does not.
        let mut peaked = vec![0.0f32; 256];
        peaked[9] = 12.0;
        let cfg = SamplingConfig { min_p: 0.1, ..plain(1.0, 1.0, 3) };
        for step in 0..32u64 {
            assert_eq!(draw(&peaked, &cfg, step), 9);
        }

        // Flat: min-p must NOT collapse it, or it is just a greedy switch.
        let flat = vec![1.0f32; 256];
        let seen: std::collections::HashSet<u32> =
            (0..64u64).map(|s| draw(&flat, &cfg, s)).collect();
        assert!(seen.len() > 1, "min-p collapsed a flat distribution to one token");
    }

    #[test]
    fn repetition_penalty_moves_logits_toward_zero_from_both_sides() {
        // The CTRL formulation, and the reason it is a divide rather than a
        // subtract: a subtraction applied to a negative logit makes a repeated
        // token MORE likely, which is backwards.
        let mut l = vec![-8.0f32; 8];
        l[1] = 6.0; // positive, repeated -> should shrink
        l[2] = 5.0; // positive, not repeated -> untouched, should now win
        let cfg = SamplingConfig {
            repeat_penalty: 2.0,
            repeat_last_n: 4,
            ..plain(0.0, 1.0, 0)
        };
        let mut v = l.clone();
        let t = sample_with(&mut v, &cfg, 0, [1u32].into_iter(), &mut Vec::new());
        assert_eq!(t, 2, "penalised token 1 should have fallen behind token 2");
        assert!(v[1] < l[1], "positive logit should shrink");

        // A negative logit must move toward zero too, never away.
        let mut v2 = l.clone();
        sample_with(&mut v2, &cfg, 0, [0u32].into_iter(), &mut Vec::new());
        assert!(v2[0] < l[0], "negative logit must be pushed further from zero, \
                               so the token becomes less likely");
    }

    #[test]
    fn repetition_penalty_only_looks_back_repeat_last_n() {
        let mut l = vec![0.0f32; 8];
        l[3] = 1.0;
        let cfg = SamplingConfig {
            repeat_penalty: 2.0,
            repeat_last_n: 2,
            ..plain(0.0, 1.0, 0)
        };
        // Token 3 is five back, outside the window, so it must be untouched.
        let mut v = l.clone();
        sample_with(&mut v, &cfg, 0, [3u32, 5, 5, 5, 5].into_iter(), &mut Vec::new());
        assert_eq!(v[3], l[3], "a token outside repeat_last_n was penalised");
    }
}
