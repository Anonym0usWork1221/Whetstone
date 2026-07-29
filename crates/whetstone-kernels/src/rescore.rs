//! Exact recomputation of the logits that decide the token.
//!
//! # What this buys, measured
//!
//! `lm_head` is one matrix read in full every token — 27.6% of decode traffic on
//! Qwen2.5-0.5B — so quantizing it is the largest single bandwidth win available
//! and also the change that most directly perturbs the output distribution.
//! From `research/experiments/head_lab.py`, on int4-hier-g32:
//!
//! | | Δ perplexity |
//! |---|---|
//! | no rescore | +0.5186 |
//! | k = 16 | +0.1595 |
//! | **k = 64** | **+0.0957** |
//! | k = 256 | +0.0634 |
//!
//! **k = 64 removes 82% of the head's quantization damage for 0.17% more
//! bandwidth** (114 KB against a 264 MB token), plus an fp16 copy of the head in
//! VRAM — 272 MB on the 0.5B, on a card with ~4 GB spare once the model is
//! resident. It spends the resource that is not binding to buy back the one that
//! is, which is the whole shape of this project.
//!
//! # Three fixed-shape launches, and why
//!
//! There is no top-k kernel here and there does not need to be one. A real top-k
//! over 151,936 logits is a radix select; a *threshold* selects the same set and
//! is two reductions. The threshold cannot be a constant — logit spreads vary per
//! token — so it is chosen on device from a geometric ladder by counting what
//! each candidate admits.
//!
//! Every launch is a fixed shape and every data-dependent quantity is read from
//! device memory, so the rescore lives inside the single captured decode graph
//! rather than forcing `k` host round-trips per token.

use crate::ffi;
use crate::{check, DeviceBuffer, Error, Result};

/// Scratch and the fp16 head copy, allocated once per model.
///
/// The fp16 head is the cost of this feature. It is *not* read per token — only
/// the selected rows are — so it costs VRAM and almost no bandwidth.
pub struct HeadRescore {
    /// Row-major `[vocab][hidden]` fp16 copy of the output projection.
    head: DeviceBuffer<u16>,
    /// Chosen threshold, one float, device-resident.
    thresh: DeviceBuffer<f32>,
    /// `[nblocks]` partial maxima, then the max, then the 12 candidate counters.
    /// One allocation because they are written and read in one launch sequence.
    scratch: DeviceBuffer<f32>,
    /// Blocks in the two grid-wide passes over the logits.
    nblocks: usize,
    /// How many logits cleared it. May exceed `cap`; consumers clamp.
    count: DeviceBuffer<i32>,
    /// Ids of the survivors, up to `cap` of them.
    idx: DeviceBuffer<i32>,
    /// Target number of rows to rescore.
    k: usize,
    /// Hard bound on rows rescored per token. Reaching it would reintroduce
    /// scheduling-dependent output, so it is set far above any realistic count.
    cap: usize,
    /// Blocks in the rescore launch. Fixed, so the graph capture is unaffected;
    /// each block strides over the selected rows.
    grid: usize,
    hidden: usize,
    vocab: usize,
}

impl HeadRescore {
    /// Uploads the fp16 head and allocates the scratch.
    ///
    /// `cap` bounds the work per token and is the grid width; `k` is what the
    /// threshold search aims for. `cap` above `k` gives the search room to
    /// overshoot without truncating, which matters because the alternative to
    /// overshooting is a threshold that admits fewer than `k`.
    pub fn new(head_fp16: &[u16], vocab: usize, hidden: usize, k: usize) -> Result<Self> {
        if head_fp16.len() != vocab * hidden {
            return Err(Error::Shape(format!(
                "head rescore: got {} elements for a [{vocab}, {hidden}] head",
                head_fp16.len()
            )));
        }
        if k == 0 || k > vocab {
            return Err(Error::Shape(format!(
                "head rescore: k {k} must be in 1..={vocab}"
            )));
        }
        // Generous, because the cap is the ONLY place non-determinism can
        // re-enter: below it every survivor is rescored and the compaction's
        // arbitrary order cannot matter, at it rows get dropped by scheduling
        // luck. 16x the target costs 4 KB of indices and makes truncation
        // unreachable for any distribution a language model actually produces.
        let cap = (k * 16).min(vocab).max(k);
        let grid = cap.min(256);
        // Enough blocks to spread 152k logits across the machine, bounded so the
        // finalize pass stays trivial. The first version reduced in a single
        // block and cost 7.7% of the decode step for a pass the bandwidth model
        // priced at 0.17% -- one SM of thirty reading 608 KB.
        let nblocks = vocab.div_ceil(256).clamp(1, 512);
        Ok(Self {
            scratch: DeviceBuffer::zeros(nblocks + 1 + 12)?,
            nblocks,
            head: DeviceBuffer::from_slice(head_fp16)?,
            thresh: DeviceBuffer::zeros(1)?,
            count: DeviceBuffer::zeros(1)?,
            idx: DeviceBuffer::zeros(cap)?,
            k,
            cap,
            grid,
            hidden,
            vocab,
        })
    }

    /// Bytes of VRAM this holds. Almost all of it is the fp16 head.
    pub fn bytes(&self) -> usize {
        self.head.bytes() + self.idx.bytes() + self.scratch.bytes() + 8
    }

    /// Rows rescored per token, at most.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Overwrites the largest entries of `logits` with values computed from the
    /// fp16 head.
    ///
    /// `x` is the same fp16 activation the quantized head GEMV consumed, so the
    /// only difference between the new logit and the old one is weight
    /// precision — which is exactly the error being removed.
    pub fn apply(&mut self, logits: &mut DeviceBuffer<f32>, x: &DeviceBuffer<u16>) -> Result<()> {
        if logits.len() != self.vocab {
            return Err(Error::Shape(format!(
                "head rescore: {} logits for a {} vocabulary",
                logits.len(),
                self.vocab
            )));
        }
        if x.len() != self.hidden {
            return Err(Error::Shape(format!(
                "head rescore: activation is {} wide, head expects {}",
                x.len(),
                self.hidden
            )));
        }

        // SAFETY: shapes are checked above against what each kernel indexes.
        // `thresh` and `count` are single live elements written by the first
        // kernel and read by the next two; `idx` holds exactly `cap` slots and
        // the compaction bounds its writes by that same `cap`.
        check(unsafe {
            ffi::wst_head_threshold(
                logits.as_ptr(),
                self.vocab as i32,
                self.k as i32,
                self.cap as i32,
                self.scratch.as_mut_ptr(),
                self.nblocks as i32,
                self.thresh.as_mut_ptr(),
                self.count.as_mut_ptr(),
            )
        })?;
        check(unsafe {
            ffi::wst_head_compact(
                logits.as_ptr(),
                self.vocab as i32,
                self.thresh.as_ptr(),
                self.count.as_mut_ptr(),
                self.idx.as_mut_ptr(),
                self.cap as i32,
            )
        })?;
        check(unsafe {
            ffi::wst_head_rescore(
                self.head.as_ptr(),
                x.as_ptr(),
                self.idx.as_ptr(),
                self.count.as_ptr(),
                logits.as_mut_ptr(),
                self.hidden as i32,
                self.cap as i32,
                self.grid as i32,
            )
        })
    }

    /// How many rows the last `apply` selected, before the cap. For tests and
    /// `--profile`; the decode path never reads it, because a copy back is a
    /// synchronisation point.
    pub fn last_count(&self) -> Result<i32> {
        Ok(self.count.to_vec()?[0])
    }

    /// The threshold the last `apply` chose. Diagnostics only.
    pub fn last_threshold(&self) -> Result<f32> {
        Ok(self.thresh.to_vec()?[0])
    }
}
