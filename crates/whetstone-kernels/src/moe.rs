//! Mixture-of-experts routing.
//!
//! # What sparsity buys, and what it does not
//!
//! A dense block reads three MLP matrices per token. A MoE block stores
//! `n_experts` copies and reads `k`, so the roofline denominator is a fraction
//! of the parameter count in the model's name — Qwen3-30B-A3B stores 30.5 B and
//! reads 3.0 B. For an engine whose entire thesis is that batch-1 decode is
//! bounded by bytes rather than FLOPs, that is the architecture the thesis was
//! waiting for.
//!
//! It buys nothing the weights are not resident for. Experts that overflow VRAM
//! are read across PCIe at 5.77 GB/s against DRAM's 278, which is the difference
//! between a 109 tok/s roofline and 5 tok/s measured reality. See
//! `research/01-V6-PLAN.md` §0.1 — the arithmetic there is calibrated against a
//! measured dense model, not assumed.
//!
//! # Everything the router produces stays on the device
//!
//! Expert selection is data-dependent, and a CUDA graph bakes its kernel
//! arguments in at instantiation. So the indices and weights are written to
//! device buffers and read by the kernels that consume them, exactly as the
//! position cursor is. Routing through the host instead would cost `k` sync
//! points per layer, which at 48 layers is the whole token budget.

use crate::ffi;
use crate::{check, DeviceBuffer, Error, Result};

/// The router's decision for one token: which experts, and how much each counts.
///
/// Device-resident for the reason in the module docs. Allocated once per model
/// and overwritten every token, so a decode step makes no allocations.
pub struct ExpertChoice {
    /// Selected expert ids, `k` of them, best first.
    pub(crate) idx: DeviceBuffer<i32>,
    /// Their routing weights, aligned with `idx`.
    pub(crate) weight: DeviceBuffer<f32>,
    /// Experts selected per token.
    pub k: usize,
    /// Experts available to select from.
    pub n_experts: usize,
}

impl ExpertChoice {
    /// Allocates routing scratch for a `k`-of-`n_experts` layer.
    pub fn new(n_experts: usize, k: usize) -> Result<Self> {
        if n_experts == 0 || n_experts > 1024 {
            return Err(Error::Shape(format!(
                "moe: n_experts {n_experts} must be in 1..=1024 (one block routes them)"
            )));
        }
        if k == 0 || k > n_experts {
            return Err(Error::Shape(format!(
                "moe: k {k} must be in 1..={n_experts}"
            )));
        }
        Ok(Self {
            idx: DeviceBuffer::zeros(k)?,
            weight: DeviceBuffer::zeros(k)?,
            k,
            n_experts,
        })
    }

    /// The chosen expert ids, copied to the host.
    ///
    /// For tests and for `--profile`. The decode path never calls it: a copy
    /// back is a synchronisation point, and the whole design exists to avoid one.
    pub fn indices_to_host(&self) -> Result<Vec<i32>> {
        self.idx.to_vec()
    }

    /// The routing weights, copied to the host. Same caveat as
    /// [`ExpertChoice::indices_to_host`].
    pub fn weights_to_host(&self) -> Result<Vec<f32>> {
        self.weight.to_vec()
    }
}

/// Softmax over every expert logit, then the `k` largest.
///
/// The order is HuggingFace's and is not the obvious one: softmax runs over
/// **all** experts and the top-k is taken afterwards, so each weight carries the
/// full denominator. Selecting first and softmaxing the survivors gives
/// different weights whenever the discarded experts hold real mass, and produces
/// a perfectly valid-looking distribution while doing so.
///
/// `norm_topk` renormalises the survivors to sum to 1. Qwen3-MoE and Mixtral set
/// it; OLMoE does not. It is a config flag rather than a constant because
/// getting it wrong scales every expert's contribution by one shared factor,
/// which reads as a slightly mis-tuned model rather than as a bug.
pub fn router(
    logits: &DeviceBuffer<f32>,
    choice: &mut ExpertChoice,
    norm_topk: bool,
) -> Result<()> {
    if logits.len() != choice.n_experts {
        return Err(Error::Shape(format!(
            "moe router: {} logits for {} experts",
            logits.len(),
            choice.n_experts
        )));
    }
    // SAFETY: the logit count is checked against the expert count the kernel
    // indexes, and `idx`/`weight` were allocated with exactly `k` elements by
    // `ExpertChoice::new`, which is the only constructor.
    check(unsafe {
        ffi::wst_moe_router(
            logits.as_ptr(),
            choice.n_experts as i32,
            choice.k as i32,
            norm_topk as i32,
            choice.idx.as_mut_ptr(),
            choice.weight.as_mut_ptr(),
        )
    })
}

/// `dst += weight[slot] * src`, with the scalar read from device memory.
///
/// This is what turns `k` independent expert outputs into their convex
/// combination. The weight cannot be a kernel argument: it is only known on the
/// device, and passing it would mean a copy back per expert per layer.
pub fn accumulate(
    dst: &mut DeviceBuffer<f32>,
    src: &DeviceBuffer<f32>,
    choice: &ExpertChoice,
    slot: usize,
) -> Result<()> {
    if dst.len() != src.len() {
        return Err(Error::Shape(format!(
            "moe accumulate: dst[{}] and src[{}] must agree",
            dst.len(),
            src.len()
        )));
    }
    if slot >= choice.k {
        return Err(Error::Shape(format!(
            "moe accumulate: slot {slot} is outside the {} selected experts",
            choice.k
        )));
    }
    // SAFETY: lengths are checked equal and `slot` is checked against the
    // allocation the kernel indexes.
    check(unsafe {
        ffi::wst_moe_accumulate(
            dst.as_mut_ptr(),
            src.as_ptr(),
            choice.weight.as_ptr(),
            slot as i32,
            dst.len() as i32,
        )
    })
}
