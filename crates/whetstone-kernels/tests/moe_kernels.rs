//! Differential tests for mixture-of-experts routing.
//!
//! The failure modes here are all silent. A router that softmaxes the top-k
//! instead of softmaxing everything still produces a valid distribution; one
//! that renormalises when it should not still produces fluent text at a slightly
//! wrong temperature; one that breaks ties nondeterministically produces
//! *correct* text that cannot be reproduced. None of them look like bugs, so
//! each is pinned against an explicit reference rather than a sanity check.

use whetstone_kernels::moe::{accumulate, router, ExpertChoice};
use whetstone_kernels::{Device, DeviceBuffer};

fn gpu() -> bool {
    Device::default_device().is_ok()
}

/// HuggingFace's routing, written the obvious way: softmax over everything, then
/// take the k largest, then optionally renormalise.
fn reference(logits: &[f32], k: usize, norm: bool) -> (Vec<i32>, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let denom: f64 = exp.iter().sum();
    let probs: Vec<f64> = exp.iter().map(|e| e / denom).collect();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    // Descending by logit, ties to the lower index — the kernel's rule.
    order.sort_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b))
    });

    let idx: Vec<i32> = order[..k].iter().map(|&i| i as i32).collect();
    let mut w: Vec<f32> = order[..k].iter().map(|&i| probs[i] as f32).collect();
    if norm {
        let s: f32 = w.iter().sum();
        for v in &mut w {
            *v /= s;
        }
    }
    (idx, w)
}

fn run_case(logits: &[f32], k: usize, norm: bool) {
    let n = logits.len();
    let ld = DeviceBuffer::from_slice(logits).unwrap();
    let mut choice = ExpertChoice::new(n, k).unwrap();
    router(&ld, &mut choice, norm).unwrap();

    let (want_idx, want_w) = reference(logits, k, norm);
    let got_idx = choice.indices_to_host().unwrap();
    let got_w = choice.weights_to_host().unwrap();

    assert_eq!(got_idx, want_idx, "expert ids for n={n} k={k} norm={norm}");
    for (i, (g, w)) in got_w.iter().zip(&want_w).enumerate() {
        assert!(
            (g - w).abs() < 1e-5 * w.max(1e-3),
            "weight {i} for n={n} k={k} norm={norm}: {g} vs {w}"
        );
    }
    if norm {
        let s: f32 = got_w.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "renormalised weights sum to {s}");
    }
}

/// The real geometries, plus the awkward ones.
///
/// 64-of-8 is OLMoE, 128-of-8 is Qwen3-30B-A3B, 8-of-2 is Mixtral. 65 is there
/// because it is one past a warp boundary times two, which is where a reduction
/// that forgets its partial warp goes wrong.
#[test]
fn router_matches_the_huggingface_reference() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    for &(n, k) in &[(8usize, 2usize), (64, 8), (128, 8), (65, 3), (1, 1), (32, 32)] {
        let logits: Vec<f32> = (0..n)
            .map(|i| ((i * 2_654_435_761usize) % 1997) as f32 / 200.0 - 5.0)
            .collect();
        run_case(&logits, k, false);
        run_case(&logits, k, true);
    }
}

/// OLMoE does **not** renormalise and Qwen3-MoE does, so the flag has to change
/// the answer. If it did not, one of the two families would be silently wrong
/// and every test above would still pass.
#[test]
fn renormalisation_actually_changes_the_weights() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 64;
    let logits: Vec<f32> = (0..n).map(|i| (i % 11) as f32 * 0.3).collect();
    let ld = DeviceBuffer::from_slice(&logits).unwrap();

    let mut plain = ExpertChoice::new(n, 8).unwrap();
    router(&ld, &mut plain, false).unwrap();
    let raw: f32 = plain.weights_to_host().unwrap().iter().sum();

    let mut normed = ExpertChoice::new(n, 8).unwrap();
    router(&ld, &mut normed, true).unwrap();
    let one: f32 = normed.weights_to_host().unwrap().iter().sum();

    assert!(raw < 0.95, "top-8 of 64 should not already carry the mass: {raw}");
    assert!((one - 1.0).abs() < 1e-5, "normalised weights sum to {one}");
}

/// Ties are not hypothetical — a saturated or freshly initialised router emits
/// them — and a nondeterministic choice would end the bit-exact reproducibility
/// every other differential test in this project depends on.
#[test]
fn tied_logits_break_toward_the_lower_index_every_time() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let n = 64;
    let logits = vec![1.0f32; n]; // every expert identical
    let ld = DeviceBuffer::from_slice(&logits).unwrap();

    let mut first = ExpertChoice::new(n, 8).unwrap();
    router(&ld, &mut first, true).unwrap();
    let a = first.indices_to_host().unwrap();
    assert_eq!(a, (0..8).collect::<Vec<i32>>(), "ties must pick the lowest ids in order");

    for _ in 0..8 {
        let mut again = ExpertChoice::new(n, 8).unwrap();
        router(&ld, &mut again, true).unwrap();
        assert_eq!(again.indices_to_host().unwrap(), a, "routing is not deterministic");
    }
}

/// The weighted accumulate is what combines the experts. Its scalar comes from
/// device memory, so a wrong slot index is an off-by-one that silently applies
/// the wrong expert's weight.
#[test]
fn accumulate_applies_the_slot_weight_and_sums() {
    if !gpu() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (n, k, width) = (8usize, 3usize, 257usize); // 257: not a block multiple
    let logits: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let ld = DeviceBuffer::from_slice(&logits).unwrap();
    let mut choice = ExpertChoice::new(n, k).unwrap();
    router(&ld, &mut choice, true).unwrap();
    let w = choice.weights_to_host().unwrap();

    let mut dst = DeviceBuffer::<f32>::zeros(width).unwrap();
    let mut want = vec![0f32; width];
    for (slot, &wt) in w.iter().enumerate().take(k) {
        let src: Vec<f32> = (0..width).map(|i| (i + slot * 7) as f32 * 0.01).collect();
        let sd = DeviceBuffer::from_slice(&src).unwrap();
        accumulate(&mut dst, &sd, &choice, slot).unwrap();
        for (o, s) in want.iter_mut().zip(&src) {
            *o += wt * s;
        }
    }

    for (i, (g, e)) in dst.to_vec().unwrap().iter().zip(&want).enumerate() {
        assert!((g - e).abs() < 1e-5 * e.abs().max(1.0), "element {i}: {g} vs {e}");
    }
    // A slot past the selected experts must be refused, not read out of bounds.
    let spare = DeviceBuffer::<f32>::zeros(width).unwrap();
    assert!(accumulate(&mut dst, &spare, &choice, k).is_err());
}
